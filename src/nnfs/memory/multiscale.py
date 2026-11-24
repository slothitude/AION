"""Fast/Slow multi-timescale memory system."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class FastSlowMemory(nn.Module):
    """
    Dual timescale memory system.
    
    Fast state: Reacts quickly to incoming signals (short-term)
    Slow state: Maintains context and longer-lived patterns (long-term)
    
    h_fast_{t+1} = f(x_t, h_fast_t)
    h_slow_{t+1} = g(h_fast_{t+1}, h_slow_t)
    
    This gives the network expressive internal memory beyond weights
    and external attention context.
    """
    
    def __init__(
        self,
        input_dim: int,
        fast_dim: int,
        slow_dim: int,
        fast_update_rate: float = 0.3,   # How fast fast-state adapts
        slow_update_rate: float = 0.01,  # How fast slow-state adapts
        coupling_strength: float = 0.1,  # Fast -> Slow coupling
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.fast_update_rate = fast_update_rate
        self.slow_update_rate = slow_update_rate
        self.coupling_strength = coupling_strength
        
        # Fast state update: f(x_t, h_fast_t)
        self.fast_updater = nn.Sequential(
            nn.Linear(input_dim + fast_dim, fast_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fast_dim * 2, fast_dim),
        )
        
        # Slow state update: g(h_fast, h_slow)
        self.slow_updater = nn.Sequential(
            nn.Linear(fast_dim + slow_dim, slow_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(slow_dim * 2, slow_dim),
        )
        
        # Output projection combining both states
        self.output_proj = nn.Sequential(
            nn.Linear(fast_dim + slow_dim, fast_dim + slow_dim),
            self._get_activation(activation),
            nn.Linear(fast_dim + slow_dim, fast_dim),  # Project to fast_dim for compatibility
        )
        
        # Initialize states
        self.register_buffer("fast_state", torch.zeros(1, fast_dim))
        self.register_buffer("slow_state", torch.zeros(1, slow_dim))
        
        # Statistics
        self.register_buffer("step_count", torch.tensor(0))
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.Tanh())
    
    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset memory states."""
        if device is None:
            device = self.fast_state.device
        
        self.fast_state = torch.zeros(batch_size, self.fast_dim, device=device)
        self.slow_state = torch.zeros(batch_size, self.slow_dim, device=device)
        self.step_count.zero_()
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Update memory with new input.
        
        Args:
            x: Input (batch, input_dim) or (batch, seq, input_dim)
            return_states: Whether to return fast/slow states
            
        Returns:
            Combined memory output, optionally (fast_state, slow_state)
        """
        # Handle sequence input
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            outputs = []
            
            for t in range(seq_len):
                out = self._step(x[:, t])
                outputs.append(out)
            
            output = torch.stack(outputs, dim=1)
        else:
            output = self._step(x)
        
        if return_states:
            return output, self.fast_state, self.slow_state
        return output
    
    def _step(self, x: torch.Tensor) -> torch.Tensor:
        """Single memory update step."""
        batch = x.shape[0]
        device = x.device
        
        # Recreate states if batch size changed (expand only works on singleton dims)
        if self.fast_state.shape[0] != batch:
            self.fast_state = torch.zeros(batch, self.fast_dim, device=device)
            self.slow_state = torch.zeros(batch, self.slow_dim, device=device)
        
        # Fast state update: h_fast_{t+1} = (1-α) h_fast_t + α f(x_t, h_fast_t)
        fast_input = torch.cat([x, self.fast_state], dim=-1)
        fast_proposed = self.fast_updater(fast_input)
        self.fast_state = (1 - self.fast_update_rate) * self.fast_state + self.fast_update_rate * fast_proposed
        
        # Slow state update: h_slow_{t+1} = (1-β) h_slow_t + β g(h_fast_{t+1}, h_slow_t)
        slow_input = torch.cat([self.fast_state, self.slow_state], dim=-1)
        slow_proposed = self.slow_updater(slow_input)
        self.slow_state = (1 - self.slow_update_rate) * self.slow_state + self.slow_update_rate * slow_proposed
        
        # Combined output
        combined = torch.cat([self.fast_state, self.slow_state], dim=-1)
        output = self.output_proj(combined)
        
        self.step_count += 1
        return output
    
    def get_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current fast and slow states."""
        return self.fast_state, self.slow_state
    
    def set_states(self, fast: torch.Tensor, slow: torch.Tensor):
        """Set memory states (e.g., for initialization from checkpoint)."""
        self.fast_state = fast
        self.slow_state = slow
    
    def extra_repr(self) -> str:
        return (f"fast_dim={self.fast_dim}, slow_dim={self.slow_dim}, "
                f"fast_rate={self.fast_update_rate}, slow_rate={self.slow_update_rate}")


class MultiScaleMemory(nn.Module):
    """
    Multi-scale memory with multiple timescales.
    
    Maintains a hierarchy of memory states at different timescales:
    - Ultra-fast (reactive)
    - Fast (working memory)
    - Medium (episodic)
    - Slow (semantic/contextual)
    
    Each scale has its own update rate and capacity.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_scales: int = 4,
        base_update_rate: float = 0.3,
        rate_decay: float = 0.3,  # Each scale is rate_decay times slower
        activation: str = "tanh",
        dropout: float = 0.0,
        cross_scale_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.base_update_rate = base_update_rate
        self.rate_decay = rate_decay
        
        # Compute update rates for each scale (geometric progression)
        self.update_rates = [
            base_update_rate * (rate_decay ** i) for i in range(num_scales)
        ]
        
        # Memory cells for each scale
        self.memory_cells = nn.ModuleList()
        for i, rate in enumerate(self.update_rates):
            cell = MemoryCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                update_rate=rate,
                activation=activation,
                dropout=dropout,
            )
            self.memory_cells.append(cell)
        
        # Cross-scale attention
        self.cross_scale_attention = cross_scale_attention
        if cross_scale_attention:
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.scale_norm = nn.LayerNorm(hidden_dim)
        
        # Output fusion
        self.output_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Initialize states
        self.register_buffer("states", torch.zeros(num_scales, 1, hidden_dim))
        self.register_buffer("step_count", torch.tensor(0))
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.Tanh())
    
    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset all memory states."""
        if device is None:
            device = self.states.device
        self.states = torch.zeros(self.num_scales, batch_size, self.hidden_dim, device=device)
        self.step_count.zero_()
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_all_states: bool = False
    ) -> torch.Tensor:
        """
        Update multi-scale memory.
        
        Args:
            x: Input (batch, input_dim) or (batch, seq, input_dim)
            return_all_states: Return states from all scales
            
        Returns:
            Fused memory output (batch, hidden_dim) or (batch, seq, hidden_dim)
        """
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            outputs = []
            all_states_seq = []
            
            for t in range(seq_len):
                out, states = self._step(x[:, t], return_states=True)
                outputs.append(out)
                if return_all_states:
                    all_states_seq.append(states)
            
            output = torch.stack(outputs, dim=1)
            if return_all_states:
                return output, torch.stack(all_states_seq, dim=1)
            return output
        else:
            return self._step(x, return_states=return_all_states)
    
    def _step(self, x: torch.Tensor, return_states: bool = False) -> Tuple[torch.Tensor, ...]:
        """Single step updating all scales."""
        batch = x.shape[0]
        
        # Expand states if needed
        if self.states.shape[1] != batch:
            self.states = self.states.expand(-1, batch, -1).clone()
        
        current_input = x
        new_states = []
        
        # Update each scale
        for i, cell in enumerate(self.memory_cells):
            new_state = cell(current_input, self.states[i])
            new_states.append(new_state)
            
            # Input to next scale is this scale's state
            current_input = new_state
        
        new_states = torch.stack(new_states, dim=0)  # (num_scales, batch, hidden_dim)
        self.states = new_states
        
        # Cross-scale attention
        if self.cross_scale_attention and self.num_scales > 1:
            # Attention across scales for each batch item
            states_for_attn = new_states.permute(1, 0, 2)  # (batch, num_scales, hidden_dim)
            attended, _ = self.scale_attention(states_for_attn, states_for_attn, states_for_attn)
            attended = self.scale_norm(states_for_attn + attended)
            attended = attended.permute(1, 0, 2)  # (num_scales, batch, hidden_dim)
            
            # Use attended states for output
            output_states = attended
        else:
            output_states = new_states
        
        # Fuse all scales for output
        fused = output_states.permute(1, 0, 2).reshape(batch, -1)  # (batch, num_scales * hidden_dim)
        output = self.output_fusion(fused)
        
        self.step_count += 1
        
        if return_states:
            return output, new_states
        return output
    
    def get_states(self) -> torch.Tensor:
        """Get all scale states: (num_scales, batch, hidden_dim)"""
        return self.states
    
    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                f"num_scales={self.num_scales}, rates={self.update_rates}")


class MemoryCell(nn.Module):
    """Single memory cell with gated update."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        update_rate: float = 0.1,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.update_rate = update_rate
        
        # Gated update similar to GRU but with fixed rate
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update memory cell."""
        combined = torch.cat([x, h], dim=-1)
        
        # Update gate (modulated by fixed rate)
        gate = self.update_gate(combined) * self.update_rate
        
        # Candidate state
        candidate = self.candidate(combined)
        
        # Gated update
        h_new = (1 - gate) * h + gate * candidate
        return h_new