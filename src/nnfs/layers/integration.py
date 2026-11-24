"""Layer 4: Integration - Combines patterns into higher-level representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from nnfs.neurons.standard import StandardNeuron
from nnfs.neurons.multiplicative import MultiplicativeNeuron
from nnfs.neurons.geometric import GeometricNeuron, GeometricLayer


class IntegrationLayer(nn.Module):
    """
    Integration Layer - Layer 4
    
    Combines patterns from competition layer into higher-level representations.
    This layer performs PATTERN INTEGRATION:
    - Cross-group pattern combination
    - Hierarchical composition
    - Context-dependent gating
    - Memory-augmented integration
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neuron_type: str = "standard",
        activation: str = "relu",
        dropout: float = 0.0,
        # Multiplicative-specific
        interaction_rank: int = 16,
        use_memory: bool = True,
        memory_decay: float = 0.95,
        # Geometric-specific
        use_complex_weights: bool = True,
        # Integration-specific
        num_heads: int = 8,
        use_cross_attention: bool = True,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_type = neuron_type
        self.use_cross_attention = use_cross_attention
        self.context_dim = context_dim or output_dim
        
        # Adjust num_heads to be valid
        self.num_heads = min(num_heads, output_dim)
        while output_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        
        if neuron_type == "standard":
            self.neurons = StandardNeuron(input_dim, output_dim, activation, bias=True)
            
        elif neuron_type == "multiplicative":
            self.neurons = MultiplicativeNeuron(
                input_dim, output_dim, interaction_rank, activation,
                use_memory=use_memory, memory_decay=memory_decay, bias=True
            )
            
        elif neuron_type == "geometric":
            self.neurons = GeometricLayer(
                input_dim, output_dim, activation, use_complex_weights, dropout=0.0, bias=True
            )
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")
        
        # Cross-attention for pattern integration
        if use_cross_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(output_dim)
        
        # Context gating
        self.context_gate = nn.Sequential(
            nn.Linear(output_dim + self.context_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Integrate patterns with optional context.
        
        Args:
            x: Input patterns (..., input_dim) or (mag, phase) for geometric
            context: Optional context tensor (..., context_dim)
            
        Returns:
            Integrated representation (..., output_dim) or (mag, phase)
        """
        # Primary neuron transformation
        if self.neuron_type == "geometric":
            # Handle both tensor and tuple input
            if isinstance(x, tuple):
                magnitude, phase = x
            else:
                magnitude, phase = self.neurons(x)
            magnitude = self.dropout(magnitude)
            
            # Apply cross-attention on magnitude
            if self.use_cross_attention and magnitude.dim() >= 3:
                # magnitude: (batch, seq, dim) or (batch, dim)
                if magnitude.dim() == 2:
                    magnitude = magnitude.unsqueeze(1)  # (batch, 1, dim)
                
                attended, _ = self.attention(magnitude, magnitude, magnitude)
                magnitude = self.norm(magnitude + self.dropout(attended))
                
                if magnitude.shape[1] == 1:
                    magnitude = magnitude.squeeze(1)
            
            # Context gating
            if context is not None:
                gate_input = torch.cat([magnitude, context], dim=-1)
                gate = self.context_gate(gate_input)
                magnitude = magnitude * gate
            
            return magnitude, phase
            
        else:
            out = self.neurons(x)
            out = self.dropout(out)
            
            # Cross-attention
            if self.use_cross_attention and out.dim() >= 3:
                if out.dim() == 2:
                    out = out.unsqueeze(1)
                attended, _ = self.attention(out, out, out)
                out = self.norm(out + self.dropout(attended))
                if out.shape[1] == 1:
                    out = out.squeeze(1)
            
            # Context gating
            if context is not None:
                gate_input = torch.cat([out, context], dim=-1)
                gate = self.context_gate(gate_input)
                out = out * gate
            
            return out
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, type={self.neuron_type}"


class HierarchicalIntegration(nn.Module):
    """
    Multi-scale hierarchical integration.
    Integrates patterns at multiple temporal/spatial scales.
    """
    
    def __init__(
        self,
        dim: int,
        scales: List[int] = [1, 2, 4, 8],
        neuron_type: str = "standard",
        activation: str = "relu",
    ):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.dim = dim
        
        # Integration at each scale
        self.scale_integrators = nn.ModuleList()
        for scale in scales:
            if neuron_type == "standard":
                integrator = StandardNeuron(dim * scale, dim, activation)
            elif neuron_type == "multiplicative":
                integrator = MultiplicativeNeuron(dim * scale, dim, 8, activation)
            else:
                integrator = StandardNeuron(dim * scale, dim, activation)  # fallback
            self.scale_integrators.append(integrator)
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * self.num_scales, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim) or (batch, dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)
        
        batch, seq_len, dim = x.shape
        outputs = []
        
        for i, (scale, integrator) in enumerate(zip(self.scales, self.scale_integrators)):
            if scale == 1:
                scale_input = x
            else:
                # Pool over scale
                pooled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)  # (batch, seq_len/scale, dim)
                
                # Repeat to match sequence length for concatenation
                if pooled.shape[1] < seq_len:
                    repeat_factor = (seq_len + pooled.shape[1] - 1) // pooled.shape[1]
                    pooled = pooled.repeat(1, repeat_factor, 1)[:, :seq_len]
                
                scale_input = pooled
            
            # Flatten for integrator
            scale_flat = scale_input.view(batch * seq_len, -1)
            integrated = integrator(scale_flat)
            integrated = integrated.view(batch, seq_len, -1)
            outputs.append(integrated)
        
        # Fuse scales
        fused = torch.cat(outputs, dim=-1)
        fused = self.scale_fusion(fused)
        
        if fused.shape[1] == 1:
            fused = fused.squeeze(1)
        
        return fused