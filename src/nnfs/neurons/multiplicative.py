"""Multiplicative interaction neuron with low-rank interactions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiplicativeNeuron(nn.Module):
    """
    Multiplicative interaction neuron with learnable interaction strength:
    
    z_j = Σ_i w_ji x_i + (α/√r) Σ_r a_jr [(P_r x) ⊙ (Q_r x)] + λ_j h_j
    h_j_new = φ(z_j)
    
    This neuron can detect combinations of signals through multiplicative
    interactions. The interaction term uses a low-rank factorization to
    keep computational cost manageable.
    
    Key enhancements:
    - Learnable interaction strength α (initialized small)
    - Normalization by √r to keep gradient scale consistent
    - Optional LayerNorm on interaction features
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (number of neurons)
        interaction_rank: Rank of the low-rank interaction (default: 8)
        activation: Activation function name
        use_memory: Whether to include memory term (λ_j h_j)
        memory_decay: Decay factor for memory (λ)
        bias: Whether to use bias
        interaction_strength_init: Initial value for α (default: 0.1)
        normalize_interactions: Whether to normalize interaction features (default: True)
        learn_alpha: Whether α is learnable (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        interaction_rank: int = 8,
        activation: str = "relu",
        use_memory: bool = True,
        memory_decay: float = 0.9,
        bias: bool = True,
        interaction_strength_init: float = 0.1,
        normalize_interactions: bool = True,
        learn_alpha: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.interaction_rank = interaction_rank
        self.use_memory = use_memory
        self.memory_decay = memory_decay
        self.normalize_interactions = normalize_interactions
        
        # Additive weights: W ∈ ℝ^(output_dim × input_dim)
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        
        # Low-rank interaction: 
        # P ∈ ℝ^(interaction_rank × input_dim)
        # Q ∈ ℝ^(interaction_rank × input_dim)
        # A ∈ ℝ^(output_dim × interaction_rank)
        self.P = nn.Parameter(torch.empty(interaction_rank, input_dim))
        self.Q = nn.Parameter(torch.empty(interaction_rank, input_dim))
        self.A = nn.Parameter(torch.empty(output_dim, interaction_rank))
        
        # Learnable interaction strength α
        if learn_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(interaction_strength_init)))
        else:
            self.register_buffer("log_alpha", torch.tensor(math.log(interaction_strength_init)))
        
        # Optional normalization of interaction features
        if normalize_interactions:
            self.interaction_norm = nn.LayerNorm(interaction_rank, elementwise_affine=False)
        else:
            self.interaction_norm = nn.Identity()
        
        # Memory decay per neuron: λ ∈ ℝ^(output_dim)
        if use_memory:
            self.memory_decay_param = nn.Parameter(torch.full((output_dim,), memory_decay))
            self.register_buffer("memory", torch.zeros(output_dim))
        else:
            self.register_parameter("memory_decay_param", None)
            self.register_buffer("memory", None)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        
        self.activation = self._get_activation(activation)
        self.reset_parameters()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "identity": nn.Identity(),
            "softplus": nn.Softplus(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def reset_parameters(self):
        # Additive weights: kaiming init
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        
        # Interaction factors: smaller initialization to start near additive behavior
        # Initialize P, Q with orthogonal-like structure for diverse projections
        nn.init.xavier_uniform_(self.P, gain=0.1)
        nn.init.xavier_uniform_(self.Q, gain=0.1)
        # A initialized smaller so interactions start weak
        nn.init.xavier_uniform_(self.A, gain=0.05)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        if self.use_memory:
            nn.init.zeros_(self.memory)
    
    def reset_memory(self):
        """Reset memory state (call between sequences)."""
        if self.use_memory:
            self.memory.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multiplicative interactions.
        
        Args:
            x: Input tensor of shape (..., input_dim) or (batch, seq, input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Additive term: x @ W^T
        additive = F.linear(x, self.weight, self.bias)
        
        # Multiplicative interaction term with learnable strength
        # Compute P x and Q x for each rank component
        # x: (..., input_dim), P: (rank, input_dim) -> (..., rank)
        px = torch.matmul(x, self.P.t())  # (..., rank)
        qx = torch.matmul(x, self.Q.t())  # (..., rank)
        
        # Element-wise product: (..., rank)
        interaction_features = px * qx
        
        # Normalize interaction features to stabilize gradients
        interaction_features = self.interaction_norm(interaction_features)
        
        # Project to output: (..., rank) @ A^T -> (..., output_dim)
        interaction = F.linear(interaction_features, self.A)
        
        # Scale by learnable α / √r
        alpha = torch.exp(self.log_alpha)
        scale = alpha / math.sqrt(self.interaction_rank)
        interaction = interaction * scale
        
        # Combine additive and interaction
        z = additive + interaction
        
        # Memory term
        if self.use_memory:
            # Expand memory to match batch dimensions
            mem_shape = [1] * (z.dim() - 1) + [self.output_dim]
            memory_expanded = self.memory.view(mem_shape)
            z = z + self.memory_decay_param * memory_expanded
            
            # Update memory (detach to avoid gradient flow through time)
            # Use a separate buffer update that doesn't interfere with autograd
            if self.training:
                with torch.no_grad():
                    new_memory = z.detach().mean(dim=tuple(range(z.dim() - 1)))
                    self.memory.data.copy_(new_memory)
        
        return self.activation(z)
    
    def get_interaction_strength(self) -> float:
        """Return current interaction strength α."""
        return torch.exp(self.log_alpha).item()
    
    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"interaction_rank={self.interaction_rank}, use_memory={self.use_memory}")


class MultiplicativeLayer(nn.Module):
    """A layer of multiplicative neurons."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        interaction_rank: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
        use_memory: bool = True,
        memory_decay: float = 0.9,
        bias: bool = True,
    ):
        super().__init__()
        self.neuron = MultiplicativeNeuron(
            input_dim, output_dim, interaction_rank, activation,
            use_memory, memory_decay, bias
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.neuron(x))
    
    def reset_memory(self):
        self.neuron.reset_memory()