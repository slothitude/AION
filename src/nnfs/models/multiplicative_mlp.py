"""MLP with multiplicative interaction neurons."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from nnfs.neurons.multiplicative import MultiplicativeNeuron, MultiplicativeLayer
from nnfs.config import ModelConfig


class MultiplicativeMLP(nn.Module):
    """
    MLP with multiplicative interaction neurons.
    
    Each hidden layer uses multiplicative neurons that can detect
    signal combinations through low-rank multiplicative interactions.
    
    Architecture: Input -> [Multiplicative Hidden Layers] -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        interaction_rank: int = 8,
        use_memory: bool = True,
        memory_decay: float = 0.9,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        
        if config is not None:
            hidden_dims = config.hidden_dims
            activation = config.activation
            dropout = config.dropout
            interaction_rank = config.interaction_rank
        
        hidden_dims = hidden_dims or [256, 128, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.interaction_rank = interaction_rank
        self.use_memory = use_memory
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Use memory in all but first layer (to avoid initial instability)
            layer_use_memory = use_memory and i > 0
            layers.append(MultiplicativeLayer(
                prev_dim, hidden_dim, interaction_rank, activation, dropout,
                use_memory=layer_use_memory, memory_decay=memory_decay
            ))
            prev_dim = hidden_dim
        
        # Output layer: standard (no interactions needed for final classification)
        from nnfs.neurons.standard import StandardLayer
        layers.append(StandardLayer(prev_dim, output_dim, "identity", 0.0))
        
        self.layers = nn.Sequential(*layers)
        self._param_count = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self._param_count
    
    def reset_memory(self):
        """Reset memory in all multiplicative layers."""
        for layer in self.layers:
            if hasattr(layer, 'reset_memory'):
                layer.reset_memory()
            elif hasattr(layer, 'neuron') and hasattr(layer.neuron, 'reset_memory'):
                layer.neuron.reset_memory()
    
    def get_interaction_strengths(self) -> List[torch.Tensor]:
        """Get interaction weight norms for analysis."""
        strengths = []
        for layer in self.layers:
            if hasattr(layer, 'neuron') and hasattr(layer.neuron, 'A'):
                # A matrix norm indicates interaction strength
                strength = layer.neuron.A.norm(dim=1).detach()
                strengths.append(strength)
        return strengths
    
    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
                f"output_dim={self.output_dim}, interaction_rank={self.interaction_rank}")


class MultiplicativeMLPWithCompetition(nn.Module):
    """
    Multiplicative MLP with competition layers between hidden layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        interaction_rank: int = 8,
        group_size: int = 8,
        adaptive_temp: bool = True,
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [256, 128, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Multiplicative layer
            layers.append(MultiplicativeLayer(
                prev_dim, hidden_dim, interaction_rank, activation, dropout,
                use_memory=True, memory_decay=0.9
            ))
            
            # Competition layer (except last hidden)
            if i < len(hidden_dims) - 1:
                from nnfs.layers.competition import CompetitionLayer
                layers.append(CompetitionLayer(
                    hidden_dim, group_size=group_size,
                    adaptive_temperature=adaptive_temp
                ))
            
            prev_dim = hidden_dim
        
        # Output
        from nnfs.neurons.standard import StandardLayer
        layers.append(StandardLayer(prev_dim, output_dim, "identity", 0.0))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_memory(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_memory'):
                layer.reset_memory()