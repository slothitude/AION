"""Standard MLP baseline architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from nnfs.neurons.standard import StandardNeuron, StandardLayer
from nnfs.config import ModelConfig


class StandardMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron baseline.
    
    Architecture: Input -> [Hidden Layers] -> Output
    Each layer: Linear + Activation + Dropout
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        
        if config is not None:
            hidden_dims = config.hidden_dims
            activation = config.activation
            dropout = config.dropout
        
        hidden_dims = hidden_dims or [256, 128, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(StandardLayer(prev_dim, hidden_dim, activation, dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation, no dropout)
        layers.append(StandardLayer(prev_dim, output_dim, "identity", 0.0))
        
        self.layers = nn.Sequential(*layers)
        
        # Parameter count tracking
        self._param_count = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self._param_count
    
    def get_layer_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate layer outputs for analysis."""
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, output_dim={self.output_dim}"