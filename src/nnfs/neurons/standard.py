"""Standard weighted-sum neuron (baseline)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StandardNeuron(nn.Module):
    """
    Standard neuron: y = σ(Σ w_i x_i + b)
    
    This is the conventional neuron that computes a weighted sum
    followed by a non-linear activation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
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
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Standard linear transformation: x @ W^T + b
        out = F.linear(x, self.weight, self.bias)
        return self.activation(out)
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


class StandardLayer(nn.Module):
    """A layer of standard neurons (equivalent to nn.Linear + activation)."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.neuron = StandardNeuron(input_dim, output_dim, activation, bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.neuron(x))