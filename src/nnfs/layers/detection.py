"""Layer 1: Detection - Identifies individual features and signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from nnfs.neurons.standard import StandardNeuron
from nnfs.neurons.multiplicative import MultiplicativeNeuron
from nnfs.neurons.geometric import GeometricNeuron, GeometricLayer


class DetectionLayer(nn.Module):
    """
    Detection Layer - Layer 1
    
    Identifies individual features and signals from input.
    Uses diverse receptive fields to detect different patterns.
    
    This layer focuses on FEATURE EXTRACTION:
    - Multiple filter sizes/kernels
    - Sparse activations
    - Local receptive fields (for structured data)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neuron_type: str = "standard",
        activation: str = "relu",
        dropout: float = 0.0,
        # Multiplicative-specific
        interaction_rank: int = 8,
        use_memory: bool = False,
        # Geometric-specific
        use_complex_weights: bool = True,
        # Detection-specific
        num_filters: int = 4,
        filter_sizes: Optional[List[int]] = None,
        sparsity_target: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_type = neuron_type
        self.sparsity_target = sparsity_target
        
        if neuron_type == "standard":
            self.neurons = StandardNeuron(input_dim, output_dim, activation, bias=True)
        elif neuron_type == "multiplicative":
            self.neurons = MultiplicativeNeuron(
                input_dim, output_dim, interaction_rank, activation,
                use_memory=use_memory, bias=True
            )
        elif neuron_type == "geometric":
            self.neurons = GeometricLayer(
                input_dim, output_dim, activation, use_complex_weights, dropout=0.0, bias=True
            )
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Sparsity regularization (encourages selective feature detection)
        self.register_buffer("sparsity_loss", torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (..., input_dim)
            
        Returns:
            If standard/multiplicative: (..., output_dim)
            If geometric: (magnitude, phase) each (..., output_dim)
        """
        if self.neuron_type == "geometric":
            magnitude, phase = self.neurons(x)
            magnitude = self.dropout(magnitude)
            
            # Compute sparsity loss for monitoring
            if self.training:
                active = (magnitude > 0.01).float().mean()
                self.sparsity_loss = F.mse_loss(active, torch.tensor(self.sparsity_target, device=x.device))
            
            return magnitude, phase
        else:
            out = self.neurons(x)
            out = self.dropout(out)
            
            # Sparsity monitoring
            if self.training:
                active = (out > 0.01).float().mean()
                self.sparsity_loss = F.mse_loss(active, torch.tensor(self.sparsity_target, device=x.device))
            
            return out
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, type={self.neuron_type}"


class MultiScaleDetectionLayer(nn.Module):
    """
    Detection layer with multiple filter scales.
    Useful for structured inputs (images, sequences).
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_sizes: List[int] = [1, 3, 5, 7],
        activation: str = "relu",
        neuron_type: str = "standard",
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        channels_per_scale = output_channels // self.num_scales
        
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            conv = nn.Conv1d(input_channels, channels_per_scale, k, padding=padding)
            self.convs.append(conv)
        
        self.activation = self._get_activation(activation)
        self.output_channels = channels_per_scale * self.num_scales
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, length) for 1D or (batch, channels, H, W) for 2D
        """
        outputs = []
        for conv in self.convs:
            out = conv(x)
            out = self.activation(out)
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)