"""Magnitude-phase (geometric) neuron using complex-valued representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GeometricNeuron(nn.Module):
    """
    Geometric neuron with magnitude and phase (complex-valued).
    
    Each neuron's activation is a complex number: z = r * e^(iθ)
    - r (magnitude): signal strength
    - θ (phase): signal direction/phase
    
    Operations in complex domain naturally model:
    - Reinforcement: same phase → magnitudes add
    - Cancellation: opposite phase → magnitudes subtract
    - Rotation: phase shifts transform relationships
    
    Args:
        input_dim: Input dimension (real-valued inputs)
        output_dim: Output dimension (complex neurons)
        activation: Activation for magnitude (phase passes through)
        use_complex_weights: Whether weights are complex (more params)
        bias: Whether to use complex bias
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        use_complex_weights: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_complex_weights = use_complex_weights
        
        if use_complex_weights:
            # Complex weights: W = W_real + i*W_imag
            self.weight_real = nn.Parameter(torch.empty(output_dim, input_dim))
            self.weight_imag = nn.Parameter(torch.empty(output_dim, input_dim))
        else:
            # Real weights applied to complex inputs (magnitude only)
            self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
            self.register_parameter("weight_real", None)
            self.register_parameter("weight_imag", None)
        
        if bias:
            self.bias_real = nn.Parameter(torch.empty(output_dim))
            self.bias_imag = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        
        self.magnitude_activation = self._get_activation(activation)
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
            "modrelu": ModReLU(),  # Complex activation
            "softplus": nn.Softplus(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def reset_parameters(self):
        if self.use_complex_weights:
            # Initialize complex weights with proper scaling
            nn.init.kaiming_uniform_(self.weight_real, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.weight_imag, nonlinearity="relu")
            # Scale imaginary part down initially
            with torch.no_grad():
                self.weight_imag.mul_(0.1)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (magnitude, phase).
        
        Args:
            x: Input tensor of shape (..., input_dim) - real valued
            
        Returns:
            magnitude: Tensor of shape (..., output_dim)
            phase: Tensor of shape (..., output_dim) in [-π, π]
        """
        if self.use_complex_weights:
            # Complex linear transformation: (W_real + i*W_imag) @ x
            real_part = F.linear(x, self.weight_real, self.bias_real)
            imag_part = F.linear(x, self.weight_imag, self.bias_imag)
        else:
            # Real weights -> real output, zero imaginary
            real_part = F.linear(x, self.weight, self.bias_real)
            imag_part = torch.zeros_like(real_part)
        
        # Convert to polar coordinates
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        # Apply activation to magnitude only (phase preserved/modified)
        if isinstance(self.magnitude_activation, ModReLU):
            magnitude = self.magnitude_activation(magnitude, phase)
        else:
            magnitude = self.magnitude_activation(magnitude)
        
        return magnitude, phase
    
    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning complex tensor. Handles both real and complex input."""
        if x.dtype.is_complex:
            # Complex input: (W_real + i*W_imag) @ x
            if self.use_complex_weights:
                real_part = F.linear(x.real, self.weight_real, self.bias_real) - F.linear(x.imag, self.weight_imag, self.bias_imag)
                imag_part = F.linear(x.real, self.weight_imag, self.bias_imag) + F.linear(x.imag, self.weight_real, self.bias_real)
            else:
                real_part = F.linear(x.real, self.weight, self.bias_real)
                imag_part = F.linear(x.imag, self.weight, self.bias_imag)
        else:
            # Real input
            if self.use_complex_weights:
                real_part = F.linear(x, self.weight_real, self.bias_real)
                imag_part = F.linear(x, self.weight_imag, self.bias_imag)
            else:
                real_part = F.linear(x, self.weight, self.bias_real)
                imag_part = torch.zeros_like(real_part)
        
        # Convert to polar coordinates
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        # Apply activation to magnitude only
        if isinstance(self.magnitude_activation, ModReLU):
            magnitude = self.magnitude_activation(magnitude, phase)
        else:
            magnitude = self.magnitude_activation(magnitude)
        
        return magnitude * torch.exp(1j * phase)
    
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, complex_weights={self.use_complex_weights}"


class ModReLU(nn.Module):
    """ModReLU activation for complex numbers: max(|z| + b, 0) * e^(i*arg(z))"""
    
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return F.relu(magnitude + self.bias)


class GeometricLayer(nn.Module):
    """A layer of geometric neurons."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        use_complex_weights: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.neuron = GeometricNeuron(
            input_dim, output_dim, activation, use_complex_weights, bias
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        magnitude, phase = self.neuron(x)
        # Apply dropout to magnitude only
        magnitude = self.dropout(magnitude)
        return magnitude, phase
    
    def forward_complex(self, x: torch.Tensor) -> torch.complex:
        return self.neuron.forward_complex(x)


def complex_multiply(z1: Tuple[torch.Tensor, torch.Tensor], z2: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiply two complex numbers in polar form: (r1, θ1) * (r2, θ2) = (r1*r2, θ1+θ2)"""
    r1, theta1 = z1
    r2, theta2 = z2
    return r1 * r2, theta1 + theta2


def complex_add(z1: Tuple[torch.Tensor, torch.Tensor], z2: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add two complex numbers in polar form via Cartesian conversion."""
    r1, theta1 = z1
    r2, theta2 = z2
    
    x1 = r1 * torch.cos(theta1)
    y1 = r1 * torch.sin(theta1)
    x2 = r2 * torch.cos(theta2)
    y2 = r2 * torch.sin(theta2)
    
    x = x1 + x2
    y = y1 + y2
    
    r = torch.sqrt(x**2 + y**2 + 1e-8)
    theta = torch.atan2(y, x)
    
    return r, theta


def complex_conjugate(z: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complex conjugate: (r, θ) -> (r, -θ)"""
    r, theta = z
    return r, -theta


def magnitude(z: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Get magnitude from polar representation."""
    return z[0]


def phase(z: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Get phase from polar representation."""
    return z[1]