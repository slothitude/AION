"""MLP with geometric (magnitude-phase) neurons."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from nnfs.neurons.geometric import GeometricNeuron, GeometricLayer, magnitude, phase
from nnfs.config import ModelConfig


class GeometricMLP(nn.Module):
    """
    MLP with geometric (complex-valued) neurons.
    
    Each neuron has magnitude (strength) and phase (direction).
    This enables geometric computation: reinforcement, cancellation, rotation.
    
    Architecture: Input -> [Geometric Hidden Layers] -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        use_complex_weights: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        
        if config is not None:
            hidden_dims = config.hidden_dims
            activation = config.activation
            dropout = config.dropout
            use_complex_weights = config.use_complex_weights
        
        hidden_dims = hidden_dims or [256, 128, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_complex_weights = use_complex_weights
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(GeometricLayer(
                prev_dim, hidden_dim, activation, use_complex_weights, dropout
            ))
            prev_dim = hidden_dim
        
        # Output layer: project magnitude to logits
        from nnfs.neurons.standard import StandardLayer
        layers.append(StandardLayer(prev_dim, output_dim, "identity", 0.0))
        
        self.layers = nn.Sequential(*layers)
        self._param_count = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning final logits (magnitude only).
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Logits tensor (batch, output_dim)
        """
        magnitude = None
        phase = None
        
        for layer in self.layers:
            if isinstance(layer, GeometricLayer):
                magnitude, phase = layer(x)
                x = magnitude  # Pass magnitude to next layer
            else:
                # Standard output layer
                x = layer(magnitude)
        
        return x
    
    def forward_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both magnitude and phase at each layer.
        
        Returns:
            List of (magnitude, phase) tuples for each geometric layer,
            plus final logits
        """
        outputs = []
        magnitude = None
        phase = None
        
        for layer in self.layers:
            if isinstance(layer, GeometricLayer):
                magnitude, phase = layer(x)
                outputs.append((magnitude, phase))
                x = magnitude
            else:
                x = layer(magnitude)
        
        outputs.append(("logits", x))
        return outputs
    
    def forward_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using complex arithmetic throughout."""
        for layer in self.layers:
            if isinstance(layer, GeometricLayer):
                x = layer.forward_complex(x)
            else:
                # Convert complex to magnitude for final layer
                x = torch.abs(x)
                x = layer(x)
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self._param_count
    
    def get_phase_statistics(self, x: torch.Tensor) -> dict:
        """Analyze phase distributions across layers."""
        self.eval()
        with torch.no_grad():
            layer_outputs = self.forward_full(x)
        
        stats = {}
        for i, out in enumerate(layer_outputs[:-1]):  # Exclude logits
            mag, ph = out
            stats[f"layer_{i}"] = {
                "mean_magnitude": mag.mean().item(),
                "std_magnitude": mag.std().item(),
                "mean_phase": ph.mean().item(),
                "std_phase": ph.std().item(),
                "phase_entropy": self._phase_entropy(ph).item(),
            }
        return stats
    
    def _phase_entropy(self, phase: torch.Tensor, bins: int = 32) -> torch.Tensor:
        """Compute phase distribution entropy."""
        hist = torch.histc(phase, bins=bins, min=-torch.pi, max=torch.pi)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return -(probs * torch.log(probs)).sum()
    
    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
                f"output_dim={self.output_dim}, complex_weights={self.use_complex_weights}")


class GeometricMLPWithCompetition(nn.Module):
    """Geometric MLP with geometric competition layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        use_complex_weights: bool = True,
        group_size: int = 8,
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [256, 128, 64]
        
        from nnfs.layers.competition import GeometricCompetitionLayer
        from nnfs.neurons.standard import StandardLayer
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(GeometricLayer(
                prev_dim, hidden_dim, activation, use_complex_weights, dropout
            ))
            
            if i < len(hidden_dims) - 1:
                layers.append(GeometricCompetitionLayer(
                    hidden_dim, group_size=group_size
                ))
            
            prev_dim = hidden_dim
        
        layers.append(StandardLayer(prev_dim, output_dim, "identity", 0.0))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = None
        phase = None
        
        for layer in self.layers:
            if isinstance(layer, GeometricLayer):
                magnitude, phase = layer(x)
                x = magnitude
            elif hasattr(layer, 'competition'):  # GeometricCompetitionLayer
                magnitude, phase = layer(magnitude, phase)
                x = magnitude
            else:
                x = layer(magnitude)
        
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)