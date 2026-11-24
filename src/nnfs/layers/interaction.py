"""Layer 2: Interaction - Combines signals using multiplication and geometric relationships."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nnfs.neurons.multiplicative import MultiplicativeNeuron
from nnfs.neurons.geometric import GeometricNeuron, GeometricLayer, complex_multiply, complex_add
from nnfs.neurons.standard import StandardNeuron


class InteractionLayer(nn.Module):
    """
    Interaction Layer - Layer 2
    
    Combines signals using multiplicative interactions and geometric relationships.
    This is where signal COMBINATION happens - detecting conjunctions, 
    correlations, and higher-order patterns.
    
    Key operations:
    - Multiplicative gating (signal * signal)
    - Geometric combination (phase relationships)
    - Cross-feature interactions
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neuron_type: str = "multiplicative",
        activation: str = "relu",
        dropout: float = 0.0,
        # Multiplicative-specific
        interaction_rank: int = 16,
        use_memory: bool = True,
        memory_decay: float = 0.95,
        # Geometric-specific
        use_complex_weights: bool = True,
        # Interaction-specific
        interaction_mode: str = "multiplicative",  # "multiplicative", "geometric", "bilinear"
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_type = neuron_type
        self.interaction_mode = interaction_mode
        self.num_heads = num_heads
        
        if neuron_type == "standard":
            # Standard neuron with bilinear interaction
            self.neurons = StandardNeuron(input_dim, output_dim, activation, bias=True)
            # Add bilinear interaction term
            self.bilinear_W = nn.Parameter(torch.empty(num_heads, input_dim, input_dim))
            self.bilinear_proj = nn.Parameter(torch.empty(output_dim, num_heads))
            nn.init.xavier_uniform_(self.bilinear_W, gain=0.1)
            nn.init.xavier_uniform_(self.bilinear_proj, gain=0.1)
            
        elif neuron_type == "multiplicative":
            self.neurons = MultiplicativeNeuron(
                input_dim, output_dim, interaction_rank, activation,
                use_memory=use_memory, memory_decay=memory_decay, bias=True
            )
            
        elif neuron_type == "geometric":
            self.neurons = GeometricLayer(
                input_dim, output_dim, activation, use_complex_weights, dropout=0.0, bias=True
            )
            # Geometric interaction: phase-aware combination
            self.phase_mixer = nn.Parameter(torch.empty(num_heads, output_dim))
            nn.init.xavier_uniform_(self.phase_mixer)
        
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with interaction.
        
        Args:
            x: Input tensor (..., input_dim) or for geometric: (mag, phase)
            
        Returns:
            Output tensor (..., output_dim) or for geometric: (mag, phase)
        """
        if self.neuron_type == "standard":
            # Standard path with bilinear interaction
            linear_out = self.neurons(x)
            
            # Bilinear interaction: x^T W_h x for each head
            # x: (..., D) -> (..., H) after einsum
            bilinear_terms = torch.einsum('...i,hij,...j->...h', x, self.bilinear_W, x)
            bilinear_out = F.linear(bilinear_terms, self.bilinear_proj)
            
            out = linear_out + bilinear_out
            return self.dropout(out)
            
        elif self.neuron_type == "multiplicative":
            out = self.neurons(x)
            return self.dropout(out)
            
        elif self.neuron_type == "geometric":
            magnitude, phase = self.neurons(x)
            
            # Phase mixing across heads for geometric interaction
            # This allows phase relationships to create new patterns
            if self.training and self.num_heads > 1:
                # Create head-specific phase shifts
                phase_shifts = torch.linspace(-torch.pi/4, torch.pi/4, self.num_heads, device=phase.device)
                phase_shifts = phase_shifts.view(1, -1, 1)  # (1, H, 1)
                
                # Apply shifts and average (phase-aware ensemble)
                shifted_phases = phase.unsqueeze(1) + phase_shifts  # (B, H, D)
                # Convert to complex, average, back to polar
                real = magnitude.unsqueeze(1) * torch.cos(shifted_phases)
                imag = magnitude.unsqueeze(1) * torch.sin(shifted_phases)
                avg_real = real.mean(dim=1)
                avg_imag = imag.mean(dim=1)
                
                magnitude = torch.sqrt(avg_real**2 + avg_imag**2 + 1e-8)
                phase = torch.atan2(avg_imag, avg_real)
            
            magnitude = self.dropout(magnitude)
            return magnitude, phase


class BilinearInteraction(nn.Module):
    """
    Explicit bilinear interaction module.
    Computes x^T W y for pairs of inputs.
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: int, rank: int = 32):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.rank = rank
        
        # Low-rank factorization: W = U @ V^T where U: (dim1, rank), V: (dim2, rank)
        self.U = nn.Parameter(torch.empty(dim1, rank))
        self.V = nn.Parameter(torch.empty(dim2, rank))
        self.W_out = nn.Parameter(torch.empty(output_dim, rank))
        
        nn.init.xavier_uniform_(self.U, gain=0.1)
        nn.init.xavier_uniform_(self.V, gain=0.1)
        nn.init.xavier_uniform_(self.W_out, gain=0.1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim1)
            y: (..., dim2)
        Returns:
            (..., output_dim)
        """
        # Project to rank space
        x_proj = torch.matmul(x, self.U)  # (..., rank)
        y_proj = torch.matmul(y, self.V)  # (..., rank)
        
        # Element-wise product (bilinear in original space)
        interaction = x_proj * y_proj  # (..., rank)
        
        # Project to output
        out = F.linear(interaction, self.W_out)
        return out


class GeometricInteraction(nn.Module):
    """
    Geometric interaction using complex-valued operations.
    Models phase relationships between signals.
    """
    
    def __init__(self, dim: int, num_relations: int = 4):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations
        
        # Relation-specific phase rotations
        self.phase_rotations = nn.Parameter(torch.empty(num_relations, dim))
        # Relation weights
        self.relation_weights = nn.Parameter(torch.empty(num_relations, dim))
        
        nn.init.uniform_(self.phase_rotations, -torch.pi/4, torch.pi/4)
        nn.init.xavier_uniform_(self.relation_weights)
    
    def forward(self, mag: torch.Tensor, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply geometric interactions.
        
        Args:
            mag: (..., dim) magnitudes
            phase: (..., dim) phases
            
        Returns:
            (new_mag, new_phase)
        """
        # Convert to complex
        z_real = mag * torch.cos(phase)
        z_imag = mag * torch.sin(phase)
        
        # Apply each relation
        outputs_real = []
        outputs_imag = []
        
        for r in range(self.num_relations):
            # Rotate phase
            rot = self.phase_rotations[r]
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            
            rotated_real = z_real * cos_rot - z_imag * sin_rot
            rotated_imag = z_real * sin_rot + z_imag * cos_rot
            
            # Weight
            w = self.relation_weights[r]
            outputs_real.append(rotated_real * w)
            outputs_imag.append(rotated_imag * w)
        
        # Sum relations
        out_real = torch.stack(outputs_real, dim=0).sum(dim=0)
        out_imag = torch.stack(outputs_imag, dim=0).sum(dim=0)
        
        # Back to polar
        new_mag = torch.sqrt(out_real**2 + out_imag**2 + 1e-8)
        new_phase = torch.atan2(out_imag, out_real)
        
        return new_mag, new_phase