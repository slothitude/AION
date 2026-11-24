"""Layer 3: Competition - Local inhibition and adaptive competition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CompetitionLayer(nn.Module):
    """
    Competition Layer - Layer 3
    
    Neurons compete within groups; useful patterns suppress irrelevant ones.
    Implements adaptive temperature softmax competition.
    
    Key features:
    - Local competition groups (not global softmax)
    - Adaptive temperature (τ) per group
    - Sparse activity emergence
    - Winner-take-all or soft competition modes
    """
    
    def __init__(
        self,
        dim: int,
        group_size: int = 8,
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        temp_min: float = 0.1,
        temp_max: float = 10.0,
        competition_mode: str = "softmax",  # "softmax", "wta", "sparsemax"
        learnable_groups: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.group_size = group_size
        self.num_groups = dim // group_size
        assert dim % group_size == 0, f"dim {dim} must be divisible by group_size {group_size}"
        
        self.adaptive_temperature = adaptive_temperature
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.competition_mode = competition_mode
        
        if adaptive_temperature:
            # Learnable log-temperature per group
            self.log_temp = nn.Parameter(torch.full((self.num_groups,), math.log(initial_temperature)))
        else:
            self.register_buffer("log_temp", torch.full((self.num_groups,), math.log(initial_temperature)))
        
        if learnable_groups:
            # Learnable grouping via attention
            self.group_assignment = nn.Parameter(torch.empty(dim, self.num_groups))
            nn.init.xavier_uniform_(self.group_assignment)
        else:
            self.register_parameter("group_assignment", None)
        
        # Statistics for adaptive temperature
        self.register_buffer("activation_stats", torch.zeros(self.num_groups, 2))  # mean, std
        self.register_buffer("update_count", torch.zeros(self.num_groups))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply local competition.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Competed output of shape (..., dim)
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.dim)  # (batch*, dim)
        batch_size = x_flat.shape[0]
        
        if self.group_assignment is not None:
            # Soft group assignment
            group_weights = F.softmax(self.group_assignment, dim=1)  # (dim, num_groups)
            # This is more complex - for now use fixed groups
            pass
        
        # Reshape to groups: (batch*, num_groups, group_size)
        x_grouped = x_flat.view(batch_size, self.num_groups, self.group_size)
        
        # Get temperatures
        temps = torch.exp(self.log_temp.clamp(math.log(self.temp_min), math.log(self.temp_max)))
        
        # Apply competition per group
        if self.competition_mode == "softmax":
            out_grouped = self._softmax_competition(x_grouped, temps)
        elif self.competition_mode == "wta":
            out_grouped = self._wta_competition(x_grouped)
        elif self.competition_mode == "sparsemax":
            out_grouped = self._sparsemax_competition(x_grouped)
        else:
            raise ValueError(f"Unknown competition_mode: {self.competition_mode}")
        
        # Flatten back
        out = out_grouped.view(batch_size, self.dim)
        out = out.view(original_shape)
        
        # Update temperature adaptation statistics
        if self.training and self.adaptive_temperature:
            self._update_temperature_stats(x_grouped.detach())
        
        return out
    
    def _softmax_competition(self, x: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        """Softmax with per-group temperature."""
        # x: (batch, num_groups, group_size)
        # temps: (num_groups,)
        temps_expanded = temps.view(1, -1, 1)  # (1, num_groups, 1)
        return F.softmax(x / temps_expanded, dim=-1)
    
    def _wta_competition(self, x: torch.Tensor) -> torch.Tensor:
        """Winner-take-all: only max in each group survives."""
        # x: (batch, num_groups, group_size)
        max_vals, max_indices = x.max(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        out.scatter_(-1, max_indices, max_vals)
        return out
    
    def _sparsemax_competition(self, x: torch.Tensor) -> torch.Tensor:
        """Sparsemax: projects onto probability simplex with sparsity."""
        # Sparsemax implementation
        # Sort descending
        sorted_x, _ = torch.sort(x, dim=-1, descending=True)
        # Cumulative sum
        cumsum = sorted_x.cumsum(dim=-1)
        # k = max { k : 1 + k * sorted_x_k > cumsum_k }
        k = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
        k = k.view(1, 1, -1)
        support = 1 + k * sorted_x > cumsum
        k_max = support.sum(dim=-1, keepdim=True).clamp(min=1)
        # Threshold
        tau = (cumsum.gather(-1, k_max - 1) - 1) / k_max
        # Output
        out = torch.clamp(x - tau, min=0)
        return out
    
    def _update_temperature_stats(self, x_grouped: torch.Tensor):
        """Update running statistics for adaptive temperature."""
        # x_grouped: (batch, num_groups, group_size)
        with torch.no_grad():
            group_mean = x_grouped.mean(dim=(0, 2))  # (num_groups,)
            group_std = x_grouped.std(dim=(0, 2))    # (num_groups,)
            
            # Exponential moving average
            alpha = 0.01
            self.activation_stats[:, 0] = (1 - alpha) * self.activation_stats[:, 0] + alpha * group_mean
            self.activation_stats[:, 1] = (1 - alpha) * self.activation_stats[:, 1] + alpha * group_std
            self.update_count += 1
            
            # Adapt temperature based on activity statistics
            # High mean + low std -> increase temperature (more uniform)
            # Low mean + high std -> decrease temperature (more selective)
            if self.update_count[0] > 100:  # Warmup
                selectivity = self.activation_stats[:, 1] / (self.activation_stats[:, 0] + 1e-8)
                # Normalize selectivity to [0, 1]
                selectivity_norm = (selectivity - selectivity.min()) / (selectivity.max() - selectivity.min() + 1e-8)
                # Map to temperature range
                target_log_temp = math.log(self.temp_min) + selectivity_norm * (math.log(self.temp_max) - math.log(self.temp_min))
                # Slow adaptation
                self.log_temp.data = 0.99 * self.log_temp.data + 0.01 * target_log_temp
    
    def get_temperatures(self) -> torch.Tensor:
        """Get current temperatures per group."""
        return torch.exp(self.log_temp.clamp(math.log(self.temp_min), math.log(self.temp_max)))
    
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, group_size={self.group_size}, "
                f"mode={self.competition_mode}, adaptive={self.adaptive_temperature}")


class GeometricCompetitionLayer(nn.Module):
    """
    Competition layer for geometric (magnitude-phase) neurons.
    Competition happens in magnitude space; phase is preserved for winners.
    """
    
    def __init__(
        self,
        dim: int,
        group_size: int = 8,
        initial_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        temp_min: float = 0.1,
        temp_max: float = 10.0,
        competition_mode: str = "softmax",
    ):
        super().__init__()
        self.dim = dim
        self.group_size = group_size
        self.num_groups = dim // group_size
        
        self.competition = CompetitionLayer(
            dim, group_size, initial_temperature, adaptive_temperature, 
            temp_min, temp_max, competition_mode
        )
    
    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply competition to magnitudes, preserve phase of winners.
        
        Args:
            magnitude: (..., dim)
            phase: (..., dim)
            
        Returns:
            (competed_magnitude, competed_phase)
        """
        original_shape = magnitude.shape
        mag_flat = magnitude.view(-1, self.dim)
        phase_flat = phase.view(-1, self.dim)
        
        # Apply competition to magnitudes
        competed_mag = self.competition(mag_flat)
        
        # For softmax: phase is weighted average
        # For WTA: phase of winner
        if self.competition.competition_mode == "wta":
            # Find winners
            mag_grouped = mag_flat.view(-1, self.num_groups, self.group_size)
            _, winner_idx = mag_grouped.max(dim=-1)  # (batch, num_groups)
            
            # Gather winner phases
            phase_grouped = phase_flat.view(-1, self.num_groups, self.group_size)
            batch_idx = torch.arange(phase_grouped.shape[0], device=phase.device).unsqueeze(1)
            group_idx = torch.arange(self.num_groups, device=phase.device).unsqueeze(0)
            competed_phase_flat = phase_grouped[batch_idx, group_idx, winner_idx]
        else:
            # Soft competition: phase weighted by magnitude
            # Convert to complex, weight, convert back
            z_real = mag_flat * torch.cos(phase_flat)
            z_imag = mag_flat * torch.sin(phase_flat)
            
            # Apply same competition weights to complex
            competed_real = self.competition(z_real)
            competed_imag = self.competition(z_imag)
            
            competed_mag = torch.sqrt(competed_real**2 + competed_imag**2 + 1e-8)
            competed_phase_flat = torch.atan2(competed_imag, competed_real)
        
        competed_mag = competed_mag.view(original_shape)
        competed_phase = competed_phase_flat.view(original_shape)
        
        return competed_mag, competed_phase