"""Circular patterns and interference benchmarks for geometric neurons."""

import torch
from torch.utils.data import Dataset, DataLoader
import math


class CircularPatterns(Dataset):
    """
    Circular pattern classification - tests magnitude-phase representations.
    
    Patterns are points on a circle with angular relationships.
    Geometric neurons should naturally handle angular arithmetic.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_points: int = 8,
        n_classes: int = 4,
        noise: float = 0.1,
    ):
        self.n_samples = n_samples
        self.n_points = n_points
        self.n_classes = n_classes
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        X = []
        y = []
        
        samples_per_class = self.n_samples // self.n_classes
        
        for class_idx in range(self.n_classes):
            for _ in range(samples_per_class):
                # Base angle for this class
                base_angle = class_idx * 2 * math.pi / self.n_classes
                
                # Generate n_points with this base angle + noise
                angles = base_angle + torch.randn(self.n_points) * self.noise
                
                # Convert to Cartesian coordinates (input features)
                coords = torch.stack([
                    torch.cos(angles),
                    torch.sin(angles)
                ], dim=-1)  # (n_points, 2)
                
                X.append(coords.flatten())  # Flatten to (n_points * 2,)
                y.append(class_idx)
        
        self.X = torch.stack(X)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class InterferencePatterns(Dataset):
    """
    Interference pattern dataset - tests phase cancellation and reinforcement.
    
    Two waves interfere: constructive (same phase) or destructive (opposite phase).
    Geometric neurons with phase should naturally model this.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_waves: int = 2,
        grid_size: int = 16,
        noise: float = 0.1,
    ):
        self.n_samples = n_samples
        self.n_waves = n_waves
        self.grid_size = grid_size
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        X = []
        y = []  # 0 = destructive, 1 = constructive
        
        # Create spatial grid
        x_coords = torch.linspace(-math.pi, math.pi, self.grid_size)
        y_coords = torch.linspace(-math.pi, math.pi, self.grid_size)
        X_grid, Y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        for _ in range(self.n_samples):
            # Random wave parameters
            is_constructive = torch.randint(0, 2, (1,)).item()
            
            if is_constructive:
                # Same phase (constructive interference)
                phase = torch.rand(1) * 2 * math.pi
                phases = [phase] * self.n_waves
            else:
                # Opposite phase (destructive interference)
                phase = torch.rand(1) * 2 * math.pi
                phases = [phase, phase + math.pi]
            
            # Generate interference pattern
            pattern = torch.zeros(self.grid_size, self.grid_size)
            for i, p in enumerate(phases):
                kx = torch.rand(1) * 2 + 0.5
                ky = torch.rand(1) * 2 + 0.5
                wave = torch.sin(kx * X_grid + ky * Y_grid + p)
                pattern += wave
            
            # Normalize
            pattern = pattern / self.n_waves
            
            if self.noise > 0:
                pattern += torch.randn_like(pattern) * self.noise
            
            X.append(pattern.flatten())
            y.append(is_constructive)
        
        self.X = torch.stack(X)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PhaseRotationDataset(Dataset):
    """
    Phase rotation dataset - tests whether geometric neurons track rotations.
    
    Input: a complex vector (magnitude + phase)
    Target: the same vector rotated by a known angle
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        dim: int = 32,
        noise: float = 0.0,
    ):
        self.n_samples = n_samples
        self.dim = dim
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        X = []
        y_angles = []
        y_mags = []
        
        for _ in range(self.n_samples):
            # Random complex vector
            mag = torch.rand(self.dim) + 0.5  # [0.5, 1.5]
            phase = torch.rand(self.dim) * 2 * math.pi
            
            # Random rotation angle
            rotation = torch.rand(1) * 2 * math.pi
            
            # Target: rotated phase
            target_phase = (phase + rotation) % (2 * math.pi)
            
            # Input: magnitude and phase concatenated
            input_vec = torch.cat([mag, phase])
            
            # Target: magnitude and rotated phase
            target_vec = torch.cat([mag, target_phase])
            
            X.append(input_vec)
            y_mags.append(mag)
            y_angles.append(rotation)
        
        self.X = torch.stack(X)
        self.y_angles = torch.stack(y_angles)
        self.y_mags = torch.stack(y_mags)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_mags[idx], self.y_angles[idx]


def get_circular_loaders(
    n_points: int = 8,
    n_classes: int = 4,
    batch_size: int = 128,
    n_samples: int = 10000,
    noise: float = 0.1,
):
    """Get loaders for circular patterns."""
    dataset = CircularPatterns(
        n_samples=n_samples,
        n_points=n_points,
        n_classes=n_classes,
        noise=noise,
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def get_interference_loaders(
    n_waves: int = 2,
    grid_size: int = 16,
    batch_size: int = 128,
    n_samples: int = 10000,
    noise: float = 0.1,
):
    """Get loaders for interference patterns."""
    dataset = InterferencePatterns(
        n_samples=n_samples,
        n_waves=n_waves,
        grid_size=grid_size,
        noise=noise,
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def get_phase_rotation_loaders(
    dim: int = 32,
    batch_size: int = 128,
    n_samples: int = 10000,
    noise: float = 0.0,
):
    """Get loaders for phase rotation."""
    dataset = PhaseRotationDataset(
        n_samples=n_samples,
        dim=dim,
        noise=noise,
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )