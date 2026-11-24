"""Rotated MNIST benchmark for testing geometric representations."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import math
import random


class RotatedMNIST(Dataset):
    """
    Rotated MNIST - tests geometric representations and transformation robustness.
    
    Each digit is rotated by a random angle. The network must learn 
    rotation-invariant features or use phase to encode orientation.
    """
    
    def __init__(
        self, 
        train: bool = True,
        max_angle: float = 180.0,
        n_samples: int = None,
        download: bool = True,
    ):
        self.max_angle = max_angle
        
        # Base MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.mnist = datasets.MNIST(
            root='./data', 
            train=train, 
            download=download, 
            transform=transform
        )
        
        if n_samples is not None:
            indices = torch.randperm(len(self.mnist))[:n_samples]
            self.mnist = torch.utils.data.Subset(self.mnist, indices)
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # Random rotation
        angle = random.uniform(-self.max_angle, self.max_angle)
        angle_rad = math.radians(angle)
        
        # Rotate image (img is 1x28x28)
        img = img.unsqueeze(0)  # Add batch dim: 1x1x28x28
        
        # Create rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Apply affine transformation
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        rotated = F.grid_sample(img, grid, align_corners=False)
        
        return rotated.squeeze(0), label, torch.tensor(angle_rad, dtype=torch.float32)


class RotatedMNISTFixed(Dataset):
    """
    Rotated MNIST with fixed rotation angles per sample.
    Useful for testing whether the network learns rotation-invariant features
    or encodes rotation in phase.
    """
    
    def __init__(
        self, 
        train: bool = True,
        angles: list = None,  # List of angles in degrees
        n_samples: int = None,
        download: bool = True,
    ):
        if angles is None:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        self.angles = [math.radians(a) for a in angles]
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        self.mnist = datasets.MNIST(
            root='./data', 
            train=train, 
            download=download, 
            transform=transform
        )
        
        if n_samples is not None:
            indices = torch.randperm(len(self.mnist))[:n_samples]
            self.mnist = torch.utils.data.Subset(self.mnist, indices)
        
        # Pre-assign random angle to each sample
        self.assigned_angles = torch.randint(0, len(self.angles), (len(self.mnist),))
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        angle = self.angles[self.assigned_angles[idx]]
        
        img = img.unsqueeze(0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        rotated = F.grid_sample(img, grid, align_corners=False)
        
        return rotated.squeeze(0), label, torch.tensor(angle, dtype=torch.float32)


def get_rotated_mnist_loaders(
    max_angle: float = 180.0,
    batch_size: int = 128,
    n_samples: int = 50000,
):
    """Get train/val/test loaders for Rotated MNIST."""
    train_dataset = RotatedMNIST(train=True, max_angle=max_angle, n_samples=n_samples)
    test_dataset = RotatedMNIST(train=False, max_angle=max_angle, n_samples=n_samples//5)
    
    # Split train into train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def get_rotated_mnist_fixed_loaders(
    angles: list = None,
    batch_size: int = 128,
    n_samples: int = 50000,
):
    """Get train/val/test loaders for Fixed-angle Rotated MNIST."""
    train_dataset = RotatedMNISTFixed(train=True, angles=angles, n_samples=n_samples)
    test_dataset = RotatedMNISTFixed(train=False, angles=angles, n_samples=n_samples//5)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )