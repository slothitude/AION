"""XOR and Parity benchmark datasets."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class XORDataset(Dataset):
    """
    XOR dataset - tests whether neurons can discover combinations of signals.
    
    XOR is the classic test for non-linear separability. A neuron with 
    multiplicative interactions should be able to solve it with fewer layers.
    """
    
    def __init__(self, n_samples: int = 10000, noise: float = 0.0):
        self.n_samples = n_samples
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        # All 4 XOR combinations
        X = torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
        ], dtype=torch.float32)
        
        y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        
        # Repeat to get desired sample count
        repeats = self.n_samples // 4 + 1
        X = X.repeat(repeats, 1)[:self.n_samples]
        y = y.repeat(repeats)[:self.n_samples]
        
        # Add noise if specified
        if self.noise > 0:
            X = X + torch.randn_like(X) * self.noise
            X = torch.clamp(X, 0, 1)
        
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ParityDataset(Dataset):
    """
    Parity dataset - computes parity (even/odd) of k binary inputs.
    
    This tests whether the network can compose multiple XOR-like operations.
    With k inputs, there are 2^k possible patterns.
    """
    
    def __init__(self, n_inputs: int = 4, n_samples: int = 10000, noise: float = 0.0):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        # Generate all possible binary combinations
        n_patterns = min(2 ** self.n_inputs, self.n_samples)
        
        # Generate random binary patterns
        X = torch.randint(0, 2, (self.n_samples, self.n_inputs), dtype=torch.float32)
        
        # Parity: 1 if odd number of 1s, 0 if even
        y = (X.sum(dim=1) % 2).long()
        
        if self.noise > 0:
            X = X + torch.randn_like(X) * self.noise
            X = torch.clamp(X, 0, 1)
        
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiBitXORDataset(Dataset):
    """
    Multi-bit XOR - XOR of multiple pairs.
    
    Tests whether the network can learn multiple independent XOR operations
    simultaneously and combine them.
    """
    
    def __init__(self, n_pairs: int = 3, n_samples: int = 10000):
        self.n_pairs = n_pairs
        self.n_samples = n_samples
        self._generate_data()
    
    def _generate_data(self):
        # Each pair is an independent XOR
        X = torch.randint(0, 2, (self.n_samples, self.n_pairs * 2), dtype=torch.float32)
        
        # Compute XOR for each pair
        y = torch.zeros(self.n_samples, self.n_pairs, dtype=torch.long)
        for i in range(self.n_pairs):
            y[:, i] = (X[:, 2*i] != X[:, 2*i + 1]).long()
        
        # Multi-label classification: predict all XORs
        # For simplicity, use single label: parity of all XOR results
        y = (y.sum(dim=1) % 2).long()
        
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_xor_loaders(batch_size: int = 128, n_samples: int = 10000, noise: float = 0.0):
    """Get train/val/test loaders for XOR."""
    dataset = XORDataset(n_samples=n_samples, noise=noise)
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


def get_parity_loaders(n_inputs: int = 4, batch_size: int = 128, n_samples: int = 10000, noise: float = 0.0):
    """Get train/val/test loaders for Parity."""
    dataset = ParityDataset(n_inputs=n_inputs, n_samples=n_samples, noise=noise)
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