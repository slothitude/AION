"""Sequential prediction and pattern completion benchmarks."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequentialPrediction(Dataset):
    """
    Sequential next-step prediction benchmark.
    
    Tests fast/slow memory and temporal integration.
    Sequences are generated from simple dynamical systems.
    """
    
    def __init__(
        self,
        seq_length: int = 20,
        n_sequences: int = 5000,
        system: str = "sine",  # "sine", "lorenz", "random_walk", "markov"
        input_dim: int = 1,
        noise: float = 0.0,
    ):
        self.seq_length = seq_length
        self.n_sequences = n_sequences
        self.system = system
        self.input_dim = input_dim
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        sequences = []
        targets = []
        
        for _ in range(self.n_sequences):
            if self.system == "sine":
                # Multiple sine waves with different frequencies
                t = torch.linspace(0, 4*math.pi, self.seq_length + 1)
                freq = torch.rand(1) * 2 + 0.5  # 0.5 to 2.5
                phase = torch.rand(1) * 2 * math.pi
                seq = torch.sin(freq * t + phase).unsqueeze(-1)
                
            elif self.system == "lorenz":
                # Lorenz system (chaotic)
                dt = 0.01
                x, y, z = 1.0, 1.0, 1.0
                seq = []
                for _ in range(self.seq_length + 1):
                    seq.append([x])
                    dx = 10 * (y - x) * dt
                    dy = (x * (28 - z) - y) * dt
                    dz = (x * y - 8/3 * z) * dt
                    x += dx; y += dy; z += dz
                seq = torch.tensor(seq, dtype=torch.float32)
                
            elif self.system == "random_walk":
                seq = torch.cumsum(torch.randn(self.seq_length + 1, 1) * 0.1, dim=0)
                
            elif self.system == "markov":
                # Simple 2-state Markov chain with continuous observations
                states = torch.randint(0, 2, (self.seq_length + 1,))
                seq = states.float().unsqueeze(-1) + torch.randn(self.seq_length + 1, 1) * 0.1
                
            else:
                raise ValueError(f"Unknown system: {self.system}")
            
            if self.noise > 0:
                seq = seq + torch.randn_like(seq) * self.noise
            
            # Input is all but last, target is all but first (next-step prediction)
            sequences.append(seq[:-1])
            targets.append(seq[1:])
        
        self.sequences = torch.stack(sequences)
        self.targets = torch.stack(targets)
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class NoisyPatternCompletion(Dataset):
    """
    Noisy pattern completion benchmark.
    
    Tests whether the network can reconstruct missing information from relationships.
    Given a corrupted pattern, predict the clean version.
    """
    
    def __init__(
        self,
        pattern_dim: int = 64,
        n_patterns: int = 1000,
        n_samples_per_pattern: int = 10,
        corruption_rate: float = 0.3,
        noise: float = 0.1,
    ):
        self.pattern_dim = pattern_dim
        self.corruption_rate = corruption_rate
        self.noise = noise
        self._generate_data(n_patterns, n_samples_per_pattern)
    
    def _generate_data(self, n_patterns, n_samples_per_pattern):
        # Generate base patterns (e.g., sparse codes, structured patterns)
        base_patterns = torch.randn(n_patterns, self.pattern_dim)
        # Make them structured: sparse + smooth
        base_patterns = torch.sign(base_patterns) * torch.relu(torch.abs(base_patterns) - 0.5)
        
        clean_patterns = []
        corrupted_patterns = []
        
        for i in range(n_patterns):
            for _ in range(n_samples_per_pattern):
                clean = base_patterns[i]
                
                # Add noise
                if self.noise > 0:
                    clean = clean + torch.randn_like(clean) * self.noise
                
                # Corrupt: randomly zero out elements
                mask = torch.rand(self.pattern_dim) > self.corruption_rate
                corrupted = clean * mask.float()
                
                clean_patterns.append(clean)
                corrupted_patterns.append(corrupted)
        
        self.clean = torch.stack(clean_patterns)
        self.corrupted = torch.stack(corrupted_patterns)
    
    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, idx):
        return self.corrupted[idx], self.clean[idx]


class SequenceClassification(Dataset):
    """
    Sequence classification benchmark.
    
    Tests whether the network can integrate information over time
    to classify the generating process.
    """
    
    def __init__(
        self,
        seq_length: int = 50,
        n_sequences: int = 5000,
        n_classes: int = 4,
    ):
        self.seq_length = seq_length
        self.n_classes = n_classes
        self._generate_data(n_sequences)
    
    def _generate_data(self, n_sequences):
        sequences = []
        labels = []
        
        # Class 0: Sine wave
        # Class 1: Square wave
        # Class 2: Sawtooth
        # Class 3: Random walk
        
        for class_idx in range(self.n_classes):
            n_per_class = n_sequences // self.n_classes
            for _ in range(n_per_class):
                t = torch.linspace(0, 4*math.pi, self.seq_length)
                
                if class_idx == 0:
                    # Sine with random freq/phase
                    freq = torch.rand(1) * 2 + 0.5
                    phase = torch.rand(1) * 2 * math.pi
                    seq = torch.sin(freq * t + phase)
                elif class_idx == 1:
                    # Square wave
                    freq = torch.rand(1) * 2 + 0.5
                    phase = torch.rand(1) * 2 * math.pi
                    seq = torch.sign(torch.sin(freq * t + phase))
                elif class_idx == 2:
                    # Sawtooth
                    freq = torch.rand(1) * 2 + 0.5
                    phase = torch.rand(1) * 2 * math.pi
                    seq = 2 * ((freq * t + phase) / (2 * math.pi) - torch.floor((freq * t + phase) / (2 * math.pi))) - 1
                else:
                    # Random walk
                    seq = torch.cumsum(torch.randn(self.seq_length) * 0.1, dim=0)
                
                seq = seq.unsqueeze(-1)  # Add feature dim
                sequences.append(seq)
                labels.append(class_idx)
        
        self.sequences = torch.stack(sequences)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def get_sequential_loaders(
    system: str = "sine",
    seq_length: int = 20,
    batch_size: int = 64,
    n_sequences: int = 5000,
):
    """Get loaders for sequential prediction."""
    dataset = SequentialPrediction(
        seq_length=seq_length, 
        n_sequences=n_sequences, 
        system=system
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


def get_pattern_completion_loaders(
    pattern_dim: int = 64,
    batch_size: int = 64,
    n_patterns: int = 1000,
):
    """Get loaders for noisy pattern completion."""
    dataset = NoisyPatternCompletion(
        pattern_dim=pattern_dim,
        n_patterns=n_patterns,
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


def get_sequence_classification_loaders(
    seq_length: int = 50,
    batch_size: int = 64,
    n_sequences: int = 5000,
):
    """Get loaders for sequence classification."""
    dataset = SequenceClassification(
        seq_length=seq_length,
        n_sequences=n_sequences,
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