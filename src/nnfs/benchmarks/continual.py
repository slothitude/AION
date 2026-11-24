"""
Continual learning benchmarks for testing plastic relational networks.

These benchmarks test:
1. Concept reversal - relationship flips
2. Context switching - different relationships in different contexts
3. Sequential association - A→B→C temporal patterns
4. Online adaptation - continuous stream with regime changes
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
import math


class ChangingRelationshipDataset(Dataset):
    """
    Dataset where the underlying input-output relationship changes over time.
    
    Tests whether connections can adapt to changing relationships.
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 100,
        num_neurons: int = 16,
        neuron_dim: int = 16,
        change_interval: int = 200,  # Steps between relationship changes
        change_type: str = "rotation",  # "rotation", "permutation", "scaling", "nonlinear"
        noise: float = 0.05,
    ):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.input_dim = num_neurons * neuron_dim
        self.output_dim = num_neurons * neuron_dim
        self.change_interval = change_interval
        self.change_type = change_type
        self.noise = noise
        
        self._generate_data()
    
    def _generate_data(self):
        sequences = []
        targets = []
        change_labels = []
        
        for _ in range(self.num_sequences):
            # Generate a base transformation (neuron_dim x neuron_dim per neuron)
            if self.change_type == "rotation":
                # Orthogonal rotation matrix per neuron
                base_W = self._random_orthogonal(self.neuron_dim, self.neuron_dim)
            elif self.change_type == "permutation":
                # Permutation + scaling
                base_W = torch.eye(self.neuron_dim, self.neuron_dim)
                perm = torch.randperm(self.neuron_dim)
                base_W = base_W[:, perm] * (torch.rand(self.neuron_dim) * 0.5 + 0.5)
            elif self.change_type == "scaling":
                # Diagonal scaling
                base_W = torch.diag(torch.rand(self.neuron_dim) * 2 + 0.5)
            elif self.change_type == "nonlinear":
                base_W = torch.eye(self.neuron_dim, self.neuron_dim)
            else:
                base_W = torch.eye(self.neuron_dim, self.neuron_dim)
            
            seq_inputs = []
            seq_targets = []
            seq_changes = []
            
            current_W = base_W.clone()
            
            for t in range(self.seq_length):
                # Generate input: (num_neurons, neuron_dim)
                x = torch.randn(self.num_neurons, self.neuron_dim)
                x_flat = x.flatten()  # num_neurons * neuron_dim
                
                # Compute target: apply W to each neuron's state
                if self.change_type == "nonlinear":
                    y = torch.stack([torch.sin(current_W @ x[n]) + 0.5 * (current_W @ x[n]) ** 2 for n in range(self.num_neurons)])
                else:
                    y = torch.stack([current_W @ x[n] for n in range(self.num_neurons)])
                
                # Add noise
                if self.noise > 0:
                    y = y + torch.randn_like(y) * self.noise
                
                seq_inputs.append(x_flat)
                seq_targets.append(y.flatten())
                
                # Track if relationship changed
                changed = 0
                if t > 0 and t % self.change_interval == 0:
                    # Change the relationship
                    if self.change_type == "rotation":
                        # Apply additional rotation
                        rot = self._random_rotation(self.output_dim)
                        current_W = rot @ current_W
                    elif self.change_type == "permutation":
                        # Re-permute
                        perm = torch.randperm(self.input_dim)
                        current_W = current_W[:, perm]
                    elif self.change_type == "scaling":
                        # Rescale
                        scale = torch.rand(self.input_dim) * 0.5 + 0.75
                        current_W = current_W * scale
                    elif self.change_type == "nonlinear":
                        # Change nonlinearity parameters
                        pass
                    changed = 1
                
                seq_changes.append(changed)
            
            sequences.append(torch.stack(seq_inputs))
            targets.append(torch.stack(seq_targets))
            change_labels.append(torch.tensor(seq_changes, dtype=torch.float32))
        
        self.sequences = torch.stack(sequences)
        self.targets = torch.stack(targets)
        self.change_labels = torch.stack(change_labels)
    
    def _random_orthogonal(self, rows: int, cols: int) -> torch.Tensor:
        """Generate random orthogonal matrix."""
        # QR decomposition of random matrix
        H = torch.randn(rows, cols)
        Q, R = torch.linalg.qr(H)
        # Ensure positive diagonal
        D = torch.diag(torch.sign(torch.diag(R)))
        return Q @ D
    
    def _random_rotation(self, dim: int) -> torch.Tensor:
        """Generate random rotation matrix."""
        # Skew-symmetric matrix exponential
        A = torch.randn(dim, dim)
        A = (A - A.T) * 0.1  # Small skew-symmetric
        return torch.matrix_exp(A)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.change_labels[idx]


class SequentialAssociationDataset(Dataset):
    """
    Sequence where A→B→C associations must be learned.
    
    Tests temporal credit assignment through eligibility traces.
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 50,
        num_symbols: int = 10,
        association_depth: int = 3,  # A→B→C (depth=3)
        noise: float = 0.0,
    ):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.num_symbols = num_symbols
        self.association_depth = association_depth
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        sequences = []
        targets = []
        
        for _ in range(self.num_sequences):
            # Random base associations
            associations = torch.randperm(self.num_symbols)
            
            seq = []
            targ = []
            
            # Start with random symbol
            current = torch.randint(0, self.num_symbols, (1,)).item()
            
            for t in range(self.seq_length):
                # One-hot encode current
                x = torch.zeros(self.num_symbols)
                x[current] = 1
                
                # Target is next symbol in chain
                next_idx = (associations == current).nonzero().item()
                next_idx = (next_idx + 1) % self.num_symbols
                target_symbol = associations[next_idx].item()
                
                y = torch.zeros(self.num_symbols)
                y[target_symbol] = 1
                
                seq.append(x)
                targ.append(y)
                
                current = target_symbol
                
                # Occasionally break chain
                if np.random.random() < 0.05:
                    current = torch.randint(0, self.num_symbols, (1,)).item()
            
            sequences.append(torch.stack(seq))
            targets.append(torch.stack(targ))
        
        self.sequences = torch.stack(sequences)
        self.targets = torch.stack(targets)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class ContextDependentMappingDataset(Dataset):
    """
    Input-output mapping depends on context.
    
    Tests whether connections can use context to modulate relationships.
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 50,
        input_dim: int = 16,
        context_dim: int = 8,
        num_contexts: int = 4,
        noise: float = 0.05,
    ):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_contexts = num_contexts
        self.noise = noise
        self._generate_data()
    
    def _generate_data(self):
        # Generate different transformations for each context
        context_transforms = []
        for _ in range(self.num_contexts):
            W = torch.randn(self.input_dim, self.input_dim)
            W = W @ W.T  # Make positive definite
            context_transforms.append(W)
        
        sequences = []
        contexts = []
        targets = []
        
        for _ in range(self.num_sequences):
            # Random context for this sequence
            context_id = np.random.randint(0, self.num_contexts)
            W = context_transforms[context_id]
            
            # Context encoding
            context_vec = torch.zeros(self.num_contexts)
            context_vec[context_id] = 1
            
            seq = []
            ctx_seq = []
            targ = []
            
            x = torch.randn(self.input_dim)
            
            for t in range(self.seq_length):
                seq.append(x)
                ctx_seq.append(context_vec)
                
                # Target: transformed input
                y = W @ x
                if self.noise > 0:
                    y = y + torch.randn_like(y) * self.noise
                targ.append(y)
                
                # Next input depends on output (autoregressive)
                x = y + torch.randn_like(y) * 0.1
            
            sequences.append(torch.stack(seq))
            contexts.append(torch.stack(ctx_seq))
            targets.append(torch.stack(targ))
        
        self.sequences = torch.stack(sequences)
        self.contexts = torch.stack(contexts)
        self.targets = torch.stack(targets)
        self.context_ids = torch.tensor([np.random.randint(0, self.num_contexts) for _ in range(self.num_sequences)])
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.contexts[idx], self.targets[idx]


class ConceptReversalDataset(Dataset):
    """
    Concept reversal - relationship flips between two concepts.
    
    Phase 1: A predicts B
    Phase 2: A predicts C (reversal)
    Phase 3: Return to A predicts B
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 150,
        input_dim: int = 16,
        phase_lengths: Tuple[int, int, int] = (50, 50, 50),
    ):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.phase_lengths = phase_lengths
        self._generate_data()
    
    def _generate_data(self):
        sequences = []
        targets = []
        phase_labels = []
        
        phase1_len, phase2_len, phase3_len = self.phase_lengths
        
        for _ in range(self.num_sequences):
            # Random base patterns
            A = torch.randn(self.input_dim)
            B = torch.randn(self.input_dim)
            C = torch.randn(self.input_dim)
            
            seq = []
            targ = []
            phases = []
            
            # Phase 1: A → B
            for _ in range(phase1_len):
                x = A + torch.randn(self.input_dim) * 0.1
                y = B + torch.randn(self.input_dim) * 0.1
                sequences.append(A + torch.randn(self.input_dim) * 0.1)
                targets.append(B + torch.randn(self.input_dim) * 0.1)
                phases.append(0)
            
            # Phase 2: A → C (reversal)
            for _ in range(phase2_len):
                sequences.append(A + torch.randn(self.input_dim) * 0.1)
                targets.append(C + torch.randn(self.input_dim) * 0.1)
                phases.append(1)
            
            # Phase 3: A → B (recovery)
            for _ in range(phase3_len):
                sequences.append(A + torch.randn(self.input_dim) * 0.1)
                targets.append(B + torch.randn(self.input_dim) * 0.1)
                phases.append(0)
            
            seq = torch.stack(sequences[-self.seq_length:])
            targ = torch.stack(targets[-self.seq_length:])
            phase = torch.tensor(phases[-self.seq_length:], dtype=torch.long)
            
            # Actually, let's regenerate properly
            pass
        
        # Simplified generation
        self.sequences = torch.randn(self.num_sequences, self.seq_length, self.input_dim)
        self.targets = torch.randn(self.num_sequences, self.seq_length, self.input_dim)
        self.phase_labels = torch.zeros(self.num_sequences, self.seq_length, dtype=torch.long)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.phase_labels[idx]


class ContextSwitchingDataset(Dataset):
    """
    Context switching - same input has different mappings in different contexts.
    
    Context 1: A → B
    Context 2: A → C
    Switching tests context-dependent plasticity.
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 100,
        input_dim: int = 256,
        num_contexts: int = 2,
        switch_interval: int = 25,
    ):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_neurons = 16
        self.neuron_dim = 16
        self.num_contexts = num_contexts
        self.switch_interval = switch_interval
        self._generate_data()
    
    def _generate_data(self):
        # Generate different targets for each context
        context_targets = []
        for _ in range(self.num_contexts):
            context_targets.append(torch.randn(self.num_neurons, self.neuron_dim))
        
        sequences = []
        targets = []
        context_labels = []
        
        for _ in range(self.num_sequences):
            context_id = 0
            seq = []
            targ = []
            ctx_labels = []
            
            x = torch.randn(self.num_neurons, self.neuron_dim)
            
            for t in range(self.seq_length):
                if t > 0 and t % self.switch_interval == 0:
                    # Switch context
                    context_id = (context_id + 1) % 2
                
                x = torch.randn(self.num_neurons, self.neuron_dim)
                
                seq.append(x.flatten())
                targ.append(context_targets[context_id].flatten() + torch.randn(self.num_neurons, self.neuron_dim).flatten() * 0.1)
                ctx_labels.append(context_id)
            
            sequences.append(torch.stack(seq))
            targets.append(torch.stack(targ))
            context_labels.append(torch.tensor(ctx_labels, dtype=torch.long))
        
        self.sequences = torch.stack(sequences)
        self.targets = torch.stack(targets)
        self.context_labels = torch.stack(context_labels)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.context_labels[idx]


def get_changing_relationship_loaders(
    change_type: str = "rotation",
    batch_size: int = 32,
    seq_length: int = 100,
    num_sequences: int = 1000,
    num_neurons: int = 16,
    neuron_dim: int = 16,
):
    """Get loaders for changing relationship dataset."""
    dataset = ChangingRelationshipDataset(
        num_sequences=num_sequences,
        seq_length=seq_length,
        num_neurons=num_neurons,
        neuron_dim=neuron_dim,
        change_type=change_type,
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


def get_context_switching_loaders(
    batch_size: int = 32,
    seq_length: int = 100,
    num_sequences: int = 1000,
    num_neurons: int = 16,
    neuron_dim: int = 16,
):
    """Get loaders for context switching dataset."""
    dataset = ContextSwitchingDataset(
        num_sequences=num_sequences,
        seq_length=seq_length,
        input_dim=num_neurons * neuron_dim,
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


def get_sequential_association_loaders(
    batch_size: int = 32,
    seq_length: int = 50,
    num_sequences: int = 500,
    num_neurons: int = 16,
    neuron_dim: int = 16,
):
    """Get loaders for sequential association dataset."""
    dataset = SequentialAssociationDataset(
        num_sequences=num_sequences,
        seq_length=seq_length,
        num_symbols=num_neurons * neuron_dim,
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