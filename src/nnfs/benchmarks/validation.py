"""
Scientifically valid benchmarks for NNFS validation.

Each benchmark has known ground truth and learnable structure.
Designed to test specific capabilities: adaptation, retention, discovery, self-organisation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class BenchmarkConfig:
    """Configuration for validation benchmarks."""
    num_neurons: int = 16
    neuron_dim: int = 16
    seq_length: int = 100
    num_sequences: int = 100
    batch_size: int = 16
    noise: float = 0.05
    phase_length: int = 300


class ChangingRotationDataset(Dataset):
    """
    Benchmark 1: Changing relationships with known ground truth.
    
    Phase structure:
    - Phase A (steps 0-299): 30° rotation
    - Phase B (steps 300-599): -45° rotation  
    - Phase C (steps 600-899): Return to 30°
    
    The rotation is applied per-neuron (neuron_dim x neuron_dim).
    This is learnable because the transformation is stable within each phase.
    """
    
    def __init__(
        self,
        num_neurons: int = 16,
        neuron_dim: int = 16,
        seq_length: int = None,  # Defaults to 3 × phase_length
        num_sequences: int = 100,
        noise: float = 0.05,
        phase_length: int = 300,
    ):
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.phase_length = phase_length
        self.seq_length = seq_length if seq_length is not None else 3 * phase_length
        self.num_sequences = num_sequences
        self.noise = noise

        # Phase boundaries
        self.phase_boundaries = [0, phase_length, 2 * phase_length, 3 * phase_length]
        self.phase_angles = [30.0, -45.0, 30.0]  # degrees

        self._generate_data()
    
    def _rotation_matrix(self, angle_deg: float) -> torch.Tensor:
        """Create 2D rotation matrix, extended to neuron_dim."""
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        # Base 2D rotation
        R2 = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        
        # Extend to neuron_dim by tiling 2x2 blocks
        if self.neuron_dim % 2 == 0:
            blocks = self.neuron_dim // 2
            R = torch.block_diag(*[R2 for _ in range(blocks)])
        else:
            # Odd dimension: add 1x1 identity
            R = torch.block_diag(*[R2 for _ in range(self.neuron_dim // 2)], torch.tensor([[1.0]]))
        
        return R
    
    def _generate_data(self):
        # Pre-compute rotation matrices for each phase
        phase_matrices = [self._rotation_matrix(angle) for angle in self.phase_angles]
        
        sequences = []
        targets = []
        phase_labels = []
        
        for _ in range(self.num_sequences):
            seq_inputs = []
            seq_targets = []
            seq_phases = []

            x_prev = None
            for t in range(self.seq_length):
                # Determine phase
                phase = 0
                for i, boundary in enumerate(self.phase_boundaries[1:], 1):
                    if t < boundary:
                        phase = i - 1
                        break
                else:
                    phase = len(self.phase_boundaries) - 2

                R = phase_matrices[phase]

                # Generate input: random per-neuron vectors
                x = torch.randn(self.num_neurons, self.neuron_dim)

                # Target is the previous input under the CURRENT phase's
                # rotation (lag-1): a fresh random x each step makes y = R@x_t
                # informationally unlearnable from the network's state history
                # (connections lane diagnosis, gate 4). First step has no
                # history, so it targets the current input.
                x_src = x if x_prev is None else x_prev
                x_prev = x

                # Apply rotation to each neuron's state
                y = torch.stack([R @ x_src[n] for n in range(self.num_neurons)])

                # Add noise
                if self.noise > 0:
                    y = y + torch.randn_like(y) * self.noise

                seq_inputs.append(x.flatten())
                seq_targets.append(y.flatten())
                seq_phases.append(phase)
            
            sequences.append(torch.stack(seq_inputs))
            targets.append(torch.stack(seq_targets))
            phase_labels.append(torch.tensor(seq_phases, dtype=torch.long))
        
        self.sequences = torch.stack(sequences)  # (num_seq, seq_len, flat_dim)
        self.targets = torch.stack(targets)
        self.phase_labels = torch.stack(phase_labels)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.phase_labels[idx]
    
    def get_phase_boundaries(self) -> List[int]:
        """Return step indices where phases change."""
        return self.phase_boundaries[1:-1]  # Exclude 0 and final


class ContextDependentDataset(Dataset):
    """
    Benchmark 2: Context-dependent relationships.
    
    Same input produces different outputs depending on context.
    Context switches every 200 steps.
    
    Context 1: 30° rotation
    Context 2: 90° rotation (quarter turn)
    
    Tests: contextual plasticity, memory separation, context switching.
    """
    
    def __init__(
        self,
        num_neurons: int = 16,
        neuron_dim: int = 16,
        seq_length: int = 800,
        num_sequences: int = 100,
        switch_interval: int = 200,
        noise: float = 0.05,
    ):
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.seq_length = seq_length
        self.num_sequences = num_sequences
        self.switch_interval = switch_interval
        self.noise = noise
        
        # Context transformations
        self.context_matrices = self._create_context_matrices()
        
        self._generate_data()
    
    def _create_context_matrices(self) -> List[torch.Tensor]:
        """Create rotation matrices for each context."""
        matrices = []
        angles = [30.0, 90.0]  # Context 0: 30°, Context 1: 90°
        
        for angle in angles:
            angle_rad = math.radians(angle)
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            R2 = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            
            if self.neuron_dim % 2 == 0:
                blocks = self.neuron_dim // 2
                R = torch.block_diag(*[R2 for _ in range(blocks)])
            else:
                R = torch.block_diag(*[R2 for _ in range(self.neuron_dim // 2)], torch.tensor([[1.0]]))
            matrices.append(R)
        
        return matrices
    
    def _generate_data(self):
        sequences = []
        targets = []
        context_labels = []
        
        for _ in range(self.num_sequences):
            seq_inputs = []
            seq_targets = []
            seq_contexts = []

            context_id = 0
            x_prev = None

            for t in range(self.seq_length):
                # Switch context periodically
                if t > 0 and t % self.switch_interval == 0:
                    context_id = 1 - context_id

                R = self.context_matrices[context_id]

                # Generate input
                x = torch.randn(self.num_neurons, self.neuron_dim)

                # Lag-1 target (see ChangingRotationDataset: fresh random x
                # each step is informationally unlearnable from state history)
                x_src = x if x_prev is None else x_prev
                x_prev = x

                # Apply context-specific rotation
                y = torch.stack([R @ x_src[n] for n in range(self.num_neurons)])
                
                if self.noise > 0:
                    y = y + torch.randn_like(y) * self.noise
                
                seq_inputs.append(x.flatten())
                seq_targets.append(y.flatten())
                seq_contexts.append(context_id)
            
            sequences.append(torch.stack(seq_inputs))
            targets.append(torch.stack(seq_targets))
            context_labels.append(torch.tensor(seq_contexts, dtype=torch.long))
        
        self.sequences = torch.stack(sequences)
        self.targets = torch.stack(targets)
        self.context_labels = torch.stack(context_labels)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.context_labels[idx]


class CompositionalDiscoveryDataset(Dataset):
    """
    Benchmark 3: Compositional discovery through dreaming.
    
    This is the KEY test for dreaming capability.
    
    The network observes:
    - A → B (direct training)
    - B → C (direct training)
    
    But NEVER observes A → C directly.
    
    During dreaming, PatternRecombiner should compose:
    T_AC = T_BC ∘ T_AB
    
    Test: Can it predict C from previously unseen A?
    
    Structure:
    - Phase 1 (steps 0-299): Train A→B with random A inputs
    - Phase 2 (steps 300-599): Train B→C with random B inputs  
    - Phase 3 (steps 600-899): TEST A→C with NEW A inputs (unseen during training)
    
    The transformations are fixed orthogonal matrices so composition is exact.
    """
    
    def __init__(
        self,
        num_neurons: int = 16,
        neuron_dim: int = 16,
        seq_length: int = None,  # Defaults to 3 × phase_length
        num_sequences: int = 100,
        noise: float = 0.05,
        phase_length: int = 300,
    ):
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.phase_length = phase_length
        self.seq_length = seq_length if seq_length is not None else 3 * phase_length
        self.num_sequences = num_sequences
        self.noise = noise

        # Fixed transformations (orthogonal for exact composition)
        self.T_AB = self._random_orthogonal()
        self.T_BC = self._random_orthogonal()
        self.T_AC = self.T_BC @ self.T_AB  # Ground truth composition

        self._generate_data()
    
    def _random_orthogonal(self) -> torch.Tensor:
        """Generate random orthogonal matrix for neuron_dim."""
        H = torch.randn(self.neuron_dim, self.neuron_dim)
        Q, R = torch.linalg.qr(H)
        D = torch.diag(torch.sign(torch.diag(R)))
        return Q @ D
    
    def _generate_data(self):
        sequences = []
        targets = []
        phase_labels = []
        
        for _ in range(self.num_sequences):
            seq_inputs = []
            seq_targets = []
            seq_phases = []
            
            # Lag-1 targets within each phase block (blocks reset because A,
            # B and C are different variable spaces — the first step of a
            # block targets its own current input). Fresh random inputs each
            # step make y = T@x_t unlearnable from state history (gate-4
            # diagnosis); lagging restores learnability.
            x_prev = None

            # Phase 1: A → B (steps 0 to phase_length-1)
            for t in range(self.phase_length):
                A = torch.randn(self.num_neurons, self.neuron_dim)
                A_src = A if x_prev is None else x_prev
                x_prev = A
                B = torch.stack([self.T_AB @ A_src[n] for n in range(self.num_neurons)])
                if self.noise > 0:
                    B = B + torch.randn_like(B) * self.noise
                seq_inputs.append(A.flatten())
                seq_targets.append(B.flatten())
                seq_phases.append(0)  # Phase 0: A→B

            x_prev = None

            # Phase 2: B → C (steps phase_length to 2*phase_length-1)
            for t in range(self.phase_length):
                B = torch.randn(self.num_neurons, self.neuron_dim)
                B_src = B if x_prev is None else x_prev
                x_prev = B
                C = torch.stack([self.T_BC @ B_src[n] for n in range(self.num_neurons)])
                if self.noise > 0:
                    C = C + torch.randn_like(C) * self.noise
                seq_inputs.append(B.flatten())
                seq_targets.append(C.flatten())
                seq_phases.append(1)  # Phase 1: B→C

            x_prev = None

            # Phase 3: TEST A → C with NEW A inputs (never trained directly)
            for t in range(self.phase_length):
                A_new = torch.randn(self.num_neurons, self.neuron_dim)  # UNSEEN A inputs
                A_src = A_new if x_prev is None else x_prev
                x_prev = A_new
                C_true = torch.stack([self.T_AC @ A_src[n] for n in range(self.num_neurons)])
                if self.noise > 0:
                    C_true = C_true + torch.randn_like(C_true) * self.noise
                seq_inputs.append(A_new.flatten())
                seq_targets.append(C_true.flatten())
                seq_phases.append(2)  # Phase 2: Test A→C (never trained directly)
            
            sequences.append(torch.stack(seq_inputs))
            targets.append(torch.stack(seq_targets))
            phase_labels.append(torch.tensor(seq_phases, dtype=torch.long))
        
        self.sequences = torch.stack(sequences)
        self.targets = torch.stack(targets)
        self.phase_labels = torch.stack(phase_labels)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.phase_labels[idx]
    
    def get_ground_truth_AC(self) -> torch.Tensor:
        """Return the true A→C transformation for evaluation."""
        return self.T_AC
    
    def get_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return T_AB and T_BC for verification."""
        return self.T_AB, self.T_BC


# Metric contract for network.step() results (see HANDOVER).
# Primary benchmark metric is connection_prediction_mse — the actual plastic
# network output vs ground truth. state_prediction_mse and readout_mse are
# recorded SEPARATELY; the untrained prediction head (readout) must never
# determine whether plastic learning succeeded. Legacy out["mse"] is RETIRED:
# the harness never falls back to it.
REQUIRED_METRIC_KEYS = ("connection_prediction_mse", "state_prediction_mse", "readout_mse")


def benchmark_metrics(out: Dict) -> Dict[str, float]:
    """Read the three metric-contract keys from a network step result.

    Raises loudly if any key is missing — never silently falls back to the
    retired legacy 'mse'.
    """
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in out]
    if missing:
        raise KeyError(
            f"network output missing required metric keys {missing}; "
            "legacy 'mse' is retired and must not be used as a fallback"
        )
    return {k: float(out[k]) for k in REQUIRED_METRIC_KEYS}


def primary_mse(out: Dict) -> float:
    """Primary benchmark metric: connection_prediction_mse."""
    return benchmark_metrics(out)["connection_prediction_mse"]


def tensor_pair_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MSE between a prediction and its target — fails loudly on shape mismatch."""
    assert pred.shape == target.shape, (
        f"prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}"
    )
    return (pred - target).pow(2).mean().item()


def get_validation_loaders(
    benchmark: str,
    config: BenchmarkConfig,
    dataset: Optional[Dataset] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test loaders for validation benchmarks.

    An explicitly provided dataset is used as-is so callers can keep a
    single ground-truth instance (e.g. T_AC) consistent with the data.
    """

    if dataset is None:
        if benchmark == "changing_rotation":
            dataset = ChangingRotationDataset(
                num_neurons=config.num_neurons,
                neuron_dim=config.neuron_dim,
                seq_length=config.seq_length,
                num_sequences=config.num_sequences,
                noise=config.noise,
                phase_length=config.phase_length,
            )
        elif benchmark == "context_switching":
            dataset = ContextDependentDataset(
                num_neurons=config.num_neurons,
                neuron_dim=config.neuron_dim,
                seq_length=config.seq_length,
                num_sequences=config.num_sequences,
                noise=config.noise,
            )
        elif benchmark == "compositional_discovery":
            dataset = CompositionalDiscoveryDataset(
                num_neurons=config.num_neurons,
                neuron_dim=config.neuron_dim,
                seq_length=config.seq_length,
                num_sequences=config.num_sequences,
                noise=config.noise,
                phase_length=config.phase_length,
            )
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    train_size = int(0.7 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
    )


def evaluate_compositional_discovery(
    network,
    test_loader: DataLoader,
    ground_truth_AC: torch.Tensor,
    device: str = "cpu",
    num_neurons: int = 16,
    neuron_dim: int = 16,
) -> Dict[str, float]:
    """
    Evaluate compositional discovery performance.
    
    Metrics:
    - MSE on A→C test phase (never directly trained)
    - Comparison to random baseline
    - Comparison to oracle (ground truth T_AC)
    """
    network.eval()
    network.to(device)

    metric_sums = {k: 0.0 for k in REQUIRED_METRIC_KEYS}
    oracle_mse = 0.0
    random_mse = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            inp, target, phase = batch
            if not (phase == 2).any():
                continue

            # Per-sequence iteration so the lag-1 oracle uses the same source
            # input the dataset used to build each target (phase blocks reset
            # the lag; the first test step of each sequence targets itself)
            batch_size = inp.shape[0]
            for b in range(batch_size):
                seq_x = inp[b].to(device)   # (seq_len, flat_dim)
                seq_y = target[b].to(device)
                seq_p = phase[b]
                test_steps = torch.nonzero(seq_p == 2).flatten().tolist()
                if not test_steps:
                    continue

                x_prev = None
                for t in test_steps:
                    step_inp = seq_x[t].view(1, num_neurons, neuron_dim)
                    step_target = seq_y[t].view(1, num_neurons, neuron_dim)
                    x_src = step_inp if x_prev is None else x_prev
                    x_prev = step_inp

                    out = network.step(step_inp, target_next=step_target, learn=False)
                    metrics = benchmark_metrics(out)
                    for k, v in metrics.items():
                        metric_sums[k] += v

                    # Oracle prediction on the SAME lagged source the target
                    # was generated from
                    oracle_pred = torch.stack(
                        [ground_truth_AC @ x_src[0, n] for n in range(num_neurons)])
                    oracle_pred = oracle_pred.unsqueeze(0)
                    assert oracle_pred.shape == step_target.shape, (
                        f"oracle shape {tuple(oracle_pred.shape)} != "
                        f"target shape {tuple(step_target.shape)}"
                    )
                    oracle_mse += (oracle_pred - step_target).pow(2).mean().item()

                    # Random baseline
                    random_pred = torch.randn_like(step_target)
                    random_mse += (random_pred - step_target).pow(2).mean().item()

                    count += 1

    if count == 0:
        return {"error": "No test phase data found"}

    avg_connection_mse = metric_sums["connection_prediction_mse"] / count
    avg_oracle_mse = oracle_mse / count
    avg_random_mse = random_mse / count
    result = {
        # Primary benchmark metric: actual plastic network output vs ground truth
        "test_mse": avg_connection_mse,
        "state_mse": metric_sums["state_prediction_mse"] / count,
        "readout_mse": metric_sums["readout_mse"] / count,
        "oracle_mse": avg_oracle_mse,
        "random_mse": avg_random_mse,
        "relative_to_oracle": avg_connection_mse / max(avg_oracle_mse, 1e-8),
        "relative_to_random": avg_connection_mse / max(avg_random_mse, 1e-8),
    }
    # Unclamped held-out prediction error on A→C (the discovery error).
    # No clamping: a static control sits at baseline, better models score lower.
    result["discovery_score"] = avg_connection_mse
    return result


def evaluate_adaptation_retention(
    network,
    test_loader: DataLoader,
    phase_boundaries: List[int],
    device: str = "cpu",
    num_neurons: int = 16,
    neuron_dim: int = 16,
) -> Dict[str, float]:
    """
    Evaluate adaptation speed and retention on changing rotation.

    The network LEARNS on the test stream (learn=True): adaptation speed is
    only measurable if the model is allowed to adapt to the new phase.

    Metrics:
    - Steps to adapt after each phase change, counted from the start of the
      phase (None if the criterion is never reached)
    - Retention of Phase A performance when returning in Phase C
    """
    network.eval()
    network.to(device)

    # Track each metric per phase (one entry per network step, not per batch element)
    phase_metrics = {0: {k: [] for k in REQUIRED_METRIC_KEYS},
                     1: {k: [] for k in REQUIRED_METRIC_KEYS},
                     2: {k: [] for k in REQUIRED_METRIC_KEYS}}

    def _record(step_phase, metrics):
        for p in step_phase.unique().tolist():
            for k, v in metrics.items():
                phase_metrics[p][k].append(v)

    with torch.no_grad():
        for batch in test_loader:
            inp, target, phase = batch

            # Reshape sequence data
            if inp.dim() == 3:
                batch_size, seq_len = inp.shape[0], inp.shape[1]
                inp = inp.view(batch_size, seq_len, num_neurons, neuron_dim)
                target = target.view(batch_size, seq_len, num_neurons, neuron_dim)
                phase = phase.view(batch_size, seq_len)

                for t in range(seq_len):
                    step_inp = inp[:, t].to(device)
                    step_target = target[:, t].to(device)
                    step_phase = phase[:, t]

                    out = network.step(step_inp, target_next=step_target, learn=True)
                    _record(step_phase, benchmark_metrics(out))
            else:
                inp = inp.to(device)
                target = target.to(device)
                out = network.step(inp, target_next=target, learn=True)
                _record(phase, benchmark_metrics(out))

    # Compute adaptation (primary metric): steps to reach 50% of the error
    # right after the phase switch, counted 1-indexed from the start of the
    # phase (phase switch itself = step 1). None if never reached.
    adaptation_steps = []
    for phase_id in [1, 2]:
        mses = phase_metrics.get(phase_id, {}).get("connection_prediction_mse", [])
        if len(mses) > 10:
            initial_error = mses[0]
            threshold = initial_error * 0.5
            steps_to_criterion = None
            if initial_error > 0:
                for i, mse in enumerate(mses):
                    if mse < threshold:
                        steps_to_criterion = i + 1
                        break
            adaptation_steps.append(steps_to_criterion)

    def _phase_tail(phase_id: int, key: str) -> float:
        values = phase_metrics.get(phase_id, {}).get(key, [])
        return float(np.mean(values[-50:])) if values else 0.0

    # Retention: compare Phase C (return to 30°) performance with Phase A
    phase_a_mse = _phase_tail(0, "connection_prediction_mse")
    phase_b_mse = _phase_tail(1, "connection_prediction_mse")
    phase_c_mse = _phase_tail(2, "connection_prediction_mse")
    retention_ratio = phase_c_mse / max(phase_a_mse, 1e-8)

    result = {
        "adaptation_steps": adaptation_steps,
        "phase_a_mse": phase_a_mse,
        "phase_b_mse": phase_b_mse,
        "phase_c_mse": phase_c_mse,
        "retention_ratio": retention_ratio,
        # The same phase measurements for the other two contract metrics
        "phase_a_state_mse": _phase_tail(0, "state_prediction_mse"),
        "phase_b_state_mse": _phase_tail(1, "state_prediction_mse"),
        "phase_c_state_mse": _phase_tail(2, "state_prediction_mse"),
        "phase_a_readout_mse": _phase_tail(0, "readout_mse"),
        "phase_b_readout_mse": _phase_tail(1, "readout_mse"),
        "phase_c_readout_mse": _phase_tail(2, "readout_mse"),
    }
    result["phase_mses"] = {
        p: float(np.mean(m["connection_prediction_mse"])) if m["connection_prediction_mse"] else 0
        for p, m in phase_metrics.items()
    }
    return result


def evaluate_context_switching(
    network,
    test_loader: DataLoader,
    device: str = "cpu",
    num_neurons: int = 16,
    neuron_dim: int = 16,
) -> Dict[str, float]:
    """
    Evaluate context-dependent learning.
    
    Metrics:
    - MSE per context
    - Interference (performance drop when switching)
    - Context classification accuracy (if applicable)
    """
    network.eval()
    network.to(device)
    
    context_metrics = {0: {k: [] for k in REQUIRED_METRIC_KEYS},
                       1: {k: [] for k in REQUIRED_METRIC_KEYS}}

    with torch.no_grad():
        for batch in test_loader:
            inp, target, context = batch

            # Reshape sequence data
            if inp.dim() == 3:
                batch_size, seq_len = inp.shape[0], inp.shape[1]
                inp = inp.view(batch_size, seq_len, num_neurons, neuron_dim)
                target = target.view(batch_size, seq_len, num_neurons, neuron_dim)
                context = context.view(batch_size, seq_len)

                for t in range(seq_len):
                    step_inp = inp[:, t].to(device)
                    step_target = target[:, t].to(device)
                    step_context = context[:, t]

                    out = network.step(step_inp, target_next=step_target, learn=False)
                    metrics = benchmark_metrics(out)
                    for c in step_context.unique().tolist():
                        for k, v in metrics.items():
                            context_metrics[c][k].append(v)
            else:
                inp = inp.to(device)
                target = target.to(device)
                out = network.step(inp, target_next=target, learn=False)
                metrics = benchmark_metrics(out)
                for c in context.unique().tolist():
                    for k, v in metrics.items():
                        context_metrics[c][k].append(v)

    def _ctx_mean(c: int, key: str) -> float:
        values = context_metrics.get(c, {}).get(key, [])
        return float(np.mean(values)) if values else 0.0

    context_0_mse = _ctx_mean(0, "connection_prediction_mse")
    context_1_mse = _ctx_mean(1, "connection_prediction_mse")

    # Switch cost: performance on first few steps after switch
    # (would need sequence-level evaluation for precise measurement)

    return {
        "context_0_mse": context_0_mse,
        "context_1_mse": context_1_mse,
        "context_gap": abs(context_0_mse - context_1_mse),
        "avg_mse": (context_0_mse + context_1_mse) / 2,
        # Same per-context measurements for the other two contract metrics
        "context_0_state_mse": _ctx_mean(0, "state_prediction_mse"),
        "context_1_state_mse": _ctx_mean(1, "state_prediction_mse"),
        "context_0_readout_mse": _ctx_mean(0, "readout_mse"),
        "context_1_readout_mse": _ctx_mean(1, "readout_mse"),
    }


# Control baselines
class RandomPredictor:
    """Random prediction baseline."""
    
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
    
    def step(self, inp: torch.Tensor, target_next: torch.Tensor, learn: bool = False) -> Dict:
        pred = torch.randn_like(target_next)
        mse = (pred - target_next).pow(2).mean().item()
        return {"mse": mse, "predicted_next": pred}


class FixedMappingOracle:
    """Oracle that knows the true transformation."""
    
    def __init__(self, transform: torch.Tensor, num_neurons: int, neuron_dim: int):
        self.transform = transform
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
    
    def step(self, inp: torch.Tensor, target_next: torch.Tensor, learn: bool = False) -> Dict:
        batch_size = inp.shape[0]
        inp_reshaped = inp.view(batch_size, self.num_neurons, self.neuron_dim)
        pred = torch.stack([
            torch.stack([self.transform @ inp_reshaped[b, n] for n in range(self.num_neurons)])
            for b in range(batch_size)
        ])
        mse = (pred.flatten(1) - target_next).pow(2).mean().item()
        return {"mse": mse, "predicted_next": pred}


def run_control_baselines(
    benchmark: str,
    config: BenchmarkConfig,
    test_loader: DataLoader,
) -> Dict[str, float]:
    """Run control baselines for a benchmark."""
    
    if benchmark == "compositional_discovery":
        # Need dataset for ground truth
        dataset = CompositionalDiscoveryDataset(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            seq_length=config.seq_length,
            num_sequences=config.num_sequences,
            noise=config.noise,
        )
        T_AC = dataset.get_ground_truth_AC()
        
        oracle = FixedMappingOracle(T_AC, config.num_neurons, config.neuron_dim)
        random = RandomPredictor(config.num_neurons * config.neuron_dim)
        
        oracle_mse = 0.0
        random_mse = 0.0
        count = 0
        
        for batch in test_loader:
            inp, target, phase = batch
            mask = (phase == 2)
            if not mask.any():
                continue
            inp = inp[mask]
            target = target[mask]
            
            oracle_out = oracle.step(inp, target)
            random_out = random.step(inp, target)
            
            oracle_mse += oracle_out['mse'] * inp.shape[0]
            random_mse += random_out['mse'] * inp.shape[0]
            count += inp.shape[0]
        
        return {
            "oracle_mse": oracle_mse / max(count, 1),
            "random_mse": random_mse / max(count, 1),
        }
    
    return {"oracle_mse": 0, "random_mse": 0}


if __name__ == "__main__":
    print("Testing validation benchmarks...")
    
    config = BenchmarkConfig(num_neurons=8, neuron_dim=8, num_sequences=20, seq_length=900)
    
    # Test 1: Changing Rotation
    print("\n1. Changing Rotation Dataset:")
    train, val, test = get_validation_loaders("changing_rotation", config)
    for batch in train:
        print(f"  Batch shapes: {[b.shape for b in batch]}")
        break
    
    # Test 2: Context Switching
    print("\n2. Context Switching Dataset:")
    train, val, test = get_validation_loaders("context_switching", config)
    for batch in train:
        print(f"  Batch shapes: {[b.shape for b in batch]}")
        break
    
    # Test 3: Compositional Discovery
    print("\n3. Compositional Discovery Dataset:")
    train, val, test = get_validation_loaders("compositional_discovery", config)
    for batch in train:
        print(f"  Batch shapes: {[b.shape for b in batch]}")
        # Check phases
        _, _, phase = batch
        print(f"  Phase distribution: {torch.bincount(phase.flatten())}")
        break
    
    print("\nAll validation benchmarks created successfully!")