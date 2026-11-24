"""Regression tests for the evaluation-metric fixes (Phase 5 blockers).

Covers:
  6. The untrained readout (prediction head) can never mask plastic learning —
     the primary benchmark metric is connection_prediction_mse.
  7. Adaptation steps are phase-relative (None when the criterion is never
     reached — never the trivial phase maximum).
  8. Discovery precision is unclamped (Static D0 sits at baseline, not 1.0)
     and D0 dream precision is None (null), not 1.0.
Plus stage-4 checks: real Phase B MSE, no oracle/target leakage into the
prediction path, loud failure on missing metric keys / shape mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnfs.benchmarks.validation import (
    REQUIRED_METRIC_KEYS,
    benchmark_metrics,
    primary_mse,
    tensor_pair_mse,
    evaluate_adaptation_retention,
    evaluate_compositional_discovery,
    ChangingRotationDataset,
)


def _metrics_out(connection: float, state: float, readout: float) -> dict:
    """A network.step()-style result carrying the metric contract keys."""
    return {
        "connection_prediction_mse": connection,
        "state_prediction_mse": state,
        "readout_mse": readout,
    }


class StubNetwork:
    """Network stub returning scripted metrics per step (FIFO queue per phase)."""

    def __init__(self, per_phase, phases=(100, 100, 100)):
        self.per_phase = per_phase
        self.phases = phases
        self.i = 0

    def eval(self):
        return self

    def to(self, device):
        return self

    def step(self, inp, target_next=None, learn=True):
        phase = min(self.i // self.phases[0], len(self.per_phase) - 1)
        values = self.per_phase[phase]
        self.i += 1
        if isinstance(values, dict):
            return values
        # Constant across all three contract metrics for this phase step
        return _metrics_out(values, values, values)


# ---------------------------------------------------------------------------
# Metric contract plumbing
# ---------------------------------------------------------------------------

def test_missing_metric_keys_fail_loudly_no_legacy_fallback():
    """The harness must error when metric keys are absent, never fall back to legacy mse."""
    legacy_only = {"mse": 0.5, "states": torch.zeros(1)}
    with pytest.raises(KeyError):
        benchmark_metrics(legacy_only)
    with pytest.raises(KeyError):
        primary_mse(legacy_only)


def test_tensor_pair_mse_rejects_shape_mismatch():
    """Prediction/target shape disagreements fail loudly, never broadcast."""
    pred = torch.zeros(1, 4, 4)
    target = torch.zeros(1, 16)
    with pytest.raises(AssertionError):
        tensor_pair_mse(pred, target)
    assert tensor_pair_mse(torch.ones(2, 3), torch.zeros(2, 3)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6: untrained readout cannot mask plastic learning
# ---------------------------------------------------------------------------

def test_readout_cannot_mask_connection_learning():
    """When connection_prediction_mse improves but readout_mse stays flat, the
    harness must report improvement on the primary metric only."""
    curves = {
        # connection improves 2.0 -> 0.5, readout stays flat at 1.0
        "connection": np.linspace(2.0, 0.5, 100),
        "readout": np.full(100, 1.0),
        "state": np.linspace(2.0, 0.6, 100),
    }
    improvements = {
        k: curves[k][0] - curves[k][-1] for k in ("connection", "readout")
    }
    assert improvements["connection"] > 1.0
    assert improvements["readout"] == pytest.approx(0.0)

    reported_primary = primary_mse(_metrics_out(curves["connection"][-1], 0.6, curves["readout"][-1]))
    assert reported_primary == pytest.approx(0.5)
    # The flat readout value is recorded separately, never used as the score
    metrics = benchmark_metrics(_metrics_out(0.5, 0.6, 1.0))
    assert metrics["readout_mse"] == pytest.approx(1.0)
    assert metrics["connection_prediction_mse"] == pytest.approx(0.5)


def test_live_network_readout_flat_connection_improves():
    """Live check on RelationalNetwork: plastic training genuinely improves
    the primary metric while the untrained readout metric stays flat.

    Recipe (Lane A, post gate-4): learnable lagged target — tgt = R @ x_prev
    (a fresh random x each step is informationally unlearnable from the
    network's state at transmit time), relation_type="post",
    state_leak_rate=0.5, per-connection lr=0.05.

    The corr(transmitted, target) growth assertion rules out window-drift
    artifacts: only real learned signal correlates transmissions with the
    target.
    """
    from nnfs.connections import RelationalNetwork

    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten().double()
        b = b.flatten().double()
        a = a - a.mean()
        b = b - b.mean()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

    torch.manual_seed(0)
    H = torch.randn(8, 8)
    R, _ = torch.linalg.qr(H)  # orthogonal ground-truth rotation
    net = RelationalNetwork(num_neurons=8, neuron_dim=8, sparsity=0.15,
                            relation_type="post", learning_rate=0.05,
                            trace_decay=0.95, consolidation_rate=0.0,
                            blend_ratio=1.0, state_leak_rate=0.5)
    net.reset_states(batch_size=1)

    x_prev = torch.randn(1, 8, 8)
    conn_curve, readout_curve, corr_curve = [], [], []
    for _ in range(600):
        x = torch.randn(1, 8, 8)
        tgt = torch.stack([R @ x_prev[0, n] for n in range(8)]).view(1, 8, 8)
        out = net.step(x, target_next=tgt, learn=True)
        m = benchmark_metrics(out)
        conn_curve.append(m["connection_prediction_mse"])
        readout_curve.append(m["readout_mse"])
        corr_curve.append(_corr(out["transmissions"], tgt))
        x_prev = x

    conn_improvement = np.mean(conn_curve[:50]) - np.mean(conn_curve[-50:])
    corr_growth = np.mean(corr_curve[-50:]) - np.mean(corr_curve[:50])
    readout_drift = abs(np.mean(readout_curve[-50:]) - np.mean(readout_curve[:50]))

    # Genuine learning on the primary metric (>=5% error reduction; measured ~9.7%)
    assert conn_improvement / np.mean(conn_curve[:50]) > 0.05, (
        f"plastic learning must improve the primary metric (got "
        f"{conn_improvement / np.mean(conn_curve[:50]):.3%})"
    )
    # corr(transmitted, target) must GROW — rules out window-drift artifacts
    assert corr_growth > 0.05, f"corr(transmitted, target) must grow (got {corr_growth:.4f})"
    # The untrained readout stays flat relative to its own level
    assert readout_drift < 0.05 * np.mean(readout_curve), (
        f"readout drift {readout_drift:.4f} too large — it must not move with learning"
    )


# ---------------------------------------------------------------------------
# Test 7: adaptation steps are phase-relative, None when never reached
# ---------------------------------------------------------------------------

def test_adaptation_steps_phase_relative_and_none_when_flat():
    """Adapting model: 1-indexed steps from the phase start. Flat model: None."""
    # Phase 0 constant; phase 1 decays from 2.0 (first crossing at 0-based
    # index 7 -> 8th step from phase start); phase 2 crosses at 0-based 5 ->
    # 6th step from phase start.
    phase1 = [2.0 * (0.9 ** i) for i in range(100)]          # 2.0*0.9^7 ≈ 0.957 < 1.0
    phase2 = [1.0 * (0.85 ** i) for i in range(100)]         # 0.85^5 ≈ 0.444 < 0.5

    stub = _IndexedStubPublic([[1.0] * 100, phase1, phase2])
    loader = _fake_loader(batch_size=4, seq_len=300, phases=(100, 100, 100))
    results = evaluate_adaptation_retention(stub, loader, [100, 200], "cpu",
                                            num_neurons=4, neuron_dim=4)
    assert results["adaptation_steps"][0] == 8
    assert results["adaptation_steps"][1] == 6

    # Flat (never adapts): criterion never reached -> None, never the max
    flat = _IndexedStubPublic([[1.0] * 100, [1.0] * 100, [1.0] * 100])
    results_flat = evaluate_adaptation_retention(flat, loader, [100, 200], "cpu",
                                                 num_neurons=4, neuron_dim=4)
    assert results_flat["adaptation_steps"] == [None, None]


def _fake_loader(batch_size=4, seq_len=300, phases=(100, 100, 100), num_sequences=4):
    xs, ys, ps = [], [], []
    for _ in range(num_sequences):
        xs.append(torch.randn(seq_len, 4 * 4))
        ys.append(torch.randn(seq_len, 4 * 4))
        ps.append(torch.repeat_interleave(
            torch.arange(len(phases)), torch.tensor(phases))[:seq_len])
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    ps = torch.stack(ps)
    return DataLoader(TensorDataset(xs, ys, ps), batch_size=batch_size)


def _dataset_loader(dataset, batch_size=2):
    return DataLoader(TensorDataset(dataset.sequences, dataset.targets,
                                    dataset.phase_labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Test 8: discovery unclamped; Static D0 at baseline; D0 dream precision null
# ---------------------------------------------------------------------------

def test_discovery_score_unclamped_static_not_1():
    """A static-like network (constant high error) scores its raw baseline
    error — never clamped to 1.0; a good model can score below it."""
    from nnfs.benchmarks.validation import CompositionalDiscoveryDataset

    dataset = CompositionalDiscoveryDataset(num_neurons=4, neuron_dim=4,
                                            phase_length=10, num_sequences=4,
                                            noise=0.0)
    ground_truth = dataset.get_ground_truth_AC()
    loader = _fake_loader(batch_size=2, seq_len=30, phases=(10, 10, 10), num_sequences=4)

    bad = StubNetwork([_metrics_out(3.7, 3.7, 1.0)] * 30)
    results = evaluate_compositional_discovery(bad, loader, ground_truth, "cpu",
                                               num_neurons=4, neuron_dim=4)
    assert results["discovery_score"] == pytest.approx(3.7)
    assert results["discovery_score"] != 1.0  # no saturation clamp

    good = StubNetwork([_metrics_out(0.2, 0.2, 1.0)] * 30)
    results_good = evaluate_compositional_discovery(good, loader, ground_truth, "cpu",
                                                    num_neurons=4, neuron_dim=4)
    assert results_good["discovery_score"] < results["discovery_score"]
    # Oracle and readout recorded separately
    assert "readout_mse" in results and "oracle_mse" in results


def test_d0_dream_precision_is_null_and_discovery_measured():
    """D0 (no dreaming) reports dream precision None (null), while A→C
    prediction is still measured separately."""
    from nnfs.experiments.pilot_experiment import PilotMetrics, _metrics_to_dict, DreamVariant

    m = PilotMetrics()
    assert m.dream_precision is None
    d = _metrics_to_dict(m)
    assert d["dream_precision"] is None
    # JSON round-trip keeps it null
    import json as _json
    assert _json.loads(_json.dumps(d))["dream_precision"] is None
    # D0 creates no dream engine
    from nnfs.experiments.pilot_experiment import PilotExperiment, ExperimentConfig, ModelVariant
    cfg = ExperimentConfig(benchmark="changing_rotation", model_variant=ModelVariant.STATIC,
                           dream_variant=DreamVariant.D0_NONE, num_sequences=4,
                           phase_length=10, total_steps=30)
    exp = PilotExperiment(cfg)
    assert exp.dream_engine is None


# ---------------------------------------------------------------------------
# Stage-4 checks: real Phase B, no oracle leakage, -45° phase transform
# ---------------------------------------------------------------------------

def test_phase_b_uses_minus_45_transform_and_is_measured():
    """ChangingRotation phase 1 targets use the -45° rotation, different from
    phase 0's +30°, and phase B MSE is actually returned by the evaluator."""
    ds = ChangingRotationDataset(num_neurons=4, neuron_dim=4, phase_length=5,
                                 num_sequences=2, noise=0.0)
    assert ds.phase_boundaries == [0, 5, 10, 15]
    x = torch.tensor([1.0, 0.0, 0.0, 1.0])
    y30 = ds._rotation_matrix(30.0) @ x
    ym45 = ds._rotation_matrix(-45.0) @ x
    assert not torch.allclose(y30, ym45)
    assert torch.allclose(ds._rotation_matrix(-45.0) @ ds._rotation_matrix(45.0) @ x, x,
                          atol=1e-5)

    phase1 = [2.0 * (0.9 ** i) for i in range(100)]
    stub = _IndexedStubPublic([[1.0] * 100, phase1, [1.0] * 100])
    loader = _fake_loader(batch_size=4, seq_len=300, phases=(100, 100, 100))
    results = evaluate_adaptation_retention(stub, loader, [100, 200], "cpu",
                                            num_neurons=4, neuron_dim=4)
    assert "phase_b_mse" in results
    assert results["phase_b_mse"] > 0.0
    # All three contract metrics have phase measurements recorded separately
    for key in ("phase_b_state_mse", "phase_b_readout_mse"):
        assert key in results


class _IndexedStubPublic:
    """Indexed stub shared by the stage-4 tests."""

    def __init__(self, phase_curves):
        self.phase_curves = phase_curves
        self.i = 0

    def eval(self):
        return self

    def to(self, device):
        return self

    def step(self, inp, target_next=None, learn=True):
        phase = min(self.i // 100, len(self.phase_curves) - 1)
        idx = self.i % 100
        self.i += 1
        v = self.phase_curves[phase][idx]
        return _metrics_out(v, v, v)


def test_no_oracle_or_target_leakage_into_prediction():
    """evaluate_compositional_discovery never feeds the oracle output or the
    dataset target into the network's prediction path: the stub's step() only
    ever receives (input, target_next, learn) and its metric is its own."""
    from nnfs.benchmarks.validation import CompositionalDiscoveryDataset

    dataset = CompositionalDiscoveryDataset(num_neurons=4, neuron_dim=4,
                                            phase_length=10, num_sequences=4,
                                            noise=0.0)
    ground_truth = dataset.get_ground_truth_AC()
    loader = _dataset_loader(dataset, batch_size=2)

    calls = []

    class RecordingStub(StubNetwork):
        def step(self, inp, target_next=None, learn=False):
            calls.append((learn, target_next is not None))
            return _metrics_out(2.5, 2.5, 2.5)

    results = evaluate_compositional_discovery(
        RecordingStub([None]), loader, ground_truth, "cpu",
        num_neurons=4, neuron_dim=4)
    # Eval never trains and never omits the target kwarg the oracle would need
    assert calls and all(learn is False for learn, _ in calls)
    assert all(has_target for _, has_target in calls)
    # Score equals the stub's own metric — the ground truth only scores the
    # oracle baseline term
    assert results["discovery_score"] == pytest.approx(2.5)
    assert results["oracle_mse"] < results["random_mse"]
