# AION LEARNING BASELINE — FROZEN (owner directive, 2026-11-24)

This is the known-good AION learning baseline that passed Gate 4. **No changes to the
plasticity mathematics or the Dream Engine while the six-configuration pilot runs.**
Any proposed mechanism change waits for the owner's decision.

## Scope of freeze
- `src/nnfs/connections/**` (plastic.py, network.py, consolidation.py, __init__.py)
- `tests/test_connections_learning.py` (7/7 green)
- `scripts/diagnostic_learning.py` (full gate-4 rerun in ~30 s)

Snapshot: `freeze/aion_learning_baseline_20261124.tar.gz`
SHA-256: `a39013c90fb5f36118a76c35d3a6d5dde01d8c903a576a892d89a75218132daf`

Frozen file hashes:
```
f3f750e70b1cb65bebe054874a4cf72b057dd3520bfe6804022bb772f12782e6  src/nnfs/connections/plastic.py
413e8d45cebd34d610cd926d34bb8088f0d32c75e182851509febf8accd64fcf  src/nnfs/connections/network.py
2aed7ed292f76e6202c8fba0ae25c9c271ef635ca332cef34b7fa581de29e3bf  src/nnfs/connections/consolidation.py
```

## Calibration settings (the chosen point)
- `relation_type="post"` (ψ = post·W) — the bilinear pre-factor made E[Δu]=0 (starvation)
- W-init gain `g = 1.0`, learning rate `η = 0.05`, `state_leak_rate = 0.5`
- `blend_ratio = 1.0`, consolidation `transfer_fraction = 0.1`, `consolidation_interval` default 1000
- Update: `u_fast ← u_fast + blend·η·mean_batch((y−ŷ)·ψ)` — 0.7 blend factor applied exactly once
- Consolidation: `u ← u − d, θ ← θ + (blend/(1−blend))·d`, d = transfer_fraction·u on gate-passing entries
- Credit assignment: incoming connections share `target / in_degree`

## Gate-4 evidence (why this version)
| Criterion | Result |
|---|---|
| Held-out connection_prediction_mse | 1.026 → 0.577 (−43.8%); seeds: −51.6%, −49.2% |
| Coefficient recovery (effective mapping) | a = [2.0003, 2.9989] vs [2, 3]; abs err [0.0003, 0.0011]; cos = 0.999998 |
| Predictions on y=2x₁+3x₂ | [2.003, 2.993, 4.997, 7.000], MSE 1.7e-5 |
| Finite-difference agreement | max rel err 1.95e-4 |
| g² calibration law | \|Δŷ\| = 0.000946 identical across (1,0.05)/(2,0.0125)/(4,0.003125); spread 0.0 |
| Signed/zero-mean stability | no divergence; zero-mean floor is representational, not instability |
| Plastic-state growth | max\|u\| = 0.63, 93.7% margin from the ±10 clamp |
| Test suites | 7/7 connections; 96 passed / 3 pre-existing test_training.py failures (MNIST/env, unrelated) |

## Known limits (recorded honestly)
- The "post" channel learns per-dimension state maps; full off-diagonal cross-neuron mixing
  needs a trained W (autograd path) or correlated inputs across neurons.
- Consolidation correctness is unit-proven; network-level consolidation behaviour (rate > 0)
  has not yet been exercised in a live continual run — the pilot is its first outing.
- Lane B's original live-test recipe (fresh random target per step) was informationally
  unlearnable; fixed to lagged-target / fixed-pattern persistence.
