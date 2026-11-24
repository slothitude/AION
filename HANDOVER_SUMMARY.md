# NNFS Project Handover Summary
**Date:** 2026-11-24
**Archive:** `nnfs_handover_20261124_complete.tar.gz`

## Project Overview

**NNFS (Neural Network Fundamental Structures)** - An experimental neural architecture implementing:
- Alternative neuron mathematics (Multiplicative, Geometric)
- Specialised neural layers (Detection, Interaction, Competition, Integration, Prediction)
- Plastic Relational Connections with dual-timescale plasticity
- Dream Engine for offline relational learning
- Structural plasticity for network topology evolution

---

## What's Implemented & Working

### Phase 1: Alternative Neurons ✅
- **StandardNeuron** - Conventional baseline
- **MultiplicativeNeuron** - Signal interaction detection
- **GeometricNeuron** - Phase/magnitude representation

### Phase 2: Specialised Layers ✅
- Detection, Interaction, Competition, Integration, Prediction layers

### Phase 3: Plastic Relational Connections ✅
- Dual-timescale plasticity (fast/slow)
- Adaptive eligibility traces (context/uncertainty-dependent)
- Selective consolidation gates
- Uncertainty-aware plasticity
- Contextual plasticity
- Dynamic connection creation/pruning

### Phase 4: Dream Engine ✅ (7 Components)
1. **MemoryRetriever** - Experience selection
2. **ExperienceReplay** - Observation rehearsal
3. **PatternRecombiner** - Hypothesis generation (composition, analogy)
4. **DreamSimulator** - Hypothesis evaluation via predictive model
5. **DreamEvaluator** - Scoring (consistency, novelty, efficiency, forgetting)
6. **DreamConsolidator** - Long-term memory updates, new connections
7. **StructuralDreamer** - Topology evolution (create/prune/rewire)

### Phase 5: Validation Framework ✅
- 3 benchmarks with known ground truth
- 5×4 experimental matrix (5 models × 4 dream levels)
- Pilot experiment runner
- Proper controls and metrics

---

## Key Files in Archive

```
src/nnfs/
├── neurons/          # Standard, Multiplicative, Geometric
├── layers/           # Detection, Interaction, Competition, Integration, Prediction
├── connections/      # PlasticRelationalConnection, RelationalNetwork
├── dreaming/         # All 7 Dream Engine components
├── benchmarks/       # Validation benchmarks
├── experiments/      # Pilot & continual experiments
└── training/         # Trainer, metrics

scripts/
├── run_pilot.py      # Pilot experiments
├── run_validation.py # Full validation
└── run_dreaming.py   # Dreaming experiments
```

---

## Current Status (As of 2026-11-24)

### ✅ Working
- All architecture components compile and run
- Dream engine components integrate correctly
- Structural dreaming creates/prunes connections
- Evaluation framework executes without errors

### ❌ Critical Issues (Need Fix Before Full Matrix)
1. **Network not actually learning** - MSE constant at ~1.02 across all phases/models
2. **Phase B MSE = 0.0** - Evaluation bug
3. **Adaptation steps = 4500** - Metric bug (max value)
4. **Discovery score clamped to 1.0** - Masks real differences
5. **Static = Plastic = Consolidating** - All models identical

### 🔧 Root Cause Hypothesis
Plastic connections may not be updating weights during `step(learn=True)`. The `PlasticRelationalConnection.step()` with `learn=True` needs verification.

---

## Next Steps for New Session

### 1. Diagnostic (Priority 1)
```python
# Test if plastic connections actually update
network = RelationalNetwork(...)
inp = torch.randn(2, 16, 16)
target = torch.randn(2, 16, 16)

# Check weights before
w_before = network.connections['conn_0'].plasticity.u_fast.clone()

out = network.step(inp, target_next=target, learn=True)

# Check weights after
w_after = network.connections['conn_0'].plasticity.u_fast.clone()
print('Weight change:', (w_after - w_before).abs().mean())
```

### 2. Fix Evaluation Metrics
- Phase B MSE calculation
- Adaptation step counting (relative to phase start)
- Discovery score without clamping

### 3. Run Focused Pilot
Only 6 configurations × 2 benchmarks × 3 seeds after fixes

---

## Archive Contents
- `src/nnfs/` - Complete source code
- `scripts/` - Experiment runners
- `configs/` - Default configurations
- `tests/` - Unit tests
- `pilot_results/` - Latest pilot experiment results
- `continual_results/` - Earlier continual learning results
- `validation_results/` - Validation benchmark results
- `README.md`, `README_DREAMING.md` - Documentation
- `pyproject.toml` - Package configuration

---

## Key Design Principles (From Specification)

1. **Dreams generate hypotheses; reality decides which survive**
2. **Provenance separation** - Imagined vs. actual experiences never confused
3. **Candidate topology** - Structural changes tested before commitment
4. **Evaluation before update** - Measure retention before learning
5. **Uncertainty-aware plasticity** - Lower learning rate for uncertain predictions

---

## Contact / Context
This is an experimental research architecture. The goal is scientific validation of whether dreaming improves continual learning, not production deployment.

The most important experiment: **Does Plastic D2 (replay + imagination) beat Plastic D1 (replay only) on Compositional Discovery?**

That comparison has not yet been validly measured.