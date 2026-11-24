# NNFS Dreaming: Offline Relational Learning

## Overview

The NNFS Dreaming system implements **offline relational learning** - a network that learns continuously from experience through simulated "dreaming" cycles. This is the next evolutionary stage of NNFS after plastic relational connections.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DREAM ENGINE                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. MemoryRetriever     │ Selects important experiences        │
│  2. ExperienceReplay    │ Rehearses actual observations         │
│  3. PatternRecombiner   │ Creates hypothetical relationships    │
│  4. DreamSimulator      │ Evaluates imagined relationships      │
│  5. DreamEvaluator      │ Scores usefulness & consistency       │
│  6. DreamConsolidator   │ Updates long-term memory              │
│  7. StructuralDreamer   │ Experiments with network topology     │
└─────────────────────────────────────────────────────────────────┘
```

## Core Mathematical Framework

### Dual-Timescale Plasticity
```
a_ij(t) = θ_ij^slow + u_ij^fast(t)          # Dual-timescale relationship
m_ij(t) = a_ij(t) · ψ(h_i, h_j, c_t)        # Transmitted signal
```

### Prediction-Driven Plasticity
```
Prediction:  ĥ_j(t+1) = Σ_i m_ij(t)
Error:       δ_j = h_j(t+1) - ĥ_j(t+1)
Plasticity:  u_ij^fast ← u_ij^fast + η δ_j e_ij
```

### Adaptive Eligibility Traces
```
e_ij(t) = β_t e_ij(t-1) + (1-β_t) h_i(t) h_j(t)
β_t = f(context, uncertainty)  # Adaptive decay
```

### Selective Consolidation
```
θ_slow ← θ_slow + α · g_ij · (u_fast - θ_slow)
g_ij = sigmoid(consolidation_gate_ij)  # Per-connection consolidation gate
```

### Uncertainty-Aware Plasticity
```
uncertainty = error_std
uncertainty_scale = 1 / (1 + error_std)
effective_learning_rate = η · uncertainty_scale
```

### Selective Consolidation Gates
```
θ_slow ← θ_slow + α · σ(consolidation_gate) · (u_fast - θ_slow)
g_ij = sigmoid(evidence_ij)
evidence_ij = trace_strength / (1 + error_magnitude)
```

## Installation

```bash
cd C:\Users\aaron\Desktop\nnfs
pip install -e .
```

## Quick Start

```python
import torch
from nnfs.connections import RelationalNetwork
from nnfs.dreaming import DreamEngine

# Create network
network = RelationalNetwork(
    num_neurons=16,
    neuron_dim=16,
    sparsity=0.15,
    relation_type='bilinear',
    learning_rate=0.01,
    trace_decay=0.95,
    consolidation_rate=0.001,
)

# Create dream engine
dream_engine = DreamEngine(
    network=network,
    dream_cycle_interval=1000,
    dream_duration=100,
    enable_replay=True,
    enable_recombination=True,
    enable_consolidation=True,
    enable_structural=True,
)

# Training loop
for step in range(10000):
    inp = torch.randn(2, 16, 16)
    target = torch.randn(2, 16, 16)
    
    # Online learning
    out = network.step(inp, target_next=target, learn=True)
    
    # Dream engine checks if it's time to dream
    dream_stats = dream_engine.step()
    
    if dream_stats.get("dreamed"):
        print(f"Dream cycle: {dream_stats}")

# Manual dream
dream_result = dream_engine.run_manual_dream(duration=200)
print(f"Dream cycle: {dream_result.accepted} accepted, {dream_result.connections_created} created")
```

## Running Experiments

```bash
# Run dreaming experiments (concept reversal, context switching, structural efficiency)
python scripts/run_dreaming.py --experiments

# Run dreaming benchmarks (dreaming vs non-dreaming comparison)
python scripts/run_dreaming.py --benchmarks

# Run both
python scripts/run_dreaming.py --all
```

## Architecture Components

### 1. MemoryRetriever
Selects important experiences for replay based on significance, recency, uncertainty, and diversity.

```python
retriever = MemoryRetriever(
    max_memory_size=10000,
    significance_weight=0.4,
    recency_weight=0.2,
    uncertainty_weight=0.2,
    diversity_weight=0.2,
)
retriever.store(input_signal, target_next, significance=1.0, uncertainty=0.5)
experiences = retriever.retrieve(num_experiences=32)
```

### 2. ExperienceReplay
Replays stored experiences through the network for consolidation.

```python
replay = ExperienceReplay(network, retriever)
stats = replay.replay_episodic(num_experiences=32)
```

### 3. PatternRecombiner
Generates hypothetical relationships through composition, analogy, and context transfer.

```python
recombiner = PatternRecombiner(network)
hypotheses = recombiner.generate_hypotheses(max_hypotheses=20)
```

### 4. DreamSimulator
Evaluates hypothesized relationships through simulation.

```python
simulator = DreamSimulator(network)
results = simulator.simulate_batch(hypotheses, initial_state)
```

### 5. DreamEvaluator
Scores dreamed experiences for consolidation.

```python
evaluator = DreamEvaluator(network)
eval_result = evaluator.evaluate(simulation_result)
```

### 6. DreamConsolidator
Updates long-term memory from accepted dreams.

```python
consolidator = DreamConsolidator(network)
results = consolidator.consolidate(accepted_evaluations)
```

### 7. StructuralDreamer
Experiments with network topology (creation, pruning, rewiring).

```python
structural = StructuralDreamer(network)
changes = structural.dream_step()
```

## DreamEngine Integration

```python
dream_engine = DreamEngine(
    network=network,
    dream_cycle_interval=1000,  # Steps between dream cycles
    dream_duration=100,         # Steps per dream cycle
    enable_replay=True,
    enable_recombination=True,
    enable_consolidation=True,
    enable_structural=True,
)

# Automatic dreaming
for step in range(10000):
    out = network.step(inp, target_next=target, learn=True)
    dream_stats = dream_engine.step()  # Checks if should dream
    
# Manual dream
dream_result = dream_engine.run_manual_dream(duration=200)
```

## Running Experiments

```bash
# Run dreaming experiments (concept reversal, context switching, structural efficiency)
python scripts/run_dreaming.py --experiments

# Run dreaming benchmarks (dreaming vs non-dreaming comparison)
python scripts/run_dreaming.py --benchmarks

# Run both
python scripts/run_dreaming.py --all
```

## Configuration

```python
# Network config
config = {
    'num_neurons': 16,
    'neuron_dim': 16,
    'sparsity': 0.15,
    'relation_type': 'bilinear',
    'learning_rate': 0.01,
    'trace_decay': 0.95,
    'consolidation_rate': 0.001,
    'consolidation_threshold': 0.1,
    'blend_ratio': 0.7,
}

# Dreaming config
dream_config = {
    'dream_cycle_interval': 1000,  # Steps between dream cycles
    'dream_duration': 100,          # Steps per dream cycle
    'enable_replay': True,
    'enable_recombination': True,
    'enable_consolidation': True,
    'enable_structural': True,
}
```

## Key Innovations

1. **Dual-timescale plasticity**: Fast adaptation + slow consolidation
2. **Adaptive eligibility traces**: Context-dependent decay
3. **Selective consolidation**: Per-connection gates based on evidence
4. **Uncertainty-aware plasticity**: Lower learning rate for uncertain predictions
5. **Relational dreaming**: Network dreams in relationship space, not pixel space
6. **Structural dreaming**: Network topology evolves during dreaming
6. **Provenance separation**: Imagined vs. actual experiences never confused

## Research Directions

1. **Dream content analysis**: What does the network dream about?
2. **Dream duration scaling**: How does dream duration affect learning?
2. **Multi-scale dreaming**: Hierarchical dreaming at multiple timescales
3. **Dream sharing**: Multiple networks sharing dream experiences
4. **Lucid dreaming**: Network becomes aware it's dreaming
5. **Dream-guided exploration**: Using dreams to guide awake exploration

## Citation

If you use NNFS Dreaming in your research, please cite:

```
@article{nnfs_dreaming,
  title={NNFS: Neural Networks with Plastic Relational Connections and Offline Dreaming},
  author={...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.