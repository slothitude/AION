"""
Dreaming experiments - Run comparative experiments with dreaming vs non-dreaming networks.

Experiments:
1. Retention after interference: Learn A, learn B, test A retention
2. Concept reversal: A→B then A→C then A→B again
3. Context switching: Different relationships in different contexts
4. Structural efficiency: Does dreaming improve network topology?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time
from pathlib import Path

from nnfs.connections import RelationalNetwork
from nnfs.dreaming import DreamEngine, create_dream_engine
from nnfs.benchmarks import (
    get_changing_relationship_loaders,
    get_context_switching_loaders,
)


@dataclass
class ExperimentConfig:
    """Configuration for dreaming experiments."""
    # Network
    num_neurons: int = 16
    neuron_dim: int = 16
    sparsity: float = 0.15
    relation_type: str = "bilinear"
    
    # Plasticity
    learning_rate: float = 0.01
    trace_decay: float = 0.95
    consolidation_rate: float = 0.001
    consolidation_threshold: float = 0.1
    blend_ratio: float = 0.7
    
    # Dreaming
    dream_cycle_interval: int = 1000
    dream_duration: int = 100
    enable_replay: bool = True
    enable_recombination: bool = True
    enable_consolidation: bool = True
    enable_structural: bool = True
    
    # Training
    batch_size: int = 16
    steps: int = 5000
    eval_interval: int = 500
    
    # Benchmark
    benchmark: str = "changing_rotation"
    
    device: str = "cpu"
    seed: int = 42


def run_concept_reversal_experiment(config: 'ExperimentConfig') -> Dict[str, Any]:
    """
    Concept Reversal Experiment:
    
    Phase 1: A→B (learn relationship)
    Phase 2: A→C (reverse relationship)  
    Phase 3: A→B (return to original)
    
    Measures:
    - Adaptation speed to reversal
    - Recovery of original relationship
    - Retention of original knowledge
    """
    print("Running Concept Reversal Experiment...")
    
    # Create network
    network = RelationalNetwork(
        num_neurons=config.num_neurons,
        neuron_dim=config.neuron_dim,
        sparsity=config.sparsity,
        relation_type=config.relation_type,
        learning_rate=config.learning_rate,
        trace_decay=config.trace_decay,
        consolidation_rate=config.consolidation_rate,
        consolidation_threshold=config.consolidation_threshold,
        blend_ratio=config.blend_ratio,
    )
    
    dream_engine = DreamEngine(
        network=network,
        dream_cycle_interval=config.dream_cycle_interval,
        dream_duration=config.dream_duration,
    )
    
    # Create fixed patterns for A, B, C
    A = torch.randn(config.neuron_dim)
    B = torch.randn(config.neuron_dim)
    C = torch.randn(config.neuron_dim)
    
    # Expand to network shape
    A_net = A.unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    B_net = B.unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    C_net = C.unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    
    results = {
        "phase1_losses": [],
        "phase2_losses": [],
        "phase3_losses": [],
        "recovery_mse": [],
    }
    
    # Phase 1: Learn A→B
    print("  Phase 1: Learning A→B")
    for step in range(200):
        inp = A_net + torch.randn_like(A_net) * 0.1
        target = B_net + torch.randn_like(B_net) * 0.1
        out = network.step(inp, target_next=target, learn=True)
        
        if step % 50 == 0:
            print(f"  Step {step}: MSE={out['mse']:.4f}")
        results["phase1_losses"].append(out['mse'])
    
    # Test retention of A→B
    test_mse = test_relationship(network, A_net, B_net)
    print(f"  Retention after Phase 1: MSE={test_mse:.6f}")
    
    # Phase 2: Reverse to A→C
    print("  Phase 2: Reversing to A→C")
    for step in range(200):
        inp = A_net + torch.randn_like(A_net) * 0.1
        target = C_net + torch.randn_like(C_net) * 0.1
        out = network.step(inp, target_next=target, learn=True)
        
        if step % 50 == 0:
            print(f"  Step {step}: MSE={out['mse']:.4f}")
        results["phase2_losses"].append(out['mse'])
    
    # Test A→C
    test_mse_2 = test_relationship(network, A_net, C_net)
    print(f"  A→C MSE after Phase 2: {test_mse_2:.6f}")
    
    # Test retention of A→B (should be worse now)
    test_mse_b = test_relationship(network, A_net, B_net)
    print(f"  A→B retention after reversal: MSE={test_mse_b:.6f}")
    
    # Phase 3: Return to A→B
    print("  Phase 3: Returning to A→B")
    for step in range(200):
        inp = A_net + torch.randn_like(A_net) * 0.1
        target = B_net + torch.randn_like(B_net) * 0.1
        out = network.step(inp, target_next=target, learn=True)
        
        if step % 50 == 0:
            print(f"  Step {step}: MSE={out['mse']:.4f}")
        results["phase3_losses"].append(out['mse'])
    
    # Test recovery of A→B
    test_mse_3 = test_relationship(network, A_net, B_net)
    print(f"  A→B MSE after recovery: {test_mse_3:.6f}")
    results["recovery_mse"].append(test_mse_3)
    
    return results


def test_relationship(network, inp_pattern, target_pattern, num_tests=20) -> float:
    """Test a learned relationship."""
    network.eval()
    total_mse = 0.0
    with torch.no_grad():
        for _ in range(20):
            inp = inp_pattern + torch.randn_like(inp_pattern) * 0.1
            target = target_pattern + torch.randn_like(target_pattern) * 0.1
            out = network.step(inp, target_next=target, learn=False)
            total_mse += out.get("mse", 0)
    network.train()
    return total_mse / 20


def run_context_switching_experiment(config: 'ExperimentConfig') -> Dict[str, Any]:
    """
    Context Switching Experiment:
    
    Context 1: A→B
    Context 2: A→C
    
    Measures:
    - Context-dependent adaptation
    - Interference between contexts
    - Context-specific retention
    """
    print("Running Context Switching Experiment...")
    
    network = RelationalNetwork(
        num_neurons=config.num_neurons,
        neuron_dim=config.neuron_dim,
        sparsity=config.sparsity,
        relation_type=config.relation_type,
        learning_rate=config.learning_rate,
        trace_decay=config.trace_decay,
        consolidation_rate=config.consolidation_rate,
        consolidation_threshold=config.consolidation_threshold,
        blend_ratio=config.blend_ratio,
    )
    
    # Create context-dependent targets
    A = torch.randn(config.neuron_dim)
    B = torch.randn(config.neuron_dim)
    C = torch.randn(config.neuron_dim)
    
    A_net = A.unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    B_net = B.unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    C_net = C.unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    
    # Context encoding (one-hot)
    context1 = torch.tensor([1., 0.]).unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    context2 = torch.tensor([0., 1.]).unsqueeze(0).unsqueeze(0).repeat(1, config.num_neurons, 1)
    
    results = {"context1_mse": [], "context2_mse": [], "interference": []}
    
    # Alternate contexts
    for epoch in range(10):
        # Context 1: A→B
        for _ in range(20):
            inp = A_net + torch.randn_like(A_net) * 0.1
            out = network.step(inp, target_next=B_net, learn=True, context=context1)
            results["context1_mse"].append(out['mse'])
        
        # Context 2: A→C
        for _ in range(20):
            inp = A_net + torch.randn_like(A_net) * 0.1
            out = network.step(inp, target_next=C_net, learn=True, context=context2)
            results["context2_mse"].append(out['mse'])
        
        # Test interference
        network.eval()
        with torch.no_grad():
            out1 = network.step(A_net, target_next=B_net, learn=False, context=context1)
            out2 = network.step(A_net, target_next=C_net, learn=False, context=context2)
            interference = abs(out1['mse'] - out2['mse'])
            results["interference"].append(interference)
        network.train()
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: C1 MSE={results['context1_mse'][-1]:.4f}, C2 MSE={results['context2_mse'][-1]:.4f}, Interference={interference:.4f}")
    
    return results


def run_structural_efficiency_experiment(config: 'ExperimentConfig') -> Dict[str, Any]:
    """
    Structural Efficiency Experiment:
    
    Compare network topology with and without structural dreaming.
    
    Measures:
    - Number of connections
    - Connection strength distribution
    - Prediction efficiency (MSE per connection)
    - Adaptation speed
    """
    print("Running Structural Efficiency Experiment...")
    
    # Network with structural dreaming
    net_struct = RelationalNetwork(
        num_neurons=config.num_neurons,
        neuron_dim=config.neuron_dim,
        sparsity=config.sparsity,
        relation_type=config.relation_type,
        learning_rate=config.learning_rate,
        trace_decay=config.trace_decay,
        consolidation_rate=config.consolidation_rate,
        consolidation_threshold=config.consolidation_threshold,
        blend_ratio=config.blend_ratio,
    )
    
    dream_engine = DreamEngine(
        network=net_struct,
        dream_cycle_interval=500,
        dream_duration=50,
        enable_structural=True,
    )
    
    # Network without structural dreaming
    net_no_struct = RelationalNetwork(
        num_neurons=config.num_neurons,
        neuron_dim=config.neuron_dim,
        sparsity=config.sparsity,
        relation_type=config.relation_type,
        learning_rate=config.learning_rate,
        trace_decay=config.trace_decay,
        consolidation_rate=config.consolidation_rate,
        consolidation_threshold=config.consolidation_threshold,
        blend_ratio=config.blend_ratio,
    )
    
    dream_engine_no_struct = DreamEngine(
        network=net_no_struct,
        dream_cycle_interval=500,
        dream_duration=50,
        enable_structural=False,
    )
    
    # Training data
    target = torch.randn(2, config.num_neurons, config.neuron_dim)
    
    results = {
        "with_struct": {"mse": [], "connections": [], "efficiency": []},
        "without_struct": {"mse": [], "connections": [], "efficiency": []},
    }
    
    # Train both
    for step in range(1000):
        inp = torch.randn(2, config.num_neurons, config.num_neurons)
        
        # With structural dreaming
        out_struct = net_struct.step(inp, target_next=target, learn=True)
        if step % 100 == 0:
            n_conns = len(net_struct.connections)
            results["with_struct"]["mse"].append(out_struct['mse'])
            results["with_struct"]["connections"].append(n_conns)
            results["with_struct"]["efficiency"].append(out_struct['mse'] / max(1, n_conns))
        
        # Without structural dreaming
        out_no_struct = net_no_struct.step(inp, target_next=target, learn=True)
        if step % 100 == 0:
            n_conns = len(net_no_struct.connections)
            results["without_struct"]["mse"].append(out_no_struct['mse'])
            results["without_struct"]["connections"].append(n_conns)
            results["without_struct"]["efficiency"].append(out_no_struct['mse'] / max(1, n_conns))
    
    return results


def run_all_experiments():
    """Run all dreaming experiments."""
    config = type('Config', (), {
        'num_neurons': 16,
        'neuron_dim': 16,
        'sparsity': 0.15,
        'relation_type': 'bilinear',
        'learning_rate': 0.01,
        'trace_decay': 0.95,
        'consolidation_rate': 0.001,
        'consolidation_threshold': 0.1,
        'blend_ratio': 0.7,
        'dream_cycle_interval': 500,
        'dream_duration': 100,
    })()
    
    print("="*60)
    print("NNFS DREAMING EXPERIMENTS")
    print("="*60)
    
    all_results = {}
    
    # 1. Concept Reversal
    print("\n1. CONCEPT REVERSAL EXPERIMENT")
    all_results["concept_reversal"] = run_concept_reversal_experiment(type('Config', (), vars(config))())
    
    # 2. Context Switching
    print("\n2. CONTEXT SWITCHING EXPERIMENT")
    all_results["context_switching"] = run_context_switching_experiment(type('Config', (), vars(config))())
    
    # 3. Structural Efficiency
    print("\n3. STRUCTURAL EFFICIENCY EXPERIMENT")
    all_results["structural_efficiency"] = run_structural_efficiency_experiment(type('Config', (), vars(config))())
    
    # Save results
    output_path = Path("dreaming_experiments")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        if isinstance(results, dict):
            for key, val in results.items():
                if isinstance(val, list) and val:
                    print(f"  {key}: {val[-1]:.6f} (final)")
                else:
                    print(f"  {key}: {val}")
    
    return all_results


def main():
    """Run all dreaming experiments."""
    import sys
    sys.path.insert(0, "src")
    
    print("Running NNFS Dreaming Experiments...")
    results = run_all_experiments()
    
    print("\nExperiments complete. Results saved to dreaming_experiments/results.json")


if __name__ == "__main__":
    main()