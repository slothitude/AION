"""
Experiments for online relational learning with plastic connections.

Compares:
- Static: Fixed connections after training
- Plastic: Fast online relationship updates
- Consolidating: Fast updates + slow memory consolidation
- Contextual: Context-conditioned relationships
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time
from pathlib import Path

from nnfs.connections import RelationalNetwork
from nnfs.connections.consolidation import ConsolidationManager, ConsolidationConfig
from nnfs.benchmarks import (
    get_changing_relationship_loaders,
    get_context_switching_loaders,
)


@dataclass
class ContinualExperimentConfig:
    """Configuration for continual learning experiments."""
    # Network
    num_neurons: int = 32
    neuron_dim: int = 16
    sparsity: float = 0.15
    relation_type: str = "bilinear"
    
    # Plasticity
    learning_rate: float = 0.01
    trace_decay: float = 0.95
    consolidation_rate: float = 0.001
    consolidation_threshold: float = 0.1
    blend_ratio: float = 0.7
    
    # Consolidation
    consolidation_interval: int = 500
    trace_threshold: float = 0.1
    
    # Training
    batch_size: int = 16
    steps: int = 5000
    eval_interval: int = 500
    
    # Benchmark
    benchmark: str = "changing_rotation"  # "changing_rotation", "changing_permutation", "context_switching"
    
    # Comparison configs
    compare_configs: List[str] = None  # "static", "plastic", "consolidating", "contextual"
    
    device: str = "cpu"
    seed: int = 42


def create_network(config: ContinualExperimentConfig, variant: str = "consolidating") -> RelationalNetwork:
    """Create network with specified plasticity variant."""
    if variant == "static":
        # Fixed connections after training
        lr = 1e-6
        trace_decay = 1.0
        cons_rate = 0.0
    elif variant == "plastic":
        # Fast online learning only
        lr = config.learning_rate
        trace_decay = config.trace_decay
        cons_rate = 0.0
    elif variant == "consolidating":
        # Fast + slow plasticity
        lr = config.learning_rate
        trace_decay = config.trace_decay
        cons_rate = config.consolidation_rate
    elif variant == "contextual":
        # Context-conditioned (uses same network, different interpretation)
        lr = config.learning_rate
        trace_decay = config.trace_decay
        cons_rate = config.consolidation_rate
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    net = RelationalNetwork(
        num_neurons=config.num_neurons,
        neuron_dim=config.neuron_dim,
        sparsity=config.sparsity,
        relation_type=config.relation_type,
        learning_rate=lr,
        trace_decay=trace_decay,
        consolidation_rate=cons_rate,
        consolidation_threshold=config.consolidation_threshold,
        blend_ratio=config.blend_ratio,
    )
    return net


def get_benchmark_loaders(config: ContinualExperimentConfig):
    """Get data loaders for the specified benchmark."""
    if config.benchmark == "changing_rotation":
        return get_changing_relationship_loaders(
            change_type="rotation",
            batch_size=config.batch_size,
            seq_length=100,
            num_sequences=1000,
        )
    elif config.benchmark == "changing_permutation":
        return get_changing_relationship_loaders(
            change_type="permutation",
            batch_size=config.batch_size,
            seq_length=100,
            num_sequences=1000,
        )
    elif config.benchmark == "context_switching":
        return get_context_switching_loaders(
            batch_size=config.batch_size,
            seq_length=100,
            num_sequences=1000,
        )
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")


def run_continual_experiment(
    config: ContinualExperimentConfig,
    variant: str = "consolidating",
    output_dir: str = "continual_results",
) -> Dict[str, Any]:
    """
    Run a continual learning experiment.
    
    Returns metrics including adaptation speed, retention, forgetting.
    """
    output_path = Path(output_dir) / config.benchmark / variant
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Get data
    train_loader, val_loader, test_loader = get_benchmark_loaders(config)
    
    # Create network
    network = create_network(config, variant)
    network.to(config.device)
    
    # Consolidation manager
    cons_config = ConsolidationConfig(
        consolidation_interval=config.consolidation_interval,
        trace_threshold=0.1,
        consolidation_rate=config.consolidation_rate,
    )
    cons_manager = ConsolidationManager(
        dict(network.connections),
        ConsolidationConfig(
            consolidation_interval=config.consolidation_interval,
            trace_threshold=0.1,
            consolidation_rate=config.consolidation_rate,
        ),
    )
    
    # Metrics tracking
    metrics = {
        "variant": variant,
        "benchmark": config.benchmark,
        "steps": [],
        "train_mse": [],
        "val_mse": [],
        "adaptation_speed": [],
        "retention": [],
        "forgetting": [],
        "consolidation_events": [],
        "trace_stats": [],
    }
    
    # Training loop
    network.train()
    step = 0
    epoch = 0
    
    print(f"Running {variant} on {config.benchmark}...")
    
    while step < config.steps:
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            epoch += 1
            train_loader = iter(train_loader)
            batch = next(train_loader)
        
        step += 1
        
        # Prepare batch
        inp, target, change = batch
        inp = inp.to(config.device)
        target = target.to(config.device)
        
        # Forward step
        out = network.step(inp, target_next=target, learn=True)
        
        # Track metrics
        train_mse = out.get("mse", 0)
        metrics["steps"].append(step)
        metrics["train_mse"].append(train_mse)
        
        # Consolidation check
        cons_stats = {
            "consolidation_triggered": step % config.consolidation_interval == 0
        }
        if step % config.consolidation_interval == 0:
            cons_manager.force_consolidation()
        
        metrics["consolidation_events"].append(cons_stats.get("consolidation_triggered", False))
        
        # Trace stats
        trace_means = [c.eligibility_trace.trace.abs().mean().item() for c in network.connections.values()]
        metrics["trace_stats"].append(sum(trace_means) / len(trace_means))
        
        # Evaluation
        if step % config.eval_interval == 0:
            val_mse = evaluate_network(network, val_loader, config.device)
            metrics["val_mse"].append(val_mse)
            
            # Retention test
            if len(metrics["val_mse"]) > 1:
                prev_mse = metrics["val_mse"][-2]
                curr_mse = metrics["val_mse"][-1]
                forgetting = max(0, curr_mse - prev_mse)
                adaptation = max(0, prev_mse - curr_mse)
                metrics["forgetting"].append(forgetting)
                metrics["adaptation_speed"].append(adaptation)
            
            print(f"  Step {step}: Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}, "
                  f"Forgetting={metrics['forgetting'][-1]:.6f}, "
                  f"Adaptation={metrics['adaptation_speed'][-1]:.6f}")
        
        step += 1
    
    # Final evaluation
    final_val_mse = evaluate_network(network, test_loader, config.device)
    
    metrics["final_val_mse"] = final_val_mse
    metrics["total_steps"] = step
    
    # Save results
    output_path = Path("continual_results") / config.benchmark / variant
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"Results saved to {results_file}")
    return metrics


def evaluate_network(
    network: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Evaluate network on a dataset."""
    network.eval()
    total_mse = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            inp, target, change = batch
            inp = inp.to(device)
            target = target.to(device)
            
            out = network.step(inp, target_next=target, learn=False)
            mse = out.get("mse", 0)
            total_mse += mse * inp.shape[0]
            count += inp.shape[0]
    
    network.train()
    return total_mse / count if count > 0 else float('inf')


def run_comparison(
    config: ContinualExperimentConfig,
    variants: List[str] = None,
    output_dir: str = "continual_results",
) -> Dict[str, Dict]:
    """Run comparison across plasticity variants."""
    if variants is None:
        variants = ["static", "plastic", "consolidating", "contextual"]
    
    results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant}")
        print(f"{'='*60}")
        
        try:
            metrics = run_continual_experiment(config, variant)
            results[variant] = metrics
        except Exception as e:
            print(f"Error in {variant}: {e}")
            import traceback
            traceback.print_exc()
            results[variant] = {"error": str(e)}
    
    # Save comparison
    output_path = Path(output_dir) / config.benchmark
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nComparison saved to {comparison_file}")
    return results


def print_comparison_summary(results: Dict[str, Dict]):
    """Print summary of comparison results."""
    print("\n" + "="*80)
    print("CONTINUAL LEARNING COMPARISON")
    print("="*80)
    print(f"{'Variant':<15} {'Final MSE':>12} {'Best MSE':>12} {'Forgetting':>12} {'Adaptation':>12}")
    print("-"*80)
    
    for variant, metrics in results.items():
        if "error" in metrics:
            print(f"{variant:<15} ERROR: {metrics['error']}")
            continue
            
        final_mse = metrics.get("final_val_mse", 0)
        best_mse = min(metrics.get("val_mse", [0])) if metrics.get("val_mse") else 0
        forgetting = sum(metrics.get("forgetting", [0]))
        adaptation = sum(metrics.get("adaptation_speed", [0]))
        
        print(f"{variant:<15} {final_mse:>12.6f} {best_mse:>12.6f} {forgetting:>12.6f} {adaptation:>12.6f}")
    
    print("="*80)


def main():
    """Run continual learning experiments."""
    import sys
    sys.path.insert(0, "src")
    
    # Configuration
    config = ContinualExperimentConfig(
        num_neurons=32,
        neuron_dim=16,
        sparsity=0.15,
        relation_type="bilinear",
        learning_rate=0.01,
        trace_decay=0.95,
        consolidation_rate=0.001,
        consolidation_threshold=0.1,
        blend_ratio=0.7,
        batch_size=16,
        steps=3000,
        eval_interval=300,
        benchmark="changing_rotation",
        device="cpu",
        seed=42,
    )
    
    # Run comparison
    results = run_comparison(config, variants=["static", "plastic", "consolidating", "contextual"])
    print_comparison_summary(results)


if __name__ == "__main__":
    main()