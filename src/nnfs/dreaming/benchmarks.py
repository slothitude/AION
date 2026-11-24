"""
Dreaming benchmarks - Evaluate the effectiveness of dreaming on continual learning.

Benchmarks:
1. Retention after interference: Learn A, learn B, test retention of A
2. Adaptation speed: How quickly does the network adapt to new relationships?
3. Recovery: After reversal, can the network recover original relationships?
4. Context switching: Can the network maintain multiple contexts?
5. Structural efficiency: Does dreaming lead to more efficient network topology?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
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
class DreamingBenchmarkConfig:
    """Configuration for dreaming benchmarks."""
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


def get_benchmark_loaders(config: 'DreamingBenchmarkConfig'):
    """Get data loaders for the specified benchmark."""
    data_dim = config.num_neurons * config.neuron_dim
    
    if config.benchmark == "changing_rotation":
        from nnfs.benchmarks import get_changing_relationship_loaders
        return get_changing_relationship_loaders(
            change_type="rotation",
            batch_size=config.batch_size,
            seq_length=100,
            num_sequences=100,
            input_dim=config.num_neurons * config.neuron_dim,
            output_dim=config.num_neurons * config.neuron_dim,
        )
    elif config.benchmark == "context_switching":
        from nnfs.benchmarks import get_context_switching_loaders
        return get_context_switching_loaders(
            batch_size=config.batch_size,
            seq_length=100,
            num_sequences=100,
        )
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")


class DreamingBenchmark:
    """
    Benchmark for evaluating dreaming effectiveness.
    
    Compares:
    - Dreaming network: full dream engine enabled
    - No-dream network: dream engine disabled (only online learning)
    """
    
    def __init__(self, config: DreamingBenchmarkConfig):
        self.config = config
        self.device = config.device
        
    def run_comparison(self, output_dir: str = "dreaming_results") -> Dict[str, Any]:
        """Run comparison between dreaming and non-dreaming networks."""
        
        # Create networks
        network_dreaming = RelationalNetwork(
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
            sparsity=self.config.sparsity,
            relation_type=self.config.relation_type,
            learning_rate=self.config.learning_rate,
            trace_decay=self.config.trace_decay,
            consolidation_rate=self.config.consolidation_rate,
            consolidation_threshold=self.config.consolidation_threshold,
            blend_ratio=self.config.blend_ratio,
        ).to(self.device)
        
        network_nodream = RelationalNetwork(
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
            sparsity=self.config.sparsity,
            relation_type=self.config.relation_type,
            learning_rate=self.config.learning_rate,
            trace_decay=self.config.trace_decay,
            consolidation_rate=0.0,  # No consolidation
            consolidation_threshold=self.config.consolidation_threshold,
            blend_ratio=self.config.blend_ratio,
        ).to(self.device)
        
        # Get data
        train_loader, val_loader, test_loader = get_benchmark_loaders(self.config)
        
        # Create dream engines
        dream_engine = DreamEngine(
            network=network_dreaming,
            dream_cycle_interval=self.config.dream_cycle_interval,
            dream_duration=self.config.dream_duration,
        ) if self.config.enable_replay or self.config.enable_consolidation else None
        
        # Metrics tracking
        results = {
            "dreaming": {"train_mse": [], "val_mse": [], "retention": [], "forgetting": []},
            "no_dream": {"train_mse": [], "val_mse": [], "retention": [], "forgetting": []},
        }
        
        # Training loop
        print("Training dreaming network...")
        dream_results = self._train_network(
            network_dreaming, train_loader, val_loader, 
            dream_engine=dream_engine,
            variant="dreaming"
        )
        
        print("Training no-dream network...")
        nodream_results = self._train_network(
            network_nodream, train_loader, val_loader,
            dream_engine=None,
            variant="no_dream"
        )
        
        results["dreaming"] = dream_results
        results["no_dream"] = nodream_results
        
        # Save results
        output_path = Path("dreaming_results") / "comparison"
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(Path("dreaming_results") / "comparison.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _train_network(
        self, 
        network: 'RelationalNetwork',
        train_loader: DataLoader,
        val_loader: DataLoader,
        dream_engine: Optional['DreamEngine'],
        variant: str,
    ) -> Dict[str, Any]:
        """Train a network with optional dreaming."""
        
        metrics = {
            "train_mse": [],
            "val_mse": [],
            "retention": [],
            "forgetting": [],
            "trace_stats": [],
        }
        
        # Dream engine
        dream_engine = DreamEngine(
            network=network,
            dream_cycle_interval=1000,
            dream_duration=100,
        ) if self.config.enable_replay or self.config.enable_consolidation else None
        
        step = 0
        epoch = 0
        val_mse_prev = float('inf')
        
        while step < 2000:  # Simplified
            for batch in torch.utils.data.DataLoader:
                pass
            # Simplified for now
            pass
        
        return metrics
    
    def evaluate_retention(self, network, test_loader) -> float:
        """Evaluate retention of early tasks."""
        network.eval()
        total_mse = 0.0
        count = 0
        
        with torch.no_grad():
            for i, (inp, target, change) in enumerate(test_loader):
                if i > 10:  # Only first few batches
                    break
                out = network.step(inp, target_next=target, learn=False)
                mse = out.get("mse", 0)
                total_mse += mse * inp.shape[0]
                count += inp.shape[0]
        
        return total_mse / count if count > 0 else float('inf')


def run_dreaming_benchmark(
    benchmark_type: str = "changing_rotation",
    output_dir: str = "dreaming_results",
) -> Dict[str, Any]:
    """Run the dreaming benchmark."""
    config = DreamingBenchmarkConfig(benchmark=benchmark_type)
    
    benchmark = DreamingBenchmark(config)
    return benchmark.run_comparison(output_dir)


def main():
    """Run dreaming benchmark experiments."""
    import sys
    sys.path.insert(0, "src")
    
    print("Running dreaming benchmark...")
    results = run_dreaming_benchmark("changing_rotation")
    
    print("\nResults:")
    for variant, metrics in results.items():
        print(f"{variant}: {metrics}")


if __name__ == "__main__":
    main()