#!/usr/bin/env python
"""Run ablation study experiments."""

import sys
sys.path.insert(0, "src")

import torch
from nnfs.experiments.ablation import AblationConfig, run_ablation_study, print_ablation_summary
from nnfs.benchmarks import get_xor_loaders, get_parity_loaders, get_rotated_mnist_loaders
from nnfs.benchmarks import get_sequential_loaders, get_pattern_completion_loaders
from nnfs.benchmarks import get_circular_loaders, get_interference_loaders
from nnfs.training.trainer import create_data_loaders


def get_benchmark_loaders(benchmark: str, config: AblationConfig):
    """Get data loaders for a benchmark."""
    if benchmark == "mnist":
        return create_data_loaders("mnist", batch_size=config.batch_size)
    elif benchmark == "fashion_mnist":
        return create_data_loaders("fashion_mnist", batch_size=config.batch_size)
    elif benchmark == "xor":
        return get_xor_loaders(batch_size=config.batch_size, n_samples=20000)
    elif benchmark == "parity":
        return get_parity_loaders(n_inputs=4, batch_size=config.batch_size, n_samples=20000)
    elif benchmark == "rotated_mnist":
        return get_rotated_mnist_loaders(max_angle=180, batch_size=config.batch_size)
    elif benchmark == "sequential":
        return get_sequential_loaders(system="sine", seq_length=20, batch_size=config.batch_size)
    elif benchmark == "pattern_completion":
        return get_pattern_completion_loaders(pattern_dim=64, batch_size=config.batch_size)
    elif benchmark == "circular":
        return get_circular_loaders(n_points=8, n_classes=4, batch_size=config.batch_size)
    elif benchmark == "interference":
        return get_interference_loaders(n_waves=2, grid_size=16, batch_size=config.batch_size)
    elif benchmark == "phase_rotation":
        from nnfs.benchmarks import get_phase_rotation_loaders
        return get_phase_rotation_loaders(dim=32, batch_size=config.batch_size)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def main():
    # Configuration
    config = AblationConfig(
        input_dim=784,
        output_dim=10,
        hidden_dim=128,
        target_params=150000,
        batch_size=128,
        learning_rate=1e-3,
        epochs=15,
        early_stopping_patience=5,
        device="cpu",
        seed=42,
        num_runs=3,
        ablations={
            "multiplicative": True,
            "competition": True,
            "memory": True,
            "prediction": True,
        },
        benchmarks=["mnist", "xor", "parity", "rotated_mnist", "sequential"],
    )
    
    # Run ablation for each benchmark
    for benchmark in config.benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {benchmark}")
        print(f"{'='*60}")
        
        try:
            train_loader, val_loader, test_loader, input_dim, num_classes = get_benchmark_loaders(benchmark, config)
            
            # Update config for this benchmark
            bench_config = AblationConfig(**config.to_dict())
            bench_config.input_dim = input_dim
            bench_config.output_dim = num_classes
            
            results = run_ablation_study(
                bench_config,
                train_loader,
                val_loader,
                test_loader,
                output_dir=f"ablation_results/{benchmark}",
            )
            
            print_ablation_summary(results)
            
        except Exception as e:
            print(f"Error on {benchmark}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()