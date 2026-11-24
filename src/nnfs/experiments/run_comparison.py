"""Comparative experiment runner for standard vs multiplicative vs geometric architectures."""

import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
import json
import time

from nnfs.config import ExperimentConfig, TrainingConfig
from nnfs.models import StandardMLP, MultiplicativeMLP, GeometricMLP, SpecializedNet
from nnfs.training import Trainer, create_data_loaders, measure_inference_time, analyze_model_complexity
from nnfs.training.metrics import compute_parameter_efficiency


def run_single_experiment(
    model: nn.Module,
    model_name: str,
    task: str,
    config: TrainingConfig,
    device: str,
    seed: int,
    run_idx: int,
) -> Dict[str, Any]:
    """Run a single training run for one model on one task."""
    
    # Set seeds
    torch.manual_seed(seed + run_idx)
    np.random.seed(seed + run_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + run_idx)
    
    # Create data loaders
    train_loader, val_loader, test_loader, input_dim, num_classes = create_data_loaders(
        task, batch_size=config.batch_size
    )
    
    # Adjust model input/output dims if needed
    if hasattr(model, 'input_dim') and model.input_dim != input_dim:
        # Recreate model with correct dims
        pass  # Assume model already has correct dims
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        log_dir=f"logs/{task}/{model_name}/run_{run_idx}",
        save_checkpoints=True,
    )
    
    # Train
    start_time = time.time()
    history = trainer.train()
    train_time = time.time() - start_time
    
    # Measure inference time
    inference_metrics = measure_inference_time(
        model, (config.batch_size, input_dim), device=device
    )
    
    # Model complexity
    complexity = analyze_model_complexity(model)
    
    # Parameter efficiency
    test_acc = history.get("test_acc", history["val_acc"][-1])
    efficiency = compute_parameter_efficiency(model, test_acc)
    
    # Compile results
    results = {
        "model": model_name,
        "task": task,
        "run": run_idx,
        "seed": seed + run_idx,
        "parameters": complexity["trainable_parameters"],
        "train_time_sec": train_time,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_acc": test_acc,
        "best_val_acc": max(history["val_acc"]),
        "inference_time_ms": inference_metrics["mean_time_ms"],
        "throughput_samples_per_sec": inference_metrics["throughput_samples_per_sec"],
        "history": history,
        "efficiency": efficiency,
    }
    
    return results


def run_comparison(
    config_path: str = "configs/default.yaml",
    device: str = "cpu",
    output_dir: str = "results",
):
    """Run full comparative experiment."""
    
    # Load config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    exp_config = ExperimentConfig.from_yaml(config_path)
    train_config = exp_config.training
    train_config.device = device
    
    # Override device if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        train_config.device = "cpu"
    
    tasks = exp_config.tasks
    num_runs = exp_config.num_runs
    seed = train_config.seed
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        # Get input/output dims
        _, _, _, input_dim, num_classes = create_data_loaders(task, batch_size=1)
        
        # Create models with matched parameter counts
        models = create_models_for_task(input_dim, num_classes, task)
        
        for model_name, model in models.items():
            print(f"\n--- {model_name.upper()} ---")
            
            for run_idx in range(num_runs):
                print(f"Run {run_idx+1}/{num_runs}")
                
                try:
                    results = run_single_experiment(
                        model, model_name, task, train_config, device, seed, run_idx
                    )
                    all_results.append(results)
                    
                    print(f"  Test Acc: {100*results['test_acc']:.2f}% | "
                          f"Params: {results['parameters']:,} | "
                          f"Time: {results['train_time_sec']:.1f}s")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save results
    results_file = output_path / f"results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def create_models_for_task(input_dim: int, num_classes: int, task: str) -> Dict[str, nn.Module]:
    """Create models with matched parameter counts for a task."""
    
    # Target parameter count based on task complexity
    if "mnist" in task:
        target_params = 300000
        hidden_dim = 256
    elif "cifar" in task:
        target_params = 800000
        hidden_dim = 512
    else:
        target_params = 500000
        hidden_dim = 256
    
    models = {}
    
    # Standard MLP
    standard = StandardMLP(input_dim, num_classes, [hidden_dim, hidden_dim//2, hidden_dim//4])
    standard_params = standard.count_parameters()
    
    # Scale other models to match
    scale = (target_params / standard_params) ** 0.5
    adj_hidden = int(hidden_dim * scale)
    adj_hidden = max(64, min(adj_hidden, 512))
    
    models["standard"] = StandardMLP(input_dim, num_classes, [adj_hidden, adj_hidden//2, adj_hidden//4])
    
    # Multiplicative MLP
    mult_hidden = int(adj_hidden * 0.8)
    models["multiplicative"] = MultiplicativeMLP(
        input_dim, num_classes, [mult_hidden, mult_hidden//2, mult_hidden//4],
        interaction_rank=8
    )
    
    # Geometric MLP
    geom_hidden = int(adj_hidden * 0.7)
    models["geometric"] = GeometricMLP(
        input_dim, num_classes, [geom_hidden, geom_hidden//2, geom_hidden//4]
    )
    
    # Specialized net
    models["specialized"] = SpecializedNet(
        input_dim, num_classes, hidden_dim=adj_hidden, neuron_type="multiplicative"
    )
    
    # Report
    for name, model in models.items():
        print(f"  {name}: {model.count_parameters():,} parameters")
    
    return models


def print_summary(results: List[Dict]):
    """Print summary table of results."""
    from collections import defaultdict
    
    # Group by model and task
    grouped = defaultdict(list)
    for r in results:
        key = (r["model"], r["task"])
        grouped[key].append(r)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Task':<20} {'Params':>10} {'Test Acc':>10} {'Time(s)':>8} {'Infer(ms)':>10}")
    print("-"*80)
    
    for (model, task), runs in grouped.items():
        test_accs = [r["test_acc"] for r in runs]
        params = [r["parameters"] for r in runs]
        times = [r["train_time_sec"] for r in runs]
        infer = [r["inference_time_ms"] for r in runs]
        
        print(f"{model:<20} {task:<20} {np.mean(params):>10,.0f} "
              f"{100*np.mean(test_accs):>9.2f}% {np.mean(times):>8.1f} "
              f"{np.mean(infer):>10.2f}")
    
    print("="*80)


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run NNFS comparative experiments")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--task", help="Single task to run (overrides config)")
    parser.add_argument("--model", help="Single model to run (overrides config)")
    parser.add_argument("--runs", type=int, help="Number of runs (overrides config)")
    
    args = parser.parse_args()
    
    # Load and potentially override config
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    
    if args.task:
        config_data.setdefault("experiment", {})["tasks"] = [args.task]
    if args.runs:
        config_data.setdefault("experiment", {})["num_runs"] = args.runs
    
    # Save modified config temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config = f.name
    
    try:
        run_comparison(temp_config, args.device, args.output)
    finally:
        Path(temp_config).unlink()


if __name__ == "__main__":
    main()