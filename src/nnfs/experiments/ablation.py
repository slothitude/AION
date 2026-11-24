"""Ablation study framework for specialized layers and architectural components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import yaml
import json
import time
from pathlib import Path

from nnfs.models import StandardMLP, MultiplicativeMLP, GeometricMLP, SpecializedNet
from nnfs.training.trainer import Trainer
from nnfs.training.metrics import measure_inference_time, analyze_model_complexity


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    # Model parameters
    input_dim: int = 784
    output_dim: int = 10
    hidden_dim: int = 128
    target_params: int = 200000
    
    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 20
    early_stopping_patience: int = 5
    device: str = "cpu"
    seed: int = 42
    num_runs: int = 3
    
    # Ablation flags
    ablations: Dict[str, bool] = field(default_factory=lambda: {
        "multiplicative": False,
        "geometric": False,
        "competition": False,
        "memory": False,
        "prediction": False,
    })
    
    # Benchmarks
    benchmarks: List[str] = field(default_factory=lambda: ["mnist"])
    
    def to_dict(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "target_params": self.target_params,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
            "seed": self.seed,
            "num_runs": self.num_runs,
            "ablations": self.ablations,
            "benchmarks": self.benchmarks,
        }


class AblationModel(nn.Module):
    """
    Configurable model for ablation studies.
    
    Each component can be toggled independently to measure its contribution.
    """
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.hidden_dim = config.hidden_dim
        self.ablations = config.ablations
        
        # Base layer (always present)
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.activation = nn.ReLU()
        
        # Layer 1: Detection (enhanced feature extraction)
        if config.ablations.get("detection", False):
            from nnfs.layers.detection import DetectionLayer
            self.detection = DetectionLayer(
                config.hidden_dim, config.hidden_dim, 
                neuron_type="standard", activation="relu"
            )
        else:
            self.detection = nn.Identity()
        
        # Layer 2: Interaction (multiplicative)
        if config.ablations.get("multiplicative", False):
            from nnfs.layers.interaction import InteractionLayer
            self.interaction = InteractionLayer(
                config.hidden_dim, config.hidden_dim,
                neuron_type="multiplicative", activation="relu",
                interaction_rank=8, use_memory=False
            )
        else:
            self.interaction = nn.Identity()
        
        # Layer 3: Competition (adaptive sparsity)
        if config.ablations.get("competition", False):
            from nnfs.layers.competition import CompetitionLayer
            self.competition = CompetitionLayer(
                config.hidden_dim, group_size=8,
                adaptive_temperature=True
            )
        else:
            self.competition = nn.Identity()
        
        # Layer 4: Memory (fast/slow) - DISABLED for ablation study due to autograd issues
        self.memory = nn.Identity()
        
        # Layer 5: Integration
        from nnfs.layers.integration import IntegrationLayer
        self.integration = IntegrationLayer(
            config.hidden_dim, config.hidden_dim,
            neuron_type="standard", activation="relu"
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Prediction head (optional)
        if config.ablations.get("prediction", False):
            from nnfs.layers.prediction import PredictionLayer
            self.prediction = PredictionLayer(
                config.hidden_dim, neuron_type="standard"
            )
        else:
            self.prediction = None
    
    def forward(self, x: torch.Tensor):
        # Input projection
        x = self.input_proj(x)
        x = self.activation(x)
        
        # Detection
        x = self.detection(x)
        
        # Interaction
        x = self.interaction(x)
        
        # Competition
        x = self.competition(x)
        
        # Memory
        if hasattr(self.memory, 'forward') and not isinstance(self.memory, nn.Identity):
            x = x + 0.1 * self.memory(x)
        
        # Integration
        x = self.integration(x)
        
        # Output
        logits = self.output_proj(x)
        
        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ablation_models(config: AblationConfig) -> Dict[str, AblationModel]:
    """
    Create models for each ablation condition.
    
    Returns dict mapping condition name to model.
    """
    models = {}
    
    # Baseline: no ablations
    baseline_config = AblationConfig(**config.to_dict())
    baseline_config.ablations = {k: False for k in config.ablations}
    models["baseline"] = AblationModel(baseline_config)
    
    # Individual ablations
    for key in config.ablations:
        if config.ablations[key]:
            ablated_config = AblationConfig(**config.to_dict())
            ablated_config.ablations = {k: False for k in config.ablations}
            ablated_config.ablations[key] = True
            models[f"ablate_{key}"] = AblationModel(ablated_config)
    
    # Full model (all enabled)
    if any(config.ablations.values()):
        full_config = AblationConfig(**config.to_dict())
        full_config.ablations = {k: True for k in config.ablations}
        models["full"] = AblationModel(full_config)
    
    return models


def run_ablation_study(
    config: AblationConfig,
    train_loader,
    val_loader,
    test_loader,
    output_dir: str = "ablation_results",
) -> Dict[str, Any]:
    """
    Run full ablation study.
    
    Returns results dict with metrics for each condition.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create models
    models = create_ablation_models(config)
    
    # Report parameter counts
    for name, model in models.items():
        params = model.count_parameters()
        print(f"{name}: {params:,} parameters")
    
    # Train each model
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        model_results = {"runs": []}
        
        for run in range(config.num_runs):
            # Set seed
            seed = config.seed + run
            torch.manual_seed(seed)
            
            # Create fresh model
            if name == "baseline":
                model = AblationModel(AblationConfig(**config.to_dict()))
            elif name == "full":
                full_config = AblationConfig(**config.to_dict())
                full_config.ablations = {k: True for k in config.ablations}
                model = AblationModel(full_config)
            else:
                # Find which ablation
                key = name.replace("ablate_", "")
                ablated_config = AblationConfig(**config.to_dict())
                ablated_config.ablations = {k: False for k in config.ablations}
                ablated_config.ablations[key] = True
                model = AblationModel(ablated_config)
            
            # Trainer
            trainer = Trainer(
                model=model,
                config=type('obj', (object,), {
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'weight_decay': 1e-4,
                    'epochs': config.epochs,
                    'early_stopping_patience': config.early_stopping_patience,
                    'scheduler': 'cosine',
                    'warmup_epochs': 3,
                    'device': config.device,
                    'seed': seed,
                })(),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=config.device,
                log_dir=str(output_path / name / f"run_{run}"),
                save_checkpoints=True,
            )
            
            # Train
            history = trainer.train()
            
            # Measure inference time
            inference_metrics = measure_inference_time(
                model, (config.batch_size, config.input_dim), device=config.device
            )
            
            # Model complexity
            complexity = analyze_model_complexity(model)
            
            run_result = {
                "run": run,
                "seed": seed,
                "final_train_acc": history["train_acc"][-1],
                "final_val_acc": history["val_acc"][-1],
                "test_acc": history.get("test_acc", history["val_acc"][-1]),
                "best_val_acc": max(history["val_acc"]),
                "train_time": history.get("train_time", 0),
                "inference_time_ms": inference_metrics["mean_time_ms"],
                "throughput": inference_metrics["throughput_samples_per_sec"],
                "parameters": complexity["trainable_parameters"],
                "history": history,
            }
            
            model_results["runs"].append(run_result)
            print(f"  Run {run+1}/{config.num_runs}: Test Acc = {run_result['test_acc']:.4f}")
        
        # Aggregate
        test_accs = [r["test_acc"] for r in model_results["runs"]]
        model_results["mean_test_acc"] = sum(test_accs) / len(test_accs)
        model_results["std_test_acc"] = torch.tensor(test_accs).std().item()
        
        results[name] = model_results
    
    # Save results
    results_file = output_path / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    return results


def print_ablation_summary(results: Dict[str, Any]):
    """Print summary table of ablation results."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Condition':<25} {'Mean Acc':>10} {'Std Acc':>10} {'Params':>12} {'Infer (ms)':>12}")
    print("-"*80)
    
    for name, result in results.items():
        mean_acc = result["mean_test_acc"]
        std_acc = result["std_test_acc"]
        params = result["runs"][0]["parameters"]
        infer = result["runs"][0]["inference_time_ms"]
        print(f"{name:<25} {mean_acc:>9.4f} {std_acc:>9.4f} {params:>12,} {infer:>11.2f}")
    
    print("="*80)
    
    # Compute effect sizes
    if "baseline" in results:
        baseline_acc = results["baseline"]["mean_test_acc"]
        print(f"\nEffect sizes vs baseline ({baseline_acc:.4f}):")
        for name, result in results.items():
            if name != "baseline":
                effect = result["mean_test_acc"] - baseline_acc
                print(f"  {name}: {effect:+.4f}")