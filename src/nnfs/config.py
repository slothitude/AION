"""Configuration dataclasses for NNFS experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    """Base model configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    # Multiplicative-specific
    interaction_rank: int = 8
    # Geometric-specific
    use_complex: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "dropout": self.dropout,
            "interaction_rank": self.interaction_rank,
            "use_complex": self.use_complex,
        }


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    device: str = "cpu"
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tasks: List[str] = field(default_factory=lambda: ["mnist", "fashion_mnist"])
    num_runs: int = 3
    log_dir: str = "logs"
    save_checkpoints: bool = True
    plot_results: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        models = {}
        for name, cfg in data.get("models", {}).items():
            models[name] = ModelConfig(**cfg)
        
        training = TrainingConfig(**data.get("training", {}))
        
        return cls(
            models=models,
            training=training,
            tasks=data.get("experiment", {}).get("tasks", ["mnist", "fashion_mnist"]),
            num_runs=data.get("experiment", {}).get("num_runs", 3),
            log_dir=data.get("logging", {}).get("log_dir", "logs"),
            save_checkpoints=data.get("logging", {}).get("save_checkpoints", True),
            plot_results=data.get("logging", {}).get("plot_results", True),
        )


def get_activation_fn(name: str):
    """Get activation function by name."""
    import torch.nn as nn
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "silu": nn.SiLU(),
        "mish": nn.Mish(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]