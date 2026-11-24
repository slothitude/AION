"""Tests for training utilities."""

import torch
import torch.nn as nn
import pytest
from pathlib import Path
from nnfs.training import create_data_loaders, measure_inference_time, analyze_model_complexity
from nnfs.training.metrics import compute_metrics, compute_parameter_efficiency, MetricsTracker


class TestDataLoaders:
    def test_mnist_loaders(self):
        if not Path("./data/MNIST/raw").exists():
            pytest.skip("MNIST not cached locally (offline); cannot test loading")
        train_loader, val_loader, test_loader, input_dim, num_classes = create_data_loaders("mnist", batch_size=32)
        assert input_dim == 784
        assert num_classes == 10
        
        # Check one batch
        data, target = next(iter(train_loader))
        assert data.shape == (32, 1, 28, 28)
        assert target.shape == (32,)
        assert data.min() >= 0 and data.max() <= 1
    
    def test_fashion_mnist_loaders(self):
        train_loader, val_loader, test_loader, input_dim, num_classes = create_data_loaders("fashion_mnist", batch_size=32)
        assert input_dim == 784
        assert num_classes == 10
    
    def test_cifar10_loaders(self):
        train_loader, val_loader, test_loader, input_dim, num_classes = create_data_loaders("cifar10_flattened", batch_size=32)
        assert input_dim == 3072
        assert num_classes == 10


class TestMetrics:
    def test_compute_metrics(self):
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 1, 0, 2, 2])
        metrics = compute_metrics(preds, targets, num_classes=3)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 4/6  # 4 correct out of 6
        assert "macro_accuracy" in metrics
        assert "per_class_accuracy" in metrics
        assert len(metrics["per_class_accuracy"]) == 3
    
    def test_compute_metrics_perfect(self):
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        metrics = compute_metrics(preds, targets)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_accuracy"] == 1.0
    
    def test_compute_parameter_efficiency(self):
        model = nn.Linear(100, 10)
        metrics = compute_parameter_efficiency(model, 0.95)
        assert metrics["parameters"] == 1010  # 100*10 + 10
        assert metrics["accuracy"] == 0.95
    
    def test_parameter_efficiency_with_baseline(self):
        model = nn.Linear(100, 10)
        metrics = compute_parameter_efficiency(model, 0.95, baseline_params=2000, baseline_acc=0.90)
        assert metrics["param_ratio"] == 1010/2000
        assert metrics["acc_ratio"] == 0.95/0.90
        assert "efficiency_score" in metrics
    
    def test_metrics_tracker(self):
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5, "acc": 0.8})
        tracker.update({"loss": 0.4, "acc": 0.85})
        
        assert tracker.get_history("loss") == [0.5, 0.4]
        assert tracker.get_history("acc") == [0.8, 0.85]
        assert tracker.get_best("acc", maximize=True) == 0.85
        assert tracker.get_best("loss", maximize=False) == 0.4


class TestInferenceTime:
    def test_measure_inference_time_cpu(self):
        model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        metrics = measure_inference_time(model, (32, 784), device="cpu", num_runs=10, warmup=2)
        
        assert "mean_time_ms" in metrics
        assert "std_time_ms" in metrics
        assert "throughput_samples_per_sec" in metrics
        assert metrics["mean_time_ms"] > 0
        assert metrics["throughput_samples_per_sec"] > 0


class TestModelComplexity:
    def test_analyze_model_complexity(self):
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        analysis = analyze_model_complexity(model)
        
        assert "total_parameters" in analysis
        assert "trainable_parameters" in analysis
        assert "layer_type_counts" in analysis
        assert "parameters_by_layer_type" in analysis
        
        assert analysis["total_parameters"] == analysis["trainable_parameters"]
        assert "Linear" in analysis["layer_type_counts"]
        assert "ReLU" in analysis["layer_type_counts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])