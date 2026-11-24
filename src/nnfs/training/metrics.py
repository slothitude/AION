"""Metrics computation and tracking."""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class MetricsTracker:
    """Track metrics across training."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.current = {}
    
    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.history[k].append(v)
        self.current = metrics
    
    def get_history(self, key: str) -> List[float]:
        return self.history.get(key, [])
    
    def get_best(self, key: str, maximize: bool = True) -> float:
        vals = self.history.get(key, [])
        if not vals:
            return 0.0
        return max(vals) if maximize else min(vals)
    
    def to_dict(self) -> Dict:
        return dict(self.history)


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        preds: Predicted class indices (N,)
        targets: True class indices (N,)
        num_classes: Number of classes (inferred if None)
        
    Returns:
        Dict of metrics
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    if num_classes is None:
        num_classes = max(preds.max(), targets.max()) + 1
    
    metrics = {}
    
    # Overall accuracy
    metrics["accuracy"] = (preds == targets).mean()
    
    # Per-class accuracy
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    for p, t in zip(preds, targets):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1
    
    # Exact ratio for present classes; absent classes count as 0 accuracy (no epsilon artefact)
    class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total > 0)
    metrics["per_class_accuracy"] = class_acc.tolist()
    metrics["macro_accuracy"] = class_acc.mean()
    
    # Confusion matrix (for analysis)
    # cm = np.zeros((num_classes, num_classes), dtype=int)
    # for p, t in zip(preds, targets):
    #     cm[t, p] += 1
    # metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def compute_parameter_efficiency(
    model: torch.nn.Module,
    accuracy: float,
    baseline_params: int = None,
    baseline_acc: float = None
) -> Dict[str, float]:
    """
    Compute parameter efficiency metrics.
    
    Args:
        model: The model
        accuracy: Model accuracy
        baseline_params: Baseline model parameter count
        baseline_acc: Baseline model accuracy
        
    Returns:
        Efficiency metrics
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        "parameters": params,
        "accuracy": accuracy,
        "params_per_accuracy": params / (accuracy + 1e-8),
    }
    
    if baseline_params and baseline_acc:
        metrics["param_ratio"] = params / baseline_params
        metrics["acc_ratio"] = accuracy / baseline_acc
        metrics["efficiency_score"] = (accuracy / baseline_acc) / (params / baseline_params)
    
    return metrics


def measure_inference_time(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    num_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure inference time and throughput.
    
    Returns:
        Dict with mean_time_ms, std_time_ms, throughput_samples_per_sec
    """
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.randn(input_shape).to(device)
            _ = model(x)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            x = torch.randn(input_shape).to(device)
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
            end = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
            
            if device.startswith("cuda"):
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                import time
                t0 = time.perf_counter()
                _ = model(x)
                elapsed = (time.perf_counter() - t0) * 1000
            
            times.append(elapsed)
    
    times = np.array(times)
    batch_size = input_shape[0]
    
    return {
        "mean_time_ms": float(times.mean()),
        "std_time_ms": float(times.std()),
        "throughput_samples_per_sec": batch_size / (times.mean() / 1000),
        "times_ms": times.tolist(),
    }


def analyze_model_complexity(model: torch.nn.Module) -> Dict:
    """Analyze model architecture complexity."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by layer type
    layer_counts = {}
    param_by_layer = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                param_by_layer[layer_type] = param_by_layer.get(layer_type, 0) + module_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layer_type_counts": layer_counts,
        "parameters_by_layer_type": param_by_layer,
    }