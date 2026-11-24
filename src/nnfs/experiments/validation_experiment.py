"""
Validation Experiment Runner for NNFS Phase 5.

Runs controlled experiments on the three validation benchmarks:
1. Changing Rotation (adaptation + retention)
2. Context Switching (contextual plasticity)
3. Compositional Discovery (dreaming hypothesis testing)

Includes proper controls: random baseline, oracle, conventional network, NNFS without dreaming, NNFS with dreaming.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import copy
from pathlib import Path
import numpy as np

from nnfs.connections import RelationalNetwork
from nnfs.dreaming import DreamEngine, create_dream_engine
from nnfs.benchmarks.validation import (
    BenchmarkConfig,
    get_validation_loaders,
    evaluate_compositional_discovery,
    evaluate_adaptation_retention,
    evaluate_context_switching,
    run_control_baselines,
    CompositionalDiscoveryDataset,
    RandomPredictor,
    FixedMappingOracle,
)


class ModelVariant(Enum):
    RANDOM = "random"
    ORACLE = "oracle"
    STATIC = "static"
    HEBBIAN = "hebbian"
    PLASTIC = "plastic"
    CONSOLIDATING = "consolidating"
    CONTEXTUAL = "contextual"


class DreamVariant(Enum):
    D0_NONE = "d0_none"
    D1_REPLAY = "d1_replay"
    D2_REPLAY_IMAGINATION = "d2_replay_imagination"
    D3_FULL = "d3_full"


@dataclass
class ExperimentConfig:
    """Configuration for a validation experiment."""
    benchmark: str  # "changing_rotation", "context_switching", "compositional_discovery"
    model_variant: ModelVariant
    dream_variant: DreamVariant
    seed: int = 42
    
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
    dream_cycle_interval: int = 300  # Match phase boundaries
    dream_duration: int = 50
    
    # Training
    batch_size: int = 16
    total_steps: int = 1800  # 2 full cycles
    eval_interval: int = 100
    
    # Benchmark
    noise: float = 0.05
    
    device: str = "cpu"


@dataclass
class ValidationMetrics:
    """Metrics for validation experiments."""
    # Core metrics
    final_mse: float = 0.0
    adaptation_speed: List[int] = field(default_factory=list)
    retention_ratio: float = 0.0
    
    # Discovery (for compositional)
    discovery_score: float = 0.0
    relative_to_oracle: float = 0.0
    relative_to_random: float = 0.0
    
    # Context switching
    context_0_mse: float = 0.0
    context_1_mse: float = 0.0
    context_gap: float = 0.0
    
    # Structural
    final_connections: int = 0
    connections_created: int = 0
    connections_pruned: int = 0
    structural_efficiency: float = 0.0
    
    # Dream metrics
    dream_hypotheses_generated: int = 0
    dream_hypotheses_accepted: int = 0
    dream_precision: float = 0.0
    
    # Cost
    total_time: float = 0.0
    dream_time: float = 0.0
    
    # Online learning curve
    online_mse_curve: List[float] = field(default_factory=list)
    pre_update_mse_curve: List[float] = field(default_factory=list)


class ValidationNetwork:
    """Wrapper for creating and managing NNFS models with different variants."""
    
    @staticmethod
    def create(config: ExperimentConfig) -> RelationalNetwork:
        """Create network based on model variant."""
        
        if config.model_variant == ModelVariant.RANDOM:
            return None  # Handled separately
        elif config.model_variant == ModelVariant.ORACLE:
            return None  # Handled separately
        elif config.model_variant == ModelVariant.STATIC:
            return ValidationNetwork._create_static(config)
        elif config.model_variant == ModelVariant.HEBBIAN:
            return ValidationNetwork._create_hebbian(config)
        elif config.model_variant == ModelVariant.PLASTIC:
            return ValidationNetwork._create_plastic(config)
        elif config.model_variant == ModelVariant.CONSOLIDATING:
            return ValidationNetwork._create_consolidating(config)
        elif config.model_variant == ModelVariant.CONTEXTUAL:
            return ValidationNetwork._create_contextual(config)
        else:
            raise ValueError(f"Unknown model variant: {config.model_variant}")
    
    @staticmethod
    def _create_static(config: ExperimentConfig) -> RelationalNetwork:
        network = RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=0.0,
            trace_decay=config.trace_decay,
            consolidation_rate=0.0,
            blend_ratio=1.0,
        )
        for param in network.parameters():
            param.requires_grad = False
        return network
    
    @staticmethod
    def _create_hebbian(config: ExperimentConfig) -> RelationalNetwork:
        network = RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
            consolidation_rate=0.0,
            blend_ratio=1.0,
        )
        for conn in network.connections.values():
            conn.plasticity._hebbian_mode = True
        return network
    
    @staticmethod
    def _create_plastic(config: ExperimentConfig) -> RelationalNetwork:
        network = RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
            consolidation_rate=0.0,
            blend_ratio=1.0,
        )
        for conn in network.connections.values():
            conn.plasticity.theta_slow.data.zero_()
            conn.blend_ratio = 1.0
        return network
    
    @staticmethod
    def _create_consolidating(config: ExperimentConfig) -> RelationalNetwork:
        return RelationalNetwork(
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
    
    @staticmethod
    def _create_contextual(config: ExperimentConfig) -> RelationalNetwork:
        return RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
            consolidation_rate=config.consolidation_rate,
            consolidation_threshold=config.consolidation_threshold,
            blend_ratio=config.blend_ratio,
            context_dim=8,
        )


class ValidationExperiment:
    """Runs a single validation experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self.metrics = ValidationMetrics()
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create network
        self.network = ValidationNetwork.create(config)
        if self.network is not None:
            self.network = self.network.to(self.device)
        
        # Create dream engine
        self.dream_engine = None
        if self.network is not None and config.dream_variant != DreamVariant.D0_NONE:
            self.dream_engine = self._create_dream_engine()
        
        # Data loaders
        bench_config = BenchmarkConfig(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            seq_length=config.total_steps,
            num_sequences=100,
            batch_size=config.batch_size,
            noise=config.noise,
        )
        self.train_loader, self.val_loader, self.test_loader = get_validation_loaders(
            config.benchmark, bench_config
        )
        
        # For compositional discovery, get ground truth
        self.ground_truth_AC = None
        if config.benchmark == "compositional_discovery":
            dataset = CompositionalDiscoveryDataset(
                num_neurons=config.num_neurons,
                neuron_dim=config.neuron_dim,
                seq_length=config.total_steps,
                num_sequences=100,
                noise=config.noise,
            )
            self.ground_truth_AC = dataset.get_ground_truth_AC().to(self.device)
        
        # For changing rotation, get phase boundaries
        self.phase_boundaries = None
        if config.benchmark == "changing_rotation":
            dataset = CompositionalDiscoveryDataset(
                num_neurons=config.num_neurons,
                neuron_dim=config.neuron_dim,
                seq_length=config.total_steps,
                num_sequences=100,
                noise=config.noise,
            )
            self.phase_boundaries = [300, 600]  # Hardcoded for validation
        
        # Track dream hypotheses
        self.dream_hypotheses_log = []
    
    def _create_dream_engine(self) -> DreamEngine:
        if self.config.dream_variant == DreamVariant.D1_REPLAY:
            return DreamEngine(
                network=self.network,
                dream_cycle_interval=self.config.dream_cycle_interval,
                dream_duration=self.config.dream_duration,
                enable_replay=True,
                enable_recombination=False,
                enable_consolidation=True,
                enable_structural=False,
            )
        elif self.config.dream_variant == DreamVariant.D2_REPLAY_IMAGINATION:
            return DreamEngine(
                network=self.network,
                dream_cycle_interval=self.config.dream_cycle_interval,
                dream_duration=self.config.dream_duration,
                enable_replay=True,
                enable_recombination=True,
                enable_consolidation=True,
                enable_structural=False,
            )
        elif self.config.dream_variant == DreamVariant.D3_FULL:
            return DreamEngine(
                network=self.network,
                dream_cycle_interval=self.config.dream_cycle_interval,
                dream_duration=self.config.dream_duration,
                enable_replay=True,
                enable_recombination=True,
                enable_consolidation=True,
                enable_structural=True,
            )
        return None
    
    def run(self) -> ValidationMetrics:
        """Run the validation experiment."""
        start_time = time.time()
        step = 0
        
        # Control baselines
        if self.config.model_variant == ModelVariant.RANDOM:
            return self._run_random_baseline()
        elif self.config.model_variant == ModelVariant.ORACLE:
            return self._run_oracle_baseline()
        
        # Main training loop
        for epoch in range(100):
            for batch in self.train_loader:
                if step >= self.config.total_steps:
                    break
                
                inp, target, labels = batch
                
                # Reshape sequence data
                if inp.dim() == 3:
                    batch_size, seq_len = inp.shape[0], inp.shape[1]
                    expected_shape = (self.config.num_neurons, self.config.neuron_dim)
                    inp = inp.view(batch_size, seq_len, *expected_shape)
                    target = target.view(batch_size, seq_len, *expected_shape)
                    labels = labels.view(batch_size, seq_len)
                else:
                    raise ValueError(f"Unexpected input dim: {inp.dim()}")
                
                inp = inp.to(self.device)
                target = target.to(self.device)
                labels = labels.to(self.device)
                
                # Iterate through sequence
                for t in range(seq_len):
                    if step >= self.config.total_steps:
                        break
                    
                    step_inp = inp[:, t]
                    step_target = target[:, t]
                    step_label = labels[:, t]
                    
                    # EVALUATE BEFORE UPDATE
                    if step % self.config.eval_interval == 0 and step > 0:
                        pre_mse = self._evaluate_retention()
                        self.metrics.pre_update_mse_curve.append(pre_mse)
                    
                    # Online learning
                    self.network.train()
                    out = self.network.step(step_inp, target_next=step_target, learn=True)
                    
                    mse = out.get('mse', 0)
                    self.metrics.online_mse_curve.append(mse)
                    
                    # Store experience for dreaming
                    if self.dream_engine is not None:
                        self.dream_engine.store_experience(
                            step_inp[0], step_target[0],
                            significance=1.0 / (1.0 + mse),
                            uncertainty=out.get('prediction_error', torch.tensor(0)).std().item() 
                                if 'prediction_error' in out else 0.1,
                        )
                        
                        # Dream step
                        dream_start = time.time()
                        dream_stats = self.dream_engine.step()
                        self.metrics.dream_time += time.time() - dream_start
                        
                        if hasattr(dream_stats, 'hypotheses_generated'):
                            self.metrics.dream_hypotheses_generated += dream_stats.hypotheses_generated
                            self.metrics.dream_hypotheses_accepted += dream_stats.hypotheses_accepted
                            self.dream_hypotheses_log.append({
                                'step': step,
                                'generated': dream_stats.hypotheses_generated,
                                'accepted': dream_stats.hypotheses_accepted,
                            })
                        
                        if hasattr(dream_stats, 'connections_created'):
                            self.metrics.connections_created += dream_stats.connections_created
                        if hasattr(dream_stats, 'connections_pruned'):
                            self.metrics.connections_pruned += dream_stats.connections_pruned
                    
                    step += 1
                
                if step >= self.config.total_steps:
                    break
            
            if step >= self.config.total_steps:
                break
        
        self.metrics.total_time = time.time() - start_time
        self.metrics.final_connections = self.network.num_connections if self.network else 0
        
        # Final evaluation
        self._final_evaluation()
        
        return self.metrics
    
    def _run_random_baseline(self) -> ValidationMetrics:
        random = RandomPredictor(self.config.num_neurons * self.config.neuron_dim)
        total_mse = 0.0
        count = 0
        
        for batch in self.test_loader:
            inp, target, _ = batch
            inp = inp.to(self.device)
            target = target.to(self.device)
            
            # Reshape and iterate sequence
            if inp.dim() == 3:
                batch_size, seq_len = inp.shape[0], inp.shape[1]
                expected_shape = (self.config.num_neurons, self.config.neuron_dim)
                inp = inp.view(batch_size, seq_len, *expected_shape)
                target = target.view(batch_size, seq_len, *expected_shape)
                
                for t in range(seq_len):
                    out = random.step(inp[:, t].to(self.device), target[:, t].to(self.device))
                    total_mse += out['mse'] * inp[:, t].shape[0]
                    count += inp[:, t].shape[0]
        
        self.metrics.final_mse = total_mse / max(count, 1)
        return self.metrics
    
    def _run_oracle_baseline(self) -> ValidationMetrics:
        if self.ground_truth_AC is None:
            return self.metrics
        
        oracle = FixedMappingOracle(
            self.ground_truth_AC, 
            self.config.num_neurons, 
            self.config.neuron_dim
        )
        
        total_mse = 0.0
        count = 0
        
        for batch in self.test_loader:
            inp, target, labels = batch
            mask = (labels == 2)  # Only test phase
            if not mask.any():
                continue
            
            inp = inp[mask].to(self.device)
            target = target[mask].to(self.device)
            
            out = oracle.step(inp, target)
            total_mse += out['mse'] * inp.shape[0]
            count += inp.shape[0]
        
        self.metrics.final_mse = total_mse / max(count, 1)
        return self.metrics
    
    def _evaluate_retention(self) -> float:
        """Evaluate on validation set."""
        if self.network is None:
            return 0.0
        
        self.network.eval()
        total_mse = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inp, target, labels = batch
                
                if inp.dim() == 3:
                    batch_size, seq_len = inp.shape[0], inp.shape[1]
                    expected_shape = (self.config.num_neurons, self.config.neuron_dim)
                    inp = inp.view(batch_size, seq_len, *expected_shape)
                    target = target.view(batch_size, seq_len, *expected_shape)
                    
                    for t in range(seq_len):
                        out = self.network.step(
                            inp[:, t].to(self.device), 
                            target_next=target[:, t].to(self.device), 
                            learn=False
                        )
                        total_mse += out.get('mse', 0) * inp[:, t].shape[0]
                        count += inp[:, t].shape[0]
        
        return total_mse / max(count, 1)
    
    def _final_evaluation(self):
        """Run benchmark-specific final evaluation."""
        if self.config.benchmark == "compositional_discovery":
            self._evaluate_compositional()
        elif self.config.benchmark == "changing_rotation":
            self._evaluate_changing_rotation()
        elif self.config.benchmark == "context_switching":
            self._evaluate_context_switching()
        
        # Structural efficiency
        if self.metrics.final_connections > 0 and self.metrics.online_mse_curve:
            avg_mse = np.mean(self.metrics.online_mse_curve[-100:])
            self.metrics.structural_efficiency = 1.0 / (avg_mse * self.metrics.final_connections)
        
        # Dream precision
        if self.metrics.dream_hypotheses_generated > 0:
            self.metrics.dream_precision = (
                self.metrics.dream_hypotheses_accepted / self.metrics.dream_hypotheses_generated
            )
    
    def _evaluate_compositional(self):
        """Evaluate compositional discovery."""
        if self.ground_truth_AC is None:
            return
        
        results = evaluate_compositional_discovery(
            self.network, self.test_loader, self.ground_truth_AC, self.device,
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
        )
        
        self.metrics.final_mse = results.get("test_mse", 0)
        self.metrics.discovery_score = results.get("discovery_score", 0)
        self.metrics.relative_to_oracle = results.get("relative_to_oracle", 0)
        self.metrics.relative_to_random = results.get("relative_to_random", 0)
    
    def _evaluate_changing_rotation(self):
        """Evaluate adaptation and retention."""
        results = evaluate_adaptation_retention(
            self.network, self.test_loader, self.phase_boundaries, self.device,
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
        )
        
        self.metrics.adaptation_speed = results.get("adaptation_steps", [])
        self.metrics.retention_ratio = results.get("retention_ratio", 0)
        self.metrics.final_mse = results.get("phase_c_mse", 0)
    
    def _evaluate_context_switching(self):
        """Evaluate context-dependent learning."""
        results = evaluate_context_switching(
            self.network, self.test_loader, self.device,
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
        )
        
        self.metrics.context_0_mse = results.get("context_0_mse", 0)
        self.metrics.context_1_mse = results.get("context_1_mse", 0)
        self.metrics.context_gap = results.get("context_gap", 0)
        self.metrics.final_mse = results.get("avg_mse", 0)


def run_validation_experiment(config: ExperimentConfig) -> ValidationMetrics:
    """Run a single validation experiment."""
    experiment = ValidationExperiment(config)
    return experiment.run()


def run_full_validation(
    benchmarks: List[str] = ["changing_rotation", "context_switching", "compositional_discovery"],
    model_variants: List[ModelVariant] = [
        ModelVariant.RANDOM, ModelVariant.ORACLE, ModelVariant.STATIC,
        ModelVariant.HEBBIAN, ModelVariant.PLASTIC, ModelVariant.CONSOLIDATING, ModelVariant.CONTEXTUAL
    ],
    dream_variants: List[DreamVariant] = [
        DreamVariant.D0_NONE, DreamVariant.D1_REPLAY, 
        DreamVariant.D2_REPLAY_IMAGINATION, DreamVariant.D3_FULL
    ],
    seeds: List[int] = [42, 43, 44],
    output_dir: str = "validation_results"
) -> Dict[str, Any]:
    """Run the full validation matrix."""
    
    results = {}
    
    for benchmark in benchmarks:
        results[benchmark] = {}
        
        # Oracle and Random don't need dream variants
        for model_variant in [ModelVariant.RANDOM, ModelVariant.ORACLE]:
            results[benchmark][model_variant.value] = {}
            for seed in seeds:
                config = ExperimentConfig(
                    benchmark=benchmark,
                    model_variant=model_variant,
                    dream_variant=DreamVariant.D0_NONE,
                    seed=seed,
                )
                metrics = run_validation_experiment(config)
                results[benchmark][model_variant.value][f"seed_{seed}"] = _metrics_to_dict(metrics)
        
        # Other models with dream variants
        for model_variant in [ModelVariant.STATIC, ModelVariant.HEBBIAN, 
                               ModelVariant.PLASTIC, ModelVariant.CONSOLIDATING, ModelVariant.CONTEXTUAL]:
            results[benchmark][model_variant.value] = {}
            
            for dream_variant in dream_variants:
                results[benchmark][model_variant.value][dream_variant.value] = {}
                
                for seed in seeds:
                    config = ExperimentConfig(
                        benchmark=benchmark,
                        model_variant=model_variant,
                        dream_variant=dream_variant,
                        seed=seed,
                    )
                    
                    print(f"Running: {benchmark} | {model_variant.value} | {dream_variant.value} | seed={seed}")
                    
                    try:
                        metrics = run_validation_experiment(config)
                        results[benchmark][model_variant.value][dream_variant.value][f"seed_{seed}"] = _metrics_to_dict(metrics)
                        print(f"  MSE={metrics.final_mse:.4f}, "
                              f"Disc={metrics.discovery_score:.4f}, "
                              f"Ret={metrics.retention_ratio:.4f}, "
                              f"Conn={metrics.final_connections}")
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        results[benchmark][model_variant.value][dream_variant.value][f"seed_{seed}"] = {"error": str(e)}
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def _metrics_to_dict(metrics: ValidationMetrics) -> Dict:
    """Convert metrics to dictionary."""
    return {
        "final_mse": metrics.final_mse,
        "adaptation_speed": metrics.adaptation_speed,
        "retention_ratio": metrics.retention_ratio,
        "discovery_score": metrics.discovery_score,
        "relative_to_oracle": metrics.relative_to_oracle,
        "relative_to_random": metrics.relative_to_random,
        "context_0_mse": metrics.context_0_mse,
        "context_1_mse": metrics.context_1_mse,
        "context_gap": metrics.context_gap,
        "final_connections": metrics.final_connections,
        "connections_created": metrics.connections_created,
        "connections_pruned": metrics.connections_pruned,
        "structural_efficiency": metrics.structural_efficiency,
        "dream_hypotheses_generated": metrics.dream_hypotheses_generated,
        "dream_hypotheses_accepted": metrics.dream_hypotheses_accepted,
        "dream_precision": metrics.dream_precision,
        "total_time": metrics.total_time,
        "dream_time": metrics.dream_time,
        "online_mse_final": np.mean(metrics.online_mse_curve[-100:]) if metrics.online_mse_curve else 0,
    }


def analyze_validation_results(results: Dict) -> Dict:
    """Analyze validation results."""
    analysis = {}
    
    for benchmark, benchmark_results in results.items():
        analysis[benchmark] = {}
        
        # Get baselines
        random_mse = np.mean([
            v.get('final_mse', 0) for v in benchmark_results.get('random', {}).values() 
            if 'error' not in v
        ])
        oracle_mse = np.mean([
            v.get('final_mse', 0) for v in benchmark_results.get('oracle', {}).values() 
            if 'error' not in v
        ])
        
        for model_type, model_results in benchmark_results.items():
            if model_type in ['random', 'oracle']:
                continue
            
            analysis[benchmark][model_type] = {}
            
            for dream_level, level_results in model_results.items():
                if isinstance(level_results, dict) and 'error' not in str(level_results):
                    # Average over seeds
                    valid_seeds = [v for v in level_results.values() if 'error' not in v]
                    if not valid_seeds:
                        continue
                    
                    avg_mse = np.mean([v.get('final_mse', 0) for v in valid_seeds])
                    avg_disc = np.mean([v.get('discovery_score', 0) for v in valid_seeds])
                    avg_ret = np.mean([v.get('retention_ratio', 0) for v in valid_seeds])
                    avg_eff = np.mean([v.get('structural_efficiency', 0) for v in valid_seeds])
                    avg_prec = np.mean([v.get('dream_precision', 0) for v in valid_seeds])
                    
                    analysis[benchmark][model_type][dream_level] = {
                        'mse_vs_random': (random_mse - avg_mse) / max(random_mse, 1e-8) if random_mse > 0 else 0,
                        'mse_vs_oracle': (avg_mse - oracle_mse) / max(oracle_mse, 1e-8) if oracle_mse > 0 else 0,
                        'discovery_score': avg_disc,
                        'retention_ratio': avg_ret,
                        'structural_efficiency': avg_eff,
                        'dream_precision': avg_prec,
                    }
    
    return analysis


if __name__ == "__main__":
    print("Running NNFS Validation Experiments (Phase 5)")
    print("=" * 60)
    
    # Quick test with fewer seeds
    results = run_full_validation(
        benchmarks=["changing_rotation", "context_switching", "compositional_discovery"],
        seeds=[42, 43],
        output_dir="validation_results"
    )
    
    analysis = analyze_validation_results(results)
    
    print("\n=== VALIDATION ANALYSIS ===")
    for benchmark, benchmark_analysis in analysis.items():
        print(f"\n{benchmark}:")
        for model_type, model_analysis in benchmark_analysis.items():
            print(f"  {model_type}:")
            for dream_level, metrics in model_analysis.items():
                print(f"    {dream_level}: "
                      f"MSE vs Rand={metrics['mse_vs_random']:.3f}, "
                      f"MSE vs Oracle={metrics['mse_vs_oracle']:.3f}, "
                      f"Disc={metrics['discovery_score']:.3f}, "
                      f"Ret={metrics['retention_ratio']:.3f}, "
                      f"Eff={metrics['structural_efficiency']:.3f}, "
                      f"DreamPrec={metrics['dream_precision']:.3f}")
    
    print("\nResults saved to validation_results/validation_results.json")