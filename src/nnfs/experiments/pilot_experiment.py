"""
Pilot Experiment Runner for NNFS Phase 5.

Runs the 6-configuration pilot on Changing Rotation and Compositional Discovery.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import copy
from pathlib import Path
import numpy as np

from nnfs.connections import RelationalNetwork
from nnfs.dreaming import DreamEngine
from nnfs.benchmarks.validation import (
    BenchmarkConfig,
    get_validation_loaders,
    evaluate_compositional_discovery,
    evaluate_adaptation_retention,
    evaluate_context_switching,
    benchmark_metrics,
    CompositionalDiscoveryDataset,
)


class ModelVariant(Enum):
    STATIC = "static"
    PLASTIC = "plastic"
    CONSOLIDATING = "consolidating"


class DreamVariant(Enum):
    D0_NONE = "d0_none"
    D1_REPLAY = "d1_replay"
    D2_REPLAY_IMAGINATION = "d2_replay_imagination"
    D3_FULL = "d3_full"


# Pilot configurations: (model_variant, dream_variant)
PILOT_CONFIGS = [
    (ModelVariant.STATIC, DreamVariant.D0_NONE),
    (ModelVariant.PLASTIC, DreamVariant.D0_NONE),
    (ModelVariant.CONSOLIDATING, DreamVariant.D0_NONE),
    (ModelVariant.PLASTIC, DreamVariant.D1_REPLAY),
    (ModelVariant.PLASTIC, DreamVariant.D2_REPLAY_IMAGINATION),
    (ModelVariant.PLASTIC, DreamVariant.D3_FULL),
]

# Consolidation-interval matrix (steps between consolidation sweeps).
# A comparison must use ONE interval across all its configurations so every
# variant gets the same total consolidation budget.
CONSOLIDATION_INTERVALS = [10, 25, 50, 100]


@dataclass
class ExperimentConfig:
    """Configuration for a pilot experiment."""
    benchmark: str  # "changing_rotation", "compositional_discovery"
    model_variant: ModelVariant
    dream_variant: DreamVariant
    seed: int = 42
    
    # Network
    num_neurons: int = 16
    neuron_dim: int = 16
    sparsity: float = 0.15
    relation_type: str = "post"  # "post" per connections-lane gate 4 (bilinear starves updates)

    # Plasticity
    learning_rate: float = 0.05  # connections-lane gate-4 live-run value
    trace_decay: float = 0.95
    consolidation_rate: float = 0.001
    # u_fast elements reach ~0.01-0.02 at lr 0.01 over 900 steps, so the old
    # 0.1 threshold meant consolidation never fired. Experiment knob, not
    # mechanism: lowered so the consolidating variant can express itself.
    consolidation_threshold: float = 0.005
    blend_ratio: float = 0.7
    state_leak_rate: float = 0.5  # gate-4 live-run value (old default 0.1 blurred states ~20 steps)
    
    # Dreaming
    dream_cycle_interval: int = 300
    dream_duration: int = 50
    
    # Training
    batch_size: int = 16
    total_steps: int = 900  # Full 3 phases
    eval_interval: int = 50

    # Plasticity
    consolidation_interval: int = 50  # Steps between consolidation sweeps (10/25/50/100)

    # Benchmark
    noise: float = 0.05
    phase_length: int = 300
    num_sequences: int = 100

    device: str = "cpu"


@dataclass
class PilotMetrics:
    """Metrics for pilot experiments."""
    # Core learning (primary metric = connection_prediction_mse)
    final_mse: float = 0.0
    online_mse_curve: List[float] = field(default_factory=list)
    pre_update_mse_curve: List[float] = field(default_factory=list)
    # The three contract metrics, recorded separately (online training curves)
    state_mse_curve: List[float] = field(default_factory=list)
    readout_mse_curve: List[float] = field(default_factory=list)

    # Phase-specific (changing rotation)
    phase_a_mse: float = 0.0
    phase_b_mse: float = 0.0
    phase_c_mse: float = 0.0
    adaptation_steps_b: Optional[int] = None  # None = criterion never reached
    adaptation_steps_c: Optional[int] = None
    retention_ratio: float = 0.0
    # Same phase measurements for the other two contract metrics
    phase_a_state_mse: float = 0.0
    phase_b_state_mse: float = 0.0
    phase_c_state_mse: float = 0.0
    phase_a_readout_mse: float = 0.0
    phase_b_readout_mse: float = 0.0
    phase_c_readout_mse: float = 0.0
    
    # Compositional discovery
    discovery_score: float = 0.0
    relative_to_oracle: float = 0.0
    relative_to_random: float = 0.0
    oracle_mse: float = 0.0
    random_mse: float = 0.0
    
    # Structural
    initial_connections: int = 0
    final_connections: int = 0
    connections_created: int = 0
    connections_pruned: int = 0
    structural_efficiency: float = 0.0
    
    # Dream metrics
    dream_hypotheses_generated: int = 0
    dream_hypotheses_accepted: int = 0
    dream_precision: Optional[float] = None  # None = no dreaming (D0)
    dream_time: float = 0.0

    # Consolidation stats (network-level sweeps; per-connection means)
    consolidation_calls: int = 0
    consolidated_total: int = 0
    gated_consolidated_total: int = 0
    gate_mean_sum: float = 0.0
    gate_open_fraction_sum: float = 0.0
    theta_slow_norm_initial: float = 0.0
    theta_slow_norm_final: float = 0.0
    consolidation_output_max_diff: Optional[float] = None
    
    # Cost
    total_time: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Topology snapshots
    initial_topology: List[Tuple[int, int]] = field(default_factory=list)
    final_topology: List[Tuple[int, int]] = field(default_factory=list)


class PilotNetwork:
    """Creates and manages NNFS models for pilot experiments."""

    @staticmethod
    def create(config: ExperimentConfig) -> RelationalNetwork:
        if config.model_variant == ModelVariant.STATIC:
            return PilotNetwork._create_static(config)
        elif config.model_variant == ModelVariant.PLASTIC:
            return PilotNetwork._create_plastic(config)
        elif config.model_variant == ModelVariant.CONSOLIDATING:
            return PilotNetwork._create_consolidating(config)
        else:
            raise ValueError(f"Unknown model variant: {config.model_variant}")

    @staticmethod
    def _set_connection_learning_rates(network: RelationalNetwork, learning_rate: float):
        """Apply the variant's plasticity rate to every connection.

        Belt-and-braces: network args now propagate to connections, but this
        guarantees the per-variant rate regardless of upstream wiring.
        """
        for conn in network.connections.values():
            conn.plasticity.learning_rate = learning_rate

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
            consolidation_interval=config.consolidation_interval,
            state_leak_rate=config.state_leak_rate,
        )
        PilotNetwork._set_connection_learning_rates(network, 0.0)
        for param in network.parameters():
            param.requires_grad = False
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
            consolidation_interval=config.consolidation_interval,
            state_leak_rate=config.state_leak_rate,
        )
        PilotNetwork._set_connection_learning_rates(network, config.learning_rate)
        for conn in network.connections.values():
            conn.plasticity.theta_slow.data.zero_()
            conn.blend_ratio = 1.0
        return network

    @staticmethod
    def _create_consolidating(config: ExperimentConfig) -> RelationalNetwork:
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
            consolidation_interval=config.consolidation_interval,
            state_leak_rate=config.state_leak_rate,
        )
        PilotNetwork._set_connection_learning_rates(network, config.learning_rate)
        return network


class PilotExperiment:
    """Runs a single pilot experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self.metrics = PilotMetrics()
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create network
        self.network = PilotNetwork.create(config).to(self.device)

        # Snapshot initial topology and consolidated-weight norm
        self.metrics.initial_connections = self.network.num_connections
        self.metrics.initial_topology = self._get_topology()
        self.metrics.theta_slow_norm_initial = self._theta_slow_norm()
        
        # Create dream engine
        self.dream_engine = None
        if config.dream_variant != DreamVariant.D0_NONE:
            self.dream_engine = self._create_dream_engine()
        
        # Data loaders (single dataset instance so ground truth matches the data)
        bench_config = BenchmarkConfig(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            seq_length=config.total_steps,
            num_sequences=config.num_sequences,
            batch_size=config.batch_size,
            noise=config.noise,
            phase_length=config.phase_length,
        )
        self.bench_dataset = None
        self.ground_truth_AC = None
        if config.benchmark == "compositional_discovery":
            self.bench_dataset = CompositionalDiscoveryDataset(
                num_neurons=config.num_neurons,
                neuron_dim=config.neuron_dim,
                seq_length=config.total_steps,
                num_sequences=config.num_sequences,
                noise=config.noise,
                phase_length=config.phase_length,
            )
            self.ground_truth_AC = self.bench_dataset.get_ground_truth_AC().to(self.device)
        self.train_loader, self.val_loader, self.test_loader = get_validation_loaders(
            config.benchmark, bench_config, dataset=self.bench_dataset
        )
    
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
    
    def _get_topology(self) -> List[Tuple[int, int]]:
        """Get current network topology as list of (pre, post) pairs."""
        topology = []
        for conn in self.network.connections.values():
            pre = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
            post = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
            topology.append((pre, post))
        return topology

    def _theta_slow_norm(self) -> float:
        """Total norm of all consolidated slow weights."""
        total = 0.0
        for conn in self.network.connections.values():
            total += float(conn.plasticity.theta_slow.detach().norm())
        return total
    
    def run(self) -> PilotMetrics:
        """Run the pilot experiment."""
        start_time = time.time()
        step = 0
        
        # Training loop
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

                    step_metrics = benchmark_metrics(out)
                    mse = step_metrics["connection_prediction_mse"]
                    self.metrics.online_mse_curve.append(mse)
                    self.metrics.state_mse_curve.append(step_metrics["state_prediction_mse"])
                    self.metrics.readout_mse_curve.append(step_metrics["readout_mse"])

                    # Periodic consolidation (same interval across compared variants)
                    if (self.config.consolidation_interval > 0
                            and step % self.config.consolidation_interval == 0
                            and step > 0):
                        cons_stats = self.network.consolidate_all()
                        self.metrics.consolidation_calls += 1
                        self.metrics.consolidated_total += int(cons_stats.get("consolidated", 0))
                        self.metrics.gated_consolidated_total += int(
                            cons_stats.get("gated_consolidated", 0))
                        # consolidate_all sums per-connection stats; normalise
                        # back to per-connection means for reporting
                        n_conn = max(self.network.num_connections, 1)
                        if "gate_mean" in cons_stats:
                            self.metrics.gate_mean_sum += float(cons_stats["gate_mean"]) / n_conn
                        if "gate_open_fraction" in cons_stats:
                            self.metrics.gate_open_fraction_sum += (
                                float(cons_stats["gate_open_fraction"]) / n_conn)
                    
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
        self.metrics.final_connections = self.network.num_connections
        self.metrics.final_topology = self._get_topology()
        self.metrics.theta_slow_norm_final = self._theta_slow_norm()
        self.metrics.consolidation_output_max_diff = self._consolidation_output_check()

        # Final evaluation
        self._final_evaluation()

        return self.metrics

    def _consolidation_output_check(self) -> Optional[float]:
        """Gate-6 check: consolidation must preserve effective-connection output.

        Runs one probe step, consolidates, restores the state and repeats the
        identical probe step; returns the max |transmission| difference.
        """
        if self.config.consolidation_rate <= 0:
            return None
        batch = self.network.neuron_states.shape[0]
        probe = torch.randn(batch, self.config.num_neurons, self.config.neuron_dim,
                            device=self.device)
        saved_states = self.network.neuron_states.detach().clone()
        out1 = self.network.step(probe, target_next=None, learn=False)
        t1 = out1["transmissions"].detach().clone()
        self.network.neuron_states = saved_states.clone()
        self.network.consolidate_all()
        self.network.neuron_states = saved_states.clone()
        out2 = self.network.step(probe, target_next=None, learn=False)
        self.network.neuron_states = saved_states.clone()
        diff = (out2["transmissions"].detach() - t1).abs().max()
        return float(diff)
    
    def _evaluate_retention(self) -> float:
        """Evaluate on validation set."""
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
                        step_inp = inp[:, t].to(self.device)
                        step_target = target[:, t].to(self.device)
                        out = self.network.step(
                            step_inp,
                            target_next=step_target,
                            learn=False
                        )
                        total_mse += benchmark_metrics(out)["connection_prediction_mse"] * step_inp.shape[0]
                        count += step_inp.shape[0]
        
        return total_mse / max(count, 1)
    
    def _final_evaluation(self):
        """Run benchmark-specific final evaluation."""
        if self.config.benchmark == "compositional_discovery":
            self._evaluate_compositional()
        elif self.config.benchmark == "changing_rotation":
            self._evaluate_changing_rotation()
        
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
        
        from nnfs.benchmarks.validation import evaluate_compositional_discovery
        
        self.network.reset_states(batch_size=1)
        results = evaluate_compositional_discovery(
            self.network, self.test_loader, self.ground_truth_AC, self.device,
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
        )
        
        self.metrics.final_mse = results.get("test_mse", 0)
        self.metrics.discovery_score = results.get("discovery_score", 0)
        self.metrics.relative_to_oracle = results.get("relative_to_oracle", 0)
        self.metrics.relative_to_random = results.get("relative_to_random", 0)
        self.metrics.oracle_mse = results.get("oracle_mse", 0)
        self.metrics.random_mse = results.get("random_mse", 0)
    
    def _evaluate_changing_rotation(self):
        """Evaluate adaptation and retention with proper phase-relative measurements."""
        from nnfs.benchmarks.validation import evaluate_adaptation_retention
        
        self.network.reset_states(batch_size=1)
        results = evaluate_adaptation_retention(
            self.network, self.test_loader, [self.config.phase_length, 2 * self.config.phase_length], self.device,
            num_neurons=self.config.num_neurons,
            neuron_dim=self.config.neuron_dim,
        )
        
        self.metrics.phase_a_mse = results.get("phase_a_mse", 0)
        self.metrics.phase_b_mse = results.get("phase_b_mse", 0)
        self.metrics.phase_c_mse = results.get("phase_c_mse", 0)
        self.metrics.adaptation_steps_b = results.get("adaptation_steps", [0, 0])[0] if results.get("adaptation_steps") else 0
        self.metrics.adaptation_steps_c = results.get("adaptation_steps", [0, 0])[1] if len(results.get("adaptation_steps", [])) > 1 else 0
        self.metrics.retention_ratio = results.get("retention_ratio", 0)
        self.metrics.phase_a_state_mse = results.get("phase_a_state_mse", 0)
        self.metrics.phase_b_state_mse = results.get("phase_b_state_mse", 0)
        self.metrics.phase_c_state_mse = results.get("phase_c_state_mse", 0)
        self.metrics.phase_a_readout_mse = results.get("phase_a_readout_mse", 0)
        self.metrics.phase_b_readout_mse = results.get("phase_b_readout_mse", 0)
        self.metrics.phase_c_readout_mse = results.get("phase_c_readout_mse", 0)
        self.metrics.final_mse = results.get("phase_c_mse", 0)


def run_pilot_experiment(config: ExperimentConfig) -> PilotMetrics:
    """Run a single pilot experiment."""
    experiment = PilotExperiment(config)
    return experiment.run()


def run_pilot_matrix(
    benchmarks: List[str] = ["changing_rotation", "compositional_discovery"],
    seeds: List[int] = [42, 43, 44],
    output_dir: str = "pilot_results",
    configs: Optional[List[Tuple[ModelVariant, DreamVariant]]] = None,
    experiment_kwargs: Optional[Dict[str, Any]] = None,
    output_name: str = "pilot_results.json",
) -> Dict[str, Any]:
    """Run the 6-configuration pilot matrix.

    Incremental persistence: every completed run is appended to
    pilot_results_incremental.jsonl as it finishes, so a stop loses at most
    one run. A resume skips cells already present in that file.
    """

    if configs is None:
        configs = PILOT_CONFIGS
    if experiment_kwargs is None:
        experiment_kwargs = {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Per-output-file incremental log so matrix variants with the same cell
    # keys (e.g. the gate-5 D1x2 supplementary) don't skip each other
    incremental_path = output_path / (
        output_name.replace(".json", "") + "_incremental.jsonl")

    results: Dict[str, Any] = {}

    # Load previously completed cells (stop/crash resilience)
    done = set()
    if incremental_path.exists():
        with open(incremental_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a hard kill
                cell = (rec["benchmark"], rec["config_key"], rec["seed_key"])
                done.add(cell)
                results.setdefault(rec["benchmark"], {}).setdefault(
                    rec["config_key"], {})[rec["seed_key"]] = rec["metrics"]

    total_runs = len(benchmarks) * len(configs) * len(seeds)
    status_interval = max(1, total_runs // 3)
    completed = sum(
        len(seeds_done) for benchmark in benchmarks
        for seeds_done in results.get(benchmark, {}).values()
    )
    print(f"[matrix] total={total_runs} runs, already done={completed} "
          f"(incremental log: {incremental_path})")

    for benchmark in benchmarks:
        results.setdefault(benchmark, {})

        for model_variant, dream_variant in configs:
            key = f"{model_variant.value}_{dream_variant.value}"
            results[benchmark].setdefault(key, {})

            for seed in seeds:
                seed_key = f"seed_{seed}"
                if (benchmark, key, seed_key) in done:
                    print(f"skip (already done): {benchmark} | {key} | {seed_key}")
                    continue

                config = ExperimentConfig(
                    benchmark=benchmark,
                    model_variant=model_variant,
                    dream_variant=dream_variant,
                    seed=seed,
                    **experiment_kwargs,
                )

                print(f"Running: {benchmark} | {model_variant.value} | {dream_variant.value} | seed={seed}")

                try:
                    metrics = run_pilot_experiment(config)
                    metrics_dict = _metrics_to_dict(metrics)
                    results[benchmark][key][seed_key] = metrics_dict
                    # Durable append immediately after each completed run
                    with open(incremental_path, "a") as f:
                        f.write(json.dumps({
                            "benchmark": benchmark,
                            "config_key": key,
                            "seed_key": seed_key,
                            "metrics": metrics_dict,
                        }, default=str) + "\n")
                    print(f"  MSE={metrics.final_mse:.4f}")
                    if benchmark == "changing_rotation":
                        print(f"  Phase A={metrics.phase_a_mse:.4f}, B={metrics.phase_b_mse:.4f}, C={metrics.phase_c_mse:.4f}")
                        print(f"  Adapt B={metrics.adaptation_steps_b}, C={metrics.adaptation_steps_c}, Ret={metrics.retention_ratio:.4f}")
                    elif benchmark == "compositional_discovery":
                        print(f"  Disc={metrics.discovery_score:.4f}, vsOracle={metrics.relative_to_oracle:.4f}, vsRand={metrics.relative_to_random:.4f}")
                    dream_prec = "None" if metrics.dream_precision is None else f"{metrics.dream_precision:.4f}"
                    print(f"  Conn={metrics.final_connections} (+{metrics.connections_created}/-{metrics.connections_pruned}), DreamPrec={dream_prec}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[benchmark][key][seed_key] = {"error": str(e)}

                completed += 1
                if completed % status_interval == 0:
                    print(f"[progress] {completed}/{total_runs} runs complete")

    # Save consolidated result
    with open(output_path / output_name, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _metrics_to_dict(metrics: PilotMetrics) -> Dict:
    """Convert metrics to dictionary."""
    return {
        "final_mse": metrics.final_mse,
        "online_mse_final": np.mean(metrics.online_mse_curve[-100:]) if metrics.online_mse_curve else 0,
        # The three contract metrics recorded separately (primary first)
        "connection_mse_final": np.mean(metrics.online_mse_curve[-100:]) if metrics.online_mse_curve else 0,
        "state_mse_final": np.mean(metrics.state_mse_curve[-100:]) if metrics.state_mse_curve else 0,
        "readout_mse_final": np.mean(metrics.readout_mse_curve[-100:]) if metrics.readout_mse_curve else 0,
        "phase_a_mse": metrics.phase_a_mse,
        "phase_b_mse": metrics.phase_b_mse,
        "phase_c_mse": metrics.phase_c_mse,
        "phase_a_state_mse": metrics.phase_a_state_mse,
        "phase_b_state_mse": metrics.phase_b_state_mse,
        "phase_c_state_mse": metrics.phase_c_state_mse,
        "phase_a_readout_mse": metrics.phase_a_readout_mse,
        "phase_b_readout_mse": metrics.phase_b_readout_mse,
        "phase_c_readout_mse": metrics.phase_c_readout_mse,
        "adaptation_steps_b": metrics.adaptation_steps_b,
        "adaptation_steps_c": metrics.adaptation_steps_c,
        "retention_ratio": metrics.retention_ratio,
        "discovery_score": metrics.discovery_score,
        "relative_to_oracle": metrics.relative_to_oracle,
        "relative_to_random": metrics.relative_to_random,
        "oracle_mse": metrics.oracle_mse,
        "random_mse": metrics.random_mse,
        "initial_connections": metrics.initial_connections,
        "final_connections": metrics.final_connections,
        "connections_created": metrics.connections_created,
        "connections_pruned": metrics.connections_pruned,
        "structural_efficiency": metrics.structural_efficiency,
        "dream_hypotheses_generated": metrics.dream_hypotheses_generated,
        "dream_hypotheses_accepted": metrics.dream_hypotheses_accepted,
        "dream_precision": metrics.dream_precision,
        "dream_time": metrics.dream_time,
        "consolidation_calls": metrics.consolidation_calls,
        "consolidated_total": metrics.consolidated_total,
        "gated_consolidated_total": metrics.gated_consolidated_total,
        "gate_mean_avg": (
            metrics.gate_mean_sum / metrics.consolidation_calls
            if metrics.consolidation_calls else None
        ),
        "gate_open_fraction_avg": (
            metrics.gate_open_fraction_sum / metrics.consolidation_calls
            if metrics.consolidation_calls else None
        ),
        "theta_slow_norm_initial": metrics.theta_slow_norm_initial,
        "theta_slow_norm_final": metrics.theta_slow_norm_final,
        "consolidation_output_max_diff": metrics.consolidation_output_max_diff,
        "total_time": metrics.total_time,
        "initial_topology": metrics.initial_topology,
        "final_topology": metrics.final_topology,
    }


def _mean_or_none(values: List) -> Optional[float]:
    """Mean over non-None values; None when nothing measurable."""
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def analyze_pilot_results(results: Dict) -> Dict:
    """Analyze pilot results with focus on D1 vs D2 vs D3 comparison."""
    analysis = {}

    for benchmark, benchmark_results in results.items():
        analysis[benchmark] = {}

        for config_key, config_results in benchmark_results.items():
            # Average over seeds
            valid_seeds = [v for v in config_results.values() if 'error' not in v]
            if not valid_seeds:
                continue

            analysis[benchmark][config_key] = {
                'final_mse': np.mean([v.get('final_mse', 0) for v in valid_seeds]),
                'phase_a_mse': np.mean([v.get('phase_a_mse', 0) for v in valid_seeds]),
                'phase_b_mse': np.mean([v.get('phase_b_mse', 0) for v in valid_seeds]),
                'phase_c_mse': np.mean([v.get('phase_c_mse', 0) for v in valid_seeds]),
                'adaptation_steps_b': _mean_or_none([v.get('adaptation_steps_b') for v in valid_seeds]),
                'adaptation_steps_c': _mean_or_none([v.get('adaptation_steps_c') for v in valid_seeds]),
                'retention_ratio': np.mean([v.get('retention_ratio', 0) for v in valid_seeds]),
                'discovery_score': np.mean([v.get('discovery_score', 0) for v in valid_seeds]),
                'relative_to_oracle': np.mean([v.get('relative_to_oracle', 0) for v in valid_seeds]),
                'relative_to_random': np.mean([v.get('relative_to_random', 0) for v in valid_seeds]),
                'dream_precision': _mean_or_none([v.get('dream_precision') for v in valid_seeds]),
                'final_connections': np.mean([v.get('final_connections', 0) for v in valid_seeds]),
                'connections_created': np.mean([v.get('connections_created', 0) for v in valid_seeds]),
                'connections_pruned': np.mean([v.get('connections_pruned', 0) for v in valid_seeds]),
            }

    return analysis


if __name__ == "__main__":
    print("Running AION Pilot Experiments (Phase 5)")
    print("=" * 60)
    print("6 Configurations × 2 Benchmarks × 3 Seeds")
    print("=" * 60)
    
    results = run_pilot_matrix(
        benchmarks=["changing_rotation", "compositional_discovery"],
        seeds=[42, 43, 44],
        output_dir="pilot_results"
    )
    
    analysis = analyze_pilot_results(results)
    
    print("\n=== PILOT ANALYSIS ===")
    for benchmark, benchmark_analysis in analysis.items():
        print(f"\n{benchmark}:")
        for config_key, metrics in benchmark_analysis.items():
            print(f"  {config_key}:")
            print(f"    MSE={metrics['final_mse']:.4f}")
            if benchmark == "changing_rotation":
                print(f"    Phase A={metrics['phase_a_mse']:.4f}, B={metrics['phase_b_mse']:.4f}, C={metrics['phase_c_mse']:.4f}")
                print(f"    Adapt B={metrics['adaptation_steps_b']}, C={metrics['adaptation_steps_c']}, Ret={metrics['retention_ratio']:.4f}")
            elif benchmark == "compositional_discovery":
                print(f"    Disc={metrics['discovery_score']:.4f}, vsOracle={metrics['relative_to_oracle']:.4f}")
            dream_prec = "None" if metrics['dream_precision'] is None else f"{metrics['dream_precision']:.4f}"
            print(f"    Conn={metrics['final_connections']:.1f}, DreamPrec={dream_prec}")
    
    print("\nResults saved to pilot_results/pilot_results.json")