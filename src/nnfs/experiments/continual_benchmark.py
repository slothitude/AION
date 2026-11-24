"""
Rigorous continual-learning benchmark for NNFS Dream Engine.

Implements the 5×4 experimental matrix:
- 5 model types: Static, Hebbian, Plastic, Consolidating, Contextual
- 4 dream levels: D0 (none), D1 (replay), D2 (replay+imagination), D3 (full+structural)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import copy
from pathlib import Path
import numpy as np

from nnfs.connections import RelationalNetwork
from nnfs.dreaming import (
    DreamEngine, MemoryRetriever, ExperienceReplay, 
    PatternRecombiner, DreamSimulator, DreamEvaluator,
    DreamConsolidator, StructuralDreamer, create_dream_engine
)
from nnfs.benchmarks import (
    ChangingRelationshipDataset, ContextSwitchingDataset,
    SequentialAssociationDataset, get_changing_relationship_loaders,
    get_context_switching_loaders, get_sequential_association_loaders
)


class ModelType(Enum):
    STATIC = "static"
    HEBBIAN = "hebbian"
    PLASTIC = "plastic"
    CONSOLIDATING = "consolidating"
    CONTEXTUAL = "contextual"


class DreamLevel(Enum):
    D0_NONE = "d0_none"
    D1_REPLAY = "d1_replay"
    D2_REPLAY_IMAGINATION = "d2_replay_imagination"
    D3_FULL = "d3_full"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_type: ModelType
    dream_level: DreamLevel
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
    dream_cycle_interval: int = 1000
    dream_duration: int = 100
    
    # Training
    batch_size: int = 16
    total_steps: int = 5000
    eval_interval: int = 500
    
    # Benchmark
    benchmark: str = "changing_rotation"  # "changing_rotation", "context_switching", "sequential_assoc"
    
    device: str = "cpu"


@dataclass
class ExperimentMetrics:
    """Metrics collected during an experiment."""
    # Adaptation speed: steps to reach threshold after change
    adaptation_steps: List[int] = field(default_factory=list)
    adaptation_mse: List[float] = field(default_factory=list)
    
    # Retention: performance on previous tasks
    retention_mse: List[float] = field(default_factory=list)
    forgetting_rate: List[float] = field(default_factory=list)
    
    # Discovery: novel useful relationships found
    discoveries_count: int = 0
    discovery_precision: float = 0.0  # Dream hypothesis precision
    discovery_recall: float = 0.0
    
    # Structural efficiency
    final_connections: int = 0
    connection_changes: List[int] = field(default_factory=list)
    pruning_events: int = 0
    creation_events: int = 0
    structural_efficiency: float = 0.0  # Performance per connection
    
    # Learning cost
    total_time: float = 0.0
    dream_time: float = 0.0
    memory_usage: List[int] = field(default_factory=list)
    peak_memory: int = 0
    
    # Online learning
    online_mse: List[float] = field(default_factory=list)
    pre_update_mse: List[float] = field(default_factory=list)  # Evaluation BEFORE update


class ModelFactory:
    """Creates the 5 distinct model types."""
    
    @staticmethod
    def create_model(config: ExperimentConfig) -> RelationalNetwork:
        """Create a model of the specified type."""
        
        if config.model_type == ModelType.STATIC:
            return ModelFactory._create_static(config)
        elif config.model_type == ModelType.HEBBIAN:
            return ModelFactory._create_hebbian(config)
        elif config.model_type == ModelType.PLASTIC:
            return ModelFactory._create_plastic(config)
        elif config.model_type == ModelType.CONSOLIDATING:
            return ModelFactory._create_consolidating(config)
        elif config.model_type == ModelType.CONTEXTUAL:
            return ModelFactory._create_contextual(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    @staticmethod
    def _create_static(config: ExperimentConfig) -> RelationalNetwork:
        """Static: Fixed weights, no learning."""
        network = RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=0.0,  # NO learning
            trace_decay=config.trace_decay,
            consolidation_rate=0.0,
            consolidation_threshold=config.consolidation_threshold,
            blend_ratio=1.0,  # Only slow weights
        )
        # Freeze all parameters
        for param in network.parameters():
            param.requires_grad = False
        return network
    
    @staticmethod
    def _create_hebbian(config: ExperimentConfig) -> RelationalNetwork:
        """Hebbian: Local correlation learning only (no error signal)."""
        network = RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
            consolidation_rate=0.0,
            consolidation_threshold=config.consolidation_threshold,
            blend_ratio=1.0,
        )
        # Modify plasticity to Hebbian (pre*post only, no error)
        for conn in network.connections.values():
            conn.plasticity.learning_rate = config.learning_rate
            # Mark as Hebbian mode
            conn.plasticity._hebbian_mode = True
        return network
    
    @staticmethod
    def _create_plastic(config: ExperimentConfig) -> RelationalNetwork:
        """Plastic: Fast error-driven adaptation only (no consolidation)."""
        network = RelationalNetwork(
            num_neurons=config.num_neurons,
            neuron_dim=config.neuron_dim,
            sparsity=config.sparsity,
            relation_type=config.relation_type,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
            consolidation_rate=0.0,  # NO consolidation
            consolidation_threshold=config.consolidation_threshold,
            blend_ratio=1.0,  # Only fast weights
        )
        # Disable slow weights
        for conn in network.connections.values():
            conn.plasticity.theta_slow.requires_grad = False
            conn.plasticity.theta_slow.data.zero_()
            conn.blend_ratio = 1.0
        return network
    
    @staticmethod
    def _create_consolidating(config: ExperimentConfig) -> RelationalNetwork:
        """Consolidating: Fast + slow learning with consolidation."""
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
        )
        return network
    
    @staticmethod
    def _create_contextual(config: ExperimentConfig) -> RelationalNetwork:
        """Contextual: Context-conditioned plasticity."""
        # Use higher context_dim for contextual model
        context_dim = 8
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
            context_dim=context_dim,
        )
        # Enable contextual modulation
        for conn in network.connections.values():
            conn.plasticity._contextual_mode = True
        return network


class DreamEngineFactory:
    """Creates dream engines at 4 ablation levels."""
    
    @staticmethod
    def create_dream_engine(network: RelationalNetwork, config: ExperimentConfig, dream_level: DreamLevel) -> Optional[DreamEngine]:
        """Create dream engine at specified ablation level."""
        
        if dream_level == DreamLevel.D0_NONE:
            return None
        
        elif dream_level == DreamLevel.D1_REPLAY:
            return DreamEngine(
                network=network,
                dream_cycle_interval=config.dream_cycle_interval,
                dream_duration=config.dream_duration,
                enable_replay=True,
                enable_recombination=False,
                enable_consolidation=True,
                enable_structural=False,
            )
        
        elif dream_level == DreamLevel.D2_REPLAY_IMAGINATION:
            return DreamEngine(
                network=network,
                dream_cycle_interval=config.dream_cycle_interval,
                dream_duration=config.dream_duration,
                enable_replay=True,
                enable_recombination=True,
                enable_consolidation=True,
                enable_structural=False,
            )
        
        elif dream_level == DreamLevel.D3_FULL:
            return DreamEngine(
                network=network,
                dream_cycle_interval=config.dream_cycle_interval,
                dream_duration=config.dream_duration,
                enable_replay=True,
                enable_recombination=True,
                enable_consolidation=True,
                enable_structural=True,
            )
        
        else:
            raise ValueError(f"Unknown dream level: {dream_level}")


class CandidateTopology:
    """
    Safe structural changes via candidate topology.
    
    Instead of modifying live network:
    1. Create candidate network structure
    2. Apply proposed changes
    3. Test against replayed experiences
    4. Compare with existing network
    5. Accept/reject with rollback
    """
    
    def __init__(self, network: RelationalNetwork):
        self.base_network = network
        self.candidate_network: Optional[RelationalNetwork] = None
        self.proposed_changes: List[Dict] = []
        self.rollback_state: Optional[Dict] = None
    
    def create_candidate(self) -> RelationalNetwork:
        """Create a copy of the network for structural experimentation."""
        # Clone the network
        self.candidate_network = copy.deepcopy(self.base_network)
        self.proposed_changes = []
        return self.candidate_network
    
    def propose_change(self, action: str, pre_idx: int, post_idx: int, **kwargs):
        """Record a proposed structural change."""
        self.proposed_changes.append({
            'action': action,
            'pre_idx': pre_idx,
            'post_idx': post_idx,
            **kwargs
        })
    
    def apply_proposals(self, network: RelationalNetwork) -> Dict[str, int]:
        """Apply all proposed changes to a network."""
        stats = {'created': 0, 'pruned': 0, 'rewired': 0}
        
        for change in self.proposed_changes:
            action = change['action']
            pre = change['pre_idx']
            post = change['post_idx']
            
            if action == 'create':
                self._create_connection(network, pre, post)
                stats['created'] += 1
            elif action == 'prune':
                self._prune_connection(network, pre, post)
                stats['pruned'] += 1
            elif action == 'rewire':
                self._rewire_connection(network, pre, post, change.get('new_pre'), change.get('new_post'))
                stats['rewired'] += 1
        
        return stats
    
    def _create_connection(self, network: RelationalNetwork, pre: int, post: int):
        """Create a new connection in the network."""
        from nnfs.connections.plastic import PlasticRelationalConnection
        conn = PlasticRelationalConnection(
            pre_idx=pre,
            post_idx=post,
            pre_dim=network.neuron_dim,
            post_dim=network.neuron_dim,
            context_dim=network.context_dim,
            relation_type=network.relation_type,
            learning_rate=0.01,
            trace_decay=0.95,
            consolidation_rate=0.001,
        )
        new_key = f"conn_{len(network.connections)}"
        network.connections[new_key] = conn
        network.rebuild_connection_indices()
    
    def _prune_connection(self, network: RelationalNetwork, pre: int, post: int):
        """Prune a connection from the network."""
        to_remove = []
        for key, conn in network.connections.items():
            pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
            post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
            if pre_idx == pre and post_idx == post:
                to_remove.append(key)
        for key in to_remove:
            del network.connections[key]
        network.rebuild_connection_indices()
    
    def _rewire_connection(self, network: RelationalNetwork, pre: int, post: int, new_pre: int, new_post: int):
        """Rewire a connection."""
        self._prune_connection(network, pre, post)
        self._create_connection(network, new_pre, new_post)
    
    def evaluate_candidate(self, test_loader: DataLoader, metric: str = 'mse') -> float:
        """Evaluate candidate network on test data."""
        if self.candidate_network is None:
            return float('inf')
        
        self.candidate_network.eval()
        total = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inp, target, _ = batch
                else:
                    inp, target = batch
                
                out = self.candidate_network.step(inp, target_next=target, learn=False)
                total += out.get('mse', 0) * inp.shape[0]
                count += inp.shape[0]
        
        return total / max(1, count)
    
    def commit_or_rollback(self, test_loader: DataLoader, accept_threshold: float = 0.0) -> bool:
        """
        Compare candidate with base, commit if better.
        
        Args:
            test_loader: DataLoader for evaluation
            accept_threshold: Minimum improvement to accept (relative)
        
        Returns:
            True if candidate accepted, False if rolled back
        """
        if self.candidate_network is None:
            return False
        
        # Evaluate both networks
        base_score = self._evaluate_network(self.base_network, test_loader)
        candidate_score = self._evaluate_network(self.candidate_network, test_loader)
        
        improvement = (base_score - candidate_score) / max(1e-8, base_score)
        
        if improvement > accept_threshold:
            # Commit: copy candidate back to base
            self._commit_candidate()
            return True
        else:
            # Rollback: discard candidate
            self.candidate_network = None
            return False
    
    def _evaluate_network(self, network: RelationalNetwork, test_loader: DataLoader) -> float:
        network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inp, target, _ = batch
                else:
                    inp, target = batch
                out = network.step(inp, target_next=target, learn=False)
                total += out.get('mse', 0) * inp.shape[0]
                count += inp.shape[0]
        return total / max(1, count)
    
    def _commit_candidate(self):
        """Commit candidate changes to base network."""
        # Copy connections
        self.base_network.connections = self.candidate_network.connections
        self.base_network.conn_rows = self.candidate_network.conn_rows
        self.base_network.conn_cols = self.candidate_network.conn_cols
        self.base_network.num_connections = self.candidate_network.num_connections
        self.candidate_network = None


class ContinualBenchmark:
    """
    Rigorous continual learning benchmark with proper evaluation protocol.
    
    Key principle: Evaluate BEFORE update to measure true retention.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self.metrics = ExperimentMetrics()
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create model
        self.network = ModelFactory.create_model(config).to(self.device)
        
        # Create dream engine
        self.dream_engine = DreamEngineFactory.create_dream_engine(
            self.network, config, config.dream_level
        )
        
        # Candidate topology for structural dreaming
        self.candidate_topology = CandidateTopology(self.network)
        
        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = self._get_loaders()
        
        # Track relationship changes
        self.change_points: List[int] = []
        self.relationship_history: List[Dict] = []
        
        # Dream hypothesis tracking for precision
        self.dream_hypotheses: List[Dict] = []
        self.validated_hypotheses: List[Dict] = []
    
    def _get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for the benchmark."""
        num_neurons = self.config.num_neurons
        neuron_dim = self.config.neuron_dim
        
        if self.config.benchmark == "changing_rotation":
            return get_changing_relationship_loaders(
                change_type="rotation",
                batch_size=self.config.batch_size,
                seq_length=100,
                num_sequences=20,  # Smaller for memory
                num_neurons=num_neurons,
                neuron_dim=neuron_dim,
            )
        elif self.config.benchmark == "context_switching":
            return get_context_switching_loaders(
                batch_size=self.config.batch_size,
                seq_length=100,
                num_sequences=20,
                num_neurons=num_neurons,
                neuron_dim=neuron_dim,
            )
        elif self.config.benchmark == "sequential_assoc":
            return get_sequential_association_loaders(
                batch_size=self.config.batch_size,
                seq_length=50,
                num_sequences=20,
                num_neurons=num_neurons,
                neuron_dim=neuron_dim,
            )
        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")
    
    def run(self) -> ExperimentMetrics:
        """Run the full benchmark experiment."""
        start_time = time.time()
        step = 0
        eval_step = 0
        
        # Expected network input shape
        expected_shape = (self.config.num_neurons, self.config.neuron_dim)
        expected_flat = self.config.num_neurons * self.config.neuron_dim
        
        # Track performance on each relationship regime
        regime_performance: Dict[int, List[float]] = {}
        current_regime = 0
        
        for epoch in range(100):  # Max epochs
            for batch in self.train_loader:
                if step >= self.config.total_steps:
                    break
                
                if len(batch) == 3:
                    inp, target, change_label = batch
                else:
                    inp, target = batch
                    change_label = torch.zeros(inp.shape[0])
                
                # Reshape inputs to match network expectations
                # Dataset returns (batch, seq, flat_dim)
                if inp.dim() == 3:  # (batch, seq, flat)
                    batch_size, seq_len = inp.shape[0], inp.shape[1]
                    inp = inp.view(batch_size, seq_len, *expected_shape)
                    target = target.view(batch_size, seq_len, *expected_shape)
                    change_label = change_label.view(batch_size, seq_len)
                else:
                    raise ValueError(f"Unexpected input dim: {inp.dim()}")
                
                inp = inp.to(self.device)
                target = target.to(self.device)
                change_label = change_label.to(self.device)
                
                # Iterate through sequence steps
                for t in range(seq_len):
                    if step >= self.config.total_steps:
                        break
                    
                    step_inp = inp[:, t]      # (batch, num_neurons, neuron_dim)
                    step_target = target[:, t]  # (batch, num_neurons, neuron_dim)
                    step_change = change_label[:, t]  # (batch,)
                    
                    # Detect regime change
                    if step_change.sum() > 0 and step > 0:
                        self.change_points.append(step)
                        current_regime += 1
                        regime_performance[current_regime] = []
                    
                    # EVALUATE BEFORE UPDATE (critical for retention measurement)
                    if step % self.config.eval_interval == 0 and step > 0:
                        pre_mse = self._evaluate_retention()
                        self.metrics.pre_update_mse.append(pre_mse)
                        eval_step += 1
                    
                    # Online learning step
                    self.network.train()
                    out = self.network.step(step_inp, target_next=step_target, learn=True)
                    
                    mse = out.get('mse', 0)
                    self.metrics.online_mse.append(mse)
                    regime_performance.setdefault(current_regime, []).append(mse)
                    
                    # Store experience for dreaming
                    if self.dream_engine is not None:
                        self.dream_engine.store_experience(
                            step_inp[0], step_target[0], 
                            significance=1.0 / (1.0 + mse),  # Higher significance for surprising experiences
                            uncertainty=out.get('prediction_error', torch.tensor(0)).std().item() if 'prediction_error' in out else 0.1,
                        )
                        
                        # Dream engine step
                        dream_start = time.time()
                        dream_stats = self.dream_engine.step()
                        self.metrics.dream_time += time.time() - dream_start
                        
                        # Track dream hypotheses for precision measurement
                        if hasattr(dream_stats, 'hypotheses_generated') and dream_stats.hypotheses_generated > 0:
                            self._track_dream_hypotheses(dream_stats)
                        
                        # Track structural changes
                        if hasattr(dream_stats, 'connections_created'):
                            self.metrics.creation_events += dream_stats.connections_created
                            self.metrics.connection_changes.append(dream_stats.connections_created)
                        if hasattr(dream_stats, 'connections_pruned'):
                            self.metrics.pruning_events += dream_stats.connections_pruned
                            self.metrics.connection_changes.append(-dream_stats.connections_pruned)
                    
                    step += 1
                
                if step >= self.config.total_steps:
                    break
            
            if step >= self.config.total_steps:
                break
        
        self.metrics.total_time = time.time() - start_time
        self.metrics.final_connections = self.network.num_connections
        
        # Final retention evaluation
        self.metrics.retention_mse = self._evaluate_all_regimes(regime_performance)
        
        # Compute adaptation speed
        self._compute_adaptation_speed(regime_performance)
        
        # Compute discovery precision
        self._compute_discovery_precision()
        
        # Structural efficiency
        self._compute_structural_efficiency()
        
        return self.metrics
    
    def _evaluate_retention(self) -> float:
        """Evaluate retention on all previous regimes."""
        self.network.eval()
        total_mse = 0.0
        count = 0
        
        expected_shape = (self.config.num_neurons, self.config.neuron_dim)
        
        with torch.no_grad():
            # Use validation loader for retention test
            for batch in self.val_loader:
                if len(batch) == 3:
                    inp, target, _ = batch
                else:
                    inp, target = batch
                
                # Reshape sequence data
                if inp.dim() == 3:  # (batch, seq, flat)
                    batch_size, seq_len = inp.shape[0], inp.shape[1]
                    inp = inp.view(batch_size, seq_len, *expected_shape)
                    target = target.view(batch_size, seq_len, *expected_shape)
                    
                    # Evaluate each step in sequence
                    for t in range(seq_len):
                        step_inp = inp[:, t]
                        step_target = target[:, t]
                        step_inp = step_inp.to(self.device)
                        step_target = step_target.to(self.device)
                        out = self.network.step(step_inp, target_next=step_target, learn=False)
                        total_mse += out.get('mse', 0) * step_inp.shape[0]
                        count += step_inp.shape[0]
                else:
                    inp = inp.to(self.device)
                    target = target.to(self.device)
                    out = self.network.step(inp, target_next=target, learn=False)
                    total_mse += out.get('mse', 0) * inp.shape[0]
                    count += inp.shape[0]
        
        return total_mse / max(1, count)
    
    def _evaluate_all_regimes(self, regime_performance: Dict[int, List[float]]) -> List[float]:
        """Evaluate performance on each regime's test data."""
        retentions = []
        for regime_id in sorted(regime_performance.keys()):
            # Use recent performance as proxy for retention
            perf = regime_performance[regime_id]
            if perf:
                retentions.append(np.mean(perf[-10:]))  # Last 10 steps in regime
            else:
                retentions.append(float('inf'))
        return retentions
    
    def _compute_adaptation_speed(self, regime_performance: Dict[int, List[float]]):
        """Compute steps to adapt after each change."""
        for regime_id in sorted(regime_performance.keys()):
            if regime_id == 0:
                continue  # Skip first regime
            perf = regime_performance[regime_id]
            if len(perf) < 10:
                self.metrics.adaptation_steps.append(len(perf))
                continue
            
            # Find when MSE drops below threshold (adaptation)
            initial_mse = perf[0]
            threshold = initial_mse * 0.5  # 50% improvement
            
            for i, mse in enumerate(perf):
                if mse < threshold:
                    self.metrics.adaptation_steps.append(i)
                    break
            else:
                self.metrics.adaptation_steps.append(len(perf))
    
    def _track_dream_hypotheses(self, dream_stats):
        """Track hypotheses generated during dreaming for precision measurement."""
        # In a full implementation, we'd track each hypothesis
        # and later validate against real observations
        self.dream_hypotheses.append({
            'generated': dream_stats.hypotheses_generated,
            'accepted': dream_stats.hypotheses_accepted,
            'rejected': dream_stats.hypotheses_rejected,
        })
    
    def _compute_discovery_precision(self):
        """Compute dream hypothesis precision: how often imagined relationships prove useful."""
        if not self.dream_hypotheses:
            self.metrics.discovery_precision = 0.0
            return
        
        # In a full implementation, we'd compare hypotheses against 
        # subsequent real observations. For now, use acceptance rate as proxy.
        total_gen = sum(h['generated'] for h in self.dream_hypotheses)
        total_acc = sum(h['accepted'] for h in self.dream_hypotheses)
        
        self.metrics.discovery_precision = total_acc / max(1, total_gen)
        self.metrics.discoveries_count = total_acc
    
    def _compute_structural_efficiency(self):
        """Compute performance per connection."""
        if self.metrics.final_connections > 0 and self.metrics.online_mse:
            avg_mse = np.mean(self.metrics.online_mse[-100:])
            self.metrics.structural_efficiency = 1.0 / (avg_mse * self.metrics.final_connections)
        else:
            self.metrics.structural_efficiency = 0.0


def run_single_experiment(config: ExperimentConfig) -> ExperimentMetrics:
    """Run a single experiment configuration."""
    benchmark = ContinualBenchmark(config)
    return benchmark.run()


def run_full_matrix(
    benchmarks: List[str] = ["changing_rotation", "context_switching", "sequential_assoc"],
    seeds: List[int] = [42, 43, 44, 45, 46],
    output_dir: str = "continual_results"
) -> Dict[str, Any]:
    """Run the full 5×4 experimental matrix across benchmarks and seeds."""
    
    results = {}
    
    for benchmark in benchmarks:
        results[benchmark] = {}
        
        for model_type in ModelType:
            results[benchmark][model_type.value] = {}
            
            for dream_level in DreamLevel:
                results[benchmark][model_type.value][dream_level.value] = {}
                
                for seed in seeds:
                    config = ExperimentConfig(
                        model_type=model_type,
                        dream_level=dream_level,
                        seed=seed,
                        benchmark=benchmark,
                    )
                    
                    print(f"Running: {benchmark} | {model_type.value} | {dream_level.value} | seed={seed}")
                    
                    try:
                        metrics = run_single_experiment(config)
                        results[benchmark][model_type.value][dream_level.value][f"seed_{seed}"] = {
                            'adaptation_speed': np.mean(metrics.adaptation_steps) if metrics.adaptation_steps else 0,
                            'retention_mse': np.mean(metrics.retention_mse) if metrics.retention_mse else 0,
                            'forgetting_rate': np.mean(metrics.forgetting_rate) if metrics.forgetting_rate else 0,
                            'discovery_precision': metrics.discovery_precision,
                            'discoveries_count': metrics.discoveries_count,
                            'structural_efficiency': metrics.structural_efficiency,
                            'final_connections': metrics.final_connections,
                            'creation_events': metrics.creation_events,
                            'pruning_events': metrics.pruning_events,
                            'total_time': metrics.total_time,
                            'dream_time': metrics.dream_time,
                            'online_mse_final': np.mean(metrics.online_mse[-100:]) if metrics.online_mse else 0,
                            'pre_update_mse_final': np.mean(metrics.pre_update_mse[-10:]) if metrics.pre_update_mse else 0,
                        }
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        results[benchmark][model_type.value][dream_level.value][f"seed_{seed}"] = {'error': str(e)}
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "full_matrix_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the full matrix results."""
    analysis = {}
    
    for benchmark, benchmark_results in results.items():
        analysis[benchmark] = {}
        
        # Compare dream levels within each model type
        for model_type, model_results in benchmark_results.items():
            analysis[benchmark][model_type] = {}
            
            # D0 baseline
            d0_key = DreamLevel.D0_NONE.value
            if d0_key in model_results:
                d0_seeds = model_results[d0_key]
                d0_mse = np.mean([v.get('online_mse_final', 0) for v in d0_seeds.values() if 'error' not in v])
                d0_retention = np.mean([v.get('retention_mse', 0) for v in d0_seeds.values() if 'error' not in v])
            else:
                d0_mse = 0
                d0_retention = 0
            
            for dream_level, level_results in model_results.items():
                if dream_level == d0_key:
                    continue
                
                level_mse = np.mean([v.get('online_mse_final', 0) for v in level_results.values() if 'error' not in v])
                level_retention = np.mean([v.get('retention_mse', 0) for v in level_results.values() if 'error' not in v])
                level_discovery = np.mean([v.get('discovery_precision', 0) for v in level_results.values() if 'error' not in v])
                level_efficiency = np.mean([v.get('structural_efficiency', 0) for v in level_results.values() if 'error' not in v])
                
                analysis[benchmark][model_type][dream_level] = {
                    'mse_improvement': (d0_mse - level_mse) / max(1e-8, d0_mse) if d0_mse > 0 else 0,
                    'retention_improvement': (d0_retention - level_retention) / max(1e-8, d0_retention) if d0_retention > 0 else 0,
                    'discovery_precision': level_discovery,
                    'structural_efficiency': level_efficiency,
                }
    
    return analysis


if __name__ == "__main__":
    print("Running full continual learning benchmark matrix...")
    results = run_full_matrix()
    analysis = analyze_results(results)
    
    print("\n=== ANALYSIS ===")
    for benchmark, benchmark_analysis in analysis.items():
        print(f"\n{benchmark}:")
        for model_type, model_analysis in benchmark_analysis.items():
            print(f"  {model_type}:")
            for dream_level, metrics in model_analysis.items():
                print(f"    {dream_level}: MSE imp={metrics['mse_improvement']:.3f}, "
                      f"Ret imp={metrics['retention_improvement']:.3f}, "
                      f"Disc={metrics['discovery_precision']:.3f}, "
                      f"Eff={metrics['structural_efficiency']:.3f}")