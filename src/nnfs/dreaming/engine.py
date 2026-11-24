"""
Dream Engine - Orchestrates all dreaming subsystems.

The Dream Engine coordinates the full offline learning cycle:
1. Memory Retrieval: Select important experiences
2. Experience Replay: Rehearse actual observations
3. Pattern Recombination: Generate hypothetical relationships
4. Dream Simulation: Evaluate hypothetical relationships
5. Dream Evaluation: Score and filter discoveries
6. Consolidation: Update long-term memory
7. Structural Dreaming: Experiment with network topology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import time

from nnfs.dreaming.retriever import MemoryRetriever, RetrievedExperience, create_retriever
from nnfs.dreaming.replay import ExperienceReplay, create_experience_replay
from nnfs.dreaming.recombiner import PatternRecombiner, HypothesizedRelationship, create_pattern_recombiner
from nnfs.dreaming.simulator import DreamSimulator, SimulationResult, create_dream_simulator
from nnfs.dreaming.evaluator import DreamEvaluator, EvaluationResult, create_dream_evaluator
from nnfs.dreaming.consolidator import DreamConsolidator, ConsolidationResult, create_dream_consolidator
from nnfs.dreaming.structural import StructuralDreamer, StructuralChangeResult, create_structural_dreamer


@dataclass
class DreamCycleResult:
    """Complete result of one dream cycle."""
    cycle_id: int
    duration: float
    
    # Retrieval
    experiences_retrieved: int
    retrieval_stats: Dict
    
    # Replay
    experiences_replayed: int
    replay_stats: Dict
    
    # Recombination
    hypotheses_generated: int
    hypotheses_tested: int
    
    # Simulation
    simulations_run: int
    hypotheses_accepted: int
    hypotheses_rejected: int
    hypotheses_deferred: int
    
    # Evaluation
    evaluations_performed: int
    accepted: int
    rejected: int
    deferred: int
    
    # Consolidation
    consolidations_performed: int
    connections_updated: int
    structural_changes: Dict[str, int]
    
    # Structural dreaming
    proposals_generated: int
    proposals_accepted: int
    connections_created: int
    connections_pruned: int
    
    # Overall
    total_duration: float
    memory_stats: Dict
    safeguards_triggered: List[str]


class DreamEngine(nn.Module):
    """
    Orchestrates the complete offline learning cycle.
    
    The Dream Engine coordinates all dreaming subsystems to enable
    continuous offline learning from accumulated experience.
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        dream_cycle_interval: int = 1000,  # Steps between dream cycles
        dream_duration: int = 100,  # Steps per dream cycle
        enable_replay: bool = True,
        enable_recombination: bool = True,
        enable_consolidation: bool = True,
        enable_structural: bool = True,
        max_dream_duration: int = 200,
    ):
        super().__init__()
        self.network = network
        self.dream_cycle_interval = dream_cycle_interval
        self.dream_duration = dream_duration
        self.enable_replay = enable_replay
        self.enable_recombination = enable_recombination
        self.enable_consolidation = enable_consolidation
        self.enable_structural = enable_structural
        self.max_dream_duration = max_dream_duration
        
        # Initialize subsystems
        experience_dim = network.num_neurons * network.neuron_dim
        self.retriever = create_retriever(
            max_memory=10000,
            context_dim=network.context_dim,
            experience_dim=experience_dim,
        )
        
        self.experience_replay = ExperienceReplay(
            network=network,
            retriever=self.retriever,
            replay_batch_size=16,
            replay_steps=1,
            learning_rate_scale=0.1,
        )
        
        self.recombiner = PatternRecombiner(
            network=network,
            max_hypotheses_per_step=10,
            composition_depth=2,
            analogy_threshold=0.7,
        )
        
        self.simulator = DreamSimulator(
            network=network,
            simulation_steps=5,
            consistency_weight=0.4,
            novelty_weight=0.3,
            utility_weight=0.3,
        )
        
        self.evaluator = DreamEvaluator(
            network=network,
            consistency_weight=0.3,
            novelty_weight=0.2,
            efficiency_weight=0.2,
            forgetting_weight=0.3,
            acceptance_threshold=0.6,
        )
        
        self.consolidator = DreamConsolidator(
            network=network,
            consolidation_rate=0.001,
            max_consolidation_per_step=10,
            forgetting_protection=0.99,
        )
        
        self.structural_dreamer = StructuralDreamer(
            network=network,
            max_proposals_per_dream=5,
            creation_threshold=0.7,
            pruning_threshold=0.1,
        )
        
        # Dream scheduling
        self.register_buffer("dream_cycle_count", torch.tensor(0))
        self.register_buffer("steps_since_dream", torch.tensor(0))
        self.register_buffer("total_dream_time", torch.tensor(0.0))
        
        # Statistics
        self.register_buffer("total_dream_cycles", torch.tensor(0))
        self.register_buffer("total_dream_steps", torch.tensor(0))
        
    def should_dream(self) -> bool:
        """Check if it's time for a dream cycle."""
        return self.steps_since_dream.item() >= self.dream_cycle_interval
    
    def step(self) -> Dict:
        """
        Called each step to check if dreaming should occur.
        
        Returns dream statistics if a dream cycle occurred.
        """
        self.steps_since_dream.add_(1)
        
        if self.should_dream():
            return self.run_dream_cycle()
        
        return {"dreamed": False}
    
    def run_dream_cycle(self) -> DreamCycleResult:
        """
        Run a complete dream cycle.
        
        Returns comprehensive statistics about the dream cycle.
        """
        start_time = time.time()
        cycle_id = self.dream_cycle_count.item()
        
        # Reset dream timer
        self.steps_since_dream.zero_()
        self.dream_cycle_count.add_(1)
        
        # Initialize result tracking
        cycle_result = DreamCycleResult(
            cycle_id=cycle_id,
            duration=0.0,
            experiences_retrieved=0,
            retrieval_stats={},
            experiences_replayed=0,
            replay_stats={},
            hypotheses_generated=0,
            hypotheses_tested=0,
            simulations_run=0,
            hypotheses_accepted=0,
            hypotheses_rejected=0,
            hypotheses_deferred=0,
            evaluations_performed=0,
            accepted=0,
            rejected=0,
            deferred=0,
            consolidations_performed=0,
            connections_updated=0,
            structural_changes={},
            proposals_generated=0,
            proposals_accepted=0,
            connections_created=0,
            connections_pruned=0,
            total_duration=0.0,
            memory_stats={},
            safeguards_triggered=[],
        )
        
        # Store initial network state for comparison
        initial_state = self._get_network_state()
        
        # Phase 1: Memory Retrieval
        retrieved = self.retriever.retrieve(
            num_experiences=32,
            min_score=0.1,
        )
        cycle_result.experiences_retrieved = len(retrieved)
        cycle_result.retrieval_stats = self.retriever.get_stats()
        
        # Phase 2: Experience Replay
        if self.enable_replay and retrieved:
            replay_stats = self.experience_replay.replay_episodic(
                num_experiences=min(32, len(retrieved)),
            )
            cycle_result.experiences_replayed = replay_stats.get("replayed", 0)
            cycle_result.replay_stats = replay_stats
            self.total_dream_steps.add_(replay_stats.get("replayed", 0))
        
        # Phase 3: Pattern Recombination
        sim_results = []
        if self.enable_recombination:
            hypotheses = self.recombiner.generate_hypotheses(
                max_hypotheses=20,
            )
            cycle_result.hypotheses_generated = len(hypotheses)
            cycle_result.hypotheses_tested = len(hypotheses)
            
            # Phase 4: Dream Simulation
            if hypotheses:
                sim_results = self.simulator.simulate_batch(
                    hypotheses=hypotheses,
                    initial_state=torch.zeros(1),  # Would use actual state
                )
                cycle_result.simulations_run = len(sim_results)
                
                # Count recommendations
                for result in sim_results:
                    if result.recommendation == "accept":
                        cycle_result.hypotheses_accepted += 1
                    elif result.recommendation == "reject":
                        cycle_result.hypotheses_rejected += 1
                    else:
                        cycle_result.hypotheses_deferred += 1
        
        # Phase 5: Dream Evaluation
        evaluations = []
        if self.enable_consolidation and sim_results:
            for sim in sim_results:
                eval_result = self.evaluator.evaluate(sim)
                evaluations.append(eval_result)
            
            cycle_result.evaluations_performed = len(evaluations)
            for eval_result in evaluations:
                if eval_result.acceptance == "accept":
                    cycle_result.accepted += 1
                elif eval_result.acceptance == "reject":
                    cycle_result.rejected += 1
                else:
                    cycle_result.deferred += 1
        
        # Phase 6: Consolidation
        if self.enable_consolidation and evaluations:
            accept_evals = [e for e in evaluations if e.acceptance == "accept"]
            if accept_evals:
                cons_results = self.consolidator.consolidate(accept_evals)
                cycle_result.consolidations_performed = len(cons_results)
                for cr in cons_results:
                    cycle_result.connections_updated += cr.connections_updated
                    cycle_result.safeguards_triggered.extend(cr.safeguards_triggered)
                    # Owner fix 1 (2026-11-25): consolidation-side creations and
                    # prunes count toward the cycle's structural totals. Only
                    # Phase 7 structural-dreamer changes were counted before, so
                    # a D2 run reported connections_created=0 while the topology
                    # grew (36 -> 61 in the pilot).
                    cycle_result.connections_created += cr.structural_changes.get("created", 0)
                    cycle_result.connections_pruned += cr.structural_changes.get("pruned", 0)
                    for k, v in cr.structural_changes.items():
                        cycle_result.structural_changes[k] = cycle_result.structural_changes.get(k, 0) + v
        
        # Phase 7: Structural Dreaming
        if self.enable_structural:
            struct_results = self.structural_dreamer.dream_step()
            for scr in struct_results:
                cycle_result.proposals_generated += 1
                if scr.accepted:
                    cycle_result.proposals_accepted += 1
                    if scr.proposal.action == "create":
                        cycle_result.connections_created += 1
                    elif scr.proposal.action == "prune":
                        cycle_result.connections_pruned += 1
        
        # Finalize
        cycle_result.total_duration = time.time() - time.time() + 0.001  # Placeholder
        cycle_result.total_duration = time.time() - start_time if 'start_time' in locals() else 0.0
        cycle_result.memory_stats = self.retriever.get_stats()
        cycle_result.safeguards_triggered = list(set(cycle_result.safeguards_triggered))
        
        # Store dream state for next cycle
        self._last_state = self._get_network_state()
        
        return cycle_result
    
    def _get_network_state(self) -> Dict:
        """Get current network state for comparison."""
        return {
            "connection_weights": {
                k: conn.plasticity.get_combined_weight().data.clone()
                for k, conn in self.network.connections.items()
            },
            "slow_weights": {
                k: conn.plasticity.theta_slow.data.clone()
                for k, conn in self.network.connections.items()
                if hasattr(conn.plasticity, 'theta_slow')
            },
            "fast_weights": {
                k: conn.plasticity.u_fast.data.clone()
                for k, conn in self.network.connections.items()
                if hasattr(conn.plasticity, 'u_fast')
            },
        }
    
    def store_experience(
        self,
        input_signal: torch.Tensor,
        target_next: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        significance: float = 1.0,
        uncertainty: float = 0.0,
        context_id: Optional[int] = None,
    ):
        """Store experience for future dreaming."""
        self.retriever.store(
            input_signal=input_signal,
            target_next=target_next,
            context=context,
            significance=significance,
            uncertainty=uncertainty,
        )
    
    def run_manual_dream(self, duration: int = None) -> DreamCycleResult:
        """Manually trigger a dream cycle."""
        duration = duration or self.max_dream_duration
        old_interval = self.dream_cycle_interval
        self.dream_cycle_interval = 1
        self.steps_since_dream.fill_(self.dream_cycle_interval)
        result = self.run_dream_cycle()
        self.dream_cycle_interval = old_interval
        return result
    
    def get_stats(self) -> Dict:
        return {
            "dream_cycles": self.dream_cycle_count.item(),
            "total_dream_steps": self.total_dream_steps.item(),
            "retriever": self.retriever.get_stats(),
            "replay": self.experience_replay.get_stats(),
            "recombiner": self.recombiner.get_stats(),
            "simulator": self.simulator.get_stats(),
            "evaluator": self.evaluator.get_stats(),
            "consolidator": self.consolidator.get_stats(),
            "structural": self.structural_dreamer.get_stats(),
        }
    
    def force_consolidation(self) -> Dict:
        """Force consolidation of all pending fast weights."""
        return self.consolidator.consolidate_all()
    
    def get_memory_stats(self) -> Dict:
        return self.retriever.get_stats()


def create_dream_engine(
    network: 'RelationalNetwork',
    dream_cycle_interval: int = 1000,
    dream_duration: int = 100,
    **kwargs
) -> DreamEngine:
    """Factory for dream engine."""
    return DreamEngine(
        network=network,
        dream_cycle_interval=dream_cycle_interval,
        dream_duration=dream_duration,
        **kwargs
    )