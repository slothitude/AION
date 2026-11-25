"""
Dream Engine - Orchestrates all dreaming subsystems.

The Dream Engine coordinates the full offline learning cycle:
1. Memory Retrieval: Select important experiences (fixed-shape windows, D2)
2. Experience Replay: Rehearse actual observations through the real law (R2)
3. Hallucination Guard: reject structurally implausible compositions (E4)
4. Pattern Recombination: Generate hypothetical relationships
5. Dream Simulation: MEASURE hypotheses on a copy of the network (E1, D3)
6. Dream Evaluation: Accept on measured evidence (E2/E3/E5, owner D4)
7. Consolidation: Route through the connection's own consolidate() (cond. 1)
8. Structural Dreaming: Experiment with network topology

D1-only validation (owner D6): with enable_recombination=False the engine's
intervention is replay alone -- bounded (R4), provenance-tagged (R5), and
state-delta instrumented (V1/V4 of docs/D1_REPLAY_VALIDATION_PLAN.md).
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
from nnfs.dreaming.guard import ComposedWeightNormGuard, GuardVerdict, calibrate_guard
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

    # Guard (E4): structurally implausible hypotheses, counted in their own
    # bucket so the acceptance rate means something (E6)
    hypotheses_guard_rejected: int
    guard_rejections: List[Dict]

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

    # --- D1 validation instrumentation (docs/D1_REPLAY_VALIDATION_PLAN.md) ---
    replay_wall_clock_s: float = 0.0          # V3
    replay_budget: int = 0                    # V3
    replay_steps_executed: int = 0            # V3 (counted in total_dream_steps)
    state_delta_u_fast: float = 0.0           # V1: ||Delta u|| across the cycle
    state_delta_theta_slow: float = 0.0       # V1: ||Delta theta|| across the cycle
    replay_state_delta_u_fast: float = 0.0    # V1: ||Delta u|| across replay phase
    replay_state_delta_theta_slow: float = 0.0
    provenance_count: int = 0                 # V2: replayed=true decision records


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
        replay_budget: int = 32,  # R4: experiences replayed per cycle
        learning_rate_scale: float = 0.1,  # R2: replay at 0.1x live rate
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
        self.replay_budget = replay_budget
        self.learning_rate_scale = learning_rate_scale

        # Initialize subsystems (D2: fixed-shape window ring sized to the net)
        self.retriever = create_retriever(
            max_memory=10000,
            context_dim=network.context_dim,
            num_neurons=network.num_neurons,
            neuron_dim=network.neuron_dim,
        )

        self.experience_replay = ExperienceReplay(
            network=network,
            retriever=self.retriever,
            replay_batch_size=16,
            replay_steps=1,
            learning_rate_scale=learning_rate_scale,
            replay_budget=replay_budget,
        )

        # E4 hallucination guard: cheap first filter, runs BEFORE measurement
        self.guard = ComposedWeightNormGuard(network)

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

        # Owner D4 weights (binding): consistency 0.25, novelty 0.15,
        # utility 0.25, efficiency 0.10, non-forgetting 0.25. epsilon = 0.0
        # with the beyond-noise requirement.
        self.evaluator = DreamEvaluator(
            network=network,
            consistency_weight=0.25,
            novelty_weight=0.15,
            utility_weight=0.25,
            efficiency_weight=0.10,
            forgetting_weight=0.25,
            utility_epsilon=0.0,
            forgetting_risk_bound=0.5,
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

        # Per-cycle evidence log (D1 validation harness harvests this)
        self.cycle_log: List[Dict] = []

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

    @staticmethod
    def _state_norms(network) -> Dict[str, float]:
        u_norm = 0.0
        theta_norm = 0.0
        for conn in network.connections.values():
            u_norm += float(conn.plasticity.u_fast.detach().norm())
            theta_norm += float(conn.plasticity.theta_slow.detach().norm())
        return {"u_fast_norm": u_norm, "theta_slow_norm": theta_norm}

    @staticmethod
    def _state_snapshot(network) -> Dict[str, Dict[str, torch.Tensor]]:
        snap = {}
        for key, conn in network.connections.items():
            snap[key] = {
                "u_fast": conn.plasticity.u_fast.detach().clone(),
                "theta_slow": conn.plasticity.theta_slow.detach().clone(),
            }
        return snap

    @staticmethod
    def _delta_norms(network, before: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        du = 0.0
        dtheta = 0.0
        for key, conn in network.connections.items():
            if key in before:
                du += float((conn.plasticity.u_fast.detach() - before[key]["u_fast"]).norm())
                dtheta += float((conn.plasticity.theta_slow.detach() - before[key]["theta_slow"]).norm())
        return {"u_fast_delta_norm": du, "theta_slow_delta_norm": dtheta}

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
            hypotheses_guard_rejected=0,
            guard_rejections=[],
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

        # Network state before the cycle (V1/V4 instrumentation)
        before_state = self._state_snapshot(self.network)

        # Phase 1: Memory Retrieval
        # A ranked pool twice the budget: the top half is the replay set, the
        # bottom half is the held-out set the measurement scores consistency
        # on (experiences the replay lane did not rehearse this cycle).
        ranked_pool = self.retriever.retrieve(
            num_experiences=2 * self.replay_budget,
            min_score=0.1,
        )
        replay_set = ranked_pool[:self.replay_budget]
        heldout_set = ranked_pool[self.replay_budget:]
        cycle_result.experiences_retrieved = len(ranked_pool)
        cycle_result.retrieval_stats = self.retriever.get_stats()

        # Phase 2: Experience Replay (R2-R5: real, bounded, provenance-tagged)
        if self.enable_replay and replay_set:
            replay_start = time.time()
            replay_stats = self.experience_replay.replay_episodic(
                num_experiences=min(self.replay_budget, len(replay_set)),
                cycle_id=cycle_id,
            )
            cycle_result.replay_wall_clock_s = time.time() - replay_start
            cycle_result.experiences_replayed = replay_stats.get("replayed", 0)
            cycle_result.replay_stats = replay_stats
            # R4: every replayed step is counted in total_dream_steps.
            replayed_steps = replay_stats.get("total_dream_steps", 0)
            self.total_dream_steps.add_(replayed_steps)
            cycle_result.replay_budget = self.replay_budget
            cycle_result.replay_steps_executed = replayed_steps
            cycle_result.replay_state_delta_u_fast = replay_stats.get("u_fast_delta_norm", 0.0)
            cycle_result.replay_state_delta_theta_slow = replay_stats.get("theta_slow_delta_norm", 0.0)
            cycle_result.provenance_count = replay_stats.get("provenance_count", 0)

        # Phase 3: Hallucination guard (E4) -- cheap filter BEFORE measurement
        guarded_hypotheses: List[HypothesizedRelationship] = []
        guard_verdicts: List[GuardVerdict] = []
        sim_results: List[SimulationResult] = []
        if self.enable_recombination:
            hypotheses = self.recombiner.generate_hypotheses(
                max_hypotheses=20,
            )
            cycle_result.hypotheses_generated = len(hypotheses)
            cycle_result.hypotheses_tested = len(hypotheses)

            for hyp in hypotheses:
                verdict = self.guard.check(hyp)
                if verdict.accepted:
                    guarded_hypotheses.append(hyp)
                    guard_verdicts.append(verdict)
                else:
                    # E6: guard rejections are counted in their own bucket so
                    # the acceptance rate means something.
                    cycle_result.hypotheses_guard_rejected += 1
                    cycle_result.guard_rejections.append({
                        "pre_idx": hyp.pre_idx,
                        "post_idx": hyp.post_idx,
                        "reason": verdict.reason,
                        **verdict.as_dict(),
                    })

            # Phase 4: Dream Simulation -- MEASURED on stored replay
            # experiences (E1), hypothesis installed on a COPY (owner D3).
            if guarded_hypotheses:
                sim_results = self.simulator.simulate_batch(
                    hypotheses=guarded_hypotheses,
                    initial_state=self.network.neuron_states.detach().clone(),
                    experiences_utility=replay_set,
                    experiences_consistency=heldout_set,
                )
                cycle_result.simulations_run = len(sim_results)

                # Recommendation bins are reporting-only (E3).
                for result in sim_results:
                    if result.recommendation == "accept":
                        cycle_result.hypotheses_accepted += 1
                    elif result.recommendation == "reject":
                        cycle_result.hypotheses_rejected += 1
                    else:
                        cycle_result.hypotheses_deferred += 1

        # Phase 5: Dream Evaluation (measured acceptance law, owner D4)
        evaluations: List[EvaluationResult] = []
        accept_evals: List[EvaluationResult] = []
        if self.enable_consolidation and sim_results:
            for i, sim in enumerate(sim_results):
                verdict = guard_verdicts[i] if i < len(guard_verdicts) else None
                eval_result = self.evaluator.evaluate(
                    sim,
                    guard_passed=verdict.accepted if verdict else True,
                    guard_reason=verdict.reason if verdict else "passed",
                )
                evaluations.append(eval_result)

            cycle_result.evaluations_performed = len(evaluations)
            for eval_result in evaluations:
                if eval_result.acceptance == "accept":
                    cycle_result.accepted += 1
                elif eval_result.acceptance == "reject":
                    cycle_result.rejected += 1
                else:
                    cycle_result.deferred += 1
            accept_evals = [e for e in evaluations if e.acceptance == "accept"]

        # Phase 6: Consolidation (routed through the connection's own law)
        if self.enable_consolidation and evaluations:
            if accept_evals:
                cons_results = self.consolidator.consolidate(accept_evals)
                cycle_result.consolidations_performed = len(cons_results)
                for cr in cons_results:
                    cycle_result.connections_updated += cr.connections_updated
                    cycle_result.safeguards_triggered.extend(cr.safeguards_triggered)
                    # Owner fix 1 (2026-11-25): consolidation-side creations and
                    # prunes count toward the cycle's structural totals.
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
        cycle_result.total_duration = time.time() - start_time
        cycle_result.memory_stats = self.retriever.get_stats()
        cycle_result.safeguards_triggered = list(set(cycle_result.safeguards_triggered))

        # Cycle-wide state deltas (V1: actual network updates per cycle)
        deltas = self._delta_norms(self.network, before_state)
        cycle_result.state_delta_u_fast = deltas["u_fast_delta_norm"]
        cycle_result.state_delta_theta_slow = deltas["theta_slow_delta_norm"]

        # Per-cycle evidence log (validation harness)
        self.cycle_log.append({
            "cycle_id": cycle_id,
            "wall_clock_s": cycle_result.total_duration,
            "replay_wall_clock_s": cycle_result.replay_wall_clock_s,
            "retrieved": cycle_result.experiences_retrieved,
            "replayed": cycle_result.experiences_replayed,
            "replay_budget": cycle_result.replay_budget,
            "replay_steps_executed": cycle_result.replay_steps_executed,
            "state_delta_u_fast": cycle_result.state_delta_u_fast,
            "state_delta_theta_slow": cycle_result.state_delta_theta_slow,
            "replay_state_delta_u_fast": cycle_result.replay_state_delta_u_fast,
            "replay_state_delta_theta_slow": cycle_result.replay_state_delta_theta_slow,
            "provenance_count": cycle_result.provenance_count,
            "hypotheses_generated": cycle_result.hypotheses_generated,
            "hypotheses_guard_rejected": cycle_result.hypotheses_guard_rejected,
            "hypotheses_rejected": cycle_result.rejected,
            "hypotheses_deferred": cycle_result.deferred,
            "measured_evaluations": sum(
                1 for e in evaluations if e.simulation_result.measured),
            "accepted": cycle_result.accepted,
            # falsifiable-prediction evidence (spec §4): every acceptance must
            # carry measured, beyond-noise evidence -- the pilot's 60/60
            # constant-table collapse structurally cannot recur when this is
            # True alongside accepted==len(accept_evals).
            "accepted_all_measured": (
                all(e.simulation_result.measured and e.beyond_noise
                    for e in accept_evals)
                if accept_evals else True
            ),
            "consolidations_performed": cycle_result.consolidations_performed,
            "connections_updated": cycle_result.connections_updated,
            "connections_created": cycle_result.connections_created,
        })

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
        """Store experience as a fixed-shape window for future dreaming (D2)."""
        self.retriever.store(
            input_signal=input_signal,
            target_next=target_next,
            context=context,
            significance=significance,
            uncertainty=uncertainty,
            context_id=context_id,
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
            "guard": {
                "max_product_ratio": self.guard.max_product_ratio,
                "max_scale_ratio": self.guard.max_scale_ratio,
            },
            "recombiner": self.recombiner.get_stats(),
            "simulator": self.simulator.get_stats(),
            "evaluator": self.evaluator.get_stats(),
            "consolidator": self.consolidator.get_stats(),
            "structural": self.structural_dreamer.get_stats(),
        }

    def force_consolidation(self) -> Dict:
        """Force the lawful consolidation sweep (through each connection's
        own consolidate())."""
        return self.consolidator._consolidate_active_connections()

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
