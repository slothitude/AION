"""
Dream Evaluator - Accepts or rejects dreamed hypotheses on MEASURED evidence.

Reality-grounded per spec DREAM_REIMPL_SPEC.md §3 E1-E5 (owner-approved,
2026-11-25). The previous version summed hardcoded constants and omitted
utility from the sum entirely, so acceptance was guaranteed arithmetic.

Owner decision D4 (2026-11-25), binding:
  consistency 0.25 · novelty 0.15 · utility 0.25 · efficiency 0.10 ·
  non-forgetting 0.25   (sums to 1.0)
  epsilon = 0.0, BUT the improvement must be positive BEYOND measurement
  noise across repeated held-out measurements.

Acceptance law (E5):
  hallucination guard passed
  AND utility >= epsilon (0.0)
  AND measured improvement beyond the across-fold noise floor
  AND forgetting risk within bound
  AND (optional) stability within bound.
Only then does consolidation run -- through the connection's own
output-preserving consolidate(), never through a direct theta-write.

Dreams cannot validate themselves (README law): acceptance here is measured
improvement on stored replay experiences, never a constant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

from nnfs.dreaming.simulator import SimulationResult


@dataclass
class EvaluationResult:
    """Result of evaluating a measured hypothesis."""
    simulation_result: 'SimulationResult'
    overall_score: float
    acceptance: str  # "accept", "reject", "defer"
    consolidation_priority: float  # 0-1, priority for consolidation
    feedback: Dict[str, float]  # Detailed measured feedback
    recommended_action: str  # "consolidate", "discard", "test_more"
    guard_passed: bool = True
    guard_reason: str = ""
    beyond_noise: bool = False
    utility_epsilon: float = 0.0


class DreamEvaluator(nn.Module):
    """
    Evaluates measured hypotheses for consolidation.

    Scores are the simulator's MEASURED behaviour deltas (spec §3 E1):
    - consistency: measured degradation on non-source experiences
    - novelty: measured max cosine distance to existing weights
    - utility: measured prediction improvement on source experiences
    - efficiency: measured capacity cost
    - forgetting risk: measured error increase on consolidated connections
    """

    # Owner decision D4 weights (binding, 2026-11-25)
    OWNER_D4_WEIGHTS = {
        "consistency": 0.25,
        "novelty": 0.15,
        "utility": 0.25,
        "efficiency": 0.10,
        "forgetting": 0.25,
    }

    def __init__(
        self,
        network: 'RelationalNetwork',
        consistency_weight: float = 0.25,
        novelty_weight: float = 0.15,
        utility_weight: float = 0.25,
        efficiency_weight: float = 0.10,
        forgetting_weight: float = 0.25,
        utility_epsilon: float = 0.0,
        forgetting_risk_bound: float = 0.5,
        stability_bound: Optional[float] = None,
        acceptance_threshold: float = 0.6,  # retired from acceptance (E3);
                                            # kept only as a reporting bin
        defer_threshold: float = 0.3,
    ):
        super().__init__()
        self.network = network
        self.consistency_weight = consistency_weight
        self.novelty_weight = novelty_weight
        self.utility_weight = utility_weight
        self.efficiency_weight = efficiency_weight
        self.forgetting_weight = forgetting_weight
        self.utility_epsilon = utility_epsilon
        self.forgetting_risk_bound = forgetting_risk_bound
        self.stability_bound = stability_bound
        self.acceptance_threshold = acceptance_threshold
        self.defer_threshold = defer_threshold

        wsum = (consistency_weight + novelty_weight + utility_weight
                + efficiency_weight + forgetting_weight)
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(
                f"evaluator weights must sum to 1.0 (owner D4); got {wsum}")

        # Statistics
        self.register_buffer("evaluations_performed", torch.tensor(0))
        self.register_buffer("accepted_count", torch.tensor(0))
        self.register_buffer("rejected_count", torch.tensor(0))
        self.register_buffer("deferred_count", torch.tensor(0))
        self.register_buffer("modified_count", torch.tensor(0))
        self.register_buffer("unmeasured_count", torch.tensor(0))

    def evaluate(
        self,
        simulation_result: SimulationResult,
        network_state: Optional[Dict] = None,
        guard_passed: bool = True,
        guard_reason: str = "passed",
    ) -> EvaluationResult:
        """
        Evaluate a MEASURED simulation result for consolidation.

        Acceptance law (spec §3 E5, owner D4 epsilon=0.0 + beyond-noise):
        measured utility >= epsilon, improvement beyond the across-fold
        noise floor, forgetting risk within bound. Guard failure or
        unmeasured results are never accepted.
        """
        sim = simulation_result

        consistency = sim.consistency_score
        novelty = sim.novelty_score
        utility = sim.utility_score
        stability = sim.stability_estimate
        efficiency = sim.efficiency_score
        forgetting_risk = sim.forgetting_risk

        # Weighted overall score (E2): utility IS in the sum (owner D4).
        overall_score = (
            self.consistency_weight * consistency +
            self.novelty_weight * novelty +
            self.utility_weight * utility +
            self.efficiency_weight * efficiency +
            self.forgetting_weight * (1 - forgetting_risk)  # low forgetting = good
        )

        # --- Acceptance law (E5) ---
        beyond_noise = bool(
            sim.measured
            and sim.utility_delta_abs > sim.noise_floor
        )
        utility_gate = bool(sim.measured and utility >= self.utility_epsilon)
        forgetting_gate = bool(forgetting_risk <= self.forgetting_risk_bound)
        stability_gate = (
            self.stability_bound is None or stability >= self.stability_bound
        )

        if not sim.measured:
            acceptance = "defer"
            action = "test_more"
            self.unmeasured_count.add_(1)
        elif not guard_passed:
            acceptance = "reject"
            action = "discard"
        elif utility_gate and beyond_noise and forgetting_gate and stability_gate:
            acceptance = "accept"
            action = "consolidate"
        elif utility > 0 and forgetting_gate:
            # positive but not beyond noise: report bin only, nothing consolidates
            acceptance = "defer"
            action = "test_more"
        else:
            acceptance = "reject"
            action = "discard"

        # Consolidation priority from measured evidence
        consolidation_priority = max(0.0, min(1.0, overall_score)) * (1 - forgetting_risk)

        # Detailed measured feedback
        feedback = {
            "consistency": consistency,
            "novelty": novelty,
            "utility": utility,
            "utility_delta_abs": sim.utility_delta_abs,
            "stability": stability,
            "forgetting_risk": forgetting_risk,
            "efficiency": efficiency,
            "overall_score": overall_score,
            "noise_floor": sim.noise_floor,
            "install_mode": 0.0 if sim.install_mode == "existing_connection" else 1.0,
        }

        result = EvaluationResult(
            simulation_result=simulation_result,
            overall_score=overall_score,
            acceptance=acceptance,
            consolidation_priority=consolidation_priority,
            feedback=feedback,
            recommended_action=action,
            guard_passed=guard_passed,
            guard_reason=guard_reason,
            beyond_noise=beyond_noise,
            utility_epsilon=self.utility_epsilon,
        )

        # Update stats
        self.evaluations_performed.add_(1)
        if acceptance == "accept":
            self.accepted_count.add_(1)
        elif acceptance == "reject":
            self.rejected_count.add_(1)
        elif acceptance == "defer":
            self.deferred_count.add_(1)
        elif acceptance == "modify":
            self.modified_count.add_(1)

        return result

    def evaluate_batch(
        self,
        simulation_results: List[SimulationResult],
        network_state: Optional[Dict] = None,
        guard_verdicts: Optional[List[Tuple[bool, str]]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of simulation results, one guard verdict each."""
        results = []
        for i, sim in enumerate(simulation_results):
            if guard_verdicts is not None:
                passed, reason = guard_verdicts[i]
            else:
                passed, reason = True, "passed"
            results.append(self.evaluate(sim, network_state, passed, reason))
        return results

    def get_stats(self) -> Dict:
        total = self.evaluations_performed.item()
        if total == 0:
            return {"evaluations": 0}
        return {
            "evaluations": total,
            "accepted": self.accepted_count.item(),
            "rejected": self.rejected_count.item(),
            "deferred": self.deferred_count.item(),
            "modified": self.modified_count.item(),
            "unmeasured": self.unmeasured_count.item(),
            "acceptance_rate": self.accepted_count.item() / max(1, total),
        }


def create_dream_evaluator(
    network: 'RelationalNetwork',
    consistency_weight: float = 0.25,
    novelty_weight: float = 0.15,
    utility_weight: float = 0.25,
    efficiency_weight: float = 0.10,
    forgetting_weight: float = 0.25,
    utility_epsilon: float = 0.0,
) -> DreamEvaluator:
    """Factory for dream evaluator (owner D4 weights, epsilon 0.0)."""
    return DreamEvaluator(
        network=network,
        consistency_weight=consistency_weight,
        novelty_weight=novelty_weight,
        utility_weight=utility_weight,
        efficiency_weight=efficiency_weight,
        forgetting_weight=forgetting_weight,
        utility_epsilon=utility_epsilon,
    )
