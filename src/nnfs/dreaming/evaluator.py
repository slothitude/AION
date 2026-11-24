"""
Dream Evaluator - Scores dreamed experiences for usefulness, consistency, and forgetting risk.

Evaluates dreamed experiences to decide which should be consolidated
into long-term memory and which should be discarded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from nnfs.dreaming.simulator import SimulationResult


@dataclass
class EvaluationResult:
    """Result of evaluating a dreamed experience."""
    simulation_result: 'SimulationResult'
    overall_score: float
    acceptance: str  # "accept", "reject", "defer", "modify"
    consolidation_priority: float  # 0-1, priority for consolidation
    feedback: Dict[str, float]  # Detailed feedback scores
    recommended_action: str  # "consolidate", "discard", "modify", "test_more"


class DreamEvaluator(nn.Module):
    """
    Evaluates dreamed experiences for consolidation.
    
    Scores dreamed experiences on:
    - Predictive accuracy: does it improve predictions?
    - Consistency: does it conflict with existing knowledge?
    - Novelty: does it add genuinely new information?
    - Efficiency: does it simplify or complicate the network?
    - Forgetting risk: does it overwrite important memories?
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        consistency_weight: float = 0.3,
        novelty_weight: float = 0.2,
        efficiency_weight: float = 0.2,
        forgetting_weight: float = 0.3,
        acceptance_threshold: float = 0.6,
        defer_threshold: float = 0.3,
    ):
        super().__init__()
        self.network = network
        self.consistency_weight = consistency_weight
        self.novelty_weight = novelty_weight
        self.efficiency_weight = efficiency_weight
        self.forgetting_weight = forgetting_weight
        self.acceptance_threshold = acceptance_threshold
        self.defer_threshold = defer_threshold
        
        # Statistics
        self.register_buffer("evaluations_performed", torch.tensor(0))
        self.register_buffer("accepted_count", torch.tensor(0))
        self.register_buffer("rejected_count", torch.tensor(0))
        self.register_buffer("deferred_count", torch.tensor(0))
        self.register_buffer("modified_count", torch.tensor(0))
        
    def evaluate(
        self,
        simulation_result: SimulationResult,
        network_state: Optional[Dict] = None,
    ) -> EvaluationResult:
        """
        Evaluate a simulation result for consolidation.
        
        Returns an EvaluationResult with acceptance decision and priority.
        """
        sim = simulation_result
        
        # Extract scores
        consistency = sim.consistency_score
        novelty = sim.novelty_score
        utility = sim.utility_score
        stability = sim.stability_estimate
        
        # Forgetting risk: does this overwrite important memories?
        forgetting_risk = self._estimate_forgetting_risk(simulation_result)
        
        # Efficiency: does this simplify or complicate?
        efficiency = self._evaluate_efficiency(simulation_result)
        
        # Weighted overall score
        overall_score = (
            self.consistency_weight * consistency +
            self.novelty_weight * novelty +
            self.efficiency_weight * efficiency +
            self.forgetting_weight * (1 - forgetting_risk)  # Low forgetting = good
        )
        
        # Determine acceptance
        if overall_score >= self.acceptance_threshold:
            acceptance = "accept"
            action = "consolidate"
        elif overall_score >= 0.3:  # Defer threshold
            acceptance = "defer"
            action = "test_more"
        else:
            acceptance = "reject"
            action = "discard"
        
        # Consolidation priority
        consolidation_priority = overall_score * (1 - forgetting_risk)
        
        # Detailed feedback
        feedback = {
            "consistency": consistency,
            "novelty": novelty,
            "utility": sim.utility_score,
            "stability": stability,
            "forgetting_risk": forgetting_risk,
            "efficiency": efficiency,
            "overall_score": overall_score,
        }
        
        result = EvaluationResult(
            simulation_result=simulation_result,
            overall_score=overall_score,
            acceptance=acceptance,
            consolidation_priority=consolidation_priority,
            feedback=feedback,
            recommended_action=action,
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
    
    def _estimate_forgetting_risk(self, simulation_result: SimulationResult) -> float:
        """Estimate risk of overwriting important memories."""
        # Check if this hypothesis conflicts with strongly consolidated relationships
        # For now, return low risk for high-consistency hypotheses
        if simulation_result.consistency_score > 0.7:
            return 0.1
        elif simulation_result.consistency_score > 0.4:
            return 0.3
        return 0.5
    
    def _evaluate_efficiency(self, simulation_result: SimulationResult) -> float:
        """Evaluate whether this simplifies or complicates the network."""
        # Composition relationships tend to be efficient (compress multiple relations)
        if simulation_result.hypothesis.relationship_type == "composition":
            return 0.8
        elif simulation_result.hypothesis.relationship_type == "analogy":
            return 0.6
        return 0.5
    
    def evaluate_batch(
        self,
        simulation_results: List[SimulationResult],
        network_state: Optional[Dict] = None,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of simulation results."""
        return [self.evaluate(sim, network_state) for sim in simulation_results]
    
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
            "acceptance_rate": self.accepted_count.item() / max(1, total),
        }


def create_dream_evaluator(
    network: 'RelationalNetwork',
    consistency_weight: float = 0.3,
    novelty_weight: float = 0.2,
    efficiency_weight: float = 0.2,
    forgetting_weight: float = 0.3,
    acceptance_threshold: float = 0.6,
) -> DreamEvaluator:
    """Factory for dream evaluator."""
    return DreamEvaluator(
        network=network,
        consistency_weight=consistency_weight,
        novelty_weight=novelty_weight,
        efficiency_weight=efficiency_weight,
        forgetting_weight=forgetting_weight,
        acceptance_threshold=acceptance_threshold,
    )