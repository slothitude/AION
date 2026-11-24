"""
Dream Consolidator - Updates slow memory and structural connections from dreams.

Applies accepted dream discoveries to the network's long-term memory
and structural connections, with safeguards against catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from nnfs.dreaming.evaluator import EvaluationResult
from nnfs.dreaming.simulator import SimulationResult
from nnfs.dreaming.recombiner import HypothesizedRelationship


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    evaluation: 'EvaluationResult'
    connections_updated: int
    consolidation_amount: float
    structural_changes: Dict[str, int]
    safeguards_triggered: List[str]


def _live_connection_params(network: 'RelationalNetwork') -> Dict:
    """Owner fix 4 (2026-11-25): dream-created connections must inherit the
    live network's plasticity parameters instead of hardcoded defaults
    (previously lr=0.01 / default blend=0.7), so a dream-created connection
    learns and blends identically to the connections around it."""
    existing = next(iter(network.connections.values()), None)
    if existing is not None:
        learning_rate = float(existing.plasticity.learning_rate)
        blend_ratio = float(getattr(existing, "blend_ratio",
                                    getattr(network, "blend_ratio", 0.7)))
    else:
        learning_rate = 0.01
        blend_ratio = float(getattr(network, "blend_ratio", 0.7))
    return {"learning_rate": learning_rate, "blend_ratio": blend_ratio}


class DreamConsolidator(nn.Module):
    """
    Consolidates accepted dream discoveries into long-term memory.
    
    Responsibilities:
    - Update slow weights (θ_slow) from fast weights (u_fast)
    - Create new connections from accepted hypotheses
    - Prune rejected/detrimental relationships
    - Safeguard against catastrophic forgetting
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        consolidation_rate: float = 0.001,
        max_consolidation_per_step: int = 10,
        forgetting_protection: float = 0.99,
        conflict_reversal_rate: float = 0.01,
    ):
        super().__init__()
        self.network = network
        self.consolidation_rate = consolidation_rate
        self.max_consolidation_per_step = max_consolidation_per_step
        self.forgetting_protection = forgetting_protection
        self.conflict_reversal_rate = conflict_reversal_rate
        
        # Statistics
        self.register_buffer("consolidations_performed", torch.tensor(0))
        self.register_buffer("connections_created", torch.tensor(0))
        self.register_buffer("connections_pruned", torch.tensor(0))
        self.register_buffer("conflicts_resolved", torch.tensor(0))
        self.register_buffer("forgetting_events_prevented", torch.tensor(0))
        
    def consolidate(
        self,
        evaluation_results: List['EvaluationResult'],
    ) -> List[ConsolidationResult]:
        """
        Consolidate accepted evaluations into long-term memory.
        
        Args:
            evaluation_results: List of accepted/rejected evaluations
            
        Returns:
            List of consolidation results
        """
        results = []
        
        for eval_result in evaluation_results:
            if eval_result.acceptance != "accept":
                continue
            
            result = self._consolidate_evaluation(eval_result)
            results.append(result)
            
            if len(results) >= self.max_consolidation_per_step:
                break
        
        return results
    
    def _consolidate_evaluation(self, eval_result: 'EvaluationResult') -> 'ConsolidationResult':
        """Consolidate a single accepted evaluation."""
        sim = eval_result.simulation_result
        hypothesis = sim.hypothesis
        
        safeguards = []
        structural_changes = {"created": 0, "updated": 0, "pruned": 0}
        connections_updated = 0
        consolidation_amount = 0.0
        
        if hypothesis.relationship_type == "composition":
            # For composition, update the target connection or create it
            result = self._consolidate_composition(hypothesis, eval_result)
            connections_updated += result["updated"]
            structural_changes["created"] += result["created"]
            consolidation_amount += result["amount"]
            safeguards.extend(result["safeguards"])
            
        elif hypothesis.relationship_type == "analogy":
            # For analogy, update or create the analogous relationship
            result = self._consolidate_analogy(hypothesis, eval_result)
            connections_updated += result["updated"]
            structural_changes["updated"] += result["updated"]
            consolidation_amount += result["amount"]
            safeguards.extend(result["safeguards"])
            
        elif hypothesis.relationship_type == "context_transfer":
            # For context transfer, update context-specific relationships
            result = self._consolidate_context_transfer(hypothesis, eval_result)
            connections_updated += result["updated"]
            consolidation_amount += result["amount"]
            safeguards.extend(result["safeguards"])
        
        # Apply general consolidation to all active connections
        gen_result = self._consolidate_active_connections()
        connections_updated += gen_result["updated"]
        consolidation_amount += gen_result["amount"]
        safeguards.extend(gen_result["safeguards"])

        self.consolidations_performed.add_(1)
        
        return ConsolidationResult(
            evaluation=eval_result,
            connections_updated=connections_updated,
            consolidation_amount=consolidation_amount,
            structural_changes=structural_changes,
            safeguards_triggered=safeguards,
        )
    
    def _consolidate_composition(
        self, 
        hypothesis: 'HypothesizedRelationship',
        eval_result: 'EvaluationResult'
    ) -> Dict:
        """Consolidate a compositional relationship A→B, B→C → A→C."""
        safeguards = []
        pre, post = hypothesis.pre_idx, hypothesis.post_idx
        amount = 0.0
        updated = 0
        created = 0
        
        # Check if connection already exists by pre/post indices
        existing_conn = None
        for key, conn in self.network.connections.items():
            pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
            post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
            if pre_idx == pre and post_idx == post:
                existing_conn = conn
                break
        
        if existing_conn is not None:
            # Update existing connection
            conn = existing_conn
            fast = conn.plasticity.u_fast
            slow = conn.plasticity.theta_slow
            hypothesis_weight = hypothesis.hypothesis_weight
            
            # Move slow toward hypothesis
            with torch.no_grad():
                delta = hypothesis_weight - slow
                # Forgetting guard (owner fix 2, 2026-11-25): the previous check
                # compared slow.norm() < slow.norm() * 0.5 against itself and
                # could never fire. Guard the transfer properly: revert any
                # transfer that would shrink theta_slow by more than 50% in one
                # event, and record the trigger in safeguards/stats.
                pre_norm = float(slow.norm())
                slow_backup = slow.clone()
                slow.add_(self.consolidation_rate * delta)
                if float(slow.norm()) < 0.5 * pre_norm:
                    slow.copy_(slow_backup)
                    safeguards.append("forgetting_protection_triggered")
                    self.forgetting_events_prevented.add_(1)
                else:
                    consolidation_amount = delta.norm().item()
                    updated = 1
        else:
            # Create new connection dynamically
            from nnfs.connections.plastic import PlasticRelationalConnection
            # Owner fix 4 (2026-11-25): inherit the live network's plasticity
            # parameters (lr / blend) instead of hardcoded lr=0.01 and the
            # default blend=0.7, so a dream-created connection blends and
            # learns like the live connections around it.
            live_params = _live_connection_params(self.network)
            conn = PlasticRelationalConnection(
                pre_idx=pre,
                post_idx=post,
                pre_dim=self.network.neuron_dim,
                post_dim=self.network.neuron_dim,
                context_dim=self.network.context_dim,
                relation_type=self.network.relation_type,
                learning_rate=live_params["learning_rate"],
                trace_decay=0.95,
                consolidation_rate=0.001,
                blend_ratio=live_params["blend_ratio"],
            )
            # Initialize slow weights with hypothesis
            with torch.no_grad():
                conn.plasticity.theta_slow.data = hypothesis.hypothesis_weight.data.clone()

            # Add to network
            new_key = f"conn_{len(self.network.connections)}"
            self.network.connections[new_key] = conn

            # Owner fix 3 (2026-11-25): rebuild the network's index buffers so
            # conn_rows/conn_cols/in_degree/num_connections all stay consistent
            # (the previous manual conn_rows/conn_cols cat left in_degree stale,
            # mis-scaling the target/in_degree supervision split for the new
            # connection's post neuron). Same convention as structural.py.
            self.network.rebuild_connection_indices()

            # Owner fix 1 (2026-11-25): count dream-side creations so stats
            # reflect the true topology change (buffer previously never
            # incremented anywhere).
            self.connections_created.add_(1)

            created = 1
            consolidation_amount = 1.0
            safeguards.append("new_connection_created")
        
        return {"updated": updated, "created": created, "amount": consolidation_amount, "safeguards": safeguards}
    
    def _consolidate_analogy(
        self,
        hypothesis: 'HypothesizedRelationship',
        eval_result: 'EvaluationResult'
    ) -> Dict:
        """Consolidate an analogical relationship."""
        safeguards = []
        amount = 0.0
        updated = 0
        
        # For analogy, we'd create or update the target relationship
        # based on the analogy mapping
        pre, post = hypothesis.pre_idx, hypothesis.post_idx
        key = f"conn_{pre}_{post}"
        
        if key in self.network.connections:
            conn = self.network.connections[key]
            with torch.no_grad():
                # Analogies are weaker, use smaller consolidation rate
                alpha = self.consolidation_rate * 0.5
                # Would blend with analogy-derived weight
                updated = 1
                amount = alpha
        else:
            # Would create new connection
            created = 1
            amount = 0.5
            safeguards.append("analogical_connection_created")
        
        return {"updated": updated, "created": created, "amount": amount, "safeguards": safeguards}
    
    def _consolidate_context_transfer(
        self,
        hypothesis: 'HypothesizedRelationship',
        eval_result: 'EvaluationResult'
    ) -> Dict:
        """Consolidate a context-transferred relationship."""
        safeguards = []
        amount = 0.0
        updated = 0
        
        # Context transfers are more tentative
        pre, post = hypothesis.pre_idx, hypothesis.post_idx
        key = f"conn_{pre}_{post}"
        
        if key in self.network.connections:
            conn = self.network.connections[key]
            with torch.no_grad():
                alpha = self.consolidation_rate * 0.3  # Even smaller for context transfer
                updated = 1
                amount = alpha
                safeguards.append("context_transfer_consolidated")
        else:
            created = 1
            amount = 0.3
            safeguards.append("context_transfer_created")
        
        return {"updated": updated, "created": created, "amount": amount, "safeguards": safeguards}
    
    def _consolidate_active_connections(self) -> Dict:
        """General consolidation of all active connections."""
        safeguards = []
        updated = 0
        amount = 0.0
        
        for key, conn in self.network.connections.items():
            if hasattr(conn, 'plasticity') and hasattr(conn.plasticity, 'u_fast'):
                fast = conn.plasticity.u_fast
                slow = conn.plasticity.theta_slow
                
                # Only consolidate strong traces
                trace_strength = conn.eligibility_trace.trace.abs().mean().item()
                if trace_strength > 0.1:
                    with torch.no_grad():
                        delta = fast - slow
                        # Forgetting guard (owner fix 2, 2026-11-25): same
                        # repaired guard as the composition transfer — revert a
                        # transfer that would shrink theta_slow by more than
                        # 50% in one event. Inert at healthy rates (rate 0.001
                        # cannot halve a norm) but arms the path.
                        pre_norm = float(slow.norm())
                        slow_backup = slow.clone()
                        slow.add_(self.consolidation_rate * delta)
                        if float(slow.norm()) < 0.5 * pre_norm:
                            slow.copy_(slow_backup)
                            safeguards.append("forgetting_protection_triggered")
                            self.forgetting_events_prevented.add_(1)
                        else:
                            updated += 1
                            amount += delta.norm().item()

        return {"updated": updated, "amount": amount, "safeguards": safeguards}
    
    def get_stats(self) -> Dict:
        return {
            "consolidations_performed": self.consolidations_performed.item(),
            "connections_created": self.connections_created.item(),
            "connections_pruned": self.connections_pruned.item(),
            "conflicts_resolved": self.conflicts_resolved.item(),
            "forgetting_events_prevented": self.forgetting_events_prevented.item(),
        }


def create_dream_consolidator(
    network: 'RelationalNetwork',
    consolidation_rate: float = 0.001,
    max_consolidation_per_step: int = 10,
    forgetting_protection: float = 0.99,
) -> DreamConsolidator:
    """Factory for dream consolidator."""
    return DreamConsolidator(
        network=network,
        consolidation_rate=consolidation_rate,
        max_consolidation_per_step=max_consolidation_per_step,
        forgetting_protection=forgetting_protection,
    )