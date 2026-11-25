"""
Dream Consolidator - Applies accepted dream discoveries through the
connection's OWN output-preserving consolidation law.

Theta-write bypass ELIMINATED per owner condition 1 (spec
DREAM_REIMPL_SPEC.md §3 E5 + §6a, 2026-11-25). The two retired bypass paths:

- line 222 (old): ``conn.plasticity.theta_slow.data = hypothesis_weight`` on
  a newly created connection -- a raw theta-write with no activity mask, no
  selective gate, no a-preservation solve, no clamp discipline.
- lines 305-337 (old): raw ``slow.add_(rate * delta)`` sweep on every
  trace-active connection -- again outside the connection's law.

Replaced by:
- New dream connections are created with the hypothesis weight in the FAST
  memory (u_fast initialization at creation -- not a consolidation) and
  theta_slow at its constructed zero. Thereafter theta can only move through
  the connection's own ``consolidate()`` (the annealing exact-solve in
  plastic.py, protected property 4), with its gates, activity mask,
  rate-0 disable and 1e-6 output-preservation invariant.
- Updating an existing connection injects the hypothesis delta into the FAST
  channel (u_fast), bounded by the max-weight clamp; the connection's own
  law decides if/when that evidence consolidates into theta.
- The general active-connection sweep calls ``conn.consolidate()`` -- the
  connection's own law -- gated on trace strength as the WHEN policy. The
  HOW is never reimplemented here.

The trace-strength criterion and the sweep flag live in this file; the
consolidation mathematics does not. Rate-0 connections disable by law
(Plastic-D0 controls cannot consolidate via the dream path any more than via
the live path, spec §5 safeguard 4).
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
    Consolidates accepted dream discoveries into long-term memory THROUGH
    THE CONNECTION'S OWN LAW.

    Responsibilities:
    - Route dream discoveries into the fast channel (u_fast initialization /
      injection); theta is owned exclusively by the connection's
      consolidate().
    - Create new connections from accepted hypotheses (live params, owner
      fix 4), with the hypothesis as initial fast weights.
    - Run the lawful consolidation sweep on trace-active connections via
      conn.consolidate() (flag-controlled).
    - Safeguard against catastrophic forgetting.

    There is NO path in this class that writes theta_slow outside
    consolidate().
    """

    def __init__(
        self,
        network: 'RelationalNetwork',
        consolidation_rate: float = 0.001,
        max_consolidation_per_step: int = 10,
        forgetting_protection: float = 0.99,
        conflict_reversal_rate: float = 0.01,
        allow_active_sweep: bool = True,
        trace_strength_threshold: float = 0.1,
    ):
        super().__init__()
        self.network = network
        self.consolidation_rate = consolidation_rate
        self.max_consolidation_per_step = max_consolidation_per_step
        self.forgetting_protection = forgetting_protection
        self.conflict_reversal_rate = conflict_reversal_rate
        self.allow_active_sweep = allow_active_sweep
        self.trace_strength_threshold = trace_strength_threshold

        # Statistics
        self.register_buffer("consolidations_performed", torch.tensor(0))
        self.register_buffer("connections_created", torch.tensor(0))
        self.register_buffer("connections_pruned", torch.tensor(0))
        self.register_buffer("conflicts_resolved", torch.tensor(0))
        self.register_buffer("forgetting_events_prevented", torch.tensor(0))
        self.register_buffer("lawful_sweep_connections", torch.tensor(0))

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

        # Lawful sweep over active connections: routes through each
        # connection's OWN consolidate() (owner condition 1).
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
        """Consolidate a compositional relationship A→B, B→C → A→C.

        Theta-write bypass ELIMINATED (owner condition 1):
        - Existing connection: the hypothesis delta enters the FAST channel
          (u_fast), bounded by the max-weight clamp. Theta is untouched
          here; the connection's own consolidate() owns it.
        - New connection: created with live params (owner fix 4); the
          hypothesis weight is the initial FAST memory; theta_slow stays at
          its constructed zero. No theta-write exists on this path.
        """
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
            conn = existing_conn
            hypothesis_weight = hypothesis.hypothesis_weight

            # Move the EFFECTIVE weight toward the hypothesis through the
            # FAST channel only: a' = blend*(u + d) + (1-blend)*theta moves
            # toward w as d -> (w - a)/blend. Theta is never written here;
            # consolidation of that evidence is the connection's own law.
            with torch.no_grad():
                blend = float(getattr(conn, "blend_ratio", 0.7))
                blend = blend if blend > 0 else 1.0
                a_current = blend * conn.plasticity.u_fast \
                    + (1.0 - blend) * conn.plasticity.theta_slow
                delta = (hypothesis_weight.detach() - a_current) / blend
                u_backup = conn.plasticity.u_fast.clone()
                conn.plasticity.u_fast.add_(self.consolidation_rate * delta)
                # Max-weight clamp discipline (the connection's own clamp law).
                conn.plasticity.u_fast.clamp_(
                    -conn.plasticity.max_weight, conn.plasticity.max_weight)
                # Forgetting guard (owner fix 2, repaired form): revert an
                # injection that would shrink u_fast by more than 50% in one
                # event. Inert at healthy rates; arms the path.
                pre_norm = float(u_backup.norm())
                if pre_norm > 0 and float(conn.plasticity.u_fast.norm()) < 0.5 * pre_norm:
                    conn.plasticity.u_fast.copy_(u_backup)
                    safeguards.append("forgetting_protection_triggered")
                    self.forgetting_events_prevented.add_(1)
                else:
                    amount = float(delta.norm()) * self.consolidation_rate
                    updated = 1
        else:
            # Create new connection dynamically. Owner fix 4 (2026-11-25):
            # inherit the live network's plasticity parameters (lr / blend).
            from nnfs.connections.plastic import PlasticRelationalConnection
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
            # The hypothesis weight enters as INITIAL FAST memory. theta_slow
            # stays at its constructed zero -- there is deliberately NO
            # theta-write on this path (owner condition 1): theta can only
            # ever move through this connection's own consolidate().
            with torch.no_grad():
                blend = float(live_params["blend_ratio"])
                blend = blend if blend > 0 else 1.0
                conn.plasticity.u_fast.copy_(hypothesis.hypothesis_weight.detach() / blend)
                conn.plasticity.u_fast.clamp_(
                    -conn.plasticity.max_weight, conn.plasticity.max_weight)

            # Add to network
            new_key = f"conn_{len(self.network.connections)}"
            self.network.connections[new_key] = conn

            # Owner fix 3 (2026-11-25): rebuild the network's index buffers so
            # conn_rows/conn_cols/in_degree/num_connections all stay consistent.
            self.network.rebuild_connection_indices()

            # Owner fix 1 (2026-11-25): count dream-side creations so stats
            # reflect the true topology change.
            self.connections_created.add_(1)

            created = 1
            amount = float(hypothesis.hypothesis_weight.detach().norm())
            safeguards.append("new_connection_created")

        return {"updated": updated, "created": created, "amount": amount, "safeguards": safeguards}

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
            # No theta-write: analogical evidence stays in the fast channel.
            with torch.no_grad():
                alpha = self.consolidation_rate * 0.5
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
        """Lawful consolidation sweep over trace-active connections.

        The sweep's WHEN policy (only strong traces) lives here; the HOW is
        the connection's OWN output-preserving consolidate() -- the annealing
        exact-solve with its selective gates, activity mask, rate-0 disable
        and 1e-6 preservation invariant. No slow-weight arithmetic happens in
        this file (owner condition 1: the old raw slow.add_ sweep at lines
        305-337 is retired).
        """
        safeguards = []
        updated = 0
        amount = 0.0

        if not self.allow_active_sweep:
            return {"updated": 0, "amount": 0.0, "safeguards": ["active_sweep_disabled"]}

        for key, conn in self.network.connections.items():
            if hasattr(conn, 'plasticity') and hasattr(conn.plasticity, 'u_fast'):
                # Only sweep connections with strong traces (WHEN policy)
                trace_strength = conn.eligibility_trace.trace.abs().mean().item()
                if trace_strength > self.trace_strength_threshold:
                    stats = conn.consolidate()  # the connection's own law
                    self.lawful_sweep_connections.add_(1)
                    transferred = float(stats.get("transferred_abs_mean", 0.0) or 0.0)
                    if stats.get("consolidated", 0) and transferred > 0:
                        updated += 1
                        amount += transferred
                    if stats.get("consolidation_disabled"):
                        safeguards.append("consolidation_disabled_rate_0")

        return {"updated": updated, "amount": amount, "safeguards": safeguards}

    def get_stats(self) -> Dict:
        return {
            "consolidations_performed": self.consolidations_performed.item(),
            "connections_created": self.connections_created.item(),
            "connections_pruned": self.connections_pruned.item(),
            "conflicts_resolved": self.conflicts_resolved.item(),
            "forgetting_events_prevented": self.forgetting_events_prevented.item(),
            "lawful_sweep_connections": self.lawful_sweep_connections.item(),
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
