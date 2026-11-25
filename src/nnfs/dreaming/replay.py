"""
Experience Replay - Rehearses actual observations with temporal fidelity.

Real implementation per spec DREAM_REIMPL_SPEC.md §2 R1-R6 (owner-approved,
2026-11-25). The previous version was a stub: retrieval with placeholder
statistics and zero network passes, which made D1 bit-identical to D0.

Now (R2): for each retrieved experience the fixed-shape window is
reconstructed at its stored shape and driven through the network's REAL
learning law -- ``network.step(input, target_next=target, learn=True)`` --
with the live learning rate scaled by ``learning_rate_scale`` (default 0.1,
i.e. effective per-step eta = 0.005 against live 0.05).

(R3) Replay invents no consolidation of its own: it drives the live law, and
any periodic consolidation fires through the connection's own
output-preserving ``consolidate()`` (rate-0 connections disable by law).

(R4) Budget: one cycle replays ``min(budget, retrieved)`` experiences,
1 step each, every step counted in ``total_dream_steps`` by the engine.

(R5) Provenance: every replayed step is tagged ``replayed=true`` with
(cycle_id, step, effective_lr) in decision records, distinct from
live-observation provenance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import random

from nnfs.dreaming.retriever import MemoryRetriever, RetrievedExperience


@dataclass
class ReplayResult:
    """Result of one replayed experience."""
    experience: 'RetrievedExperience'
    prediction_mse: float
    weight_changes: Dict[str, float]
    trace_stats: Dict[str, float]
    consolidation_triggered: bool
    provenance: Dict[str, object] = field(default_factory=dict)


class ExperienceReplay(nn.Module):
    """
    Replays stored fixed-shape experience windows through the network's real
    learning law.

    Key properties:
    - Temporal fidelity: replays windows at their stored (num_neurons,
      neuron_dim) shape -- no flattening, no averaging (storage rebuild, D2).
    - Real learning law: ``network.step(..., learn=True)`` at the live rate
      scaled by ``learning_rate_scale``; consolidation only via the
      connection's own ``consolidate()``.
    - Bounded budget: ``min(replay_budget, retrieved)`` experiences, 1 step
      each, all counted.
    - Provenance: every replayed step tagged replayed=true with
      (cycle_id, step, effective_lr) decision records.
    """

    def __init__(
        self,
        network: 'RelationalNetwork',
        retriever: MemoryRetriever,
        replay_batch_size: int = 16,
        replay_steps: int = 1,
        learning_rate_scale: float = 0.1,  # Lower LR for replay (R2: 0.1x live)
        interleave_prob: float = 0.1,  # Probability of interleaving with current
        replay_budget: int = 32,  # R4: max experiences replayed per cycle
        max_decision_records: int = 100000,
    ):
        super().__init__()
        self.network = network
        self.retriever = retriever
        self.replay_batch_size = replay_batch_size
        self.replay_steps = replay_steps
        self.learning_rate_scale = learning_rate_scale
        self.interleave_prob = interleave_prob
        self.replay_budget = replay_budget

        # (R5) provenance decision records -- distinct from live observation.
        self.decision_records: deque = deque(maxlen=max_decision_records)

        # Track replay statistics
        self.register_buffer("total_replays", torch.tensor(0))
        self.register_buffer("total_consolidations", torch.tensor(0))
        self.register_buffer("replay_mse_sum", torch.tensor(0.0))
        self.register_buffer("total_steps_executed", torch.tensor(0))

    # ------------------------------------------------------------------
    # Learning-rate scaling (R2): live law at scaled rate, restored after.
    # ------------------------------------------------------------------
    def _scaled_rate_step(self, input_window, target_window, learn: bool):
        """One network step at the live rate scaled by learning_rate_scale.

        The live per-connection rate is scaled for the duration of the step
        and restored in a finally block, so the live path is untouched.
        Returns (step_result, effective_lr).
        """
        conns = list(self.network.connections.values())
        live_rates = [float(c.plasticity.learning_rate) for c in conns]
        effective_lr = live_rates[0] * self.learning_rate_scale if live_rates else 0.0
        try:
            for c, lr in zip(conns, live_rates):
                c.plasticity.learning_rate = lr * self.learning_rate_scale
            result = self.network.step(
                input_window, target_next=target_window, learn=learn)
        finally:
            for c, lr in zip(conns, live_rates):
                c.plasticity.learning_rate = lr
        return result, effective_lr

    @staticmethod
    def _state_snapshot(network) -> Dict[str, torch.Tensor]:
        snap = {}
        for key, conn in network.connections.items():
            snap[key] = {
                "u_fast": conn.plasticity.u_fast.detach().clone(),
                "theta_slow": conn.plasticity.theta_slow.detach().clone(),
            }
        return snap

    @staticmethod
    def _state_delta_norms(network, before: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        du = 0.0
        dtheta = 0.0
        for key, conn in network.connections.items():
            if key in before:
                du += float((conn.plasticity.u_fast.detach() - before[key]["u_fast"]).norm())
                dtheta += float((conn.plasticity.theta_slow.detach() - before[key]["theta_slow"]).norm())
        return {"u_fast_delta_norm": du, "theta_slow_delta_norm": dtheta}

    # ------------------------------------------------------------------
    # Replay (R2/R4)
    # ------------------------------------------------------------------
    def replay_batch(
        self,
        experiences: List['RetrievedExperience'],
        learn: bool = True,
        cycle_id: Optional[int] = None,
    ) -> List[ReplayResult]:
        """
        Replay a batch of experiences through the network's real learning law.

        Each experience's fixed-shape window is reconstructed at its stored
        shape and driven through ``network.step(learn=...)`` at the scaled
        rate (R2). Every executed step is tagged replayed=true with
        (cycle_id, step, effective_lr) provenance (R5).
        """
        results = []

        for step_i, exp in enumerate(experiences):
            # R2: reconstruct (input, target) at the stored fixed shape.
            input_window = exp.input_signal
            target_window = exp.target_next
            if input_window.dim() == 2:
                input_batch = input_window.unsqueeze(0)   # (1, N, D)
                target_batch = target_window.unsqueeze(0)  # (1, N, D)
            elif input_window.dim() == 3:
                input_batch = input_window
                target_batch = target_window
            else:
                raise ValueError(
                    f"experience window has unexpected dim {input_window.dim()}; "
                    f"fixed-shape windows are (num_neurons, neuron_dim)"
                )

            before = self._state_snapshot(self.network)

            n_steps_this_exp = max(1, int(self.replay_steps))
            mse_sum = 0.0
            consolidations = 0
            effective_lr = 0.0
            for _ in range(n_steps_this_exp):
                out, effective_lr = self._scaled_rate_step(
                    input_batch, target_batch, learn=learn)
                if out.get("connection_prediction_mse") is not None:
                    mse_sum += float(out["connection_prediction_mse"])
                # (R3) any periodic consolidation inside the step flows through
                # the connection's own consolidate(); count those events.
                consolidations += int(out["stats"].get("consolidation_consolidated", 0) > 0)

            deltas = self._state_delta_norms(self.network, before)

            provenance = {
                "replayed": True,
                "cycle_id": cycle_id,
                "step": step_i,
                "effective_lr": effective_lr,
                "learning_rate_scale": self.learning_rate_scale,
                "storage_index": exp.metadata.get("storage_index"),
            }
            # R5: tag the experience itself as replayed.
            exp.metadata["replayed"] = True
            exp.metadata["replay_cycle_id"] = cycle_id
            exp.metadata["replay_step"] = step_i
            exp.metadata["replay_effective_lr"] = effective_lr
            self.decision_records.append(provenance)

            results.append(ReplayResult(
                experience=exp,
                prediction_mse=mse_sum / n_steps_this_exp,
                weight_changes=deltas,
                trace_stats={},
                consolidation_triggered=consolidations > 0,
                provenance=provenance,
            ))

            self.total_replays.add_(1)
            self.total_steps_executed.add_(n_steps_this_exp)
            self.replay_mse_sum.add_(mse_sum / n_steps_this_exp)
            if consolidations > 0:
                self.total_consolidations.add_(1)

        return results

    def replay_episodic(
        self,
        num_experiences: int = 32,
        learn: bool = True,
        context_filter: Optional[int] = None,
        cycle_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform episodic replay of retrieved fixed-shape windows (R4 budget:
        ``min(replay_budget, num_experiences, retrieved)`` experiences, 1 step
        each). Returns real replay statistics -- measured, not placeholders.
        """
        # R4: bounded budget.
        budget = min(self.replay_budget, int(num_experiences))

        # Retrieve experiences
        experiences = self.retriever.retrieve(
            num_experiences=budget,
            min_score=0.1,
            context_filter=context_filter,
        )

        if not experiences:
            return {
                "replayed": 0,
                "avg_mse": 0.0,
                "consolidations": 0,
                "total_dream_steps": 0,
                "replay_budget": budget,
                "u_fast_delta_norm": 0.0,
                "theta_slow_delta_norm": 0.0,
                "provenance_count": 0,
            }

        before = self._state_snapshot(self.network)
        results = self.replay_batch(experiences, learn=learn, cycle_id=cycle_id)
        deltas = self._state_delta_norms(self.network, before)

        replayed = len(results)
        stats = {
            "replayed": replayed,
            "avg_mse": (
                sum(r.prediction_mse for r in results) / replayed if replayed else 0.0
            ),
            "consolidations": sum(1 for r in results if r.consolidation_triggered),
            "total_dream_steps": sum(
                max(1, int(self.replay_steps)) for _ in results),
            "replay_budget": budget,
            "u_fast_delta_norm": deltas["u_fast_delta_norm"],
            "theta_slow_delta_norm": deltas["theta_slow_delta_norm"],
            "provenance_count": len(self.decision_records),
        }
        return stats

    def interleave_replay(
        self,
        current_input: torch.Tensor,
        current_target: torch.Tensor,
        learn: bool = True,
        cycle_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Interleave replay with current experience.

        With probability ``interleave_prob``, replay one stored experience
        alongside the current input.
        """
        if random.random() > self.interleave_prob:
            return {"interleaved": False}

        # Retrieve one experience for interleaving
        experiences = self.retriever.retrieve(num_experiences=1)
        if not experiences:
            return {"interleaved": False}

        self.replay_batch(experiences, learn=learn, cycle_id=cycle_id)
        return {"interleaved": True, "replayed": 1}

    def get_stats(self) -> Dict:
        total = self.total_replays.item()
        return {
            "total_replays": total,
            "total_steps_executed": self.total_steps_executed.item(),
            "total_consolidations": self.total_consolidations.item(),
            "avg_replay_mse": self.replay_mse_sum.item() / max(1, total),
            "decision_records": len(self.decision_records),
            "replay_budget": self.replay_budget,
            "learning_rate_scale": self.learning_rate_scale,
        }


def create_experience_replay(
    network: 'RelationalNetwork',
    retriever: 'MemoryRetriever',
    replay_batch_size: int = 16,
    replay_steps: int = 1,
    learning_rate_scale: float = 0.1,
    replay_budget: int = 32,
) -> ExperienceReplay:
    """Factory for experience replay."""
    return ExperienceReplay(
        network=network,
        retriever=retriever,
        replay_batch_size=replay_batch_size,
        replay_steps=replay_steps,
        learning_rate_scale=learning_rate_scale,
        replay_budget=replay_budget,
    )
