"""
Memory Retriever - Selects important experiences for replay.

Rebuilt per owner decision D2 (spec DREAM_REIMPL_SPEC.md §2 R1, 2026-11-25):
the experience ring holds FIXED-SHAPE experience windows -- the un-averaged
(input, target) tensors at their stored (num_neurons, neuron_dim) shape --
so replay can reconstruct exactly what was observed. The old flattened
records (averaged over seq_len, padded/truncated to experience_dim) are
ARCHIVED, never replayed: the reconstruction information they lost at store
time cannot be recovered, so they are kept only as an inert record.

Selection criteria (unchanged):
- Significance: prediction error magnitude, surprise
- Recency: recent experiences weighted higher
- Uncertainty: experiences with high prediction uncertainty
- Diversity: covers different contexts and relationship types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import random


@dataclass
class RetrievedExperience:
    """An experience selected for replay.

    Since the D2 storage rebuild, ``input_signal`` and ``target_next`` are
    the FIXED-SHAPE windows stored at observation time --
    (num_neurons, neuron_dim) for a single-step window -- never flattened.
    """
    input_signal: torch.Tensor      # (num_neurons, neuron_dim) fixed-shape window
    target_next: torch.Tensor       # (num_neurons, neuron_dim) fixed-shape window
    context: Optional[torch.Tensor] # (context_dim,)
    significance: float             # How important this experience is
    recency: float                  # How recent (0=old, 1=recent)
    uncertainty: float              # Prediction uncertainty during experience
    context_id: Optional[int]       # Context identifier
    metadata: Dict                  # Additional info


class MemoryRetriever(nn.Module):
    """
    Fixed-shape experience ring (owner decision D2, spec §2 R1).

    Each slot stores one experience window: the un-averaged (input, target)
    tensors at their stored (num_neurons, neuron_dim) shape, plus context,
    context_id, significance, uncertainty and the original sequence length.
    The ring is bounded: ``max_memory_size`` windows, overwritten cyclically.

    Legacy flattened records (pre-rebuild format: averaged over seq_len,
    padded/truncated to experience_dim) can only enter through
    :meth:`archive_legacy_flattened`. They are held in an inert archive,
    reported in stats, and NEVER returned by :meth:`retrieve` -- flattened
    records are never replayed (owner D2).
    """

    def __init__(
        self,
        max_memory_size: int = 10000,
        significance_weight: float = 0.4,
        recency_weight: float = 0.2,
        uncertainty_weight: float = 0.2,
        diversity_weight: float = 0.2,
        min_significance: float = 0.1,
        context_dim: int = 0,
        num_neurons: int = 16,
        neuron_dim: int = 16,
        max_archived: int = 1000,
        experience_dim: Optional[int] = None,  # legacy kwarg, ignored (D2 rebuild)
    ):
        super().__init__()
        self.max_memory_size = max_memory_size
        self.significance_weight = significance_weight
        self.recency_weight = recency_weight
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.min_significance = min_significance
        self.context_dim = context_dim
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.max_archived = max_archived

        # Fixed-shape experience window ring (D2): one (num_neurons, neuron_dim)
        # window pair per slot. Bounded memory: 2 * max_memory_size * N * D floats.
        self.register_buffer(
            "window_inputs", torch.zeros(max_memory_size, num_neurons, neuron_dim))
        self.register_buffer(
            "window_targets", torch.zeros(max_memory_size, num_neurons, neuron_dim))
        self.register_buffer(
            "window_contexts",
            torch.zeros(max_memory_size, context_dim) if context_dim > 0 else torch.zeros(1))
        self.register_buffer("experience_significance", torch.zeros(max_memory_size))
        self.register_buffer("experience_timestamps", torch.zeros(max_memory_size))
        self.register_buffer("experience_uncertainty", torch.zeros(max_memory_size))
        self.register_buffer("experience_context_ids", torch.zeros(max_memory_size, dtype=torch.long))
        # Original sequence length of the observation the window came from.
        self.register_buffer("window_seq_lens", torch.ones(max_memory_size, dtype=torch.long))
        # Provenance at store time: 0 = live observation. Archived records never
        # enter this ring.
        self.register_buffer("window_is_live", torch.ones(max_memory_size, dtype=torch.long))

        self.register_buffer("current_size", torch.tensor(0))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0))

        # Diversity tracking
        self.register_buffer("context_counts", torch.zeros(100))  # Assuming max 100 contexts

        # Legacy flattened archive (D2): inert, never replayed.
        self.archived_flat: List[Dict] = []
        self.archived_dropped = 0

    # ------------------------------------------------------------------
    # Storage (fixed-shape windows)
    # ------------------------------------------------------------------
    def store(
        self,
        input_signal: torch.Tensor,      # (num_neurons, neuron_dim) or (seq_len, num_neurons, neuron_dim)
        target_next: torch.Tensor,       # same shape as input_signal
        context: Optional[torch.Tensor] = None,
        significance: float = 1.0,
        uncertainty: float = 0.0,
        context_id: Optional[int] = None,
    ):
        """Store one experience as a fixed-shape window (un-averaged).

        A (seq_len, num_neurons, neuron_dim) sequence stores its LAST step as
        the window (spec §2 R1: "un-averaged (or last-step)") and records the
        original seq_len alongside it. Nothing is flattened, averaged, padded
        or truncated: replay reconstructs exactly what was observed.
        """
        if input_signal.dim() == 3:
            seq_len = int(input_signal.shape[0])
            input_window = input_signal[-1].detach().clone()
            target_window = target_next[-1].detach().clone()
        elif input_signal.dim() == 2:
            seq_len = 1
            input_window = input_signal.detach().clone()
            target_window = target_next.detach().clone()
        else:
            raise ValueError(
                f"store() expects (num_neurons, neuron_dim) or "
                f"(seq_len, num_neurons, neuron_dim); got dim={input_signal.dim()}"
            )

        expected = (self.num_neurons, self.neuron_dim)
        if tuple(input_window.shape) != expected:
            raise ValueError(
                f"window shape {tuple(input_window.shape)} does not match the "
                f"retriever's fixed window shape {expected}; construct the "
                f"retriever with matching num_neurons/neuron_dim"
            )

        ptr = int(self.write_ptr.item())

        self.window_inputs[ptr] = input_window
        self.window_targets[ptr] = target_window

        if self.context_dim > 0 and context is not None:
            self.window_contexts[ptr] = context.squeeze()

        self.experience_significance[ptr] = significance
        self.experience_uncertainty[ptr] = uncertainty
        self.experience_timestamps[ptr] = self.global_step.item()
        self.experience_context_ids[ptr] = context_id if context_id is not None else -1
        self.window_seq_lens[ptr] = seq_len
        self.window_is_live[ptr] = 1  # live observation

        if context_id is not None and context_id >= 0:
            self.context_counts[context_id] += 1

        self.write_ptr.fill_((ptr + 1) % self.max_memory_size)
        self.current_size.fill_(min(self.current_size.item() + 1, self.max_memory_size))
        self.global_step += 1

    # ------------------------------------------------------------------
    # Legacy flattened archive (D2): stored inert, NEVER replayed
    # ------------------------------------------------------------------
    def archive_legacy_flattened(
        self,
        inputs_flat: torch.Tensor,
        targets_flat: torch.Tensor,
        significance: Optional[torch.Tensor] = None,
    ) -> int:
        """Archive legacy flattened records so they are never replayed.

        Accepts the pre-rebuild flattened buffers (max_memory_size,
        experience_dim) -- or any (n, d) flattened batch -- and moves them
        into the inert archive. Archived records are counted in stats and
        can never be returned by retrieve(); replay cannot reach them.
        Returns the number archived.
        """
        n = int(inputs_flat.shape[0])
        archived = 0
        for i in range(n):
            if len(self.archived_flat) >= self.max_archived:
                self.archived_dropped += n - i
                break
            self.archived_flat.append({
                "input_flat": inputs_flat[i].detach().clone(),
                "target_flat": targets_flat[i].detach().clone(),
                "significance": (
                    float(significance[i].item())
                    if significance is not None else None
                ),
            })
            archived += 1
        return archived

    def replayable_count(self) -> int:
        """Number of REPLAYABLE (fixed-shape window) experiences."""
        return int(self.current_size.item())

    # ------------------------------------------------------------------
    # Retrieval scoring (unchanged policy, windows instead of flats)
    # ------------------------------------------------------------------
    def compute_diversity_bonus(self, context_id: int) -> float:
        """Compute diversity bonus for underrepresented contexts."""
        if context_id < 0 or context_id >= self.context_counts.shape[0]:
            return 0.0
        count = self.context_counts[context_id].item()
        total = self.context_counts.sum().item() + 1e-8
        # Inverse frequency: rare contexts get higher bonus
        return 1.0 / (1.0 + count / total * 10)

    def compute_retrieval_score(self, idx: int) -> float:
        """Compute retrieval priority score for an experience."""
        sig = self.experience_significance[idx].item()
        if sig < self.min_significance:
            return 0.0

        # Recency: more recent = higher score
        age = self.global_step.item() - self.experience_timestamps[idx].item()
        recency = 1.0 / (1.0 + age / 1000.0)

        # Uncertainty: higher uncertainty = higher priority
        unc = self.experience_uncertainty[idx].item()

        # Diversity bonus
        ctx_id = self.experience_context_ids[idx].item()
        div = self.compute_diversity_bonus(ctx_id) if ctx_id >= 0 else 0.0

        # Weighted combination
        score = (
            self.significance_weight * sig +
            self.recency_weight * recency +
            self.uncertainty_weight * unc +
            self.diversity_weight * div
        )

        return score

    def retrieve(
        self,
        num_experiences: int = 32,
        min_score: float = 0.1,
        context_filter: Optional[int] = None,
    ) -> List[RetrievedExperience]:
        """
        Retrieve top fixed-shape window experiences for replay.

        Only fixed-shape window records are ever returned. Archived legacy
        flattened records are never served (owner D2).

        Args:
            num_experiences: Number of experiences to retrieve
            min_score: Minimum retrieval score threshold
            context_filter: Only retrieve experiences from this context

        Returns:
            List of retrieved experiences sorted by priority
        """
        size = self.current_size.item()
        if size == 0:
            return []

        # Score all experiences
        scores = []
        for i in range(size):
            if context_filter is not None and self.experience_context_ids[i].item() != context_filter:
                continue
            score = self.compute_retrieval_score(i)
            if score >= min_score:
                scores.append((score, i))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # Retrieve top experiences
        retrieved = []
        for score, idx in scores[:num_experiences]:
            exp = RetrievedExperience(
                input_signal=self.window_inputs[idx].clone(),
                target_next=self.window_targets[idx].clone(),
                context=self.window_contexts[idx] if self.context_dim > 0 else None,
                significance=self.experience_significance[idx].item(),
                recency=1.0 / (1.0 + (self.global_step.item() - self.experience_timestamps[idx].item()) / 1000.0),
                uncertainty=self.experience_uncertainty[idx].item(),
                context_id=self.experience_context_ids[idx].item() if self.experience_context_ids[idx].item() >= 0 else None,
                metadata={
                    "retrieval_score": score,
                    "timestamp": self.experience_timestamps[idx].item(),
                    "storage_index": int(idx),
                    "seq_len": int(self.window_seq_lens[idx].item()),
                    "is_live_observation": bool(self.window_is_live[idx].item() == 1),
                }
            )
            retrieved.append(exp)

        return retrieved

    def get_stats(self) -> Dict:
        return {
            "stored_experiences": self.current_size.item(),
            "replayable_experiences": self.current_size.item(),
            "archived_flat_records": len(self.archived_flat),
            "archived_flat_dropped": self.archived_dropped,
            "window_shape": [self.num_neurons, self.neuron_dim],
            "total_steps": self.global_step.item(),
            "avg_significance": self.experience_significance[:self.current_size].mean().item() if self.current_size > 0 else 0,
            "context_distribution": self.context_counts[:100].tolist() if self.context_dim > 0 else [],
        }

    def clear(self):
        """Clear the replayable window ring. The legacy archive is retained."""
        self.window_inputs.zero_()
        self.window_targets.zero_()
        self.window_contexts.zero_()
        self.experience_significance.zero_()
        self.experience_timestamps.zero_()
        self.experience_uncertainty.zero_()
        self.experience_context_ids.fill_(-1)
        self.window_seq_lens.fill_(1)
        self.window_is_live.fill_(1)
        self.current_size.zero_()
        self.write_ptr.zero_()
        self.global_step.zero_()
        self.context_counts.zero_()

    def clear_archive(self):
        """Clear the inert legacy flattened archive."""
        self.archived_flat = []
        self.archived_dropped = 0


def create_retriever(
    max_memory: int = 10000,
    significance_weight: float = 0.4,
    recency_weight: float = 0.2,
    uncertainty_weight: float = 0.2,
    diversity_weight: float = 0.2,
    context_dim: int = 0,
    num_neurons: int = 16,
    neuron_dim: int = 16,
    experience_dim: Optional[int] = None,  # legacy kwarg, ignored (D2 rebuild)
) -> MemoryRetriever:
    """Factory for memory retriever (fixed-shape window ring, owner D2)."""
    return MemoryRetriever(
        max_memory_size=max_memory,
        significance_weight=significance_weight,
        recency_weight=recency_weight,
        uncertainty_weight=uncertainty_weight,
        diversity_weight=diversity_weight,
        context_dim=context_dim,
        num_neurons=num_neurons,
        neuron_dim=neuron_dim,
        experience_dim=experience_dim,
    )
