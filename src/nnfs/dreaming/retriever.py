"""
Memory Retriever - Selects important experiences for replay.

Selects experiences based on:
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
    """An experience selected for replay."""
    input_signal: torch.Tensor      # (seq_len, num_neurons, neuron_dim)
    target_next: torch.Tensor       # (seq_len, num_neurons, neuron_dim)
    context: Optional[torch.Tensor] # (context_dim,)
    significance: float             # How important this experience is
    recency: float                  # How recent (0=old, 1=recent)
    uncertainty: float              # Prediction uncertainty during experience
    context_id: Optional[int]       # Context identifier
    metadata: Dict                  # Additional info


class MemoryRetriever(nn.Module):
    """
    Selects important experiences for replay based on multiple criteria.
    
    Selection criteria:
    - Significance: experiences with high prediction error (surprising)
    - Recency: recent experiences weighted higher
    - Uncertainty: experiences with high prediction uncertainty
    - Diversity: covers different contexts and relationship types
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
        experience_dim: int = 256,  # Default for 16x16 network
    ):
        super().__init__()
        self.max_memory_size = max_memory_size
        self.significance_weight = significance_weight
        self.recency_weight = recency_weight
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.min_significance = min_significance
        self.context_dim = context_dim
        self.experience_dim = experience_dim
        
        # Episodic memory buffer
        self.register_buffer("experience_inputs", torch.zeros(max_memory_size, experience_dim))
        self.register_buffer("experience_targets", torch.zeros(max_memory_size, experience_dim))
        self.register_buffer("experience_contexts", torch.zeros(max_memory_size, context_dim) if context_dim > 0 else torch.zeros(1))
        self.register_buffer("experience_significance", torch.zeros(max_memory_size))
        self.register_buffer("experience_timestamps", torch.zeros(max_memory_size))
        self.register_buffer("experience_uncertainty", torch.zeros(max_memory_size))
        self.register_buffer("experience_context_ids", torch.zeros(max_memory_size, dtype=torch.long))
        
        self.register_buffer("current_size", torch.tensor(0))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0))
        
        # Diversity tracking
        self.register_buffer("context_counts", torch.zeros(100))  # Assuming max 100 contexts
        
    def store(
        self,
        input_signal: torch.Tensor,      # (seq_len, num_neurons, neuron_dim) or (num_neurons, neuron_dim)
        target_next: torch.Tensor,       # (seq_len, num_neurons, neuron_dim) or (num_neurons, neuron_dim)
        context: Optional[torch.Tensor] = None,
        significance: float = 1.0,
        uncertainty: float = 0.0,
        context_id: Optional[int] = None,
    ):
        """Store an experience in episodic memory."""
        # Handle both single step and sequence
        if input_signal.dim() == 3:  # (seq_len, num_neurons, neuron_dim)
            # Average over sequence for single experience
            input_flat = input_signal.flatten(1).mean(dim=0)  # Average over seq_len
            target_flat = target_next.flatten(1).mean(dim=0)
        else:
            input_flat = input_signal.flatten()
            target_flat = target_next.flatten()
        
        # Ensure the flattened size matches experience_dim
        if input_flat.numel() != self.experience_dim:
            # Resize if needed (pad or truncate)
            if input_flat.numel() < self.experience_dim:
                padding = torch.zeros(self.experience_dim - input_flat.numel(), device=input_flat.device)
                input_flat = torch.cat([input_flat, padding])
                target_flat = torch.cat([target_flat, torch.zeros_like(padding)])
            else:
                input_flat = input_flat[:self.experience_dim]
                target_flat = target_flat[:self.experience_dim]
        
        ptr = int(self.write_ptr.item())
        max_size = self.max_memory_size
        
        # Store experience (flattened)
        self.experience_inputs[ptr] = input_flat
        self.experience_targets[ptr] = target_flat
        
        if self.context_dim > 0 and context is not None:
            self.experience_contexts[ptr] = context.squeeze()
        
        self.experience_significance[ptr] = significance
        self.experience_uncertainty[ptr] = uncertainty
        self.experience_timestamps[ptr] = self.global_step.item()
        self.experience_context_ids[ptr] = context_id if context_id is not None else -1
        
        if context_id is not None and context_id >= 0:
            self.context_counts[context_id] += 1
        
        self.write_ptr.fill_((ptr + 1) % self.max_memory_size)
        self.current_size.fill_(min(self.current_size.item() + 1, self.max_memory_size))
        self.global_step += 1
    
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
        Retrieve top experiences for replay.
        
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
                input_signal=self.experience_inputs[idx].clone(),
                target_next=self.experience_targets[idx].clone(),
                context=self.experience_contexts[idx] if self.context_dim > 0 else None,
                significance=self.experience_significance[idx].item(),
                recency=1.0 / (1.0 + (self.global_step.item() - self.experience_timestamps[idx].item()) / 1000.0),
                uncertainty=self.experience_uncertainty[idx].item(),
                context_id=self.experience_context_ids[idx].item() if self.experience_context_ids[idx].item() >= 0 else None,
                metadata={
                    "retrieval_score": score,
                    "timestamp": self.experience_timestamps[idx].item(),
                }
            )
            retrieved.append(exp)
        
        return retrieved
    
    def get_stats(self) -> Dict:
        return {
            "stored_experiences": self.current_size.item(),
            "total_steps": self.global_step.item(),
            "avg_significance": self.experience_significance[:self.current_size].mean().item() if self.current_size > 0 else 0,
            "context_distribution": self.context_counts[:100].tolist() if self.context_dim > 0 else [],
        }
    
    def clear(self):
        """Clear all stored experiences."""
        self.experience_inputs.zero_()
        self.experience_targets.zero_()
        self.experience_contexts.zero_()
        self.experience_significance.zero_()
        self.experience_timestamps.zero_()
        self.experience_uncertainty.zero_()
        self.experience_context_ids.fill_(-1)
        self.current_size.zero_()
        self.write_ptr.zero_()
        self.global_step.zero_()
        self.context_counts.zero_()


def create_retriever(
    max_memory: int = 10000,
    significance_weight: float = 0.4,
    recency_weight: float = 0.2,
    uncertainty_weight: float = 0.2,
    diversity_weight: float = 0.2,
    context_dim: int = 0,
    experience_dim: int = 256,
) -> MemoryRetriever:
    """Factory for memory retriever."""
    return MemoryRetriever(
        max_memory_size=max_memory,
        significance_weight=significance_weight,
        recency_weight=recency_weight,
        uncertainty_weight=uncertainty_weight,
        diversity_weight=diversity_weight,
        context_dim=context_dim,
        experience_dim=experience_dim,
    )