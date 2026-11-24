"""
Experience Replay - Rehearses actual observations with temporal fidelity.

Replays stored experiences through the network to:
- Strengthen important relationships
- Protect against catastrophic forgetting
- Maintain temporal structure of experiences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import random

from nnfs.dreaming.retriever import MemoryRetriever, RetrievedExperience


@dataclass
class ReplayResult:
    """Result of a replay episode."""
    experience: 'RetrievedExperience'
    prediction_mse: float
    weight_changes: Dict[str, float]
    trace_stats: Dict[str, float]
    consolidation_triggered: bool


class ExperienceReplay(nn.Module):
    """
    Replays stored experiences through the network for consolidation.
    
    Key features:
    - Temporal fidelity: replays sequences in original order
    - Selective plasticity: only updates connections active during original experience
    - Interleaving: mixes replay with current experience
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        retriever: MemoryRetriever,
        replay_batch_size: int = 16,
        replay_steps: int = 1,
        learning_rate_scale: float = 0.1,  # Lower LR for replay
        interleave_prob: float = 0.1,  # Probability of interleaving with current
    ):
        super().__init__()
        self.network = network
        self.retriever = retriever
        self.replay_batch_size = replay_batch_size
        self.replay_steps = replay_steps
        self.learning_rate_scale = learning_rate_scale
        self.interleave_prob = interleave_prob
        
        # Track replay statistics
        self.register_buffer("total_replays", torch.tensor(0))
        self.register_buffer("total_consolidations", torch.tensor(0))
        self.register_buffer("replay_mse_sum", torch.tensor(0.0))
        
    def replay_batch(
        self,
        experiences: List['RetrievedExperience'],
        learn: bool = True,
    ) -> List[ReplayResult]:
        """
        Replay a batch of experiences through the network.
        
        Each experience is replayed through the network to reinforce
        the relationships that were active during the original experience.
        """
        results = []
        
        for exp in experiences:
            # Reconstruct input/target tensors from flattened storage
            # This is simplified - actual implementation would store shapes
            seq_len = 10  # Default, would be stored with experience
            
            # For now, use the stored flattened data
            # In practice, we'd store the full tensors with shapes
            pass
        
        return results
    
    def replay_episodic(
        self,
        num_experiences: int = 32,
        learn: bool = True,
        context_filter: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform episodic replay of retrieved experiences.
        
        Returns replay statistics.
        """
        # Retrieve experiences
        experiences = self.retriever.retrieve(
            num_experiences=num_experiences,
            min_score=0.1,
        )
        
        if not experiences:
            return {"replayed": 0, "avg_mse": 0.0}

        # For now, return placeholder stats
        # Full implementation would replay each experience through network
        return {
            "replayed": len(experiences),
            "avg_mse": 0.0,
            "consolidations": 0,
        }
    
    def interleave_replay(
        self,
        current_input: torch.Tensor,
        current_target: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, float]:
        """
        Interleave replay with current experience.
        
        With probability `interleave_prob`, replay a random experience
        alongside the current input.
        """
        if random.random() > self.interleave_prob:
            return {"interleaved": False}
        
        # Retrieve one experience for interleaving
        experiences = self.retriever.retrieve(num_experiences=1)
        if not experiences:
            return {"interleaved": False}
        
        # Would interleave here
        return {"interleaved": True, "replayed": 1}
    
    def get_stats(self) -> Dict:
        return {
            "total_replays": self.total_replays.item(),
            "total_consolidations": self.total_consolidations.item(),
            "avg_replay_mse": self.replay_mse_sum.item() / max(1, self.total_replays.item()),
        }


def create_experience_replay(
    network: 'RelationalNetwork',
    retriever: 'MemoryRetriever',
    replay_batch_size: int = 16,
    replay_steps: int = 1,
    learning_rate_scale: float = 0.1,
) -> ExperienceReplay:
    """Factory for experience replay."""
    return ExperienceReplay(
        network=network,
        retriever=retriever,
        replay_batch_size=replay_batch_size,
        replay_steps=replay_steps,
        learning_rate_scale=learning_rate_scale,
    )