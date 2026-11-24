"""
Consolidation management for plastic relational networks.

Handles scheduled consolidation, conflict detection, and forgetting protection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation behavior."""
    # When to consolidate
    consolidation_interval: int = 1000  # Steps between consolidation
    trace_threshold: float = 0.1        # Minimum trace for consolidation
    
    # How fast to consolidate
    consolidation_rate: float = 0.001   # α for slow <- (1-α)*slow + α*fast
    
    # Safeguards
    max_consolidation_per_step: float = 0.01  # Max fraction of weights per step
    forgetting_protection: float = 0.99       # Slow params don't drop below this fraction
    
    # Conflict detection
    conflict_threshold: float = 0.5   # Detect when fast contradicts slow
    conflict_reversal_rate: float = 0.01  # Rate to reverse contradictory learning


class ConsolidationManager:
    """
    Manages consolidation across a population of plastic connections.
    
    Handles:
    - Scheduled consolidation
    - Conflict detection and resolution
    - Catastrophic forgetting protection
    - Statistics tracking
    """
    
    def __init__(
        self,
        connections: Dict[str, 'PlasticRelationalConnection'],
        config: Optional[ConsolidationConfig] = None,
    ):
        self.connections = connections
        self.config = config or ConsolidationConfig()
        self.step_count = 0
        self.consolidation_count = 0
        self.conflict_count = 0
        
    def step(self) -> Dict[str, float]:
        """
        Called each training step.
        
        Returns consolidation statistics.
        """
        self.step_count += 1
        stats = {"consolidation_triggered": False}
        
        if self.step_count % self.config.consolidation_interval == 0:
            stats = self.consolidate_all()
            stats["consolidation_triggered"] = True
            
        return stats
    
    def consolidate_all(self) -> Dict[str, float]:
        """
        Consolidate all connections.
        
        Returns aggregate statistics.
        """
        total_stats = defaultdict(float)
        total_stats["consolidated_connections"] = 0
        total_stats["conflicts_resolved"] = 0
        
        for name, conn in self.connections.items():
            # Check for conflicts before consolidating
            conflicts = self._detect_conflicts(conn)
            if conflicts > 0:
                self._resolve_conflicts(conn, conflicts)
                total_stats["conflicts_resolved"] += conflicts
            
            # Consolidate
            stats = conn.consolidate()
            for k, v in stats.items():
                total_stats[k] += v
            
            if stats.get("consolidated", 0) > 0:
                total_stats["consolidated_connections"] += 1
        
        self.consolidation_count += 1
        return dict(total_stats)
    
    def _detect_conflicts(self, conn: 'PlasticRelationalConnection') -> int:
        """
        Detect connections where fast learning contradicts slow knowledge.
        
        Conflict: sign(fast - slow) != sign(expected_change)
        """
        if hasattr(conn.plasticity, 'u_fast') and hasattr(conn.plasticity, 'theta_slow'):
            fast = conn.plasticity.u_fast
            slow = conn.plasticity.theta_slow
            # Conflict where fast and slow have opposite signs and large magnitude
            diff = fast - slow
            conflict_mask = (fast * slow < 0) & (diff.abs() > self.config.conflict_threshold)
            return int(conflict_mask.sum().item())
        return 0
    
    def _resolve_conflicts(self, conn: 'PlasticRelationalConnection', num_conflicts: int):
        """
        Resolve detected conflicts.
        
        Strategies:
        1. If conflict is small, let consolidation handle it
        2. If conflict is large, reverse fast learning on those connections
        3. Increase consolidation rate for conflicting connections
        """
        if num_conflicts == 0:
            return
            
        self.conflict_count += num_conflicts
        
        if hasattr(conn.plasticity, 'u_fast') and hasattr(conn.plasticity, 'theta_slow'):
            fast = conn.plasticity.u_fast
            slow = conn.plasticity.theta_slow
            diff = fast - slow
            # Conflict where signs differ
            conflict_mask = (fast * slow < 0) & (diff.abs() > self.config.conflict_threshold)
            
            if conflict_mask.any():
                # Reverse fast learning on conflicting connections
                reversal = self.config.conflict_reversal_rate
                fast[conflict_mask] = fast[conflict_mask] * (1 - reversal) + slow[conflict_mask] * reversal
    
    def get_stats(self) -> Dict[str, float]:
        """Get consolidation statistics."""
        return {
            "total_steps": self.step_count,
            "consolidation_events": self.consolidation_count,
            "conflicts_detected": self.conflict_count,
        }
    
    def force_consolidation(self) -> Dict[str, float]:
        """Force immediate consolidation of all connections."""
        return self.consolidate_all()


class ContinualLearningScheduler:
    """
    Schedules consolidation, plasticity, and evaluation for continual learning.
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        consolidation_manager: ConsolidationManager,
        plasticity_schedule: str = "continuous",  # "continuous", "phased", "event_driven"
        eval_interval: int = 1000,
    ):
        self.network = network
        self.consolidation_manager = consolidation_manager
        self.plasticity_schedule = plasticity_schedule
        self.eval_interval = eval_interval
        
        self.phase = "learning"  # "learning", "consolidation", "evaluation"
        self.step = 0
        
    def step(self, prediction_error: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """Take one step of the continual learning schedule."""
        self.step += 1
        stats = {"step": self.step}
        
        # Regular consolidation check
        cons_stats = self.consolidation_manager.step()
        stats.update(cons_stats)
        
        # Phase transitions
        if self.plasticity_schedule == "phased":
            # Alternate between high plasticity and consolidation
            phase_length = 5000
            phase_step = self.step % (2 * phase_length)
            
            if phase_step < phase_length:
                self.phase = "learning"
                self._set_plasticity(1.0)
            else:
                self.phase = "consolidation"
                self._set_plasticity(0.1)
                # Force consolidation at phase start
                if phase_step == phase_length:
                    cons_stats = self.consolidation_manager.force_consolidation()
                    stats.update({f"forced_{k}": v for k, v in cons_stats.items()})
        
        stats["phase"] = self.phase
        return stats
    
    def _set_plasticity(self, scale: float):
        """Scale plasticity rates."""
        for conn in self.network.connections.values():
            conn.plasticity.learning_rate *= scale
            conn.eligibility_trace.decay = 0.9 + 0.1 * scale  # Faster decay = less plasticity
    
    def evaluate_retention(self, test_inputs: torch.Tensor, test_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate retention of previous knowledge."""
        self.network.eval()
        with torch.no_grad():
            total_error = 0
            for inp, target in zip(test_inputs, test_targets):
                out = self.network.step(inp, learn=False)
                error = (out["predicted_next"] - target).pow(2).mean()
                total_error += error.item()
            
            return {
                "retention_mse": total_error / len(test_inputs),
                "phase": self.phase,
            }