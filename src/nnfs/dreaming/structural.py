"""
Structural Dreamer - Experiments with network topology during dreaming.

Experiments with creating, pruning, and reorganizing connections
to discover more efficient network structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import random

from nnfs.connections.plastic import PlasticRelationalConnection
from nnfs.dreaming.recombiner import HypothesizedRelationship


def _live_connection_params(network) -> Dict:
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


@dataclass
class StructuralProposal:
    """A proposed structural change to the network."""
    action: str  # "create", "prune", "split", "merge", "rewire"
    connection_key: Optional[str] = None
    pre_idx: Optional[int] = None
    post_idx: Optional[int] = None
    reason: str = ""
    expected_benefit: float = 0.0
    risk: float = 0.0
    confidence: float = 0.0


@dataclass
class StructuralChangeResult:
    """Result of applying a structural change."""
    proposal: StructuralProposal
    accepted: bool
    actual_benefit: float = 0.0
    side_effects: Dict[str, float] = None
    rollback_info: Any = None


class StructuralDreamer(nn.Module):
    """
    Experiments with network topology during dreaming.
    
    Operations:
    - Create: Add new connections where they might be useful
    - Prune: Remove weak or harmful connections
    - Split: Split overloaded connections
    - Merge: Merge redundant connections
    - Rewire: Change connection endpoints for better structure
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        max_proposals_per_dream: int = 5,
        creation_threshold: float = 0.3,
        pruning_threshold: float = 0.01,
        min_connection_strength: float = 0.01,
        max_connections_per_neuron: int = 8,
    ):
        super().__init__()
        self.network = network
        self.max_proposals_per_dream = max_proposals_per_dream
        self.creation_threshold = creation_threshold
        self.pruning_threshold = pruning_threshold
        self.min_connection_strength = min_connection_strength
        self.max_connections_per_neuron = max_connections_per_neuron
        
        # Statistics
        self.register_buffer("proposals_generated", torch.tensor(0))
        self.register_buffer("proposals_accepted", torch.tensor(0))
        self.register_buffer("proposals_rejected", torch.tensor(0))
        self.register_buffer("connections_created", torch.tensor(0))
        self.register_buffer("connections_pruned", torch.tensor(0))
        
    def dream_step(self) -> List[StructuralChangeResult]:
        """
        Perform one dreaming step: generate and evaluate structural proposals.
        """
        proposals = self._generate_proposals()
        
        results = []
        for proposal in proposals[:self.max_proposals_per_dream]:
            result = self._evaluate_and_apply(proposal)
            results.append(result)
            
            if result.accepted:
                self.proposals_accepted.add_(1)
            else:
                self.proposals_rejected.add_(1)
            
            self.proposals_generated.add_(1)
        
        return results
    
    def _generate_proposals(self) -> List[StructuralProposal]:
        """Generate structural proposals based on network state."""
        proposals = []
        
        # 1. Pruning proposals: weak or harmful connections
        prune_proposals = self._generate_pruning_proposals()
        proposals.extend(prune_proposals)
        
        # 2. Creation proposals: missing but potentially useful connections
        create_proposals = self._generate_creation_proposals()
        proposals.extend(create_proposals)
        
        # 3. Rewiring proposals: connections that might be better elsewhere
        rewire_proposals = self._generate_rewire_proposals()
        proposals.extend(rewire_proposals)
        
        # Sort by expected benefit
        proposals.sort(key=lambda p: p.expected_benefit - p.risk, reverse=True)
        
        return proposals
    
    def _generate_pruning_proposals(self) -> List['StructuralProposal']:
        """Generate proposals to prune weak/harmful connections."""
        proposals = []
        
        for key, conn in self.network.connections.items():
            # Compute connection utility
            strength = conn.plasticity.get_combined_weight().norm().item()
            trace_activity = conn.eligibility_trace.trace.abs().mean().item()
            
            # Predicted utility: strength * activity
            utility = strength * trace_activity
            
            if utility < self.pruning_threshold:
                # Check if neuron would still have enough connections
                pre = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
                post = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
                
                pre_connections = 0
                post_connections = 0
                for k2, conn2 in self.network.connections.items():
                    pre2 = int(conn2.pre_idx.item()) if hasattr(conn2.pre_idx, 'item') else int(conn2.pre_idx)
                    post2 = int(conn2.post_idx.item()) if hasattr(conn2.post_idx, 'item') else int(conn2.post_idx)
                    if pre2 == pre:
                        pre_connections += 1
                    if post2 == post:
                        post_connections += 1
                
                if pre_connections > 1 and post_connections > 1:
                    proposals.append(StructuralProposal(
                        action="prune",
                        connection_key=key,
                        pre_idx=pre,
                        post_idx=post,
                        reason=f"Low utility: strength={strength:.4f}, activity={trace_activity:.4f}",
                        expected_benefit=0.5,  # Free up capacity
                        risk=0.2,  # Low risk for weak connections
                        confidence=0.8,
                    ))
        
        return proposals
    
    def _generate_creation_proposals(self) -> List['StructuralProposal']:
        """Generate proposals to create new connections."""
        proposals = []
        
        # Build existing connection map
        existing_connections = set()
        pre_counts = {}
        post_counts = {}
        for key, conn in self.network.connections.items():
            pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
            post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
            existing_connections.add((pre_idx, post_idx))
            pre_counts[pre_idx] = pre_counts.get(pre_idx, 0) + 1
            post_counts[post_idx] = post_counts.get(post_idx, 0) + 1
        
        # Find neuron pairs without connections that might benefit
        # Prioritize neurons with few connections
        for i in range(self.network.num_neurons):
            for j in range(self.network.num_neurons):
                if i == j:
                    continue
                if (i, j) in existing_connections:
                    continue
                
                pre_conns = pre_counts.get(i, 0)
                post_conns = post_counts.get(j, 0)
                
                if pre_conns >= self.max_connections_per_neuron or post_conns >= self.max_connections_per_neuron:
                    continue
                
                # Heuristic: prefer neurons with some but not too many connections
                # Also prefer balanced connectivity
                expected_utility = 0.7 + 0.1 * (1.0 / (1.0 + pre_conns)) + 0.1 * (1.0 / (1.0 + post_conns))
                
                if expected_utility > self.creation_threshold:
                    proposals.append(StructuralProposal(
                        action="create",
                        pre_idx=i,
                        post_idx=j,
                        reason=f"Under-connected neurons: {i} (out={pre_conns}) -> {j} (in={post_conns})",
                        expected_benefit=expected_utility,
                        risk=0.2,  # Lower risk for creation
                        confidence=0.7,  # Higher confidence
                    ))
        
        # Sort by expected benefit
        proposals.sort(key=lambda p: p.expected_benefit, reverse=True)
        return proposals[:10]  # Limit to top 10
    
    def _generate_rewire_proposals(self) -> List['StructuralProposal']:
        """Generate proposals to rewire connections."""
        proposals = []
        
        # Find connections that might be better connected elsewhere
        for key, conn in self.network.connections.items():
            # Check if this connection would be better targeting a different neuron
            # (Simplified heuristic)
            pass
        
        return proposals
    
    def _evaluate_and_apply(self, proposal: 'StructuralProposal') -> 'StructuralChangeResult':
        """Evaluate a proposal and apply if accepted."""
        # Accept if expected benefit > risk
        net_benefit = proposal.expected_benefit - proposal.risk
        
        if net_benefit > 0 and proposal.confidence >= 0.5:
            # Apply the change
            result = self._apply_proposal(proposal)
            return StructuralChangeResult(
                proposal=proposal,
                accepted=True,
                actual_benefit=net_benefit,
                side_effects={},
                rollback_info=None,
            )
        else:
            return StructuralChangeResult(
                proposal=proposal,
                accepted=False,
                actual_benefit=0.0,
                side_effects={},
                rollback_info=None,
            )
    
    def _apply_proposal(self, proposal: 'StructuralProposal') -> Dict:
        """Apply an accepted structural proposal."""
        result = {"success": True, "details": ""}
        
        if proposal.action == "prune" and proposal.connection_key:
            if proposal.connection_key in self.network.connections:
                del self.network.connections[proposal.connection_key]
                self.connections_pruned.add_(1)
                result["details"] = f"Pruned {proposal.connection_key}"
                # Rebuild indices after pruning
                self.network.rebuild_connection_indices()
                
        elif proposal.action == "create" and proposal.pre_idx is not None and proposal.post_idx is not None:
            # Create new connection
            pre, post = proposal.pre_idx, proposal.post_idx
            
            # Check if already exists
            exists = False
            for key, conn in self.network.connections.items():
                pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
                post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
                if pre_idx == pre and post_idx == post:
                    exists = True
                    break
            
            if not exists:
                from nnfs.connections.plastic import PlasticRelationalConnection
                # Owner fix 4 (2026-11-25): inherit the live network's plasticity
                # parameters (lr / blend) instead of hardcoded lr=0.01 and the
                # default blend=0.7.
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
                new_key = f"conn_{len(self.network.connections)}"
                self.network.connections[new_key] = conn
                
                # Rebuild indices after creation
                self.network.rebuild_connection_indices()
                
                self.connections_created.add_(1)
                result["details"] = f"Created {new_key} ({pre}->{post})"
        
        return result
    
    def get_stats(self) -> Dict:
        return {
            "proposals_generated": self.proposals_generated.item(),
            "proposals_accepted": self.proposals_accepted.item(),
            "proposals_rejected": self.proposals_rejected.item(),
            "connections_created": self.connections_created.item(),
            "connections_pruned": self.connections_pruned.item(),
        }


def create_structural_dreamer(
    network: 'RelationalNetwork',
    max_proposals: int = 5,
    creation_threshold: float = 0.7,
    pruning_threshold: float = 0.1,
) -> StructuralDreamer:
    """Factory for structural dreamer."""
    return StructuralDreamer(
        network=network,
        max_proposals_per_dream=max_proposals,
        creation_threshold=creation_threshold,
        pruning_threshold=pruning_threshold,
    )