"""
Pattern Recombiner - Creates hypothetical relationships by combining existing ones.

The Pattern Recombiner explores the space of possible relationships by
combining existing learned relationships in novel ways, then testing
them through the dream simulator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import random
import itertools

from nnfs.connections.plastic import PlasticRelationalConnection


@dataclass
class HypothesizedRelationship:
    """A hypothesized relationship created by recombination."""
    pre_idx: int
    post_idx: int
    relationship_type: str  # "composition", "analogy", "inversion", "context_transfer"
    source_relationships: List[Tuple[int, int]]  # Source connection pairs
    hypothesis_weight: torch.Tensor  # Proposed relationship parameters
    confidence: float  # Confidence in the hypothesis
    source_contexts: List[Optional[int]]  # Contexts of source relationships
    target_context: Optional[int] = None  # Context where this might apply


class PatternRecombiner(nn.Module):
    """
    Creates hypothetical relationships by combining existing learned relationships.
    
    Recombination strategies:
    1. Composition: A→B and B→C implies A→C
    2. Analogy: A→B in context X, C→D in context Y, try A→D in context X
    3. Inversion: A→B suggests B→A (with different semantics)
    4. Context transfer: Relationship in context A might apply in context B
    5. Generalization: Specific relationship → more general pattern
    """
    
    def __init__(
        self,
        network: 'RelationalNetwork',
        max_hypotheses_per_step: int = 10,
        composition_depth: int = 2,
        analogy_threshold: float = 0.7,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.network = network
        self.max_hypotheses_per_step = max_hypotheses_per_step
        self.composition_depth = composition_depth
        self.analogy_threshold = analogy_threshold
        self.confidence_threshold = confidence_threshold
        
        # Track generated hypotheses
        self.register_buffer("hypotheses_generated", torch.tensor(0))
        self.register_buffer("hypotheses_tested", torch.tensor(0))
        self.register_buffer("hypotheses_accepted", torch.tensor(0))
        
    def find_compositional_chains(self, max_depth: int = 2) -> List[List[Tuple[int, int]]]:
        """
        Find compositional chains: A→B, B→C, C→D...
        
        Returns list of chains where each chain is a sequence of connection indices.
        """
        chains = []
        connections = dict(self.network.connections)
        
        # Build adjacency
        adj = {}
        for key, conn in connections.items():
            pre = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
            post = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
            if pre not in adj:
                adj[pre] = []
            adj[pre].append((post, key))
        
        # Find chains up to max_depth
        def dfs(current, path, depth):
            if depth >= self.composition_depth:
                if len(path) > 1:
                    chains.append(path[:])
                return
            
            if current in adj:
                for next_node, conn_key in adj[current]:
                    path.append((current, next_node))
                    dfs(next_node, path, depth + 1)
                    path.pop()
        
        for start in adj:
            dfs(start, [], 0)
        
        return chains
    
    def find_analogical_pairs(
        self, 
        min_similarity: float = 0.7
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Find analogous relationships across contexts.
        
        Returns pairs of connections with similarity scores as (pre1, post1, pre2, post2, sim).
        """
        analogies = []
        connections = dict(self.network.connections)
        conn_list = list(connections.items())
        
        for i, (key1, conn1) in enumerate(conn_list):
            # Get relationship pattern for conn1
            w1 = conn1.plasticity.get_combined_weight().data
            pre1 = int(conn1.pre_idx.item()) if hasattr(conn1.pre_idx, 'item') else int(conn1.pre_idx)
            post1 = int(conn1.post_idx.item()) if hasattr(conn1.post_idx, 'item') else int(conn1.post_idx)
            
            for j, (key2, conn2) in enumerate(conn_list[i+1:], i+1):
                w2 = conn2.plasticity.get_combined_weight().data
                pre2 = int(conn2.pre_idx.item()) if hasattr(conn2.pre_idx, 'item') else int(conn2.pre_idx)
                post2 = int(conn2.post_idx.item()) if hasattr(conn2.post_idx, 'item') else int(conn2.post_idx)
                
                # Compute similarity
                if w1.shape == w2.shape:
                    sim = F.cosine_similarity(w1.flatten().unsqueeze(0), w2.flatten().unsqueeze(0))
                    if sim.item() > min_similarity:
                        analogies.append((pre1, post1, pre2, post2, sim.item()))
        
        return analogies
    
    def generate_hypotheses(
        self,
        current_context: Optional[int] = None,
        max_hypotheses: int = None,
    ) -> List['HypothesizedRelationship']:
        """
        Generate hypothesized relationships through recombination.
        """
        max_hyp = max_hypotheses or self.max_hypotheses_per_step
        hypotheses = []
        
        # 1. Composition: A→B, B→C → hypothesize A→C
        chains = self.find_compositional_chains()
        for chain in chains:
            if len(chain) >= 2:
                # A→B, B→C → hypothesize A→C
                pre = chain[0][0]
                mid = chain[0][1]
                post = chain[-1][1]
                
                if pre != post:  # Avoid self-loops
                    # Find connections by pre/post indices
                    conn1 = None
                    conn2 = None
                    for key, conn in self.network.connections.items():
                        pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
                        post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
                        if pre_idx == pre and post_idx == mid:
                            conn1 = conn
                        if pre_idx == mid and post_idx == post:
                            conn2 = conn
                    
                    if conn1 and conn2:
                        w1 = conn1.plasticity.get_combined_weight()
                        w2 = conn2.plasticity.get_combined_weight()
                        
                        # Compose: w_composed = w2 @ w1 (for linear)
                        hypothesized_weight = w2 @ w1
                        
                        # Check if connection already exists
                        exists = False
                        for key, conn in self.network.connections.items():
                            pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
                            post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
                            if pre_idx == pre and post_idx == post:
                                exists = True
                                break
                        
                        if not exists:
                            hyp = HypothesizedRelationship(
                                pre_idx=pre,
                                post_idx=post,
                                relationship_type="composition",
                                source_relationships=[(pre, mid), (mid, post)],
                                hypothesis_weight=hypothesized_weight,
                                confidence=0.7,
                                source_contexts=[None, None],
                                target_context=current_context,
                            )
                            hypotheses.append(hyp)
        
        # 2. Analogy: A→B in context X, C→D in context Y → A→D in context X
        analogies = self.find_analogical_pairs()
        for pre1, post1, pre2, post2, sim in analogies:
            if len(hypotheses) >= self.max_hypotheses_per_step:
                break
                
            # If pre1≈pre2 but post1≠post2, hypothesize cross-relationship
            if pre1 == pre2 and post1 != post2:
                # Same source, different targets - could indicate context-dependence
                hyp = HypothesizedRelationship(
                    pre_idx=pre1,
                    post_idx=post2,
                    relationship_type="analogy",
                    source_relationships=[(pre1, post1), (pre2, post2)],
                    hypothesis_weight=None,  # Would need to derive
                    confidence=sim * 0.5,
                    source_contexts=[None, None],
                    target_context=current_context,
                )
                hypotheses.append(hyp)
            
            # Also check reverse
            if post1 == post2 and pre1 != pre2:
                hyp = HypothesizedRelationship(
                    pre_idx=pre2,
                    post_idx=post1,
                    relationship_type="analogy",
                    source_relationships=[(pre1, post1), (pre2, post2)],
                    hypothesis_weight=None,
                    confidence=sim * 0.5,
                    source_contexts=[None, None],
                    target_context=current_context,
                )
                hypotheses.append(hyp)
        
        # 3. Context transfer: relationship in context A might work in B
        # (This would require context-specific weight storage)
        
        # Filter by confidence
        hypotheses = [h for h in hypotheses if h.confidence >= self.confidence_threshold]
        
        # Limit number
        return hypotheses[:max_hyp]
    
    def get_stats(self) -> Dict:
        return {
            "hypotheses_generated": self.hypotheses_generated.item(),
            "hypotheses_tested": self.hypotheses_tested.item(),
            "hypotheses_accepted": self.hypotheses_accepted.item(),
            "acceptance_rate": (
                self.hypotheses_accepted.item() / max(1, self.hypotheses_tested.item())
            ),
        }


def create_pattern_recombiner(
    network: 'RelationalNetwork',
    max_hypotheses: int = 10,
    composition_depth: int = 2,
    analogy_threshold: float = 0.7,
) -> PatternRecombiner:
    """Factory for pattern recombiner."""
    return PatternRecombiner(
        network=network,
        max_hypotheses_per_step=max_hypotheses,
        composition_depth=composition_depth,
        analogy_threshold=analogy_threshold,
    )