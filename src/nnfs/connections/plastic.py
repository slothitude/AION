"""
Plastic Relational Connections - Connections that learn online from experience.

Core mathematical framework:

a_ij(t) = θ_ij^slow + u_ij^fast(t)          # Dual-timescale relationship strength
m_ij(t) = a_ij(t) * ψ(h_i, h_j, c_t)        # Transmitted signal
ψ(h_i, h_j, c_t)                              # Learned relationship function

Prediction:  ĥ_j(t+1) = Σ_i m_ij(t)
Error:       δ_j = h_j(t+1) - ĥ_j(t+1)
Plasticity:  u_ij^fast ← u_ij^fast + η δ_j e_ij
Consolidation: θ_ij^slow ← θ_ij^slow + κ s_ij u_ij^fast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class EligibilityTrace(nn.Module):
    """
    Adaptive Eligibility Trace for credit assignment.
    
    Standard: e_ij(t) = γ e_ij(t-1) + h_i(t) h_j(t)
    
    Adaptive: e_ij(t) = β_t e_ij(t-1) + (1-β_t) f(x_pre, x_post)
    
    Where β_t depends on context and prediction uncertainty.
    This allows connections to retain meaningful relationships longer
    and forget temporary associations more quickly.
    
    At β=0.95, trace half-life ≈ 13.5 updates.
    At β=0.99, trace half-life ≈ 69 updates.
    """
    
    def __init__(
        self,
        pre_dim: int,
        post_dim: int,
        decay: float = 0.95,
        threshold: float = 1e-4,
        adaptive: bool = True,
        context_dim: int = 0,
    ):
        super().__init__()
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.base_decay = decay
        self.threshold = threshold
        self.adaptive = adaptive
        
        # Trace matrix: e_ij for each connection
        self.register_buffer("trace", torch.zeros(post_dim, pre_dim))
        self.register_buffer("pre_activity", torch.zeros(pre_dim))
        self.register_buffer("post_activity", torch.zeros(post_dim))
        self.register_buffer("context_activity", torch.zeros(1))
        
        # Adaptive decay predictor
        if adaptive and context_dim > 0:
            self.decay_predictor = nn.Sequential(
                nn.Linear(context_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.decay_predictor = None
        
    def compute_decay(self, context: Optional[torch.Tensor] = None) -> float:
        """Compute adaptive decay from context and uncertainty."""
        if not self.adaptive or self.decay_predictor is None or context is None:
            return self.base_decay
        
        ctx_mean = context.mean(dim=0) if context.dim() > 1 else context
        predicted_decay = self.decay_predictor(ctx_mean.unsqueeze(0)).squeeze()
        
        # Clamp to reasonable range
        return predicted_decay.clamp(0.8, 0.999).item()
    
    def update(
        self, 
        pre: torch.Tensor, 
        post: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ):
        """Update traces with current activity using adaptive decay."""
        # Compute adaptive decay
        decay = self.compute_decay(context)
        
        # Decay
        self.trace.mul_(decay)
        self.pre_activity.mul_(decay)
        self.post_activity.mul_(decay)
        
        # Add current activity (average over batch)
        pre_mean = pre.mean(dim=0) if pre.dim() > 1 else pre
        post_mean = post.mean(dim=0) if post.dim() > 1 else post
        
        # Update traces: e = γ e + (1-γ) * (post ⊗ pre)
        # Using (1-decay) factor for the new activity
        factor = 1.0 - decay
        self.trace.add_(factor * torch.outer(post_mean, pre_mean))
        self.pre_activity.add_(factor * pre_mean)
        self.post_activity.add_(factor * post_mean)
        
        if context is not None:
            ctx_mean = context.mean(dim=0) if context.dim() > 1 else context
            self.context_activity.add_(factor * ctx_mean.mean() if ctx_mean.dim() > 0 else factor * ctx_mean)
    
    def get_active_mask(self) -> torch.Tensor:
        """Get mask of active traces (above threshold)."""
        return self.trace.abs() > self.threshold
    
    def reset(self):
        """Reset all traces to zero."""
        self.trace.zero_()
        self.pre_activity.zero_()
        self.post_activity.zero_()
        self.context_activity.zero_()
    
    def get_stats(self) -> Dict:
        """Get trace statistics."""
        return {
            "mean": self.trace.abs().mean().item(),
            "max": self.trace.abs().max().item(),
            "active_fraction": (self.trace.abs() > self.threshold).float().mean().item(),
            "active_count": (self.trace.abs() > self.threshold).sum().item(),
        }


class RelationshipFunction(nn.Module):
    """
    Learnable relationship function ψ(h_i, h_j, c_t).
    
    Computes a per-connection modulation factor for each pre-post pair.
    Output shape: (batch, post_dim, pre_dim) - modulation per connection.
    """
    
    def __init__(
        self,
        pre_dim: int,
        post_dim: int,
        context_dim: int = 0,
        hidden_dim: int = 32,
        relation_type: str = "bilinear",  # "bilinear", "post", "mlp", "attention"
    ):
        super().__init__()
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.context_dim = context_dim
        self.relation_type = relation_type

        if relation_type == "bilinear":
            # ψ_ij = h_j^T W h_i * context_gate
            # W: (post_dim, pre_dim)
            self.W = nn.Parameter(torch.empty(post_dim, pre_dim))
            self.context_gate = nn.Linear(context_dim, post_dim) if context_dim > 0 else None
            nn.init.xavier_uniform_(self.W, gain=0.1)

        elif relation_type == "post":
            # Post-side relationship: psi_ij = h_i * W_ij, broadcast over pre.
            # Gives each output dim a learnable self-channel via the receiving
            # neuron's own activity. "bilinear" multiplies psi by the pre
            # activity; with untrained W and zero-mean states that factor
            # zeroes the EXPECTED error-driven gradient for self-prediction
            # targets (pre is an independent neuron's state), leaving u_fast
            # on an unbiased random walk.
            self.W = nn.Parameter(torch.empty(post_dim, pre_dim))
            self.context_gate = nn.Linear(context_dim, post_dim) if context_dim > 0 else None
            nn.init.xavier_uniform_(self.W, gain=1.0)
        elif relation_type == "mlp":
            # ψ = MLP([h_i; h_j; c_t]) -> (post_dim, pre_dim)
            input_dim = pre_dim + post_dim + context_dim
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, post_dim * pre_dim)
            )
            
        elif relation_type == "attention":
            # Attention-based: Q(post) * K(pre)^T
            self.q_proj = nn.Linear(post_dim, hidden_dim)
            self.k_proj = nn.Linear(pre_dim, hidden_dim)
            self.scale = math.sqrt(hidden_dim)
        
    def forward(
        self, 
        pre: torch.Tensor, 
        post: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute relationship ψ(h_i, h_j, c_t).
        
        Args:
            pre: Pre-synaptic activity (batch, pre_dim)
            post: Post-synaptic activity (batch, post_dim)
            context: Context signal (batch, context_dim)
            
        Returns:
            Relationship output (batch, post_dim, pre_dim) - per-connection modulation
        """
        batch_size = pre.shape[0]
        
        if self.relation_type == "bilinear":
            # ψ_ij = (h_j^T W) ⊗ h_i  -> (B, post_dim, pre_dim)
            # post @ W: (B, post_dim, pre_dim)  -- wait, W is (post_dim, pre_dim)
            # post: (B, post_dim) -> (B, post_dim, 1)
            # W: (post_dim, pre_dim)
            # Result: (B, post_dim, pre_dim)
            # Actually: for each post neuron j, we want post[j] * W[j] * pre[i]
            # So: post.unsqueeze(-1) @ W.unsqueeze(0) @ pre.unsqueeze(1)
            # Or simpler: post.unsqueeze(-1) * (W @ pre.unsqueeze(-1)).squeeze(-1)
            
            # post: (B, post_dim), W: (post_dim, pre_dim), pre: (B, pre_dim)
            # We want (B, post_dim, pre_dim)
            # For each batch: post_j * W_jk * pre_i
            W_expanded = self.W.unsqueeze(0).expand(batch_size, -1, -1)  # (B, post, pre)
            post_expanded = post.unsqueeze(-1)  # (B, post, 1)
            pre_expanded = pre.unsqueeze(1)  # (B, 1, pre)
            
            rel = post_expanded * W_expanded * pre_expanded  # (B, post, pre)
            
            if context is not None and self.context_gate is not None:
                gate = torch.sigmoid(self.context_gate(context))  # (B, post_dim)
                rel = rel * gate.unsqueeze(-1)  # (B, post, pre)
            return rel

        elif self.relation_type == "post":
            # psi = post * W -> (B, post_dim, pre_dim), constant along pre
            W_expanded = self.W.unsqueeze(0).expand(batch_size, -1, -1)
            rel = post.unsqueeze(-1) * W_expanded
            if context is not None and self.context_gate is not None:
                gate = torch.sigmoid(self.context_gate(context))  # (B, post_dim)
                rel = rel * gate.unsqueeze(-1)
            return rel

        elif self.relation_type == "mlp":
            inputs = [pre, post]
            if context is not None:
                inputs.append(context)
            x = torch.cat(inputs, dim=-1)
            out = self.mlp(x)  # (B, post_dim * pre_dim)
            return out.view(batch_size, self.post_dim, self.pre_dim)
            
        elif self.relation_type == "attention":
            # Q(post) * K(pre)^T -> (B, post_dim, pre_dim)
            query = self.q_proj(post)  # (B, H)
            key = self.k_proj(pre)     # (B, H)
            # For full pairwise: need to project pre to (B, pre_dim, H)
            # Actually simpler: use separate projections per pre_dim
            # For now, use a simplified version
            key_full = self.k_proj(pre).unsqueeze(1)  # (B, 1, H)
            query_full = self.q_proj(post).unsqueeze(-1)  # (B, H, 1)
            # This is getting complex - return simple bilinear for now
            return torch.ones(batch_size, self.post_dim, self.pre_dim, device=pre.device)
            
        return torch.ones(batch_size, self.post_dim, self.pre_dim, device=pre.device)


class FastSlowPlasticity(nn.Module):
    """
    Dual-timescale plasticity mechanism with selective consolidation.
    
    Slow weights: θ_slow - consolidated, stable knowledge
    Fast weights: u_fast - rapid adaptation from experience
    
    Consolidation: θ_slow ← θ_slow + α * g_ij * (u_fast - θ_slow)
    Where g_ij is a per-connection consolidation gate based on evidence of stability.
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        learning_rate: float = 0.01,
        consolidation_rate: float = 0.001,
        consolidation_threshold: float = 0.1,
        weight_decay: float = 1e-5,
        max_weight: float = 10.0,
        selective_consolidation: bool = True,
        blend_ratio: float = 0.7,
        transfer_fraction: float = 0.1,
        blend_anneal: float = 0.05,
        blend_floor: float = 0.5,
    ):
        super().__init__()
        self.shape = shape
        self.learning_rate = learning_rate
        self.consolidation_rate = consolidation_rate
        self.consolidation_threshold = consolidation_threshold
        self.weight_decay = weight_decay
        self.max_weight = max_weight
        self.selective_consolidation = selective_consolidation
        self.blend_ratio = blend_ratio
        self.transfer_fraction = transfer_fraction
        self.blend_anneal = blend_anneal
        self.blend_floor = blend_floor
        
        # Fast weights (rapidly adapting)
        self.u_fast = nn.Parameter(torch.zeros(*shape))
        
        # Slow weights (consolidated knowledge)
        self.theta_slow = nn.Parameter(torch.zeros(*shape), requires_grad=False)
        
        # Consolidation gate: learns which connections are stable
        if selective_consolidation:
            self.consolidation_gate = nn.Parameter(torch.ones(*shape) * 0.5)
            # Evidence accumulators for consolidation decision
            self.register_buffer("stability_evidence", torch.zeros(*shape))
            self.register_buffer("evidence_count", torch.zeros(*shape))
        else:
            self.register_parameter("consolidation_gate", None)
            self.register_buffer("stability_evidence", None)
            self.register_buffer("evidence_count", None)
        
    def plasticity_step(
        self,
        error: torch.Tensor,
        eligibility: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Update fast weights from prediction error.

        Δu_fast = η * error * eligibility

        With eligibility = psi (batch, post, pre) and error = y - y_hat
        (batch, post), the first branch is exact gradient descent on the
        actual forward loss L = 0.5*mean((y - y_hat)^2):
            u_fast <- u_fast + blend_ratio * lr * mean_batch(delta * psi)
        For L = 0.5*(y-y_hat)^2, dL/du = -blend_ratio * delta * psi, so the
        blend factor (0.7) enters the gradient explicitly, exactly once.
        """
        stats = {}

        with torch.no_grad():
            # Modulate error by eligibility traces
            if error.dim() == 2 and eligibility.dim() == 3:
                # error: (batch, post), eligibility: (batch, post, pre)
                # Batch-mean of the per-sample product = gradient of the mean loss
                modulated = (error.unsqueeze(-1) * eligibility).mean(dim=0)
            elif error.dim() == 1 and eligibility.dim() == 2:
                # error: (post_dim,), eligibility: (post_dim, pre_dim)
                modulated = error.unsqueeze(1) * eligibility
            elif error.dim() == 1 and eligibility.dim() == 1:
                modulated = error.unsqueeze(1) * eligibility.unsqueeze(0)
            else:
                modulated = error * eligibility

            # Apply only to active connections
            if active_mask is not None:
                modulated = modulated * active_mask.float()

            # Apply learning rate with the explicit J_u factor (blend_ratio)
            self.u_fast.add_(self.blend_ratio * self.learning_rate * modulated)

            # Weight decay
            self.u_fast.mul_(1 - self.weight_decay)
            self.u_fast.clamp_(-10.0, 10.0)

            stats["weight_change"] = (
                self.blend_ratio * self.learning_rate * modulated.abs()
            ).mean().item()
            stats["u_fast_norm"] = self.u_fast.norm().item()

        return stats
    
    def consolidate(
        self,
        active_mask: Optional[torch.Tensor] = None,
        blend_ratio: Optional[float] = None,
    ) -> Dict:
        """
        Output-preserving consolidation (Fix 3 + blend-anneal deadlock fix).

        Per-connection transfer amount d = transfer_fraction * u_fast on
        masked entries. Given the (possibly annealed) blend b', the slow
        weights are solved EXACTLY from a-preservation:

            a_before = b * u_fast + (1 - b) * theta      (before any change)
            u'       = u_fast - d
            theta'   = (a_before - b' * u') / (1 - b')   (elementwise)

        which leaves the effective connection
            a = blend * u_fast + (1 - blend) * theta
        unchanged for EVERY entry (masked and unmasked), within float
        precision, so transmitted output y_hat is invariant.

        Blend annealing breaks the blend=1.0 deadlock: at b = 1 the slow
        weights have zero weight in a, so transfer is impossible while b
        stays 1. Each consolidation event that transfers also anneals
        b -> max(blend_floor, b - blend_anneal); a is preserved through the
        anneal by the same solve (theta absorbs the blend change).
        """
        stats = {}

        blend = self.blend_ratio if blend_ratio is None else blend_ratio

        with torch.no_grad():
            if self.consolidation_rate <= 0.0:
                # consolidation_rate == 0 means consolidation is DISABLED for
                # this connection (e.g. Plastic-D0 controls); the interval
                # alone must not fire sweeps. Otherwise rate==0 variants
                # consolidate identically to consolidating variants and the
                # control is invalid.
                stats["consolidated"] = 0
                stats["consolidation_disabled"] = 1
                stats["blend_after"] = blend
                return stats

            if blend >= 1.0 - 1e-9 and self.blend_anneal <= 0.0:
                # theta has zero weight in a and annealing is disabled: no
                # transfer can preserve the effective connection.
                stats["consolidated"] = 0
                stats["skipped_blend_ge_1"] = 1
                return stats

            # Determine which connections to consolidate
            if active_mask is None:
                # Use consolidation threshold on fast weights
                consolidate_mask = self.u_fast.abs() > self.consolidation_threshold
            else:
                consolidate_mask = active_mask

            if self.selective_consolidation and self.consolidation_gate is not None:
                # Per-connection consolidation gates
                # g_ij = sigmoid(consolidation_gate) ∈ (0, 1)
                consolidation_gates = torch.sigmoid(self.consolidation_gate)

                # Only consolidate where gate is open AND mask is active
                consolidate_mask = consolidate_mask & (consolidation_gates > 0.5)

                stats["gated_consolidated"] = int(consolidate_mask.sum().item())
                stats["gate_mean"] = consolidation_gates.mean().item()
                stats["gate_open_fraction"] = (consolidation_gates > 0.5).float().mean().item()

            stats["consolidated"] = int(consolidate_mask.sum().item())

            if not consolidate_mask.any():
                stats["theta_slow_norm"] = self.theta_slow.norm().item()
                stats["blend_after"] = blend
                return stats

            # Exact a-preserving solve (handles transfer AND blend anneal)
            blend_after = blend
            if blend - self.blend_anneal >= self.blend_floor:
                blend_after = blend - self.blend_anneal
            a_before = blend * self.u_fast + (1.0 - blend) * self.theta_slow
            d = torch.zeros_like(self.u_fast)
            d[consolidate_mask] = self.transfer_fraction * self.u_fast[consolidate_mask]
            u_new = self.u_fast - d
            theta_new = (a_before - blend_after * u_new) / (1.0 - blend_after)
            self.u_fast.copy_(u_new)
            self.theta_slow.copy_(theta_new)

            stats["blend_before"] = blend
            stats["blend_after"] = blend_after
            stats["transferred_abs_mean"] = d.abs().mean().item()
            stats["consolidation_rate"] = self.consolidation_rate
            stats["theta_slow_norm"] = self.theta_slow.norm().item()

        return stats

    def update_evidence(self, eligibility: torch.Tensor, error_magnitude: float = None):
        """
        Update stability evidence for consolidation gates (deadlock fix).

        The gate encodes evidence of STABILITY, not absolute error level.
        Statistic over the coactivity magnitude |e|:
            stability = mean(|e|)^2 / mean(|e|^2)   in [0, 1]  (= 1/(1+CV^2))
        1 = steadily exercised relationship; -> 0 for intermittent exercise
        (bursts and silences) or absent coactivity. Scale-free, so
        consolidation is reachable in realistic O(1)-error regimes. The
        previous mapping folded 1/(1+error) into the evidence, which pinned
        gates near 0 whenever error stayed O(1) and consolidation never
        fired (gate-6 pilot: gate_mean 0.0697, transfers 0). A signed-mean
        variant was also rejected: real-network trace entries fluctuate
        around zero, so mean(e)^2/mean(e^2) ~ 0 for EVERY entry and gates
        never open. Noise/dead entries are excluded by the consolidation
        activity mask (trace threshold), not by the gate.
        `error_magnitude` is kept in the signature for caller compatibility
        and intentionally does not influence the gate.
        """
        if not self.selective_consolidation or self.stability_evidence is None:
            return

        with torch.no_grad():
            strength = eligibility.detach().abs().float()
            # Running mean and second moment of the magnitude (0.99 EMA)
            self.stability_evidence.mul_(0.99).add_(0.01 * strength)
            self.evidence_count.mul_(0.99).add_(0.01 * strength.square())
            mean = self.stability_evidence
            m2 = self.evidence_count
            stability = (mean.square() / (m2 + 1e-12)).clamp_(0.0, 1.0)
            # Gate opens (> 0.5) once stability exceeds ~0.5
            target_gate = torch.sigmoid((stability - 0.5) * 10.0)
            self.consolidation_gate.data = torch.logit(
                torch.sigmoid(self.consolidation_gate) * 0.99 + 0.01 * target_gate
            ).clamp(-10, 10)
    
    def get_combined_weight(self, blend_ratio: float = 0.7) -> torch.Tensor:
        """
        Get combined weight: blend_ratio * u_fast + (1-blend_ratio) * θ_slow
        """
        return blend_ratio * self.u_fast + (1 - blend_ratio) * self.theta_slow
    
    def reset_fast(self):
        """Reset fast weights (e.g., between episodes)."""
        self.u_fast.zero_()
    
    def get_stats(self) -> Dict:
        stats = {
            "u_fast_norm": self.u_fast.norm().item(),
            "theta_slow_norm": self.theta_slow.norm().item(),
            "u_fast_max": self.u_fast.abs().max().item(),
            "theta_slow_max": self.theta_slow.abs().max().item(),
        }
        if self.selective_consolidation:
            stats["gate_mean"] = torch.sigmoid(self.consolidation_gate).mean().item()
            stats["gate_open"] = (torch.sigmoid(self.consolidation_gate) > 0.5).float().mean().item()
        return stats


class ContextualRelation(nn.Module):
    """
    Context-conditioned relationship.
    
    Learns when a relationship should be active:
    a_ij = θ_slow + u_fast * gate(c_t)
    Where gate(c_t) ∈ [0, 1] determines relationship strength in current context.
    """
    
    def __init__(
        self,
        pre_dim: int,
        post_dim: int,
        context_dim: int,
        num_contexts: int = 10,
    ):
        super().__init__()
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.context_dim = context_dim
        self.num_contexts = num_contexts
        
        # Context classifier: which context regime are we in?
        self.context_classifier = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_contexts)
        )
        
        # Per-context relationship modulators
        self.context_gates = nn.Parameter(torch.ones(num_contexts, post_dim, 1))
        self.context_biases = nn.Parameter(torch.zeros(num_contexts, post_dim, 1))
        
        # Slow relationship weights
        self.theta_slow = nn.Parameter(torch.zeros(post_dim, post_dim), requires_grad=False)
        self.u_fast = nn.Parameter(torch.zeros(post_dim, post_dim))
        
    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contextual relationship.
        
        Returns:
            modulated_strength: (batch, post_dim, post_dim) - relationship matrix
            context_probs: (batch, num_contexts) - context classification
        """
        # Classify context
        context_logits = self.context_classifier(context)  # (batch, num_contexts)
        context_probs = F.softmax(context_logits, dim=-1)
        
        # Compute context-dependent gates
        # gates: (batch, post_dim, 1)
        gates = torch.einsum('bn,nji->bji', context_probs, self.context_gates)
        biases = torch.einsum('bn,ni->bni', context_probs, self.context_biases)
        
        # Combined weight: slow + fast * context_gate
        combined = self.theta_slow.unsqueeze(0) + self.u_fast.unsqueeze(0) * gates + biases
        
        return combined, context_probs


class StructuralPlasticity(nn.Module):
    """
    Structural plasticity - creation and removal of connections.
    
    Experimental: dynamically creates/removes connections based on utility.
    """
    
    def __init__(
        self,
        num_neurons: int,
        neuron_dim: int,
        max_connections: int = None,
        creation_threshold: float = 0.01,
        deletion_threshold: float = 0.001,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.max_connections = max_connections or (num_neurons * 4)
        self.creation_threshold = creation_threshold
        self.deletion_threshold = deletion_threshold
        
        # Connection utility tracker
        self.register_buffer("connection_utility", torch.zeros(num_neurons, num_neurons))
        self.register_buffer("connection_exists", torch.zeros(num_neurons, num_neurons, dtype=torch.bool))
        
    def update_utility(self, connections: Dict, eligibility_traces: Dict):
        """Update connection utility from activity."""
        for key, conn in connections.items():
            if hasattr(conn, 'eligibility_trace'):
                trace = conn.eligibility_trace.trace
                self.connection_utility[conn.post_idx, conn.pre_idx] += trace.abs().mean().item()
    
    def prune_and_grow(self, connections: Dict) -> Dict[str, int]:
        """Remove weak connections, create new ones."""
        stats = {"pruned": 0, "created": 0}
        
        # Prune weak connections
        if self.connection_utility.sum() > 0:
            norm_utility = self.connection_utility / self.connection_utility.max()
            prune_mask = (norm_utility < self.deletion_threshold) & self.connection_exists
            # Implementation would remove connections here
            stats["pruned"] = int(prune_mask.sum().item())
            
        # Create new connections where utility is high
        grow_mask = self.connection_utility > self.creation_threshold
        # Implementation would add connections here
        stats["created"] = int(grow_mask.sum().item())
        
        return stats


class PlasticRelationalConnection(nn.Module):
    """
    Complete plastic relational connection.
    
    Combines:
    - Relationship function ψ(h_i, h_j, c_t)
    - Dual-timescale plasticity (FastSlowPlasticity)
    - Eligibility traces
    - Prediction error driven learning
    """
    
    def __init__(
        self,
        pre_idx: int,
        post_idx: int,
        pre_dim: int,
        post_dim: int,
        context_dim: int = 0,
        relation_type: str = "bilinear",
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        trace_decay: float = 0.95,
        trace_threshold: float = 1e-4,
        consolidation_rate: float = 0.001,
        consolidation_threshold: float = 0.1,
        weight_decay: float = 1e-5,
        max_weight: float = 10.0,
        blend_ratio: float = 0.7,
        consolidation_interval: int = 1000,
        transfer_fraction: float = 0.1,
        blend_anneal: float = 0.05,
        blend_floor: float = 0.5,
    ):
        super().__init__()
        self.pre_idx = pre_idx
        self.post_idx = post_idx
        self.pre_dim = pre_dim
        self.post_dim = post_dim
        self.context_dim = context_dim
        self.blend_ratio = blend_ratio
        self.consolidation_interval = consolidation_interval

        # Relationship function
        self.relationship = RelationshipFunction(
            pre_dim, post_dim, context_dim=context_dim,
            hidden_dim=hidden_dim, relation_type=relation_type
        )

        # Dual-timescale plasticity with selective consolidation
        self.plasticity = FastSlowPlasticity(
            shape=(post_dim, pre_dim),
            learning_rate=learning_rate,
            consolidation_rate=consolidation_rate,
            consolidation_threshold=consolidation_threshold,
            selective_consolidation=True,
            blend_ratio=blend_ratio,
            transfer_fraction=transfer_fraction,
            blend_anneal=blend_anneal,
            blend_floor=blend_floor,
        )
        
        # Eligibility trace (adaptive)
        self.eligibility_trace = EligibilityTrace(
            pre_dim, post_dim, 
            decay=trace_decay, 
            threshold=trace_threshold,
            adaptive=True,
            context_dim=context_dim
        )
        
        # Statistics
        self.register_buffer("update_count", torch.tensor(0))
        self.register_buffer("total_weight_change", torch.tensor(0.0))
        
    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        learn: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with optional online learning.
        
        Args:
            pre: Pre-synaptic activity (batch, pre_dim)
            post: Post-synaptic activity (batch, post_dim)
            context: Context signal (batch, context_dim)
            target: Target next state for prediction error (batch, post_dim)
            learn: Whether to update traces and plasticity
            
        Returns:
            transmitted: Transmitted signal (batch, post_dim)
            stats: Dictionary of statistics
        """
        stats = {}
        
        # Get combined weight (fast + slow)
        weight = self.plasticity.get_combined_weight(self.blend_ratio)
        
        # Compute relationship
        psi = self.relationship(pre, post, context)
        
        # Transmitted signal: a_ij * ψ
        transmitted = weight.unsqueeze(0) * psi
        
        stats["psi_norm"] = psi.norm(dim=-1).mean().item()
        stats["weight_norm"] = weight.norm().item()
        
        if learn:
            # Update coactivity eligibility trace (masks / consolidation gates
            # / relationship statistics only - NOT the error-driven update)
            self.eligibility_trace.update(pre, post, context)
            trace_stats = self.eligibility_trace.get_stats()
            stats.update({f"trace_{k}": v for k, v in trace_stats.items()})

            # Apply plasticity if target provided
            if target is not None:
                # Prediction uses the ACTUAL forward output (Fix 2):
                # y_hat_j = sum_i a_ij * psi_ij  == transmitted.sum(-1) above
                predicted = transmitted.sum(dim=-1)  # (batch, post_dim)

                # Prediction error: δ = target - predicted
                # SIGN CONVENTION: error = target - predicted
                # Positive error means prediction was too low, need to increase weights
                error = target - predicted  # (batch, post_dim)

                # Uncertainty estimate: variance of error across batch
                if error.shape[0] > 1:
                    error_std = error.std(dim=0).mean().item()
                else:
                    error_std = error.abs().mean().item()  # Use mean abs error for single sample
                stats["error_uncertainty"] = error_std

                # Uncertainty scale is REPORTED only; the update is exact
                # gradient descent on the actual loss (Fix 2), so no scaling.
                uncertainty_scale = 1.0 / (1.0 + error_std)
                stats["uncertainty_scale"] = uncertainty_scale

                # Plasticity: u_fast += blend * lr * mean_batch(delta * psi)
                # psi is the per-step forward-path derivative; the coactivity
                # trace no longer enters the error-driven update. The update
                # mask is forward participation (|psi| this step) - masking by
                # the coactivity trace zeroes zero-mean directions and breaks
                # gradient descent exactly in the signed-post regime.
                if psi.dim() == 3:
                    active_mask = (psi.abs().mean(dim=0) > 1e-4).float()
                else:
                    active_mask = None
                plasticity_stats = self.plasticity.plasticity_step(
                    error,   # (batch, post_dim)
                    psi,     # (batch, post_dim, pre_dim) forward derivative
                    active_mask
                )
                stats.update({f"plasticity_{k}": v for k, v in plasticity_stats.items()})

                # Update evidence for selective consolidation
                self.plasticity.update_evidence(
                    self.eligibility_trace.trace,
                    error_std
                )

                # Update statistics
                stats["error_norm"] = error.norm(dim=-1).mean().item()
                stats["prediction_mse"] = (error ** 2).mean().item()

            # Periodic consolidation (interval configurable, Fix 3)
            if (self.consolidation_interval > 0
                    and self.update_count % self.consolidation_interval == 0
                    and self.update_count > 0):
                cons_stats = self.plasticity.consolidate(blend_ratio=self.blend_ratio)
                if "blend_after" in cons_stats:
                    self.blend_ratio = cons_stats["blend_after"]
                stats.update({f"consolidation_{k}": v for k, v in cons_stats.items()})

            self.update_count.add_(1)
        
        return transmitted.sum(dim=-1), stats  # Sum over pre_dim -> (batch, post_dim)
    
    def consolidate(self) -> Dict:
        """Manual consolidation call (output-preserving, Fix 3)."""
        active_mask = self.eligibility_trace.get_active_mask()
        stats = self.plasticity.consolidate(active_mask, blend_ratio=self.blend_ratio)
        if "blend_after" in stats:
            self.blend_ratio = stats["blend_after"]
        return stats
    
    def reset_traces(self):
        """Reset eligibility traces."""
        self.eligibility_trace.reset()
    
    def get_state_dict(self) -> Dict:
        """Get full state for checkpointing."""
        return {
            "relationship": self.relationship.state_dict(),
            "plasticity": {
                "u_fast": self.plasticity.u_fast.data.clone(),
                "theta_slow": self.plasticity.theta_slow.data.clone(),
            },
            "eligibility_trace": self.eligibility_trace.trace.clone(),
            "stats": {
                "updates": self.update_count.item(),
                "total_weight_change": self.total_weight_change.item(),
            }
        }
    
    def extra_repr(self) -> str:
        return (f"pre={self.pre_dim}, post={self.post_dim}, "
                f"blend={self.blend_ratio}, lr={self.plasticity.learning_rate}")


def create_plastic_connection(
    pre_idx: int,
    post_idx: int,
    pre_dim: int,
    post_dim: int,
    **kwargs
) -> PlasticRelationalConnection:
    """Factory for plastic connections."""
    return PlasticRelationalConnection(
        pre_idx=pre_idx,
        post_idx=post_idx,
        pre_dim=pre_dim,
        post_dim=post_dim,
        **kwargs
    )