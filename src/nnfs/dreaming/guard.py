"""
Composed-weight norm guard (spec DREAM_REIMPL_SPEC.md §3 E4, owner condition 2).

Rejects a composed hypothesis BEFORE measurement when its weight is
structurally implausible -- the cheap first filter so the expensive
measurement loop is never spent on composition artifacts.

Two sanity checks, both required to pass:

1. Product-of-norms sanity: the composed weight must not exceed the
   product of its source factors' norms by more than ``max_product_ratio``.
   In exact arithmetic ||w2 @ w1|| <= ||w2|| * ||w1||, so a ratio above 1
   (beyond float slack) means the weight is corrupted (NaN/inf propagation,
   a scaling bug), not discovered.

2. Network-scale sanity ("the sane band"): the composed norm must not exceed
   ``max_scale_ratio`` times the largest existing connection weight norm.
   Matrix products can legitimately reach at most the product of norms, but
   a product far outside the network's own weight scale is a composition
   artifact (the pilot wrote ||w2 @ w1|| ~= 9.81 into theta against
   connection weights of O(0.1-1)), not a discovery.

Owner condition 2: the guard must be CALIBRATED against valid compositions
as well as pathological ones -- :func:`calibrate_guard` builds a
deterministic valid/pathological set, runs the guard, and returns the
calibration table. A guard that rejects legitimate discoveries is another
failure, so the calibration asserts margins: valid cases accept with room
below the thresholds, pathological cases reject well above them.
"""

import math
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class GuardVerdict:
    """Outcome of the composed-weight norm guard for one hypothesis."""
    accepted: bool
    reason: str
    composed_norm: float
    product_of_norms: Optional[float]      # None when no composition factors
    product_ratio: Optional[float]         # ||composed|| / (||w1||*||w2||)
    reference_scale: float                 # max existing connection weight norm
    scale_ratio: float                     # ||composed|| / reference_scale
    detail: str = ""

    def as_dict(self) -> Dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "composed_norm": self.composed_norm,
            "product_of_norms": self.product_of_norms,
            "product_ratio": self.product_ratio,
            "reference_scale": self.reference_scale,
            "scale_ratio": self.scale_ratio,
            "detail": self.detail,
        }


def _connection_norms(network) -> List[float]:
    norms = []
    for conn in network.connections.values():
        w = conn.plasticity.get_combined_weight().detach()
        norms.append(float(w.norm()))
    return norms


class ComposedWeightNormGuard:
    """Hallucination guard for composed hypothesis weights (spec §3 E4)."""

    def __init__(
        self,
        network,
        max_product_ratio: float = 1.001,
        max_scale_ratio: float = 10.0,
        scale_floor: float = 1.0,
    ):
        self.network = network
        self.max_product_ratio = max_product_ratio
        self.max_scale_ratio = max_scale_ratio
        # Fresh networks have all-zero weights; the scale reference falls back
        # to this floor so an empty network does not reject everything.
        self.scale_floor = scale_floor

    # ------------------------------------------------------------------
    def reference_scale(self) -> float:
        norms = _connection_norms(self.network)
        return max(norms) if norms and max(norms) > 0 else self.scale_floor

    def composition_factors(self, hypothesis) -> List[torch.Tensor]:
        """Combined weights of the hypothesis's source connections."""
        factors = []
        connections = self.network.connections
        for (pre, post) in hypothesis.source_relationships:
            for conn in connections.values():
                pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
                post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
                if pre_idx == pre and post_idx == post:
                    factors.append(conn.plasticity.get_combined_weight().detach())
                    break
        return factors

    def check(self, hypothesis, factors: Optional[List[torch.Tensor]] = None) -> GuardVerdict:
        w = hypothesis.hypothesis_weight
        if w is None:
            return GuardVerdict(
                accepted=False, reason="no_weight", composed_norm=0.0,
                product_of_norms=None, product_ratio=None,
                reference_scale=self.reference_scale(), scale_ratio=0.0,
                detail="hypothesis carries no weight tensor",
            )
        w = w.detach()

        reference = self.reference_scale()
        composed_norm = float(w.norm())

        if not torch.isfinite(w).all() or not math.isfinite(composed_norm):
            return GuardVerdict(
                accepted=False, reason="non_finite", composed_norm=composed_norm,
                product_of_norms=None, product_ratio=None,
                reference_scale=reference, scale_ratio=float('inf'),
                detail="composed weight contains NaN/inf",
            )

        scale_ratio = composed_norm / reference

        product_of_norms = None
        product_ratio = None
        if hypothesis.relationship_type == "composition":
            if factors is None:
                factors = self.composition_factors(hypothesis)
            if factors:
                product_of_norms = 1.0
                for f in factors:
                    product_of_norms *= float(f.norm())
                product_ratio = composed_norm / max(product_of_norms, 1e-12)

        # Check 1: product-of-norms sanity (corruption detector).
        if product_ratio is not None and product_ratio > self.max_product_ratio:
            return GuardVerdict(
                accepted=False, reason="product_norm_exceeded",
                composed_norm=composed_norm, product_of_norms=product_of_norms,
                product_ratio=product_ratio, reference_scale=reference,
                scale_ratio=scale_ratio,
                detail=(f"||composed||/{product_of_norms:.6g} = {product_ratio:.4g} "
                        f"exceeds max_product_ratio {self.max_product_ratio}"),
            )

        # Check 2: sane-band / network-scale sanity (artifact detector).
        if scale_ratio > self.max_scale_ratio:
            return GuardVerdict(
                accepted=False, reason="outside_sane_band",
                composed_norm=composed_norm, product_of_norms=product_of_norms,
                product_ratio=product_ratio, reference_scale=reference,
                scale_ratio=scale_ratio,
                detail=(f"||composed||/{reference:.6g} = {scale_ratio:.4g} "
                        f"exceeds max_scale_ratio {self.max_scale_ratio}"),
            )

        detail = (f"scale_ratio={scale_ratio:.4g} <= {self.max_scale_ratio}")
        if product_ratio is not None:
            detail += f"; product_ratio={product_ratio:.4g} <= {self.max_product_ratio}"
        return GuardVerdict(
            accepted=True, reason="passed", composed_norm=composed_norm,
            product_of_norms=product_of_norms, product_ratio=product_ratio,
            reference_scale=reference, scale_ratio=scale_ratio, detail=detail,
        )


# ----------------------------------------------------------------------
# Calibration (owner condition 2)
# ----------------------------------------------------------------------
def _compose(w2: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    return w2 @ w1


def calibrate_guard(network=None, seed: int = 0, verbose: bool = False) -> Dict:
    """Calibrate the guard against valid AND pathological compositions.

    Builds a deterministic case set at the scale of the given network
    (a fresh 16-neuron/16-dim network when none supplied), runs the guard on
    each case and returns the calibration table plus pass/fail margins:

    Valid cases (must be ACCEPTED -- rejecting a legitimate discovery is
    itself a failure, owner condition 2):
      - identity chain: w2 @ (c*I) recovers a scaled known weight
      - aligned moderate pair: smooth factors whose product sits at network
        weight scale
      - small-norm pair: product well below network scale

    Pathological cases (must be REJECTED):
      - misaligned oversized pair: Gaussian factors far above network scale
        whose product lands far outside the sane band
      - near-singular blow-up pair: ill-conditioned factors aligned in their
        top singular directions
      - corrupted weight: product-of-norms ratio above 1 (NaN/inf/scaling)
    """
    import torch as _torch

    if network is None:
        from nnfs.connections.network import RelationalNetwork
        network = RelationalNetwork(num_neurons=16, neuron_dim=16, sparsity=0.15,
                                    relation_type="post", learning_rate=0.05,
                                    blend_ratio=1.0)
        # Give the network a realistic live-scale weight profile.
        generator = _torch.Generator().manual_seed(seed)
        with _torch.no_grad():
            for conn in network.connections.values():
                conn.plasticity.u_fast.add_(
                    0.5 * _torch.randn(conn.plasticity.u_fast.shape, generator=generator))
    else:
        generator = _torch.Generator().manual_seed(seed)

    guard = ComposedWeightNormGuard(network)
    dim = network.neuron_dim
    reference = guard.reference_scale()

    def factors(norm_target: float, aligned: bool, cond: float = 1.0):
        # Deterministic pair with approximate Frobenius norm norm_target.
        a = _torch.randn(dim, dim, generator=generator)
        b = _torch.randn(dim, dim, generator=generator)
        if aligned:
            b = a.t().contiguous()  # top singular directions aligned
        if cond != 1.0:
            u, s, vt = _torch.svd(a)
            s = _torch.linspace(cond, 1.0, dim)
            a = u @ _torch.diag(s) @ vt
        a = a / a.norm() * norm_target
        b = b / b.norm() * norm_target
        return a, b

    identity = _torch.eye(dim)
    target_weight = 0.5 * _torch.randn(dim, dim, generator=generator)

    valid_cases = [
        ("valid_identity_chain", [0.5 * identity, target_weight]),
        ("valid_aligned_moderate", list(factors(reference, aligned=True))),
        ("valid_small_norm_pair", list(factors(0.25 * reference, aligned=False))),
    ]

    pathological_cases = [
        ("path_misaligned_oversized", list(factors(30.0 * reference, aligned=False))),
        ("path_near_singular_blowup", list(factors(8.0 * reference, aligned=True, cond=100.0))),
        ("path_corrupted_norm", list(factors(reference, aligned=False))),
    ]

    table = []

    for name, factor_list in valid_cases + pathological_cases:
        expected_valid = name.startswith("valid")

        w1, w2 = factor_list[0], factor_list[1]
        composed = _compose(w2, w1)
        if name == "path_corrupted_norm":
            composed = 1000.0 * composed

        class _Hyp:
            pass

        hyp = _Hyp()
        hyp.hypothesis_weight = composed
        hyp.relationship_type = "composition"
        hyp.source_relationships = [(0, 1), (1, 2)]  # factors supplied explicitly

        verdict = guard.check(hyp, factors=[w1, w2])
        row = {
            "case": name,
            "expected": "accept" if expected_valid else "reject",
            "verdict": "accept" if verdict.accepted else "reject",
            "pass": verdict.accepted == expected_valid,
            "composed_norm": verdict.composed_norm,
            "product_of_norms": verdict.product_of_norms,
            "product_ratio": verdict.product_ratio,
            "reference_scale": verdict.reference_scale,
            "scale_ratio": verdict.scale_ratio,
            "reason": verdict.reason,
            "detail": verdict.detail,
        }
        table.append(row)

    valid_rows = [r for r in table if r["expected"] == "accept"]
    pathological_rows = [r for r in table if r["expected"] == "reject"]

    max_valid_scale_ratio = max((r["scale_ratio"] for r in valid_rows), default=0.0)
    min_path_scale_ratio = min((r["scale_ratio"] for r in pathological_rows), default=0.0)

    result = {
        "thresholds": {
            "max_product_ratio": guard.max_product_ratio,
            "max_scale_ratio": guard.max_scale_ratio,
        },
        "reference_scale": reference,
        "table": table,
        "valid_accepted": sum(1 for r in valid_rows if r["pass"]),
        "valid_total": len(valid_rows),
        "pathological_rejected": sum(1 for r in pathological_rows if r["pass"]),
        "pathological_total": len(pathological_rows),
        "max_valid_scale_ratio": max_valid_scale_ratio,
        "min_pathological_scale_ratio": min_path_scale_ratio,
        "margin_ratio": (min_path_scale_ratio / max(max_valid_scale_ratio, 1e-12)),
        "calibrated": (
            all(r["pass"] for r in table)
            and min_path_scale_ratio > guard.max_scale_ratio
            and max_valid_scale_ratio < guard.max_scale_ratio
        ),
    }

    if verbose:
        print(f"[guard-calibration] reference_scale={reference:.4g} "
              f"max_product_ratio={guard.max_product_ratio} "
              f"max_scale_ratio={guard.max_scale_ratio}")
        for r in table:
            print(f"  {r['case']:<28} expected={r['expected']:<6} "
                  f"verdict={r['verdict']:<6} pass={r['pass']} "
                  f"|w|={r['composed_norm']:.4g} scale_ratio={r['scale_ratio']:.4g} "
                  f"product_ratio={r['product_ratio'] if r['product_ratio'] is not None else 'n/a'}"
                  f" reason={r['reason']}")
        print(f"  valid accepted {result['valid_accepted']}/{result['valid_total']}, "
              f"pathological rejected {result['pathological_rejected']}/"
              f"{result['pathological_total']}, "
              f"margin {result['margin_ratio']:.2f}x, "
              f"calibrated={result['calibrated']}")

    return result
