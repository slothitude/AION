"""Consolidation ledger: wrapper around the FROZEN consolidation machinery.

OWNER BRIEF (2026-11-24): schedule consolidation events, record before/after
snapshots (u, theta, effective mapping), and enforce the output-preservation
invariant on every consolidation event — the 1e-6 tolerance test from the frozen
baseline (tests/test_connections_learning.py Test 5). CALL IT, DON'T REIMPLEMENT:
the consolidation itself is always the frozen
    PlasticRelationalConnection.consolidate()      (-> FastSlowPlasticity.consolidate)
whose output-preserving transfer (u <- u - d ; theta += (blend/(1-blend)) * d,
d = transfer_fraction * u) leaves the effective mapping a = blend*u_fast +
(1-blend)*theta_slow — and therefore the transmitted output y_hat — invariant
to float tolerance. This module only photographs the state before/after and
checks the invariant with torch.allclose(atol=1e-6, rtol=1e-6).

FROZEN-CODE LAW: src/nnfs/connections/plastic.py is imported read-only (through
aion/relational/interface.py's file-path loader). No consolidation math lives
here.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch

from aion.memory.state import conn_tensor_map

TOLERANCE = 1e-6          # the frozen baseline's output-preservation tolerance

ProbeSource = Callable[[int], Tuple[torch.Tensor, torch.Tensor]]


def _default_probe_source(hidden_dim: int) -> ProbeSource:
    """Deterministic per-pair probe states, used when no real probe is supplied."""
    def probe(pair_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(1000 + int(pair_index))
        pre = torch.randn(hidden_dim, generator=g) / (hidden_dim ** 0.5)
        post = torch.randn(hidden_dim, generator=g) / (hidden_dim ** 0.5)
        return pre, post
    return probe


class ConsolidationLedger:
    """Schedules + audits consolidation events; enforces the 1e-6 invariant."""

    def __init__(self, conns: List, hidden_dim: int,
                 probe_source: Optional[ProbeSource] = None,
                 tolerance: float = TOLERANCE):
        self.conns = conns
        self.hidden_dim = int(hidden_dim)
        self.probe_source = probe_source or _default_probe_source(hidden_dim)
        self.tolerance = float(tolerance)
        self.events: List[Dict] = []
        self.violations: List[Dict] = []

    # -- photographing ----------------------------------------------------------- #

    def readout(self, pair_index: int) -> Dict[str, torch.Tensor]:
        """Public before/after readout: effective mapping `a` and transmitted `y`
        on this pair's probe, straight from the frozen objects."""
        return self._readout(pair_index)

    def _readout(self, pair_index: int) -> Dict[str, torch.Tensor]:
        conn = self.conns[pair_index]
        pre, post = self.probe_source(pair_index)
        with torch.no_grad():
            a = conn.plasticity.get_combined_weight(conn.blend_ratio)
            y = conn(pre.unsqueeze(0), post.unsqueeze(0), learn=False)[0]
        return {"a": a.detach().clone(), "y": y.detach().clone(),
                "u_norm": float(conn.plasticity.u_fast.norm().item()),
                "theta_norm": float(conn.plasticity.theta_slow.norm().item())}

    # -- the invariant ------------------------------------------------------------ #

    def verify_preservation(self, before: Dict[str, torch.Tensor], pair_index: int
                            ) -> Tuple[bool, str]:
        """Output-preservation check: the effective mapping `a` and the
        transmitted readout `y` must be unchanged by consolidation, within the
        frozen baseline's 1e-6 tolerance. Read-only measurement around the
        frozen call — no math of our own."""
        after = self._readout(pair_index)
        ok_a = torch.allclose(before["a"], after["a"], atol=self.tolerance, rtol=self.tolerance)
        ok_y = torch.allclose(before["y"], after["y"], atol=self.tolerance, rtol=self.tolerance)
        detail = (f"a_maxdiff={float((before['a'] - after['a']).abs().max().item()):.3e}, "
                  f"y_maxdiff={float((before['y'] - after['y']).abs().max().item()):.3e}, "
                  f"tol={self.tolerance:.1e}")
        return (ok_a and ok_y), detail

    # -- scheduling ---------------------------------------------------------------- #

    def consolidate_pair(self, pair_index: int, step: Optional[int] = None) -> Dict:
        """One consolidation event: photograph -> FROZEN consolidate -> verify."""
        conn = self.conns[pair_index]
        before = self._readout(pair_index)
        frozen_stats = conn.consolidate()          # <-- the frozen machinery, called
        ok, detail = self.verify_preservation(before, pair_index)
        after = self._readout(pair_index)
        event = {
            "pair_index": int(pair_index), "step": step,
            "before": {k: (v if isinstance(v, float) else "tensor") for k, v in before.items()},
            "after": {k: (v if isinstance(v, float) else "tensor") for k, v in after.items()},
            "u_norm_before": before["u_norm"], "u_norm_after": after["u_norm"],
            "theta_norm_before": before["theta_norm"], "theta_norm_after": after["theta_norm"],
            "frozen_stats": dict(frozen_stats),
            "invariant_ok": ok, "invariant_detail": detail,
        }
        self.events.append(event)
        if not ok:
            self.violations.append(event)
        return event

    def consolidate_all(self, step: Optional[int] = None) -> Dict:
        results = [self.consolidate_pair(k, step=step) for k in range(len(self.conns))]
        return {
            "events": results,
            "n_pairs": len(results),
            "transferred_pairs": sum(1 for e in results
                                     if e["frozen_stats"].get("consolidated", 0) > 0),
            "invariant_ok": all(e["invariant_ok"] for e in results),
        }

    def maybe_consolidate(self, step: int, interval: int) -> Optional[Dict]:
        """Schedule helper: consolidate when `step` crosses `interval`."""
        if interval > 0 and step > 0 and step % interval == 0:
            return self.consolidate_all(step=step)
        return None

    def assert_clean(self) -> None:
        if self.violations:
            v = self.violations[0]
            raise AssertionError(
                f"consolidation output-preservation INVARIANT VIOLATED on pair "
                f"{v['pair_index']}: {v['invariant_detail']}")

    # -- audit ---------------------------------------------------------------------- #

    def report(self) -> Dict:
        return {
            "n_events": len(self.events),
            "n_violations": len(self.violations),
            "invariant_ok": len(self.violations) == 0,
            "tolerance": self.tolerance,
            "total_consolidated_entries": sum(
                e["frozen_stats"].get("consolidated", 0) for e in self.events),
            "u_norm_trajectories": {
                f"pair{e['pair_index']}": [e["u_norm_before"], e["u_norm_after"]]
                for e in self.events},
            "theta_norm_trajectories": {
                f"pair{e['pair_index']}": [e["theta_norm_before"], e["theta_norm_after"]]
                for e in self.events},
        }
