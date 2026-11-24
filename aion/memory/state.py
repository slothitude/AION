"""Exact serialization of the AION-side plastic state (save/restart round-trip).

OWNER LAW (retention apparatus, 2026-11-24): on shutdown, everything AION-side is
serialized — plastic state (u_fast, theta_slow, consolidation gates, evidence
buffers, eligibility traces, relationship.W, residual gate gamma, connection
scalars), memory stores. The FOUNDATION is frozen weights on disk and simply
reloads; it is never serialized here.

FROZEN-CODE LAW: this module only READS tensors off the frozen
PlasticRelationalConnection objects (public attributes) and copies them back on
restore. The frozen code in src/nnfs/connections/plastic.py is never edited and
its update/consolidation logic is never reimplemented.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# Tensor names captured per PlasticRelationalConnection. `relationship.W` exists
# for the "bilinear"/"post" relation types the interface uses; context_gate only
# exists when context_dim > 0 (the interface builds context_dim=0, but the map is
# defensive so a future context-carrying interface still round-trips exactly).
_U_FAST = "u_fast"
_THETA = "theta_slow"
_GATE = "consolidation_gate"


def conn_tensor_map(conn) -> Dict[str, torch.Tensor]:
    """Read every learned/derived tensor off one frozen plastic connection.

    Read-only: returns detached references (callers clone before storing).
    """
    t: Dict[str, torch.Tensor] = {}
    t["relationship.W"] = conn.relationship.W.detach()
    for name in (_U_FAST, _THETA, _GATE):
        p = getattr(conn.plasticity, name, None)
        if p is not None:
            t[name] = p.detach()
    for name in ("stability_evidence", "evidence_count"):
        b = getattr(conn.plasticity, name, None)
        if b is not None and torch.is_tensor(b):
            t[name] = b.detach()
    et = conn.eligibility_trace
    for name in ("trace", "pre_activity", "post_activity", "context_activity"):
        v = getattr(et, name, None)
        if v is not None and torch.is_tensor(v):
            t[f"eligibility.{name}"] = v.detach()
    for name in ("update_count", "total_weight_change"):
        v = getattr(conn, name, None)
        if v is not None and torch.is_tensor(v):
            t[name] = v.detach()
    cg = getattr(conn.relationship, "context_gate", None)
    if cg is not None:
        for n, p in cg.named_parameters():
            t[f"relationship.context_gate.{n}"] = p.detach()
    return t


def _copy_into(dst, src: torch.Tensor) -> None:
    """Copy `src` into the live parameter/buffer `dst` without autograd effects."""
    with torch.no_grad():
        if isinstance(dst, torch.nn.Parameter):
            dst.data.copy_(src)
        else:
            dst.copy_(src)


@dataclass
class PlasticStateSnapshot:
    """Everything needed to reconstruct AION-side learned state bit-exactly."""

    cfg: Dict[str, object]                      # InterfaceConfig scalars
    hidden_dim: int
    pairs: List[Tuple[int, int]]                # hooked (pre, post) layer pairs
    layer_names: List[str]                      # informational
    gates: List[torch.Tensor]                   # residual gate gamma per pair
    conns: List[Dict[str, torch.Tensor]]        # per-connection tensor map
    meta: Dict[str, object] = field(default_factory=dict)

    # -- capture / restore --------------------------------------------------- #

    @classmethod
    def capture(cls, iface, meta: Optional[Dict[str, object]] = None) -> "PlasticStateSnapshot":
        cfg = iface.cfg
        scalars = {
            "trace_decay": cfg.trace_decay,
            "learning_rate": cfg.learning_rate,
            "blend_ratio": cfg.blend_ratio,
            "gate_lr": cfg.gate_lr,
            "consolidation_interval": cfg.consolidation_interval,
            "relation_type": cfg.relation_type,
            "normalize_states": cfg.normalize_states,
        }
        return cls(
            cfg=scalars,
            hidden_dim=int(iface.hidden_dim),
            pairs=[(int(a), int(b)) for (a, b) in iface.pairs],
            layer_names=[l.name for l in iface.layers],
            gates=[g.detach().clone() for g in iface.gates],
            conns=[{k: v.clone() for k, v in conn_tensor_map(c).items()} for c in iface.conns],
            meta=dict(meta or {}),
        )

    def build_interface(self, base):
        """Construct a fresh RelationalInterface configured exactly like the one
        that was captured, then restore the captured tensors into it. (Read-only
        import of aion/relational/interface.py — landed code, never edited.)"""
        from aion.relational.interface import InterfaceConfig, RelationalInterface

        cfg = InterfaceConfig(**self.cfg)
        iface = RelationalInterface(base, cfg=cfg, layer_pairs=[tuple(p) for p in self.pairs])
        self.restore(iface)
        return iface

    def restore(self, iface) -> bool:
        """Copy the captured tensors into the live interface. Returns True when
        every captured tensor found a destination of matching shape."""
        if [(int(a), int(b)) for (a, b) in iface.pairs] != self.pairs:
            raise ValueError("snapshot pairs do not match live interface pairs")
        if int(iface.hidden_dim) != int(self.hidden_dim):
            raise ValueError("snapshot hidden_dim does not match live interface")
        ok = True
        if len(iface.gates) == len(self.gates):
            for g, s in zip(iface.gates, self.gates):
                if tuple(g.shape) != tuple(s.shape):
                    ok = False
                    continue
                _copy_into(g, s)
        else:
            ok = False
        if len(iface.conns) != len(self.conns):
            return False
        for conn, snap in zip(iface.conns, self.conns):
            live = conn_tensor_map(conn)
            for name, src in snap.items():
                dst = live.get(name)
                if dst is None or tuple(dst.shape) != tuple(src.shape):
                    ok = False
                    continue
                _copy_into(dst, src)
        return ok

    # -- comparison ----------------------------------------------------------- #

    def equals(self, other: "PlasticStateSnapshot", atol: float = 0.0) -> Tuple[bool, List[str]]:
        """Exact (atol=0 → torch.equal, bitwise) or tolerant comparison of two
        snapshots. Returns (identical, list of differing tensor names)."""
        diffs: List[str] = []
        if self.cfg != other.cfg:
            diffs.append("cfg")
        if self.hidden_dim != other.hidden_dim or self.pairs != other.pairs:
            diffs.append("topology")
        if len(self.gates) != len(other.gates) or len(self.conns) != len(other.conns):
            diffs.append("structure")
            return (not diffs), diffs
        for i, (ga, gb) in enumerate(zip(self.gates, other.gates)):
            if not _tensor_eq(ga, gb, atol):
                diffs.append(f"gates[{i}]")
        for k, (ca, cb) in enumerate(zip(self.conns, other.conns)):
            if ca.keys() != cb.keys():
                diffs.append(f"conn{k}.keys")
                continue
            for name in ca:
                if not _tensor_eq(ca[name], cb[name], atol):
                    diffs.append(f"conn{k}.{name}")
        return (not diffs), diffs

    # -- persistence ----------------------------------------------------------- #

    def to_payload(self) -> Dict[str, object]:
        return {
            "cfg": self.cfg,
            "hidden_dim": self.hidden_dim,
            "pairs": self.pairs,
            "layer_names": self.layer_names,
            "gates": [g.clone() for g in self.gates],
            "conns": [{k: v.clone() for k, v in c.items()} for c in self.conns],
            "meta": self.meta,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "PlasticStateSnapshot":
        return cls(
            cfg=dict(payload["cfg"]),
            hidden_dim=int(payload["hidden_dim"]),
            pairs=[(int(a), int(b)) for (a, b) in payload["pairs"]],
            layer_names=list(payload["layer_names"]),
            gates=[t.clone() for t in payload["gates"]],
            conns=[{k: v.clone() for k, v in c.items()} for c in payload["conns"]],
            meta=dict(payload.get("meta", {})),
        )

    def save(self, path: str) -> None:
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str) -> "PlasticStateSnapshot":
        return cls.from_payload(torch.load(path, weights_only=False))


def _tensor_eq(a: torch.Tensor, b: torch.Tensor, atol: float) -> bool:
    if a.shape != b.shape:
        return False
    if atol <= 0.0:
        return torch.equal(a, b)
    return torch.allclose(a, b, atol=atol, rtol=0.0)
