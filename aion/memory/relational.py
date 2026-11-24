"""The owned relational knowledge store: an index over PLASTIC state.

OWNER BRIEF (2026-11-24): this is AIONCore's knowledge inventory — the thing the
disconnect test interrogates. It reads state from the plastic pathway (u_fast,
theta_slow per connection, through the FROZEN public API in
src/nnfs/connections/plastic.py, imported read-only via
aion/relational/interface.py) and maintains a queryable index of learned
relationships: subject-relation-object -> WHERE it lives in plastic state
(which hooked pair, which probe states exercise it, how strongly it transmits
NOW under fast / slow / blended weights).

Strength is measured, not assumed: for each indexed relation we keep the
(pre_state, post_state) probe that exercises it and recompute the transmitted
signal through the FROZEN forward math whenever strength is asked for:
    w   = plasticity.get_combined_weight(blend)     # frozen API
    psi = relationship(pre, post)                   # frozen API
    y   = (w * psi).sum(-1)                         # == conn.forward(learn=False)
Three ablated readings are first-class (the provenance buckets):
    blend=conn's own  -> FULL          (natural effective mapping)
    blend=1.0         -> PLASTIC       (u_fast alone; theta contributes 0)
    blend=0.0         -> CONSOLIDATED  (theta_slow alone; u_fast contributes 0)
RETRIEVED is not a plastic reading at all — it is the episodic store (variant B).

Pathway-agnostic by design: the host lane feeds probes from the interface's
hooked hidden states; the WITHOUT-host lane (knowledge_discovery.py) feeds
probes from direct observations with the host disconnected entirely. Both use
this same index and the same FROZEN PlasticRelationalConnection machinery.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from aion.memory.episodic import EpisodicStore

ProbeMaker = Callable[[List[int]], Tuple[torch.Tensor, torch.Tensor]]
LearnFn = Callable[[List[int]], Dict]


def relation_key(subject: str, relation: str, obj: str) -> str:
    return f"{subject}|{relation}|{obj}"


def parse_key(key: str) -> Tuple[str, str, str]:
    s, r, o = key.split("|", 2)
    return s, r, o


@dataclass
class RelationRecord:
    key: str                                   # "subject|relation|object"
    pair_index: int                            # which hooked pair carries it
    pre_probe: torch.Tensor                    # (hidden_dim,) unit-L2
    post_probe: torch.Tensor                   # (hidden_dim,) unit-L2
    registered_at_seq: int
    meta: Dict[str, object] = field(default_factory=dict)
    superseded_by: Optional[str] = None        # set when a correction replaces it

    def to_dict(self) -> Dict:
        return {
            "key": self.key, "pair_index": self.pair_index,
            "registered_at_seq": self.registered_at_seq,
            "meta": dict(self.meta), "superseded_by": self.superseded_by,
        }


class RelationalMemory:
    """Knowledge inventory over the plastic relational pathway."""

    def __init__(self, conns: List, hidden_dim: int,
                 episodic: Optional[EpisodicStore] = None,
                 make_probe: Optional[ProbeMaker] = None,
                 blend_ratio: float = 0.7,
                 strength_threshold: float = 1e-8):
        self.conns = conns
        self.hidden_dim = int(hidden_dim)
        self.episodic = episodic
        self.make_probe = make_probe            # (prompt_ids) -> (pre, post)
        self.blend_ratio = float(blend_ratio)
        self.strength_threshold = float(strength_threshold)
        self.records: Dict[str, RelationRecord] = {}
        self.update_counter = 0

    # -- learning + registration ----------------------------------------------- #

    def next_update_id(self) -> int:
        self.update_counter += 1
        return self.update_counter

    def register(self, key: str, pair_index: int, pre_probe: torch.Tensor,
                 post_probe: torch.Tensor, meta: Optional[Dict] = None) -> RelationRecord:
        rec = RelationRecord(
            key=key, pair_index=int(pair_index),
            pre_probe=F.normalize(pre_probe.detach().float().flatten(), dim=0),
            post_probe=F.normalize(post_probe.detach().float().flatten(), dim=0),
            registered_at_seq=self.next_update_id(),
            meta=dict(meta or {}),
        )
        self.records[key] = rec                 # re-registration overwrites (latest wins)
        return rec

    def learn_relation(self, key: str, prompt_ids: List[int], learn_fn: LearnFn,
                       pair_index: int = 0, teacher: str = "battery",
                       transcript_id: Optional[str] = None,
                       channel: str = "transcript",
                       kind: str = "lesson", tags: Optional[Dict] = None,
                       n_steps: int = 1) -> RelationRecord:
        """Drive plastic learning for a relation and index where it landed.

        `learn_fn(prompt_ids)` performs one real learning pass through the
        pathway (host lane: interface.nnfs_update; bare lane: conn forward with
        learn=True). Update IDs recorded here are the provenance link from the
        relational index back to the episodic log.
        """
        update_ids = []
        stats_last = {}
        # Protocol: learn_fn(prompt_ids) and make_probe(prompt_ids) both receive a
        # (1, T) LongTensor — the same shape the interface consumes.
        ids_tensor = torch.as_tensor([list(int(i) for i in prompt_ids)],
                                     dtype=torch.long)
        for _ in range(max(1, int(n_steps))):
            self.update_counter += 1
            stats_last = learn_fn(ids_tensor) or {}
            update_ids.append(self.update_counter)
        if self.make_probe is not None:
            pre, post = self.make_probe(ids_tensor)
        else:
            pre, post = self._fallback_probe(prompt_ids)
        rec = self.register(key, pair_index, pre, post,
                            meta={"update_ids": update_ids, "stats": stats_last,
                                  "teacher": teacher, "tags": dict(tags or {})})
        if self.episodic is not None:
            self.episodic.record(
                kind=kind, text=key.replace("|", " "), channel=channel,
                teacher=teacher, transcript_id=transcript_id,
                plastic_update_ids=update_ids, tags=dict(tags or {}))
        return rec

    def _fallback_probe(self, prompt_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(sum(int(i) for i in prompt_ids) + 7)
        pre = F.normalize(torch.randn(self.hidden_dim, generator=g), dim=0)
        post = F.normalize(torch.randn(self.hidden_dim, generator=g), dim=0)
        return pre, post

    # -- live readings from plastic state --------------------------------------- #

    def strength(self, key: str, blend: Optional[float] = None) -> Optional[float]:
        """Transmitted-signal norm for one indexed relation, read live from
        plastic state. blend=None -> the connection's own blend (FULL);
        blend=1.0 -> PLASTIC (u_fast only); blend=0.0 -> CONSOLIDATED (theta only)."""
        rec = self.records.get(key)
        if rec is None:
            return None
        conn = self.conns[rec.pair_index]
        w = conn.plasticity.get_combined_weight(
            conn.blend_ratio if blend is None else float(blend))
        with torch.no_grad():
            psi = conn.relationship(rec.pre_probe.unsqueeze(0), rec.post_probe.unsqueeze(0))
            transmitted = (w.unsqueeze(0) * psi).sum(dim=-1)
        return float(transmitted.norm().item())

    def strengths(self, key: str) -> Dict[str, Optional[float]]:
        return {"full": self.strength(key, None),
                "plastic": self.strength(key, 1.0),
                "consolidated": self.strength(key, 0.0)}

    def present(self, key: str, blend: Optional[float] = None) -> bool:
        s = self.strength(key, blend)
        return bool(s is not None and s > self.strength_threshold)

    # -- queries --------------------------------------------------------------- #

    def lookup(self, subject: Optional[str] = None, relation: Optional[str] = None,
               obj: Optional[str] = None) -> List[RelationRecord]:
        out = []
        for key, rec in sorted(self.records.items()):
            s, r, o = parse_key(key)
            if subject is not None and s != subject:
                continue
            if relation is not None and r != relation:
                continue
            if obj is not None and o != obj:
                continue
            out.append(rec)
        return out

    def nearest(self, query_probe: torch.Tensor, k: int = 1,
                active_only: bool = True) -> List[Tuple[RelationRecord, float]]:
        """Cosine-nearest indexed relations to a probe (the disconnect test's
        interrogation path: give me what the state knows, ranked)."""
        q = F.normalize(query_probe.detach().float().flatten(), dim=0)
        scored = []
        for key, rec in self.records.items():
            if active_only and not self.present(key):
                continue
            sim = float(torch.dot(q, rec.pre_probe).item())
            scored.append((rec, sim))
        scored.sort(key=lambda p: (-p[1], p[0].registered_at_seq, p[0].key))
        return scored[:k]

    def inventory(self) -> List[Dict]:
        """Full knowledge inventory with live per-bucket strengths — what the
        disconnect test prints before and after the host is disconnected."""
        rows = []
        for key, rec in sorted(self.records.items()):
            s, r, o = parse_key(key)
            conn = self.conns[rec.pair_index]
            rows.append({
                "subject": s, "relation": r, "object": o, "key": key,
                "pair_index": rec.pair_index,
                "u_fast_norm": float(conn.plasticity.u_fast.norm().item()),
                "theta_slow_norm": float(conn.plasticity.theta_slow.norm().item()),
                **self.strengths(key),
                "registered_at_seq": rec.registered_at_seq,
                "superseded_by": rec.superseded_by,
                "meta": dict(rec.meta),
            })
        return rows

    # -- persistence ------------------------------------------------------------ #

    def to_dict(self) -> Dict:
        return {"hidden_dim": self.hidden_dim, "blend_ratio": self.blend_ratio,
                "update_counter": self.update_counter,
                "records": {k: r.to_dict() for k, r in self.records.items()}}

    @classmethod
    def from_dict(cls, d: Dict, conns: List, episodic: Optional[EpisodicStore] = None,
                  make_probe: Optional[ProbeMaker] = None,
                  probes: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> "RelationalMemory":
        mem = cls(conns=conns, hidden_dim=int(d["hidden_dim"]), episodic=episodic,
                  make_probe=make_probe, blend_ratio=float(d.get("blend_ratio", 0.7)))
        mem.update_counter = int(d.get("update_counter", 0))
        for key, rd in d.get("records", {}).items():
            pr = (probes or {}).get(key, {})
            pre = pr.get("pre", torch.zeros(mem.hidden_dim))
            post = pr.get("post", torch.zeros(mem.hidden_dim))
            rec = RelationRecord(
                key=rd["key"], pair_index=int(rd["pair_index"]),
                pre_probe=pre.clone(), post_probe=post.clone(),
                registered_at_seq=int(rd.get("registered_at_seq", 0)),
                meta=dict(rd.get("meta", {})),
                superseded_by=rd.get("superseded_by"),
            )
            mem.records[key] = rec
        return mem

    def save_probes(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {k: {"pre": r.pre_probe.clone(), "post": r.post_probe.clone()}
                for k, r in self.records.items()}
