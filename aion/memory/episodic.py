"""Episodic experience store: lessons / observations / corrections with provenance.

OWNER BRIEF (2026-11-24): experiences are stored as timestamped events tagged
with provenance — which teacher, which transcript, which plastic-update IDs.
The store exists for AUDIT and RETRIEVAL (variant B), not as the substrate of
knowledge: the transcript-REMOVAL test purges transcript history while the
PLASTIC STATE survives. Memory of experiences must live in the plastic /
consolidated state, not in this log.

Determinism law: `clock` is injectable (tests pass a fixed clock); sequence
numbers are monotonic integers so ordering is stable regardless of wall time.
Persistence is plain JSON — no tensors live here.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

PURGE_TOMBSTONE = "purge_tombstone"

KINDS = ("lesson", "observation", "correction")
CHANNELS = ("transcript", "direct")


@dataclass
class Episode:
    event_id: str
    seq: int
    ts: float
    kind: str                      # "lesson" | "observation" | "correction"
    channel: str                   # "transcript" (came through a teacher chat)
                                   # | "direct" (never had a transcript)
    text: str
    teacher: str = "unknown"
    transcript_id: Optional[str] = None
    plastic_update_ids: List[int] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id, "seq": self.seq, "ts": self.ts,
            "kind": self.kind, "channel": self.channel, "text": self.text,
            "teacher": self.teacher, "transcript_id": self.transcript_id,
            "plastic_update_ids": list(self.plastic_update_ids),
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Episode":
        return cls(
            event_id=d["event_id"], seq=int(d["seq"]), ts=float(d["ts"]),
            kind=d["kind"], channel=d["channel"], text=d["text"],
            teacher=d.get("teacher", "unknown"),
            transcript_id=d.get("transcript_id"),
            plastic_update_ids=[int(x) for x in d.get("plastic_update_ids", [])],
            tags=dict(d.get("tags", {})),
        )


def _tokens(text: str) -> set:
    return {t for t in text.lower().replace(",", " ").replace("?", " ").replace(".", " ").split() if t}


class EpisodicStore:
    """Append-only event log with provenance tags + transcript purge."""

    def __init__(self, clock: Optional[Callable[[], float]] = None):
        self._clock = clock or time.time
        self._events: List[Episode] = []
        self._seq = 0

    # -- recording ------------------------------------------------------------ #

    def record(self, kind: str, text: str, channel: str = "transcript",
               teacher: str = "unknown", transcript_id: Optional[str] = None,
               plastic_update_ids: Optional[List[int]] = None,
               tags: Optional[Dict[str, str]] = None) -> Episode:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
        self._seq += 1
        ev = Episode(
            event_id=f"ev{self._seq:06d}", seq=self._seq, ts=float(self._clock()),
            kind=kind, channel=channel, text=text, teacher=teacher,
            transcript_id=transcript_id,
            plastic_update_ids=list(plastic_update_ids or []),
            tags=dict(tags or {}),
        )
        self._events.append(ev)
        return ev

    def attach_update(self, event_id: str, update_id: int) -> bool:
        for ev in self._events:
            if ev.event_id == event_id:
                if update_id not in ev.plastic_update_ids:
                    ev.plastic_update_ids.append(int(update_id))
                return True
        return False

    # -- reading ---------------------------------------------------------------- #

    def events(self, kind: Optional[str] = None, channel: Optional[str] = None) -> List[Episode]:
        out = []
        for ev in self._events:
            if kind is not None and ev.kind != kind:
                continue
            if channel is not None and ev.channel != channel:
                continue
            out.append(ev)
        return out

    def __len__(self) -> int:
        return len(self._events)

    def texts(self) -> List[str]:
        """All non-tombstone texts currently in the log (for internalisation
        checks: was this exact question ever written down?)."""
        return [ev.text for ev in self._events if ev.kind != PURGE_TOMBSTONE]

    def updates_index(self) -> Dict[int, List[str]]:
        """plastic_update_id -> event ids that reference it (provenance both ways)."""
        idx: Dict[int, List[str]] = {}
        for ev in self._events:
            for uid in ev.plastic_update_ids:
                idx.setdefault(int(uid), []).append(ev.event_id)
        return idx

    # -- transcript REMOVAL (the whole point) ----------------------------------- #

    def purge_transcripts(self) -> Dict:
        """Remove every transcript-backed event from the log.

        The plastic/consolidated state is NOT touched by this module — purging
        here is pure log deletion, which is exactly the point of the
        transcript-REMOVAL test: if knowledge lived in the log, purging would
        destroy it; because it lives in plastic state, answers survive.

        A single tombstone record is appended so the audit trail knows a purge
        happened and which plastic-update IDs lost their transcript backing.
        """
        removed = [ev for ev in self._events if ev.channel == "transcript"]
        kept = [ev for ev in self._events if ev.channel != "transcript"]
        self._events = kept
        touched = sorted({u for ev in removed for u in ev.plastic_update_ids})
        self._seq += 1
        self._events.append(Episode(
            event_id=f"ev{self._seq:06d}", seq=self._seq, ts=float(self._clock()),
            kind=PURGE_TOMBSTONE, channel="direct",
            text=f"purged {len(removed)} transcript events",
            plastic_update_ids=touched,
            tags={"purged": str(len(removed))},
        ))
        return {"removed": len(removed), "kept": len(kept),
                "orphaned_update_ids": touched}

    # -- retrieval (variant B: conversation-retrieval) --------------------------- #

    def retrieve(self, query: str, k: int = 1) -> List[tuple]:
        """Best-matching events by deterministic token overlap (Jaccard).

        Ties broken by lower seq — stable across runs and processes. Tombstones
        are never returned. Score in [0, 1]; 1.0 == every query token present.
        """
        qt = _tokens(query)
        scored = []
        for ev in self._events:
            if ev.kind == PURGE_TOMBSTONE:
                continue
            et = _tokens(ev.text)
            union = qt | et
            score = (len(qt & et) / len(union)) if union else 0.0
            scored.append((score, ev))
        scored.sort(key=lambda p: (-p[0], p[1].seq))
        return [(ev, score) for score, ev in scored[:k]]

    # -- persistence -------------------------------------------------------------- #

    def to_dict(self) -> Dict:
        return {"seq": self._seq, "events": [ev.to_dict() for ev in self._events]}

    @classmethod
    def from_dict(cls, d: Dict, clock: Optional[Callable[[], float]] = None) -> "EpisodicStore":
        store = cls(clock=clock)
        store._seq = int(d.get("seq", 0))
        store._events = [Episode.from_dict(e) for e in d.get("events", [])]
        return store

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str, clock: Optional[Callable[[], float]] = None) -> "EpisodicStore":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), clock=clock)
