"""Correction events: teacher corrections become plastic UPDATES that override prior state.

Owner brief (2026-11-24): when the teacher corrects, the fact is re-entered into the plastic
pathway as a fresh, repeated training signal (NNFS plasticity is error-driven, so repeated
presentation of the corrected rendering is the override mechanism) — and every correction is
recorded as an event with BEFORE/AFTER fact pairs, which is exactly the hook the retention
lane needs to later test that the old relationship was not destroyed. FactLedger keeps the
current fact per (subject, relation) and the full supersession history for those tests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aion.dialogue.lesson_extractor import Fact


@dataclass
class CorrectionEvent:
    """One applied correction: before-state -> after-state, provenance attached."""

    before: Optional[Fact]            # previously held fact for (subject, relation), if any
    after: Fact                       # the corrected fact now trained into the pathway
    kind: str                         # "update" (before known) | "insert" (nothing prior)
    transcript_id: Optional[str] = None
    turn_index: Optional[int] = None
    repeats: int = 1
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind,
                "before": self.before.to_dict() if self.before else None,
                "after": self.after.to_dict(),
                "transcript_id": self.transcript_id, "turn_index": self.turn_index,
                "repeats": self.repeats, "meta": self.meta}


class FactLedger:
    """Current-belief ledger: one fact per (subject, relation), plus supersession history.

    Bookkeeping only — the ledger NEVER feeds the pathway. It exists so corrections can carry
    an explicit before/after pair and the retention lane can query what was superseded."""

    def __init__(self):
        self.facts: Dict[tuple, Fact] = {}
        self.superseded: List[CorrectionEvent] = []

    def current(self, subject: str, relation: str) -> Optional[Fact]:
        return self.facts.get((subject.lower(), relation))

    def record(self, fact: Fact) -> Optional[Fact]:
        """Store `fact` as the current belief; returns the fact it replaced (if any)."""
        previous = self.current(fact.subject, fact.relation)
        self.facts[fact.key] = fact
        return previous

    def history_for(self, subject: str, relation: str) -> List[CorrectionEvent]:
        """All supersessions recorded against one (subject, relation) — retention-lane hook."""
        key = (subject.lower(), relation)
        return [e for e in self.superseded
                if e.after.key == key or (e.before is not None and e.before.key == key)]


class CorrectionHandler:
    """Applies corrections to the student as plastic overrides and emits before/after events."""

    def __init__(self, ledger: Optional[FactLedger] = None, repeats: Optional[int] = None):
        self.ledger = ledger or FactLedger()
        self.repeats = repeats            # None -> fall back to the student's own config
        self.events: List[CorrectionEvent] = []

    def apply(self, student, after: Fact, before: Optional[Fact] = None,
              transcript_id: Optional[str] = None, turn_index: Optional[int] = None,
              repeats: Optional[int] = None) -> CorrectionEvent:
        """Train the corrected fact into the pathway (repeated = override weight) and emit the
        CorrectionEvent with its before/after pair. `before` defaults to the ledger's prior
        belief for the same (subject, relation)."""
        n = repeats if repeats is not None else (
            self.repeats if self.repeats is not None else getattr(
                getattr(student, "cfg", None), "correction_repeats", 1))
        stats = student.learn_fact(after, repeats=n) if student is not None else {}
        previous = self.ledger.record(after)
        event = CorrectionEvent(
            before=before or previous, after=after,
            kind="update" if (before or previous) else "insert",
            transcript_id=transcript_id, turn_index=turn_index, repeats=n,
            meta={"learn": {k: v for k, v in stats.items() if k != "nnfs"}})
        self.events.append(event)
        if event.kind == "update":
            self.ledger.superseded.append(event)
        return event

    def apply_reroute(self, student, intervention, transcript_id: Optional[str] = None,
                      turn_index: Optional[int] = None) -> CorrectionEvent:
        """Intervention events ('Tav is rerouted to Kimo') are plastic updates too: the conduit's
        routes-to edge is overridden in the pathway, with its prior routing as the before-state."""
        return self.apply(student, after=intervention.fact,
                          transcript_id=transcript_id, turn_index=turn_index)
