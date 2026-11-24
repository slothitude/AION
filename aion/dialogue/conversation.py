"""AION-1 dialogue session: the lesson/question/correction state machine (owner protocol).

Owner protocol (2026-11-24):
    Teacher teaches -> AION (student) answers/predicts -> Teacher confirms or corrects
    -> lesson event emitted.
States: idle -> taught -> questioned -> (verdict closes the round) -> idle. Illegal
transitions raise IllegalTransition. Every turn is tagged with PROVENANCE (lesson/correction/
question) and the session's transcript_id, so the retention lane can later do transcript-
removal tests without touching the learning loop.

LAW: relationship facts enter the plastic system through the Student; the transcript and the
lesson events exist only for provenance/battery bookkeeping and never feed the pathway.
"""

import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from aion.dialogue.correction import CorrectionEvent, CorrectionHandler
from aion.dialogue.lesson_extractor import ExtractionResult, Fact, RuleExtractor  # noqa: F401
from aion.dialogue.lesson_extractor import extract, extract_correction

STATE_IDLE = "idle"
STATE_TAUGHT = "taught"
STATE_QUESTIONED = "questioned"


class IllegalTransition(RuntimeError):
    """A protocol action was attempted from a state that forbids it."""


def default_grader(a: str, b: str) -> bool:
    """Case/punctuation-tolerant string match for short entity answers."""

    def norm(s: str) -> str:
        s = s.strip().lower()
        if s.endswith("."):
            s = s[:-1]
        return " ".join(s.split())

    return norm(a) == norm(b) and norm(a) != ""


@dataclass
class Turn:
    """One transcript row: tagged action + provenance + transcript id (bookkeeping only)."""

    index: int
    kind: str                       # teacher_teach | question | student_answer | teacher_answer
                                    # | confirmation | correction | intervention | lesson_event
    provenance: str                 # lesson | correction | question | system
    content: str
    transcript_id: str
    meta: Dict = field(default_factory=dict)


@dataclass
class LessonEvent:
    """The emitted record of a completed protocol round (the owner's 'lesson event')."""

    kind: str                       # lesson_round | question_round
    facts: List[Fact]
    question: Optional[str]
    student_answer: Optional[str]
    teacher_answer: Optional[str]
    verdict: str                    # confirmed | corrected | no_student | no_reference
    corrections: List[CorrectionEvent]
    transcript_id: str
    source_turn_index: int
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind,
                "facts": [f.to_dict() for f in self.facts],
                "question": self.question, "student_answer": self.student_answer,
                "teacher_answer": self.teacher_answer, "verdict": self.verdict,
                "corrections": [c.to_dict() for c in self.corrections],
                "transcript_id": self.transcript_id,
                "source_turn_index": self.source_turn_index, "meta": self.meta}


class DialogueSession:
    """Teacher <-> student session state machine. The teacher and extractor are duck-typed:
    anything with teach/answer/correct and extract/extract_correction works (ScriptedTeacher
    and RuleExtractor are the offline pair used in tests and build-ahead dry runs)."""

    def __init__(self, teacher, student=None, extractor=None, corrections=None,
                 transcript_id: Optional[str] = None,
                 grader: Callable[[str, str], bool] = default_grader):
        self.teacher = teacher
        self.student = student
        self.extractor = extractor or RuleExtractor()
        self.corrections = corrections or CorrectionHandler()
        self.transcript_id = transcript_id or uuid.uuid4().hex
        self.grader = grader
        self.state = STATE_IDLE
        self.turns: List[Turn] = []
        self.events: List[LessonEvent] = []
        self._pending_facts: List[Fact] = []

    # -- plumbing ------------------------------------------------------------ #

    def _record(self, kind: str, provenance: str, content: str,
                meta: Optional[Dict] = None) -> Turn:
        turn = Turn(index=len(self.turns), kind=kind, provenance=provenance, content=content,
                    transcript_id=self.transcript_id, meta=meta or {})
        self.turns.append(turn)
        return turn

    def _require(self, allowed: tuple, action: str):
        if self.state not in allowed:
            raise IllegalTransition(
                f"{action} not allowed in state {self.state!r} (allowed: {allowed})")

    # -- protocol: teach ------------------------------------------------------ #

    def teach(self, lesson_text: str) -> List[Fact]:
        """Teacher teaches -> extractor pulls facts -> each fact enters the PLASTIC pathway
        (repeats=1 for plain lessons) -> state taught. Interventions inside the lesson text
        are applied as plastic updates with their prior routing as the before-state."""
        self._require((STATE_IDLE,), "teach")
        self.teacher.teach(lesson_text)
        self._record("teacher_teach", "lesson", lesson_text)
        res: ExtractionResult = self.extractor.extract(lesson_text, provenance="lesson")
        plain = [f for f in res.facts if f.provenance != "intervention"]
        for fact in plain:
            if self.student is not None:
                self.student.learn_fact(fact)
            self.corrections.ledger.record(fact)     # bookkeeping only; never feeds the pathway
        for iv in res.interventions:                 # plastic update, prior routing as before
            self.corrections.apply_reroute(self.student, iv, transcript_id=self.transcript_id,
                                           turn_index=len(self.turns))
            self._record("intervention", "correction", iv.raw,
                         {"subject": iv.subject, "new_object": iv.new_object})
        self._pending_facts = plain
        self.state = STATE_TAUGHT
        return list(self._pending_facts)

    # -- protocol: ask -> answer -> confirm/correct -> event ------------------- #

    def ask(self, question: str) -> LessonEvent:
        """One Q&A round: student answers, teacher confirms or corrects, lesson event emitted.
        Allowed from idle (battery questions) or taught (post-lesson comprehension)."""
        self._require((STATE_IDLE, STATE_TAUGHT), "ask")
        opening_state = self.state
        self.state = STATE_QUESTIONED
        self._record("question", "question", question)

        teacher_turn = self.teacher.answer(question)
        reference = None
        if teacher_turn.structured and isinstance(teacher_turn.parsed, dict):
            reference = teacher_turn.parsed.get("answer")
        elif teacher_turn.content:
            reference = teacher_turn.content.strip()
        self._record("teacher_answer", "question", reference or "",
                     {"structured": teacher_turn.structured})

        student_answer = self.student.answer(question) if self.student is not None else None
        if student_answer is not None:
            self._record("student_answer", "question", student_answer)

        correction_events: List[CorrectionEvent] = []
        if student_answer is None:
            verdict = "no_student"
        elif reference is None:
            verdict = "no_reference"
        elif self.grader(student_answer, reference):
            verdict = "confirmed"
        else:
            verdict = "corrected"
            correction_events = self._correct(student_answer, reference)

        if verdict == "confirmed":
            self._record("confirmation", "question", f"confirmed: {student_answer}")
        elif verdict == "corrected":
            self._record("correction", "correction",
                         f"student={student_answer!r} -> teacher={reference!r}")

        event = LessonEvent(
            kind="lesson_round" if opening_state == STATE_TAUGHT else "question_round",
            facts=list(self._pending_facts), question=question,
            student_answer=student_answer, teacher_answer=reference, verdict=verdict,
            corrections=correction_events, transcript_id=self.transcript_id,
            source_turn_index=len(self.turns) - 1,
            meta={"opening_state": opening_state})
        self.events.append(event)
        self._record("lesson_event", "system", event.kind, {"verdict": verdict})
        self._pending_facts = []
        self.state = STATE_IDLE
        return event

    def _correct(self, mistake: str, reference: str) -> List[CorrectionEvent]:
        """Teacher corrects -> extractor pulls the corrected facts -> each becomes a plastic
        UPDATE that overrides prior state (repeats>1), emitting before/after correction events."""
        self.teacher.correct(mistake=mistake, correction=reference)
        cf = self.extractor.extract_correction(reference)
        events: List[CorrectionEvent] = []
        for fact in cf.corrected:
            before = self.corrections.ledger.current(fact.subject, fact.relation)
            events.append(self.corrections.apply(
                self.student, after=fact,
                before=before or (cf.rejected[0] if cf.rejected else None),
                transcript_id=self.transcript_id, turn_index=len(self.turns)))
        return events

    # -- convenience ----------------------------------------------------------- #

    def run_lesson_round(self, lesson_text: str, question: str) -> tuple:
        """Full owner protocol in one call: teach -> ask -> (confirm|correct) -> lesson event."""
        facts = self.teach(lesson_text)
        event = self.ask(question)
        return facts, event
