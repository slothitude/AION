"""The GYM-C stub runtime: a REAL AION runtime bundle over a scripted pathway.

No real model anywhere (law: no real model in tests — stubs). The dialogue side
is the REAL learn path (DialogueSession -> RuleExtractor -> Student.learn_fact
-> pathway.nnfs_update, exactly what the bridge tests exercise); the pathway is
a stub that spells a scripted answer so ask is deterministic. The relational
side is a stub whose strengths are strengthened BY THE STUB RUNTIME'S OWN
teach side effect — the same plastic side effect the real runtime lane owns.
The bridge only OBSERVES that change; it never writes it.

Everything the gate needs lives on one object: build_runtime() -> StubRuntime
with .bundle (the GymBridge RuntimeBundle + dialogue/student/corrections), the
runtime-side tick helpers (consolidation tick, dream cycle, growth/naming,
nav decision) that the real runtime lane would call.
"""

import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aion.dialogue.conversation import DialogueSession               # noqa: E402
from aion.dialogue.lesson_extractor import RuleExtractor             # noqa: E402
from aion.dialogue.student import Student, StudentConfig             # noqa: E402
from aion.dialogue.teacher import ScriptedTeacher                    # noqa: E402
from aion.gym_bridge.emitter import RuntimeBundle                    # noqa: E402
from aion.hostmode.authority import HostRuntime                      # noqa: E402
from aion.hostmode.policy import Policy                              # noqa: E402
from aion.memory.devhistory import DevelopmentalHistory              # noqa: E402
from aion.memory.episodic import EpisodicStore                       # noqa: E402

# The teaching arc: Kav is the subject of the first fact the student learns.
CHAIN_LESSON = ("Kav operates Tav. Tav routes to relay Kimo. Kimo controls "
                "device Zor. Zor produces heat.")
ASK_Q = "who operates Tav?"
ASK_A = "Tav"


class FakeClock:
    """Injectable clock (the repo's determinism law)."""

    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


ALPHABET = "\x00" + "abcdefghijklmnopqrstuvwxyz .,-'"


class StubTokenizer:
    eos_id = 0
    vocab_size = len(ALPHABET)

    def encode(self, text):
        return [ALPHABET.find(ch) if ch in ALPHABET else ALPHABET.index(" ")
                for ch in text.lower()]

    def decode(self, ids):
        return "".join(ALPHABET[i] for i in ids)


class StubPathway:
    """Plastic-pathway stub: nnfs_update counts presentations (the real learn
    path drives it); forward spells the scripted answer (same prompt-tracking
    as the bridge suite's proven stub)."""

    def __init__(self):
        self.vocab = StubTokenizer.vocab_size
        self.script = None
        self._prompt = None
        self.presentations = 0

    def nnfs_update(self, input_ids):
        self.presentations += 1
        return {"nnfs": {"calls": self.presentations}}

    def forward(self, input_ids, attention_mask=None, learn=False):
        import torch
        cur = input_ids[0].tolist()
        if self._prompt is None or len(cur) == len(self._prompt):
            self._prompt = list(cur)
        offset = len(cur) - len(self._prompt)
        target = None
        if self.script is not None and 0 <= offset < len(self.script):
            target = self.script[offset]
        logits = torch.zeros(1, len(cur), self.vocab)
        logits[0, -1, target if target is not None else StubTokenizer.eos_id] = 10.0
        return {"logits": logits, "base_logits": logits, "nnfs_stats": {}}


class StubRelational:
    """Relational inventory stub. set_strength is the RUNTIME's mutator (the
    stub runtime's teach side effect calls it — never the bridge)."""

    def __init__(self):
        self.rows: Dict[str, Dict] = {}

    def set_strength(self, key, full, plastic, consolidated):
        self.rows[key] = {"key": key, "pair_index": len(self.rows),
                          "full": float(full), "plastic": float(plastic),
                          "consolidated": float(consolidated),
                          "superseded_by": None}
        return self.rows[key]

    def inventory(self):
        return [dict(self.rows[k]) for k in sorted(self.rows)]


class StubConsolidation:
    """Consolidation-ledger shape (events + violations) the emitter reads."""

    def __init__(self):
        self.events: List[Dict] = []
        self.violations: List[Dict] = []


class StubDialogue:
    """The stub runtime's dialogue front: the REAL DialogueSession learn path,
    plus the runtime-owned plastic side effect — teaching a fact strengthens
    the matching relational pair (that is the runtime's job, and it is exactly
    what connection.updated then observes)."""

    def __init__(self, session: DialogueSession, relational: StubRelational,
                 episodic: EpisodicStore, plastic_step: float = 0.1):
        self.session = session
        self.relational = relational
        self.episodic = episodic
        self.plastic_step = float(plastic_step)

    @property
    def student(self):
        return self.session.student

    @property
    def corrections(self):
        return self.session.corrections

    @property
    def extractor(self):
        return self.session.extractor

    def teach(self, lesson_text: str):
        facts = self.session.teach(lesson_text)     # THE REAL LEARN PATH
        for f in facts:                             # the runtime's plastic side effect
            key = f"{f.subject}|{f.relation}|{f.object}"
            prev = self.relational.rows.get(key)
            plastic = (prev["plastic"] if prev else 0.0) + self.plastic_step
            consolidated = prev["consolidated"] if prev else 0.0
            self.relational.set_strength(key, plastic + consolidated, plastic,
                                         consolidated)
        # the runtime lane records the experience it just learned
        self.episodic.record("lesson", lesson_text, teacher="script",
                             plastic_update_ids=list(range(len(facts))))
        return facts

    def ask(self, question: str):
        return self.session.ask(question)


class StubRuntime:
    """One assembled stub AION runtime + the runtime-side tick helpers."""

    def __init__(self, sandbox_root: str, instance_id: str = "kav-gym",
                 seed: int = 7):
        os.makedirs(sandbox_root, exist_ok=True)
        self.clock = FakeClock(1000.0)
        self.relational = StubRelational()
        self.episodic = EpisodicStore(clock=self.clock)
        self.devhistory = DevelopmentalHistory.begin(
            instance_id, {"model": "stub", "hidden_dim": 8},
            clock=self.clock, seed=seed)
        policy = Policy(jail_root=os.path.join(sandbox_root, "jail"),
                        state_root=os.path.join(sandbox_root, "state"))
        self.host = HostRuntime(policy, clock=self.clock)
        self.consolidation = StubConsolidation()

        tok = StubTokenizer()
        pathway = StubPathway()
        pathway.script = tok.encode(ASK_A)
        student = Student(pathway, tok, StudentConfig(answer_tokens=8))
        teacher = ScriptedTeacher(reference_answers={ASK_Q: ASK_A})
        session = DialogueSession(teacher, student=student, extractor=RuleExtractor())
        self.student = student
        self.pathway = pathway
        self.dialogue = StubDialogue(session, self.relational, self.episodic)

        self.bundle = RuntimeBundle(
            instance_id=instance_id,
            relational=self.relational,
            episodic=self.episodic,
            devhistory=self.devhistory,
            host=self.host,
            consolidation=self.consolidation,
        )
        self.bundle.dialogue = self.dialogue
        self.bundle.student = student

    # -- runtime-side ticks (the real runtime lane calls these; the harness ---- #
    # -- calls them here because there IS no real runtime lane in the gate) ---- #

    def consolidation_tick(self, pair_index: int = 0) -> None:
        """One consolidation cycle: the ledger records the folded pair and the
        preservation invariant held (stub: violations stay empty)."""
        self.consolidation.events.append({"pair_index": pair_index,
                                          "kind": "consolidate"})

    def dream_cycle(self, dream_id: str = "d-gate-1",
                    hypothesis_id: str = "h-gate-1",
                    statement: str = "Kav operates Tav") -> None:
        """One locked dream cycle, filed exactly as DevelopmentalHistory files it."""
        self.devhistory.record_dream({"dream_id": dream_id})
        self.devhistory.record_dream({"dream_id": dream_id, "kind": "hypothesis",
                                      "hypothesis_id": hypothesis_id,
                                      "statement": statement, "confidence": 0.7})
        self.devhistory.record_dream({"dream_id": dream_id, "kind": "evaluated",
                                      "hypothesis_id": hypothesis_id,
                                      "verdict": "confirmed", "score": 0.75,
                                      "detail": "preservation held"})

    def growth(self, new_name: str = "Parasite",
               milestone: str = "first lesson taught") -> None:
        """One growth event that lands on a NEW NAMED STATE (nomen change)."""
        self.devhistory.record_growth({"milestone": milestone}, new_name=new_name)

    def nav_decision(self, emitter) -> None:
        """One recorded NAV decision (the bridge only files it)."""
        emitter.queue_nav_action("ASK_TEACHER", "gate-scenario-1",
                                 goal_text="answer the teacher's question",
                                 options=["QUERY", "ASK_TEACHER"],
                                 rationale="not covered by plastic alone")
