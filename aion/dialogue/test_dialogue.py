"""AION-1 dialogue apparatus unit tests — CPU only, stubs + tiny model, NO real endpoint calls.

Proves (owner brief, 2026-11-24):
  1. lesson extractor correctness on templated lessons (incl. converts-into subject swap)
  2. intervention extraction ("Tav is rerouted to Kimo")
  3. conversation state machine: owner protocol, provenance/transcript tags, illegal transitions
  4. corrections: plastic UPDATE with before/after fact pairs (retention-lane hook)
  5. world generator: determinism, 100 relationships, every battery answer graph-derivable
  6. student loop on a tiny stand-in model through the REAL relational interface (skipif no
     transformers) — facts reach pathway.nnfs_update, questions format answers
Run:  python -m pytest aion/dialogue/test_dialogue.py -q   (from the repo root)
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aion.dialogue.conversation import (  # noqa: E402
    DialogueSession, IllegalTransition, default_grader)
from aion.dialogue.correction import CorrectionHandler, FactLedger  # noqa: E402
from aion.dialogue.lesson_extractor import (  # noqa: E402
    Fact, RuleExtractor, extract, extract_correction)
from aion.dialogue.student import Student, StudentConfig  # noqa: E402
from aion.dialogue.teacher import (  # noqa: E402
    ScriptedTeacher, TeacherClient, TeacherConfig, parse_structured)
from aion.worlds.generator import (  # noqa: E402
    generate_world, verify_world)

CHAIN_LESSON = ("Neral operates Tav. Tav routes to relay Kimo. Kimo controls device Zor. "
                "Zor produces heat.")
CONVERT_LESSON = "A motor converts electrical energy into mechanical energy."


# --------------------------------------------------------------------------- #
# Offline stubs: tokenizer + pathway + scripted student answers
# --------------------------------------------------------------------------- #

ALPHABET = "\x00" + "abcdefghijklmnopqrstuvwxyz .,-'"  # id 0 = EOS, never produced by encode


class StubTokenizer:
    """Deterministic char tokenizer over ALPHABET; id = alphabet index (unknown -> space)."""

    eos_id = 0
    vocab_size = len(ALPHABET)

    def encode(self, text):
        return [ALPHABET.find(ch) if ch in ALPHABET else ALPHABET.index(" ") for ch in text.lower()]

    def decode(self, ids):
        return "".join(ALPHABET[i] for i in ids)


class StubPathway:
    """Pathway stub: nnfs_update counts plastic presentations; forward produces scripted
    next-token logits so `question -> answer` is fully deterministic without a model."""

    def __init__(self, vocab=StubTokenizer.vocab_size):
        self.vocab = vocab
        self.learned: list = []
        self.script = None       # token ids the pathway spells out for the current question
        self._prompt = None      # prompt token ids currently being answered

    def nnfs_update(self, input_ids):
        ids = input_ids[0].tolist()
        self.learned.append(ids)
        return {"nnfs": {"calls": len(self.learned), "n_tokens": len(ids)}}

    def forward(self, input_ids, attention_mask=None, learn=False):
        import torch

        cur = input_ids[0].tolist()
        if self._prompt is None or len(cur) == len(self._prompt):
            self._prompt = list(cur)          # a fresh prompt begins a new query
        offset = len(cur) - len(self._prompt)
        target = None
        if self.script is not None and 0 <= offset < len(self.script):
            target = self.script[offset]
        logits = torch.zeros(1, len(cur), self.vocab)
        logits[0, -1, target if target is not None else StubTokenizer.eos_id] = 10.0
        return {"logits": logits, "base_logits": logits, "nnfs_stats": {}}


def scripted_student(answer_text):
    """Student whose pathway spells `answer_text` for the next question asked."""
    tok = StubTokenizer()
    path = StubPathway()
    path.script = tok.encode(answer_text)
    return Student(path, tok, StudentConfig(answer_tokens=8))


# --------------------------------------------------------------------------- #
# 1. lesson extractor
# --------------------------------------------------------------------------- #

def test_extractor_templated_lessons():
    res = extract(CHAIN_LESSON, provenance="lesson")
    facts = res.facts
    assert res.unparsed == []
    assert len(facts) == 4
    assert [f.relation for f in facts] == ["operates", "routes-to", "controls", "produces"]
    assert (facts[0].subject, facts[0].object) == ("Neral", "Tav")
    assert (facts[1].subject, facts[1].object) == ("Tav", "Kimo")
    assert (facts[2].subject, facts[2].object) == ("Kimo", "Zor")
    assert (facts[3].subject, facts[3].object) == ("Zor", "heat")
    assert all(f.provenance == "lesson" for f in facts)


def test_extractor_converts_swaps_subject():
    res = extract(CONVERT_LESSON)
    assert len(res.facts) == 1
    f = res.facts[0]
    assert (f.subject, f.relation, f.object) == ("electrical energy", "converts-to",
                                                 "mechanical energy")
    assert f.agent == "A motor"


def test_extractor_facts_have_keys_and_raw():
    f = extract("Neral operates Tav.").facts[0]
    assert f.key == ("neral", "operates")
    assert f.raw == "Neral operates Tav."


# --------------------------------------------------------------------------- #
# 2. intervention extraction
# --------------------------------------------------------------------------- #

def test_extractor_intervention():
    res = extract("Tav is rerouted to relay Kimo.", provenance="intervention")
    assert len(res.facts) == 1 and len(res.interventions) == 1
    fact, iv = res.facts[0], res.interventions[0]
    assert fact.provenance == "intervention"
    assert (fact.subject, fact.relation, fact.object) == ("Tav", "routes-to", "Kimo")
    assert iv.kind == "reroute" and iv.new_object == "Kimo" and iv.subject == "Tav"
    assert iv.fact is fact


def test_extractor_correction_pairs():
    cf = extract_correction("No - Tav routes to relay Vex, not Kimo.")
    assert len(cf.corrected) == 1 and len(cf.rejected) == 1
    assert (cf.corrected[0].subject, cf.corrected[0].relation, cf.corrected[0].object) == \
        ("Tav", "routes-to", "Vex")
    assert (cf.rejected[0].subject, cf.rejected[0].relation, cf.rejected[0].object) == \
        ("Tav", "routes-to", "Kimo")
    assert cf.has_override


def test_extractor_rule_seam_class():
    ex = RuleExtractor()
    assert ex.extract("Kimo controls device Zor.").facts[0].relation == "controls"
    assert ex.extract_correction("Zor produces heat, not light.").rejected[0].object == "light"


# --------------------------------------------------------------------------- #
# 3. conversation state machine
# --------------------------------------------------------------------------- #

def test_session_full_lesson_round_confirmed():
    student = scripted_student("Tav")
    teacher = ScriptedTeacher(reference_answers={"Who does Neral operate?": "Tav"})
    session = DialogueSession(teacher, student)

    facts = session.teach(CHAIN_LESSON)
    assert len(facts) == 4
    assert session.state == "taught"
    # LAW: facts reached the plastic pathway, not just the transcript
    assert len(student.pathway.learned) == 4
    assert student.n_facts_learned == 4

    event = session.ask("Who does Neral operate?")
    assert session.state == "idle"
    assert event.verdict == "confirmed"
    assert event.student_answer.lower() == "tav" and event.teacher_answer == "Tav"
    assert len(session.events) == 1 and event.kind == "lesson_round"

    # every turn tagged with provenance + transcript id
    assert all(t.transcript_id == session.transcript_id for t in session.turns)
    provs = {t.kind: t.provenance for t in session.turns}
    assert provs["teacher_teach"] == "lesson"
    assert provs["question"] == "question"
    assert provs["confirmation"] == "question"


def test_session_correction_becomes_plastic_update_with_before_after():
    student = scripted_student("Sula")
    teacher = ScriptedTeacher(reference_answers={"Who does Neral operate?": "Neral operates Sula."})
    session = DialogueSession(teacher, student)
    session.teach(CHAIN_LESSON)

    event = session.ask("Who does Neral operate?")
    assert event.verdict == "corrected"
    assert session.turns[-2].kind == "correction"  # correction turn before the lesson_event
    assert session.turns[-2].provenance == "correction"
    # plastic override: corrected fact trained with repeats > 1 (4 lesson facts + 3 repeats)
    assert len(student.pathway.learned) == 7
    assert event.corrections and event.corrections[0].repeats == \
        student.cfg.correction_repeats
    # before/after hook for the retention lane
    ev = event.corrections[0]
    assert ev.kind == "update"
    assert (ev.before.subject, ev.before.object) == ("Neral", "Tav")
    assert (ev.after.subject, ev.after.object) == ("Neral", "Sula")


def test_session_illegal_transitions():
    session = DialogueSession(ScriptedTeacher())
    session.teach("Neral operates Tav.")
    with pytest.raises(IllegalTransition):
        session.teach("Sula operates Rinpo.")     # teach twice without finishing the round


def test_session_intervention_updates_ledger_and_pathway():
    student = scripted_student("Vex")
    teacher = ScriptedTeacher(reference_answers={"Where does Tav route?": "Vex"})
    session = DialogueSession(teacher, student)
    session.teach(CHAIN_LESSON)
    event = session.ask("Where does Tav route?")
    assert event.verdict == "confirmed"

    reroute = session.teach("Tav is rerouted to relay Vex.")
    assert reroute == []                       # an intervention is not a plain lesson fact
    assert len(session.events[-1].corrections) == 0
    iv_events = session.corrections.ledger.superseded
    assert len(iv_events) == 1
    assert (iv_events[0].before.subject, iv_events[0].before.object) == ("Tav", "Kimo")
    assert (iv_events[0].after.subject, iv_events[0].after.object) == ("Tav", "Vex")
    # plastic: the rerouted routing was re-trained with the override weight (4 + 3 repeats)
    assert len(student.pathway.learned) == 7


def test_session_battery_questions_run_from_idle():
    student = scripted_student("Rinpo")
    teacher = ScriptedTeacher(reference_answers={"Who does Sula operate?": "Rinpo"})
    session = DialogueSession(teacher, student)
    event = session.ask("Who does Sula operate?")     # no lesson taught first
    assert session.state == "idle" and event.verdict == "confirmed"
    assert event.kind == "question_round"


def test_default_grader_normalization():
    assert default_grader("Tav.", "tav")
    assert not default_grader("Tav", "Kimo")
    assert not default_grader("", "")


# --------------------------------------------------------------------------- #
# 4. corrections module
# --------------------------------------------------------------------------- #

def test_fact_ledger_before_after_history():
    ledger = FactLedger()
    old = Fact("Tav", "routes-to", "Kimo")
    new = Fact("Tav", "routes-to", "Vex", provenance="correction")
    assert ledger.record(old) is None
    assert ledger.current("Tav", "routes-to") is old
    assert ledger.record(new) is old
    handler = CorrectionHandler(ledger)
    handler.apply(None, after=new, before=old, transcript_id="t1", turn_index=3)
    ev = handler.events[-1]
    assert ev.kind == "update" and ev.before is old and ev.after is new
    assert ledger.history_for("Tav", "routes-to") == [ev]
    assert ev.to_dict()["before"]["object"] == "Kimo"


def test_correction_insert_when_no_prior():
    handler = CorrectionHandler(FactLedger())
    ev = handler.apply(None, after=Fact("Zor", "produces", "heat"))
    assert ev.kind == "insert" and ev.before is None


# --------------------------------------------------------------------------- #
# 5. teacher client (no network) + scripted teacher
# --------------------------------------------------------------------------- #

def test_parse_structured_tolerant():
    assert parse_structured('{"action":"answer","answer":"Tav"}')["answer"] == "Tav"
    assert parse_structured("```json\n{\"action\":\"teach\"}\n```")["action"] == "teach"
    assert parse_structured('Teacher says: {"action":"answer","answer":"Kimo"} done')\
        ["answer"] == "Kimo"
    assert parse_structured("no json here") is None


class FakeEndpointTeacher(TeacherClient):
    """Overrides the single HTTP seam with a canned OpenAI-shaped payload (no network)."""

    def __init__(self, content):
        super().__init__(TeacherConfig(base_url="http://stub.invalid/v1", model="stub-teacher"))
        self.content = content
        self.payloads = []

    def post_chat(self, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": self.content}}]}


def test_teacher_client_structured_no_network():
    content = json.dumps({"action": "answer", "question": "q", "answer": "Tav"})
    client = FakeEndpointTeacher(content)
    turn = client.answer("Who does Neral operate?")
    assert turn.structured and turn.parsed["answer"] == "Tav"
    assert client.payloads[0]["model"] == "stub-teacher"
    assert client.payloads[0]["messages"][0]["role"] == "system"
    assert turn.content == content

    client2 = FakeEndpointTeacher("I cannot answer that.")
    turn2 = client2.teach("Neral operates Tav.")
    assert turn2.action == "teach" and not turn2.structured and turn2.parsed is None


def test_scripted_teacher_reference_answers():
    t = ScriptedTeacher(reference_answers={"q": "Kimo"})
    turn = t.answer("q")
    assert turn.structured and turn.parsed["answer"] == "Kimo"
    assert t.calls[-1].startswith("answer:")


# --------------------------------------------------------------------------- #
# 6. world generator
# --------------------------------------------------------------------------- #

def test_world_shape_and_owner_example_chain():
    w = generate_world(seed=0)
    assert len(w.relationships) == 100                       # owner spec: 100 relationships
    assert w.n_chains == 25 and len(w.chains) == 25
    assert [r.sentence for r in w.relationships[:4]] == [
        "Neral operates Tav.", "Tav routes to relay Kimo.",
        "Kimo controls device Zor.", "Zor produces heat."]
    assert w.chains[0] == ["Neral", "Tav", "Kimo", "Zor", "heat"]
    # ALL NAMES INVENTED: owner-canon held-out entity names lead each pool
    assert w.actors[0] == "Neral" and w.conduits[0] == "Tav"
    assert w.relays[0] == "Kimo" and w.devices[0] == "Zor"


def test_world_determinism_and_seed_sensitivity():
    a, b, c = generate_world(seed=7), generate_world(seed=7), generate_world(seed=8)
    assert a.to_dict() == b.to_dict()                        # same seed -> identical world
    assert a.to_dict() != c.to_dict()                        # seed actually drives generation
    counts = {}
    for q in a.questions:
        counts[q.battery] = counts.get(q.battery, 0) + 1
    assert counts == {"direct": 100, "combination": 75, "paraphrase": 100, "intervention": 50}


def test_world_ground_truth_consistency():
    for seed in (0, 1, 7):
        w = generate_world(seed=seed)
        assert verify_world(w) == [], verify_world(w)[:3]
    # combinations are genuinely unseen: no relationship ever spans actor -> device
    w = generate_world(seed=0)
    pairs = {(r.subject, r.object) for r in w.relationships}
    for q in w.questions:
        if q.battery == "combination" and q.meta.get("kind") == "reach":
            assert (q.meta["start"], q.answer) not in pairs


def test_world_intervention_answers_change():
    w = generate_world(seed=0)
    chain0 = w.chains[0]
    from aion.worlds.generator import apply_reroute
    for q in (x for x in w.questions if x.id == "int-reach-000"):
        g2 = apply_reroute(w.graph, q.meta["conduit"], q.meta["new_relay"])
        assert w.terminal(q.meta["start"], g2) == q.answer
        assert w.terminal(q.meta["start"]) == chain0[4]      # pre-intervention terminal
        assert q.answer != chain0[4] and q.meta["new_relay"] != chain0[2]


def test_world_json_roundtrip(tmp_path):
    w = generate_world(seed=0)
    path = tmp_path / "world.json"
    payload = w.save_json(str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["relationships"][0]["subject"] == "Neral"
    assert len(loaded["questions"]) == len(payload["questions"]) == 325
    assert len(loaded["graph"]["Neral"][0]) == 2             # edges serialized as pairs


def test_world_sample_for_report():
    w = generate_world(seed=0)
    print("\nSAMPLE RELATIONSHIPS:", [r.sentence for r in w.relationships[:5]])
    print("SAMPLE QUESTIONS:", [(q.id, q.text, q.answer) for q in w.questions[:5]])


# --------------------------------------------------------------------------- #
# 7. student loop through the REAL relational interface (tiny stand-in model)
# --------------------------------------------------------------------------- #

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture(scope="module")
def tiny_student():
    """Tiny 4-layer Llama stand-in + the real RelationalInterface (read-only import)."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from aion.relational.interface import RelationalInterface

    cfg = LlamaConfig(vocab_size=StubTokenizer.vocab_size, hidden_size=64,
                      intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    iface = RelationalInterface(model)
    return Student(iface, StubTokenizer(), StudentConfig(answer_tokens=6))


def test_student_learn_fact_reaches_pathway(tiny_student):
    stats = tiny_student.learn_fact(Fact("Neral", "operates", "Tav"))
    assert stats["text"] == "Neral operates Tav."
    assert stats["n_tokens"] > 0 and stats["nnfs"]
    assert tiny_student.n_facts_learned == 1
    # foundation stayed frozen through the NNFS update (interface law)
    iface = tiny_student.pathway
    assert not any(p.requires_grad for p in iface.base.parameters())


def test_student_render_fact_templates(tiny_student):
    assert tiny_student.render_fact(Fact("Neral", "operates", "Tav")) == "Neral operates Tav."
    assert tiny_student.render_fact(Fact("Tav", "routes-to", "Kimo")) == "Tav routes to Kimo."
    assert tiny_student.render_fact(Fact("a motor", "converts-to", "x", agent=None)) == \
        "a motor converts into x."
    conv = Fact("electrical energy", "converts-to", "mechanical energy", agent="a motor")
    assert tiny_student.render_fact(conv) == \
        "a motor converts electrical energy into mechanical energy."


def test_student_ask_formats_answer(tiny_student):
    ans = tiny_student.ask("Who does Neral operate?")
    assert isinstance(ans.text, str)
    assert ans.stop_reason in {"budget", "eos", "sentence_end"}
    assert len(ans.meta["first_step_top"]) == 5              # top-k reported
    assert ans.token_ids == tiny_student.tokenizer.encode(ans.text)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
