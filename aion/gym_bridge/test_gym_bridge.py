"""AION-Gym bridge unit tests — CPU only, seeded, stub runtime, NO real model.

Proves (owner law: "Python remains authoritative. Godot subscribes to events and
requests permitted actions. It never writes directly into plastic state or
bypasses the AION runtime's authority controls."):

  1. schema: the 11-type envelope validates/rejects explicitly (closed types,
     closed vocabularies, no silent coercion), serialize/deserialize round-trips
  2. snapshot completeness on a stub runtime bundle (all sections, counts, goal)
  3. event emission on injected state changes (experience / connection /
     consolidation / dream / nomen / stage / hostmode) + the connection.updated
     throttle (emit on change, max 1/second, changes never dropped)
  4. action authority checks: unauthenticated / unknown / paused denials return
     structured reasons; approve_request routes through the REAL hostmode
     authority gate (executed + authority_denied outcomes); deny_request refuses
     before the gate; set_goal stays in bridge-owned state
  5. ZERO plastic writes: observation touches only read APIs (strict read
     proxy), snapshot/poll cause no plastic mutation, and every plastic
     mutation in the action flow has the dialogue learn path
     (aion.dialogue.student) as its immediate caller — never a gym_bridge frame
  6. server: action-envelope handling + a real loopback WebSocket/HTTP
     roundtrip with token auth (denial without token)
Run:  python -m pytest aion/gym_bridge/test_gym_bridge.py -q   (from the repo root)
"""

import asyncio
import inspect
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aion.dialogue.conversation import DialogueSession               # noqa: E402
from aion.dialogue.correction import CorrectionHandler               # noqa: E402
from aion.dialogue.lesson_extractor import RuleExtractor             # noqa: E402
from aion.dialogue.student import Student, StudentConfig             # noqa: E402
from aion.dialogue.teacher import ScriptedTeacher                    # noqa: E402
from aion.gym_bridge.actions import GymActions                       # noqa: E402
from aion.gym_bridge.emitter import GymEmitter, RuntimeBundle        # noqa: E402
from aion.gym_bridge.schema import (                                 # noqa: E402
    EVENT_TYPES, SCHEMA_VERSION, SchemaError, deserialize, make_event,
    serialize, validate)
from aion.gym_bridge.server import GymServer                         # noqa: E402
from aion.hostmode.authority import HostRuntime, Proposal            # noqa: E402
from aion.hostmode.policy import PermittedOperation, Policy          # noqa: E402
from aion.memory.consolidation import ConsolidationLedger            # noqa: E402
from aion.memory.devhistory import DevelopmentalHistory              # noqa: E402
from aion.memory.episodic import EpisodicStore                       # noqa: E402
from aion.nav.state import NAV_ACTIONS                               # noqa: E402

CHAIN_LESSON = ("Neral operates Tav. Tav routes to relay Kimo. Kimo controls "
                "device Zor. Zor produces heat.")
ASK_Q = "who operates Tav?"
ASK_A = "Tav"


# --------------------------------------------------------------------------- #
# Offline stubs (CPU, seeded, deterministic) + strict read guard
# --------------------------------------------------------------------------- #

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
    """Plastic-pathway stub: nnfs_update records EVERY plastic presentation with
    its immediate caller module (the zero-plastic-write proof) and forward
    spells a scripted answer so ask is deterministic without a model."""

    def __init__(self):
        self.vocab = StubTokenizer.vocab_size
        self.script = None
        self._prompt = None
        self.presentations = 0
        self.mutation_callers: List[str] = []

    def nnfs_update(self, input_ids):
        caller = inspect.currentframe().f_back.f_globals.get("__name__", "?")
        self.mutation_callers.append(caller)
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
    """Relational inventory stub: inventory() is the READ the bridge may use;
    set_strength is the RUNTIME's own mutator (caller-traced — the bridge must
    never appear among its callers)."""

    def __init__(self):
        self.rows: Dict[str, Dict] = {}
        self.mutation_callers: List[str] = []

    def set_strength(self, key, full, plastic, consolidated):
        caller = inspect.currentframe().f_back.f_globals.get("__name__", "?")
        self.mutation_callers.append(caller)
        self.rows[key] = {"key": key, "pair_index": 0, "full": full,
                          "plastic": plastic, "consolidated": consolidated,
                          "superseded_by": None}
        return self.rows[key]

    def inventory(self):
        return [dict(self.rows[k]) for k in sorted(self.rows)]


class StrictReadProxy:
    """Guards one observable: any attribute access outside the allowed read set
    FAILS the bridge (an explicit crash beats a silent plastic write)."""

    def __init__(self, inner, allowed=("inventory",)):
        self._inner = inner
        self._allowed = set(allowed)
        self.accesses: List[tuple] = []

    def __getattr__(self, name):
        if name not in self._allowed:
            self.accesses.append(("DENIED", name))
            raise AssertionError(f"bridge touched non-read attribute {name!r}")
        self.accesses.append(("read", name))
        return getattr(self._inner, name)


@dataclass
class StubConsolidation:
    events: List[Dict] = field(default_factory=list)
    violations: List[Dict] = field(default_factory=list)


@dataclass
class BridgeBundle:
    """One bundle carrying everything the bridge may observe or route through.
    All handles are duck-typed runtime objects or stubs; the bridge itself is
    the only new code in the picture."""
    instance_id: str = "t1"
    relational: Any = None
    interface: Any = None
    episodic: Any = None
    devhistory: Any = None
    host: Any = None
    consolidation: Any = None
    dialogue: Any = None
    student: Any = None
    corrections: Any = None


def make_dialogue(answer_map: Optional[Dict[str, str]] = None):
    tok = StubTokenizer()
    pathway = StubPathway()
    pathway.script = tok.encode(ASK_A)
    student = Student(pathway, tok, StudentConfig(answer_tokens=8))
    teacher = ScriptedTeacher(reference_answers=answer_map or {ASK_Q: ASK_A})
    session = DialogueSession(teacher, student=student, extractor=RuleExtractor())
    return session, student, pathway


def make_host(tmp_path) -> HostRuntime:
    policy = Policy(jail_root=str(tmp_path / "jail"), state_root=str(tmp_path / "state"))
    return HostRuntime(policy, clock=FakeClock(2000.0))


def make_bundle(tmp_path, with_all: bool = True) -> tuple:
    clock = FakeClock(1000.0)
    episodic = EpisodicStore(clock=clock)
    dev = DevelopmentalHistory.begin("t1", {"model": "stub", "hidden_dim": 8},
                                     clock=clock, seed=7)
    host = make_host(tmp_path) if with_all else None
    relational = StubRelational()
    consolidation = StubConsolidation()
    bundle = BridgeBundle(instance_id="t1", relational=relational,
                          episodic=episodic, devhistory=dev, host=host,
                          consolidation=consolidation)
    return bundle, clock, relational, episodic, dev, host


# --------------------------------------------------------------------------- #
# 1. schema
# --------------------------------------------------------------------------- #

def test_schema_all_11_types_roundtrip():
    payloads = {
        "state.snapshot": {"instance_id": "t1", "name": "AION", "paused": False},
        "experience.created": {"event_id": "ev1", "kind": "lesson", "text": "x"},
        "connection.updated": {"relations": [{"key": "a|operates|b"}]},
        "memory.consolidated": {"n_events": 2},
        "dream.started": {"dream_id": "d1"},
        "dream.hypothesis": {"dream_id": "d1", "hypothesis_id": "h1", "statement": "s"},
        "dream.evaluated": {"dream_id": "d1", "hypothesis_id": "h1", "verdict": "confirmed"},
        "nomen.changed": {"new_name": "Two", "old_name": "One"},
        "nav.action": {"action": "QUERY", "scenario_id": "s1"},
        "hostmode.request": {"request_id": "r1", "op": "WRITE_FILE", "status": "executed"},
        "development.stage": {"kind": "growth", "description": "d"},
    }
    assert set(payloads) == set(EVENT_TYPES) and len(EVENT_TYPES) == 11
    clock = FakeClock(5.0)
    for etype, payload in payloads.items():
        ev = make_event(etype, payload, clock=clock)
        assert ev["schema_version"] == SCHEMA_VERSION
        assert ev["timestamp"] == 5.0 and isinstance(ev["timestamp"], float)
        validate(ev)
        text = serialize(ev)
        assert deserialize(text) == ev


def test_schema_rejects_unknown_type_missing_field_unknown_field():
    with pytest.raises(SchemaError, match="unknown event type"):
        make_event("world.destroyed", {})
    with pytest.raises(SchemaError, match="missing required field 'dream_id'"):
        make_event("dream.started", {})
    with pytest.raises(SchemaError, match="unknown payload field"):
        make_event("dream.started", {"dream_id": "d1", "cheat": True})


def test_schema_rejects_wrong_field_types_and_bad_envelope():
    with pytest.raises(SchemaError, match="must be int"):
        validate({"schema_version": 1, "type": "experience.created", "timestamp": 1.0,
                  "payload": {"event_id": "e", "kind": "lesson", "text": "t", "seq": "x"}})
    with pytest.raises(SchemaError, match="must be bool"):
        make_event("state.snapshot", {"instance_id": "a", "name": "b", "paused": "yes"})
    with pytest.raises(SchemaError, match="nan"):
        make_event("memory.consolidated", {"n_events": 1}, timestamp=float("nan"))
    with pytest.raises(SchemaError, match="missing envelope key"):
        validate({"schema_version": 1, "type": "memory.consolidated", "payload": {}})
    with pytest.raises(SchemaError, match="unknown envelope key"):
        validate({"schema_version": 1, "type": "memory.consolidated", "timestamp": 1.0,
                  "payload": {"n_events": 1}, "extra": 1})
    with pytest.raises(SchemaError, match="schema_version"):
        validate({"schema_version": 2, "type": "memory.consolidated", "timestamp": 1.0,
                  "payload": {"n_events": 1}})
    with pytest.raises(SchemaError, match="payload must be a dict"):
        validate({"schema_version": 1, "type": "memory.consolidated", "timestamp": 1.0,
                  "payload": []})
    with pytest.raises(SchemaError, match="not valid JSON"):
        deserialize("{oops")


def test_schema_closed_vocabularies():
    for action in NAV_ACTIONS:                       # the frozen nav enum passes
        make_event("nav.action", {"action": action, "scenario_id": "s"})
    with pytest.raises(SchemaError, match="not in NAV_ACTIONS"):
        make_event("nav.action", {"action": "FLY", "scenario_id": "s"})
    with pytest.raises(SchemaError, match="not in PermittedOperation"):
        make_event("hostmode.request", {"request_id": "r", "op": "SELF_DESTRUCT",
                                        "status": "executed"})
    with pytest.raises(SchemaError, match="status"):
        make_event("hostmode.request", {"request_id": "r", "op": "WRITE_FILE",
                                        "status": "vibes"})


def test_serialize_rejects_non_json_safe():
    # passes the field-type checks (tags is a dict) but is not JSON-encodable
    ev = make_event("experience.created", {"event_id": "e1", "kind": "lesson",
                                           "text": "t"})
    bad = dict(ev)
    bad["payload"] = {"event_id": "e1", "kind": "lesson", "text": "t",
                      "tags": {"x": object()}}
    with pytest.raises(SchemaError, match="not JSON-safe"):
        serialize(bad)


# --------------------------------------------------------------------------- #
# 2. snapshot completeness on a stub runtime
# --------------------------------------------------------------------------- #

def test_snapshot_completeness_on_stub_runtime(tmp_path):
    bundle, clock, relational, episodic, dev, host = make_bundle(tmp_path)
    relational.set_strength("neral|operates|tav", 1.5, 0.5, 1.0)
    episodic.record("lesson", "Neral operates Tav.", teacher="script",
                    plastic_update_ids=[3])
    dev.record_experience({"event_id": "evx", "kind": "observation", "text": "saw it"})
    actions = GymActions(bundle)
    actions.request("set_goal", {"goal": "learn heat"}, authenticated=True)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", relational=relational,
                                       episodic=episodic, devhistory=dev, host=host,
                                       consolidation=bundle.consolidation),
                         clock=clock, goal_state=actions.goal_state, paused=True)
    snap = emitter.snapshot()
    validate(snap)
    p = snap["payload"]
    assert p["instance_id"] == "t1"
    assert p["name"] == "AION-t1"                       # the nomen current name
    assert p["paused"] is True
    assert p["goal"]["goal"] == "learn heat"
    assert p["relational"]["n_relations"] == 1
    assert p["relational"]["rows"][0]["full"] == pytest.approx(1.5)
    assert p["episodic"]["n_events"] == 1 and p["episodic"]["kinds"]["lesson"] == 1
    assert p["devhistory"]["current_name"] == "AION-t1" and p["devhistory"]["n_stages"] >= 1
    assert p["hostmode"]["n_records"] == 0
    assert "per_write_bytes" in p["hostmode"]["caps"]
    assert p["runtime_uptime_s"] >= 0.0


def test_snapshot_minimal_bundle_still_valid():
    emitter = GymEmitter(RuntimeBundle(instance_id="solo"), clock=FakeClock(1.0))
    snap = emitter.snapshot()
    validate(snap)
    p = snap["payload"]
    assert p["name"] == "AION" and p["paused"] is False
    assert "relational" not in p and "hostmode" not in p


# --------------------------------------------------------------------------- #
# 3. event emission on injected state changes
# --------------------------------------------------------------------------- #

def test_poll_experience_created_from_episodic(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", episodic=bundle.episodic),
                         clock=clock)
    assert emitter.poll_events() == []                  # baseline: nothing new
    ev = bundle.episodic.record("correction", "No - Tav routes to Vex.",
                                teacher="owner", plastic_update_ids=[9, 11])
    events = emitter.poll_events()
    assert len(events) == 1 and events[0]["type"] == "experience.created"
    p = events[0]["payload"]
    assert p["event_id"] == ev.event_id and p["kind"] == "correction"
    assert p["teacher"] == "owner" and p["plastic_update_ids"] == [9, 11]
    assert p["source"] == "episodic"
    assert emitter.poll_events() == []                  # no replay


def test_poll_hostmode_request_executed_and_denied(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    host = bundle.host
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", host=host), clock=clock)
    host.execute(Proposal(op=PermittedOperation.WRITE_FILE,
                          args={"path": "a.txt", "content": "hi"}, arm="test"))
    events = emitter.poll_events()
    assert len(events) == 1 and events[0]["type"] == "hostmode.request"
    p = events[0]["payload"]
    assert p["status"] == "executed" and p["op"] == "WRITE_FILE" and p["arm"] == "test"
    host.execute(Proposal(op=PermittedOperation.WRITE_FILE,
                          args={"path": "big.txt", "content": "x" * 4096}, arm="test"))
    events = emitter.poll_events()
    assert len(events) == 1
    p = events[0]["payload"]
    assert p["status"] == "authority_denied"
    assert p["reason"] == "capacity_exceeded"


def test_poll_memory_consolidated_and_dream_events(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", episodic=bundle.episodic,
                                       devhistory=bundle.devhistory,
                                       consolidation=bundle.consolidation),
                         clock=clock)
    bundle.consolidation.events.append({"pair_index": 0, "kind": "consolidate"})
    bundle.devhistory.record_dream({"dream_id": "d9"})
    bundle.devhistory.record_dream({"dream_id": "d9", "kind": "hypothesis",
                                    "hypothesis_id": "h1", "statement": "Tav->Vex"})
    bundle.devhistory.record_dream({"dream_id": "d9", "kind": "evaluated",
                                    "hypothesis_id": "h1", "verdict": "confirmed",
                                    "score": 0.75})
    events = emitter.poll_events()
    kinds = [e["type"] for e in events]
    assert kinds.count("memory.consolidated") == 1
    assert kinds.count("dream.started") == 1
    assert kinds.count("dream.hypothesis") == 1
    assert kinds.count("dream.evaluated") == 1
    by_type = {e["type"]: e["payload"] for e in events}
    assert by_type["memory.consolidated"]["n_events"] == 1
    assert by_type["dream.started"]["dream_id"] == "d9"
    assert by_type["dream.hypothesis"]["statement"] == "Tav->Vex"
    assert by_type["dream.evaluated"]["verdict"] == "confirmed"
    assert by_type["dream.evaluated"]["score"] == pytest.approx(0.75)


def test_poll_nomen_changed_and_development_stage(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", devhistory=bundle.devhistory),
                         clock=clock)
    bundle.devhistory.record_growth({"milestone": "first consolidation"},
                                    new_name="Wanderer")
    events = emitter.poll_events()
    names = {e["type"]: e["payload"] for e in events}
    assert "nomen.changed" in names and "development.stage" in names
    assert names["nomen.changed"]["new_name"] == "Wanderer"
    assert names["nomen.changed"]["old_name"] == "AION-t1"
    assert names["nomen.changed"]["lineage"][-1] == "Wanderer"
    stage = names["development.stage"]
    assert stage["kind"] == "growth" and stage["name"] == "Wanderer"
    assert stage["node_id"].startswith("n")
    assert emitter.poll_events() == []                  # no replay


def test_connection_updated_throttle_never_drops_a_change(tmp_path):
    bundle, clock, relational, *_ = make_bundle(tmp_path)
    relational.set_strength("a|operates|b", 1.0, 0.0, 1.0)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", relational=relational),
                         clock=clock, connection_update_interval=1.0)
    assert emitter.poll_events() == []                  # baseline armed
    clock.advance(0.5)
    relational.set_strength("a|operates|b", 2.0, 0.0, 1.0)     # change #1
    events = emitter.poll()
    assert len(events) == 1 and events[0]["type"] == "connection.updated"
    assert events[0]["payload"]["changed_keys"] == ["a|operates|b"]
    clock.advance(0.1)
    relational.set_strength("a|operates|b", 3.0, 0.0, 1.0)     # change #2 inside interval
    assert emitter.poll() == []                         # throttled, REMEMBERED
    clock.advance(1.2)                                  # past the 1s interval
    events = emitter.poll()
    assert len(events) == 1 and events[0]["type"] == "connection.updated"
    assert events[0]["payload"]["reason"] == "throttled change flushed"


def test_subscribe_push_and_nav_action_validation(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1"), clock=clock)
    received: List[Dict] = []
    emitter.subscribe(received.append)
    ev1 = emitter.queue_nav_action("QUERY", "sc-1", goal_text="reach heat",
                                   options=["QUERY", "INFER"])
    ev2 = emitter.queue_nav_action("ASK_TEACHER", "sc-2", rationale="not covered")
    assert received == [ev1, ev2]                       # push mode got both
    polled = emitter.poll_events()                      # poll drains the same queue
    assert len(polled) == 2
    assert polled[-1]["payload"]["action"] == "ASK_TEACHER"
    assert emitter.poll_events() == []                  # no replay
    with pytest.raises(SchemaError, match="not in NAV_ACTIONS"):
        emitter.queue_nav_action("FLY", "sc-3")          # closed enum, explicit
    with pytest.raises(SchemaError):
        emitter.queue_dream("pondered", "d1")            # not a dream kind


# --------------------------------------------------------------------------- #
# 4. action authority checks
# --------------------------------------------------------------------------- #

def test_actions_denials_are_structured_and_logged(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    actions = GymActions(bundle, clock=clock)
    r = actions.request("teach", {"text": "x"}, authenticated=False)
    assert r["ok"] is False and r["reason"] == "unauthenticated"
    r = actions.request("teleport", {}, authenticated=True)
    assert r["ok"] is False and r["reason"] == "unknown_action"
    r = actions.request("teach", {"text": ""}, authenticated=True)
    assert r["ok"] is False and r["reason"] == "bad_args"
    r = actions.request("approve_request",
                        {"op": "SELF_DESTRUCT", "args": {}}, authenticated=True)
    assert r["ok"] is False and r["reason"] == "unknown_operation"
    r = actions.request("teach", {"text": "x"}, authenticated=True)   # no dialogue wired
    assert r["ok"] is False and r["reason"] == "no_handler"
    assert len(actions.log) == 5                       # every request logged


def test_actions_pause_resume_gates(tmp_path):
    bundle, clock, dialogue, *_ = make_bundle(tmp_path)
    session, _student, _pathway = make_dialogue()
    bundle.dialogue = session
    actions = GymActions(bundle, clock=clock)
    r = actions.request("pause", {}, authenticated=True)
    assert r["result"]["paused"] is True
    for act, args in (("teach", {"text": CHAIN_LESSON}),
                      ("ask", {"question": ASK_Q}),
                      ("correct", {"correction": "No - Tav routes to Vex, not Kimo."}),
                      ("approve_request", {"op": "CHECK_RESOURCES", "args": {}}),
                      ("deny_request", {"request_id": "r1", "op": "WRITE_FILE"})):
        r = actions.request(act, args, authenticated=True)
        assert r["ok"] is False and r["reason"] == "paused", act
    r = actions.request("resume", {}, authenticated=True)     # pause controls never blocked
    assert r["result"]["paused"] is False
    r = actions.request("teach", {"text": CHAIN_LESSON}, authenticated=True)
    assert r["ok"] is True and r["result"]["n_facts"] == 4


def test_actions_teach_ask_correct_through_real_dialogue(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    session, student, pathway = make_dialogue()
    bundle.dialogue = session
    actions = GymActions(bundle, clock=clock)
    r = actions.request("teach", {"text": CHAIN_LESSON}, authenticated=True)
    assert r["ok"] is True and r["result"]["n_facts"] == 4
    r = actions.request("ask", {"question": ASK_Q}, authenticated=True)
    assert r["result"]["verdict"] == "confirmed"
    assert r["result"]["student_answer"].strip().lower() == ASK_A.lower()
    r = actions.request("correct",
                        {"correction": "No - Tav routes to relay Vex, not Kimo."},
                        authenticated=True)
    assert r["ok"] is True and r["result"]["n_corrections"] == 1
    c = r["result"]["corrections"][0]
    assert c["kind"] == "update" and c["after"]["object"] == "Vex"
    assert c["before"]["object"] == "Kimo"
    assert session.corrections.ledger.current("Tav", "routes-to").object == "Vex"


def test_actions_approve_request_routes_through_the_real_gate(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    actions = GymActions(bundle, clock=clock)
    r = actions.request("approve_request",
                        {"op": "WRITE_FILE", "args": {"path": "gym.txt",
                                                      "content": "hello"},
                         "arm": "godot"}, authenticated=True)
    assert r["ok"] is True
    assert r["result"]["outcome_status"] == "executed"
    assert r["result"]["outcome_reason"] is None
    r = actions.request("approve_request",
                        {"op": "REQUEST_CAPACITY",
                         "args": {"resource": "per_write_bytes", "requested": 8192}},
                        authenticated=True)
    assert r["result"]["outcome_status"] == "authority_denied"    # no demonstrated need
    assert r["result"]["outcome_reason"] == "no_demonstrated_need"
    r = actions.request("approve_request",
                        {"op": "WRITE_FILE",
                         "args": {"path": "big.txt", "content": "x" * 4096}},
                        authenticated=True)
    assert r["result"]["outcome_status"] == "authority_denied"    # THE GATE said no
    assert r["result"]["outcome_reason"] == "capacity_exceeded"
    r = actions.request("approve_request",
                        {"op": "REQUEST_CAPACITY",
                         "args": {"resource": "per_write_bytes", "requested": 8192}},
                        authenticated=True)
    assert r["result"]["outcome_status"] == "executed"            # need demonstrated
    assert r["result"]["grant"]["resource"] == "per_write_bytes"
    records = bundle.host.records
    assert [rec["arm"] for rec in records] == ["godot"] * 4


def test_actions_deny_request_never_reaches_the_gate_and_emits(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", host=bundle.host),
                         clock=clock)
    actions = GymActions(bundle, emitter=emitter, clock=clock)
    before = len(bundle.host.records)
    r = actions.request("deny_request",
                        {"request_id": "req-7", "op": "RUN_SCRIPT",
                         "reason": "host veto"}, authenticated=True)
    assert r["ok"] is True and r["result"]["denied"] == "req-7"
    assert len(bundle.host.records) == before           # nothing was forwarded
    events = emitter.poll_events()
    assert len(events) == 1 and events[0]["type"] == "hostmode.request"
    p = events[0]["payload"]
    assert p["status"] == "denied_by_gym" and p["op"] == "RUN_SCRIPT"
    assert p["reason"] == "host veto"


def test_actions_set_goal_stays_in_bridge_state(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", devhistory=bundle.devhistory),
                         clock=clock)
    actions = GymActions(bundle, emitter=emitter, clock=clock)
    r = actions.request("set_goal", {"goal": "answer reach questions"},
                        authenticated=True)
    assert r["ok"] is True and actions.goal_state["goal"] == "answer reach questions"
    snap = emitter.snapshot()
    assert snap["payload"]["goal"]["goal"] == "answer reach questions"
    # the devhistory ledger was NOT written by the bridge (goal is bridge state
    # for the runtime lane to fold when it chooses)
    kinds = [e["kind"] for e in bundle.devhistory.entries]
    assert kinds == ["birth"]


# --------------------------------------------------------------------------- #
# 5. ZERO plastic writes
# --------------------------------------------------------------------------- #

def test_observation_is_pure_read_strict_proxy(tmp_path):
    bundle, clock, relational, *_ = make_bundle(tmp_path)
    relational.set_strength("a|operates|b", 1.0, 0.2, 0.9)   # the TEST plays the runtime
    relational.mutation_callers.clear()                     # baseline: nothing seen yet
    proxy = StrictReadProxy(relational, allowed=("inventory",))
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", relational=proxy),
                         clock=clock)
    emitter.snapshot()
    emitter.poll_events()
    emitter.poll_events()
    touched = {name for (_, name) in proxy.accesses}
    assert touched == {"inventory"}                     # ONE read API, nothing else
    assert relational.mutation_callers == []            # no plastic mutation at all


def test_snapshot_and_poll_cause_no_plastic_mutation(tmp_path):
    bundle, clock, dialogue, *_ = make_bundle(tmp_path)
    session, student, pathway = make_dialogue()
    session.teach(CHAIN_LESSON)                         # pre-existing plastic state
    bundle.dialogue = session
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", relational=StubRelational(),
                                       episodic=bundle.episodic,
                                       devhistory=bundle.devhistory,
                                       host=bundle.host),
                         clock=clock)
    before = pathway.presentations
    for _ in range(3):
        emitter.snapshot()
        emitter.poll()
    assert pathway.presentations == before              # zero plastic mutation


def test_all_plastic_mutation_flows_through_runtime_paths(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    session, student, pathway = make_dialogue()
    bundle.dialogue = session
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", episodic=bundle.episodic,
                                       devhistory=bundle.devhistory,
                                       host=bundle.host),
                         clock=clock)
    actions = GymActions(bundle, emitter=emitter, clock=clock)
    assert actions.request("teach", {"text": CHAIN_LESSON}, authenticated=True)["ok"]
    assert actions.request("ask", {"question": ASK_Q}, authenticated=True)["ok"]
    assert actions.request(
        "correct", {"correction": "No - Tav routes to relay Vex, not Kimo."},
        authenticated=True)["ok"]
    assert pathway.presentations == 4 + 3               # 4 teach facts + 3 correction repeats
                                                        # (ask only reads: forward, no update)
    # every plastic presentation was driven by the dialogue learn path
    # (Student.learn_fact) — never by a gym_bridge frame
    assert set(pathway.mutation_callers) == {"aion.dialogue.student"}
    assert not any(c.startswith("aion.gym_bridge")
                   for c in pathway.mutation_callers)


# --------------------------------------------------------------------------- #
# 6. server
# --------------------------------------------------------------------------- #

def test_handle_action_request_envelope_units(tmp_path):
    bundle, clock, *_ = make_bundle(tmp_path)
    emitter = GymEmitter(RuntimeBundle(instance_id="t1"), clock=clock)
    actions = GymActions(bundle, emitter=emitter, clock=clock)
    server = GymServer(emitter, actions, token="sekrit")
    out = server.handle_action_request("not-a-dict", authenticated=True)
    assert out["type"] == "action.error"
    out = server.handle_action_request({"type": "chat", "text": "hi"},
                                       authenticated=True)
    assert out["type"] == "action.error" and "action.request" in out["detail"]
    out = server.handle_action_request(
        {"type": "action.request", "action": "set_goal", "args": {"goal": "g"},
         "request_id": "wire-1"}, authenticated=True)
    assert out["type"] == "action.result" and out["ok"] is True
    assert out["request_id"] == "wire-1"


def test_server_loopback_websocket_and_http(tmp_path):
    import websockets
    bundle, clock, *_ = make_bundle(tmp_path)
    session, _student, _pathway = make_dialogue()
    bundle.dialogue = session
    emitter = GymEmitter(RuntimeBundle(instance_id="t1", episodic=bundle.episodic,
                                       devhistory=bundle.devhistory,
                                       host=bundle.host),
                         clock=clock)
    actions = GymActions(bundle, emitter=emitter, clock=clock)
    server = GymServer(emitter, actions, host="127.0.0.1", token="sekrit")
    ports = {}

    async def scenario():
        ports.update(await server.start_async())
        try:
            await _exercise(server, ports)
        finally:
            await server.stop_async()          # same loop: the ws Server lives here

    async def _exercise(server, ports):
        ws_url = f"ws://127.0.0.1:{ports['ws_port']}"
        async with websockets.connect(ws_url) as ws:
            snap = deserialize(await asyncio.wait_for(ws.recv(), 5))
            assert snap["type"] == "state.snapshot"
            assert snap["payload"]["instance_id"] == "t1"
            # an authenticated action
            await ws.send(json.dumps({"type": "action.request", "action": "set_goal",
                                      "args": {"goal": "reach heat"},
                                      "request_id": "w1", "token": "sekrit"}))
            res = json.loads(await asyncio.wait_for(ws.recv(), 5))
            assert res["type"] == "action.result" and res["ok"] is True
            assert res["result"]["goal"] == "reach heat"
            # an UNauthenticated action (bad token) -> structured denial
            await ws.send(json.dumps({"type": "action.request", "action": "set_goal",
                                      "args": {"goal": "evil"}, "token": "wrong"}))
            res = json.loads(await asyncio.wait_for(ws.recv(), 5))
            assert res["ok"] is False and res["reason"] == "unauthenticated"
        # HTTP side
        base = f"http://127.0.0.1:{ports['http_port']}"
        health = json.load(urllib.request.urlopen(f"{base}/health", timeout=5))
        assert health["ok"] is True and health["schema_version"] == 1
        assert health["actions_logged"] >= 2
        snap = deserialize(urllib.request.urlopen(f"{base}/snapshot", timeout=5).read())
        assert snap["type"] == "state.snapshot"
        events = json.load(urllib.request.urlopen(f"{base}/events", timeout=5))
        assert isinstance(events["events"], list)
        # POST /action without token -> 403 with structured reason
        req = urllib.request.Request(f"{base}/action", method="POST",
                                     data=json.dumps({"action": "set_goal",
                                                      "args": {"goal": "x"}}).encode())
        try:
            urllib.request.urlopen(req, timeout=5)
            raised = False
        except urllib.error.HTTPError as exc:
            raised = True
            body = json.load(exc)
            assert exc.code == 403 and body["reason"] == "unauthenticated"
        assert raised
        # POST /action with token -> executed through the REAL hostmode gate
        req = urllib.request.Request(
            f"{base}/action", method="POST",
            data=json.dumps({"action": "approve_request",
                             "args": {"op": "CHECK_RESOURCES", "args": {}}}).encode(),
            headers={"X-AION-Token": "sekrit"})
        res = json.load(urllib.request.urlopen(req, timeout=5))
        assert res["ok"] is True
        assert res["result"]["outcome_status"] == "executed"

    async def finish():
        await server.stop_async()

    asyncio.run(scenario())
    assert emitter.goal_state.get("goal") == "reach heat"   # only the authed one landed
