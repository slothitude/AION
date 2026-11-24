"""Unit tests for the developmental-history ledger (CPU, seeded, stubs only).

Owner acceptance (2026-11-24): begin -> record a few of each event type ->
transitions appear as NAMED nodes in the NOMEN graph -> biography renders
markdown + JSON -> persistence round-trip EXACT -> dream/growth stubs hold
events without dreaming anything. NO network, NO model calls, NO dream engine
(anything importable or not).

Run:
    python -m pytest aion/memory/test_devhistory.py -q        # pytest
    python aion/memory/test_devhistory.py                     # standalone
"""

import json
import math
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.memory.devhistory import DevelopmentalHistory, _jsonable   # noqa: E402
from aion.memory.episodic import EpisodicStore                       # noqa: E402
from aion.nomen.evolution_tracker import TransitionEvent             # noqa: E402

FIXED_TS = 1700000000.0


class StepClock:
    """Clock that steps one second per call: birth, then every event later."""

    def __init__(self, start: float = FIXED_TS):
        self.value = float(start)

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def make_history(instance_id="alpha", **kw):
    substrate = {"brain": "pith-e2b", "artifact_sha256": "abc123", "bpw": 1.5}
    return DevelopmentalHistory.begin(instance_id, substrate, clock=StepClock(),
                                      seed=42, **kw)


def stub_inventory_rows():
    """Rows shaped exactly like RelationalMemory.inventory() (landed format)."""
    return [
        {"subject": "spark", "relation": "melts", "object": "ice",
         "key": "spark|melts|ice", "pair_index": 0,
         "u_fast_norm": 0.4, "theta_slow_norm": 0.1,
         "full": 2.75, "plastic": 2.5, "consolidated": 0.8,
         "registered_at_seq": 1, "superseded_by": None, "meta": {"update_ids": [1, 2]}},
        {"subject": "ice", "relation": "freezes", "object": "water",
         "key": "ice|freezes|water", "pair_index": 0,
         "u_fast_norm": 0.2, "theta_slow_norm": 0.3,
         "full": 1.9, "plastic": 0.9, "consolidated": 1.7,
         "registered_at_seq": 3, "superseded_by": None, "meta": {}},
    ]


def stub_retention_report():
    """Shaped like the retention probe-phase report (aion/experiments/retention.py)."""
    return {"variant": "C", "battery_id": "tiny", "phase": "probe",
            "restart": {"loaded_from_disk_only": True, "state_restored_exact": True},
            "summary": {"internalisation_score": 0.5, "internalisation_tests": 4,
                        "internalisation_passes": 2}}


def stub_discovery_report():
    """Shaped like the knowledge_discovery run_battery report, inf survival
    included on purpose (the plain-JSON sanitiser must hold)."""
    return {"battery_id": "tiny", "seed": 0,
            "with_host": {"scores": {"recognise": 1.0}},
            "without_host": {"scores": {"recognise": 1.0}},
            "survival_ratio": {"recognise": 1.0, "predict": 1.0, "learn_new": math.inf,
                               "combine": 0.5, "self_correct": 1.0},
            "verdict": {"recognise": "fully-survives", "learn_new":
                        "impossible-gain(check-scoring)"}}


# --------------------------------------------------------------------------- #
# 1. begin: identity, substrate, birth as a named state
# --------------------------------------------------------------------------- #

def test_begin_opens_a_biography():
    hist = make_history()
    assert hist.instance_id == "alpha"
    assert hist.substrate == {"brain": "pith-e2b", "artifact_sha256": "abc123", "bpw": 1.5}
    assert hist.name == "AION-alpha"          # initial identity SUPPLIED (NOMEN law)

    ent = hist.tracker.self_entity
    assert ent.entity_kind == "self"
    assert ent.verification_status == "observed"
    assert len(ent.lineage) == 1 and ent.lineage[0].provenance == "initial"

    # the birth is itself the first event, attached to the first NAMED node
    assert len(hist.entries) == 1
    assert hist.entries[0]["kind"] == "birth"
    assert hist.entries[0]["payload"]["substrate"]["artifact_sha256"] == "abc123"
    node = hist.nodes[0]
    assert node.name == "AION-alpha" and node.node_id == hist.entries[0]["node_id"]
    assert node.events[0]["kind"] == "birth"

    # scalar substrate facts became observed properties (name seed material)
    assert ent.observed_properties["brain"] == "pith-e2b"

    # birth timestamp is from the injected clock (deterministic) and is exactly
    # the birth event's own ts — the biography's "Born" line matches the walk
    assert hist.birth_ts == hist.entries[0]["ts"] > FIXED_TS


def test_begin_rejects_non_json_substrate():
    try:
        DevelopmentalHistory.begin("beta", {"bad": object()}, clock=StepClock())
    except ValueError as exc:
        assert "plain JSON" in str(exc)
    else:
        raise AssertionError("non-JSON substrate must be refused at begin()")


# --------------------------------------------------------------------------- #
# 2. every event type routes through NOMEN and lands as a named node event
# --------------------------------------------------------------------------- #

def record_one_of_each(hist):
    episodic = EpisodicStore(clock=StepClock(FIXED_TS + 100))
    ep = episodic.record(kind="lesson", text="spark melts ice", channel="transcript",
                         teacher="battery", plastic_update_ids=[1, 2],
                         tags={"answer": "ice"})
    hist.record_experience(ep)
    hist.record_relationship_snapshot(stub_inventory_rows())
    hist.record_test_result(stub_retention_report(), label="strict restart")
    hist.record_test_result({"n_events": 2, "n_violations": 0, "invariant_ok": True,
                             "total_consolidated_entries": 5})
    hist.record_dream({"stage": "learn", "n_lessons": 3})
    hist.record_growth({"milestone": "chain-1 complete"})
    hist.record_independence(stub_discovery_report())


def test_every_event_type_is_recorded_through_nomen():
    hist = make_history()
    before = len(hist.tracker.events)
    record_one_of_each(hist)
    kinds = [e["kind"] for e in hist.entries]
    assert kinds == ["birth", "experience", "relationships", "test", "test",
                     "dream", "growth", "independence"]
    assert len(hist.tracker.events) == before + 7          # NOMEN saw them all
    # each ledger entry's node exists in the graph and carries the event
    for e in hist.entries:
        node = hist.graph.node(e["node_id"])
        assert any(int(ev["seq"]) == e["seq"] for ev in node.events)


def test_experience_payload_keeps_provenance():
    hist = make_history()
    episodic = EpisodicStore(clock=StepClock(FIXED_TS + 100))
    ep = episodic.record(kind="correction", text="ice freezes water", teacher="reality",
                         channel="transcript", plastic_update_ids=[9],
                         tags={"replaces": "spark|melts|ice"})
    hist.record_experience(ep)
    e = hist.entries[-1]
    assert e["payload"]["episode"]["teacher"] == "reality"
    assert e["payload"]["episode"]["plastic_update_ids"] == [9]
    assert e["description"] == "correction (transcript, teacher reality): ice freezes water"


def test_relationship_snapshot_records_states_and_strengths():
    hist = make_history()
    hist.record_relationship_snapshot(stub_inventory_rows())
    e = hist.entries[-1]
    assert e["payload"]["n_relations"] == 2
    assert e["payload"]["n_transmitting"] == 2
    row = e["payload"]["rows"][0]
    assert row["key"] == "spark|melts|ice"
    assert (row["full"], row["plastic"], row["consolidated"]) == (2.75, 2.5, 0.8)
    assert e["description"] == "knowledge inventory: 2 relations indexed, 2 transmitting"


def test_test_result_accepts_landed_report_shapes_and_sanitises():
    hist = make_history()
    hist.record_test_result(stub_retention_report(), label="strict restart")
    hist.record_independence(stub_discovery_report())
    retention, independence = hist.entries[-2], hist.entries[-1]
    assert "retention test, variant C" in retention["description"]
    assert retention["payload"]["report"]["restart"]["state_restored_exact"] is True
    # inf survival sanitised to a string: plain JSON survives exactly
    assert independence["payload"]["report"]["survival_ratio"]["learn_new"] == "inf"
    assert "learn_new=inf" in independence["description"]
    json.dumps(hist.to_dict(), allow_nan=False)   # the whole payload is strict JSON


# --------------------------------------------------------------------------- #
# 3. named stages: renames grow NAMED nodes in the NOMEN graph
# --------------------------------------------------------------------------- #

def test_rename_grows_named_stages_in_the_nomen_graph():
    hist = make_history()
    assert len(hist.nodes) == 1
    # the instance proposes its next name from its own observed properties
    proposals = hist.tracker.propose_name(include_teacher=False)
    assert proposals, "NOMEN must propose observation-derived candidates"
    chosen = proposals[0].name
    # the rename happens THROUGH the ledger (record_transition -> NOMEN seam)
    hist.record_transition(
        TransitionEvent(kind="growth", description="grew past its first chain"),
        new_name=chosen)
    assert len(hist.nodes) == 2                     # two NAMED stages now
    assert hist.nodes[1].name == chosen
    assert hist.name == chosen                      # registry rename landed
    assert hist.tracker.self_entity.verification_status == "observed"  # safeguard
    assert hist.tracker.self_entity.entity_id == hist.nodes[0].entity_id  # identity
    # the event that caused the rename is attached to the NEW named state
    last = hist.entries[-1]
    assert last["node_id"] == hist.nodes[1].node_id
    assert last["node_name"] == chosen
    # and one transition edge connects the stages with a GENERATED label
    edges = [e for e in hist.graph.edges() if e.entity_id == hist.tracker.self_entity_id]
    assert len(edges) == 1
    assert edges[0].src == hist.nodes[0].node_id and edges[0].dst == hist.nodes[1].node_id
    assert edges[0].label                           # NOMEN-generated verb phrase


def test_stub_growth_with_new_name_becomes_a_named_stage():
    hist = make_history()
    chosen = hist.tracker.propose_name(include_teacher=False)[0].name
    hist.record_growth({"milestone": "first consolidation"}, new_name=chosen)
    assert hist.nodes[-1].name == chosen
    assert hist.entries[-1]["kind"] == "growth"


# --------------------------------------------------------------------------- #
# 4. biography: markdown + machine JSON
# --------------------------------------------------------------------------- #

def test_biography_renders_markdown():
    hist = make_history()
    record_one_of_each(hist)
    chosen = hist.tracker.propose_name(include_teacher=False)[0].name
    hist.record_growth({"milestone": "first consolidation"}, new_name=chosen)
    md = hist.biography()

    assert md.startswith(f"# {hist.name} — a developmental history")
    assert "## Starting substrate" in md
    assert "artifact_sha256: abc123" in md          # substrate record shown
    assert "**Instance:** alpha" in md
    # stage headings are exactly NOMEN's names — nothing hardcoded
    for node in hist.nodes:
        assert f"### {node.name}" in md
    assert "### First Light" not in md              # no invented stage names
    # every event kind appears with its description
    assert "**birth**" in md and "**experience**" in md
    assert "spark melts ice" in md
    assert "**dream**" in md and "stub" in md.lower()
    assert "learn_new=inf" in md                    # independence survival line
    # the transition between stages is rendered with NOMEN's generated label
    edge = hist.graph.edges()[-1]
    assert f"--> *{edge.label}*" in md and chosen in md


def test_biography_json_round_trips():
    hist = make_history()
    record_one_of_each(hist)
    blob = hist.biography_json()
    assert blob["instance_id"] == "alpha"
    assert blob["current_name"] == hist.name
    assert len(blob["nodes"]) == len(hist.nodes) == 1
    assert len(blob["entries"]) == len(hist.entries) == 8
    assert blob["edges"] == []                      # no renames yet: no transition edges
    json.dumps(blob, allow_nan=False)               # machine view is strict JSON


# --------------------------------------------------------------------------- #
# 5. persistence: exact round-trip, and the ledger keeps recording after load
# --------------------------------------------------------------------------- #

def test_persistence_round_trip_exact(tmp_path):
    hist = make_history()
    record_one_of_each(hist)
    chosen = hist.tracker.propose_name(include_teacher=False)[0].name
    hist.record_growth({"milestone": "first consolidation"}, new_name=chosen)
    path = str(tmp_path / "runs" / "alpha" / "devhistory.json")
    hist.save(path)

    loaded = DevelopmentalHistory.load(path, clock=StepClock(FIXED_TS + 1000))
    assert hist.equals(loaded), "restart must restore the biography EXACTLY"
    assert loaded.to_dict() == hist.to_dict()
    # name, lineage, substrate, entries, named nodes, edges all survive
    assert loaded.name == chosen
    assert loaded.substrate == hist.substrate
    assert [n.name for n in loaded.nodes] == [n.name for n in hist.nodes]
    assert loaded.entries[-1]["payload"]["event"] == \
        {"milestone": "first consolidation"}

    # byte-stability: two saves of the same state are byte-identical
    path2 = str(tmp_path / "runs" / "alpha" / "devhistory2.json")
    loaded.save(path2)
    with open(path, "rb") as a, open(path2, "rb") as b:
        assert a.read() == b.read()

    # and the loaded instance keeps living: the tracker sequence continues
    chosen2 = loaded.tracker.propose_name(include_teacher=False)[0].name
    loaded.record_dream({"stage": "post-restart"}, new_name=chosen2)
    assert loaded.name == chosen2
    assert len(loaded.entries) == len(hist.entries) + 1


def test_jsonable_sanitiser_law():
    assert _jsonable(float("inf")) == "inf" and _jsonable(float("-inf")) == "-inf"
    assert _jsonable(float("nan")) == "nan"
    assert _jsonable((1, 2)) == [1, 2]
    assert _jsonable({"a": True}) == {"a": True}
    import torch
    t = torch.tensor([1.0, 2.0])
    assert _jsonable(t) == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# 6. the stubs are honest: dream/growth hold events WITHOUT dreaming anything
# --------------------------------------------------------------------------- #

def test_dream_and_growth_stubs_hold_events_without_dreaming():
    before_modules = set(sys.modules)
    hist = make_history()
    hist.record_dream({"stage": "learn", "n_lessons": 3})
    hist.record_growth({"milestone": "chain-1 complete"})
    # no dream mechanism, no model mechanism, nothing new imported at all
    new_modules = set(sys.modules) - before_modules
    offenders = [m for m in new_modules if "dream" in m or "nnfs" in m]
    assert not offenders, f"stub hooks must import nothing: {offenders}"
    # events are FILED with the honest locked marker
    dream = hist.entries[-2]
    growth = hist.entries[-1]
    assert dream["payload"]["engine"] is None
    assert growth["payload"]["engine"] is None
    assert dream["payload"]["dream"] == "stub-locked"
    assert growth["payload"]["growth"] == "stub-locked"
    assert dream["payload"]["event"] == {"stage": "learn", "n_lessons": 3}
    assert growth["payload"]["event"] == {"milestone": "chain-1 complete"}
    assert dream["tags"]["stub"] == "locked" and growth["tags"]["stub"] == "locked"
    # nothing was invented: the graph only grew events, no dreaming side effects
    assert len(hist.nodes) == 1 and len(hist.graph.edges()) == 0


# --------------------------------------------------------------------------- #
# standalone entry (mirrors test_apparatus.py)
# --------------------------------------------------------------------------- #

def main() -> int:
    import pytest
    here = os.path.abspath(__file__)
    rc = pytest.main(["-q", here])
    return 0 if rc in (0, pytest.ExitCode.NO_TESTS_COLLECTED) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
