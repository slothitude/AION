"""Tests for the ownership SCALE run (aion/experiments/ownership_scale.py).

The scale question (100 relationships) needs only proven machinery, so these tests
prove the HARNESS wiring on the tiny CPU stand-in: preregistered plan helpers are
pure functions of the world, and one full mini-cycle (teach -> test -> correct ->
retain/restart -> purge -> report) runs end to end with every mechanical gate green
and every JSONL battery file carrying exactly the asked rows. The REAL 100-rel
numbers come from the Lappy run (runs/ownership_scale/), never from these tests.
"""

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.worlds.generator import BATTERY_DIRECT, BATTERY_PARAPHRASE, generate_world, verify_world
from aion.experiments.ownership_scale import (
    OwnershipScaleRun, correction_plan, direct_battery, grading_row,
    paraphrase_subset, probe_set, stride_pick, summarize_rows)


# --------------------------------------------------------------------------- #
# Preregistered plan helpers (pure, no model)
# --------------------------------------------------------------------------- #

def test_world_at_scale_is_sound():
    world = generate_world(seed=0, n_chains=25)
    assert len(world.relationships) == 100
    assert len(world.questions) == 325   # 100 direct + 75 combo + 100 para + 50 intervention
    assert verify_world(world) == []


def test_direct_battery_is_full_count():
    world = generate_world(seed=0, n_chains=25)
    qs = direct_battery(world)
    assert len(qs) == 100
    assert all(q.battery == BATTERY_DIRECT for q in qs)


def test_paraphrase_subset_stride_is_deterministic():
    world = generate_world(seed=0, n_chains=25)
    paras = [q for q in world.questions if q.battery == BATTERY_PARAPHRASE]
    sub = paraphrase_subset(world, 20)
    assert len(sub) == 20
    assert sub == stride_pick(paras, 20)
    assert [q.id for q in sub] == [f"para-{i:03d}" for i in range(0, 100, 5)]
    # same world -> identical subset (preregistration means frozen, not re-derived)
    assert paraphrase_subset(generate_world(seed=0, n_chains=25), 20) == sub


def test_stride_pick_edges():
    qs = list(range(7))
    assert stride_pick(qs, 0) == qs
    assert stride_pick(qs, 10) == qs
    assert stride_pick(qs, 3) == [0, 2, 4]


def test_correction_plan_ten_midchain_reroutes():
    world = generate_world(seed=0, n_chains=25)
    plan = correction_plan(world, 10)
    assert len(plan) == 10
    assert len({p["conduit"] for p in plan}) == 10          # distinct conduits
    for p in plan:
        assert p["new_relay"] != p["old_relay"]
        assert p["new_relay"] in world.relays
        assert p["sentence"] == f"{p['conduit']} is rerouted to relay {p['new_relay']}."
        assert p["old_key"] != p["new_key"]
        assert p["old_key"] == f"{p['conduit']}|routes-to|{p['old_relay']}"
    # mid-chain: every corrected subject is a conduit (the chain's routes-to edge)
    assert all(p["conduit"] in world.conduits for p in plan)


def test_probe_set_one_direct_question_per_chain():
    world = generate_world(seed=0, n_chains=25)
    probes = probe_set(world, 20)
    assert len(probes) == 20
    assert [q.chain_id for q in probes] == list(range(20))
    assert all(q.battery == BATTERY_DIRECT for q in probes)


def test_grading_row_provenance_priority():
    world = generate_world(seed=0, n_chains=3)
    q = [x for x in world.questions if x.battery == BATTERY_DIRECT][0]
    gold = q.answer
    row = grading_row(q, {"FULL": gold, "PLASTIC": gold, "CONSOLIDATED": gold,
                          "RETRIEVED": gold}, world)
    assert row["provenance_exact"] == "PLASTIC"             # strongest-first priority
    row = grading_row(q, {"FULL": gold, "PLASTIC": "", "CONSOLIDATED": "",
                          "RETRIEVED": ""}, world)
    assert row["provenance_exact"] == "FULL_ONLY"
    row = grading_row(q, {"FULL": "", "PLASTIC": "", "CONSOLIDATED": "",
                          "RETRIEVED": ""}, world)
    assert row["provenance_exact"] == "NONE"
    summ = summarize_rows([row])
    assert summ["n"] == 1 and summ["provenance_histogram"] == {"NONE": 1}


# --------------------------------------------------------------------------- #
# Full mini-cycle on the tiny stand-in (harness wiring, fast on CPU)
# --------------------------------------------------------------------------- #

def _run(capsys, tmp_path):
    class Args:
        foundation = "tiny"
        local_dir = None
        device = "cpu"
        seed = 0
        chains = 3
        para_subset = 4
        n_corrections = 2
        probe_size = 3
        learning_rate = 0.5
        gate_steps = 1
        gate_lr = 300.0
        answer_tokens = 8
        correction_repeats = 3
        run_dir = str(tmp_path / "ownership_tiny")
    run = OwnershipScaleRun(Args())
    assert run.run() == 0
    return run


def test_full_small_run_end_to_end(tmp_path):
    run = _run(None, tmp_path)
    rd = run.run_dir
    world = run.world
    n_direct = len(direct_battery(world))
    n_para = len(paraphrase_subset(world, 4))

    # -- artifacts exist, and battery JSONLs carry EXACTLY the asked rows -- #
    assert os.path.exists(os.path.join(rd, "report.json"))
    for phase in ("protocol", "teach", "test", "corrections", "restart", "retain",
                  "purge"):
        assert os.path.exists(os.path.join(rd, f"phase_{phase}.json")), phase
    assert os.path.getsize(os.path.join(rd, "battery_test.jsonl")) > 0
    assert sum(1 for _ in open(os.path.join(rd, "battery_test.jsonl"))) == \
        n_direct + n_para
    assert sum(1 for _ in open(os.path.join(rd, "battery_survival-pre.jsonl"))) == \
        n_direct + n_para
    assert sum(1 for _ in open(os.path.join(rd, "battery_survival-post.jsonl"))) == \
        n_direct + n_para

    report = json.load(open(os.path.join(rd, "report.json"), encoding="utf-8"))

    # -- TEACH: every fact stored in the plastic pathway -- #
    assert report["teach"]["facts_learned"] == 4 * world.n_chains
    assert report["teach"]["all_100_facts_stored"] is True
    assert any(u > 0 for u in report["teach"]["u_fast_norms_after"])

    # -- CORRECT: without-destroying counts are exact -- #
    corr = report["corrections"]
    n_corr = 2
    assert corr["new_facts_learned"] == f"{n_corr}/{n_corr}"
    assert corr["untouched_old_facts"] == 4 * world.n_chains - n_corr
    assert corr["untouched_old_facts_intact"] == \
        f"{4 * world.n_chains - n_corr}/{4 * world.n_chains - n_corr}"
    assert corr["adaptation_ok"] is True            # strengths-based gate
    assert "unaffected_probe_identical" in corr     # answer identity reported as diagnostic
    assert "probe_identity_note" in corr

    # -- RETAIN: strict restart is bitwise + survival per battery -- #
    assert report["retain"]["restart"]["state_restored_exact"] is True
    assert report["retain"]["restart"]["loaded_from_disk_only"] is True
    assert report["retain"]["restart"]["fact_index_reloaded"] == \
        4 * world.n_chains + n_corr
    for battery, per in report["retain"]["per_battery"].items():
        assert per["survival_ok"] is True, battery
        assert all(v == 1.0 for v in per["answers_identical_fraction"].values()), battery

    # -- PURGE: plastic answers unchanged, transcript gone -- #
    purge = report["transcript_purge"]
    assert purge["purge"]["removed"] == report["teach"]["episodic_events"] + n_corr
    assert purge["plastic_unchanged"] is True

    # -- foundation untouched by the whole run -- #
    assert report["foundation_integrity"]["bit_identical_after_full_run"] is True

    # -- all four mechanical gates green on the mini-cycle -- #
    assert report["acceptance_score"] == "4/4 mechanical gates"


def test_run_is_repeatable_from_preregistered_plan(tmp_path):
    """The persisted protocol file matches the live plan (frozen before learning)."""
    run = _run(None, tmp_path)
    proto = json.load(open(os.path.join(run.run_dir, "phase_protocol.json"),
                           encoding="utf-8"))
    assert proto["preregistered"] is True
    assert proto["corrections"] == correction_plan(run.world, run.args.n_corrections)
    assert proto["direct_battery_n"] == len(direct_battery(run.world))
    assert proto["paraphrase_subset_n"] == len(paraphrase_subset(run.world, 4))
    assert proto["batteries_not_measured_note"].startswith("combination+intervention")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
