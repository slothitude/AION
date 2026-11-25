"""Tests for the AION-1 INTERVENTION EXPERIMENT (aion/experiments/interventions.py).

Covers, without any network:
  * the PREREGISTERED reroute plan is deterministic, valid (every reroute changes the
    edge) and matches the frozen formula in the module docstring;
  * gold derivation: only the rerouted actor's reach gold changes per epoch; route gold
    is the new relay; the heuristic control reproduces every gold (the ceiling is exact
    by construction);
  * the intervention statement shape extracts as exactly ONE intervention (the proven
    RuleExtractor path);
  * ablation (retention.ablated) restores blend ratios and gate values after scoring;
  * a FULL tiny-standin END-TO-END run (teach -> baseline -> reroutes -> report) with
    incremental JSONL writes, plastic-override evidence per reroute, and a report whose
    aggregates are internally consistent.

Run:  python -m pytest aion/experiments/test_interventions.py -q
"""

import json
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.dialogue.lesson_extractor import extract                    # noqa: E402
from aion.worlds.generator import generate_world, verify_world        # noqa: E402
from aion.experiments.interventions import (                          # noqa: E402
    INTERVENTION_STATEMENT, N_REROUTES, InterventionRun, apply_plan, heuristic_answer,
    reach_gold, reroute_plan, route_gold)
from aion.experiments.retention import ablated, CONSOLIDATED, FULL, PLASTIC  # noqa: E402
from aion.experiments.teacher_student import build_tiny_foundation    # noqa: E402
from aion.relational.interface import InterfaceConfig, RelationalInterface  # noqa: E402


# --------------------------------------------------------------------------- #
# The frozen plan
# --------------------------------------------------------------------------- #

def test_preregistered_plan_is_deterministic_and_valid():
    world = generate_world(seed=0, n_chains=5)
    assert verify_world(world) == []
    plan1 = reroute_plan(world, N_REROUTES)
    plan2 = reroute_plan(world, N_REROUTES)
    assert plan1 == plan2 and len(plan1) == N_REROUTES
    # every reroute changes the edge, and the cumulative walk never repeats a relay
    current = {c: world.relays[i] for i, c in enumerate(world.conduits)}
    for item in plan1:
        assert item["new_relay"] != current[item["conduit"]]
        assert item["old_relay"] == current[item["conduit"]]
        current[item["conduit"]] = item["new_relay"]


def test_plan_matches_preregistered_formula():
    """reroutes 1..5 = conduits[i] -> relays[(i+1)%5]; reroutes 6..8 = conduits[i] ->
    relays[(i+3)%5] for i in (0, 2, 4) — the docstring formula."""
    world = generate_world(seed=0, n_chains=5)
    plan = reroute_plan(world, N_REROUTES)
    for k, i in enumerate(list(range(5)) + [0, 2, 4]):
        offset = 1 if k < 5 else 3
        assert plan[k]["conduit"] == world.conduits[i]
        assert plan[k]["new_relay"] == world.relays[(i + offset) % 5]


def test_intervention_statement_extracts_as_one_intervention():
    res = extract(INTERVENTION_STATEMENT.format(conduit="Tav", new_relay="Drel"))
    assert len(res.interventions) == 1
    # the intervention fact carries provenance="intervention", so DialogueSession.teach's
    # plain-fact pass sees NO plain facts (only the plastic-override path fires)
    assert len(res.facts) == 1 and res.facts[0].provenance == "intervention"
    iv = res.interventions[0]
    assert (iv.subject, iv.kind, iv.new_object) == ("Tav", "reroute", "Drel")
    assert (iv.fact.subject, iv.fact.relation, iv.fact.object) == ("Tav", "routes-to", "Drel")


# --------------------------------------------------------------------------- #
# Gold derivation + heuristic control
# --------------------------------------------------------------------------- #

def test_gold_changes_only_for_affected_actor():
    world = generate_world(seed=0, n_chains=5)
    applied = apply_plan(reroute_plan(world, N_REROUTES), world.graph)
    for k, item in enumerate(applied):
        g = item["graph"]
        g_prev = applied[k - 1]["graph"] if k else world.graph
        for actor in world.actors:
            changed = reach_gold(actor, g) != reach_gold(actor, g_prev)
            assert changed == (actor == item["actor"]), (k, actor)
        # the rerouted actor lands on the new chain's output via the new relay
        assert reach_gold(item["actor"], g) == item["gold_reach"]
        assert route_gold(item["actor"], g) == item["gold_route"] == item["new_relay"]


def test_heuristic_control_matches_gold_everywhere():
    """The graph-lookup control answers every measured question correctly by construction:
    the epoch-0 plain reach set plus every epoch's own intervention questions."""
    world = generate_world(seed=0, n_chains=5)
    plan = reroute_plan(world, N_REROUTES)
    applied = apply_plan(plan, world.graph)

    checks = []                                   # (question, gold, graph)
    for actor in world.actors:
        checks.append((f"What does {actor} reach?", reach_gold(actor, world.graph),
                       world.graph))
    for item in applied:
        checks.append((f"What does {item['actor']} reach now?", item["gold_reach"],
                       item["graph"]))
        checks.append((f"Which relay does {item['actor']} route through now?",
                       item["gold_route"], item["graph"]))
    for q, gold, g in checks:
        assert heuristic_answer(q, world.actors, g) == gold, q


# --------------------------------------------------------------------------- #
# Ablation restores live pathway state
# --------------------------------------------------------------------------- #

def test_scoring_ablation_restores_pathway_state():
    model, tok = build_tiny_foundation(seed=0)
    iface = RelationalInterface(model, cfg=InterfaceConfig())
    blends_before = [c.blend_ratio for c in iface.conns]
    gates_before = [g.detach().clone() for g in iface.gates]
    with ablated(iface.conns, iface.gates, PLASTIC):
        assert all(c.blend_ratio == 1.0 for c in iface.conns)
    with ablated(iface.conns, iface.gates, CONSOLIDATED):
        assert all(c.blend_ratio == 0.0 for c in iface.conns)
    with ablated(iface.conns, iface.gates, FULL):
        pass
    assert [c.blend_ratio for c in iface.conns] == blends_before
    assert all(torch.equal(a, b) for a, b in zip(iface.gates, gates_before))


# --------------------------------------------------------------------------- #
# Full tiny end-to-end run (2 reroutes; the machinery, not the exam config)
# --------------------------------------------------------------------------- #

def _make_args(tmp_path, n_reroutes=2):
    from types import SimpleNamespace
    return SimpleNamespace(
        foundation="tiny", local_dir=None, device="cpu", seed=0, chains=5,
        n_reroutes=n_reroutes, learning_rate=0.5, gate_steps=6, gate_lr=300.0,
        answer_tokens=6, correction_repeats=3,
        run_dir=str(tmp_path / "interventions-tiny"))


def test_full_tiny_run_end_to_end(tmp_path):
    args = _make_args(tmp_path)
    run = InterventionRun(args)
    assert run.run() == 0
    run_dir = args.run_dir

    # -- artifacts -- #
    for name in ("results.jsonl", "report.json", "world.json"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    rows = [json.loads(line) for line in
            open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8")]
    kinds = {}
    for r in rows:
        kinds[r["row"]] = kinds.get(r["row"], 0) + 1
    assert kinds["prereg_plan"] == 1
    assert kinds["teach_summary"] == 1
    assert kinds["reroute"] == 2
    # 5 stability (epoch 0) + 2 reroutes x (2 intervention + 5 stability) = 19 questions
    assert kinds["question"] == 5 + 2 * 7
    assert kinds["epoch_summary"] == 3

    # -- teaching evidence -- #
    teach = next(r for r in rows if r["row"] == "teach_summary")
    assert teach["facts_learned"] == 20
    assert teach["weak_facts"] == []
    assert any(u > 0 for u in teach["u_fast_norms_after"])
    assert teach["theta_slow_empty"]

    # -- plastic-override evidence per reroute -- #
    reroutes = [r for r in rows if r["row"] == "reroute"]
    for r in reroutes:
        assert r["intervention_applied_as_plastic_override"]
        assert r["new_fact_learned"] and r["new_fact_plastic_strength"] > 0
        assert r["old_key_superseded"]
        assert r["old_facts_all_present"]

    # -- report aggregates are internally consistent -- #
    report = json.load(open(os.path.join(run_dir, "report.json"), encoding="utf-8"))
    iq = report["aggregates"]["intervention_questions"]
    assert iq["n"] == 4 and iq["chance"] == 0.2
    assert iq["heuristic_ceiling"] == 1.0            # the control is exact by construction
    for m in (FULL, PLASTIC, CONSOLIDATED):
        assert 0.0 <= iq["modes"][m]["acc@1"] <= 1.0
    assert iq["plastic_ablation_delta"] == round(
        iq["modes"][PLASTIC]["acc@1"] - iq["modes"][CONSOLIDATED]["acc@1"], 4)
    stab = report["aggregates"]["stability"]
    assert stab["n_affected_comparisons"] == 2 * 3          # 2 reroutes x 3 forward modes
    assert stab["n_unaffected_comparisons"] == 2 * 4 * 3    # 2 reroutes x 4 other actors x 3
    assert report["theta_slow_empty"]
    assert report["foundation_integrity"]["bit_identical_after_full_run"]
    assert report["verdict"]["heuristic_ceiling_ok"]

    # -- every scored question row is complete (3 forward modes + retrieval + heuristic) -- #
    qrows = [r for r in rows if r["row"] == "question"]
    for r in qrows:
        assert set(r["modes"]) == {FULL, PLASTIC, CONSOLIDATED}
        assert r["gold"] in r["candidates"]
        assert len(r["heuristic_answer"]) > 0


def test_run_is_repeatable_from_the_preregistered_plan(tmp_path):
    """Two runs, same seed: identical plan rows in the JSONL (the frozen-config law)."""
    dirs = []
    for i in range(2):
        args = _make_args(tmp_path / f"run{i}", n_reroutes=1)
        run = InterventionRun(args)
        run.run()
        dirs.append(args.run_dir)
    plans = []
    for d in dirs:
        rows = [json.loads(line) for line in
                open(os.path.join(d, "results.jsonl"), encoding="utf-8")]
        plans.append(next(r for r in rows if r["row"] == "prereg_plan")["plan"])
    assert plans[0] == plans[1]
