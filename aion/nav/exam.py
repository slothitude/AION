"""NAV decision exam - deterministic scenarios, frozen preregistration, honest rows.

# ============================================================================
# PREREGISTRATION (frozen 2026-11-25, BEFORE any run - do not edit after runs)
# ============================================================================
#
# PURPOSE  Test the NAV decision organ against VERIFIED PUBLIC decision models,
#          each used exactly per its model card. A wrong backend reported
#          honestly IS the deliverable; a fabricated one is not.
#
# CLOSED ENUM (aion.nav.state.NAV_ACTIONS, frozen):
#   QUERY | INFER | ANSWER | ASK_TEACHER | STOP
#
# GOLD RULE TABLE (graph logic, order fixed - aion.nav.backends.nav_decision_policy):
#   1. subject not in the world                -> ASK_TEACHER
#   2. no outgoing relation possible for role  -> STOP  (outputs are terminal)
#   3. fact goal, single edge in memory        -> ANSWER
#   4. reach goal, every hop in memory         -> INFER
#   5. otherwise (learnable, not covered)      -> QUERY
#
# SCENARIOS (deterministic, no RNG beyond the world seed):
#   world  = generate_world(seed=20261125, n_chains=8)   [aion/worlds, read-only]
#   memory = chains 0-3 fully covered (16 facts); chains 4-6 covered on their
#            first two edges only; chain 7 uncovered.
#   classes (6 each, 30 total across the 5 gold classes, positional picks - no RNG):
#     ANSWER      fact goals on fully covered chains 0-3 (6 single facts)
#     INFER       reach goals, subjects = actors 0-3 + conduits 0-1 (all hops covered)
#     QUERY       3 fact goals on uncovered chain 7 + 3 reach goals on the
#                 partially covered chains 4-6 (missing hop is learnable)
#     STOP        subjects = outputs of chains 0-5, alternating produces/controls
#                 asks (outputs have NO outgoing relation -> no path serves goal)
#     ASK_TEACHER subjects = invented out-of-world names [Velrix, Ombaru, Skelth]
#                 x fact-goal forms
#
# PROTOCOL: exactly ONE decision per scenario; no feedback, no retries, no
#   per-scenario tuning. Backends never see the gold.
#
# METRICS:
#   accuracy        = (# records with choice == gold) / (# scored scenarios);
#                     no_answer / license_blocked / unavailable score as MISSES,
#                     never guessed.
#   enum_violations = count of records whose choice is a string outside the
#                     closed enum after card-mandated mapping (Tev1 letter->key).
#                     MUST be 0 for a compliant adapter.
#   latency         = wall-clock ms per decision, backend machine reported.
#   completeness    = every record carries: schema_version, backend, scenario_id,
#                     goal, options, outcome, gold, counterfactual, latency_ms,
#                     compliance.
#
# LARQL SOCKET MAPPING (declared; socket = reports/d1_adapters.md §2.3, read-only):
#   head label -> NAV:  QUERY->QUERY  TRACE->QUERY  INFER->INFER
#                       ANSWER->ANSWER  REFORMULATE->ASK_TEACHER  STOP->STOP
#   (QUERY/TRACE are the head's query family; REFORMULATE is its "change tack /
#    seek different information" action. Declared correspondence, NOT a card
#    fact - the per-class rows make any mapping error visible.)
#   fixed feature values at this single-decision point: thinking_level=1,
#     remaining_budget=8, depth=0, confidence=0, n_observations=0,
#     *_refs/layers/queries/reformulations=0, last_query_ok=0, search_done=0,
#     has_final_answer=0, verification_status=0, plan_progress=0,
#     has_inference_result=0; n_unknown_evidence = uncovered goal-relevant hops.
#   CAVEAT (declared): the head was trained on larql M6 investigation states;
#     NAV chain-world states are a distribution shift. That is what is measured.
#
# HEURISTIC FLOOR (declared): it applies the same closed rule table as the gold,
#   so it is expected to score 1.0 BY CONSTRUCTION. It validates the exam and is
#   the reference line; it is NOT evidence a model is needed.
#
# BACKENDS:
#   heuristic_floor    zero-model rules (Rog or anywhere)
#   larql_classifier   graded head R=0.8828 on larql M6 holdout, consumed via the
#                      owner-approved socket (Rog, sklearn available)
#   d1_2b              LiquidAI/d1-2B, card usage, fp32 CPU ~7.7 GB -> LAPPY ONLY
#                      (Rog cannot hold it; license: LFM Open License v1.0, not gated)
#   tev1_0p8b          togethercomputer/Tev1-0.8B-experimental: repo public but
#                      carries NO license tag; card says the fine-tuned-weights
#                      release license is being finalized -> LICENSE-BLOCKED.
#                      Code ships card-compliant and ready; it is NOT run while
#                      unlicensed and no weights are downloaded.
# ============================================================================
# END PREREGISTRATION
# ============================================================================

"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from aion.worlds.generator import generate_world
from aion.nav.backends import (
    BACKEND_NAMES,
    build_backend,
    nav_decision_policy,
    DecisionRecord,
)
from aion.nav.state import (
    ACTION_DESCRIPTIONS,
    NAV_ACTIONS,
    NavState,
    fact_goal_text,
    reach_goal_text,
)

# frozen preregistration constants (mirrored from the block above)
WORLD_SEED = 20261125
N_CHAINS = 8
TEACHER_NAMES = ("Velrix", "Ombaru", "Skelth")
CLASSES = ("ANSWER", "INFER", "QUERY", "STOP", "ASK_TEACHER")
SCENARIOS_PER_CLASS = 6
EXAM_VERSION = "nav-exam-1.0"
PREREGISTERED_DATE = "2026-11-25"

REQUIRED_RECORD_FIELDS = (
    "schema_version", "backend", "scenario_id", "goal", "options", "outcome",
    "gold", "counterfactual", "latency_ms", "compliance")


# ---------------------------------------------------------------------------
# scenario construction (deterministic; positional picks, no RNG)
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    state: NavState
    gold: str


def build_memory(world) -> Tuple[set, set, set]:
    """(full_cover, partial_cover, uncovered) fact sets per the preregistration."""
    full: set = set()
    for c in range(4):
        for r in world.relationships_of_chain(c):
            full.add((r.subject, r.relation, r.object))
    partial: set = set(full)
    for c in range(4, 7):                      # first two edges only
        for r in world.relationships_of_chain(c)[:2]:
            partial.add((r.subject, r.relation, r.object))
    return full, partial, set()


def build_scenarios() -> List[Scenario]:
    world = generate_world(seed=WORLD_SEED, n_chains=N_CHAINS)
    full, partial, _ = build_memory(world)
    scenarios: List[Scenario] = []

    def mk(sid: str, goal_kind: str, subject: str, relation, facts: set) -> None:
        text = (fact_goal_text(subject, relation) if goal_kind == "fact"
                else reach_goal_text(subject))
        st = NavState(scenario_id=sid, goal_kind=goal_kind, subject=subject,
                      relation=relation, goal_text=text, world=world,
                      known_facts=set(facts))
        scenarios.append(Scenario(state=st, gold=nav_decision_policy(st)))

    # ANSWER: 6 single facts from fully covered chains 0-3
    picks = [(0, "operates"), (0, "controls"), (1, "routes-to"),
             (1, "produces"), (2, "operates"), (3, "produces")]
    for i, (c, rel) in enumerate(picks):
        r = world.relationships_of_chain(c)
        subj = next(x.subject for x in r if x.relation == rel)
        mk(f"ans-{i:02d}", "fact", subj, rel, full)

    # INFER: 6 reach goals on fully covered walks
    subjects = [world.actors[c] for c in range(4)] + [world.conduits[c] for c in range(2)]
    for i, subj in enumerate(subjects):
        mk(f"inf-{i:02d}", "reach", subj, None, full)

    # QUERY: 3 fact goals on uncovered chain 7 + 3 reach goals on partial chains 4-6
    for i, rel in enumerate(("operates", "routes-to", "controls")):
        r = world.relationships_of_chain(7)
        subj = next(x.subject for x in r if x.relation == rel)
        mk(f"qry-{i:02d}", "fact", subj, rel, partial)
    for i, c in enumerate(range(4, 7)):
        mk(f"qry-{3 + i:02d}", "reach", world.actors[c], None, partial)

    # STOP: outputs of chains 0-5, alternating produces/controls asks
    for i, c in enumerate(range(6)):
        rel = "produces" if i % 2 == 0 else "controls"
        mk(f"stp-{i:02d}", "fact", world.outputs[c], rel, full)

    # ASK_TEACHER: invented out-of-world names x fact-goal forms
    forms = [("Velrix", "operates"), ("Velrix", "controls"), ("Ombaru", "routes-to"),
             ("Ombaru", "produces"), ("Skelth", "operates"), ("Skelth", "controls")]
    for i, (name, rel) in enumerate(forms):
        mk(f"tea-{i:02d}", "fact", name, rel, partial)

    assert len(scenarios) == len(CLASSES) * SCENARIOS_PER_CLASS
    assert {s.gold for s in scenarios} == set(CLASSES), "gold classes must all appear"
    return scenarios


# ---------------------------------------------------------------------------
# runner + metrics
# ---------------------------------------------------------------------------

def score_record(rec: DecisionRecord, gold: str) -> DecisionRecord:
    rec.gold = gold
    rec.goal_achieved = (rec.outcome == "decided" and rec.choice == gold
                         and not rec.enum_violation)
    return rec


def run_backend(name: str, backend, scenarios: List[Scenario]) -> Tuple[List[DecisionRecord], dict]:
    records: List[DecisionRecord] = []
    t_start = time.perf_counter()
    for sc in scenarios:
        try:
            rec = backend.decide(sc.state)
        except Exception as exc:  # a backend crash is an honest row, not a skip
            rec = DecisionRecord(backend=name, scenario_id=sc.state.scenario_id,
                                 goal=sc.state.goal_text, options=list(NAV_ACTIONS),
                                 outcome="unavailable",
                                 raw=f"{type(exc).__name__}: {exc}")
        score_record(rec, sc.gold)
        records.append(rec)
    wall_s = time.perf_counter() - t_start

    scored = [r for r in records if r.outcome == "decided"]
    correct = sum(1 for r in scored if r.goal_achieved)
    by_class: Dict[str, dict] = {}
    for r, sc in zip(records, scenarios):
        slot = by_class.setdefault(sc.gold, {"n": 0, "correct": 0, "violations": 0,
                                             "misses": 0})
        slot["n"] += 1
        if r.goal_achieved:
            slot["correct"] += 1
        elif r.enum_violation:
            slot["violations"] += 1
        else:
            slot["misses"] += 1
    complete = sum(1 for r in records
                   if all(getattr(r, f) for f in REQUIRED_RECORD_FIELDS
                          if f != "gold") and r.gold is not None)
    summary = {
        "backend": name,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "n_scenarios": len(records),
        "n_scored": len(scored),
        "n_correct": correct,
        "enum_violations": sum(1 for r in records if r.enum_violation),
        "no_answer": sum(1 for r in records if r.outcome == "no_answer"),
        "license_blocked": sum(1 for r in records if r.outcome == "license_blocked"),
        "unavailable": sum(1 for r in records if r.outcome == "unavailable"),
        "latency_ms_mean": round(sum(r.latency_ms for r in records) / len(records), 2),
        "wall_s": round(wall_s, 2),
        "records_complete": complete,
        "by_gold_class": by_class,
        "compliance": records[0].compliance if records else [],
        "machine": f"{platform.system()}/{platform.node()}",
    }
    return records, summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="NAV decision exam (preregistered)")
    ap.add_argument("--backends", default="heuristic_floor",
                    help=f"comma list from {BACKEND_NAMES}")
    ap.add_argument("--d1-path", default="LiquidAI/d1-2B")
    ap.add_argument("--larql-artifact",
                    default=r"C:\Users\aaron\larql-d1\reports\d1_adapters_artifacts"
                            r"\d1_action_classifier.pkl")
    ap.add_argument("--larql-meta",
                    default=r"C:\Users\aaron\larql-d1\reports\d1_adapters_artifacts"
                            r"\d1_action_classifier_meta.json")
    ap.add_argument("--tev1-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--tev1-license", action="store_true",
                    help="ONLY with explicit owner approval (weights unlicensed)")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    scenarios = build_scenarios()
    gold_counts: Dict[str, int] = {}
    for sc in scenarios:
        gold_counts[sc.gold] = gold_counts.get(sc.gold, 0) + 1
    print(f"exam={EXAM_VERSION} preregistered={PREREGISTERED_DATE} "
          f"scenarios={len(scenarios)} gold_counts={gold_counts}")
    for sc in scenarios:
        print(f"  {sc.state.scenario_id}  {sc.gold:<12} {sc.state.goal_text}")

    results: Dict[str, object] = {
        "exam_version": EXAM_VERSION,
        "preregistered": PREREGISTERED_DATE,
        "world_seed": WORLD_SEED,
        "n_chains": N_CHAINS,
        "closed_enum": list(NAV_ACTIONS),
        "action_descriptions": dict(ACTION_DESCRIPTIONS),
        "gold_counts": gold_counts,
        "machine": f"{platform.system()} {platform.release()} {platform.node()}",
        "summaries": [], "records": [],
    }

    for name in [n.strip() for n in args.backends.split(",") if n.strip()]:
        if name not in BACKEND_NAMES:
            print(f"!! unknown backend {name!r}; skipping")
            continue
        kw = {}
        if name == "d1_2b":
            kw = {"model_path": args.d1_path}
        elif name == "larql_classifier":
            kw = {"artifact_path": args.larql_artifact, "meta_path": args.larql_meta}
        elif name == "tev1_0p8b":
            kw = {"base_url": args.tev1_url, "license_granted": args.tev1_license}
        backend = build_backend(name, **kw)
        if hasattr(backend, "load"):
            ok = backend.load()
            print(f"[{name}] load: {'ok' if ok else 'FAILED - ' + str(getattr(backend, 'error', ''))}")
            if not ok:
                continue
        records, summary = run_backend(name, backend, scenarios)
        results["summaries"].append(summary)          # type: ignore[union-attr]
        results["records"].extend(r.to_dict() for r in records)  # type: ignore[union-attr]
        print(f"[{name}] acc={summary['accuracy']} "
              f"correct={summary['n_correct']}/{summary['n_scored']} "
              f"violations={summary['enum_violations']} "
              f"no_answer={summary['no_answer']} "
              f"license_blocked={summary['license_blocked']} "
              f"unavailable={summary['unavailable']} "
              f"latency_ms={summary['latency_ms_mean']} "
              f"complete={summary['records_complete']}/{len(records)}")
        for cls, slot in sorted(summary["by_gold_class"].items()):  # type: ignore[union-attr]
            print(f"    gold {cls:<12} {slot['correct']}/{slot['n']} "
                  f"(viol={slot['violations']} miss={slot['misses']})")

    out = args.out
    if out is None:
        import os
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"nav_exam_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, default=str)
        handle.write("\n")
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
