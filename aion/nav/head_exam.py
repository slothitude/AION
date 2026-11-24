"""NAV head exam - the amended preregistered protocol (owner amendments A1+A2).

Runs BOTH frozen exam variants:
  E1 ORIGINAL  aion.nav.exam.build_scenarios() verbatim (seed 20261125,
               n_chains=8, 30 scenarios, 6 per gold class)
  E2 FRESH     aion.nav.nav_head_prereg.build_fresh_scenarios() (seed
               20261126, n_chains=10, unseen world, 30 scenarios)

Scored backend rows (frozen):
  d1_asis          PRIMARY - d1-2B as-is, decomposed card usage (A1).
                   fp32 CPU ~7.7 GB -> run on LAPPY only.
  trained_head     ARM 2 - the NAV-domain head (artifacts/nav_decision_head.pkl)
  heuristic_floor  reference line (rule table; confidence None -> redo-exempt)
  larql_classifier reference - the larql M6 head via the owner-approved socket
  tev1_0p8b        reference row - LICENSE-BLOCKED, never runs

AMENDMENT A2 redo rule (applies to every scored backend with a calibrated
confidence): confidence in [0.49, 0.51] -> re-present the item with the next
frozen formulation (0->1->2), max 2 re-presentations (3 attempts); the FINAL
attempt is scored, whatever it gives. Redos never change gold. Reported per
backend: redo_count (total re-presentations) and ambiguity_rate (fraction of
items that hit the band). Feature-vector heads are formulation-invariant
(declared): their redos are bounded no-ops.

Backends never see gold. Results go to aion/nav/artifacts/ - the original
preregistered aion/nav/results/ outputs stay untouched.

CLI (Rog):
    python -m aion.nav.head_exam --backends trained_head,heuristic_floor,larql_classifier
CLI (Lappy, ssa-train-venv):
    python -m aion.nav.head_exam --backends d1_asis,heuristic_floor
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Dict, List, Optional, Tuple

from aion.nav.backends import (
    BACKEND_NAMES,
    LarqlClassifierBackend,
    DecisionRecord,
    nav_decision_policy,
)
from aion.nav.exam import build_scenarios
from aion.nav.nav_head_prereg import (
    CLASSES,
    HEAD_EXAM_VERSION,
    N_FORMS,
    PREREG_DATE,
    REDO_BAND_HIGH,
    REDO_BAND_LOW,
    REDO_MAX_REPRESENTATIONS,
    AMENDMENTS,
    build_fresh_scenarios,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "artifacts")

REQUIRED_RECORD_FIELDS = (
    "schema_version", "backend", "scenario_id", "goal", "options", "outcome",
    "gold", "counterfactual", "latency_ms", "compliance")


# ---------------------------------------------------------------------------
# backend wrappers: uniform present(state, form) ->
#   (choice, confidence, aux, outcome)
# ---------------------------------------------------------------------------

class HeuristicPresent:
    """Reference floor: the gold rule table itself. No calibrated probability
    exists (rule-determined) -> confidence None -> exempt from the redo band
    (declared in the preregistration)."""

    name = "heuristic_floor"

    def present(self, state, form: int
                ) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        return nav_decision_policy(state), None, {
            "form_invariant": True, "confidence_declared": None}, "decided"


class TrainedHeadPresent:
    """ARM 2: the trained NAV head. Formulation-invariant redo (declared)."""

    name = "trained_head"

    def __init__(self, artifact_path: str) -> None:
        import joblib                      # sklearn lane, Rog only
        self.head = joblib.load(artifact_path)

    def present(self, state, form: int
                ) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        return self.head.present(state, form)


class LarqlPresent:
    """Reference: the larql M6 head through the portable socket. Redo is a
    declared no-op (feature-vector backend)."""

    name = "larql_classifier"

    def __init__(self, artifact_path: str, meta_path: Optional[str]) -> None:
        self.backend = LarqlClassifierBackend(artifact_path=artifact_path,
                                              meta_path=meta_path)
        if not self.backend.load():
            raise RuntimeError(f"larql artifact failed to load: "
                               f"{self.backend.error}")

    def present(self, state, form: int
                ) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        rec = self.backend.decide(state)
        if rec.outcome != "decided":
            return None, None, {"error": rec.raw}, rec.outcome
        conf = rec.counterfactual.get("head_confidence")
        return rec.choice, (float(conf) if conf is not None else None), {
            "form_invariant": True, "head_raw": rec.raw,
            "counterfactual": rec.counterfactual}, rec.outcome


class Tev1Present:
    """Reference row: LICENSE-BLOCKED. The card says the fine-tuned-weights
    release license is being finalized; this backend NEVER runs and records
    outcome=license_blocked for every item."""

    name = "tev1_0p8b"

    def present(self, state, form: int
                ) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        return None, None, {"license": "blocked per card; no run"}, \
            "license_blocked"


class D1AsIsPresent:
    """PRIMARY arm (A1): d1-2B as-is, decomposed. Text backend - redos
    re-present the state under the next frozen formulation."""

    name = "d1_asis"

    def __init__(self, model_path: str) -> None:
        from aion.nav.d1_asis import D1AsIsBackend
        self.backend = D1AsIsBackend(model_path=model_path)

    def load(self) -> Tuple[bool, Optional[str]]:
        ok = self.backend.load()
        return ok, self.backend.error

    def present(self, state, form: int
                ) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        return self.backend.present(state, form)


def build_present(name: str, args) -> object:
    if name == "heuristic_floor":
        return HeuristicPresent()
    if name == "trained_head":
        return TrainedHeadPresent(args.head_artifact)
    if name == "larql_classifier":
        return LarqlPresent(args.larql_artifact, args.larql_meta)
    if name == "tev1_0p8b":
        return Tev1Present()
    if name == "d1_asis":
        return D1AsIsPresent(args.d1_path)
    raise ValueError(f"unknown scored backend {name!r}")


SCORED_BACKEND_NAMES = ("d1_asis", "trained_head", "heuristic_floor",
                        "larql_classifier", "tev1_0p8b")


# ---------------------------------------------------------------------------
# the amended runner (A2 redo rule)
# ---------------------------------------------------------------------------

def run_item(present, state, gold: str) -> Dict[str, object]:
    """One exam item under the amended protocol: up to 3 attempts (form
    0 -> 1 -> 2), redo when the deciding confidence lands in the frozen band;
    the FINAL attempt is scored; gold never enters the backend."""
    attempts: List[Dict[str, object]] = []
    choice: Optional[str] = None
    confidence: Optional[float] = None
    outcome = "unavailable"
    aux: Dict[str, object] = {}
    n_redos = 0
    for attempt in range(REDO_MAX_REPRESENTATIONS + 1):
        form = min(attempt, N_FORMS - 1)
        t0 = time.perf_counter()
        choice, confidence, aux, outcome = present.present(state, form)
        latency = (time.perf_counter() - t0) * 1000.0
        attempts.append({
            "attempt": attempt + 1, "form": form, "choice": choice,
            "confidence": confidence, "outcome": outcome,
            "latency_ms": round(latency, 3),
        })
        if outcome != "decided" or confidence is None:
            break                      # nothing to re-present on
        if not (REDO_BAND_LOW <= confidence <= REDO_BAND_HIGH):
            break                      # decisive attempt - keep it
        if attempt < REDO_MAX_REPRESENTATIONS:
            n_redos += 1               # bounded re-presentation, fresh form
    correct = bool(outcome == "decided" and choice == gold)
    if choice is not None and choice not in CLASSES and outcome == "decided":
        pass                           # enum violation recorded below
    return {
        "choice": choice, "confidence": confidence, "outcome": outcome,
        "aux": aux, "attempts": attempts, "redo_count": n_redos,
        "ambiguous": n_redos > 0, "correct": correct,
    }


def run_backend(name: str, present, scenarios, variant: str = ""
                ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    records: List[Dict[str, object]] = []
    t_start = time.perf_counter()
    enum_violations = 0
    for sc in scenarios:
        res = run_item(present, sc.state, sc.gold)
        if res["outcome"] == "decided" and res["choice"] not in CLASSES:
            enum_violations += 1
        rec = {
            "schema_version": "nav-head-exam-1.0",
            "backend": name,
            "exam_variant": variant,
            "scenario_id": sc.state.scenario_id,
            "goal": sc.state.goal_text,
            "options": list(CLASSES),
            "choice": res["choice"],
            "outcome": res["outcome"],
            "goal_achieved": res["correct"],
            "gold": sc.gold,
            "counterfactual": {"slot": None, "score": res["confidence"],
                               "source": f"{name}_deciding_confidence"},
            "confidence": res["confidence"],
            "attempts": res["attempts"],
            "redo_count": res["redo_count"],
            "ambiguous": res["ambiguous"],
            "enum_violation": bool(res["outcome"] == "decided"
                                   and res["choice"] not in CLASSES),
            "latency_ms": round(sum(a["latency_ms"] for a in res["attempts"]),
                                3),
            "compliance": ["amended protocol: A2 bounded redos, final attempt "
                           "scored; gold never an input"],
            "aux": res["aux"],
        }
        missing = [f for f in REQUIRED_RECORD_FIELDS if f not in rec]
        rec["record_complete"] = not missing
        records.append(rec)
    wall_s = time.perf_counter() - t_start

    scored = [r for r in records if r["outcome"] == "decided"]
    correct = sum(1 for r in records if r["goal_achieved"])
    by_class: Dict[str, Dict[str, int]] = {}
    for r in records:
        slot = by_class.setdefault(str(r["gold"]),
                                   {"n": 0, "correct": 0, "violations": 0,
                                    "misses": 0})
        slot["n"] += 1
        if r["goal_achieved"]:
            slot["correct"] += 1
        elif r["enum_violation"]:
            slot["violations"] += 1
        else:
            slot["misses"] += 1
    confs = [r["confidence"] for r in records if r["confidence"] is not None]
    summary = {
        "backend": name,
        "accuracy": round(correct / len(records), 4) if records else None,
        "accuracy_scored": round(correct / len(scored), 4) if scored else None,
        "n_scenarios": len(records),
        "n_scored": len(scored),
        "n_correct": correct,
        "enum_violations": enum_violations,
        "no_answer": sum(1 for r in records if r["outcome"] == "no_answer"),
        "license_blocked": sum(1 for r in records
                               if r["outcome"] == "license_blocked"),
        "unavailable": sum(1 for r in records if r["outcome"] == "unavailable"),
        "redo_count": int(sum(r["redo_count"] for r in records)),
        "n_ambiguous": int(sum(1 for r in records if r["ambiguous"])),
        "ambiguity_rate": round(sum(1 for r in records if r["ambiguous"])
                                / len(records), 4) if records else None,
        "confidence_mean": round(sum(confs) / len(confs), 4) if confs else None,
        "latency_ms_mean": round(sum(r["latency_ms"] for r in records)
                                 / len(records), 2),
        "wall_s": round(wall_s, 2),
        "records_complete": sum(1 for r in records if r["record_complete"]),
        "by_gold_class": by_class,
        "machine": f"{platform.system()} {platform.release()} {platform.node()}",
    }
    return records, summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="NAV head exam (amended protocol)")
    ap.add_argument("--backends", default="trained_head,heuristic_floor",
                    help=f"comma list from {SCORED_BACKEND_NAMES}")
    ap.add_argument("--exams", default="original,fresh",
                    help="comma list: original, fresh")
    ap.add_argument("--d1-path", default="LiquidAI/d1-2B")
    ap.add_argument("--head-artifact",
                    default=os.path.join(ARTIFACT_DIR, "nav_decision_head.pkl"))
    ap.add_argument("--larql-artifact",
                    default=r"C:\Users\aaron\larql-d1\reports\d1_adapters_artifacts"
                            r"\d1_action_classifier.pkl")
    ap.add_argument("--larql-meta",
                    default=r"C:\Users\aaron\larql-d1\reports\d1_adapters_artifacts"
                            r"\d1_action_classifier_meta.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exam_sets = {}
    if "original" in args.exams:
        exam_sets["original"] = build_scenarios()
    if "fresh" in args.exams:
        exam_sets["fresh"] = build_fresh_scenarios()
    for label, scs in exam_sets.items():
        gold_counts: Dict[str, int] = {}
        for sc in scs:
            gold_counts[sc.gold] = gold_counts.get(sc.gold, 0) + 1
        print(f"exam[{label}] n={len(scs)} gold={gold_counts}")

    results: Dict[str, object] = {
        "exam_version": HEAD_EXAM_VERSION,
        "preregistered": PREREG_DATE,
        "amendments": list(AMENDMENTS),
        "redo_band": [REDO_BAND_LOW, REDO_BAND_HIGH],
        "redo_max_representations": REDO_MAX_REPRESENTATIONS,
        "machine": f"{platform.system()} {platform.release()} {platform.node()}",
        "exam_gold_counts": {
            k: {sc.gold: 1 for sc in v} for k, v in exam_sets.items()},
        "summaries": [], "records": [],
    }

    for name in [n.strip() for n in args.backends.split(",") if n.strip()]:
        if name not in SCORED_BACKEND_NAMES:
            print(f"!! unknown scored backend {name!r}; skipping")
            continue
        try:
            present = build_present(name, args)
            load_note = ""
            if hasattr(present, "load"):
                ok, err = present.load()          # type: ignore[attr-defined]
                load_note = "load ok" if ok else f"load FAILED - {err}"
                if not ok:
                    print(f"[{name}] {load_note}")
                    continue
        except Exception as exc:
            print(f"[{name}] setup FAILED - {type(exc).__name__}: {exc}")
            continue
        for label, scs in exam_sets.items():
            records, summary = run_backend(name, present, scs, variant=label)
            summary["exam_variant"] = label
            results["summaries"].append(summary)          # type: ignore[union-attr]
            results["records"].extend(records)            # type: ignore[union-attr]
            print(f"[{name}/{label}] acc={summary['accuracy']} "
                  f"({summary['n_correct']}/{summary['n_scenarios']}) "
                  f"scored={summary['n_scored']} viol={summary['enum_violations']} "
                  f"redo_count={summary['redo_count']} "
                  f"ambiguity={summary['ambiguity_rate']} "
                  f"latency_ms={summary['latency_ms_mean']}")
            for cls, slot in sorted(summary["by_gold_class"].items()):
                print(f"    gold {cls:<12} {slot['correct']}/{slot['n']} "
                      f"(viol={slot['violations']} miss={slot['misses']})")

    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(ARTIFACT_DIR, f"nav_head_exam_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(results, fh, indent=2, default=str)
        fh.write("\n")
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
