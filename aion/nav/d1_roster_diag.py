"""POST-HOC DIAGNOSTIC (clearly labelled - NOT a preregistered exam row).

Question the diagnostic answers: is d1-as-is's QUERY failure (0/6 on both
exam variants, routed ASK_TEACHER via gate_in_world on uncovered subjects) a
MODEL gap or a STATE-EXACTNESS gap? The preregistered state text lists only
memory facts, so "is the subject an entity of this world?" is unanswerable
from the given evidence for any subject absent from memory - and d1 answers
that unanswerable question confidently (p~0.03-0.08), never entering the
0.49-0.51 redo band.

The diagnostic re-derives the state from the graph (the owner amendment's own
words): same four gates, same thresholds, same redo rule - but the state text
now also carries the WORLD ROSTER and the role->relation table. If d1's gates
then route correctly, the failure was representation, not capacity; if not,
it is a genuine zero-training model limitation. Gold is never shown.

CLI (Lappy):
    python -m aion.nav.d1_roster_diag --d1-path /path/to/LiquidAI_d1-2B
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

from aion.nav.exam import build_scenarios
from aion.nav.nav_head_prereg import (
    REDO_BAND_HIGH,
    REDO_BAND_LOW,
    REDO_MAX_REPRESENTATIONS,
    N_FORMS,
    build_fresh_scenarios,
    render_state_text,
)
from aion.nav.d1_asis import D1AsIsBackend, route

ROLE_RELATION_TABLE = (
    "- actor -> operates\n"
    "- conduit -> routes-to\n"
    "- relay -> controls\n"
    "- device -> produces\n"
    "- output -> (none; outputs are terminal)")


def render_state_roster(state, form: int) -> str:
    """Diagnostic state: world roster + role->relation table + memory facts
    + goal question. Everything derived from the graph; gold never enters."""
    w = state.world
    lines: List[str] = ["World roster (every entity of this world, by role):"]
    lines.append(f"- actors: {', '.join(w.actors)}")
    lines.append(f"- conduits: {', '.join(w.conduits)}")
    lines.append(f"- relays: {', '.join(w.relays)}")
    lines.append(f"- devices: {', '.join(w.devices)}")
    lines.append(f"- outputs: {', '.join(w.outputs)}"
                 " (outputs are terminal phenomena)")
    lines.append("Role -> possible outgoing relation in this world:")
    lines.append(ROLE_RELATION_TABLE)
    lines.append(render_state_text(state, form))
    return "\n".join(lines)


def run_variant(backend: D1AsIsBackend, scenarios, label: str) -> List[dict]:
    rows: List[dict] = []
    for sc in scenarios:
        attempts: List[dict] = []
        choice, conf, stage = None, None, None
        outcome = "unavailable"
        for attempt in range(REDO_MAX_REPRESENTATIONS + 1):
            form = min(attempt, N_FORMS - 1)
            t0 = time.perf_counter()
            try:
                # THE diagnostic difference: roster-enriched state text
                out = backend.attempt(
                    sc.state, form,
                    state_text=render_state_roster(sc.state, form))
                probs = out["probs"]
                residual = out["residual"]
                choice, conf, stage = route(probs, residual)
                outcome = "decided" if choice is not None else "no_answer"
                attempts.append({
                    "attempt": attempt + 1, "form": form, "choice": choice,
                    "confidence": conf, "stage": stage, "probs": probs,
                    "residual": residual, "outcome": outcome,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
            except Exception as exc:
                outcome = "unavailable"
                attempts.append({"attempt": attempt + 1, "form": form,
                                 "error": str(exc), "outcome": outcome})
                break
            if outcome != "decided" or conf is None:
                break
            if not (REDO_BAND_LOW <= conf <= REDO_BAND_HIGH):
                break
        correct = bool(outcome == "decided" and choice == sc.gold)
        rows.append({
            "scenario_id": sc.state.scenario_id, "gold": sc.gold,
            "choice": choice, "confidence": conf, "stage": stage,
            "outcome": outcome, "correct": correct,
            "redo_count": max(0, len(attempts) - 1), "attempts": attempts})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="post-hoc roster-state diagnostic (NOT preregistered)")
    ap.add_argument("--d1-path", default="LiquidAI/d1-2B")
    args = ap.parse_args()

    backend = D1AsIsBackend(model_path=args.d1_path)
    if not backend.load():
        print(f"load FAILED: {backend.error}")
        return 1

    report: Dict[str, object] = {
        "diagnostic": "d1_asis_roster_state_posthoc",
        "label": ("POST-HOC DIAGNOSTIC - not a preregistered exam row; "
                  "tests state-exactness vs model capacity for the QUERY "
                  "failure"),
        "state": "roster + role->relation table + memory + goal (graph-derived)",
        "arms": {}, "records": {},
    }
    for label, scs in (("original", build_scenarios()),
                       ("fresh", build_fresh_scenarios())):
        rows = run_variant(backend, scs, label)
        by_class: Dict[str, Dict[str, int]] = {}
        for r in rows:
            slot = by_class.setdefault(r["gold"], {"n": 0, "correct": 0})
            slot["n"] += 1
            slot["correct"] += 1 if r["correct"] else 0
        acc = sum(1 for r in rows if r["correct"]) / len(rows)
        report["arms"][label] = {
            "accuracy": round(acc, 4), "n": len(rows), "by_gold_class": by_class,
            "redo_count": sum(r["redo_count"] for r in rows)}
        report["records"][label] = rows            # type: ignore[index]
        print(f"[roster-diag/{label}] acc={acc:.4f} "
              f"({sum(1 for r in rows if r['correct'])}/{len(rows)})")
        for cls, slot in sorted(by_class.items()):
            print(f"    gold {cls:<12} {slot['correct']}/{slot['n']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "artifacts", "d1_roster_diag.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, default=str)
        fh.write("\n")
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
