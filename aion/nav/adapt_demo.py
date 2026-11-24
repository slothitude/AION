"""STEP-7 SEAM DEMO - outcome-feedback adaptation on a held-out stream.

Owner order step 7: implement observe(outcome) weight adaptation on the
trained head and show post-feedback improvement on a held-out stream. The
stream comes from the frozen DEMO world family (seeds 530001+29*i, its own
RNG stream) - worlds never used for training or the dataset holdout.

Arms (frozen):
  control         trained head, adaptation OFF (no observe calls)
  adapt           trained head, observe(chosen, gold) after EVERY item
  degraded+adapt  trained head with a frozen wrong-domain prior
                  (bias QUERY +8.0, all others -2.0 - the larql-head story:
                  an organ that arrived believing everything is a query;
                  sized to the head's measured log-proba margins, min 3.9 /
                  median 6.5, so it genuinely degrades), observe() after
                  every item

Protocol: 4 chunks x 100 items (class-balanced stream, 20 per gold class per
chunk); per-chunk accuracy reported for every arm. Adaptation acts ONLY on
the per-class bias vector (lr 0.25, frozen); the fitted weights never change
at demo time.

FUTURE NNFS-PLASTICITY INTEGRATION (documented, not implemented): the
connections-lane adapter would drive observe() from NNFS plasticity events -
each (chosen, gold, reward) outcome becomes a plastic signal on the head's
per-class weights exactly as the connections lane updates weights on
co-activation, with consolidation periodically freezing the biases back into
the artifact. Single-method seam: NavDecisionHead.observe().

CLI:  python -m aion.nav.adapt_demo
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from aion.nav.nav_dataset import build_stream
from aion.nav.nav_head_prereg import ADAPTATION_LR, PREREG_DATE
from aion.nav.nav_head import ARTIFACT_DIR, NavDecisionHead

DEGRADED_BIAS = {"QUERY": 8.0, "ANSWER": -2.0, "INFER": -2.0,
                 "STOP": -2.0, "ASK_TEACHER": -2.0}
N_CHUNKS = 4
CHUNK_SIZE = 100


def run_arm(head: NavDecisionHead, stream: List[Dict[str, object]],
            adapt: bool) -> List[float]:
    """Run the stream; per-chunk accuracy; observe() after each item when
    adapt is on. Items are scored against the recorded gold (graph-derived at
    dataset build time)."""
    accs: List[float] = []
    correct_in_chunk = 0
    for i, row in enumerate(stream):
        state_like = _StreamItem(row)
        choice, _proba, _conf = head.decide_scores(state_like)
        hit = (choice == row["gold"])
        correct_in_chunk += 1 if hit else 0
        if adapt:
            head.observe({"chosen": choice, "gold": row["gold"]})
        if (i + 1) % CHUNK_SIZE == 0:
            accs.append(correct_in_chunk / CHUNK_SIZE)
            correct_in_chunk = 0
    return accs


class _StreamItem:
    """Minimal NavState-shaped view over a dataset row (features are computed
    from the row's stored state; the head never needs the full world)."""

    def __init__(self, row: Dict[str, object]) -> None:
        self.row = row
        self.scenario_id = str(row["scenario_id"])
        self.goal_text = f"{row['goal_kind']}:{row['subject']}:{row['relation']}"

    # state_features() needs world + known_facts; dataset rows already carry
    # the frozen feature vector, so bypass recomputation for speed and to
    # guarantee byte-identical inputs.
    def features(self) -> List[float]:
        return [float(x) for x in self.row["features"]]  # type: ignore[index]


def _patch_features(head: NavDecisionHead) -> None:
    """Use the row's stored frozen vector instead of recomputing from the
    world (identical values; avoids rebuilding 6 worlds per decision)."""
    import numpy as np

    def features(self, state):
        return np.array([state.features()], dtype=float)
    head.__class__.features = features  # type: ignore[assignment]


def main() -> int:
    t0 = time.perf_counter()
    stream = build_stream(N_CHUNKS * CHUNK_SIZE)
    gold_counts: Dict[str, int] = {}
    for r in stream:
        gold_counts[str(r["gold"])] = gold_counts.get(str(r["gold"]), 0) + 1

    import joblib
    head_control: NavDecisionHead = joblib.load(
        os.path.join(ARTIFACT_DIR, "nav_decision_head.pkl"))
    head_adapt: NavDecisionHead = joblib.load(
        os.path.join(ARTIFACT_DIR, "nav_decision_head.pkl"))
    head_degraded: NavDecisionHead = joblib.load(
        os.path.join(ARTIFACT_DIR, "nav_decision_head.pkl"))
    for h in (head_control, head_adapt, head_degraded):
        h.reset_adaptation()
    head_degraded.perturb_biases(DEGRADED_BIAS)
    _patch_features(head_control)
    _patch_features(head_adapt)
    _patch_features(head_degraded)

    acc_control = run_arm(head_control, stream, adapt=False)
    acc_adapt = run_arm(head_adapt, stream, adapt=True)
    acc_degraded = run_arm(head_degraded, stream, adapt=True)

    report = {
        "demo": "step7_outcome_feedback_adaptation",
        "preregistered": PREREG_DATE,
        "stream": {"n": len(stream), "chunks": N_CHUNKS,
                   "chunk_size": CHUNK_SIZE,
                   "world_family": "DEMO (seeds 530001+29*i, n_chains=8)",
                   "rng_seed_note": "own frozen stream; unseen by training "
                                    "and by the dataset holdout",
                   "gold_counts": gold_counts},
        "seam": {"method": "NavDecisionHead.observe (per-class bias "
                           "perceptron)", "lr": ADAPTATION_LR},
        "degraded_prior": DEGRADED_BIAS,
        "per_chunk_accuracy": {
            "control": acc_control,
            "adapt": acc_adapt,
            "degraded_plus_adapt": acc_degraded,
        },
        "adaptation_stats": {
            "n_observed": head_adapt.n_observed,
            "n_adapted": head_adapt.n_adapted,
            "biases_end_adapt": head_adapt.biases,
            "biases_end_degraded": head_degraded.biases,
            "degraded_n_adapted": head_degraded.n_adapted,
        },
        "seconds": round(time.perf_counter() - t0, 2),
    }
    out = os.path.join(ARTIFACT_DIR, "adapt_demo.json")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, default=str)
        fh.write("\n")

    print(f"stream n={len(stream)} gold={gold_counts} "
          f"({report['seconds']}s)")
    print(f"control           per-chunk {acc_control}")
    print(f"adapt             per-chunk {acc_adapt}")
    print(f"degraded + adapt  per-chunk {acc_degraded}")
    print(f"biases after adapt: {head_adapt.biases}")
    print(f"biases after degraded+adapt: {head_degraded.biases}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
