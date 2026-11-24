"""ARM 2 - the NAV-domain trained decision head + the STEP-7 adaptation seam.

Trains the two preregistered candidates (nav_head_prereg PART 2) on the frozen
train split, selects by the frozen validation slice (train worlds only), and
saves the winner. The head exposes:

  decide(state)          one DecisionRecord (closed enum, never sees gold)
  present(state, form)   (choice, confidence, aux, outcome) for the exam
                         runner - formulation-INVARIANT by declared design
  observe(chosen, gold)  the STEP-7 outcome-feedback adaptation seam: a light
                         per-class online bias update (multi-class perceptron
                         on top of the frozen weights, lr 0.25, frozen)

NO LLM training; CPU-only; seconds per fit.

FUTURE NNFS-PLASTICITY INTEGRATION (documented, NOT implemented here): the
connections-lane adapter would feed observe() from NNFS plasticity events -
each outcome (chosen, gold, reward) becomes a plastic signal on the head's
per-class weights exactly as the NNFS connections lane updates weights on
co-activation, with consolidation periodically freezing the biases back into
the artifact. The seam is deliberately a single method so the adapter needs
no knowledge of sklearn internals.

CLI:
    python -m aion.nav.nav_head train
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from aion.nav.backends import DecisionRecord, nav_decision_policy
from aion.nav.nav_head_prereg import (
    ADAPTATION_LR,
    CLASSES,
    FEATURE_NAMES,
    HEAD_EXAM_VERSION,
    MODEL_SEED,
    PREREG_DATE,
    SPLIT_FREEZE_DATE,
    VAL_SLICE_FRACTION,
    state_features,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "artifacts")
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "nav_decision_head.pkl")
META_PATH = os.path.join(ARTIFACT_DIR, "nav_decision_head_meta.json")


# ---------------------------------------------------------------------------
# training (frozen: PART 2)
# ---------------------------------------------------------------------------

def _fit_one(kind: str, X: np.ndarray, y: np.ndarray):
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       random_state=MODEL_SEED)),
        ]).fit(X, y)
    if kind == "boost":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            random_state=MODEL_SEED).fit(X, y)
    raise ValueError(f"unknown model kind {kind!r}")


def train_nav_head(dataset: Dict[str, object],
                   out_dir: str = ARTIFACT_DIR
                   ) -> Tuple[object, Dict[str, object]]:
    """Train both preregistered candidates on the train split, select on the
    frozen validation slice (train worlds only), return (winner, meta)."""
    train = dataset["train"]                      # type: ignore[index]
    X = np.array([r["features"] for r in train], dtype=float)  # type: ignore[index]
    y = np.array([r["gold"] for r in train])      # type: ignore[index]
    n = len(X)
    n_val = max(1, int(round(n * VAL_SLICE_FRACTION)))
    tr_x, tr_y = X[:n - n_val], y[:n - n_val]
    va_x, va_y = X[n - n_val:], y[n - n_val:]

    results: Dict[str, Dict[str, object]] = {}
    fitted: Dict[str, object] = {}
    for kind in ("logistic", "boost"):
        t0 = time.perf_counter()
        model = _fit_one(kind, tr_x, tr_y)
        fit_s = time.perf_counter() - t0
        va_pred = model.predict(va_x)
        tr_pred = model.predict(tr_x)
        results[kind] = {
            "val_accuracy": float((va_pred == va_y).mean()),
            "train_fit_accuracy": float((tr_pred == tr_y).mean()),
            "fit_seconds": round(fit_s, 2),
        }
        fitted[kind] = model

    # frozen selection rule: higher val accuracy, tie -> logistic (simpler)
    winner = "logistic"
    if results["boost"]["val_accuracy"] > results["logistic"]["val_accuracy"]:
        winner = "boost"
    model = fitted[winner]

    hold = dataset["holdout"]                     # type: ignore[index]
    hx = np.array([r["features"] for r in hold], dtype=float)  # type: ignore[index]
    hy = np.array([r["gold"] for r in hold])      # type: ignore[index]
    hold_pred = model.predict(hx)
    holdout_accuracy = float((hold_pred == hy).mean())

    head = NavDecisionHead(model, list(model.classes_ if hasattr(
        model, "classes_") else model[-1].classes_), list(FEATURE_NAMES),
        kind=winner)

    meta = {
        "artifact": "nav_decision_head",
        "exam_version": HEAD_EXAM_VERSION,
        "preregistered": PREREG_DATE,
        "split_freeze_date": SPLIT_FREEZE_DATE,
        "model_kind": winner,
        "candidates": results,
        "selection_rule": (f"higher val accuracy on the last "
                           f"{int(VAL_SLICE_FRACTION * 100)}% of train rows; "
                           "tie -> logistic"),
        "n_train": int(n - n_val), "n_val": int(n_val),
        "holdout_accuracy": holdout_accuracy,
        "n_holdout": int(len(hx)),
        "feature_names": list(FEATURE_NAMES),
        "classes": list(head.classes_),
        "machine_import": "sklearn+joblib, CPU only",
        "adaptation": {"kind": "per_class_bias_perceptron", "lr": ADAPTATION_LR},
    }
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(head, os.path.join(out_dir, "nav_decision_head.pkl"))
    with open(os.path.join(out_dir, "nav_decision_head_meta.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, default=str)
        fh.write("\n")
    return head, meta


# ---------------------------------------------------------------------------
# the head + STEP-7 adaptation seam
# ---------------------------------------------------------------------------

class NavDecisionHead:
    """Economical NAV-domain decision head (sklearn) with outcome-feedback
    adaptation. All decisions use the frozen structural feature vector; the
    closed enum is enforced; gold is never an input."""

    name = "trained_head"

    def __init__(self, model, classes: Sequence[str],
                 feature_names: Sequence[str], kind: str = "logistic") -> None:
        self.model = model
        self.classes_: List[str] = list(classes)
        self.feature_names = list(feature_names)
        self.kind = kind
        self.biases: Dict[str, float] = {c: 0.0 for c in self.classes_}
        self.n_observed = 0
        self.n_adapted = 0

    # -- features ----------------------------------------------------------- #

    def features(self, state) -> np.ndarray:
        return np.array([state_features(state)], dtype=float)

    # -- probabilities + adaptation ----------------------------------------- #

    def probabilities(self, state) -> Dict[str, float]:
        proba = self.model.predict_proba(self.features(state))[0]
        cls = getattr(self.model, "classes_", None)
        if cls is None and hasattr(self.model, "__getitem__"):
            cls = self.model[-1].classes_
        return {str(c): float(p) for c, p in zip(cls, proba)}

    def decide_scores(self, state) -> Tuple[str, Dict[str, float], float]:
        """Argmax over log-proba + adapted per-class bias (the adaptation seam
        acts on the score, not the frozen weights)."""
        proba = self.probabilities(state)
        scores = {c: float(np.log(max(p, 1e-12))) + self.biases.get(c, 0.0)
                  for c, p in proba.items()}
        choice = max(scores, key=scores.get)          # type: ignore[arg-type]
        return choice, proba, proba[choice]

    # -- the STEP-7 seam ----------------------------------------------------- #

    def observe(self, outcome: Dict[str, object]) -> Dict[str, float]:
        """Outcome-feedback adaptation (STEP-7 seam). outcome = {"chosen": ...,
        "gold": ...}. Multi-class perceptron on the per-class bias: on a miss,
        gold's bias rises and the wrongly chosen bias falls (lr 0.25, frozen);
        a hit leaves the biases untouched. Returns the updated biases."""
        chosen, gold = str(outcome.get("chosen")), str(outcome.get("gold"))
        self.n_observed += 1
        if chosen == gold or gold not in self.biases:
            return dict(self.biases)
        self.biases[gold] = self.biases.get(gold, 0.0) + ADAPTATION_LR
        if chosen in self.biases:
            self.biases[chosen] = self.biases.get(chosen, 0.0) - ADAPTATION_LR
        self.n_adapted += 1
        return dict(self.biases)

    def reset_adaptation(self) -> None:
        self.biases = {c: 0.0 for c in self.classes_}
        self.n_observed = 0
        self.n_adapted = 0

    def perturb_biases(self, delta: Dict[str, float]) -> None:
        """Frozen deterministic degradation, used by the demo to simulate a
        wrong-domain prior (the larql-head story). Not part of the exam."""
        for c, d in delta.items():
            self.biases[c] = self.biases.get(c, 0.0) + float(d)

    # -- exam interfaces ------------------------------------------------------ #

    def present(self, state, form: int = 0) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        """Uniform exam interface: (choice, confidence, aux, outcome).
        Feature-vector head: formulation-INVARIANT (declared in the
        preregistration) - form cannot change the vector, so redos are
        bounded no-ops."""
        choice, proba, conf = self.decide_scores(state)
        aux = {"proba": proba, "bias": dict(self.biases), "form_invariant": True}
        return choice, conf, aux, "decided"

    def decide(self, state) -> DecisionRecord:
        choice, proba, conf = self.decide_scores(state)
        return DecisionRecord(
            backend=self.name, scenario_id=state.scenario_id,
            goal=state.goal_text, options=list(CLASSES),
            choice=choice, outcome="decided",
            counterfactual={"slot": None, "score": None,
                            "source": "nav_head_proba", "probabilities": proba},
            latency_ms=0.0,
            compliance=[
                "nav-domain head: trained on the frozen NAV split "
                f"(prereg {PREREG_DATE}); structural features only",
                "closed enum enforced; gold never an input",
                "formulation-invariant redo (declared)",
            ],
            state_features=dict(zip(self.feature_names,
                                    state_features(state))),
            raw=json.dumps({"choice": choice, "confidence": conf,
                            "proba": proba}))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="train the NAV decision head")
    ap.add_argument("--dataset", default=os.path.join(ARTIFACT_DIR,
                                                      "nav_dataset.json"))
    ap.add_argument("--out-dir", default=ARTIFACT_DIR)
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    # `python -m` executes this file as __main__, which would pickle the head
    # class under an unresolvable module name; rebind to the properly
    # imported class object so joblib.load() resolves it anywhere.
    import aion.nav.nav_head as _proper
    globals()["NavDecisionHead"] = _proper.NavDecisionHead
    head, meta = train_nav_head(dataset, args.out_dir)
    print(f"winner={meta['model_kind']}  "
          f"cand={ {k: v['val_accuracy'] for k, v in meta['candidates'].items()} }")
    print(f"holdout_accuracy={meta['holdout_accuracy']} "
          f"({meta['n_holdout']} fresh-world examples)")
    print(f"artifact -> {os.path.join(args.out_dir, 'nav_decision_head.pkl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
