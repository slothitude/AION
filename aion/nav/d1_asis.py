"""ARM 1 (PRIMARY) - d1-2B AS-IS with decomposed card usage. ZERO training.

Owner amendment A1 (2026-11-25): LiquidAI/d1-2B is a READY-MADE post-trained
decision model - no training step is required to use it. The original exam's
single 5-way choice left d1 with a QUERY blind spot (0.533, never chose
QUERY). The fix is USAGE: decompose the NAV decision into the card's typed
primitives, per the preregistration (nav_head_prereg PART 3):

  ONE model.system_one(state_text, questions) call per attempt, four named
  questions, one forward pass, zero generated tokens (card):

    gate_in_world  (noul)  does the subject exist in this world?
    gate_path      (noul)  can the subject's role emit the goal relation?
    gate_covered   (noul)  is what the goal needs already in memory?
    residual       (choice over ANSWER, INFER)

  ROUTING (application logic over card-typed answers, threshold 0.5):
    P(in_world=yes) < 0.5  -> ASK_TEACHER
    else P(path=yes) < 0.5 -> STOP
    else P(covered=yes)<0.5-> QUERY      (the blind-spot fix, structural)
    else residual winner   -> ANSWER / INFER

  REDO (amendment A2): if the DECIDING probability is in [0.49, 0.51] the
  item is re-presented with the next frozen formulation (0->1->2, max 2
  re-presentations); the final attempt is scored. All four probabilities are
  recorded per attempt. Redos never change gold.

Card compliance (LiquidAI/d1-2B card): system_one(state, questions) with a
noul/choice question schema; trust_remote_code=True; fp32 on CPU; single
forward pass; no sampling parameters exist. fp32 needs ~7.7 GB -> LAPPY ONLY.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from aion.nav.backends import DecisionRecord
from aion.nav.nav_head_prereg import (
    RESIDUAL_CRITERIA,
    gate_covered_text,
    gate_in_world_text,
    gate_path_text,
    render_state_text,
    residual_instructions,
)


def route(probs: Dict[str, float], residual: Dict[str, float]
          ) -> Tuple[Optional[str], Optional[float], str]:
    """Frozen routing over one attempt's probabilities. Returns
    (choice, deciding_confidence, fired_stage). The deciding confidence is
    the probability of the DECIDING answer: for a gate, the probability of
    the branch taken (P(yes) when the gate said yes, P(no) when it said no);
    for the residual, the chosen option's probability."""
    p_in = float(probs["gate_in_world"])
    if p_in < 0.5:
        return "ASK_TEACHER", 1.0 - p_in, "gate_in_world"
    p_path = float(probs["gate_path"])
    if p_path < 0.5:
        return "STOP", 1.0 - p_path, "gate_path"
    p_cov = float(probs["gate_covered"])
    if p_cov < 0.5:
        return "QUERY", 1.0 - p_cov, "gate_covered"
    if not residual:
        return None, None, "residual_empty"
    best = max(residual, key=residual.get)        # type: ignore[arg-type]
    return str(best), float(residual[best]), "residual"


def build_questions(state, form: int) -> Dict[str, Dict[str, object]]:
    """The four named card questions for one formulation (frozen pool)."""
    return {
        "gate_in_world": {"type": "noul",
                          "instructions": gate_in_world_text(state, form)},
        "gate_path": {"type": "noul",
                      "instructions": gate_path_text(state, form)},
        "gate_covered": {"type": "noul",
                         "instructions": gate_covered_text(state, form)},
        "residual": {"type": "choice",
                     "instructions": residual_instructions(form),
                     "criteria": dict(RESIDUAL_CRITERIA)},
    }


class D1AsIsBackend:
    """d1-2B as-is, decomposed usage, card-compliant, zero training."""

    name = "d1_asis"

    def __init__(self, model_path: str = "LiquidAI/d1-2B",
                 device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        self.error: Optional[str] = None

    def load(self) -> bool:
        try:
            import torch
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, dtype=torch.float32,
            ).to(self.device)
            self.model.eval()
            return True
        except Exception as exc:            # typed honesty, never silent
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    # -- one attempt --------------------------------------------------------- #

    def attempt(self, state, form: int,
                state_text: Optional[str] = None) -> Dict[str, object]:
        """One system_one call under one formulation. Returns routing +
        all probabilities + latency. Raises on backend failure (caller
        records an honest unavailable row). state_text overrides the
        rendered preregistered formulation (used ONLY by the post-hoc
        roster diagnostic; exam rows always use the frozen forms)."""
        t0 = time.perf_counter()
        questions = build_questions(state, form)
        text = render_state_text(state, form) if state_text is None \
            else state_text
        out = self.model.system_one(text, questions)
        ans = out["answers"]
        usage = out.get("usage", {})
        probs = {
            "gate_in_world": float(ans["gate_in_world"]["noul"]),
            "gate_path": float(ans["gate_path"]["noul"]),
            "gate_covered": float(ans["gate_covered"]["noul"]),
        }
        res_ans = ans["residual"]
        residual = {str(k): float(v) for k, v in
                    (res_ans.get("probabilities") or {}).items()}
        choice, conf, stage = route(probs, residual)
        return {
            "choice": choice, "confidence": conf, "stage": stage,
            "probs": probs, "residual": residual,
            "residual_choice": res_ans.get("choice"),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "form": form,
            "usage": usage,
        }

    # -- uniform exam interface ---------------------------------------------- #

    def present(self, state, form: int = 0
                ) -> Tuple[Optional[str], Optional[float], Dict[str, object], str]:
        if self.model is None:
            return None, None, {"error": self.error or "model not loaded"}, \
                "unavailable"
        try:
            att = self.attempt(state, form)
        except Exception as exc:
            return None, None, {"error": f"{type(exc).__name__}: {exc}"}, \
                "unavailable"
        aux = {k: att[k] for k in ("stage", "probs", "residual", "form",
                                   "usage", "latency_ms", "residual_choice")}
        if att["choice"] is None:
            return None, None, aux, "no_answer"    # empty routing: honest miss
        return att["choice"], att["confidence"], aux, "decided"

    def decide(self, state) -> DecisionRecord:
        """Single-attempt record (compatibility interface, form 0)."""
        rec = DecisionRecord(
            backend=self.name, scenario_id=state.scenario_id,
            goal=state.goal_text, options=["QUERY", "INFER", "ANSWER",
                                           "ASK_TEACHER", "STOP"],
            counterfactual={"slot": None, "score": None, "source": "d1_asis"},
            compliance=[
                "card: system_one(state, questions) - one forward pass",
                "card: noul gates + residual choice (typed answers)",
                "card: trust_remote_code=True, fp32 on CPU, no sampling",
                "ZERO training - decomposed usage of the ready-made model",
                "decomposition prereg: nav_head_prereg PART 3",
            ])
        if self.model is None:
            rec.outcome = "unavailable"
            rec.raw = self.error or "model not loaded"
            return rec
        try:
            att = self.attempt(state, 0)
        except Exception as exc:
            rec.outcome = "unavailable"
            rec.raw = f"{type(exc).__name__}: {exc}"
            return rec
        rec.latency_ms = att["latency_ms"]
        rec.raw = json.dumps({k: att[k] for k in
                              ("stage", "probs", "residual", "usage")},
                             default=str)
        if att["choice"] is None:
            rec.outcome = "no_answer"
        else:
            rec.choice = str(att["choice"])
        rec.counterfactual = {
            "slot": None, "score": att["confidence"],
            "source": "d1_asis_decomposed", "stage": att["stage"],
            "probs": att["probs"],
        }
        return rec
