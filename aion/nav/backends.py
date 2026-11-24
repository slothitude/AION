"""NAV backends - pluggable decision adapters, each used EXACTLY per its card.

Backends return a choice from the CLOSED ENUM only (aion.nav.state.NAV_ACTIONS)
plus a decision record (goal, options, choice, outcome, goal_achieved,
counterfactual slot). Any returned string outside the enum is recorded as an
enum_violation and counted - never silently remapped.

Card sources (verified 2026-11-25, the ONLY usage authority for each backend):
- LiquidAI/d1-2B card: system_one(state, questions); choice question dict
  {type: choice, instructions, criteria}; answer fields choice/confidence/
  probabilities; transformers>=5.19 + trust_remote_code=True; fp32 on CPU;
  single forward pass - no sampling parameters exist.
- togethercomputer/Tev1-0.8B-experimental card: system instruction + structured
  decision {state, question, 2-24 labeled options}; model returns ONE option
  letter, application code maps letter->key; temperature 0, max_tokens 8,
  chat_template_kwargs {enable_thinking: false}. Card states the fine-tuned
  weights release license is BEING FINALIZED (repo carries no license tag) ->
  LICENSE-BLOCKED until the owner approves; the code path ships ready but is
  never called while unlicensed.
- larql-d1 socket contract (reports/d1_adapters.md §2.3, owner-approved
  read-only consumption): load reports/d1_adapters_artifacts/
  d1_action_classifier.pkl PORTABLY (pickle.load -> dict -> ["pipeline"] ->
  predict([vector])); feature order = the artifact's own feature_names (bound
  to DECISION_FEATURE_NAMES); no larql module import required, no internals
  touched.
"""

from __future__ import annotations

import json
import pickle
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aion.nav.state import (
    ACTION_DESCRIPTIONS,
    NAV_ACTIONS,
    NavState,
    build_d1_question,
    build_state_text,
    build_tev1_payload,
    d1_question_name,
    possible_relations,
    role_of,
    tev1_letter_to_key,
    TEV1_REQUEST_PARAMS,
    TEV1_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Decision record (field names mirror the larql socket record schema §2.2)
# ---------------------------------------------------------------------------

RECORD_SCHEMA_VERSION = "nav-1.0"


@dataclass
class DecisionRecord:
    schema_version: str = RECORD_SCHEMA_VERSION
    backend: str = ""
    scenario_id: str = ""
    goal: str = ""
    options: List[str] = field(default_factory=list)
    choice: Optional[str] = None
    outcome: str = "decided"          # decided|no_answer|license_blocked|unavailable
    goal_achieved: bool = False       # set by the exam (choice == gold)
    gold: Optional[str] = None        # set by the exam
    counterfactual: Dict[str, object] = field(default_factory=dict)
    enum_violation: bool = False
    latency_ms: float = 0.0
    compliance: List[str] = field(default_factory=list)
    raw: str = ""
    state_features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Shared NAV policy (graph rules) - the gold rule table AND the heuristic floor.
# Honest framing: the floor encodes the same graph policy as the gold, so it is
# the exam's solvability check and the reference line; models are judged
# against it, not expected to be "needed" by it.
# ---------------------------------------------------------------------------

def nav_decision_policy(state: NavState) -> str:
    """Closed graph rule table (order fixed, preregistered):
    1. subject not in the world            -> ASK_TEACHER (information missing)
    2. no outgoing relation for the role   -> STOP (no path serves the goal)
    3. fact goal, single edge covered      -> ANSWER
    4. reach goal, every hop covered       -> INFER
    5. otherwise (learnable, not covered)  -> QUERY
    """
    if state.subject_role() is None:
        return "ASK_TEACHER"
    rel = state.goal_relation()
    if rel is None or rel not in possible_relations(state.subject_role()):
        return "STOP"
    if state.goal_kind == "fact":
        return "ANSWER" if state.covered_single_fact() is not None else "QUERY"
    return "INFER" if state.walk_covered() is not None else "QUERY"


class HeuristicFloor:
    """The zero-model floor: nav_decision_policy applied directly. Deterministic,
    no model, ~0 latency. Expected to score 1.0 by construction (it IS the gold
    rule table) - it validates the exam and gives the model backends their
    reference line."""

    name = "heuristic_floor"

    def decide(self, state: NavState) -> DecisionRecord:
        t0 = time.perf_counter()
        choice = nav_decision_policy(state)
        rec = DecisionRecord(
            backend=self.name, scenario_id=state.scenario_id, goal=state.goal_text,
            options=list(NAV_ACTIONS), choice=choice, outcome="decided",
            counterfactual={"slot": None, "score": None, "source": "rule_table"},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            compliance=["zero-model rules; deterministic; no download"],
            raw=f"rule_table->{choice}")
        return rec


# ---------------------------------------------------------------------------
# D1Backend - LiquidAI/d1-2B via model.system_one() (card-compliant)
# ---------------------------------------------------------------------------

class D1Backend:
    """d1-2B choice query over the NAV action menu. NON-autoregressive: typed
    answers + probabilities in ONE forward pass, zero generated tokens, no
    sampling parameters (card)."""

    name = "d1_2b"

    def __init__(self, model_path: str, device: str = "cpu") -> None:
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
        except Exception as exc:  # typed honesty, never a silent skip
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def decide(self, state: NavState) -> DecisionRecord:
        t0 = time.perf_counter()
        rec = DecisionRecord(
            backend=self.name, scenario_id=state.scenario_id, goal=state.goal_text,
            options=list(NAV_ACTIONS),
            counterfactual={"slot": None, "score": None, "source": "not_run"},
            compliance=[
                "card: system_one(state, questions) - one forward pass",
                "card: choice question type (criteria = named action options)",
                "card: trust_remote_code=True, fp32 on CPU",
                "card: no sampling params exist; output_tokens always 0",
            ])
        if self.model is None:
            rec.outcome = "unavailable"
            rec.raw = self.error or "model not loaded"
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            return rec
        questions = {d1_question_name(): build_d1_question(state)}
        out = self.model.system_one(build_state_text(state), questions)
        ans = out["answers"][d1_question_name()]
        usage = out.get("usage", {})
        choice = ans.get("choice")
        probs: Dict[str, float] = dict(ans.get("probabilities") or {})
        rec.latency_ms = (time.perf_counter() - t0) * 1000.0
        rec.raw = json.dumps(
            {"answer": {k: ans.get(k) for k in ("choice", "confidence")},
             "usage": usage}, default=str)
        if choice is None:
            rec.outcome = "no_answer"
        elif str(choice) not in NAV_ACTIONS:
            # criteria keys ARE the enum, so an off-enum string is a real violation
            rec.choice = str(choice)
            rec.outcome = "decided"
            rec.enum_violation = True
        else:
            rec.choice = str(choice)
        alt = sorted(((k, v) for k, v in probs.items() if k != rec.choice),
                     key=lambda kv: -(kv[1] or 0.0))
        rec.counterfactual = {
            "slot": alt[0][0] if alt else None,
            "score": alt[0][1] if alt else None,
            "source": "d1_probabilities",
            "probabilities": probs,
        }
        return rec


# ---------------------------------------------------------------------------
# Tev1Backend - structured chat-completions per card. LICENSE-BLOCKED by
# default: the repo lists weights but carries NO license tag and the card says
# the fine-tuned-weights release license is being finalized. While
# license_granted is False this backend NEVER runs and records
# outcome=license_blocked.
# ---------------------------------------------------------------------------

class Tev1Backend:
    name = "tev1_0p8b"

    def __init__(self, base_url: str, model: str = "tev1-0.8b-experimental",
                 api_key: Optional[str] = None, license_granted: bool = False,
                 timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.license_granted = license_granted
        self.timeout_s = timeout_s

    def decide(self, state: NavState) -> DecisionRecord:
        t0 = time.perf_counter()
        rec = DecisionRecord(
            backend=self.name, scenario_id=state.scenario_id, goal=state.goal_text,
            options=list(NAV_ACTIONS),
            counterfactual={"slot": None, "score": None, "source": "not_run"},
            compliance=[
                "card: system instruction verbatim; state treated as data",
                "card: {state, question, 2-24 labeled options}; 5 options here",
                "card: temperature 0, max_tokens 8, enable_thinking=false",
                "card: one option letter -> key mapped by application code",
                "license: repo untagged + card says release license being "
                "finalized -> LICENSE-BLOCKED while license_granted=False",
            ])
        if not self.license_granted:
            rec.outcome = "license_blocked"
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            return rec
        payload = build_tev1_payload(state)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEV1_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            **TEV1_REQUEST_PARAMS,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            rec.outcome = "unavailable"
            rec.raw = f"{type(exc).__name__}: {exc}"
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            return rec
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        rec.latency_ms = (time.perf_counter() - t0) * 1000.0
        rec.raw = text
        key = tev1_letter_to_key(payload, text)
        if not text.strip():
            rec.outcome = "no_answer"
        elif key is None:
            rec.choice = text.strip()[:32]
            rec.enum_violation = True  # returned a token that is no listed letter
        else:
            rec.choice = key
        rec.counterfactual = {
            "slot": None, "score": None, "source": "tev1_single_letter",
        }
        return rec


# ---------------------------------------------------------------------------
# LarqlClassifierBackend - the graded socket consumer (R=0.8828 on the larql
# M6 holdout). Read-only consumption per reports/d1_adapters.md §2.3: PORTABLE
# artifact load (no larql module import, no internals touched).
# ---------------------------------------------------------------------------

#: Preregistered head-label -> NAV mapping (see exam.py preregistration).
#: QUERY/TRACE are the query family; REFORMULATE is the head's "change tack /
#: seek different information" action -> ASK_TEACHER. This is a declared
#: correspondence, NOT a card fact - reported as such.
HEAD_LABEL_TO_NAV: Dict[str, str] = {
    "QUERY": "QUERY",
    "TRACE": "QUERY",
    "INFER": "INFER",
    "ANSWER": "ANSWER",
    "REFORMULATE": "ASK_TEACHER",
    "STOP": "STOP",
}

#: Fixed exam-time values for features NAV states do not vary (declared in the
#: exam preregistration; documented here where they are produced).
LARQL_FIXED_FEATURES: Dict[str, float] = {
    "thinking_level": 1.0,
    "remaining_budget": 8.0,
    "depth": 0.0,
    "confidence": 0.0,
    "n_observations": 0.0,
    "n_directly_observed_refs": 0.0,
    "n_layer_references": 0.0,
    "n_queries": 0.0,
    "n_reformulations": 0.0,
    "last_query_ok": 0.0,
    "search_done": 0.0,
    "has_final_answer": 0.0,
    "verification_status": 0.0,
    "plan_progress": 0.0,
    "has_inference_result": 0.0,
}


def nav_larql_features(state: NavState, feature_names: List[str]) -> List[float]:
    """NAV state -> the artifact's documented 16-feature vector (order taken
    from the artifact's own feature_names). n_unknown_evidence is the count of
    goal-relevant hops not covered by memory (fact goal: 0/1; reach goal:
    missing hops); every other feature is fixed at its declared exam value.
    Honest caveat: the head was trained on larql M6 investigation states, so
    NAV states are a distribution shift - that is the finding being measured."""
    rel = state.goal_relation()
    role = state.subject_role()
    if state.goal_kind == "fact":
        unknown = 0.0 if state.covered_single_fact() is not None else 1.0
    else:
        # count hops missing from memory along the true graph walk
        unknown = 0.0
        node = state.subject
        while role_of(node, state.world) is not None:
            r = possible_relations(role_of(node, state.world))
            if not r:
                break
            nxt = [o for (rr, o) in state.world.graph.get(node, []) if rr == r[0]]
            if not nxt:
                break
            covered = any(s == node and rr == r[0]
                          for (s, rr, _o) in state.known_facts)
            if not covered:
                unknown += 1.0
            node = nxt[0]
    values = dict(LARQL_FIXED_FEATURES)
    values["n_unknown_evidence"] = unknown
    return [float(values.get(name, 0.0)) for name in feature_names]


class LarqlClassifierBackend:
    """Graded sklearn head over the socket's documented feature vector,
    consumed PORTABLY per socket §2.3 (pickle dict -> pipeline.predict)."""

    name = "larql_classifier"

    def __init__(self, artifact_path: str, meta_path: Optional[str] = None) -> None:
        self.artifact_path = artifact_path
        self.meta_path = meta_path
        self.pipeline = None
        self.feature_names: List[str] = []
        self.labels: List[str] = []
        self.training_meta: dict = {}
        self.feature_order_digest: Optional[str] = None
        self.error: Optional[str] = None

    def load(self) -> bool:
        try:
            with open(self.artifact_path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("artifact_kind") != "larql_d1_d1_action_classifier":
                self.error = "unrecognised artifact kind"
                return False
            self.pipeline = payload["pipeline"]
            self.feature_names = [str(x) for x in payload["feature_names"]]
            self.labels = [str(x) for x in payload["labels"]]
            self.training_meta = dict(payload.get("training_meta") or {})
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        if self.meta_path:
            try:
                with open(self.meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                self.feature_order_digest = meta.get("feature_order_digest")
                self.training_meta.update(
                    {k: meta.get(k) for k in ("n_train_points", "n_holdout_points",
                                              "holdout_accuracy")
                     if meta.get(k) is not None})
            except Exception:
                pass  # meta sidecar is informational
        return True

    def decide(self, state: NavState) -> DecisionRecord:
        t0 = time.perf_counter()
        rec = DecisionRecord(
            backend=self.name, scenario_id=state.scenario_id, goal=state.goal_text,
            options=list(NAV_ACTIONS),
            counterfactual={"slot": None, "score": None, "source": "not_run"},
            compliance=[
                "socket §2.3: portable pickle load, pipeline.predict([vector])",
                "socket: feature order from the artifact's feature_names "
                "(bound to DECISION_FEATURE_NAMES)",
                "declared mapping: head QUERY/TRACE->QUERY, REFORMULATE->"
                "ASK_TEACHER (correspondence, not a card fact)",
                "distribution shift: head trained on larql M6 states, probed "
                "on NAV chain-world states",
            ])
        if self.pipeline is None:
            rec.outcome = "unavailable"
            rec.raw = self.error or "head not loaded"
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            return rec
        vector = nav_larql_features(state, self.feature_names)
        rec.state_features = dict(zip(self.feature_names, vector))
        try:
            label = str(self.pipeline.predict([vector])[0])
            proba: Dict[str, float] = {}
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba([vector])[0]
                classes = [str(c) for c in getattr(self.pipeline, "classes_", [])]
                if len(classes) == len(probs):
                    proba = {c: float(p) for c, p in zip(classes, probs)}
        except Exception as exc:
            rec.outcome = "unavailable"
            rec.raw = f"{type(exc).__name__}: {exc}"
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            return rec
        rec.latency_ms = (time.perf_counter() - t0) * 1000.0
        rec.raw = json.dumps({"head_label": label, "proba": proba})
        nav_choice = HEAD_LABEL_TO_NAV.get(label)
        if nav_choice is None:
            rec.outcome = "no_answer"   # unmapped head label: never guessed
        else:
            rec.choice = nav_choice
        alt = sorted(((k, v) for k, v in proba.items()
                      if HEAD_LABEL_TO_NAV.get(k) != rec.choice),
                     key=lambda kv: -(kv[1] or 0.0))
        rec.counterfactual = {
            "slot": (HEAD_LABEL_TO_NAV.get(alt[0][0]) if alt else None),
            "score": alt[0][1] if alt else None,
            "source": "classifier_proba",
            "head_label": label,
            "head_confidence": proba.get(label),
        }
        return rec


#: Registry used by exam.py (--backend names)
BACKEND_NAMES = ("heuristic_floor", "d1_2b", "tev1_0p8b", "larql_classifier")


def build_backend(name: str, **kwargs):
    if name == "heuristic_floor":
        return HeuristicFloor()
    if name == "d1_2b":
        return D1Backend(model_path=kwargs["model_path"],
                         device=kwargs.get("device", "cpu"))
    if name == "tev1_0p8b":
        return Tev1Backend(base_url=kwargs["base_url"],
                           model=kwargs.get("model", "tev1-0.8b-experimental"),
                           api_key=kwargs.get("api_key"),
                           license_granted=bool(kwargs.get("license_granted")))
    if name == "larql_classifier":
        return LarqlClassifierBackend(
            artifact_path=kwargs["artifact_path"],
            meta_path=kwargs.get("meta_path"))
    raise ValueError(f"unknown backend {name!r}; known: {BACKEND_NAMES}")
