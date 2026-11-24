"""The proposer arms. They only PROPOSE; the runtime decides (owner law).

Three arms, one shared task-interface, one dimension of difference: HOW THE OP IS
CHOSEN. All arms build operation arguments with the same code from the same
observations, so any measured difference between arms is op-selection, not
arg-crafting.

  FixedRandom  uniform over the menu. The baseline: never learns, never stops.
  Heuristic    ordered rules ("if a requestable write was denied, request
               capacity"; "if the report cannot fit even the policy maximum,
               stop"). Good from episode 1, but it never adapts beyond its rules.
  PLASTIC      a weight per op, updated from OUTCOME FEEDBACK (amendment 4:
               denials are experiences, not log lines). capacity_exceeded raises
               REQUEST_CAPACITY; a beyond_policy_max refusal or a
               capacity_exceeded_policy_max denial lowers it and — once seen —
               becomes a stop signal: the agent learns to report
               insufficient-capacity instead of spamming (owner amendment 5 and
               the impossible-task control: more requests != more intelligence).

PLASTICITY SEAM (documented for the connections lane, NOT built here): the local
update rule lives entirely in PlasticProposer.observe(); the learned state is
PlasticProposer.weights. Swapping this rule for real NNFS plastic updates (the
frozen src/nnfs adapter) must not touch the runtime, the gate, or the task.

NAV SEAM (documented for Lane K, NOT built here — aion/nav is read-only): the
integration point is BaseProposer.choose(observation) -> Proposal and the
Observation dataclass below; a NAV lane can propose into the same gate unchanged.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aion.hostmode.authority import Outcome, Proposal
from aion.hostmode.policy import CAPACITY_RESOURCES, PermittedOperation
from aion.hostmode.task import (DATASET_PATH, REPORT_PATH, SCRIPT_PATH,
                                SCRIPT_TEMPLATE)

MENU: Tuple[PermittedOperation, ...] = tuple(PermittedOperation)
REQUEST_GRANT_SIZE = 65536          # what a well-formed capacity request asks for


@dataclass
class Observation:
    """Everything the proposer may SEE at one trial. No task-class label: class
    is host scoring information, never an input. Caps appear only after the arm
    has actually run CHECK_RESOURCES in this episode."""

    trial: int = 0
    goal: Dict = field(default_factory=dict)
    granted: bool = False
    caps_known: bool = False
    caps: Dict = field(default_factory=dict)
    dataset_present: bool = False
    script_written: bool = False
    script_run: bool = False
    report_text: Optional[str] = None      # computed report (from captured stdout)
    last_denial_op: Optional[str] = None
    last_denial_reason: Optional[str] = None
    request_refused_seen: bool = False
    policy_max_denial_seen: bool = False
    own_state_tampered: bool = False


def build_args(op: PermittedOperation, obs: Observation) -> Dict:
    """SHARED argument builder (identical for every arm)."""
    if op == PermittedOperation.WRITE_FILE:
        if obs.report_text is not None:
            # the answer exists -> persist it (the host manipulation)
            return {"path": REPORT_PATH, "content": obs.report_text}
        return {"path": SCRIPT_PATH, "content": SCRIPT_TEMPLATE}
    if op == PermittedOperation.RUN_SCRIPT:
        return {"script": SCRIPT_PATH, "timeout_s": 10}
    if op == PermittedOperation.ALLOCATE:
        return {"name": "scratch", "budget_bytes": 4096}
    if op == PermittedOperation.READ_OWN_STATE:
        return {"key": "aion_state.json"}
    if op == PermittedOperation.CHECK_RESOURCES:
        return {}
    if op == PermittedOperation.REQUEST_CAPACITY:
        return {"resource": "per_write_bytes", "requested": REQUEST_GRANT_SIZE}
    return {}


# --------------------------------------------------------------------------- #
# The arm interface: choose() proposes, observe() learns, stop_requested() ends
# the episode early (stopping needs no permission — refusing to act further is
# always available and is the honest response to an unsolvable task).
# --------------------------------------------------------------------------- #

class BaseProposer:
    name = "base"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.experiences: List[Dict] = []

    def reset(self) -> None:
        """Fresh episode: per-episode scratch only; learned state persists."""

    def choose(self, obs: Observation) -> Proposal:
        raise NotImplementedError

    def observe(self, obs_before: Observation, proposal: Proposal,
                outcome: Outcome) -> None:
        """Outcome feedback. Every arm records experiences (denials included);
        only Plastic updates weights here."""

    def stop_requested(self, obs: Observation) -> bool:
        return False


class FixedRandomProposer(BaseProposer):
    """Uniform over the closed menu. No learning, no stopping (baseline)."""

    name = "fixed_random"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._rng = random.Random(seed * 7717 + 11)
        self._cf_rng = random.Random(seed * 9931 + 3)

    def choose(self, obs: Observation) -> Proposal:
        op = self._rng.choice(MENU)
        return Proposal(op=op, args=build_args(op, obs), arm=self.name,
                        counterfactual=self._counterfactual(obs, op))

    def _counterfactual(self, obs: Observation, chosen) -> Dict:
        others = [o.value for o in MENU if o is not chosen]
        return {"rejected_options": others,
                "heuristic_would": HeuristicProposer().pick_op(obs).value,
                "note": "fixed_random has no counterfactual policy of its own"}


class HeuristicProposer(BaseProposer):
    """Ordered rules over the same observation. Rule-following, not adapting."""

    name = "heuristic"

    def pick_op(self, obs: Observation) -> PermittedOperation:
        # R0: a request was refused for want of demonstrated need -> DEMONSTRATE
        #     it: attempt the report write, whose denial is the evidence the
        #     runtime requires before granting capacity
        if obs.last_denial_op == "REQUEST_CAPACITY" and \
                obs.last_denial_reason == "no_demonstrated_need":
            return PermittedOperation.WRITE_FILE
        # R1: a requestable write was denied -> ask for capacity (the demo pivot:
        #     the denial IS the evidence the runtime's grant policy requires)
        if obs.last_denial_op == "WRITE_FILE" and \
                obs.last_denial_reason == "capacity_exceeded" and not obs.granted:
            return PermittedOperation.REQUEST_CAPACITY
        # R1.7: the answer exists, cannot fit the current cap, no grant yet ->
        #       ATTEMPT the write. The attempt demonstrates the need the
        #       runtime's grant policy requires (and its denial is an experience)
        if obs.report_text is not None and not obs.granted and obs.caps_known:
            if len(obs.report_text.encode()) > obs.caps.get("per_write_bytes", 0):
                return PermittedOperation.WRITE_FILE
        # R2: the answer exists and is writable -> persist it
        if obs.report_text is not None and (obs.granted or
                len(obs.report_text.encode()) <= obs.caps.get("per_write_bytes", 1 << 30)):
            return PermittedOperation.WRITE_FILE
        # R3: look around once at the start (learn the caps)
        if not obs.caps_known:
            return PermittedOperation.CHECK_RESOURCES
        # R4: dataset present but no tool yet -> write the script
        if obs.dataset_present and not obs.script_written:
            return PermittedOperation.WRITE_FILE
        # R5: tool written but not run -> run it
        if obs.script_written and not obs.script_run:
            return PermittedOperation.RUN_SCRIPT
        # R6: no answer yet and idle -> check resources again
        return PermittedOperation.CHECK_RESOURCES

    def choose(self, obs: Observation) -> Proposal:
        op = self.pick_op(obs)
        return Proposal(op=op, args=build_args(op, obs), arm=self.name,
                        counterfactual={
                            "rejected_options":
                                [o.value for o in MENU if o is not op],
                            "fixedrandom_would": "uniform-sample",
                            "rule_fired": True})

    def stop_requested(self, obs: Observation) -> bool:
        # The honest rule: if the report cannot fit even the policy maximum (seen
        # directly or computable from the caps), or a beyond-any-grant refusal
        # occurred, STOP and report insufficient capacity instead of spamming.
        # A non-raising request (request_refused) is waste, not a dead end.
        if obs.policy_max_denial_seen:
            return True
        if obs.caps_known and obs.report_text is not None:
            pmax = obs.caps.get("per_write_bytes_policy_max", 0)
            if pmax and len(obs.report_text.encode()) > pmax:
                return True
        return False


class PlasticProposer(BaseProposer):
    """Adaptive arm: CONTEXT-CONDITIONED choice weights over the menu, updated
    from outcome feedback. The context is the agent's own observable state
    (tool built? answer computed? grant in hand?) — the SAME observation every
    arm sees, so the comparison stays fair. Local to aion/hostmode (NOT
    src/nnfs plasticity — that integration is the documented seam for the
    connections lane: PlasticProposer.observe is the outcome->signal hook,
    self._weights is the plastic state).

    Three small mechanisms, all outcome-driven:
      weights      (context, op) values; softmax selection; per-trial decay
                   toward baseline so no crater is permanent.
      salience     a requestable denial shifts the VERY NEXT pick toward its fix,
                   and a grant shifts it toward persisting the answer (fast
                   pathway; the weights are the slow pathway).
      eligibility  the demonstration denial that produced a grant is credited
                   backwards (one step) — the write that 'failed' is what made
                   the grant possible, so it must not be learned as pure loss.
    """

    name = "plastic"

    LR = 0.35
    TAU = 0.9
    DECAY = 0.97                    # per-trial weight decay toward the baseline
    W_MIN, W_MAX = 0.05, 12.0
    REWARD = {
        "executed_core": 1.0,        # WRITE/RUN/ALLOC that executed
        "executed_goal": 4.0,        # the executed action achieved the goal
        "executed_info": 0.1,        # CHECK_RESOURCES / READ_OWN_STATE
        "redundant": -0.4,           # re-doing the same thing: trials are scarce
        "requestable_denial": -0.5,  # capacity_exceeded: the demonstration cost;
                                     # the eligibility credit on the grant that it
                                     # buys roughly cancels it, so 'demonstrate
                                     # then request' is the learned preference
        "policymax_denial": -1.0,    # beyond any grant: STOP territory
        "refusal": -2.0,             # unfixable-constraint refusals: stop territory
        "grant": 0.8,                # a grant arrived
        "eligibility_credit": 1.0,   # backward credit to the demonstrating denial
        "authority": -3.0,           # must never happen (measured invariant)
        "error": -0.8,
    }
    REQUEST_BOOST = 0.8              # credit to REQUEST_CAPACITY on requestable denial
                                     # (small: the grant reward itself does the teaching;
                                     # a big boost here teaches 'ask before demonstrating')

    CONTEXTS = ("no_tool", "tool", "answer_granted", "answer_pre_demo",
                "answer_demo_done")

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        # plastic state: (context, op) -> weight. Starts flat: nothing known.
        self.weights: Dict[Tuple[str, str], float] = {
            (ctx, o.value): 1.0 for ctx in self.CONTEXTS for o in MENU}
        self._rng = random.Random(seed * 6151 + 29)
        self._cf_rng = random.Random(seed * 4447 + 5)
        # no-progress (redundancy) guard: last executed signature PER OP, so
        # alternation (write-tool, check, write-tool, check ...) cannot farm
        # rewards by re-doing the same thing
        self._last_sig_by_op: Dict[str, str] = {}
        self._touched_this_episode: set = set()     # ops TRIED this episode
        self._salient_op: Optional[str] = None      # one-trial follow-through
        self._eligibility: Optional[Tuple[str, str]] = None  # (ctx, op) of the
        # demonstration denial awaiting its grant
        self.EPSILON = 0.10                         # exploration floor (all tried)
        self.EPSILON_FRESH = 0.25                   # while some op never tried

    # -- context (from the same observation every arm sees) -------------------- #

    def context_of(self, obs: Observation) -> str:
        if obs.report_text is not None:
            if obs.granted:
                return "answer_granted"
            # the pivot context, split on what the agent itself just experienced:
            # after its OWN requestable denial the request will be granted
            # (demonstrated need); before it, attempting the write is what
            # produces that evidence
            if obs.last_denial_op == "WRITE_FILE" and \
                    obs.last_denial_reason == "capacity_exceeded":
                return "answer_demo_done"
            return "answer_pre_demo"
        return "tool" if obs.script_written else "no_tool"

    def weights_for(self, obs: Observation) -> Dict[str, float]:
        """Flat view of the current context's weights (for inspection/tests)."""
        ctx = self.context_of(obs)
        return {o.value: self.weights[(ctx, o.value)] for o in MENU}

    def reset(self) -> None:
        # fresh world: per-episode scratch clears; LEARNED weights and the
        # experience history persist across episodes
        self._last_sig_by_op = {}
        self._touched_this_episode = set()
        self._salient_op = None
        self._eligibility = None

    # -- selection: softmax over the context's weights ------------------------- #

    def _probs(self, obs: Observation) -> Dict[str, float]:
        ws = self.weights_for(obs)
        m = max(ws.values())
        exps = {k: math.exp((v - m) / self.TAU) for k, v in ws.items()}
        z = sum(exps.values())
        return {k: v / z for k, v in exps.items()}

    def pick_op(self, obs: Observation,
                rng: Optional[random.Random] = None) -> PermittedOperation:
        rng = rng or self._rng
        r = rng.random()
        fresh = [op for op in MENU if op.value not in self._touched_this_episode]
        epsilon = self.EPSILON_FRESH if fresh else self.EPSILON
        if r < epsilon:
            # TARGETED exploration floor (no starvation): try an op that has not
            # been tried yet this episode before re-trying known ones
            pool = fresh if fresh else list(MENU)
            return pool[rng.randrange(len(pool))]
        probs = self._probs(obs)
        acc = 0.0
        for op in MENU:                          # stable enum order
            acc += probs[op.value]
            if r <= acc:
                return op
        return MENU[-1]

    def choose(self, obs: Observation) -> Proposal:
        # WEIGHT DECAY toward the baseline (once per trial, before anything
        # reads the weights): no crater is permanent. A badly penalised
        # (context, op) drifts back toward equal standing, so one bad episode
        # can never starve the agent of a route it needs.
        ctx = self.context_of(obs)
        for op in MENU:
            key = (ctx, op.value)
            self.weights[key] = 1.0 + (self.weights[key] - 1.0) * self.DECAY
        # SALIENCE: a requestable denial shifts the very next pick toward its fix,
        # a grant shifts it toward persisting the answer (fast pathway)
        if self._salient_op:
            op = PermittedOperation(self._salient_op)
            self._salient_op = None
            probs = self._probs(obs)
            cf_rng = random.Random(self._cf_rng.random() * 2 ** 31)
            cf_pick = self.pick_op(obs, rng=cf_rng)
            return Proposal(
                op=op, args=build_args(op, obs), arm=self.name,
                counterfactual={
                    "rejected_options": {o.value: round(probs[o.value], 4)
                                         for o in MENU if o.value != op.value},
                    "salience_follow_through": True,
                    "heuristic_would": HeuristicProposer().pick_op(obs).value,
                    "resampled_would": cf_pick.value,
                })
        op = self.pick_op(obs)
        probs = self._probs(obs)
        cf_rng = random.Random(self._cf_rng.random() * 2**31)
        cf_pick = self.pick_op(obs, rng=cf_rng)
        return Proposal(
            op=op, args=build_args(op, obs), arm=self.name,
            counterfactual={
                "rejected_options": {o.value: round(probs[o.value], 4)
                                     for o in MENU if o is not op},
                "resampled_would": cf_pick.value,
                "heuristic_would": HeuristicProposer().pick_op(obs).value,
                "context": self.context_of(obs),
            })

    # -- THE update rule (the plasticity seam) --------------------------------- #

    @staticmethod
    def _sig(proposal: Proposal) -> str:
        return proposal.op.value + "|" + ",".join(
            f"{k}={str(v)[:80]}" for k, v in sorted((proposal.args or {}).items()))

    def observe(self, obs_before: Observation, proposal: Proposal,
                outcome: Outcome) -> None:
        ctx = self.context_of(obs_before)
        key = (ctx, proposal.op.value)
        self.experiences.append({
            "trial": proposal.trial, "op": proposal.op.value, "context": ctx,
            "status": outcome.status, "reason": outcome.reason,
        })

        def bump(k: Tuple[str, str], delta: float) -> None:
            self.weights[k] = min(self.W_MAX,
                                  max(self.W_MIN, self.weights[k] + self.LR * delta))

        if outcome.status in ("executed", "authority_denied"):
            # a TRIED op is a TOUCHED op: refusals are experience too, and a
            # tried op must not hog the fresh-op exploration pool afterwards
            self._touched_this_episode.add(proposal.op.value)
        if outcome.status == "executed":
            # no-progress guard: re-executing the same op with the same args
            # teaches nothing (the classic starvation loop: re-writing the same
            # tool forever, or farming CHECK_RESOURCES)
            sig = self._sig(proposal)
            redundant = (self._last_sig_by_op.get(proposal.op.value) == sig)
            self._last_sig_by_op[proposal.op.value] = sig
            if outcome.goal_achieved:
                bump(key, self.REWARD["executed_goal"])
            elif redundant:
                bump(key, self.REWARD["redundant"])
            elif proposal.op in (PermittedOperation.CHECK_RESOURCES,
                                 PermittedOperation.READ_OWN_STATE):
                bump(key, self.REWARD["executed_info"])
            else:
                bump(key, self.REWARD["executed_core"])
            if outcome.grant is not None:
                bump((ctx, "REQUEST_CAPACITY"), self.REWARD["grant"])
                if self._eligibility is not None:
                    # the demonstration denial made this grant possible: credit it
                    bump(self._eligibility, self.REWARD["eligibility_credit"])
                    self._eligibility = None
                self._salient_op = "WRITE_FILE"   # complete the pivot: persist now
            return

        # --- denials are EXPERIENCES (owner amendment 4), with different meaning ---
        reason = outcome.reason
        if reason == "capacity_exceeded":
            bump(key, self.REWARD["requestable_denial"])
            bump((ctx, "REQUEST_CAPACITY"), self.REQUEST_BOOST)
            self._salient_op = "REQUEST_CAPACITY"   # fast pathway: try the fix
            self._eligibility = key                 # this denial buys the grant
        elif reason in ("capacity_exceeded_policy_max", "beyond_policy_max",
                        "need_exceeds_policy_ceiling"):
            # unfixable: requesting more is the WRONG response — the learned
            # stop (the impossible-task control; amendment 5)
            bump(key, self.REWARD["policymax_denial"])
            bump((ctx, "REQUEST_CAPACITY"), self.REWARD["refusal"])
            self._eligibility = None
        elif reason in ("request_refused", "no_demonstrated_need"):
            # asking blindly (already at that level, or no denial yet to
            # demonstrate need): mild waste, NOT fatal
            bump(key, -0.5)
        elif outcome.status == "authority_denied":
            bump(key, self.REWARD["authority"])
        else:  # error / timed_out
            bump(key, self.REWARD["error"])

    def stop_requested(self, obs: Observation) -> bool:
        """LEARNED stop: once the agent has experienced an UNFIXABLE constraint
        (a beyond-any-grant denial or refusal), the correct action is to stop and
        report insufficient capacity — not to keep asking. A merely non-raising
        request (request_refused) is waste, not a dead end, so it does not stop."""
        return obs.policy_max_denial_seen
