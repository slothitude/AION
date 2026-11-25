"""GymActions: the permitted-action REQUEST API — requests only, never writes.

OWNER LAW (verbatim, 2026-11-25): "Python remains authoritative. Godot subscribes
to events and requests permitted actions. It never writes directly into plastic
state or bypasses the AION runtime's authority controls."

This module is the other half of that law's Python side. Every gym action maps
to an EXISTING runtime entry point, or it is denied with a structured, logged
reason:

  teach            -> DialogueSession.teach (the dialogue learn path: extractor
                      -> Student.learn_fact -> pathway.nnfs_update). The bridge
                      performs zero learning logic itself.
  ask              -> DialogueSession.ask (the student-ask protocol round), or
                      Student.answer when no session is wired.
  correct          -> RuleExtractor.extract_correction -> CorrectionHandler.apply
                      (plastic override with before/after provenance).
  pause / resume   -> the ONE flag the bridge owns. While paused, teach / ask /
                      correct / approve_request / deny_request are denied
                      ("paused"); reads (snapshot, polls) stay open.
  approve_request  -> HostRuntime.execute(Proposal(...)): the hostmode AUTHORITY
                      gate decides — approve only MEANS "submit through the
                      gate". A denial from the gate is a first-class structured
                      outcome, never bypassed.
  deny_request     -> refuses to forward a request to the authority at all;
                      recorded + emitted as hostmode.request/denied_by_gym.
  set_goal         -> stores goal state in BRIDGE-owned state (surfaced in
                      snapshots for the devhistory lane to fold later). This
                      module never writes the devhistory ledger directly.

ZERO direct plastic writes anywhere: no module in this package touches
u_fast / theta_slow / relationship.W / the gates. Plastic change happens only
inside the runtime machinery these calls reach.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from aion.dialogue.lesson_extractor import RuleExtractor      # read-only sibling
from aion.gym_bridge.emitter import GymBridgeError, GymEmitter
from aion.hostmode.authority import Proposal                  # read-only sibling
from aion.hostmode.policy import PermittedOperation           # read-only enum

# The closed vocabulary of gym action requests.
GYM_ACTIONS = (
    "teach", "ask", "correct",
    "pause", "resume",
    "approve_request", "deny_request",
    "set_goal",
)

# Actions the pause flag blocks. Reads and the pause controls themselves
# are never blocked (a paused runtime must still be observable and resumable).
PAUSED_BLOCKED = ("teach", "ask", "correct", "approve_request", "deny_request")


class GymActionRejected(GymBridgeError):
    """A malformed or out-of-vocabulary REQUEST (never a runtime failure).
    Carries its own structured denial reason; _dispatch returns it verbatim."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


class GymActions:
    """Authority-checked action requests -> EXISTING runtime entry points."""

    def __init__(self, bundle: Any,
                 emitter: Optional[GymEmitter] = None,
                 clock: Optional[Callable[[], float]] = None,
                 goal_state: Optional[Dict] = None):
        """
        bundle: the runtime handles (same shape as RuntimeBundle, plus optional
                `dialogue` = DialogueSession-like, `student` = Student-like,
                `corrections` = CorrectionHandler-like).
        emitter: optional GymEmitter used to emit hostmode.request events for
                bridge-level denials (denied_by_gym) and to share goal state.
        """
        self.bundle = bundle
        self.emitter = emitter
        self._clock = clock or time.time
        # bridge-owned goal state; shared with the emitter when wired through it
        if goal_state is not None:
            self.goal_state: Dict = goal_state
        elif emitter is not None:
            self.goal_state = emitter.goal_state
        else:
            self.goal_state = {}
        self.paused = bool(getattr(emitter, "paused", False)) if emitter else False
        self.log: List[Dict] = []             # every request, granted or denied
        self._request_seq = 0
        self._extractor = RuleExtractor()

    # ------------------------------------------------------------------ #
    # the single entry point
    # ------------------------------------------------------------------ #

    def request(self, action: str, args: Optional[Dict] = None,
                client: str = "godot", authenticated: bool = False,
                request_id: Optional[str] = None) -> Dict:
        """One action request -> one structured result. Denials never raise;
        they return {"ok": False, "reason": ..., "detail": ...} and are logged."""
        args = dict(args or {})
        self._request_seq += 1
        rid = request_id or f"gym-{self._request_seq:06d}"
        entry = {"request_id": rid, "ts": float(self._clock()), "action": str(action),
                 "client": str(client), "authenticated": bool(authenticated)}

        result = self._dispatch(action, args, authenticated, rid, entry)
        entry.update({"status": result["status"], "reason": result.get("reason")})
        self.log.append(entry)
        return result

    # -- the gates ------------------------------------------------------------- #

    def _deny(self, rid: str, action: str, reason: str, detail: str) -> Dict:
        return {"ok": False, "request_id": rid, "action": action,
                "status": "denied", "reason": reason, "detail": detail,
                "result": {}}

    def _dispatch(self, action: str, args: Dict, authenticated: bool,
                  rid: str, entry: Dict) -> Dict:
        # gate 1: authentication (the server enforces the token; this re-checks)
        if not authenticated:
            return self._deny(rid, action, "unauthenticated",
                              "action requests require a valid token")
        # gate 2: closed action vocabulary
        if action not in GYM_ACTIONS:
            return self._deny(rid, action, "unknown_action",
                              f"{action!r} is not one of {GYM_ACTIONS}")
        # gate 3: the bridge-owned pause flag
        if self.paused and action in PAUSED_BLOCKED:
            return self._deny(rid, action, "paused",
                              "the runtime is paused (resume to act)")
        try:
            handler = {
                "teach": self._do_teach,
                "ask": self._do_ask,
                "correct": self._do_correct,
                "pause": self._do_pause,
                "resume": self._do_resume,
                "approve_request": self._do_approve,
                "deny_request": self._do_deny_request,
                "set_goal": self._do_set_goal,
            }[action]
            out = handler(args, rid)
        except GymActionRejected as exc:
            return self._deny(rid, action, exc.reason, exc.detail)
        except GymBridgeError as exc:
            return self._deny(rid, action, "no_handler", str(exc))
        except Exception as exc:               # runtime machinery refused loudly
            return self._deny(rid, action, "runtime_error",
                              f"{type(exc).__name__}: {exc}")
        return {"ok": True, "request_id": rid, "action": action,
                "status": "ok", "reason": None, "detail": "",
                "result": out}

    def _require(self, handle: Any, what: str) -> Any:
        if handle is None:
            raise GymBridgeError(f"no {what} wired in the runtime bundle")
        return handle

    # ------------------------------------------------------------------ #
    # teach / ask / correct — through the dialogue learn path ONLY
    # ------------------------------------------------------------------ #

    def _do_teach(self, args: Dict, rid: str) -> Dict:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise GymActionRejected("bad_args", "teach requires non-empty args['text']")
        session = self._require(getattr(self.bundle, "dialogue", None), "dialogue session")
        facts = session.teach(text)            # EXISTING learn path (student learns)
        return {"taught": True,
                "n_facts": len(facts),
                "facts": [f.to_dict() for f in facts]}

    def _do_ask(self, args: Dict, rid: str) -> Dict:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise GymActionRejected("bad_args", "ask requires non-empty args['question']")
        session = getattr(self.bundle, "dialogue", None)
        if session is not None:
            event = session.ask(question)      # EXISTING ask protocol round
            return {
                "verdict": str(event.verdict),
                "student_answer": event.student_answer,
                "teacher_answer": event.teacher_answer,
                "corrections": [c.to_dict() for c in event.corrections],
                "kind": str(event.kind),
            }
        student = self._require(getattr(self.bundle, "student", None), "student")
        answer = student.answer(question)      # EXISTING student ask path
        return {"verdict": None, "student_answer": answer,
                "teacher_answer": None, "corrections": [], "kind": "direct_ask"}

    def _do_correct(self, args: Dict, rid: str) -> Dict:
        text = args.get("correction")
        if not isinstance(text, str) or not text.strip():
            raise GymActionRejected("bad_args",
                                    "correct requires non-empty args['correction']")
        session = getattr(self.bundle, "dialogue", None)
        if session is not None:
            student = session.student
            corrections = session.corrections
            extractor = session.extractor
        else:
            student = self._require(getattr(self.bundle, "student", None), "student")
            corrections = self._require(getattr(self.bundle, "corrections", None),
                                        "correction handler")
            extractor = self._extractor
        cf = extractor.extract_correction(text)     # EXISTING extraction
        events = []
        for fact in cf.corrected:
            before = corrections.ledger.current(fact.subject, fact.relation) \
                if corrections.ledger is not None else None
            events.append(corrections.apply(        # EXISTING plastic override path
                student, after=fact, before=before))
        return {"n_corrections": len(events),
                "corrections": [e.to_dict() for e in events]}

    # ------------------------------------------------------------------ #
    # pause / resume — the ONE flag the bridge owns
    # ------------------------------------------------------------------ #

    def _do_pause(self, args: Dict, rid: str) -> Dict:
        self.paused = True
        if self.emitter is not None:
            self.emitter.paused = True
        return {"paused": True}

    def _do_resume(self, args: Dict, rid: str) -> Dict:
        self.paused = False
        if self.emitter is not None:
            self.emitter.paused = False
        return {"paused": False}

    # ------------------------------------------------------------------ #
    # hostmode requests — through the authority gate ONLY
    # ------------------------------------------------------------------ #

    def _do_approve(self, args: Dict, rid: str) -> Dict:
        """approve_request MEANS 'submit through the HostRuntime gate'. The gate
        decides; this layer can neither execute nor bypass anything."""
        host = self._require(getattr(self.bundle, "host", None), "hostmode authority")
        op = args.get("op")
        if not isinstance(op, str) or op not in tuple(o.value for o in PermittedOperation):
            raise GymActionRejected(
                "unknown_operation",
                f"op {op!r} is not in the closed PermittedOperation enum "
                f"{[o.value for o in PermittedOperation]}")
        op_args = args.get("args")
        if not isinstance(op_args, dict):
            raise GymActionRejected("bad_args",
                                    "approve_request requires args['args'] as a dict")
        proposal = Proposal(op=PermittedOperation(op), args=op_args,
                            arm=str(args.get("arm", "godot")))
        outcome = host.execute(proposal)            # THE AUTHORITY GATE
        out = {"outcome_status": str(outcome.status),
               "outcome_reason": outcome.reason,
               "detail": str(outcome.detail),
               "seq": int(outcome.seq)}
        if outcome.grant is not None:
            out["grant"] = dict(outcome.grant)
        if outcome.state_payload is not None:
            out["state_payload"] = outcome.state_payload
        if outcome.resources is not None:
            out["resources"] = outcome.resources
        return out

    def _do_deny_request(self, args: Dict, rid: str) -> Dict:
        """Refuse to forward a request to the authority at all. Recorded in the
        bridge log and emitted as hostmode.request/denied_by_gym; nothing is
        executed, nothing reaches the gate."""
        target = args.get("request_id")
        if not isinstance(target, str) or not target.strip():
            raise GymActionRejected("bad_args",
                                    "deny_request requires args['request_id']")
        op = args.get("op")
        if not isinstance(op, str) or op not in tuple(o.value for o in PermittedOperation):
            raise GymActionRejected(
                "unknown_operation",
                f"op {op!r} is not in the closed PermittedOperation enum "
                f"{[o.value for o in PermittedOperation]}")
        reason = str(args.get("reason", "denied by gym host"))
        if self.emitter is not None:
            self.emitter.queue_event("hostmode.request", {
                "request_id": str(target),
                "op": op,
                "status": "denied_by_gym",
                "verdict": "denied_by_gym",        # client field alias
                "proposal": op,                     # client field alias
                "reason": reason[:200],
                "detail": f"denied by gym request {rid} before the authority gate",
                "arm": str(args.get("arm", "godot")),
            })
        return {"denied": target, "reason": reason}

    # ------------------------------------------------------------------ #
    # set_goal — BRIDGE-owned state, surfaced for the devhistory lane
    # ------------------------------------------------------------------ #

    def _do_set_goal(self, args: Dict, rid: str) -> Dict:
        goal_text = args.get("goal")
        if not isinstance(goal_text, str) or not goal_text.strip():
            raise GymActionRejected("bad_args",
                                    "set_goal requires non-empty args['goal']")
        self.goal_state.clear()
        self.goal_state.update({
            "goal": str(goal_text),
            "set_by": str(args.get("client", "godot")),
            "set_request_id": rid,
            "set_ts": float(self._clock()),
        })
        # The bridge never writes the devhistory ledger; the goal state lives in
        # bridge-owned memory (surfaced in every snapshot) for the runtime lane
        # to fold through DevelopmentalHistory when it chooses to.
        return {"goal": self.goal_state["goal"], "stored": True}
