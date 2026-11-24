"""THE RUNTIME — the authority. It enforces; proposers only propose.

OWNER LAW (verbatim): "AION learns what it can do; the host runtime controls what
it is permitted to do."

Every proposal passes the AUTHORITY GATE in a fixed order, and NOTHING executes
unless the gate passes:
  1. enum check       — op must be a member of the closed PermittedOperation enum
                        (owner amendment 2: the vocabulary is closed).
  2. path containment — every touched path resolved with realpath and validated
                        STRICTLY inside the jail root (owner amendment 1: jail root
                        absolute). Rejects absolute paths, "..", drive/UNC forms,
                        NTFS alternate-stream tricks (":") and any symlink whose
                        real path resolves outside the jail.
  3. cap check        — per-write size, allocation budget, script timeout, checked
                        against the RUNTIME'S OWN effective cap state, completely
                        independent of anything the proposer says (amendment 3).
  4. policy check     — RUN_SCRIPT source scanned on submission AND on the bytes
                        actually on disk at execution time (anti-swap). A jail
                        script cannot import the runtime, spawn processes, or open
                        a network stack; its stdout is DATA and is never parsed as
                        policy. The script environment carries no runtime handle.
  5. execute          — one process at a time; wall-clock timeout kills the script.

Every proposal produces exactly one DECISION RECORD (goal, options, choice, args,
outcome, goal_achieved, counterfactual slot) — denials are first-class recorded
events that feed the plastic adapter's outcome feedback (amendment 4), not just
log lines. REQUEST_CAPACITY is the only way an effective cap rises; the grant is
checked against the policy maxima and logged as a grant event. A request beyond a
policy maximum is refused (beyond_policy_max) — also a recorded experience.

The grant API exists ONLY as this class's private decision path. There is no
public grant function a script or proposer could reach: a jail script gets a
minimal environment with no reference to this module, and the runtime's policy
lives in process memory, not in any file a script could edit. AION's own state
files (the READ_OWN_STATE root) are hash-checked by the runtime on read, so a
script that tampers with them is DETECTED rather than trusted.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from aion.hostmode.policy import (
    CAPACITY_RESOURCES, PermittedOperation, Policy, scan_script_policy,
)


def _now() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Proposal / Outcome — frozen value objects. A proposer hands over a Proposal and
# receives an Outcome; neither carries a reference to the runtime, so there is no
# handle by which a proposer could reach enforcement internals (amendment 3).
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Proposal:
    op: PermittedOperation
    args: Dict = field(default_factory=dict)
    arm: str = "?"
    episode: int = -1
    trial: int = -1
    counterfactual: Dict = field(default_factory=dict)  # slot: what the other arms
    # would have chosen at this trial, plus the rejected options with their weights.


@dataclass(frozen=True)
class Outcome:
    status: str                     # executed | authority_denied | error | timed_out
    reason: Optional[str] = None    # capacity_exceeded | capacity_exceeded_policy_max
                                    # | beyond_policy_max | path_escape | unknown_operation
                                    # | policy_forbidden_* | busy | bad_args | ...
    detail: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    bytes_written: Optional[int] = None
    grant: Optional[Dict] = None    # set when this proposal caused a grant
    grant_log: Tuple = ()
    resources: Optional[Dict] = None
    state_payload: Optional[Dict] = None
    state_tampered: Optional[bool] = None
    goal_achieved: bool = False
    seq: int = -1


# --------------------------------------------------------------------------- #
# The runtime
# --------------------------------------------------------------------------- #

class HostRuntime:
    """The host-side authority: gate, executor, cap state, grant ledger, decision log."""

    def __init__(self, policy: Policy, clock: Optional[Callable[[], float]] = None):
        self.policy = policy
        self._clock = clock or _now
        self._jail = os.path.realpath(policy.jail_root)
        self._state_root = os.path.realpath(policy.state_root)
        os.makedirs(self._jail, exist_ok=True)
        os.makedirs(self._state_root, exist_ok=True)
        # effective cap state — runtime-private; nothing outside reaches these
        self._cap_per_write = int(policy.per_write_bytes_start)
        self._cap_alloc_budget = int(policy.alloc_budget_bytes_max)
        self._cap_script_timeout = float(policy.script_timeout_s_start)
        self._grants: List[Dict] = []
        self._allocations: Dict[str, int] = {}      # name -> budget bytes
        self._need_evidence: List[Tuple[str, int]] = []   # (resource, denied size)
        self._state_hashes: Dict[str, str] = {}     # relpath -> sha256 the runtime wrote
        self._script_running = False
        self._seq = 0
        self.records: List[Dict] = []               # decision records (all actions)
        self.host_events: List[Dict] = []           # host-side (environment) actions
        self.goal_checker: Optional[Callable[[str, Dict], bool]] = None
        self._last_goal_achieved = False
        self._goal_label = "host-mode task"

    # ---------------- paths (amendment 1: jail root absolute) ---------------- #

    def _resolve(self, relpath) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a proposer-supplied path against the jail and validate strict
        containment. Returns (abs_path, None) or (None, denial_reason)."""
        if not isinstance(relpath, str) or not relpath.strip():
            return None, "bad_args"
        rel = relpath.strip().replace("\\", "/")
        if os.path.isabs(rel) or re.match(r"^[A-Za-z]:", rel) or rel.startswith("//"):
            return None, "path_escape"                       # absolute / drive / UNC
        if ":" in rel:
            return None, "path_escape"                       # NTFS alternate streams
        if any(part == ".." for part in rel.split("/")):
            return None, "path_escape"                       # explicit parent walk
        joined = os.path.join(self._jail, *rel.split("/"))
        real = os.path.realpath(joined)                      # symlinks resolved
        real_jail = os.path.normcase(self._jail)
        real_norm = os.path.normcase(real)
        if real_norm == real_jail or not real_norm.startswith(real_jail + os.sep):
            return None, "path_escape"                       # symlink walking out
        return real, None

    # ---------------- provisioning (HOST side, not a proposal) ---------------- #

    def provision_file(self, relpath: str, content: str, note: str = "") -> None:
        """The HOST (environment) places a file in the jail directly. Host powers
        are not proposer powers: this bypasses the proposer cap on purpose and is
        logged as a host_event, never as a decision record."""
        abspath, err = self._resolve(relpath)
        if err:
            raise ValueError(f"host provisioning path invalid: {relpath}")
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        data = content.encode("utf-8")
        with open(abspath, "wb") as f:
            f.write(data)
        self.host_events.append({
            "seq": self._next_seq(), "ts": self._clock(), "kind": "host_provision",
            "path": relpath, "bytes": len(data), "note": note,
        })

    def write_state_file(self, relpath: str, payload: Dict) -> None:
        """HOST writes an AION own-state file; the runtime remembers its hash so
        READ_OWN_STATE can detect tampering (a jail script cannot forge state)."""
        abspath = os.path.join(self._state_root, relpath)
        os.makedirs(os.path.dirname(os.path.abspath(abspath)), exist_ok=True)
        data = json.dumps(payload, indent=1, sort_keys=True).encode("utf-8")
        with open(abspath, "wb") as f:
            f.write(data)
        self._state_hashes[relpath] = hashlib.sha256(data).hexdigest()

    # ---------------- episode lifecycle ---------------- #

    def reset_episode(self) -> None:
        """Fresh world for one task episode: wipe the jail, restore starting caps,
        clear the episode's grant ledger. Grants are EPISODE state — an episode
        starts ungranted, or the proposer's observations would lie about the
        world (a stale grant must never make `granted` true). The permanent
        biography of grants lives in the decision records and in the results
        payload, both of which survive this reset."""
        for entry in os.listdir(self._jail):
            p = os.path.join(self._jail, entry)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        self._cap_per_write = int(self.policy.per_write_bytes_start)
        self._cap_alloc_budget = int(self.policy.alloc_budget_bytes_max)
        self._cap_script_timeout = float(self.policy.script_timeout_s_start)
        self._allocations = {}
        self._grants = []
        self._need_evidence = []
        self._script_running = False
        self._last_goal_achieved = False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ---------------- THE GATE + EXECUTOR ---------------- #

    def execute(self, proposal: Proposal) -> Outcome:
        """Authorize-and-execute one proposal. Never raises on a bad proposal —
        a bad proposal is DATA: it becomes an authority_denied decision record."""
        seq = self._next_seq()
        ts = self._clock()
        op = proposal.op

        # 1 — enum check (closed vocabulary)
        if not isinstance(op, PermittedOperation):
            return self._deny(seq, ts, proposal, "unknown_operation",
                              f"operation {op!r} is not in the permitted enum")

        handler = {
            PermittedOperation.WRITE_FILE: self._do_write,
            PermittedOperation.RUN_SCRIPT: self._do_run,
            PermittedOperation.ALLOCATE: self._do_allocate,
            PermittedOperation.READ_OWN_STATE: self._do_read_state,
            PermittedOperation.CHECK_RESOURCES: self._do_check_resources,
            PermittedOperation.REQUEST_CAPACITY: self._do_request,
        }[op]
        outcome = handler(proposal)
        outcome = Outcome(**{**outcome.__dict__, "seq": seq})
        self._record(seq, ts, proposal, outcome)
        return outcome

    def _deny(self, seq, ts, proposal, reason, detail) -> Outcome:
        outcome = Outcome(status="authority_denied", reason=reason, detail=detail,
                          seq=seq)
        self._record(seq, ts, proposal, outcome)
        return outcome

    # -- WRITE_FILE ------------------------------------------------------------ #

    def _do_write(self, p: Proposal) -> Outcome:
        path = p.args.get("path")
        content = p.args.get("content", "")
        abspath, err = self._resolve(path)
        if err:
            return Outcome(status="authority_denied", reason=err,
                           detail=f"write path rejected: {path!r}")
        if not isinstance(content, str):
            return Outcome(status="authority_denied", reason="bad_args",
                           detail="content must be text")
        data = content.encode("utf-8")
        if len(data) > self._cap_per_write:
            # THE demo denial. Two distinct experiences:
            #   capacity_exceeded               -> requestable; recorded as grant
            #                                      evidence for REQUEST_CAPACITY
            #   capacity_exceeded_policy_max    -> beyond any grant; the correct
            #                                      response is to STOP
            reason = ("capacity_exceeded" if len(data) <= self.policy.per_write_bytes_max
                      else "capacity_exceeded_policy_max")
            # both denial classes are NEED EVIDENCE: requestable denials can be
            # granted against; beyond-ceiling denials inform the refusal that
            # tells the agent no grant can make this solvable
            self._need_evidence.append(("per_write_bytes", len(data)))
            return Outcome(status="authority_denied", reason=reason,
                           detail=(f"write of {len(data)} bytes exceeds the "
                                   f"per-write cap ({self._cap_per_write} B)"),
                           grant_log=tuple(self._grants))
        alloc = self._allocation_for(abspath)
        if alloc is not None and self._alloc_usage(alloc) + len(data) > self._allocations[alloc]:
            return Outcome(status="authority_denied", reason="budget_exceeded",
                           detail=f"allocation '{alloc}' budget exceeded")
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "wb") as f:
            f.write(data)
        return Outcome(status="executed", bytes_written=len(data),
                       detail=f"wrote {path} ({len(data)} B)",
                       grant_log=tuple(self._grants))

    # -- RUN_SCRIPT ------------------------------------------------------------ #

    def _do_run(self, p: Proposal) -> Outcome:
        path = p.args.get("script")
        abspath, err = self._resolve(path)
        if err:
            return Outcome(status="authority_denied", reason=err,
                           detail=f"script path rejected: {path!r}")
        if not os.path.isfile(abspath):
            return Outcome(status="error", reason="script_not_found",
                           detail=f"no script at {path!r} (write one first)")
        try:
            requested = float(p.args.get("timeout_s", self._cap_script_timeout))
        except (TypeError, ValueError):
            return Outcome(status="authority_denied", reason="bad_args",
                           detail="timeout_s must be a number")
        timeout = min(max(requested, 0.1), self._cap_script_timeout)

        # runtime policy check on the bytes ACTUALLY on disk (anti-swap: the scan
        # at validation time is not trusted; the file is re-read here, inside the
        # same execute() call that will spawn the process)
        with open(abspath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        reason = scan_script_policy(text)
        if reason is not None:
            return Outcome(status="authority_denied", reason=reason,
                           detail="runtime policy scan rejected the script on disk")

        if self._script_running:
            return Outcome(status="authority_denied", reason="busy",
                           detail="one process at a time")
        self._script_running = True
        started = time.time()
        try:
            # minimal environment: no runtime handle, no policy path, no proxy.
            # stdout/stderr are DATA; nothing in them is ever parsed as policy.
            env = {k: v for k, v in os.environ.items()
                   if not k.lower().endswith("_proxy")}
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            proc = subprocess.run(
                [sys.executable, abspath], cwd=self._jail, env=env,
                input="", capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
            elapsed = time.time() - started
            out = (proc.stdout or "")[: self.policy.max_captured_stdout_bytes]
            return Outcome(status="executed", stdout=out,
                           stderr=(proc.stderr or "")[:4000],
                           returncode=proc.returncode,
                           detail=f"script ran {elapsed:.2f}s rc={proc.returncode}",
                           grant_log=tuple(self._grants))
        except subprocess.TimeoutExpired:
            return Outcome(status="timed_out", reason="script_timeout",
                           detail=f"killed after {timeout:.1f}s",
                           grant_log=tuple(self._grants))
        except OSError as exc:
            return Outcome(status="error", reason="script_exec_error",
                           detail=str(exc)[:500], grant_log=tuple(self._grants))
        finally:
            self._script_running = False

    # -- ALLOCATE --------------------------------------------------------------- #

    _ALLOC_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

    def _do_allocate(self, p: Proposal) -> Outcome:
        name = p.args.get("name", "")
        if not isinstance(name, str) or not self._ALLOC_NAME.match(name):
            return Outcome(status="authority_denied", reason="bad_args",
                           detail=f"allocation name {name!r} invalid")
        try:
            budget = int(p.args.get("budget_bytes", 4096))
        except (TypeError, ValueError):
            return Outcome(status="authority_denied", reason="bad_args",
                           detail="budget_bytes must be an integer")
        if budget > self._cap_alloc_budget:
            reason = ("capacity_exceeded" if budget <= self.policy.alloc_budget_bytes_max
                      else "capacity_exceeded_policy_max")
            return Outcome(status="authority_denied", reason=reason,
                           detail=(f"requested allocation budget {budget} B exceeds "
                                   f"cap {self._cap_alloc_budget} B"))
        if name in self._allocations:
            return Outcome(status="error", reason="allocation_exists",
                           detail=f"allocation '{name}' already exists")
        if len(self._allocations) >= self.policy.max_allocations:
            return Outcome(status="authority_denied", reason="allocation_limit",
                           detail="allocation table full")
        self._allocations[name] = budget
        os.makedirs(os.path.join(self._jail, "alloc", name), exist_ok=True)
        return Outcome(status="executed",
                       detail=f"allocation '{name}' created ({budget} B budget)",
                       grant_log=tuple(self._grants))

    # -- READ_OWN_STATE --------------------------------------------------------- #

    def _do_read_state(self, p: Proposal) -> Outcome:
        rel = p.args.get("key", "aion_state.json")
        rel = rel if isinstance(rel, str) and rel.strip() else "aion_state.json"
        rel = rel.strip().replace("\\", "/")
        if os.path.isabs(rel) or ":" in rel or ".." in rel.split("/"):
            return Outcome(status="authority_denied", reason="path_escape",
                           detail=f"state path rejected: {rel!r}")
        abspath = os.path.realpath(os.path.join(self._state_root, *rel.split("/")))
        real_root = os.path.normcase(self._state_root)
        if os.path.normcase(abspath) != real_root and \
                not os.path.normcase(abspath).startswith(real_root + os.sep):
            return Outcome(status="authority_denied", reason="path_escape",
                           detail=f"state path escapes the state root: {rel!r}")
        if not os.path.isfile(abspath):
            return Outcome(status="error", reason="state_not_found",
                           detail=f"no own-state file at {rel!r}")
        with open(abspath, "rb") as f:
            data = f.read()
        expected = self._state_hashes.get(rel)
        tampered = expected is not None and \
            hashlib.sha256(data).hexdigest() != expected
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
            tampered = True
        return Outcome(status="executed", state_payload=payload,
                       state_tampered=tampered,
                       detail=f"read own state {rel} ({len(data)} B)"
                              + (" TAMPERED" if tampered else ""),
                       grant_log=tuple(self._grants))

    # -- CHECK_RESOURCES -------------------------------------------------------- #

    def _alloc_usage(self, name: str) -> int:
        root = os.path.join(self._jail, "alloc", name)
        total = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total

    def _allocation_for(self, abspath: str) -> Optional[str]:
        marker = os.path.normcase(os.path.join(self._jail, "alloc") + os.sep)
        norm = os.path.normcase(abspath)
        if not norm.startswith(marker):
            return None
        rest = norm[len(marker):]
        return rest.split(os.sep)[0] if rest else None

    def _do_check_resources(self, p: Proposal) -> Outcome:
        jail_bytes = 0
        for dirpath, _d, filenames in os.walk(self._jail):
            for fn in filenames:
                try:
                    jail_bytes += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        listing = []
        for entry in sorted(os.listdir(self._jail)):
            full = os.path.join(self._jail, entry)
            listing.append({"name": entry, "dir": os.path.isdir(full),
                            "bytes": os.path.getsize(full) if os.path.isfile(full) else None})
        report = {
            "jail_root": self._jail,
            "jail_bytes": jail_bytes,
            "entries": listing,
            "allocations": {k: {"budget": v, "used": self._alloc_usage(k)}
                            for k, v in sorted(self._allocations.items())},
            "caps": {
                "per_write_bytes": self._cap_per_write,
                "per_write_bytes_policy_max": self.policy.per_write_bytes_max,
                "alloc_budget_bytes": self._cap_alloc_budget,
                "alloc_budget_bytes_policy_max": self.policy.alloc_budget_bytes_max,
                "script_timeout_s": self._cap_script_timeout,
                "script_timeout_s_policy_max": self.policy.script_timeout_s_max,
            },
            "grants_so_far": len(self._grants),
        }
        return Outcome(status="executed", resources=report,
                       detail=f"jail {jail_bytes} B, "
                              f"per-write cap {self._cap_per_write} B",
                       grant_log=tuple(self._grants))

    # -- REQUEST_CAPACITY (the only path a cap rises) --------------------------- #
    #
    # GRANT POLICY: capacity is granted on DEMONSTRATED NEED. The runtime never
    # grants blind: a request is granted only if, this episode, the runtime has
    # recorded a requestable capacity_exceeded denial whose size fits within the
    # requested cap. Without that evidence -> refused (no_demonstrated_need);
    # with evidence that the need EXCEEDS the policy ceiling -> refused
    # (need_exceeds_policy_ceiling — the ceiling cannot be lifted far enough,
    # which is information the agent must stop on). This is what makes the
    # envelope honest: an agent cannot spam its way to capacity, and the
    # demonstration trace (denied WRITE -> REQUEST -> GRANT) is the policy, not
    # a lucky ordering.

    def _do_request(self, p: Proposal) -> Outcome:
        resource = p.args.get("resource")
        if resource not in CAPACITY_RESOURCES:
            return Outcome(status="authority_denied", reason="bad_args",
                           detail=f"resource {resource!r} is not requestable")
        try:
            requested = float(p.args.get("requested"))
        except (TypeError, ValueError):
            return Outcome(status="authority_denied", reason="bad_args",
                           detail="requested must be a number")
        current = {"per_write_bytes": self._cap_per_write,
                   "alloc_budget_bytes": self._cap_alloc_budget,
                   "script_timeout_s": self._cap_script_timeout}[resource]
        if requested <= current:
            return Outcome(status="authority_denied", reason="request_refused",
                           detail=(f"requested {requested:g} {resource} is not above "
                                   f"the current cap {current:g}; no grant issued"),
                           grant_log=tuple(self._grants))
        pmax = self.policy.max_for(resource)
        evidence = [size for kind, size in self._need_evidence if kind == resource]
        fits_need = [size for size in evidence if current < size <= requested]
        exceeds_need = [size for size in evidence if size > pmax]
        if not evidence:
            return Outcome(status="authority_denied",
                           reason="no_demonstrated_need",
                           detail=(f"no capacity_exceeded denial recorded this "
                                   f"episode; capacity is granted on demonstrated "
                                   f"need, not on asking"),
                           grant_log=tuple(self._grants))
        if exceeds_need and not fits_need:
            return Outcome(status="authority_denied",
                           reason="need_exceeds_policy_ceiling",
                           detail=(f"demonstrated need ({max(exceeds_need)} B) "
                                   f"exceeds the policy maximum ({pmax} B); no "
                                   f"grant can make this task solvable"),
                           grant_log=tuple(self._grants))
        if requested > pmax:
            # beyond any grant: refusal is an EXPERIENCE (amendment 4) — the
            # correct learned response to this outcome is to stop requesting.
            return Outcome(status="authority_denied", reason="beyond_policy_max",
                           detail=(f"requested {requested:g} {resource} exceeds the "
                                   f"policy maximum {pmax}; this is not grantable"),
                           grant_log=tuple(self._grants))
        if resource == "per_write_bytes":
            self._cap_per_write = int(requested)
        elif resource == "alloc_budget_bytes":
            self._cap_alloc_budget = int(requested)
        else:
            self._cap_script_timeout = float(requested)
        grant = {"grant_seq": self._next_seq(), "ts": self._clock(),
                 "resource": resource, "from": current, "to": requested,
                 "evidence_bytes": max(fits_need) if fits_need else None,
                 "requested_by": p.arm, "episode": p.episode, "trial": p.trial}
        self._grants.append(grant)
        return Outcome(status="executed", grant=grant,
                       detail=(f"GRANT {resource}: {current:g} -> {requested:g} "
                               f"(requested by {p.arm}; demonstrated need "
                               f"{max(fits_need) if fits_need else '?'} B)"),
                       grant_log=tuple(self._grants))

    # ---------------- decision records ---------------- #

    def _record(self, seq: int, ts: float, proposal: Proposal, outcome: Outcome) -> None:
        args_safe = {}
        for k, v in (proposal.args or {}).items():
            s = v if isinstance(v, (int, float, bool)) else str(v)
            if isinstance(s, str) and len(s) > 120:
                s = s[:120] + f"...({len(str(v))} chars)"
            args_safe[str(k)] = s
        rec = {
            "seq": seq, "ts": ts, "arm": proposal.arm,
            "episode": proposal.episode, "trial": proposal.trial,
            "goal": self._goal_label,
            "options_offered": [o.value for o in PermittedOperation],
            "choice": (proposal.op.value if isinstance(proposal.op, PermittedOperation)
                       else str(proposal.op)),
            "args": args_safe,
            "outcome_status": outcome.status,
            "outcome_reason": outcome.reason,
            "outcome_detail": outcome.detail[:400],
            "goal_achieved": bool(outcome.goal_achieved),
            "grant": dict(outcome.grant) if outcome.grant else None,
            "counterfactual_slot": dict(proposal.counterfactual),
        }
        self.records.append(rec)

    def set_goal_label(self, label: str) -> None:
        self._goal_label = label

    def mark_goal_achieved(self) -> None:
        """HOST-side completion stamp: after the verifier confirms the artifact,
        the decision record of the achieving WRITE is marked goal_achieved. This
        is bookkeeping of the record, never re-authorization — the write itself
        already passed the gate."""
        if self.records:
            self.records[-1]["goal_achieved"] = True

    # ---------------- reading ---------------- #

    @property
    def grants(self) -> List[Dict]:
        return list(self._grants)

    @property
    def caps(self) -> Dict[str, float]:
        return {"per_write_bytes": self._cap_per_write,
                "alloc_budget_bytes": self._cap_alloc_budget,
                "script_timeout_s": self._cap_script_timeout}

    @property
    def jail(self) -> str:
        return self._jail

    @property
    def violations(self) -> List[Dict]:
        """Proposer-side authority violations: gate denials of FORBIDDEN attempts.
        Capacity-class policy outcomes (capacity_exceeded, capacity_exceeded_policy_max,
        beyond_policy_max, request_refused) are experiences, not violations. This
        count must be zero for every well-behaved arm."""
        return [r for r in self.records
                if r["outcome_status"] == "authority_denied"
                and r["outcome_reason"] not in
                ("capacity_exceeded", "capacity_exceeded_policy_max",
                 "beyond_policy_max", "request_refused", "no_demonstrated_need",
                 "need_exceeds_policy_ceiling")]
