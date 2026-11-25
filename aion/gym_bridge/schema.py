"""The versioned AION-Gym event schema: one envelope, 11 event types, strict validation.

OWNER LAW (verbatim, 2026-11-25): "Python remains authoritative. Godot subscribes
to events and requests permitted actions. It never writes directly into plastic
state or bypasses the AION runtime's authority controls."

This module is the WIRE CONTRACT between the AION runtime (authoritative, this
side) and the Godot Gym client (subscriber, other side). Every event that crosses
the boundary is exactly:

    {"schema_version": 1, "type": <one of 11 types>, "timestamp": float,
     "payload": {...}}

VALIDATION IS EXPLICIT AND CLOSED (never silent): unknown event type, missing
required payload field, wrong field type, UNKNOWN payload field, non-finite
number, or a mismatched schema_version raises SchemaError naming the exact
problem. Nothing is dropped, coerced, or defaulted into the wire silently.

Per-type payload fields are declared below in PAYLOAD_SCHEMA (the one source of
truth; the emitter, the actions layer and the server all validate through
validate()). Domain vocabularies stay closed exactly like the rest of AION:
nav.action actions are the frozen NAV_ACTIONS enum (aion/nav/state.py,
imported read-only) and hostmode.request ops are the closed PermittedOperation
enum (aion/hostmode/policy.py, imported read-only).
"""

import json
import math
import time
from typing import Any, Callable, Dict, Optional, Union

from aion.nav.state import NAV_ACTIONS            # read-only import (closed enum)
from aion.hostmode.policy import PermittedOperation  # read-only import (closed enum)

SCHEMA_VERSION = 1

# The 11 event types (the closed vocabulary of the wire).
EVENT_TYPES = (
    "state.snapshot",
    "experience.created",
    "connection.updated",
    "memory.consolidated",
    "dream.started",
    "dream.hypothesis",
    "dream.evaluated",
    "nomen.changed",
    "nav.action",
    "hostmode.request",
    "development.stage",
)

# hostmode.request status vocabulary (closed, like everything else here).
HOSTMODE_STATUSES = (
    "submitted",          # a request entered the bridge (pre-gate record)
    "executed",           # the authority executed it
    "authority_denied",   # the AUTHORITY denied it (the gate's own decision)
    "error",
    "timed_out",
    "denied_by_gym",      # the bridge refused to forward it (never reaches the gate)
)

# hostmode.request op vocabulary = the frozen hostmode enum (read-only view).
HOSTMODE_OPS = tuple(op.value for op in PermittedOperation)

# Dream sub-event kinds (dreaming itself is LOCKED until post-pilot — the bridge
# only FILES/forwards dream records the runtime produced; it never dreams).
DREAM_KINDS = ("started", "hypothesis", "evaluated")

NAV_ACTIONS_TUPLE = tuple(NAV_ACTIONS)   # ("QUERY", "INFER", "ANSWER", "ASK_TEACHER", "STOP")

# ---------------------------------------------------------------------------
# The per-type payload contract. Field -> declared type key. Required fields
# MUST be present; optional fields MAY be absent; anything else is rejected.
# Type keys: str, int, float, number (int|float, bool excluded), bool, list,
# dict, any.
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMA: Dict[str, Dict[str, Dict[str, str]]] = {
    "state.snapshot": {
        "required": {
            "instance_id": "str",
            "name": "str",                    # the instance's CURRENT nomen name
        },
        "optional": {
            "relational": "dict",             # knowledge inventory headline + rows
            "relationships": "dict",          # key -> live strengths (display feed)
            "episodic": "dict",               # event-log counts
            "devhistory": "dict",             # biography counts + stage names
            "hostmode": "dict",               # caps + decision-record counts
            "consolidation": "dict",          # consolidation ledger counts
            "interface": "dict",              # relational-interface describe()
            "goal": "dict",                   # the bridge-owned goal state
            "paused": "bool",
            "runtime_uptime_s": "number",
            # -- Godot-client dashboard aliases (all real-state projections) ---- #
            "mode": "str",                    # "live" | "paused" (the bridge flag)
            "plasticity": "number",           # mean plastic strength per live relation
            "experience_steps": "int",        # episodic event count
            "uptime_s": "number",             # == runtime_uptime_s (client field name)
            "memory_inventory": "dict",       # {"facts": n_live_relations}
            "devhistory_stage": "str",        # current named stage
        },
    },
    "experience.created": {
        "required": {
            "event_id": "str",
            "kind": "str",                    # lesson | observation | correction
            "text": "str",
        },
        "optional": {
            "seq": "int",
            "channel": "str",                 # transcript | direct
            "teacher": "str",
            "transcript_id": "str",
            "plastic_update_ids": "list",
            "tags": "dict",
            "source": "str",                  # episodic | devhistory
            "ts": "number",
            "episode_id": "str",              # == event_id (client field name)
        },
    },
    "connection.updated": {
        "required": {
            "relations": "list",              # rows of live relational strengths
        },
        "optional": {
            "n_relations": "int",
            "changed_keys": "list",
            "reason": "str",
            "source": "str",
            # -- headline edge (one changed row; Godot graph consumes it) ------- #
            "connection_id": "str",           # the relational key, e.g. kav|operates|tav
            "from_node": "str",               # subject (first | segment)
            "to_node": "str",                 # object (last | segment)
            "u_fast_after": "number",         # plastic component after the change
            "theta_after": "number",          # consolidated component after the change
            "delta_u_fast": "number",         # plastic delta vs the last-poll baseline
        },
    },
    "memory.consolidated": {
        "required": {
            "n_events": "int",                # consolidation events recorded
        },
        "optional": {
            "invariant_ok": "bool",
            "pairs": "list",
            "note": "str",
            "preservation_verified": "bool",  # == invariant_ok (client field name)
            "moved": "list",                  # pair indices folded this cycle
            "cycle_id": "str",
        },
    },
    "dream.started": {
        "required": {"dream_id": "str"},
        "optional": {"seed": "int", "mode": "str", "note": "str",
                     "cycle_id": "str"},      # == dream_id (client field name)
    },
    "dream.hypothesis": {
        "required": {
            "dream_id": "str",
            "hypothesis_id": "str",
            "statement": "str",
        },
        "optional": {"confidence": "number", "note": "str",
                     "cycle_id": "str"},      # == dream_id (client field name)
    },
    "dream.evaluated": {
        "required": {
            "dream_id": "str",
            "hypothesis_id": "str",
            "verdict": "str",                 # e.g. confirmed | rejected | inconclusive
        },
        "optional": {"score": "number", "detail": "str",
                     "cycle_id": "str",       # == dream_id (client field name)
                     "accepted": "bool",      # verdict == "confirmed" (client view)
                     "measured_improvement": "number",  # == score
                     "reason": "str"},        # == detail
    },
    "nomen.changed": {
        "required": {"new_name": "str"},
        "optional": {
            "old_name": "str",
            "entity_id": "str",
            "reason": "str",
            "lineage": "list",
        },
    },
    "nav.action": {
        "required": {
            "action": "str",                  # member of NAV_ACTIONS (closed enum)
            "scenario_id": "str",
        },
        "optional": {"goal_text": "str", "options": "list", "rationale": "str",
                     "meta": "dict",
                     "goal": "str",           # == goal_text (client field name)
                     "choice": "str"},        # == action (client field name)
    },
    "hostmode.request": {
        "required": {
            "request_id": "str",
            "op": "str",                      # member of PermittedOperation (closed enum)
            "status": "str",                  # member of HOSTMODE_STATUSES
        },
        "optional": {
            "reason": "str",
            "detail": "str",
            "seq": "int",
            "arm": "str",
            "grant": "dict",
            "caps": "dict",
            "verdict": "str",                 # == status (client field name)
            "proposal": "str",                # == op (client field name)
        },
    },
    "development.stage": {
        "required": {
            "kind": "str",                    # growth | renamed | observed_world | ...
            "description": "str",
        },
        "optional": {
            "name": "str",                    # the named state the event landed on
            "node_id": "str",
            "seq": "int",
            "tags": "dict",
            "stage": "str",                   # == name (client field name)
            "previous_stage": "str",          # prior named state, when known
            "stage_index": "int",             # runtime-supplied arc index (0-4), if any
        },
    },
}


class SchemaError(ValueError):
    """An explicit contract violation. Never raised silently, never swallowed."""


def _check_field(what: str, value: Any, type_key: str) -> None:
    checks = {
        "str": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, float) and math.isfinite(v),
        "number": lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)
                             and math.isfinite(v)),
        "bool": lambda v: isinstance(v, bool),
        "list": lambda v: isinstance(v, list),
        "dict": lambda v: isinstance(v, dict),
        "any": lambda v: True,
    }
    fn = checks.get(type_key)
    if fn is None:
        raise SchemaError(f"{what}: unknown declared type key {type_key!r}")
    if not fn(value):
        raise SchemaError(
            f"{what}: field must be {type_key}, got {type(value).__name__} ({value!r})")


def validate(event: Any) -> Dict:
    """Validate one event dict against the contract. Returns the event unchanged
    on success; raises SchemaError naming every problem found. Never silent."""
    problems = []
    if not isinstance(event, dict):
        raise SchemaError(f"event must be a dict, got {type(event).__name__}")
    expected_keys = {"schema_version", "type", "timestamp", "payload"}
    missing = sorted(expected_keys - set(event))
    extra = sorted(set(event) - expected_keys)
    if missing:
        problems.append(f"missing envelope key(s) {missing}")
    if extra:
        problems.append(f"unknown envelope key(s) {extra}")
    if problems:
        raise SchemaError("invalid event envelope: " + "; ".join(problems))

    if event["schema_version"] != SCHEMA_VERSION or \
            isinstance(event["schema_version"], bool):
        problems.append(
            f"schema_version must be {SCHEMA_VERSION}, got {event['schema_version']!r}")
    etype = event["type"]
    if etype not in EVENT_TYPES:
        problems.append(f"type {etype!r} is not one of {EVENT_TYPES}")
    _check_field("envelope.timestamp", event["timestamp"], "number")

    payload = event["payload"]
    if not isinstance(payload, dict):
        raise SchemaError(
            f"payload must be a dict, got {type(payload).__name__}")

    if etype in PAYLOAD_SCHEMA and not problems:
        contract = PAYLOAD_SCHEMA[etype]
        for field in contract["required"]:
            if field not in payload:
                problems.append(f"payload missing required field {field!r} for {etype}")
        for field, value in sorted(payload.items()):
            if field in contract["required"]:
                _check_field(f"{etype}.payload.{field}", value, contract["required"][field])
            elif field in contract["optional"]:
                _check_field(f"{etype}.payload.{field}", value, contract["optional"][field])
            else:
                problems.append(
                    f"unknown payload field {field!r} for {etype} "
                    f"(allowed: {sorted(set(contract['required']) | set(contract['optional']))})")

    # -- closed vocabularies (the enum law, enforced at the wire) -------------- #
    if etype == "nav.action" and isinstance(payload.get("action"), str) \
            and payload["action"] not in NAV_ACTIONS_TUPLE:
        problems.append(f"nav.action {payload['action']!r} is not in NAV_ACTIONS "
                        f"{NAV_ACTIONS_TUPLE}")
    if etype == "hostmode.request":
        op = payload.get("op")
        if isinstance(op, str) and op not in HOSTMODE_OPS:
            problems.append(f"hostmode.request op {op!r} is not in PermittedOperation "
                            f"{HOSTMODE_OPS}")
        status = payload.get("status")
        if isinstance(status, str) and status not in HOSTMODE_STATUSES:
            problems.append(f"hostmode.request status {status!r} is not in "
                            f"{HOSTMODE_STATUSES}")

    if problems:
        raise SchemaError(f"invalid {etype!r} event: " + "; ".join(problems))
    return event


def make_event(event_type: str, payload: Dict,
               timestamp: Optional[float] = None,
               clock: Optional[Callable[[], float]] = None) -> Dict:
    """Build + validate one event envelope. timestamp defaults to the injectable
    clock (or wall time). The timestamp is stored as a float, always."""
    if event_type not in EVENT_TYPES:
        raise SchemaError(f"unknown event type {event_type!r} (must be one of {EVENT_TYPES})")
    ts = float(clock() if clock is not None else (timestamp if timestamp is not None
                                                  else time.time()))
    event = {
        "schema_version": SCHEMA_VERSION,
        "type": event_type,
        "timestamp": ts,
        "payload": dict(payload or {}),
    }
    return validate(event)


def serialize(event: Dict) -> str:
    """Validate then encode one event to strict JSON (sort_keys, no NaN/Inf).
    Anything non-JSON-safe raises SchemaError — never a silent stringification."""
    validate(event)
    try:
        return json.dumps(event, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"event is not JSON-safe: {exc}") from exc


def deserialize(text: Union[str, bytes]) -> Dict:
    """Decode + validate one event from JSON text. Explicit errors, never silent."""
    try:
        event = json.loads(text)
    except ValueError as exc:
        raise SchemaError(f"event text is not valid JSON: {exc}") from exc
    return validate(event)
