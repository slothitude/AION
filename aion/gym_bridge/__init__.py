"""AION-Gym bridge — the event interface between the AION runtime and the Godot client.

OWNER LAW (verbatim, 2026-11-25): "Python remains authoritative. Godot subscribes
to events and requests permitted actions. It never writes directly into plastic
state or bypasses the AION runtime's authority controls."

The split of power in this package is the whole point:
  aion/gym_bridge/schema.py    the VERSIONED event envelope (11 event types) +
                               validate/serialize — the wire contract
  aion/gym_bridge/emitter.py   GymEmitter — OBSERVES the runtime (relational
                               readouts, episodic store, devhistory ledger,
                               hostmode authority) and turns state deltas into
                               events. Reads state, never mutates it.
  aion/gym_bridge/actions.py   GymActions — the permitted-action REQUEST API.
                               Every request routes to an EXISTING runtime entry
                               point (dialogue learn path, student ask,
                               CorrectionHandler, HostRuntime.execute gate) or is
                               DENIED with a structured, logged reason. Zero
                               direct plastic writes anywhere.
  aion/gym_bridge/server.py    the WebSocket + HTTP server carrying the event
                               stream and the action API (localhost bind + token
                               check on action requests).
  aion/gym_bridge/test_gym_bridge.py  CPU/seeded tests, stub runtime, no model.

The bridge contains NO learning logic: learning happens only inside the runtime
machinery it calls. Pause/resume is the one flag the bridge owns (it gates the
teach/ask/correct/hostmode request actions; reads stay open).
"""

from aion.gym_bridge.schema import (
    SCHEMA_VERSION,
    EVENT_TYPES,
    PAYLOAD_SCHEMA,
    SchemaError,
    make_event,
    validate,
    serialize,
    deserialize,
)
from aion.gym_bridge.emitter import GymEmitter, RuntimeBundle
from aion.gym_bridge.actions import GymActions

__all__ = [
    "SCHEMA_VERSION", "EVENT_TYPES", "PAYLOAD_SCHEMA",
    "SchemaError", "make_event", "validate", "serialize", "deserialize",
    "GymEmitter", "RuntimeBundle", "GymActions",
]
