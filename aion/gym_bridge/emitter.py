"""GymEmitter: OBSERVES the AION runtime and turns state deltas into gym events.

OWNER LAW (verbatim, 2026-11-25): "Python remains authoritative. Godot subscribes
to events and requests permitted actions. It never writes directly into plastic
state or bypasses the AION runtime's authority controls."

This module is half of that law's Python side: it READS the runtime's observable
objects (relational inventory readouts, the episodic store, the devhistory
ledger, the hostmode authority's decision records, the consolidation ledger) and
projects deltas into validated schema events. It NEVER mutates any of them —
every access here is a read API call or a property read. The one thing the
bridge owns is its own bookkeeping (baselines, the goal state handed over by
GymActions.set_goal, and the pause flag set by GymActions.pause/resume), which
is bridge state, not plastic state.

Emission model:
  - snapshot()          a full state.snapshot event, built from scratch on call.
  - poll_events()       diff the observables against the last-poll baseline and
                        return the events for everything that changed since.
  - subscribe(cb)       push mode: the callback is invoked for every produced
                        event (polled or queued). A consumer should use poll OR
                        subscribe, not both (double delivery).
  - queue_nav_action / queue_dream / queue_event
                        push-side constructors for events the runtime lane
                        hands over (nav decisions, locked dream records).
Throttling: connection.updated is emitted on CHANGE with a configurable minimum
interval (default: emit on change, max 1/second). A change inside the interval
is remembered and flushed on the first later poll past the interval — a change
is never dropped, only delayed.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from aion.gym_bridge.schema import (
    DREAM_KINDS,
    SchemaError,
    make_event,
    validate,
)

SNAPSHOT_ROW_CAP = 64          # relational rows / stage names kept in one snapshot
TEXT_LIMIT = 480               # experience text kept on the wire


class GymBridgeError(RuntimeError):
    """Explicit bridge misuse or a failed runtime readout. Never silent."""


@dataclass
class RuntimeBundle:
    """The runtime handles the bridge observes. Everything is optional and
    DUCK-TYPED (the real runtime objects or test stubs); missing handles simply
    produce empty sections. The bridge never constructs runtime state itself.

      instance_id   stable id of the AION instance being observed
      relational    RelationalMemory-like: .inventory() -> rows with live strengths
      interface     RelationalInterface-like: .describe() -> dict
      episodic      EpisodicStore-like: .events() -> Episode-like (to_dict)
      devhistory    DevelopmentalHistory-like: .name, .entries, .nodes
      host          HostRuntime-like: .caps, .records, .grants, .violations
      consolidation ConsolidationLedger-like: .events, .violations
    """

    instance_id: str = "aion"
    relational: Any = None
    interface: Any = None
    episodic: Any = None
    devhistory: Any = None
    host: Any = None
    consolidation: Any = None


def _json_num(v: Any) -> Any:
    """JSON-safe float projection (None stays None; non-finite -> None)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    return v


class GymEmitter:
    """Read-only observer of the runtime -> validated gym events."""

    def __init__(self, bundle: RuntimeBundle,
                 clock: Optional[Callable[[], float]] = None,
                 connection_update_interval: float = 1.0,
                 goal_state: Optional[Dict] = None,
                 paused: bool = False):
        self.bundle = bundle
        self._clock = clock or time.time
        self.connection_update_interval = float(connection_update_interval)
        # ---- bridge-owned state (never plastic) ----
        self.goal_state: Dict = dict(goal_state or {})
        self.paused = bool(paused)
        self._started_ts = float(self._clock())
        # ---- baselines ----
        self._episodic_seen = 0
        self._dev_entries_seen = 0
        self._host_records_seen = 0
        self._consolidation_seen = 0
        self._lineage_len_seen: Optional[int] = None
        self._stage_count_seen: Optional[int] = None
        self._name_seen: Optional[str] = None
        self._relational_sig: Optional[Dict[str, tuple]] = None
        self._last_connection_emit = float("-inf")
        self._connection_pending = False
        self._pending_changed: List[str] = []   # keys changed inside the throttle window
        # ---- delivery ----
        self._queue: List[Dict] = []
        self._subscribers: List[Callable[[Dict], Any]] = []
        self._arm_baselines()   # events are for CHANGES; current state is the baseline

    # ------------------------------------------------------------------ #
    # snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict:
        """One full state.snapshot event, read entirely from the runtime handles.
        The dashboard-alias fields (mode / plasticity / experience_steps /
        uptime_s / memory_inventory / devhistory_stage / relationships) are
        projections of the SAME real state, never invented numbers."""
        uptime_s = max(0.0, float(self._clock()) - self._started_ts)
        payload: Dict[str, Any] = {
            "instance_id": self.bundle.instance_id,
            "name": self._instance_name(),
            "paused": self.paused,
            "runtime_uptime_s": uptime_s,
            # Godot dashboard aliases (bridge-owned flag + the same clock):
            "mode": "paused" if self.paused else "live",
            "uptime_s": uptime_s,
        }
        if self.goal_state:
            payload["goal"] = dict(self.goal_state)
        rel = self._relational_section()
        if rel is not None:
            payload["relational"] = rel
            payload["relationships"] = self._relationships_map()
            payload["memory_inventory"] = {"facts": int(rel["n_relations"])}
            plasticity = self._plasticity()
            if plasticity is not None:
                payload["plasticity"] = plasticity
        epi = self._episodic_section()
        if epi is not None:
            payload["episodic"] = epi
            payload["experience_steps"] = int(epi["n_events"])
        dev = self._devhistory_section()
        if dev is not None:
            payload["devhistory"] = dev
            payload["devhistory_stage"] = str(dev["current_name"])
        hm = self._hostmode_section()
        if hm is not None:
            payload["hostmode"] = hm
        con = self._consolidation_section()
        if con is not None:
            payload["consolidation"] = con
        iface = self._interface_section()
        if iface is not None:
            payload["interface"] = iface
        return make_event("state.snapshot", payload, clock=self._clock)

    # -- snapshot sections (each read-only) ---------------------------------- #

    def _instance_name(self) -> str:
        dev = self.bundle.devhistory
        if dev is not None and getattr(dev, "name", None):
            return str(dev.name)
        return "AION"

    def _relational_section(self) -> Optional[Dict]:
        rel = self.bundle.relational
        if rel is None or not callable(getattr(rel, "inventory", None)):
            return None
        rows = rel.inventory()
        out_rows = []
        for r in rows[:SNAPSHOT_ROW_CAP]:
            out_rows.append({
                "key": r.get("key"),
                "pair_index": r.get("pair_index"),
                "full": _json_num(r.get("full")),
                "plastic": _json_num(r.get("plastic")),
                "consolidated": _json_num(r.get("consolidated")),
                "superseded_by": r.get("superseded_by"),
            })
        n_transmitting = sum(1 for r in rows
                             if isinstance(r.get("full"), (int, float))
                             and r["full"] > 0)
        return {"n_relations": len(rows), "n_transmitting": n_transmitting,
                "rows": out_rows, "rows_capped": len(rows) > SNAPSHOT_ROW_CAP}

    def _relationships_map(self) -> Dict[str, Dict]:
        """key -> live strengths (the display-integrity feed the Godot client
        compares its drawn edges against). Same rows, same numbers."""
        rel = self.bundle.relational
        out: Dict[str, Dict] = {}
        for r in rel.inventory()[:SNAPSHOT_ROW_CAP]:
            out[str(r.get("key"))] = {
                "full": _json_num(r.get("full")),
                "plastic": _json_num(r.get("plastic")),
                "consolidated": _json_num(r.get("consolidated")),
            }
        return out

    def _plasticity(self) -> Optional[float]:
        """Mean plastic strength across live relations (the substrate's live
        plastic mass per relation). None when there are no rows — never a zero
        dressed up as data."""
        rel = self.bundle.relational
        rows = rel.inventory()
        vals = [_json_num(r.get("plastic")) for r in rows]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _episodic_section(self) -> Optional[Dict]:
        epi = self.bundle.episodic
        if epi is None or not callable(getattr(epi, "events", None)):
            return None
        events = epi.events()
        kinds: Dict[str, int] = {}
        for ev in events:
            kinds[getattr(ev, "kind", "?")] = kinds.get(getattr(ev, "kind", "?"), 0) + 1
        last_seq = max((int(getattr(ev, "seq", 0)) for ev in events), default=0)
        return {"n_events": len(events), "kinds": kinds, "last_seq": last_seq}

    def _devhistory_section(self) -> Optional[Dict]:
        dev = self.bundle.devhistory
        if dev is None:
            return None
        entries = list(getattr(dev, "entries", []) or [])
        nodes = list(getattr(dev, "nodes", []) or [])
        stages = [str(getattr(n, "name", n)) for n in nodes[:SNAPSHOT_ROW_CAP]]
        return {"n_entries": len(entries), "n_stages": len(nodes),
                "current_name": self._instance_name(), "stages": stages}

    def _hostmode_section(self) -> Optional[Dict]:
        host = self.bundle.host
        if host is None:
            return None
        return {
            "caps": dict(getattr(host, "caps", {}) or {}),
            "n_records": len(list(getattr(host, "records", []) or [])),
            "n_grants": len(list(getattr(host, "grants", []) or [])),
            "n_violations": len(list(getattr(host, "violations", []) or [])),
        }

    def _consolidation_section(self) -> Optional[Dict]:
        con = self.bundle.consolidation
        if con is None:
            return None
        violations = list(getattr(con, "violations", []) or [])
        out = {"n_events": len(list(getattr(con, "events", []) or []))}
        if violations is not None and hasattr(con, "violations"):
            out["invariant_ok"] = len(violations) == 0
        return out

    def _interface_section(self) -> Optional[Dict]:
        iface = self.bundle.interface
        if iface is None or not callable(getattr(iface, "describe", None)):
            return None
        return iface.describe()

    def state_fingerprint(self) -> Dict:
        """Canonical READ-ONLY fingerprint of the observed runtime state. The
        server audits every denied action against it: a denial must leave this
        EXACTLY unchanged (never a silent partial mutation). Same read pattern
        as snapshot() — present handles only, nothing constructed."""
        fp: Dict[str, Any] = {
            "instance_id": self.bundle.instance_id,
            "name": self._instance_name(),
            "goal": dict(sorted(self.goal_state.items())),
        }
        rel_sig = self._relational_signature()
        if rel_sig is not None:
            fp["relational"] = {k: list(v) for k, v in sorted(rel_sig.items())}
        epi = self.bundle.episodic
        if epi is not None and callable(getattr(epi, "events", None)):
            events = epi.events()
            fp["episodic"] = {
                "n": len(events),
                "last_event_id": str(getattr(events[-1], "event_id", "")) if events else "",
            }
        dev = self.bundle.devhistory
        if dev is not None:
            fp["devhistory"] = {
                "n_entries": len(list(getattr(dev, "entries", []) or [])),
                "n_nodes": len(list(getattr(dev, "nodes", []) or [])),
            }
        host = self.bundle.host
        if host is not None:
            fp["hostmode"] = {"n_records": len(list(getattr(host, "records", []) or []))}
        con = self.bundle.consolidation
        if con is not None and hasattr(con, "events"):
            fp["consolidation"] = {"n_events": len(list(con.events or []))}
        return fp

    # ------------------------------------------------------------------ #
    # poll: deltas -> events
    # ------------------------------------------------------------------ #

    def poll_events(self) -> List[Dict]:
        """Diff every observable against its baseline; return the new events.
        Push-side events queued since the last poll (queue_nav_action /
        queue_dream / queue_event) are drained here too — already delivered to
        subscribers when queued, so they are NOT re-notified. Baselines advance
        only for events actually returned."""
        produced: List[Dict] = []
        if self._queue:
            produced, self._queue = self._queue, []
        deltas: List[Dict] = []
        deltas += self._poll_episodic()
        deltas += self._poll_hostmode()
        deltas += self._poll_consolidation()
        deltas += self._poll_devhistory()
        deltas += self._poll_nomen()
        deltas += self._poll_connection()
        for ev in deltas:
            self._notify(ev)                 # subscribers only; poll owns delivery
        return produced + deltas

    def _emit(self, event_type: str, payload: Dict) -> Dict:
        return make_event(event_type, payload, clock=self._clock)

    # -- episodic -> experience.created ---------------------------------------- #

    def _poll_episodic(self) -> List[Dict]:
        epi = self.bundle.episodic
        if epi is None or not callable(getattr(epi, "events", None)):
            return []
        events = epi.events()
        new = events[self._episodic_seen:]
        self._episodic_seen = len(events)
        out = []
        for ev in new:
            d = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
            text = str(d.get("text", ""))
            if len(text) > TEXT_LIMIT:
                text = text[:TEXT_LIMIT] + f"... ({len(d.get('text', ''))} chars)"
            out.append(self._emit("experience.created", {
                "event_id": str(d.get("event_id", "")),
                "episode_id": str(d.get("event_id", "")),   # client field alias
                "kind": str(d.get("kind", "")),
                "text": text,
                "seq": int(d.get("seq", 0)),
                "channel": str(d.get("channel", "direct")),
                "teacher": str(d.get("teacher", "unknown")),
                "transcript_id": str(d.get("transcript_id")) if d.get("transcript_id") else "",
                "plastic_update_ids": [int(u) for u in d.get("plastic_update_ids", [])],
                "tags": {str(k): str(v) for k, v in (d.get("tags") or {}).items()},
                "source": "episodic",
                "ts": float(d.get("ts", 0.0)),
            }))
        return out

    # -- hostmode decision records -> hostmode.request -------------------------- #

    def _poll_hostmode(self) -> List[Dict]:
        host = self.bundle.host
        if host is None:
            return []
        records = list(getattr(host, "records", []) or [])
        new = records[self._host_records_seen:]
        self._host_records_seen = len(records)
        out = []
        for rec in new:
            status = str(rec.get("outcome_status", "submitted"))
            if status not in ("executed", "authority_denied", "error", "timed_out"):
                status = "submitted"
            payload = {
                "request_id": f"host-{int(rec.get('seq', 0)):06d}",
                "op": str(rec.get("choice", "")),
                "status": status,
                "verdict": status,                      # client field alias
                "proposal": str(rec.get("choice", "")),  # client field alias
                "seq": int(rec.get("seq", 0)),
                "arm": str(rec.get("arm", "?")),
                "detail": str(rec.get("outcome_detail", "")),
            }
            reason = rec.get("outcome_reason")
            if reason:
                payload["reason"] = str(reason)
            grant = rec.get("grant")
            if isinstance(grant, dict):
                payload["grant"] = grant
            caps = getattr(host, "caps", None)
            if isinstance(caps, dict):
                payload["caps"] = {k: _json_num(v) for k, v in caps.items()}
            out.append(self._emit("hostmode.request", payload))
        return out

    # -- consolidation ledger -> memory.consolidated ---------------------------- #

    def _poll_consolidation(self) -> List[Dict]:
        con = self.bundle.consolidation
        if con is None or not hasattr(con, "events"):
            return []
        events = list(con.events or [])
        new = events[self._consolidation_seen:]
        self._consolidation_seen = len(events)
        violations = list(getattr(con, "violations", []) or [])
        out = []
        for ev in new:
            payload = {"n_events": len(events), "note": _jsonable_note(ev)}
            if hasattr(con, "violations"):
                payload["invariant_ok"] = len(violations) == 0
                payload["preservation_verified"] = len(violations) == 0  # client alias
            pairs = []
            if isinstance(ev, dict) and isinstance(ev.get("pair_index"), int):
                payload["pairs"] = [{"pair_index": ev["pair_index"]}]
                pairs.append(ev["pair_index"])
                payload["moved"] = pairs                 # client field alias
            out.append(self._emit("memory.consolidated", payload))
        return out

    # -- devhistory -> experience.created / dream.* / development.stage ---------- #

    def _poll_devhistory(self) -> List[Dict]:
        dev = self.bundle.devhistory
        if dev is None:
            return []
        entries = list(getattr(dev, "entries", []) or [])
        new = entries[self._dev_entries_seen:]
        self._dev_entries_seen = len(entries)
        out = []
        for entry in new:
            kind = str(entry.get("kind", ""))
            payload = entry.get("payload") or {}
            if kind == "experience":
                # the episodic store is the experience source when both exist
                # (the same event would otherwise be delivered twice)
                if self.bundle.episodic is not None:
                    continue
                out.append(self._emit("experience.created", {
                    "event_id": str((payload.get("episode") or {}).get("event_id", "")),
                    "kind": str((payload.get("episode") or {}).get("kind", "observation")),
                    "text": str((payload.get("episode") or {}).get("text", "")),
                    "source": "devhistory",
                }))
            elif kind == "dream":
                out.append(self._dream_event_from_entry(entry, payload))
            elif kind in ("growth", "renamed", "observed_world", "relationships",
                          "test", "independence", "birth"):
                # non-dream entries only matter when they created a new NAMED STAGE;
                # stage transitions are detected in _poll_nomen/_stages below.
                pass
        out += self._poll_stages(entries)
        return out

    def _dream_event_from_entry(self, entry: Dict, payload: Dict) -> Dict:
        inner = payload.get("event") if isinstance(payload.get("event"), dict) else {}
        sub = str(inner.get("kind", "started")) if inner.get("kind") in DREAM_KINDS \
            else "started"
        dream_id = str(inner.get("dream_id") or f"dream-{int(entry.get('seq', 0)):06d}")
        if sub == "started":
            body: Dict[str, Any] = {"dream_id": dream_id,
                                    "cycle_id": dream_id}   # client field alias
            if isinstance(inner.get("seed"), int):
                body["seed"] = inner["seed"]
            note = inner.get("note") or payload.get("note")
            if note:
                body["note"] = str(note)[:200]
            return self._emit("dream.started", body)
        if sub == "hypothesis":
            body = {"dream_id": dream_id,
                    "cycle_id": dream_id,                   # client field alias
                    "hypothesis_id": str(inner.get("hypothesis_id", "h1")),
                    "statement": str(inner.get("statement", ""))}
            if isinstance(inner.get("confidence"), (int, float)) \
                    and not isinstance(inner.get("confidence"), bool):
                body["confidence"] = float(inner["confidence"])
            return self._emit("dream.hypothesis", body)
        verdict = str(inner.get("verdict", "inconclusive"))
        body = {"dream_id": dream_id,
                "cycle_id": dream_id,                       # client field alias
                "hypothesis_id": str(inner.get("hypothesis_id", "h1")),
                "verdict": verdict,
                "accepted": verdict == "confirmed"}         # client view of the verdict
        if isinstance(inner.get("score"), (int, float)) \
                and not isinstance(inner.get("score"), bool):
            body["score"] = float(inner["score"])
            body["measured_improvement"] = float(inner["score"])  # client field alias
        if inner.get("detail"):
            body["detail"] = str(inner["detail"])[:200]
            body["reason"] = str(inner["detail"])[:200]           # client field alias
        return self._emit("dream.evaluated", body)

    def _poll_stages(self, entries: List[Dict]) -> List[Dict]:
        """A new named state (node) appeared -> development.stage."""
        dev = self.bundle.devhistory
        nodes = list(getattr(dev, "nodes", []) or [])
        if self._stage_count_seen is None:
            self._stage_count_seen = len(nodes)
            return []
        new = nodes[self._stage_count_seen:]
        self._stage_count_seen = len(nodes)
        out = []
        previous_name = ""
        if len(nodes) > len(new) and new:
            previous_name = str(getattr(nodes[len(nodes) - len(new) - 1], "name", ""))
        for node in new:
            # the event that created/landed on this named state: its tracker seq
            # is the last event appended to the node (node.tick is a graph
            # counter, NOT the ledger seq — never confuse the two)
            last_seq = int((node.events or [{}])[-1].get("seq", -1))
            matching = [e for e in entries if int(e.get("seq", -1)) == last_seq]
            kind = str(matching[-1].get("kind", "stage")) if matching else "stage"
            desc = str(matching[-1].get("description", "")) if matching else ""
            node_name = str(getattr(node, "name", ""))
            body = {"kind": kind, "description": desc,
                    "name": node_name,
                    "stage": node_name,                 # client field alias
                    "node_id": str(getattr(node, "node_id", "")),
                    "seq": last_seq}
            if previous_name:
                body["previous_stage"] = previous_name  # client field alias
            previous_name = node_name
            tags = matching[-1].get("tags") if matching else None
            if isinstance(tags, dict) and tags:
                body["tags"] = {str(k): str(v) for k, v in tags.items()}
            out.append(self._emit("development.stage", body))
        return out

    # -- nomen -> nomen.changed -------------------------------------------------- #

    def _poll_nomen(self) -> List[Dict]:
        dev = self.bundle.devhistory
        if dev is None:
            return []
        ent = getattr(getattr(dev, "tracker", None), "self_entity", None)
        lineage = list(getattr(ent, "lineage", []) or [])
        if self._lineage_len_seen is None:
            self._lineage_len_seen = len(lineage)
            self._name_seen = self._instance_name()
            return []
        out = []
        if len(lineage) > self._lineage_len_seen:
            new_names = [str(getattr(le, "name", "")) for le in lineage]
            old_name = new_names[self._lineage_len_seen - 1] \
                if self._lineage_len_seen >= 1 else self._name_seen or ""
            body = {"new_name": new_names[-1], "old_name": old_name,
                    "lineage": new_names}
            if getattr(ent, "entity_id", None):
                body["entity_id"] = str(ent.entity_id)
            out.append(self._emit("nomen.changed", body))
            self._lineage_len_seen = len(lineage)
        return out

    # -- relational strengths -> connection.updated (throttled on change) -------- #

    def _relational_signature(self) -> Optional[Dict[str, tuple]]:
        rel = self.bundle.relational
        if rel is None or not callable(getattr(rel, "inventory", None)):
            return None
        sig: Dict[str, tuple] = {}
        for r in rel.inventory():
            sig[str(r.get("key"))] = (
                _json_num(r.get("full")), _json_num(r.get("plastic")),
                _json_num(r.get("consolidated")))
        return sig

    def _connection_events(self, changed_keys: List[str], sig: Dict[str, tuple],
                           prev: Optional[Dict[str, tuple]], reason: str) -> List[Dict]:
        """One connection.updated per changed row: the full relations list plus a
        headline edge (connection_id / from_node / to_node / deltas) so the
        Godot graph can draw the real endpoints. from/to split the relational
        key's first/last '|' segments; u_fast_after is the plastic component,
        theta_after the consolidated component."""
        rel = self.bundle.relational
        rows = (rel.inventory() if rel is not None
                and callable(getattr(rel, "inventory", None)) else [])
        relations = [{
            "key": r.get("key"),
            "pair_index": r.get("pair_index"),
            "full": _json_num(r.get("full")),
            "plastic": _json_num(r.get("plastic")),
            "consolidated": _json_num(r.get("consolidated")),
        } for r in rows[:SNAPSHOT_ROW_CAP]]
        by_key = {str(r.get("key")): r for r in rows}
        out = []
        for key in changed_keys:
            body: Dict[str, Any] = {
                "relations": relations,
                "n_relations": len(rows),
                "changed_keys": [key],
                "reason": reason,
                "source": "relational_inventory",
            }
            row = by_key.get(key)
            if row is not None:
                body["connection_id"] = key
                parts = key.split("|")
                if len(parts) >= 2 and parts[0] and parts[-1]:
                    body["from_node"] = parts[0]
                    body["to_node"] = parts[-1]
                plastic = _json_num(row.get("plastic"))
                consolidated = _json_num(row.get("consolidated"))
                if isinstance(plastic, (int, float)):
                    body["u_fast_after"] = float(plastic)
                if isinstance(consolidated, (int, float)):
                    body["theta_after"] = float(consolidated)
                old = (prev or {}).get(key)
                if old is not None and isinstance(old[1], (int, float)) \
                        and isinstance(plastic, (int, float)):
                    body["delta_u_fast"] = float(plastic) - float(old[1])
            out.append(self._emit("connection.updated", body))
        return out

    def _poll_connection(self) -> List[Dict]:
        sig = self._relational_signature()
        if sig is None:
            return []
        if self._relational_sig is None:
            self._relational_sig = sig
            return []
        if sig == self._relational_sig:
            return []
        prev = self._relational_sig
        changed = [k for k in sig if sig.get(k) != prev.get(k)]
        self._relational_sig = sig
        now = float(self._clock())
        if (now - self._last_connection_emit) < self.connection_update_interval:
            self._connection_pending = True     # remembered, never dropped
            for key in changed:
                if key not in self._pending_changed:
                    self._pending_changed.append(key)
            return []
        self._last_connection_emit = now
        self._connection_pending = False
        return self._connection_events(changed, sig, prev, "strengths changed")

    def _flush_pending_connection(self) -> List[Dict]:
        """Build the events for remembered (throttled) connection changes once
        the interval allows. Pure construction — the caller (poll) delivers."""
        if not self._connection_pending:
            return []
        now = float(self._clock())
        if (now - self._last_connection_emit) < self.connection_update_interval:
            return []
        self._connection_pending = False
        self._last_connection_emit = now
        pending = self._pending_changed
        self._pending_changed = []
        sig = self._relational_signature() or {}
        return self._connection_events(pending, sig, None, "throttled change flushed")

    # ------------------------------------------------------------------ #
    # push side (events the runtime lane hands over directly)
    # ------------------------------------------------------------------ #

    def queue_event(self, event_type: str, payload: Dict) -> Dict:
        """Construct + validate + deliver one event of any of the 11 types.
        Explicit SchemaError on a bad payload — never silent, never dropped."""
        ev = make_event(event_type, payload, clock=self._clock)
        self._deliver(ev)
        return ev

    def queue_nav_action(self, action: str, scenario_id: str,
                         goal_text: str = "", options: Optional[List] = None,
                         rationale: str = "", meta: Optional[Dict] = None) -> Dict:
        """Record one NAV decision (the closed NAV_ACTIONS vocabulary is enforced
        by schema validation). The bridge only FILES the decision — nav logic
        lives in aion/nav, never here."""
        payload: Dict[str, Any] = {"action": str(action), "scenario_id": str(scenario_id),
                                   "choice": str(action)}   # client field alias
        if goal_text:
            payload["goal_text"] = str(goal_text)
            payload["goal"] = str(goal_text)                # client field alias
            payload["goal_text"] = str(goal_text)
        if options:
            payload["options"] = [str(o) for o in options]
        if rationale:
            payload["rationale"] = str(rationale)[:200]
        if meta:
            payload["meta"] = {str(k): str(v) for k, v in meta.items()}
        return self.queue_event("nav.action", payload)

    def queue_dream(self, kind: str, dream_id: str, hypothesis_id: str = "",
                    statement: str = "", verdict: str = "", seed: Optional[int] = None,
                    confidence: Optional[float] = None, score: Optional[float] = None,
                    detail: str = "", note: str = "") -> Dict:
        """File one dream record on the wire. Dreaming itself is LOCKED until
        post-pilot (owner law) — this only forwards records the runtime produced."""
        etype = f"dream.{kind}"
        if kind not in DREAM_KINDS:
            raise SchemaError(f"dream kind {kind!r} must be one of {DREAM_KINDS}")
        if kind == "started":
            payload: Dict[str, Any] = {"dream_id": str(dream_id)}
            if seed is not None:
                payload["seed"] = int(seed)
            if note:
                payload["note"] = str(note)[:200]
        elif kind == "hypothesis":
            payload = {"dream_id": str(dream_id), "hypothesis_id": str(hypothesis_id),
                       "statement": str(statement)}
            if confidence is not None:
                payload["confidence"] = float(confidence)
            if note:
                payload["note"] = str(note)[:200]
        else:
            payload = {"dream_id": str(dream_id), "hypothesis_id": str(hypothesis_id),
                       "verdict": str(verdict)}
            if score is not None:
                payload["score"] = float(score)
            if detail:
                payload["detail"] = str(detail)[:200]
        return self.queue_event(etype, payload)

    # ------------------------------------------------------------------ #
    # delivery
    # ------------------------------------------------------------------ #

    def subscribe(self, callback: Callable[[Dict], Any]) -> Callable[[], None]:
        """Push mode: callback(event) for every produced event. Returns an
        unsubscribe callable. Events delivered here are also queued for poll."""
        self._subscribers.append(callback)

        def _off() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
        return _off

    def _deliver(self, event: Dict) -> None:
        validate(event)                       # never an invalid event leaves the bridge
        self._queue.append(event)
        self._notify(event)

    def _notify(self, event: Dict) -> None:
        for cb in list(self._subscribers):
            cb(event)

    def poll(self) -> List[Dict]:
        """poll_events() plus any throttled connection flush that is now due."""
        flushed = self._flush_pending_connection()
        for ev in flushed:
            self._notify(ev)
        return flushed + self.poll_events()

    def reset_baseline(self) -> None:
        """Treat the CURRENT observable state as already seen (no backfill)."""
        self._arm_baselines()

    def _arm_baselines(self) -> None:
        epi = self.bundle.episodic
        self._episodic_seen = len(epi.events()) if epi is not None \
            and callable(getattr(epi, "events", None)) else 0
        dev = self.bundle.devhistory
        self._dev_entries_seen = len(list(getattr(dev, "entries", []) or [])) \
            if dev is not None else 0
        host = self.bundle.host
        self._host_records_seen = len(list(getattr(host, "records", []) or [])) \
            if host is not None else 0
        con = self.bundle.consolidation
        self._consolidation_seen = len(list(getattr(con, "events", []) or [])) \
            if con is not None and hasattr(con, "events") else 0
        ent = getattr(getattr(dev, "tracker", None), "self_entity", None)
        self._lineage_len_seen = len(list(getattr(ent, "lineage", []) or [])) \
            if dev is not None else None
        self._stage_count_seen = len(list(getattr(dev, "nodes", []) or [])) \
            if dev is not None else None
        self._name_seen = self._instance_name()
        self._relational_sig = self._relational_signature()


def _jsonable_note(ev: Any) -> str:
    """Compact note for a consolidation ledger event (best-effort, bounded)."""
    if isinstance(ev, dict):
        parts = [f"{k}={v}" for k, v in sorted(ev.items())
                 if isinstance(v, (int, float, str, bool))]
        return "; ".join(parts)[:200]
    return str(type(ev).__name__)
