"""The developmental-history ledger: an AION instance's biography.

OWNER CANON (2026-11-24): "each AION should have a developmental history — its
starting substrate, experiences, learned relationships, dreams, growth events and
independence tests. Every AION could develop differently." The ledger IS the
individual's biography; NOMEN names its transitions.

THE SEAM (aion/nomen/evolution_tracker.py, task 21): every event this ledger
records routes through

    EvolutionTracker.record_transition(TransitionEvent, new_name=None) -> GraphNode

so each event lands attached to a NAMED state of the instance in the same
evolution graph every other named thing uses. Named stages are AION-generated
through NOMEN (observation-derived proposals, chosen by the instance) — this
module never hardcodes a stage name; the renderer only prints whatever NOMEN
produced. Renames go through the registry (identity + verification_status
untouched, lineage appended), non-renaming events attach to the CURRENT named
state.

LAWS honoured here:
  - no dream-engine calls anywhere: record_dream is an honest STUB HOOK that
    accepts and files dream events for the future (dreaming is LOCKED until
    post-pilot). Same for record_growth. Neither imports nor invokes any
    dreaming mechanism.
  - no mechanism imports, no real model calls: this module reads LANDED report
    dicts and episode objects only; it never imports src/nnfs/**, the interface,
    or any networked client.
  - plain JSON only (the NomenStore/EpisodicStore convention): reports are
    sanitised into JSON-safe payloads at record time (tensors -> lists,
    inf/nan -> strings), so persistence round-trips EXACTLY.
  - determinism: the clock is injectable (tests pass a fixed clock); event
    sequence numbers come from the tracker and are monotonic, so ordering is
    stable regardless of wall time.
"""

import json
import math
import os
import time
from typing import Callable, Dict, List, Optional

from aion.nomen.entity_registry import EntityRegistry
from aion.nomen.evolution_tracker import EvolutionTracker, TransitionEvent
from aion.nomen.lineage_graph import GraphNode, LineageGraph
from aion.nomen.name_memory import NomenStore

PAYLOAD_VERSION = 1

STUB_NOTE = ("stub hook only: dreaming/growth engines are LOCKED until post-pilot; "
             "event accepted and filed, nothing invoked")


# --------------------------------------------------------------------------- #
# JSON-safety (the plain-JSON law: nothing non-serialisable ever enters a payload)
# --------------------------------------------------------------------------- #

def _jsonable(obj):
    """Project any recorded value into plain JSON without losing meaning."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    tolist = getattr(obj, "tolist", None)          # tensors / numpy, without importing them
    if callable(tolist):
        return _jsonable(tolist())
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _check_jsonable(substrate: Dict, what: str) -> None:
    try:
        json.dumps(substrate, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{what} must be plain JSON (the ledger's persistence law): {exc}") from exc


def _fmt_value(v, limit: int = 240) -> str:
    """Compact human rendering for the markdown biography; long values truncate
    (the full value always survives in the JSON record)."""
    if isinstance(v, dict):
        s = json.dumps(_jsonable(v), sort_keys=True)
    elif isinstance(v, (list, tuple)):
        s = json.dumps(_jsonable(list(v)))
    else:
        s = str(v)
    if len(s) > limit:
        s = s[:limit] + f" … ({len(s)} chars; full value in the JSON record)"
    return s


def _survival_line(report: Dict) -> str:
    surv = report.get("survival_ratio") or report.get("survival_ratios") or {}
    if not surv:
        return ""
    parts = []
    for cap in sorted(surv):
        v = surv[cap]
        if isinstance(v, float) and math.isinf(v):
            s = "inf"
        elif v is None:
            s = "-"
        else:
            try:
                s = f"{float(v):.2f}"
            except (TypeError, ValueError):
                s = str(v)
        parts.append(f"{cap}={s}")
    return "survival " + ", ".join(parts)


def _test_description(report: Dict) -> str:
    """Best-effort description from the LANDED report shapes (retention probe
    report, knowledge_discovery run_battery report, consolidation audit)."""
    if "variant" in report and "summary" in report:                  # retention probe report
        variant = report.get("variant")
        score = report.get("summary", {}).get("internalisation_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return f"retention test, variant {variant}: internalisation score {score:.2f}"
        return f"retention test, variant {variant}"
    if "survival_ratio" in report or "survival_ratios" in report:    # discovery run_battery
        bid = report.get("battery_id", "?")
        line = _survival_line(report)
        return f"capability battery {bid}: host disconnected" + (f"; {line}" if line else "")
    if "invariant_ok" in report and "n_events" in report:            # consolidation audit
        return (f"consolidation audit: {report['n_events']} events, invariant "
                f"{'ok' if report['invariant_ok'] else 'VIOLATED'}")
    return f"test report filed ({len(report)} keys)"


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #

class DevelopmentalHistory:
    """One instance's biography: substrate, experiences, relationships, tests,
    dreams (stub), growth (stub), independence — every event named by NOMEN."""

    def __init__(self, instance_id: str, substrate: Dict, registry: EntityRegistry,
                 graph: LineageGraph, tracker: EvolutionTracker, birth_ts: float,
                 entries: Optional[List[Dict]] = None):
        self.instance_id = str(instance_id)
        self.substrate = dict(substrate)
        self.registry = registry
        self.graph = graph
        self.tracker = tracker
        self.birth_ts = float(birth_ts)
        self._entries: List[Dict] = [dict(e) for e in (entries or [])]

    # -- opening ---------------------------------------------------------------- #

    @classmethod
    def begin(cls, instance_id: str, substrate: Dict,
              clock: Optional[Callable[[], float]] = None,
              seed: Optional[int] = None, initial_name: str = "AION") -> "DevelopmentalHistory":
        """Open a biography: instance identity, starting-substrate record, birth
        timestamp, and the NOMEN self-entity (initial identity SUPPLIED —
        "AION" + instance id, provenance "initial", exactly the SelfIdentity
        convention). The birth itself is recorded as the first named state."""
        _check_jsonable(substrate, "substrate")
        clock = clock or time.time
        registry = EntityRegistry(seed=seed, clock=clock)
        ent = registry.register(
            f"{initial_name}-{instance_id}", "self", verification_status="observed",
            provenance="initial",
            reason=f"initial identity supplied: {initial_name} + instance id")
        # scalar substrate facts become observed properties — the seed material
        # future NOMEN names emerge from (names grow from what the instance IS).
        registry.observe(ent, {str(k): str(v) for k, v in substrate.items()
                               if isinstance(v, (str, int, float, bool))})
        graph = LineageGraph()
        tracker = EvolutionTracker(registry, graph, ent.entity_id, clock=clock)
        hist = cls(instance_id=instance_id, substrate=substrate, registry=registry,
                   graph=graph, tracker=tracker, birth_ts=0.0)
        hist._record(
            "birth", f"came into being on its starting substrate "
            f"({len(substrate)} keys recorded)",
            payload={"substrate": dict(substrate)},
            observed_properties={"instance_id": instance_id},
            tags={"substrate_keys": ",".join(sorted(str(k) for k in substrate))})
        hist.birth_ts = float(hist.entries[0]["ts"])   # birth ts == the birth event's ts
        return hist

    # -- THE SEAM: every event routes through NOMEN's record_transition ---------- #

    def _record(self, kind: str, description: str, payload: Dict,
                observed_properties: Optional[Dict[str, str]] = None,
                tags: Optional[Dict[str, str]] = None,
                new_name: Optional[str] = None) -> GraphNode:
        event = TransitionEvent(
            kind=kind, description=description,
            observed_properties={str(k): str(v) for k, v in (observed_properties or {}).items()},
            tags={str(k): str(v) for k, v in (tags or {}).items()})
        node = self.tracker.record_transition(event, new_name=new_name)
        self._entries.append({
            "seq": int(event.seq), "ts": float(event.ts), "kind": kind,
            "description": description, "node_id": node.node_id,
            "node_name": node.name, "tags": dict(event.tags),
            "payload": _jsonable(dict(payload)),
        })
        return node

    def record_transition(self, event: TransitionEvent, new_name: Optional[str] = None,
                          graph_kind: str = "advance",
                          provenance: str = "internal") -> GraphNode:
        """Escape hatch straight onto the NOMEN seam (full control — renames,
        branch/merge/reverse kinds) for lanes that need it. The event is filed
        into the ledger payload verbatim."""
        node = self.tracker.record_transition(event, new_name=new_name,
                                              graph_kind=graph_kind,
                                              provenance=provenance)
        self._entries.append({
            "seq": int(event.seq), "ts": float(event.ts), "kind": event.kind,
            "description": event.description, "node_id": node.node_id,
            "node_name": node.name, "payload": {"event": event.to_dict()},
        })
        return node

    # -- event recorders --------------------------------------------------------- #

    def record_experience(self, episode, description: Optional[str] = None,
                          new_name: Optional[str] = None) -> GraphNode:
        """File one experience from the EpisodicStore (an Episode, or its dict
        shape) with its provenance. The ledger FILES it — memory of the
        experience itself lives in the plastic state, never here."""
        d = episode.to_dict() if hasattr(episode, "to_dict") else dict(episode)
        teacher = str(d.get("teacher", "unknown"))
        channel = str(d.get("channel", "direct"))
        text = str(d.get("text", ""))
        if description is None:
            description = f"{d.get('kind', 'experience')} ({channel}, teacher {teacher}): {text}"
        return self._record(
            "experience", description,
            payload={"episode": _jsonable(d)},
            observed_properties={"teacher": teacher, "channel": channel},
            tags={"event_id": str(d.get("event_id", ""))},
            new_name=new_name)

    def record_relationship_snapshot(self, rows: List[Dict],
                                     description: Optional[str] = None,
                                     new_name: Optional[str] = None) -> GraphNode:
        """File a knowledge-inventory snapshot (RelationalMemory.inventory() rows):
        which relationships exist, in plastic / consolidated / full state, with
        their live strengths. Probes and meta stay out — this is the inventory
        headline, the strengths are in the plastic state itself."""
        proj = [{"key": r.get("key"), "pair_index": r.get("pair_index"),
                 "full": r.get("full"), "plastic": r.get("plastic"),
                 "consolidated": r.get("consolidated"),
                 "registered_at_seq": r.get("registered_at_seq"),
                 "superseded_by": r.get("superseded_by")} for r in rows]
        n_live = sum(1 for p in proj
                     if isinstance(p["full"], (int, float)) and p["full"] > 0)
        if description is None:
            description = (f"knowledge inventory: {len(proj)} relations indexed, "
                           f"{n_live} transmitting")
        return self._record(
            "relationships", description,
            payload={"n_relations": len(proj), "n_transmitting": n_live,
                     "rows": proj},
            observed_properties={"n_relations": str(len(proj))},
            new_name=new_name)

    def record_test_result(self, report: Dict, label: Optional[str] = None,
                           description: Optional[str] = None,
                           new_name: Optional[str] = None) -> GraphNode:
        """File one test report in a LANDED shape (retention probe report,
        knowledge_discovery run_battery report, consolidation audit — or any
        dict). The whole sanitised report is kept in the JSON record; the
        biography prints a headline, not a flood."""
        rep = _jsonable(dict(report))
        if description is None:
            description = _test_description(rep)
        if label:
            description = f"{label}: {description}"
        return self._record(
            "test", description,
            payload={"label": label, "report": rep},
            tags={"report_keys": ",".join(sorted(rep.keys()))},
            new_name=new_name)

    def record_dream(self, payload: Optional[Dict] = None,
                     description: Optional[str] = None,
                     new_name: Optional[str] = None) -> GraphNode:
        """STUB HOOK — dreaming is LOCKED until post-pilot (owner law). Accepts
        and files the dream event for the future engine; nothing is dreamed,
        nothing is imported, nothing is invoked."""
        if description is None:
            description = "dream event filed (stub — the dream engine is locked)"
        return self._record(
            "dream", description,
            payload={"dream": "stub-locked", "engine": None, "note": STUB_NOTE,
                     "event": _jsonable(dict(payload or {}))},
            tags={"stub": "locked"}, new_name=new_name)

    def record_growth(self, payload: Optional[Dict] = None,
                      description: Optional[str] = None,
                      new_name: Optional[str] = None) -> GraphNode:
        """STUB HOOK — growth events are accepted and filed for the future
        growth engine; nothing is computed, nothing is invoked. Passing
        new_name here is how a growth event becomes a NAMED STAGE (the name
        itself must come from NOMEN — propose_name/self_naming — never from
        this module)."""
        if description is None:
            description = "growth event filed (stub hook)"
        return self._record(
            "growth", description,
            payload={"growth": "stub-locked", "engine": None, "note": STUB_NOTE,
                     "event": _jsonable(dict(payload or {}))},
            tags={"stub": "locked"}, new_name=new_name)

    def record_independence(self, report: Dict, description: Optional[str] = None,
                            new_name: Optional[str] = None) -> GraphNode:
        """File a disconnect-experiment result (the knowledge_discovery
        run_battery shape): per-capability scores with and without the host,
        plus the capability-survival ratios."""
        rep = _jsonable(dict(report))
        if description is None:
            description = "independence test: host disconnected entirely"
            line = _survival_line(rep)
            if line:
                description += f"; {line}"
        return self._record(
            "independence", description,
            payload={"report": rep},
            tags={"report_keys": ",".join(sorted(rep.keys()))},
            new_name=new_name)

    # -- reading ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """The instance's CURRENT name (NOMEN's — it changes as it self-names)."""
        return self.tracker.self_entity.current_name

    @property
    def entries(self) -> List[Dict]:
        return list(self._entries)

    @property
    def nodes(self) -> List[GraphNode]:
        """Named states of this instance, in the order it first walked them."""
        return self.tracker.history()

    def biography(self) -> str:
        """The whole history as human-readable markdown. Stage headings are
        whatever names NOMEN produced — this renderer hardcodes no stage name."""
        ent = self.tracker.self_entity
        nodes = self.nodes
        entry_by_seq = {int(e["seq"]): e for e in self._entries}
        lines: List[str] = []
        lines.append(f"# {self.name} — a developmental history")
        lines.append("")
        lineage = " -> ".join(le.name for le in ent.lineage)
        if len(ent.lineage) > 1:
            lines.append(f"**Called:** {ent.current_name} (lineage: {lineage})")
        else:
            lines.append(f"**Called:** {ent.current_name}")
        lines.append(f"**Instance:** {self.instance_id}")
        lines.append(f"**Born:** ts {self.birth_ts:.1f}")
        lines.append("")
        lines.append("## Starting substrate")
        for k in sorted(self.substrate):
            lines.append(f"- {k}: {_fmt_value(self.substrate[k])}")
        lines.append("")
        lines.append("## The walk")
        for node in nodes:
            lines.append("")
            lines.append(f"### {node.name}")
            lines.append(f"*named state {node.node_id} · tick {node.tick} · "
                         f"{len(node.events)} event(s) recorded against it*")
            for ev in node.events:
                e = entry_by_seq.get(int(ev.get("seq", -1)), {})
                lines.append("")
                lines.append(f"{int(ev.get('seq', 0))}. **{ev.get('kind', '?')}** — "
                             f"{ev.get('description', '')} (ts {float(ev.get('ts', 0.0)):.1f})")
                payload = e.get("payload") or {}
                for pk in sorted(payload):
                    lines.append(f"   - {pk}: {_fmt_value(payload[pk])}")
            for edge in self.graph.edges():
                if edge.entity_id != ent.entity_id or edge.src != node.node_id:
                    continue
                dst = self.graph.node(edge.dst)
                reason = f" — {edge.reason}" if edge.reason else ""
                lines.append("")
                lines.append(f"--> *{edge.label}* ({edge.kind}): "
                             f"{node.name} -> {dst.name}{reason}")
        lines.append("")
        return "\n".join(lines)

    def biography_json(self) -> Dict:
        """The machine view: the whole biography as one JSON-safe dict."""
        ent = self.tracker.self_entity
        return {
            "version": PAYLOAD_VERSION,
            "instance_id": self.instance_id,
            "current_name": self.name,
            "birth_ts": self.birth_ts,
            "substrate": dict(self.substrate),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.graph.edges()
                      if e.entity_id == ent.entity_id],
            "entries": [dict(e) for e in self._entries],
        }

    # -- persistence (exact; plain JSON, NomenStore conventions) ------------------- #

    def to_dict(self) -> Dict:
        return {
            "version": PAYLOAD_VERSION,
            "instance_id": self.instance_id,
            "birth_ts": self.birth_ts,
            "substrate": dict(self.substrate),
            "entries": [dict(e) for e in self._entries],
            "nomen": NomenStore(self.registry, self.graph, self.tracker).to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict,
                  clock: Optional[Callable[[], float]] = None) -> "DevelopmentalHistory":
        store = NomenStore.from_dict(d["nomen"], clock=clock)
        if store.tracker is None:
            raise ValueError("devhistory payload has no evolution tracker")
        return cls(instance_id=d["instance_id"], substrate=dict(d.get("substrate", {})),
                   registry=store.registry, graph=store.graph, tracker=store.tracker,
                   birth_ts=float(d.get("birth_ts", 0.0)),
                   entries=list(d.get("entries", [])))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str,
             clock: Optional[Callable[[], float]] = None) -> "DevelopmentalHistory":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), clock=clock)

    def equals(self, other: "DevelopmentalHistory") -> bool:
        """Exact round-trip check: identical payloads (the test asserts on this)."""
        return self.to_dict() == other.to_dict()
