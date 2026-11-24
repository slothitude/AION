"""NOMEN evolution tracker: the AION INSTANCE's own developmental transitions.

OWNER DESIGN LAW (NOMEN, 2026-11-25): evolution is a graph, not a ladder — the instance
records its OWN transitions as they happen, using exactly the same lineage graph and
registry as every other named thing. There is no stage list here: each TransitionEvent
carries its own free-form kind (e.g. "observed_world", "renamed", "consolidated",
"discovered"), and the instance may branch, merge, reverse or run transitions
simultaneously just like any entity's graph.

THE INTEGRATION SEAM (planned task — the developmental-history ledger, task 21):

    record_transition(event: TransitionEvent, new_name=None, ...) -> GraphNode

is the seam the devhistory ledger will call: hand it a TransitionEvent, get back the
NAMED GRAPH NODE the event is attached to (node_id + name + tick). The ledger itself is
deliberately NOT built in this module — this is only the naming layer for it. When a
transition is a rename, the instance's name changes through the registry (lineage
appended, identity and verification_status untouched) and a graph edge is recorded;
when it is not a rename, the event attaches to the instance's current named state.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from aion.nomen.entity_registry import Entity, EntityRegistry
from aion.nomen.lineage_graph import GraphNode, LineageGraph
from aion.nomen.name_generator import NameGenerator, NameProposal


@dataclass
class TransitionEvent:
    """One developmental transition of the AION instance. The kind is free-form by
    design (a ladder would be a fixed sequence; this is a recorded event)."""

    kind: str                          # e.g. "observed_world", "renamed", "consolidated"
    description: str
    observed_properties: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    seq: int = 0                       # assigned by the tracker
    ts: float = 0.0                    # assigned by the tracker

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "description": self.description,
                "observed_properties": dict(self.observed_properties),
                "tags": dict(self.tags), "seq": int(self.seq), "ts": float(self.ts)}

    @classmethod
    def from_dict(cls, d: Dict) -> "TransitionEvent":
        return cls(kind=d["kind"], description=d.get("description", ""),
                   observed_properties=dict(d.get("observed_properties", {})),
                   tags=dict(d.get("tags", {})), seq=int(d.get("seq", 0)),
                   ts=float(d.get("ts", 0.0)))


class EvolutionTracker:
    """Naming layer for the instance's developmental history (task-21 ledger seam)."""

    def __init__(self, registry: EntityRegistry, graph: LineageGraph,
                 self_entity_id: str, name_generator: Optional[NameGenerator] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.registry = registry
        self.graph = graph
        self.self_entity_id = self_entity_id
        self.generator = name_generator or NameGenerator()
        self._clock = clock or time.time
        self._seq = 0
        self.events: List[Dict] = []       # {event payload..., node_id}

    # -- access ---------------------------------------------------------------- #

    @property
    def self_entity(self) -> Entity:
        return self.registry.get(self.self_entity_id)

    @property
    def current_node(self) -> GraphNode:
        ent = self.self_entity
        return self.graph.ensure_node(ent.entity_id, ent.current_name)

    # -- THE SEAM --------------------------------------------------------------- #

    def record_transition(self, event: TransitionEvent, new_name: Optional[str] = None,
                          reason: str = "", graph_kind: str = "advance",
                          provenance: str = "internal") -> GraphNode:
        """Record one developmental transition; return the named node it attached to.

        The devhistory ledger (task 21) should call exactly this and store the returned
        node reference — that is the whole integration contract.

        new_name given  -> registry rename (lineage appended; id + verification_status
                           untouched) + graph transition edge, event attached to the NEW
                           named state.
        new_name absent -> non-renaming transition; event attaches to the instance's
                           CURRENT named state.
        """
        self._seq += 1
        event.seq = self._seq
        event.ts = float(self._clock())
        ent = self.self_entity
        prev_name = ent.current_name

        if new_name is not None:
            self.registry.rename(ent, new_name, provenance=provenance,
                                 reason=reason or f"developmental transition: {event.kind}",
                                 basis={"event": event.to_dict()})
            node = self.graph.record_transition(ent.entity_id, prev_name, new_name,
                                                reason=reason or event.description,
                                                kind=graph_kind)
        else:
            node = self.graph.ensure_node(ent.entity_id, prev_name)

        payload = event.to_dict()
        payload["node_id"] = node.node_id
        self.events.append(payload)
        node.events.append(dict(payload))  # the named state accumulates its own history
        return node

    def propose_name(self, taken: Optional[List[str]] = None,
                     include_teacher: bool = True) -> List[NameProposal]:
        """Candidate names for the instance's NEXT rename, grown from its own observed
        properties (teacher hook optional). Pure proposal — nothing is renamed."""
        ent = self.self_entity
        taken = set(taken or [])
        taken |= {e.current_name for e in self.registry.all_entities()}
        return self.generator.propose(ent.entity_id, ent.entity_kind,
                                      ent.observed_properties.keys(),
                                      taken=taken, include_teacher=include_teacher)

    # -- reading ----------------------------------------------------------------- #

    def history(self) -> List[GraphNode]:
        return self.graph.history(self.self_entity_id)

    # -- persistence (exact; file I/O lives in name_memory) ----------------------- #

    def to_dict(self) -> Dict:
        return {"self_entity_id": self.self_entity_id, "seq": self._seq,
                "events": [dict(e) for e in self.events]}

    @classmethod
    def from_dict(cls, d: Dict, registry: EntityRegistry, graph: LineageGraph,
                  name_generator: Optional[NameGenerator] = None,
                  clock: Optional[Callable[[], float]] = None) -> "EvolutionTracker":
        tracker = cls(registry, graph, d["self_entity_id"], name_generator=name_generator,
                      clock=clock)
        tracker._seq = int(d.get("seq", 0))
        tracker.events = [dict(e) for e in d.get("events", [])]
        return tracker

    def equals(self, other: "EvolutionTracker") -> bool:
        return self.to_dict() == other.to_dict()
