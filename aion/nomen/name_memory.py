"""NOMEN persistence: the whole naming world as one exact JSON snapshot.

OWNER DESIGN LAW (NOMEN, 2026-11-25): names, lineages and the evolution graph must
survive restarts — a restart must never forget what AION called anything, or what it
used to be called. This module owns the FILE I/O for registry + lineage graph + the
instance's tracker events + the collision resolution log. Round-trip must be EXACT:
`to_dict() == to_dict()` across save/load, byte-for-byte on the sorted JSON dump.

Plain JSON only (no tensors live in the naming layer), matching the episodic store's
conventions. Clocks are injected at load time by the caller; they are not persisted.
"""

import json
import os
from typing import Optional

from aion.nomen.entity_registry import EntityRegistry
from aion.nomen.lineage_graph import LineageGraph
from aion.nomen.evolution_tracker import EvolutionTracker
from aion.nomen.collision_detector import CollisionDetector
from aion.nomen.name_generator import NameGenerator

PAYLOAD_VERSION = 1


class NomenStore:
    """One JSON snapshot of the whole naming world, and its exact restoration."""

    def __init__(self, registry: EntityRegistry, graph: LineageGraph,
                 tracker: Optional[EvolutionTracker] = None,
                 collisions: Optional[CollisionDetector] = None):
        self.registry = registry
        self.graph = graph
        self.tracker = tracker
        self.collisions = collisions

    # -- payload ---------------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "version": PAYLOAD_VERSION,
            "registry": self.registry.to_dict(),
            "graph": self.graph.to_dict(),
            "tracker": self.tracker.to_dict() if self.tracker is not None else None,
            "collisions": self.collisions.to_dict() if self.collisions is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict, generator: Optional[NameGenerator] = None,
                  clock=None) -> "NomenStore":
        """Reconstruct everything, rebinding cross-references (tracker -> registry and
        graph; detector -> registry). Pass the same clock/seed policy the original used
        when exact reproducibility across processes matters."""
        registry = EntityRegistry.from_dict(d["registry"], clock=clock)
        graph = LineageGraph.from_dict(d["graph"])
        tracker = None
        if d.get("tracker") is not None:
            tracker = EvolutionTracker.from_dict(d["tracker"], registry, graph,
                                                 name_generator=generator, clock=clock)
        collisions = None
        if d.get("collisions") is not None:
            collisions = CollisionDetector.from_dict(d["collisions"], registry,
                                                     name_generator=generator)
        return cls(registry, graph, tracker=tracker, collisions=collisions)

    # -- file I/O ----------------------------------------------------------------- #

    def save(self, path: str) -> dict:
        payload = self.to_dict()
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        return payload

    @classmethod
    def load(cls, path: str, generator: Optional[NameGenerator] = None,
             clock=None) -> "NomenStore":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), generator=generator, clock=clock)

    # -- exactness ------------------------------------------------------------------ #

    def equals(self, other: "NomenStore") -> bool:
        """Exact round-trip check: identical payloads. The round-trip test asserts on
        this — a restart that loses one lineage entry fails here."""
        return self.to_dict() == other.to_dict()

    def dump_canonical(self) -> str:
        """The canonical sorted JSON text (byte-stable for diffing two snapshots)."""
        return json.dumps(self.to_dict(), indent=1, sort_keys=True)
