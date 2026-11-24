"""NOMEN — AION's evolving naming / identity / lineage system.

Owner design laws (2026-11-25), enforced across every module here:
  - "Everything has a name. Nothing has to keep the same one."
  - NOMEN names EVERYTHING AION encounters: models, Pith brains, relationships,
    memories, dreams, discoveries, neural modules, capabilities, itself, worlds.
    Names EMERGE FROM OBSERVATION — never from a fixed list we supply.
  - Evolution is a GRAPH, not a ladder: transitions branch, merge, reverse and run
    simultaneously; nothing is a hardcoded stage sequence.
  - Three naming properties per entity: immutable Identity (uuid), Current name,
    Lineage (previous names + evolution history). Renaming never loses history.
  - Self-naming: the instance eventually chooses its own individual name
    (initial identity supplied; teacher-assisted candidates early; internal later).
  - SAFEGUARD (hard law): naming never changes an entity's underlying identity and
    never automatically turns an imagined concept into a verified fact — every named
    entity carries a verification_status independent of its name, and naming
    operations cannot mutate it.

Modules:
  entity_registry   — Entity records: id + current name + lineage + verification_status
  name_generator    — observation-driven morpheme composition + optional teacher hook
  identity_resolver — resolve a reference (id / current name / past name) canonically
  lineage_graph     — the evolution graph (branch / merge / reverse / simultaneous)
  evolution_tracker — the AION instance's own developmental transitions (task-21 seam)
  name_memory       — exact JSON persistence of registry + graphs across restarts
  collision_detector— unique-name enforcement; both lineages survive resolution
  self_naming       — the instance's own identity and its self-naming events
"""

from aion.nomen.entity_registry import (
    ENTITY_KINDS,
    NAME_PROVENANCES,
    VERIFICATION_STATUSES,
    Entity,
    EntityRegistry,
    LineageEntry,
)
from aion.nomen.name_generator import (
    GeneratedName,
    NameGenerator,
    NameProposal,
    TeacherNameAssist,
    generate_name,
)
from aion.nomen.identity_resolver import IdentityResolver, Resolution
from aion.nomen.lineage_graph import GraphEdge, GraphNode, LineageGraph
from aion.nomen.evolution_tracker import EvolutionTracker, TransitionEvent
from aion.nomen.collision_detector import Collision, CollisionDetector, ResolvedCollision
from aion.nomen.self_naming import SelfIdentity, SelfNamingEvent
from aion.nomen.name_memory import NomenStore

__all__ = [
    "ENTITY_KINDS", "NAME_PROVENANCES", "VERIFICATION_STATUSES",
    "Entity", "EntityRegistry", "LineageEntry",
    "GeneratedName", "NameGenerator", "NameProposal", "TeacherNameAssist", "generate_name",
    "IdentityResolver", "Resolution",
    "GraphEdge", "GraphNode", "LineageGraph",
    "EvolutionTracker", "TransitionEvent",
    "Collision", "CollisionDetector", "ResolvedCollision",
    "SelfIdentity", "SelfNamingEvent",
    "NomenStore",
]
