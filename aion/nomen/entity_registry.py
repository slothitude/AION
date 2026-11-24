"""NOMEN entity registry — everything has a name; nothing has to keep the same one.

OWNER DESIGN LAWS (NOMEN, 2026-11-25), enforced here:
  - Every entity carries the THREE naming properties: an immutable identity
    (entity_id), a current name, and a lineage (every previous name plus how and why
    it got there). Renaming appends to the lineage; history is never lost.
  - SAFEGUARD (hard law): naming must never change an entity's underlying identity and
    must never automatically turn an imagined concept into a verified fact. Every
    entity therefore carries a verification_status that is INDEPENDENT of its name.
    rename() physically cannot touch entity_id or verification_status — it only swaps
    current_name and appends one lineage entry. The ONLY sanctioned status change is
    reclassify(), which demands an explicit evidence note and is never reachable from
    any naming path.

The registry names EVERYTHING AION encounters: models, Pith brains, relationships,
memories, dreams, discoveries, neural modules, capabilities, the AION instance itself,
and worlds. This module only RECORDS names — it never supplies them. Names emerge from
observation through aion/nomen/name_generator.py and are proposed to the registry.

The registry is a record-keeper, not a name police: two entities may briefly claim one
current name (observation can genuinely coincide). Unique-name ENFORCEMENT lives in
collision_detector.py, which detects such claims and resolves them while preserving
both lineages.

Determinism: with `seed` set, entity ids are uuid5-derived (same seed -> same ids) and
the clock is injectable, so a whole registry is reproducible in tests. Persistence
round-trips exactly through name_memory.py.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

# observation -> naming independence: a name never carries truth about its referent.
VERIFICATION_STATUSES = ("observed", "imagined", "verified")

ENTITY_KINDS = (
    "model", "brain", "relationship", "memory", "dream", "discovery",
    "module", "capability", "self", "world",
)

# who/what supplied a lineage name: the initial identity, names coined from observed
# properties, names proposed by a teacher hook, internally coined names, a self-naming
# decision (teacher-assisted = chose from teacher candidates, internal = chose from
# own candidates), and renames forced by collision resolution.
NAME_PROVENANCES = (
    "initial", "observation-derived", "teacher-proposed", "internal",
    "teacher-assisted", "collision-resolution",
)

EntityRef = Union[str, "Entity"]


@dataclass
class LineageEntry:
    """One naming event in an entity's lineage. Appended on every rename."""

    seq: int                       # 1-based position in this entity's lineage
    name: str
    provenance: str                # one of NAME_PROVENANCES
    reason: str
    ts: float
    basis: Dict[str, object] = field(default_factory=dict)
    # basis carries WHERE the name came from: for observation-derived names the
    # property->morpheme map, for teacher/self names the candidate list and the
    # deciding provenance, for collision resolutions the collision name + keeper.

    def to_dict(self) -> Dict:
        return {"seq": int(self.seq), "name": self.name, "provenance": self.provenance,
                "reason": self.reason, "ts": float(self.ts), "basis": dict(self.basis)}

    @classmethod
    def from_dict(cls, d: Dict) -> "LineageEntry":
        return cls(seq=int(d["seq"]), name=d["name"], provenance=d["provenance"],
                   reason=d.get("reason", ""), ts=float(d["ts"]),
                   basis=dict(d.get("basis", {})))


@dataclass
class Entity:
    """A named thing. entity_id is permanent; verification_status is name-independent."""

    entity_id: str
    entity_kind: str               # one of ENTITY_KINDS
    current_name: str
    lineage: List[LineageEntry]    # [0] = initial naming; last = current name
    verification_status: str       # observed | imagined | verified — NEVER set by naming
    observed_properties: Dict[str, str] = field(default_factory=dict)
    # observed_properties: the entity's recorded features/relations/behaviour — the
    # seed material names emerge from (see name_generator.py), NOT part of the name.
    verification_log: List[Dict] = field(default_factory=list)
    # verification_log: every verification_status change, with its evidence note.
    # Naming operations append NOTHING here — the log only grows through reclassify().

    @property
    def past_names(self) -> List[str]:
        return [e.name for e in self.lineage[:-1]]

    @property
    def history(self) -> List[str]:
        return [e.name for e in self.lineage]

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "entity_kind": self.entity_kind,
            "current_name": self.current_name,
            "lineage": [e.to_dict() for e in self.lineage],
            "verification_status": self.verification_status,
            "observed_properties": dict(self.observed_properties),
            "verification_log": [dict(v) for v in self.verification_log],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Entity":
        return cls(
            entity_id=d["entity_id"], entity_kind=d["entity_kind"],
            current_name=d["current_name"],
            lineage=[LineageEntry.from_dict(e) for e in d["lineage"]],
            verification_status=d["verification_status"],
            observed_properties=dict(d.get("observed_properties", {})),
            verification_log=[dict(v) for v in d.get("verification_log", [])],
        )


def _entity_id_of(ref: EntityRef) -> str:
    return ref.entity_id if isinstance(ref, Entity) else str(ref)


class EntityRegistry:
    """Record-keeper for named entities: register / look up / rename.

    rename() appends to the lineage and swaps current_name. It CANNOT touch
    entity_id or verification_status — the safeguard is structural, not advisory.
    """

    def __init__(self, seed: Optional[int] = None, clock: Optional[Callable[[], float]] = None):
        self._clock = clock or time.time
        self._seed = seed
        self._namespace = (uuid.uuid5(uuid.NAMESPACE_URL, f"nomen:{int(seed)}")
                           if seed is not None else uuid.NAMESPACE_URL)
        self._entities: Dict[str, Entity] = {}
        self._seq = 0                  # registry-wide counter; feeds seeded id derivation

    # -- construction --------------------------------------------------------- #

    def new_id(self, entity_kind: str) -> str:
        """Immutable identity. Seeded registries derive ids from (namespace, kind, seq);
        unseeded registries fall back to uuid4."""
        self._seq += 1
        if self._seed is not None:
            return str(uuid.uuid5(self._namespace, f"{entity_kind}:{self._seq}"))
        return str(uuid.uuid4())

    def register(self, name: str, entity_kind: str,
                 verification_status: str = "observed",
                 observed_properties: Optional[Dict[str, str]] = None,
                 provenance: str = "initial", reason: str = "") -> Entity:
        if entity_kind not in ENTITY_KINDS:
            raise ValueError(f"entity_kind must be one of {ENTITY_KINDS}, got {entity_kind!r}")
        if verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"verification_status must be one of {VERIFICATION_STATUSES}, "
                             f"got {verification_status!r}")
        if provenance not in NAME_PROVENANCES:
            raise ValueError(f"provenance must be one of {NAME_PROVENANCES}, got {provenance!r}")
        name = self._check_name(name)
        eid = self.new_id(entity_kind)
        ent = Entity(
            entity_id=eid, entity_kind=entity_kind, current_name=name,
            lineage=[LineageEntry(seq=1, name=name, provenance=provenance, reason=reason,
                                  ts=float(self._clock()))],
            verification_status=verification_status,
            observed_properties=dict(observed_properties or {}),
        )
        self._entities[eid] = ent
        return ent

    # -- lookup ---------------------------------------------------------------- #

    def by_id(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(str(entity_id))

    def get(self, ref: EntityRef) -> Entity:
        ent = self.by_id(_entity_id_of(ref))
        if ent is None:
            raise KeyError(f"no entity with id {_entity_id_of(ref)!r}")
        return ent

    def by_name(self, name: str) -> List[Entity]:
        """Every entity whose CURRENT name is `name` (collision claims live here)."""
        return [e for e in self._entities.values() if e.current_name == name]

    def by_past_name(self, name: str) -> List[Entity]:
        """Every entity that ONCE carried `name` but does not now (history survives)."""
        return [e for e in self._entities.values()
                if e.current_name != name and name in e.past_names]

    def all_entities(self) -> List[Entity]:
        return list(self._entities.values())

    def __len__(self) -> int:
        return len(self._entities)

    # -- naming (the only mutations a name can cause) -------------------------- #

    def rename(self, ref: EntityRef, new_name: str, provenance: str,
               reason: str = "", basis: Optional[Dict[str, object]] = None) -> Entity:
        """Give an entity a new current name. Appends one lineage entry; the id and the
        verification_status are structurally untouched (safeguard law)."""
        if provenance not in NAME_PROVENANCES:
            raise ValueError(f"provenance must be one of {NAME_PROVENANCES}, got {provenance!r}")
        ent = self.get(ref)
        new_name = self._check_name(new_name)
        if new_name == ent.current_name:
            raise ValueError(f"{ent.entity_id} already carries the name {new_name!r}; "
                             "rename to a different name")
        ent.lineage.append(LineageEntry(
            seq=len(ent.lineage) + 1, name=new_name, provenance=provenance,
            reason=reason, ts=float(self._clock()), basis=dict(basis or {})))
        ent.current_name = new_name
        return ent

    def observe(self, ref: EntityRef, properties: Dict[str, str]) -> Entity:
        """Record newly observed properties (features/relations/behaviour). This is
        observation, not naming: it changes the material future names emerge FROM,
        never any name and never verification_status."""
        ent = self.get(ref)
        ent.observed_properties.update({str(k): str(v) for k, v in properties.items()})
        return ent

    # -- verification (the ONLY path that may touch verification_status) -------- #

    def reclassify(self, ref: EntityRef, to_status: str, evidence_note: str) -> Entity:
        """Explicitly move verification_status with a required evidence note.

        This is deliberately separate from every naming operation: an imagined concept
        becomes verified ONLY through a recorded evidence note here — never as a side
        effect of being named, renamed, resolved, or collision-resolved.
        """
        if to_status not in VERIFICATION_STATUSES:
            raise ValueError(f"verification_status must be one of {VERIFICATION_STATUSES}, "
                             f"got {to_status!r}")
        if not evidence_note or not str(evidence_note).strip():
            raise ValueError("reclassify demands a non-empty evidence_note (safeguard law: "
                             "an imagined concept is never silently promoted)")
        ent = self.get(ref)
        if to_status == ent.verification_status:
            raise ValueError(f"{ent.entity_id} is already {to_status!r}")
        ent.verification_log.append({
            "seq": len(ent.verification_log) + 1,
            "from": ent.verification_status, "to": to_status,
            "evidence_note": str(evidence_note), "ts": float(self._clock()),
        })
        ent.verification_status = to_status
        return ent

    # -- helpers ---------------------------------------------------------------- #

    @staticmethod
    def _check_name(name: str) -> str:
        name = str(name).strip()
        if not name:
            raise ValueError("names cannot be empty — everything has a name")
        return name

    # -- persistence (exact; file I/O lives in name_memory) ---------------------- #

    def to_dict(self) -> Dict:
        return {
            "seed": self._seed,
            "seq": self._seq,
            "entities": [e.to_dict() for e in
                         sorted(self._entities.values(), key=lambda e: e.lineage[0].ts)],
        }

    @classmethod
    def from_dict(cls, d: Dict, clock: Optional[Callable[[], float]] = None) -> "EntityRegistry":
        reg = cls(seed=d.get("seed"), clock=clock)
        for ed in d.get("entities", []):
            ent = Entity.from_dict(ed)
            reg._entities[ent.entity_id] = ent
        reg._seq = int(d.get("seq", 0))
        return reg

    def equals(self, other: "EntityRegistry") -> bool:
        return self.to_dict() == other.to_dict()
