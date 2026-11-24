"""NOMEN collision detector: unique-name enforcement that preserves both lineages.

OWNER DESIGN LAW (NOMEN, 2026-11-25): names are AION's interpretation of its world, so
two entities CAN briefly claim one current name (observation genuinely coincides, and
the registry is a record-keeper, not a name police). Enforcement lives here:

  detect()   — every current name claimed by more than one entity.
  resolve()  — deterministically KEEPS one claimant (earliest lineage entry = longest
               settled claim) and RENAMES the other(s), so BOTH entities and BOTH
               full lineages survive. Neither entity is deleted, merged away, or
               un-named; nothing touches any verification_status (naming cannot).

Two resolution strategies, both lineage-preserving:
  lineage-qualifier — the renamed entity's own first lineage name becomes a qualifier
                      suffix (its history literally disambiguates it), falling back to
                      a counter when the qualified form is itself taken.
  new-coinage       — a deterministic fresh coinage grown from the entity's observed
                      properties (name_generator.coined_name), salt-varied until free.

Strategy "auto" prefers the lineage qualifier (history is the honest disambiguator)
and falls back to coinage when qualification cannot produce a free name.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aion.nomen.entity_registry import Entity, EntityRegistry
from aion.nomen.name_generator import NameGenerator, coined_name

STRATEGIES = ("auto", "lineage-qualifier", "new-coinage")


@dataclass
class Collision:
    """One name claimed by several entities right now."""

    name: str
    entity_ids: List[str]
    detected_seq: int = 0

    def to_dict(self) -> Dict:
        return {"name": self.name, "entity_ids": list(self.entity_ids),
                "detected_seq": int(self.detected_seq)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Collision":
        return cls(name=d["name"], entity_ids=list(d["entity_ids"]),
                   detected_seq=int(d.get("detected_seq", 0)))


@dataclass
class ResolvedCollision:
    """What the detector did about a collision. Both entities survive with intact,
    lengthened lineages; the renamed entries carry provenance 'collision-resolution'."""

    name: str
    keeper_entity_id: str
    strategy: str
    renames: List[Dict] = field(default_factory=list)   # {entity_id, old, new, strategy}

    def to_dict(self) -> Dict:
        return {"name": self.name, "keeper_entity_id": self.keeper_entity_id,
                "strategy": self.strategy, "renames": [dict(r) for r in self.renames]}

    @classmethod
    def from_dict(cls, d: Dict) -> "ResolvedCollision":
        return cls(name=d["name"], keeper_entity_id=d["keeper_entity_id"],
                   strategy=d["strategy"], renames=[dict(r) for r in d.get("renames", [])])


class CollisionDetector:
    """Detects and resolves current-name collisions over an EntityRegistry."""

    def __init__(self, registry: EntityRegistry,
                 name_generator: Optional[NameGenerator] = None):
        self.registry = registry
        self.generator = name_generator or NameGenerator()
        self.log: List[Dict] = []          # ResolvedCollision dicts, in resolution order
        self._detect_seq = 0

    # -- detection ---------------------------------------------------------------- #

    def detect(self) -> List[Collision]:
        """Every current name claimed by >1 entity, in deterministic name order."""
        claims: Dict[str, List[Entity]] = {}
        for ent in self.registry.all_entities():
            claims.setdefault(ent.current_name, []).append(ent)
        out = []
        for name in sorted(n for n, es in claims.items() if len(es) > 1):
            self._detect_seq += 1
            out.append(Collision(name=name,
                                 entity_ids=[e.entity_id for e in claims[name]],
                                 detected_seq=self._detect_seq))
        return out

    def collisions_including_past(self) -> List[Dict]:
        """Shadowing report (no enforcement): a PAST name of one entity is another's
        CURRENT name. Renames make this normal; the resolver reports ambiguity for it,
        so this is informational only."""
        out = []
        for ent in self.registry.all_entities():
            for past in ent.past_names:
                for other in self.registry.by_name(past):
                    if other.entity_id != ent.entity_id:
                        out.append({"past_name": past, "of_entity_id": ent.entity_id,
                                    "now_claimed_by": other.entity_id})
        return out

    # -- resolution ----------------------------------------------------------------- #

    def resolve_all(self, strategy: str = "auto") -> List[ResolvedCollision]:
        """Resolve every live collision; each rename appends to the renamed entity's
        lineage (nothing deleted). After this, detect() returns empty."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        resolved = [self.resolve(c.name, strategy=strategy) for c in self.detect()]
        return resolved

    def resolve(self, name: str, strategy: str = "auto") -> ResolvedCollision:
        """Resolve one collision deterministically: the earliest claimant keeps the
        name, the rest are renamed with the chosen lineage-preserving strategy."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        claimants = self.registry.by_name(name)
        if len(claimants) < 2:
            raise ValueError(f"{name!r} is not a live collision "
                             f"({len(claimants)} current claimant(s))")
        # keeper = earliest first lineage entry (longest-settled claim); ties broken by
        # entity id so the decision is deterministic.
        claimants.sort(key=lambda e: (e.lineage[0].ts, e.lineage[0].seq, e.entity_id))
        keeper, renamed_targets = claimants[0], claimants[1:]

        resolution = ResolvedCollision(name=name, keeper_entity_id=keeper.entity_id,
                                       strategy=strategy)
        taken = {e.current_name for e in self.registry.all_entities()}
        applied: List[str] = []
        for target in renamed_targets:
            new_name, used = self._next_name(target, name, strategy, taken)
            self.registry.rename(
                target, new_name, provenance="collision-resolution",
                reason=f"resolved collision on {name!r} "
                       f"(keeper {keeper.entity_id}, strategy {used})",
                basis={"collision_name": name, "keeper_entity_id": keeper.entity_id,
                       "strategy": used})
            taken.add(new_name)
            applied.append(used)
            resolution.renames.append({"entity_id": target.entity_id,
                                       "old": name, "new": new_name, "strategy": used})
        # report the strategy actually APPLIED: "auto" resolves to the concrete strategy
        # each rename used (uniform -> that strategy; mixed -> "mixed").
        if applied:
            resolution.strategy = applied[0] if len(set(applied)) == 1 else "mixed"
        self.log.append(resolution.to_dict())
        return resolution

    # -- strategies ------------------------------------------------------------------- #

    def _next_name(self, target: Entity, collision_name: str,
                   strategy: str, taken: set) -> "tuple[str, str]":
        """Returns (free name, strategy actually applied)."""
        if strategy in ("auto", "lineage-qualifier"):
            candidate = self._qualified(target, collision_name, taken)
            if candidate is not None:
                return candidate, "lineage-qualifier"
            if strategy == "lineage-qualifier":
                raise ValueError(f"lineage-qualifier could not free a name for "
                                 f"{target.entity_id}")
        return self._coined(target, taken), "new-coinage"

    def _qualified(self, target: Entity, collision_name: str, taken: set) -> Optional[str]:
        """Strategy 1: the entity's own FIRST lineage name qualifies it — its history
        disambiguates it. 'kolu' claimed twice and one was born 'mol' -> 'kolu-mol'."""
        root = "".join(ch for ch in target.lineage[0].name.lower() if ch.isalnum())[:12]
        if not root:
            return None
        candidate = f"{collision_name}-{root}"
        counter = 0
        while candidate in taken:
            counter += 1
            candidate = f"{collision_name}-{root}-{counter}"
            if counter > 64:
                return None
        return candidate

    def _coined(self, target: Entity, taken: set) -> str:
        """Strategy 2: deterministic fresh coinage from the entity's observed
        properties, salt-varied until free. Both lineages survive untouched."""
        for salt in range(64):
            gen = coined_name(target.entity_id, target.entity_kind,
                              target.observed_properties.keys(), taken=taken, salt=salt)
            if gen.name not in taken:
                return gen.name
        raise ValueError(f"could not coin a free name for {target.entity_id}")

    # -- persistence (exact; file I/O lives in name_memory) --------------------------- #

    def to_dict(self) -> Dict:
        return {"detect_seq": self._detect_seq, "log": [dict(l) for l in self.log]}

    @classmethod
    def from_dict(cls, d: Dict, registry: EntityRegistry,
                  name_generator: Optional[NameGenerator] = None) -> "CollisionDetector":
        det = cls(registry, name_generator=name_generator)
        det._detect_seq = int(d.get("detect_seq", 0))
        det.log = [dict(l) for l in d.get("log", [])]
        return det

    def equals(self, other: "CollisionDetector") -> bool:
        return self.to_dict() == other.to_dict()
