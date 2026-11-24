"""NOMEN identity resolution: any reference -> the canonical entity, with ambiguity reported.

OWNER DESIGN LAW (NOMEN, 2026-11-25): renaming never loses history, so a reference may
be an entity's immutable ID, its CURRENT name, or any PAST name. The resolver resolves
all three and reports ambiguity instead of guessing — AION's interpretation of its world
must stay honest when two entities claim the same word.

Resolution order (documented):
  1. exact entity ID match            (identity outranks any name)
  2. current-name claims              (unique -> resolved; several -> ambiguous,
                                       which is a collision claim for the detector)
  3. past-name matches                (history survives renames; several -> ambiguous)
  4. nothing                          -> unknown, never guessed

Resolving a reference NEVER mutates anything: no verification_status change, no
lineage append, no graph edge. Resolution is pure reading.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from aion.nomen.entity_registry import Entity, EntityRegistry


@dataclass
class Resolution:
    """The honest answer to 'what does this reference point at?'."""

    reference: str
    status: str                     # "unique" | "ambiguous" | "unknown"
    matched_as: str                 # "id" | "current-name" | "past-name" | "none"
    entities: List[Entity] = field(default_factory=list)
    note: str = ""

    @property
    def entity(self) -> Optional[Entity]:
        return self.entities[0] if self.status == "unique" and self.entities else None

    @property
    def entity_id(self) -> Optional[str]:
        return self.entity.entity_id if self.entity else None

    def to_dict(self) -> dict:
        return {"reference": self.reference, "status": self.status,
                "matched_as": self.matched_as,
                "entity_ids": [e.entity_id for e in self.entities], "note": self.note}


class IdentityResolver:
    """Read-only resolver over an EntityRegistry."""

    def __init__(self, registry: EntityRegistry):
        self.registry = registry

    def resolve(self, reference: str, entity_kind: Optional[str] = None) -> Resolution:
        """Resolve `reference` (ID, current name, or past name).

        entity_kind, when given, may narrow an ambiguous match; it is only applied
        when narrowing is exact (several -> still ambiguous, reported honestly).
        """
        ref = str(reference)

        # 1. identity outranks any name
        by_id = self.registry.by_id(ref)
        if by_id is not None:
            if entity_kind is not None and by_id.entity_kind != entity_kind:
                return Resolution(ref, "unknown", "none", [], "id found but kind does not match")
            return Resolution(ref, "unique", "id", [by_id], "resolved by immutable id")

        # 2. current names
        current = self.registry.by_name(ref)
        if entity_kind is not None:
            current = [e for e in current if e.entity_kind == entity_kind]
        if len(current) == 1:
            return Resolution(ref, "unique", "current-name", current,
                              "resolved by current name")
        if len(current) > 1:
            return Resolution(ref, "ambiguous", "current-name", current,
                              f"{len(current)} entities claim {ref!r} as their current name "
                              "(collision detector should resolve this)")

        # 3. past names (history survives renames)
        past = self.registry.by_past_name(ref)
        if entity_kind is not None:
            past = [e for e in past if e.entity_kind == entity_kind]
        if len(past) == 1:
            ent = past[0]
            return Resolution(ref, "unique", "past-name", past,
                              f"past name of {ent.entity_id} (current name "
                              f"{ent.current_name!r}); lineage preserved")
        if len(past) > 1:
            return Resolution(ref, "ambiguous", "past-name", past,
                              f"{len(past)} entities once carried {ref!r}")

        return Resolution(ref, "unknown", "none", [], "no id, current name, or past name matched")
