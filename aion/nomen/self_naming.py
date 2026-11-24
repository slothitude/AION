"""NOMEN self-naming: the instance's own identity, chosen increasingly on its own terms.

OWNER DESIGN LAW (NOMEN, 2026-11-25): the AION instance eventually chooses its own
individual name. The path is staged by PROVENANCE, never by a hardcoded schedule:
  1. initial identity is SUPPLIED ("AION" + instance id) — provenance "initial";
  2. early candidates come teacher-assisted — proposals from the optional llm_assist
     hook enter the lineage as "teacher-proposed", and a chosen one is recorded with
     deciding provenance "teacher-assisted";
  3. later candidates are internal — observation-derived proposals chosen with deciding
     provenance "internal". The hook is optional forever; independence is the default.

A self-naming event is a rename in the instance's own lineage (history never lost,
identity never changed, verification_status never touched) plus one transition edge in
the same evolution graph every other entity uses. The candidate list and the deciding
provenance are stored in the lineage entry's basis, so the WHY of the choice survives
persistence exactly.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from aion.nomen.entity_registry import Entity, EntityRegistry
from aion.nomen.lineage_graph import GraphNode, LineageGraph
from aion.nomen.name_generator import NameGenerator, NameProposal

DECIDING_PROVENANCES = ("teacher-assisted", "internal")


@dataclass
class SelfNamingEvent:
    """One recorded act of the instance naming itself."""

    seq: int
    old_name: str
    chosen: str
    deciding_provenance: str           # "teacher-assisted" | "internal"
    candidates: List[str]
    candidate_provenances: List[str]   # parallel to candidates
    reason: str
    node_id: str                       # graph node of the NEW named state

    def to_dict(self) -> dict:
        return {"seq": int(self.seq), "old_name": self.old_name, "chosen": self.chosen,
                "deciding_provenance": self.deciding_provenance,
                "candidates": list(self.candidates),
                "candidate_provenances": list(self.candidate_provenances),
                "reason": self.reason, "node_id": self.node_id}


class SelfIdentity:
    """The AION instance as a named entity: initial name supplied, later names chosen."""

    def __init__(self, registry: EntityRegistry, graph: LineageGraph, instance_id: str,
                 initial_name: str = "AION", name_generator: Optional[NameGenerator] = None):
        self.registry = registry
        self.graph = graph
        self.instance_id = str(instance_id)
        self.generator = name_generator or NameGenerator()
        self._seq = 0
        supplied = f"{initial_name}-{self.instance_id}"
        self._entity = registry.register(
            supplied, "self", verification_status="observed", provenance="initial",
            reason=f"initial identity supplied: {initial_name} + instance id")

    # -- access ------------------------------------------------------------------ #

    @property
    def entity_id(self) -> str:
        return self._entity.entity_id

    @property
    def entity(self) -> Entity:
        return self.registry.get(self._entity.entity_id)

    @property
    def name(self) -> str:
        return self.entity.current_name

    # -- candidates ---------------------------------------------------------------- #

    def propose_candidates(self, k: int = 4, include_teacher: bool = True,
                           teacher_hook: Optional[object] = None) -> List[NameProposal]:
        """Candidate names for the instance's next self-naming. Teacher hook optional:
        with one attached (and include_teacher) proposals are marked "teacher-proposed";
        without one, observation-derived candidates only — independence needs nothing."""
        gen = self.generator if teacher_hook is None else NameGenerator(assist=teacher_hook)
        taken = {e.current_name for e in self.registry.all_entities()}
        ent = self.entity
        return gen.propose(ent.entity_id, ent.entity_kind, ent.observed_properties.keys(),
                           taken=taken, k=k, include_teacher=include_teacher)

    # -- THE ACT -------------------------------------------------------------------- #

    def self_naming_event(self, candidates: Sequence[Union[NameProposal, str]],
                          chosen: str, deciding_provenance: str,
                          reason: str = "") -> SelfNamingEvent:
        """Record the instance choosing its own name.

        chosen MUST be one of the offered candidates (a self-naming act is a choice,
        not an assertion). The rename lands in the instance's own lineage with the
        deciding provenance; the same graph records the transition; verification_status
        is untouched (safeguard law holds even for the instance itself).
        """
        if deciding_provenance not in DECIDING_PROVENANCES:
            raise ValueError(f"deciding_provenance must be one of {DECIDING_PROVENANCES}, "
                             f"got {deciding_provenance!r}")
        cand_names, cand_prov = self._normalize(candidates)
        if chosen not in cand_names:
            raise ValueError(f"chosen name {chosen!r} is not among the offered candidates "
                             f"{cand_names} — self-naming is a recorded choice")
        self._seq += 1
        old = self.name
        ent = self.entity
        self.registry.rename(
            ent, chosen, provenance=deciding_provenance,
            reason=reason or f"self-naming ({deciding_provenance})",
            basis={"candidates": cand_names, "candidate_provenances": cand_prov,
                   "deciding_provenance": deciding_provenance, "event_seq": self._seq})
        node = self.graph.record_transition(ent.entity_id, old, chosen,
                                            reason=reason or f"self-named "
                                            f"({deciding_provenance})", kind="advance")
        return SelfNamingEvent(seq=self._seq, old_name=old, chosen=chosen,
                               deciding_provenance=deciding_provenance,
                               candidates=cand_names, candidate_provenances=cand_prov,
                               reason=reason, node_id=node.node_id)

    def events(self) -> List[SelfNamingEvent]:
        """Every self-naming event, derived from the lineage (so persistence is free
        and exact — nothing lives outside the registry payload)."""
        out = []
        ent = self.entity
        for prev, entry in zip(ent.lineage[:-1], ent.lineage[1:]):
            basis = entry.basis
            if entry.provenance in DECIDING_PROVENANCES and "candidates" in basis:
                out.append(SelfNamingEvent(
                    seq=int(basis.get("event_seq", entry.seq)),
                    old_name=prev.name, chosen=entry.name,
                    deciding_provenance=entry.provenance,
                    candidates=list(basis.get("candidates", [])),
                    candidate_provenances=list(basis.get("candidate_provenances", [])),
                    reason=entry.reason, node_id=""))
        return out

    # -- restore / persistence --------------------------------------------------------- #

    @classmethod
    def restore(cls, registry: EntityRegistry, graph: LineageGraph, entity_id: str,
                name_generator: Optional[NameGenerator] = None) -> "SelfIdentity":
        """Rebind to a loaded registry (name_memory round-trip): find the self entity
        by its id instead of registering again."""
        slf = cls.__new__(cls)
        slf.registry = registry
        slf.graph = graph
        slf.generator = name_generator or NameGenerator()
        slf._entity = registry.get(entity_id)
        slf.instance_id = slf._entity.current_name.rsplit("-", 1)[-1]
        slf._seq = len(slf.events())
        return slf

    # -- internals ------------------------------------------------------------------------ #

    @staticmethod
    def _normalize(candidates: Sequence[Union[NameProposal, str]]
                   ) -> Tuple[List[str], List[str]]:
        names, provs = [], []
        for c in candidates:
            if isinstance(c, NameProposal):
                names.append(c.name)
                provs.append(c.provenance)
            else:
                names.append(str(c))
                provs.append("internal")
        return names, provs
