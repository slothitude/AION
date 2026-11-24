"""NOMEN name generation: names EMERGE FROM OBSERVATION, never from a fixed list.

OWNER DESIGN LAW (NOMEN, 2026-11-25): a name is seeded from the entity's OBSERVED
PROPERTIES (features / relations / behaviour) through a deterministic morpheme-
composition procedure. There is no canned name list: the property vocabulary maps each
observed property word to a stem morpheme, and any UNSEEN property still deterministically
yields a stem (hash fallback), so a name can always grow from whatever AION actually saw.

Collision-aware: generation is told which names are taken and varies until the candidate
is free.

Teacher seam (the early path): an `llm_assist` hook — anything duck-typed with
propose_names(entity_id, entity_kind, observed_properties, k) -> list[str] — may propose
candidates. Teacher proposals are recorded in the lineage as provenance
"teacher-proposed"; they never bypass the record. The hook is OPTIONAL: internal
observation-derived generation always works alone (the later-independence path).
TeacherNameAssist adapts a aion/dialogue TeacherClient-style object (the owner's
structured chat client) to that hook without any network call in tests.

Determinism: pure md5-hash composition — same inputs -> same name, on any machine, with
no RNG state at all.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# observed property word -> stem morpheme. This is PRIMITIVE VOCABULARY, not a name
# list: names are composed from stems, unseen properties fall through to the hash
# syllable rule below, and nothing here is ever returned as a name by itself.
PROP_STEMS = {
    "warm": "ta", "hot": "fy", "cold": "ry", "bright": "lu", "dark": "nu",
    "fast": "vi", "slow": "so", "dense": "ka", "sparse": "fe", "linked": "ne",
    "alone": "mo", "echoing": "ko", "humming": "zu", "growing": "pa", "fading": "shi",
    "branching": "tre", "merging": "fu", "reversed": "ve", "dreaming": "on",
    "quiet": "mi", "loud": "bo", "smooth": "se", "rough": "gr", "heavy": "do",
    "fluid": "wa", "electric": "jo", "magnetic": "ma", "curved": "ba",
    "straight": "li", "cyclic": "ro", "nested": "he", "open": "aer", "closed": "ido",
}

# structural tail per entity kind (a kind marker, still not a name by itself).
KIND_TAILS = {
    "model": "ex", "brain": "ce", "relationship": "ra", "memory": "mn", "dream": "on",
    "discovery": "gn", "module": "fa", "capability": "pt", "self": "au", "world": "or",
}

_CONS = "bdfghjklmnprstvz"
_VOW = "aeiou"


def _hash_int(*parts) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest(), 16)


def _syllable(*parts) -> str:
    """Deterministic consonant+vowel syllable from any tuple of parts."""
    h = _hash_int(*parts)
    return _CONS[h % len(_CONS)] + _VOW[(h // len(_CONS)) % len(_VOW)]


def stem_for(property_word: str) -> str:
    """One observed property -> one stem morpheme. Unseen properties still compose."""
    word = str(property_word).strip().lower()
    if word in PROP_STEMS:
        return PROP_STEMS[word]
    return _syllable("prop", word)


@dataclass
class GeneratedName:
    """One internally generated candidate, with the observation basis it grew from."""

    name: str
    basis: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"name": self.name, "basis": dict(self.basis)}


@dataclass
class NameProposal:
    """A candidate offered to the registry, tagged with where it came from."""

    name: str
    provenance: str                 # "observation-derived" | "teacher-proposed"
    basis: Dict[str, object] = field(default_factory=dict)


def generate_name(entity_id: str, entity_kind: str,
                  observed_properties: Iterable[str],
                  taken: Iterable[str] = (), max_props: int = 2) -> GeneratedName:
    """Compose a candidate from observed properties (deterministic, collision-aware).

    name = stem(prop_1) + stem(prop_2) + kind_tail [+ variant syllable when taken].
    The basis records the property->stem map so the lineage can show WHY this name.
    """
    props = sorted({str(p).strip().lower() for p in (observed_properties or [])
                    if str(p).strip()})
    chosen = props[:max_props]
    stems = [stem_for(p) for p in chosen]
    if not stems:
        stems = [_syllable("bare", entity_id)]   # nothing observed yet: id-derived stem
    base = "".join(stems).capitalize()
    tail = KIND_TAILS.get(entity_kind, _syllable("kind", entity_kind))
    taken_set = set(taken)
    counter = 0
    while True:
        candidate = base + tail + (_syllable("var", entity_id, counter) if counter else "")
        if candidate not in taken_set:
            return GeneratedName(
                name=candidate,
                basis={"method": "observation-derived", "properties": {p: stem_for(p) for p in chosen},
                       "kind_tail": tail, "variations": counter})
        counter += 1            # vary the variant syllable until the name is free


def coined_name(entity_id: str, entity_kind: str, observed_properties: Iterable[str],
                taken: Iterable[str] = (), salt: int = 0) -> GeneratedName:
    """Deterministic NEW coinage (used by collision_detector strategy 2): a fresh
    id-salt-derived syllable is prefixed so the coinage cannot silently reproduce the
    observation-derived name it collided with."""
    inner = generate_name(entity_id, entity_kind, observed_properties, taken=())
    taken_set = set(taken)
    counter = 0
    while True:
        candidate = (_syllable("coin", entity_id, salt) + inner.name.lower()).capitalize()
        if candidate not in taken_set:
            return GeneratedName(
                name=candidate,
                basis={"method": "coined", "salt": int(salt), "variations": counter,
                       "inner": inner.name})
        counter += 1
        salt += 1


class NameGenerator:
    """Proposes candidates for one entity: internal observation-derived names, plus
    teacher-hook proposals when an assist hook is supplied (never required)."""

    def __init__(self, assist: Optional[object] = None):
        self.assist = assist            # duck-typed: propose_names(...) or None

    def propose(self, entity_id: str, entity_kind: str,
                observed_properties: Iterable[str],
                taken: Iterable[str] = (), k: int = 3,
                include_teacher: bool = True) -> List[NameProposal]:
        taken_set = set(taken)
        proposals: List[NameProposal] = []
        local_taken = set(taken_set)
        for i in range(int(k)):
            gen = generate_name(entity_id, entity_kind, observed_properties,
                                taken=local_taken)
            local_taken.add(gen.name)
            proposals.append(NameProposal(name=gen.name, provenance="observation-derived",
                                          basis=gen.basis))
        if include_teacher and self.assist is not None:
            for cand in self._teacher_candidates(entity_id, entity_kind,
                                                 observed_properties):
                if cand in taken_set or any(p.name == cand for p in proposals):
                    continue
                proposals.append(NameProposal(
                    name=cand, provenance="teacher-proposed",
                    basis={"method": "teacher-assist", "source": type(self.assist).__name__}))
        return proposals

    def _teacher_candidates(self, entity_id: str, entity_kind: str,
                            observed_properties: Iterable[str]) -> List[str]:
        props = [str(p) for p in (observed_properties or [])]
        return list(self.assist.propose_names(entity_id, entity_kind, props))


class TeacherNameAssist:
    """Adapts a TeacherClient-style chat object (aion/dialogue seam) into the
    llm_assist hook. The client only needs `.chat(messages) -> str` (TeacherClient
    provides it; ScriptedTeacher-style stubs can too). Zero network in tests: tests
    pass a stub client returning strict JSON.

    The teacher PROPOSES; it never names anything by itself — proposals enter the
    lineage as provenance "teacher-proposed" only if the instance (or caller) adopts
    them through a registry rename.
    """

    SYSTEM_PROMPT = (
        "You are the NAMING TEACHER in AION's naming protocol. Reply with STRICT JSON "
        'only: {"action":"name","candidates":["<name1>","<name2>", ...]}. '
        "Names are short invented words derived from the observed properties given."
    )

    def __init__(self, client, k: int = 4):
        self.client = client            # duck-typed TeacherClient-like object
        self.k = int(k)

    def propose_names(self, entity_id: str, entity_kind: str,
                      observed_properties: Sequence[str]) -> List[str]:
        user = (f"Propose {self.k} candidate names for a {entity_kind} "
                f"(internal id {entity_id}). Observed properties: "
                f"{', '.join(observed_properties) if observed_properties else '(none yet)'}.")
        content = self.client.chat([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        parsed = self._parse(content)
        if not parsed or parsed.get("action") != "name":
            return []
        cands = [str(c).strip() for c in parsed.get("candidates", []) if str(c).strip()]
        return cands[:self.k]

    @staticmethod
    def _parse(content: str) -> Optional[Dict]:
        # read-only reuse of the dialogue apparatus' tolerant strict-JSON parser
        from aion.dialogue.teacher import parse_structured
        return parse_structured(content)
