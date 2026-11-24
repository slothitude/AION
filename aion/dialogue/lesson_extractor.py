"""Lesson extractor: teacher turns -> discrete relationship facts for the PLASTIC pathway.

THE KEY COMPONENT (owner brief, 2026-11-24): a lesson or correction turn is converted into
discrete (subject, relation, object) facts that become NNFS training signal through
aion/relational/interface.py — relationships enter the plastic system, NOT merely a transcript
row. v1 is deterministic rule-based extraction over the templated synthetic world:

    "Neral operates Tav"                                  -> (Neral, operates, Tav)
    "A motor converts electrical energy into mechanical"  -> (electrical energy, converts-to,
        " energy"                                            mechanical energy, agent=a motor)
    "Tav is rerouted to Kimo"                             -> intervention event (routes-to update)

LLM-extractor slot (later, not wired): the conversation layer only requires an object with
`.extract(text, provenance=...)` / `.extract_correction(text)`, so a teacher-backed extractor
(e.g. structured-JSON fact list from the same endpoint) can replace RuleExtractor without
touching any other module.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# relations the world vocabulary uses (see aion/worlds/generator.py CHAIN_LINKS)
RELATIONS = ("operates", "routes-to", "controls", "produces", "converts-to")

_NOUN_PREFIX = r"(?:(?:relay|device|the)\s+)?"

# "Tav is rerouted to relay Kimo."  -> intervention (routes-to update)
_REROUTE = re.compile(
    r"^(?P<subject>[A-Za-z][\w\-']*)\s+is\s+(?:rerouted|redirected|switched)\s+to\s+"
    + _NOUN_PREFIX + r"(?P<object>.+?)\.?$",
    re.IGNORECASE)

# "A motor converts electrical energy into mechanical energy." -> subject/object swap:
# fact is (src, converts-to, dst) with the grammatical subject kept as `agent`.
_CONVERTS = re.compile(
    r"^(?P<agent>.+?)\s+converts?\s+(?P<src>.+?)\s+into\s+(?P<dst>.+?)\.?$",
    re.IGNORECASE)

# correction form: "No - Tav routes to relay Vex, not Kimo." (right fact asserted, wrong negated)
_CORRECTION = re.compile(
    r"^(?:no\b[\s.,!:\-—]*|actually[,.]?\s*|wrong[,.]?\s*|not\s+quite[,.]?\s*)?"
    r"(?P<subject>[A-Za-z][\w\-']*)\s+"
    r"(?P<verb>operates|controls|produces|routes\s+to|is\s+rerouted\s+to)\s+"
    + _NOUN_PREFIX + r"(?P<right>.+?)\s*(?:,|\s-+|\.)\s*not\s+" + _NOUN_PREFIX
    + r"(?P<wrong>.+?)\.?$",
    re.IGNORECASE)

# plain chain triple: "Neral operates Tav." / "Tav routes to relay Kimo." / ...
_TRIPLE = re.compile(
    r"^(?P<subject>[A-Za-z][\w\-']*)\s+"
    r"(?P<verb>operates|controls|produces|routes\s+to)\s+"
    + _NOUN_PREFIX + r"(?P<object>.+?)\.?$",
    re.IGNORECASE)


@dataclass
class Fact:
    """Discrete relationship fact: subject --relation--> object (the plastic training unit)."""

    subject: str
    relation: str
    object: str
    provenance: str = "lesson"        # lesson | correction | intervention
    agent: Optional[str] = None       # converter for converts-to facts ("a motor")
    raw: Optional[str] = None

    @property
    def key(self) -> tuple:
        """Ledger key: one current fact per (subject, relation)."""
        return (self.subject.lower(), self.relation)

    def to_dict(self) -> dict:
        return {"subject": self.subject, "relation": self.relation, "object": self.object,
                "provenance": self.provenance, "agent": self.agent, "raw": self.raw}


@dataclass
class InterventionEvent:
    """An action the teacher states as happening NOW: reroute of a conduit to a new target."""

    subject: str
    kind: str                     # "reroute"
    new_object: str
    fact: Fact
    raw: str = ""


@dataclass
class CorrectionFacts:
    """What a correction turn asserts vs negates: 'X routes to relay RIGHT, not WRONG.'"""

    corrected: List[Fact] = field(default_factory=list)   # after-state facts the teacher asserts
    rejected: List[Fact] = field(default_factory=list)    # before-state candidates ("not X")
    raw: str = ""

    @property
    def has_override(self) -> bool:
        return bool(self.corrected)


@dataclass
class ExtractionResult:
    """Everything one lesson turn yields: facts for the plastic pathway + intervention events."""

    facts: List[Fact] = field(default_factory=list)
    interventions: List[InterventionEvent] = field(default_factory=list)
    unparsed: List[str] = field(default_factory=list)     # sentences no template matched


def _norm_object(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _verb_to_relation(verb: str) -> str:
    return "routes-to" if "route" in verb.lower() else verb.strip().lower().replace(" ", "-")


def extract(text: str, provenance: str = "lesson") -> ExtractionResult:
    """Deterministic rule-based extraction over templated sentences. Tries, per sentence:
    intervention -> converts -> correction-shaped triple -> plain triple."""
    result = ExtractionResult()
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue

        m = _REROUTE.match(sentence)
        if m:
            fact = Fact(subject=_norm_object(m.group("subject")), relation="routes-to",
                        object=_norm_object(m.group("object")), provenance="intervention",
                        raw=sentence)
            result.facts.append(fact)
            result.interventions.append(InterventionEvent(
                subject=fact.subject, kind="reroute", new_object=fact.object,
                fact=fact, raw=sentence))
            continue

        m = _CONVERTS.match(sentence)
        if m:
            result.facts.append(Fact(subject=_norm_object(m.group("src")),
                                     relation="converts-to", object=_norm_object(m.group("dst")),
                                     provenance=provenance, agent=_norm_object(m.group("agent")),
                                     raw=sentence))
            continue

        m = _CORRECTION.match(sentence)
        if m:
            relation = _verb_to_relation(m.group("verb"))
            subject = _norm_object(m.group("subject"))
            result.facts.append(Fact(subject=subject, relation=relation,
                                     object=_norm_object(m.group("right")),
                                     provenance=provenance, raw=sentence))
            continue

        m = _TRIPLE.match(sentence)
        if m:
            result.facts.append(Fact(subject=_norm_object(m.group("subject")),
                                     relation=_verb_to_relation(m.group("verb")),
                                     object=_norm_object(m.group("object")),
                                     provenance=provenance, raw=sentence))
            continue

        result.unparsed.append(sentence)
    return result


def extract_correction(text: str) -> CorrectionFacts:
    """Extract the after-state facts a correction asserts and the before-state facts it negates
    ('... RIGHT, not WRONG'). Sentences without the ', not X' shape yield corrected-only facts
    (the ledger then supplies the before state)."""
    out = CorrectionFacts(raw=text)
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        m = _CORRECTION.match(sentence)
        if m:
            relation = _verb_to_relation(m.group("verb"))
            subject = _norm_object(m.group("subject"))
            out.corrected.append(Fact(subject=subject, relation=relation,
                                      object=_norm_object(m.group("right")),
                                      provenance="correction", raw=sentence))
            out.rejected.append(Fact(subject=subject, relation=relation,
                                     object=_norm_object(m.group("wrong")),
                                     provenance="correction", raw=sentence))
            continue
        plain = extract(sentence, provenance="correction")
        out.corrected.extend(f for f in plain.facts if f.relation != "converts-to")
        for iv in plain.interventions:
            out.corrected.append(iv.fact)
    return out


class RuleExtractor:
    """Duck-type seam the conversation layer holds: same method surface a later LLM-backed
    extractor would implement (.extract / .extract_correction). Delegates to the functions
    above; swapping in a teacher-JSON extractor touches nothing else."""

    def extract(self, text: str, provenance: str = "lesson") -> ExtractionResult:
        return extract(text, provenance=provenance)

    def extract_correction(self, text: str) -> CorrectionFacts:
        return extract_correction(text)
