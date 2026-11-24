"""AION-1 synthetic dialogue world: 100 invented chain relationships + four question batteries.

Owner spec (2026-11-24): chain structures like
    Neral operates Tav -> Tav routes to relay Kimo -> Kimo controls device Zor -> Zor produces heat
ALL entity names INVENTED (held-out): pretrained knowledge must not supply answers, so every
battery answer is computable from the generated chain graph alone.

Batteries are DATA, not prompts:
  (a) direct facts        - one question per relationship
  (b) unseen combinations - actor -> downstream node never directly taught (reach / membership)
  (c) paraphrases         - same fact, different wording (seeded template choice)
  (d) interventions       - conduit rerouted to another relay; ground truth recomputed on the
                            MODIFIED graph ("Tav is rerouted to relay X - what does Neral
                            reach now?")

Deterministic under `seed` (random.Random(seed)); JSON output via World.to_dict() / save_json().
The learning-loop lane consumes relationships (-> plastic signal) and questions (-> batteries);
verify_world() re-derives every answer from the graph and returns failures (empty list = sound).
"""

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# Invented entity pools (held-out; owner's canon names lead each role). 25 each so the default
# 25-chain world uses every pool exactly once. Outputs are generic phenomena (entity->output
# mapping is what is invented, and it is distinct per chain so every battery answer is unique).
ACTORS = ["Neral", "Sula", "Dravy", "Ombit", "Quelos", "Vantro", "Bexil", "Corvan", "Ithra",
          "Mavok", "Talen", "Rhoka", "Juvex", "Kelthan", "Orvyn", "Gholm", "Nydra", "Ashvi",
          "Fendor", "Wexla", "Xyphor", "Brunev", "Thalex", "Iovar", "Zundi"]
CONDUITS = ["Tav", "Rinpo", "Skell", "Morva", "Plin", "Drex", "Volim", "Harpa", "Nixo", "Qarbe",
            "Lomo", "Yentz", "Ufka", "Grev", "Bokka", "Sifel", "Tormu", "Vlask", "Zenna", "Ektar",
            "Pyx", "Wroll", "Aumsa", "Chesk", "Ondri"]
RELAYS = ["Kimo", "Drel", "Vash", "Osten", "Belro", "Thop", "Mirda", "Cuvex", "Ralt", "Sorni",
          "Fydd", "Glova", "Ikto", "Norva", "Pavok", "Ruzel", "Trimm", "Wonda", "Xalor", "Yibbo",
          "Zaffi", "Angth", "Blokt", "Criva", "Duvra"]
DEVICES = ["Zor", "Blun", "Crat", "Dovel", "Errik", "Fendu", "Grix", "Huvla", "Ixen", "Jarvo",
           "Krenn", "Lypox", "Mirno", "Nyvox", "Oztek", "Prava", "Quist", "Ryvok", "Skerv",
           "Tolvo", "Urska", "Vikra", "Wexu", "Xorn", "Zaph"]
OUTPUTS = ["heat", "light", "pressure", "torque", "steam", "glow", "sparks", "hum", "vapor",
           "motion", "ripple", "chime", "current", "draft", "echo", "frost", "grit", "haze",
           "ion", "jolt", "mist", "note", "pulse", "rime", "shade"]

# canonical chain links: (relation, subject_role, object_role)
CHAIN_LINKS = [
    ("operates", "actor", "conduit"),
    ("routes-to", "conduit", "relay"),
    ("controls", "relay", "device"),
    ("produces", "device", "output"),
]

ROLE_POOLS = {"actor": ACTORS, "conduit": CONDUITS, "relay": RELAYS,
              "device": DEVICES, "output": OUTPUTS}

DIRECT_ASK = {
    "operates": "Who does {subject} operate?",
    "routes-to": "Where does {subject} route?",
    "controls": "Who does {subject} control?",
    "produces": "What does {subject} produce?",
}

SENTENCE_TEMPLATES = {
    "operates": "{subject} operates {object}.",
    "routes-to": "{subject} routes to relay {object}.",
    "controls": "{subject} controls device {object}.",
    "produces": "{subject} produces {object}.",
}

PARAPHRASE_TEMPLATES = {
    "operates": [
        ("Which actor operates {object}?", "subject"),
        ("{object} is operated by whom?", "subject"),
    ],
    "routes-to": [
        ("Which relay does {subject} route to?", "object"),
        ("{subject} sends its flow through which relay?", "object"),
    ],
    "controls": [
        ("Which device is controlled by {subject}?", "object"),
        ("{subject} governs which device?", "object"),
    ],
    "produces": [
        ("{object} is produced by which device?", "subject"),
        ("{subject} emits what?", "object"),
    ],
}

BATTERY_DIRECT = "direct"
BATTERY_COMBINATION = "combination"
BATTERY_PARAPHRASE = "paraphrase"
BATTERY_INTERVENTION = "intervention"


@dataclass
class Relationship:
    """One taught fact: subject --relation--> object, plus the canonical lesson sentence."""

    chain_id: int
    index: int
    subject: str
    relation: str
    object: str
    sentence: str


@dataclass
class Question:
    """One battery question with its ground-truth answer and derivation metadata."""

    id: str
    battery: str
    text: str
    answer: str
    chain_id: int
    meta: dict = field(default_factory=dict)


@dataclass
class World:
    """Generated world: relationships, chain graph, and the four question batteries."""

    seed: int
    n_chains: int
    actors: List[str]
    conduits: List[str]
    relays: List[str]
    devices: List[str]
    outputs: List[str]
    relationships: List[Relationship]
    chains: List[List[str]]                       # [actor, conduit, relay, device, output]
    graph: Dict[str, List[Tuple[str, str]]]       # subject -> [(relation, object)]
    questions: List[Question]

    # -- access helpers ----------------------------------------------------- #

    def relationships_of_chain(self, chain_id: int) -> List[Relationship]:
        return [r for r in self.relationships if r.chain_id == chain_id]

    def reachable(self, start: str, graph: Optional[Dict] = None) -> List[str]:
        """Chain world: every node has <=1 outgoing edge, so reachability is a linear walk
        (cycle-guarded). Returns [start, ...] inclusive path order."""
        g = self.graph if graph is None else graph
        node, path, seen = start, [start], {start}
        while node in g:
            _, nxt = g[node][0]
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            node = nxt
        return path

    def terminal(self, start: str, graph: Optional[Dict] = None) -> Optional[str]:
        """Last node reachable from `start` (the chain's output for an actor)."""
        path = self.reachable(start, graph)
        return path[-1] if len(path) > 1 else None

    # -- serialization ------------------------------------------------------- #

    def to_dict(self) -> dict:
        d = asdict(self)
        d["graph"] = {k: [list(e) for e in v] for k, v in self.graph.items()}
        return d

    def save_json(self, path: str) -> dict:
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return payload


def apply_reroute(graph: Dict[str, List[Tuple[str, str]]], conduit: str, new_relay: str) -> Dict:
    """Copy `graph` with conduit's routes-to edge pointed at `new_relay`
    (intervention semantics)."""
    g = copy.deepcopy(graph)
    g[conduit] = [("routes-to", new_relay)]
    return g


def generate_world(seed: int = 0, n_chains: int = 25) -> World:
    """Build the synthetic world: n_chains chain structures -> 4*n_chains relationships (100 by
    default), then the four batteries. Deterministic: same seed -> identical JSON dump."""
    for role, pool in ROLE_POOLS.items():
        if role == "output":
            continue
        assert len(pool) >= n_chains, f"{role} pool too small for n_chains={n_chains}"
    assert n_chains >= 2, "intervention battery needs at least 2 chains"

    rng = random.Random(seed)
    actors, conduits = ACTORS[:n_chains], CONDUITS[:n_chains]
    relays, devices, outputs = RELAYS[:n_chains], DEVICES[:n_chains], OUTPUTS[:n_chains]

    relationships: List[Relationship] = []
    chains: List[List[str]] = []
    graph: Dict[str, List[Tuple[str, str]]] = {}
    for c in range(n_chains):
        roles = {"actor": actors[c], "conduit": conduits[c],
                 "relay": relays[c], "device": devices[c], "output": outputs[c]}
        chains.append([roles[l[1]] for l in CHAIN_LINKS] + [roles["output"]])
        for k, (relation, subj_role, obj_role) in enumerate(CHAIN_LINKS):
            subj, obj = roles[subj_role], roles[obj_role]
            relationships.append(Relationship(
                chain_id=c, index=len(relationships), subject=subj, relation=relation, object=obj,
                sentence=SENTENCE_TEMPLATES[relation].format(subject=subj, object=obj)))
            graph[subj] = [(relation, obj)]

    questions: List[Question] = []

    # (a) direct facts - one question per relationship
    for r in relationships:
        questions.append(Question(
            id=f"direct-{r.index:03d}", battery=BATTERY_DIRECT,
            text=DIRECT_ASK[r.relation].format(subject=r.subject), answer=r.object,
            chain_id=r.chain_id,
            meta={"subject": r.subject, "relation": r.relation, "object": r.object}))

    # (b) unseen combinations - actor -> downstream node never directly taught
    for c, chain in enumerate(chains):
        actor, device, output = chain[0], chain[3], chain[4]
        questions.append(Question(
            id=f"combo-reach-{c:03d}", battery=BATTERY_COMBINATION,
            text=f"What does {actor} reach?", answer=output, chain_id=c,
            meta={"kind": "reach", "start": actor}))
        # a DIFFERENT chain's device must be picked; n_chains=2 admits step 2 (0 mod 2),
        # which would point combo-no at this very chain and break verify_world. Filtered
        # steps keep generation identical for n_chains>=3 and make n_chains=2 sound.
        step_opts = [o for o in (1, 2) if o % n_chains]
        other = (c + rng.choice(step_opts)) % n_chains
        questions.append(Question(
            id=f"combo-yes-{c:03d}", battery=BATTERY_COMBINATION,
            text=f"Does {actor} reach {device}?", answer="yes", chain_id=c,
            meta={"kind": "membership", "start": actor, "target": device}))
        questions.append(Question(
            id=f"combo-no-{c:03d}", battery=BATTERY_COMBINATION,
            text=f"Does {actor} reach {devices[other]}?", answer="no", chain_id=c,
            meta={"kind": "membership", "start": actor, "target": devices[other]}))

    # (c) paraphrases - same fact, different wording (seeded template choice)
    for r in relationships:
        template, asked_side = PARAPHRASE_TEMPLATES[r.relation][rng.randrange(2)]
        answer = r.subject if asked_side == "subject" else r.object
        questions.append(Question(
            id=f"para-{r.index:03d}", battery=BATTERY_PARAPHRASE,
            text=template.format(subject=r.subject, object=r.object), answer=answer,
            chain_id=r.chain_id,
            meta={"subject": r.subject, "relation": r.relation, "object": r.object,
                  "asked_side": asked_side}))

    # (d) interventions - conduit rerouted to a different relay; truth from the modified graph
    offsets = [o for o in (1, 3) if o < n_chains] or [1]
    for c, chain in enumerate(chains):
        j = (c + rng.choice(offsets)) % n_chains
        conduit, actor, new_relay, new_output = chain[1], chain[0], relays[j], outputs[j]
        questions.append(Question(
            id=f"int-reach-{c:03d}", battery=BATTERY_INTERVENTION,
            text=(f"{conduit} is rerouted to relay {new_relay}. What does {actor} reach now?"),
            answer=new_output, chain_id=c,
            meta={"kind": "reach_now", "conduit": conduit, "new_relay": new_relay, "start": actor}))
        questions.append(Question(
            id=f"int-route-{c:03d}", battery=BATTERY_INTERVENTION,
            text=(f"{conduit} is rerouted to relay {new_relay}. Which relay does {actor} route "
                  f"through now?"), answer=new_relay, chain_id=c,
            meta={"kind": "route_now", "conduit": conduit, "new_relay": new_relay, "start": actor}))

    return World(seed=seed, n_chains=n_chains, actors=actors, conduits=conduits, relays=relays,
                 devices=devices, outputs=outputs, relationships=relationships, chains=chains,
                 graph=graph, questions=questions)


def verify_world(world: World) -> List[str]:
    """Re-derive every battery answer from the (possibly intervened) chain graph.

    Returns a list of failure strings; empty list means every answer is graph-derivable.
    This is the ground-truth consistency gate the dialogue tests assert on.
    """
    failures: List[str] = []
    edges = {(s, rel, o) for s, es in world.graph.items() for (rel, o) in es}

    def fact_in_graph(subject: str, relation: str, obj: str) -> bool:
        return (subject, relation, obj) in edges

    for q in world.questions:
        m = q.meta
        if q.battery == BATTERY_DIRECT:
            ok = fact_in_graph(m["subject"], m["relation"], q.answer) and q.answer == m["object"]
        elif q.battery == BATTERY_PARAPHRASE:
            ok = (fact_in_graph(m["subject"], m["relation"], m["object"])
                  and q.answer == m[m["asked_side"]])
        elif q.battery == BATTERY_COMBINATION:
            if m["kind"] == "reach":
                ok = world.terminal(m["start"]) == q.answer
            else:  # membership
                ok = ((m["target"] in world.reachable(m["start"])) == (q.answer == "yes"))
        elif q.battery == BATTERY_INTERVENTION:
            g2 = apply_reroute(world.graph, m["conduit"], m["new_relay"])
            if m["kind"] == "reach_now":
                path = world.reachable(m["start"], g2)
                ok = (path[-1] == q.answer and m["new_relay"] in path
                      and world.relationships[4 * q.chain_id + 1].object not in path)
            else:  # route_now
                ok = (dict(world.graph[m["conduit"]]).get("routes-to") is not None
                      and dict(g2[m["conduit"]])["routes-to"] == q.answer == m["new_relay"])
        else:
            ok = False
        if not ok:
            failures.append(f"{q.id}: answer={q.answer!r} not derivable (meta={m})")
    return failures


if __name__ == "__main__":
    w = generate_world()
    print(f"relationships={len(w.relationships)} questions={len(w.questions)}")
    print("verify:", verify_world(w) or "ALL ANSWERS GRAPH-DERIVABLE")
