"""AION-1 CAPABILITY BATTERY + the CAPABILITY-SURVIVAL METRIC (owner spec).

The owner's five questions, as capabilities:
    recognise    does the system still know what it learned?
    predict      can it predict the next relationship from what it learned?
    learn_new    can it learn a NEW relationship without a host feeding it?
    combine      can it combine OLD relationships into a NEW discovery?
    self_correct can it correct itself when reality disagrees?

Run in TWO modes:
    WITH host    the relational pathway is fed by the HOST's hidden states
                 (RelationalInterface forward hooks on the frozen foundation).
    WITHOUT host the host is DISCONNECTED ENTIRELY: no model, no hidden states,
                 no hooks. The same FROZEN PlasticRelationalConnection machinery
                 is fed by DIRECT OBSERVATIONS only (DirectFeed: prompt ids ->
                 deterministic synthetic states via a seeded projector).

CAPABILITY-SURVIVAL METRIC: per capability,
    survival = capability_score(WITHOUT host) / capability_score(WITH host)
Read as: how much of what was learned survives cutting the host off — the
disconnect test in one number per capability. (score==0 with host -> ratio
1.0 if bare also 0, else +inf.)

DATA-DRIVEN: the battery is consumed from aion/worlds (concurrent lane, NOT
imported here — it delivers generator.py). This harness only reads the JSON
shape documented below; the local CPU fixture is
aion/memory/fixtures/tiny_battery.json.

------------------------------------------------------------------------------
EXPECTED-INTERFACE CONTRACT — aion/worlds battery JSON (this lane's input)
------------------------------------------------------------------------------
{
  "battery_id": str,
  "seed_relations": [ {"subject": str, "relation": str, "object": str}, ... ],
  "questions": [
      {"qid": str,
       "capability": "recognise" | "predict" | "learn_new" | "combine" | "self_correct",
       "prompt": str, "paraphrases": [str, ...], "gold": str},
      ...
  ],
  "new_relations":   [ {"subject","relation","object"}, ... ],   # planted AFTER
                                                                   # the seed phase
  "combinations":    [ {"chain": [{"subject","relation","object"}, ...],
                        "subject": str, "relation": str, "gold": str}, ... ],
  "corrections":     [ {"subject","relation","object",                        # new fact
                        "replaces": {"subject","relation","object"},          # old fact
                        "correction_text": str}, ... ]
}
Both lanes learn the same seed_relations, the same new_relations, the same
corrections; capabilities are scored identically against each lane's
RelationalMemory index. No dialogue-lane code is imported (concurrent lane);
when it lands, ModelPathway scoring plugs in through the AIONAnswerer contract
documented in aion/experiments/retention.py.

CLI (build-ahead; do NOT run real experiments until the pilot greenlights):
    python aion/experiments/knowledge_discovery.py --battery aion/memory/fixtures/tiny_battery.json --seed 0
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.memory.relational import (                                     # noqa: E402
    RelationalMemory, relation_key,
)
from aion.memory.episodic import EpisodicStore                           # noqa: E402
from aion.memory.cpu_stubs import build_stub_base, DirectFeed, StubVocab  # noqa: E402
from aion.memory.consolidation import ConsolidationLedger                # noqa: E402
from aion.relational.interface import RelationalInterface, InterfaceConfig  # noqa: E402
from aion.relational.interface import PlasticRelationalConnection           # noqa: E402

DEFAULT_BATTERY = os.path.join(_REPO_ROOT, "aion", "memory", "fixtures", "tiny_battery.json")

CAPABILITIES = ("recognise", "predict", "learn_new", "combine", "self_correct")


# --------------------------------------------------------------------------- #
# The two lanes
# --------------------------------------------------------------------------- #

class HostLane:
    """WITH host: the interface's hooked hidden states feed the pathway."""

    name = "host"

    def __init__(self, seed: int = 0, hidden_size: int = 64):
        torch.manual_seed(seed)
        base = build_stub_base(seed=seed, hidden_size=hidden_size)
        self.iface = RelationalInterface(base, cfg=InterfaceConfig())
        self.vocab = StubVocab(vocab_size=128)
        self.episodic = EpisodicStore(clock=lambda: 0.0)
        self.hidden_dim = self.iface.hidden_dim
        self.pair_index = 0

        def make_probe(prompt_ids: torch.Tensor):
            with torch.no_grad():
                res = self.iface.forward(prompt_ids)
            pre_i, post_i = self.iface.pairs[self.pair_index]
            return (res["hidden"][pre_i].mean(dim=1)[0].float(),
                    res["hidden"][post_i].mean(dim=1)[0].float())

        def learn_fn(prompt_ids: torch.Tensor):
            return self.iface.nnfs_update(prompt_ids)

        self.make_probe = make_probe
        self.learn_fn = learn_fn
        self.rmem = RelationalMemory(conns=list(self.iface.conns),
                                     hidden_dim=self.hidden_dim, episodic=self.episodic,
                                     blend_ratio=self.iface.cfg.blend_ratio,
                                     strength_threshold=1e-8)
        self.ledger = ConsolidationLedger(conns=list(self.iface.conns),
                                          hidden_dim=self.hidden_dim)


class BareLane:
    """WITHOUT host: host disconnected ENTIRELY.

    The same frozen PlasticRelationalConnection machinery (same relation type,
    learning rate, trace decay, blend, consolidation interval as the interface
    builds) is instantiated bare and fed DirectFeed synthetic observation states
    only. No model, no hidden states, no hooks, no host call of any kind.
    """

    name = "bare"

    def __init__(self, seed: int = 0, hidden_size: int = 64, vocab_size: int = 128):
        torch.manual_seed(seed + 1)              # independent lane seed, still deterministic
        cfg = InterfaceConfig()
        self.conn = PlasticRelationalConnection(
            pre_idx=0, post_idx=1, pre_dim=hidden_size, post_dim=hidden_size,
            context_dim=0, relation_type=cfg.relation_type,
            learning_rate=cfg.learning_rate, trace_decay=cfg.trace_decay,
            blend_ratio=cfg.blend_ratio,
            consolidation_interval=cfg.consolidation_interval)
        for p in self.conn.parameters():
            p.requires_grad_(False)              # NNFS owns these (interface convention)
        self.feed = DirectFeed(vocab_size=vocab_size, dim=hidden_size, seed=99)
        self.vocab = StubVocab(vocab_size=vocab_size)
        self.episodic = EpisodicStore(clock=lambda: 0.0)
        self.hidden_dim = hidden_size
        self.pair_index = 0

        def make_probe(prompt_ids: torch.Tensor):
            ids = [int(i) for i in prompt_ids.flatten().tolist()]
            return self.feed.encode(ids)

        def learn_fn(prompt_ids: torch.Tensor):
            pre, post = make_probe(prompt_ids)
            with torch.no_grad():
                _, stats = self.conn(pre.unsqueeze(0), post.unsqueeze(0),
                                     target=post.unsqueeze(0), learn=True)
            return {"nnfs": {"pair0": stats}}

        self.make_probe = make_probe
        self.learn_fn = learn_fn
        self.rmem = RelationalMemory(conns=[self.conn], hidden_dim=self.hidden_dim,
                                     episodic=self.episodic, blend_ratio=cfg.blend_ratio,
                                     strength_threshold=1e-8)
        self.ledger = ConsolidationLedger(conns=[self.conn], hidden_dim=self.hidden_dim)


LANES = {"host": HostLane, "bare": BareLane}


# --------------------------------------------------------------------------- #
# Battery execution (identical code path for both lanes)
# --------------------------------------------------------------------------- #

def _learn_relation(lane, rel: Dict, kind: str = "lesson", n_steps: int = 2,
                    teacher: str = "battery") -> str:
    key = relation_key(rel["subject"], rel["relation"], rel["object"])
    fact = f"{rel['subject']} {rel['relation']} {rel['object']}"
    ids = lane.vocab.encode(fact)
    lane.rmem.learn_relation(key, ids.flatten().tolist(), learn_fn=lane.learn_fn,
                             teacher=teacher, kind=kind, channel="transcript",
                             n_steps=n_steps)
    return key


def run_lane(lane_cls, battery: Dict, seed: int = 0, n_steps: int = 2) -> Dict:
    lane = lane_cls(seed=seed)
    details: Dict[str, Dict] = {cap: {"n": 0, "pass": 0} for cap in CAPABILITIES}

    # phase 1: seed relations
    seed_keys = [_learn_relation(lane, rel, n_steps=n_steps)
                 for rel in battery.get("seed_relations", [])]

    # phase 2: NEW relations (learn_new capability)
    new_keys = []
    for rel in battery.get("new_relations", []):
        new_keys.append(_learn_relation(lane, rel, kind="lesson", n_steps=n_steps,
                                        teacher="battery_new"))
    for rel in battery.get("new_relations", []):
        key = relation_key(rel["subject"], rel["relation"], rel["object"])
        ok = lane.rmem.present(key)
        details["learn_new"]["n"] += 1
        details["learn_new"]["pass"] += int(ok)

    # phase 3: corrections (self_correct capability) — new fact must be learned,
    # the replaced fact must not be destroyed.
    for corr in battery.get("corrections", []):
        old_key = relation_key(corr["replaces"]["subject"], corr["replaces"]["relation"],
                               corr["replaces"]["object"])
        old_before = lane.rmem.strength(old_key)
        new_key = _learn_relation(lane, corr, kind="correction", n_steps=n_steps,
                                  teacher="reality")
        if old_key in lane.rmem.records:
            lane.rmem.records[old_key].superseded_by = new_key
        old_after = lane.rmem.strength(old_key)
        new_ok = lane.rmem.present(new_key)
        old_kept = bool(old_before is not None and old_after is not None
                        and old_after >= 0.5 * max(old_before, 1e-12))
        details["self_correct"]["n"] += 1
        details["self_correct"]["pass"] += int(new_ok and old_kept)

    # phase 4: consolidation (both lanes, frozen machinery, audited)
    consolidation = lane.ledger.consolidate_all(step=n_steps * 4)

    # phase 5: score recognise / predict / combine
    for rel in battery.get("seed_relations", []):
        key = relation_key(rel["subject"], rel["relation"], rel["object"])
        details["recognise"]["n"] += 1
        details["recognise"]["pass"] += int(lane.rmem.present(key))

        # predict: query (subject, relation) probe -> nearest indexed relation
        # should be the gold object's relation.
        ids = lane.vocab.encode(f"{rel['subject']} {rel['relation']}")
        pre, _ = lane.make_probe(ids)
        hits = lane.rmem.nearest(pre, k=1)
        details["predict"]["n"] += 1
        details["predict"]["pass"] += int(bool(hits) and hits[0][0].key == key)

    for combo in battery.get("combinations", []):
        qtext = f"{combo['subject']} {combo['relation']}"
        ids = lane.vocab.encode(qtext)
        pre, _ = lane.make_probe(ids)
        hits = lane.rmem.nearest(pre, k=1)
        reached = bool(hits) and combo["gold"] in hits[0][0].key.split("|")
        details["combine"]["n"] += 1
        details["combine"]["pass"] += int(reached)

    scores = {cap: (d["pass"] / d["n"] if d["n"] else 0.0) for cap, d in details.items()}
    return {
        "lane": lane.name,
        "scores": scores,
        "details": details,
        "consolidation_invariant_ok": consolidation["invariant_ok"],
        "indexed_relations": len(lane.rmem.records),
        "inventory": lane.rmem.inventory(),
    }


# --------------------------------------------------------------------------- #
# CAPABILITY-SURVIVAL METRIC (pure function — unit-tested on fixture numbers)
# --------------------------------------------------------------------------- #

def survival_ratio(with_score: float, without_score: float) -> Optional[float]:
    """capability-without-host / capability-with-host.

    with==0 and without==0 -> 1.0 (nothing to lose: survival undefined-but-full;
    reported as 1.0 so a dead-with-host capability never hides bare capability).
    with==0 and without>0  -> math.inf (impossible gain: flags a scoring artefact).
    """
    if with_score > 0:
        return without_score / with_score
    return 1.0 if without_score <= 0 else math.inf


def survival_ratios(with_scores: Dict[str, float],
                    without_scores: Dict[str, float]) -> Dict[str, Optional[float]]:
    return {cap: survival_ratio(with_scores.get(cap, 0.0),
                                without_scores.get(cap, 0.0))
            for cap in CAPABILITIES}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_battery(battery_path: str, seed: int = 0, n_steps: int = 2,
                out_path: Optional[str] = None, verbose: bool = True) -> Dict:
    with open(battery_path, "r", encoding="utf-8") as f:
        battery = json.load(f)

    with_host = run_lane(HostLane, battery, seed=seed, n_steps=n_steps)
    without_host = run_lane(BareLane, battery, seed=seed, n_steps=n_steps)
    survival = survival_ratios(with_host["scores"], without_host["scores"])

    report = {
        "battery_id": battery.get("battery_id", "?"),
        "seed": seed,
        "with_host": with_host,
        "without_host": without_host,
        "survival_ratio": survival,
        "verdict": {
            cap: ("fully-survives" if r is not None and math.isfinite(r) and r >= 1.0
                  else "degrades" if r is not None and math.isfinite(r)
                  else "impossible-gain(check-scoring)")
            for cap, r in survival.items()},
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    if verbose:
        print(f"[discovery] battery={report['battery_id']} seed={seed}")
        print(f"{'capability':14s} {'with':>7s} {'without':>8s} {'survival':>10s}")
        for cap in CAPABILITIES:
            r = survival[cap]
            rs = "inf" if r == math.inf else f"{r:.3f}" if r is not None else "-"
            print(f"{cap:14s} {with_host['scores'][cap]:7.3f} "
                  f"{without_host['scores'][cap]:8.3f} {rs:>10s}")
        print(f"[discovery] consolidation invariant: host={with_host['consolidation_invariant_ok']} "
              f"bare={without_host['consolidation_invariant_ok']}")
        if out_path:
            print(f"[discovery] report: {out_path}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="AION-1 capability battery + survival metric")
    ap.add_argument("--battery", default=DEFAULT_BATTERY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn-steps", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(_REPO_ROOT, "runs", "discovery_report.json")
    run_battery(args.battery, seed=args.seed, n_steps=args.learn_steps, out_path=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
