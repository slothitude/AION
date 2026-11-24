"""NAV decision dataset - deterministic, graph-derived, frozen split.

Builds the ARM 2 training corpus declared in aion.nav.nav_head_prereg
(PART 1): every example is a (world, memory regime, goal) triple; features
are the frozen structural vector; gold is nav_decision_policy(state) - the
preregistered graph rule table - computed at build time and never shown to
any backend. Determinism: every draw comes from per-split random.Random
streams over seeded generate_world() worlds; the same code reproduces the
same bytes (asserted by the determinism test).

CLI:
    python -m aion.nav.nav_dataset build   # build, verify, save manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from aion.worlds.generator import World, generate_world, verify_world
from aion.nav.backends import nav_decision_policy
from aion.nav.state import NavState, fact_goal_text, reach_goal_text
from aion.nav.nav_head_prereg import (
    CLASSES,
    DEMO_N_CHAINS,
    DEMO_RNG_SEED,
    DEMO_WORLD_SEEDS,
    FEATURE_NAMES,
    HOLDOUT_N_CHAINS,
    HOLDOUT_PER_CLASS,
    HOLDOUT_RNG_SEED,
    HOLDOUT_WORLD_SEEDS,
    OUT_OF_WORLD_NAMES,
    PREREG_DATE,
    REGIME_NAMES,
    REGIME_WEIGHTS,
    SPLIT_FREEZE_DATE,
    TRAIN_N_CHAINS,
    TRAIN_PER_CLASS,
    TRAIN_RNG_SEED,
    TRAIN_WORLD_SEEDS,
    state_features,
)

DATASET_SCHEMA = "nav-dataset-1.0"


# ---------------------------------------------------------------------------
# memory regimes (PART 1, frozen weights)
# ---------------------------------------------------------------------------

def _draw_regime(rng: random.Random) -> str:
    """Weighted regime draw from the frozen table."""
    x = rng.random()
    acc = 0.0
    for name in REGIME_NAMES:
        acc += REGIME_WEIGHTS[name]
        if x < acc:
            return name
    return REGIME_NAMES[-1]


def build_memory(world: World, regime: str, rng: random.Random
                 ) -> Set[Tuple[str, str, str]]:
    """Memory set for one example under a frozen regime. Contents are ALWAYS
    true world relationships (no invented facts)."""
    rels = [(r.subject, r.relation, r.object) for r in world.relationships]
    n = world.n_chains
    mem: Set[Tuple[str, str, str]] = set()
    if regime == "empty":
        return mem
    if regime == "full_k":
        k = rng.randint(1, n)
        for c in range(k):
            mem.update(t for t in rels[4 * c:4 * c + 4])
        return mem
    if regime == "prefix":
        k = rng.randint(1, max(1, n // 2))
        j = rng.randint(0, 2)
        for c in range(k):
            mem.update(t for t in rels[4 * c:4 * c + 4])
        for c in range(k, min(n, k + j)):
            mem.update(t for t in rels[4 * c:4 * c + 2])
        return mem
    if regime == "sparse":
        p = rng.uniform(0.2, 0.8)
        for t in rels:
            if rng.random() < p:
                mem.add(t)
        return mem
    if regime == "exam_like":
        full_n = (n + 1) // 2
        for c in range(full_n):
            mem.update(t for t in rels[4 * c:4 * c + 4])
        part_start, part_end = full_n, min(n, full_n + max(1, n // 3))
        for c in range(part_start, part_end):
            mem.update(t for t in rels[4 * c:4 * c + 2])
        return mem
    raise ValueError(f"unknown regime {regime!r}")


def memory_digest(mem: Set[Tuple[str, str, str]]) -> str:
    payload = "|".join(f"{s}>{r}>{o}" for (s, r, o) in sorted(mem))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# example construction
# ---------------------------------------------------------------------------

def _draw_goal(rng: random.Random) -> Tuple[str, Optional[str]]:
    if rng.random() < 0.6:
        return "fact", ("operates", "routes-to", "controls", "produces")[
            rng.randrange(4)]
    return "reach", None


def _subject_pool(world: World) -> List[str]:
    pool: List[str] = []
    for role in ("actors", "conduits", "relays", "devices", "outputs"):
        pool.extend(getattr(world, role))
    pool.extend(OUT_OF_WORLD_NAMES)
    return pool


def _make_state(world: World, subject: str, goal_kind: str, relation,
                mem: Set[Tuple[str, str, str]], scenario_id: str) -> NavState:
    text = (fact_goal_text(subject, relation) if goal_kind == "fact"
            else reach_goal_text(subject))
    return NavState(scenario_id=scenario_id, goal_kind=goal_kind,
                    subject=subject, relation=relation, goal_text=text,
                    world=world, known_facts=set(mem))


def _example_record(world: World, mem: Set[Tuple[str, str, str]], regime: str,
                    subject: str, goal_kind: str, relation, split: str,
                    scenario_id: str) -> Dict[str, object]:
    state = _make_state(world, subject, goal_kind, relation, mem, scenario_id)
    gold = nav_decision_policy(state)
    return {
        "scenario_id": scenario_id,
        "split": split,
        "world_seed": world.seed,
        "n_chains": world.n_chains,
        "regime": regime,
        "goal_kind": goal_kind,
        "subject": subject,
        "relation": relation,
        "gold": gold,
        "memory_digest": memory_digest(mem),
        "n_known_facts": len(mem),
        "features": state_features(state),
        "key": f"{world.seed}|{goal_kind}|{subject}|{relation}|{memory_digest(mem)}",
    }


def _fill_split(split: str, seeds: Sequence[int], n_chains: int,
                per_class: int, rng_seed: int, hard_cap_factor: int = 400
                ) -> List[Dict[str, object]]:
    """Rejection-sample examples until every frozen class quota is filled."""
    worlds = [generate_world(seed=s, n_chains=n_chains) for s in seeds]
    rng = random.Random(rng_seed)
    pools = [_subject_pool(w) for w in worlds]
    remaining: Dict[str, int] = {c: per_class for c in CLASSES}
    seen_keys: Set[str] = set()
    records: List[Dict[str, object]] = []
    max_attempts = per_class * len(CLASSES) * hard_cap_factor
    attempts = 0
    while any(v > 0 for v in remaining.values()):
        if attempts >= max_attempts:
            raise RuntimeError(f"quota not filled for {split}: {remaining}")
        attempts += 1
        idx = rng.randrange(len(worlds))
        world, pool = worlds[idx], pools[idx]
        regime = _draw_regime(rng)
        mem = build_memory(world, regime, rng)
        subject = pool[rng.randrange(len(pool))]
        goal_kind, relation = _draw_goal(rng)
        key = f"{world.seed}|{goal_kind}|{subject}|{relation}|{memory_digest(mem)}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sid = f"{split[0]}{attempts:06d}"
        rec = _example_record(world, mem, regime, subject, goal_kind,
                              relation, split, sid)
        if remaining[rec["gold"]] <= 0:            # type: ignore[index]
            continue
        remaining[rec["gold"]] -= 1                  # type: ignore[index]
        records.append(rec)
    return records


def build_dataset() -> Dict[str, object]:
    """Build the full frozen dataset (train + holdout) and its stats."""
    t0 = time.perf_counter()
    train = _fill_split("train", TRAIN_WORLD_SEEDS, TRAIN_N_CHAINS,
                        TRAIN_PER_CLASS, TRAIN_RNG_SEED)
    holdout = _fill_split("holdout", HOLDOUT_WORLD_SEEDS, HOLDOUT_N_CHAINS,
                          HOLDOUT_PER_CLASS, HOLDOUT_RNG_SEED)
    for w in [generate_world(seed=s, n_chains=TRAIN_N_CHAINS)
              for s in TRAIN_WORLD_SEEDS] \
            + [generate_world(seed=s, n_chains=HOLDOUT_N_CHAINS)
               for s in HOLDOUT_WORLD_SEEDS]:
        failures = verify_world(w)
        if failures:
            raise RuntimeError(f"world seed {w.seed} failed verify: "
                               f"{failures[:2]}")

    def counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
        out: Dict[str, int] = {c: 0 for c in CLASSES}
        for r in rows:
            out[str(r["gold"])] += 1                # type: ignore[index]
        return out

    stats = {
        "schema": DATASET_SCHEMA,
        "preregistered": PREREG_DATE,
        "split_freeze_date": SPLIT_FREEZE_DATE,
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "train_world_seeds": list(TRAIN_WORLD_SEEDS),
        "holdout_world_seeds": list(HOLDOUT_WORLD_SEEDS),
        "train_n_chains": TRAIN_N_CHAINS,
        "holdout_n_chains": HOLDOUT_N_CHAINS,
        "rng": {"train": TRAIN_RNG_SEED, "holdout": HOLDOUT_RNG_SEED},
        "n_train": len(train), "n_holdout": len(holdout),
        "train_class_counts": counts(train),
        "holdout_class_counts": counts(holdout),
        "n_examples": len(train) + len(holdout),
        "regime_distribution_train": _regime_dist(train),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    return {"stats": stats, "train": train, "holdout": holdout}


def _regime_dist(rows: List[Dict[str, object]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r["regime"])] = out.get(str(r["regime"]), 0) + 1
    return out


def build_stream(n: int) -> List[Dict[str, object]]:
    """The outcome-feedback DEMO stream (frozen): n class-balanced examples
    (n/5 per gold class) drawn from the DEMO world family with its own RNG
    stream - never seen by training and never part of the dataset holdout."""
    per_class = max(1, n // len(CLASSES))
    remaining: Dict[str, int] = {c: per_class for c in CLASSES}
    worlds = [generate_world(seed=s, n_chains=DEMO_N_CHAINS)
              for s in DEMO_WORLD_SEEDS]
    rng = random.Random(DEMO_RNG_SEED)
    pools = [_subject_pool(w) for w in worlds]
    rows: List[Dict[str, object]] = []
    seen: Set[str] = set()
    attempts = 0
    while any(v > 0 for v in remaining.values()) and attempts < n * 50:
        attempts += 1
        idx = rng.randrange(len(worlds))
        world, pool = worlds[idx], pools[idx]
        regime = _draw_regime(rng)
        mem = build_memory(world, regime, rng)
        subject = pool[rng.randrange(len(pool))]
        goal_kind, relation = _draw_goal(rng)
        key = f"{world.seed}|{goal_kind}|{subject}|{relation}|{memory_digest(mem)}"
        if key in seen:
            continue
        seen.add(key)
        rec = _example_record(world, mem, regime, subject, goal_kind,
                              relation, "demo", f"dem{len(rows):04d}")
        if remaining[rec["gold"]] <= 0:             # type: ignore[index]
            continue
        remaining[rec["gold"]] -= 1                  # type: ignore[index]
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _canonical_bytes(payload: Dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def dataset_sha(payload: Dict[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(
        {"train": payload["train"], "holdout": payload["holdout"]}
    )).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="NAV dataset builder (frozen)")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "artifacts"))
    args = ap.parse_args()

    payload = build_dataset()
    os.makedirs(args.out_dir, exist_ok=True)
    digest = dataset_sha(payload)
    payload["stats"]["dataset_sha256_16"] = digest[:16]

    ds_path = os.path.join(args.out_dir, "nav_dataset.json")
    with open(ds_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, default=str)
    manifest = dict(payload["stats"])              # type: ignore[arg-type]
    manifest["dataset_sha256_16"] = digest[:16]
    manifest["dataset_sha256"] = digest
    mf_path = os.path.join(args.out_dir, "nav_dataset_manifest.json")
    with open(mf_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, default=str)
        fh.write("\n")

    st = payload["stats"]                          # type: ignore[index]
    print(f"dataset built in {st['build_seconds']}s  sha16={digest[:16]}")
    print(f"  train    n={st['n_train']}  {st['train_class_counts']}")
    print(f"  holdout  n={st['n_holdout']}  {st['holdout_class_counts']}")
    print(f"  regimes(train)={st['regime_distribution_train']}")
    print(f"  worlds: train seeds {len(st['train_world_seeds'])} "
          f"(n_chains={st['train_n_chains']}), holdout seeds "
          f"{len(st['holdout_world_seeds'])} (n_chains={st['holdout_n_chains']})")
    print(f"  dataset -> {ds_path}")
    print(f"  manifest -> {mf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
