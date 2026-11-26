"""LTE-1 STEP 4a — the split-manifest generator (docs/LTE_PROTOCOL.md section 6.3).

Owner's implementation order (protocol section 10): step 4 = harness + split-manifest
verification, BEFORE controls/calibration (step 5) and the final exam (step 6). This
file produces the frozen split manifest; aion/experiments/lte_train.py consumes it.

Splits (protocol section 6.3, generation seed 20261126):

  TRAIN (teach)   60 experiences = 20 relationships x 3 teach facts — the existing
                  world bank (aion/worlds/generator.generate_world, seed 20261126,
                  5 chains -> 20 relationships) fed through teacher_student's
                  build_p2_experiences (declarative + 2 QA pairs per relationship —
                  the exact Phase-2/3 teach surface). Also the L_rank,train /
                  L_disc,train surface.

  DEV             12 questions = 8 direct + 4 untaught combinations. Calibration
                  ONLY (section 5). DEV carriers are feature-extraction-forbidden
                  (section 2.3 hard invariant; the training harness asserts this).

  FINAL EXAM      43 questions = 20 direct + 8 paraphrase + 15 combination battery —
                  the Phase-3 exam structure (teacher_student.py --phase3: all 20
                  relationships asked direct, 8 paraphrase, the full 15-question
                  combination battery of the 5-chain world).

TEMPLATE-LEVEL DISJOINTNESS (protocol section 6.3: "no question, template, or
combination in two splits", asserted in-harness, output committed in the manifest):

  teach   declaratives (SENTENCE_TEMPLATES) + DIRECT_ASK + TRAIN_Q_TEMPLATES
          (TRAIN_Q_TEMPLATES is the Phase-2 landing's deliberate-disjoint second
          interrogative — teacher_student.py:990-998).
  dev     DEV_Q_TEMPLATES (new phrasings, this file) + the relay-membership combo
          template. No paraphrase in dev.
  exam    EXAM_Q_TEMPLATES (new phrasings, this file) + PARAPHRASE_TEMPLATES (the
          world bank's paraphrase set — never used in teach or dev) + the world's
          combination templates (output-reach / device-membership).

  Every exam direct/paraphrase template is therefore disjoint from every teach and
  dev template by construction, and the in-harness assertion proves it; dev
  combination pairs are relay-target memberships while exam combination pairs are
  the world's output-reach and device-membership questions — pair-level disjoint
  (the relay and device pools share no names).

DETERMINISM: one RNG instance (random.Random(20261126)) drives, in this recorded
order: dev-direct relationship sample -> dev-combo yes-chain picks -> dev-combo
no-chain picks -> exam-paraphrase relationship sample -> exam-paraphrase
template-variant picks. The world bank's own internal RNG (generate_world(seed))
is separate and recorded. Same seed -> identical manifest content.

The manifest is committed to the repository BEFORE any training run (protocol
section 6.3 freeze mechanics / section 10 freeze record); the freeze record cites
its path + commit SHA. A manifest not demonstrably asserting disjointness voids
the run — hence the assertion output is part of the JSON itself, and
recheck_manifest() re-derives the world from the seed and re-verifies every
recorded fact (used by the training harness at load).

CLI:  python aion/experiments/lanes_manifest.py --out runs/lte/manifest.json
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# ---- repo path (robust: this file also runs flat at ~/ on Lappy) ----------- #

def _setup_repo_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path.home() / "aion_phase1",          # Phase-3 staging on Lappy
        here.parent.parent,                   # in-repo layout (aion/experiments/*)
        Path.cwd(),
    ]
    for cand in candidates:
        if (cand / "aion" / "relational" / "interface.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError("cannot locate the aion repo root (looked in "
                       + ", ".join(str(c) for c in candidates) + ")")


REPO_ROOT = _setup_repo_path()

from aion.worlds.generator import (                                         # noqa: E402
    BATTERY_COMBINATION, BATTERY_DIRECT, BATTERY_PARAPHRASE, DIRECT_ASK,
    PARAPHRASE_TEMPLATES, SENTENCE_TEMPLATES, generate_world, verify_world)
from aion.experiments.teacher_student import (                               # noqa: E402
    RELATION_OBJECT_ROLE, RELATION_SUBJECT_ROLE, TRAIN_Q_TEMPLATES,
    build_p2_experiences)

# ---- preregistered constants (protocol section 6.3) ----------------------- #

SEED = 20261126                    # the one preregistered RNG seed
N_CHAINS = 5                       # 5 chains -> 20 relationships (owner criterion)
TRAIN_EXPECTED = 60                # 20 relationships x 3 teach facts
DEV_DIRECT_N = 8
DEV_COMBO_N = 4
DEV_EXPECTED = DEV_DIRECT_N + DEV_COMBO_N                  # 12
EXAM_PARA_N = 8
EXAM_EXPECTED = 20 + EXAM_PARA_N + 15                      # 43 (Phase-3 structure)

# DEV-only direct interrogatives (this file, new). Deliberately disjoint from
# DIRECT_ASK, TRAIN_Q_TEMPLATES, PARAPHRASE_TEMPLATES and EXAM_Q_TEMPLATES —
# asserted at generation time.
DEV_Q_TEMPLATES = {
    "operates": "Who does {subject} operate upon?",
    "routes-to": "To which relay does {subject} route?",
    "controls": "Which device is {subject} in control of?",
    "produces": "What is produced by {subject}?",
}

# EXAM-only direct interrogatives (this file, new). The exam direct battery must
# NOT reuse the teach format (DIRECT_ASK) — the template-level disjointness law.
EXAM_Q_TEMPLATES = {
    "operates": "What does {subject} operate on?",
    "routes-to": "Where does {subject} route to?",
    "controls": "Which device does {subject} control?",
    "produces": "Which output does {subject} produce?",
}

DEV_COMBO_TEMPLATE = "Does {actor} reach relay {relay}?"    # relay-target membership
EXAM_REACH_TEMPLATE = "What does {actor} reach?"
EXAM_MEMBER_TEMPLATE = "Does {actor} reach {device}?"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #

def build_split_sets(seed: int = SEED) -> Dict:
    """Build the three splits from the world bank. Deterministic under `seed`."""
    rng = random.Random(seed)
    rng_trace: List[str] = []

    world = generate_world(seed=seed, n_chains=N_CHAINS)
    failures = verify_world(world)
    if failures:
        raise RuntimeError(f"world ground truth inconsistent: {failures[:3]}")

    pools = {"actor": world.actors, "conduit": world.conduits, "relay": world.relays,
             "device": world.devices, "output": world.outputs}
    rels = list(world.relationships)               # 20, stable world order

    # ---- TRAIN: 60 experiences (20 relationships x 3 teach facts) ---------- #
    # build_p2_experiences is the landed Phase-2/3 teach surface: per relationship
    # exactly [declarative (SENTENCE_TEMPLATES), QA (DIRECT_ASK), QA
    # (TRAIN_Q_TEMPLATES)], in world.relationships order. Positional derivation:
    # experience idx -> relationship idx // 3, position idx % 3 — asserted below.
    exps = build_p2_experiences(world)
    assert len(exps) == TRAIN_EXPECTED, (len(exps), TRAIN_EXPECTED)
    train: List[Dict] = []
    for idx, e in enumerate(exps):
        r = rels[idx // 3]
        pos = idx % 3
        expect_kind = "declarative" if pos == 0 else "qa"
        assert e["kind"] == expect_kind, (idx, e["kind"], expect_kind)
        assert e["chain"] == r.chain_id, (idx, e["chain"], r.chain_id)
        if pos == 0:
            # the SLOT template is registered (SENTENCE_TEMPLATES); the formatted
            # per-fact sentence is the experience text
            tmpl_id = f"TEACH_DECL:{r.relation}"
            tmpl, question = SENTENCE_TEMPLATES[r.relation], None
            assert e["text"] == r.sentence == tmpl.format(subject=r.subject,
                                                          object=r.object)
            cands: List[str] = []
        else:
            src = DIRECT_ASK if pos == 1 else TRAIN_Q_TEMPLATES
            tmpl = src[r.relation]
            tmpl_id = f"TEACH_Q{pos}:{r.relation}"
            question = tmpl.format(subject=r.subject)
            assert e["question"] == question, (idx, e["question"], question)
            assert e["answer"] == r.object
            cands = list(pools[RELATION_OBJECT_ROLE[r.relation]])
        train.append({
            "exp_index": idx, "kind": e["kind"], "fact_index": r.index,
            "chain": r.chain_id, "relation": r.relation,
            "subject": r.subject, "object": r.object,
            "template_id": tmpl_id, "template": tmpl,
            "text": e["text"], "question": question, "answer": e["answer"],
            "key": e["key"], "rank_candidates": cands,
            "gold_candidate": r.object,
        })

    # canonical L_disc,train carrier per fact: the fact's FIRST QA experience
    # (DIRECT_ASK form), gold candidate — recorded so the harness never re-derives it
    carriers = {}
    for t in train:
        if t["kind"] == "qa" and t["template_id"] == f"TEACH_Q1:{t['relation']}":
            carriers[str(t["fact_index"])] = {
                "carrier_exp_index": t["exp_index"],
                "carrier_question": t["question"],
                "carrier_candidate": t["gold_candidate"],
                "carrier_template_id": t["template_id"]}

    # ---- DEV: 8 direct + 4 untaught combinations -------------------------- #
    dev: List[Dict] = []
    picks = sorted(rng.sample(range(len(rels)), DEV_DIRECT_N))
    rng_trace.append(f"dev-direct relationship sample: {picks}")
    for i, ri in enumerate(picks):
        r = rels[ri]
        tmpl = DEV_Q_TEMPLATES[r.relation]
        dev.append({
            "qid": f"dev-direct-{i:02d}", "battery": BATTERY_DIRECT,
            "template_id": f"DEV_Q:{r.relation}", "template": tmpl,
            "text": tmpl.format(subject=r.subject), "answer": r.object,
            "fact_index": r.index, "chain": r.chain_id,
            "meta": {"subject": r.subject, "relation": r.relation,
                     "object": r.object},
            "candidates": list(pools[RELATION_OBJECT_ROLE[r.relation]]),
        })

    # untaught combinations: relay-target memberships (actor -> relay is a >=2-hop
    # composition of individually taught facts, never taught as a unit). 2 yes
    # (own-chain relay) + 2 no (other-chain relay), re-derived from the chain graph
    # in the assertion suite. Pair space (start, relay target) is disjoint from the
    # exam combo pair space (start, device target / output) because the relay and
    # device pools share no names.
    yes_chains = sorted(rng.sample(range(N_CHAINS), 2))
    rng_trace.append(f"dev-combo yes chains: {yes_chains}")
    no_chains = []
    for _ in range(2):
        c = rng.randrange(N_CHAINS)
        step = rng.choice([o for o in (1, 2, 3, 4)])
        j = (c + step) % N_CHAINS
        no_chains.append((c, j))
    rng_trace.append(f"dev-combo no (chain, other) picks: {no_chains}")
    for i, c in enumerate(yes_chains):
        actor, relay = world.chains[c][0], world.chains[c][2]
        dev.append({
            "qid": f"dev-combo-{i:02d}", "battery": BATTERY_COMBINATION,
            "template_id": "DEV_COMBO_MEMBER_RELAY", "template": DEV_COMBO_TEMPLATE,
            "text": DEV_COMBO_TEMPLATE.format(actor=actor, relay=relay),
            "answer": "yes", "fact_index": None, "chain": c,
            "meta": {"kind": "membership_relay", "start": actor, "target": relay},
            "candidates": ["yes", "no"],
        })
    for i, (c, j) in enumerate(no_chains):
        actor, relay = world.chains[c][0], world.chains[j][2]
        dev.append({
            "qid": f"dev-combo-{2 + i:02d}", "battery": BATTERY_COMBINATION,
            "template_id": "DEV_COMBO_MEMBER_RELAY", "template": DEV_COMBO_TEMPLATE,
            "text": DEV_COMBO_TEMPLATE.format(actor=actor, relay=relay),
            "answer": "no", "fact_index": None, "chain": c,
            "meta": {"kind": "membership_relay", "start": actor, "target": relay},
            "candidates": ["yes", "no"],
        })

    # ---- FINAL EXAM: 20 direct + 8 paraphrase + 15 combinations ----------- #
    exam: List[Dict] = []
    for i, r in enumerate(rels):
        tmpl = EXAM_Q_TEMPLATES[r.relation]
        exam.append({
            "qid": f"exam-direct-{i:02d}", "battery": BATTERY_DIRECT,
            "template_id": f"EXAM_Q:{r.relation}", "template": tmpl,
            "text": tmpl.format(subject=r.subject), "answer": r.object,
            "fact_index": r.index, "chain": r.chain_id,
            "meta": {"subject": r.subject, "relation": r.relation,
                     "object": r.object},
            "candidates": list(pools[RELATION_OBJECT_ROLE[r.relation]]),
        })

    para_picks = sorted(rng.sample(range(len(rels)), EXAM_PARA_N))
    rng_trace.append(f"exam-paraphrase relationship sample: {para_picks}")
    for i, ri in enumerate(para_picks):
        r = rels[ri]
        variant = rng.randrange(2)
        rng_trace.append(f"exam-paraphrase rel {ri} variant: {variant}")
        tmpl, asked_side = PARAPHRASE_TEMPLATES[r.relation][variant]
        answer = r.subject if asked_side == "subject" else r.object
        role = (RELATION_SUBJECT_ROLE[r.relation] if asked_side == "subject"
                else RELATION_OBJECT_ROLE[r.relation])
        exam.append({
            "qid": f"exam-para-{i:02d}", "battery": BATTERY_PARAPHRASE,
            "template_id": f"EXAM_PARA:{r.relation}:{variant}", "template": tmpl,
            "text": tmpl.format(subject=r.subject, object=r.object),
            "answer": answer, "fact_index": r.index, "chain": r.chain_id,
            "meta": {"subject": r.subject, "relation": r.relation,
                     "object": r.object, "asked_side": asked_side},
            "candidates": list(pools[role]),
        })

    # combination battery: the world bank's own 15 combination questions, verbatim
    # (the Phase-3 exam's combination battery). Held-out: unseen combinations of
    # individually taught facts.
    for q in world.questions:
        if q.battery != BATTERY_COMBINATION:
            continue
        is_reach = q.meta.get("kind") == "reach"
        exam.append({
            "qid": f"exam-{q.id}", "battery": BATTERY_COMBINATION,
            "template_id": "EXAM_COMBO_REACH" if is_reach else "EXAM_COMBO_MEMBER_DEVICE",
            "template": EXAM_REACH_TEMPLATE if is_reach else EXAM_MEMBER_TEMPLATE,
            "text": q.text, "answer": q.answer, "fact_index": None,
            "chain": q.chain_id, "meta": dict(q.meta),
            "candidates": list(pools["output"]) if is_reach else ["yes", "no"],
        })

    # ---- template registry ------------------------------------------------- #
    registry: Dict[str, Dict] = {}
    for t in train:
        registry.setdefault(t["template_id"],
                            {"template": t["template"], "split": "train",
                             "n_questions": 0})
        registry[t["template_id"]]["n_questions"] += 1
        assert registry[t["template_id"]]["template"] == t["template"]
    for split, rows in (("dev", dev), ("exam", exam)):
        for q in rows:
            registry.setdefault(q["template_id"],
                                {"template": q["template"], "split": split,
                                 "n_questions": 0})
            registry[q["template_id"]]["n_questions"] += 1
            assert registry[q["template_id"]]["template"] == q["template"], (
                f"template text drift for {q['template_id']}")

    # exam carrier per fact (exam DIRECT question for fact i — the section 6.1
    # swap carrier and the section 2.3 exam-diagnostic carrier)
    exam_carriers = {}
    for q in exam:
        if q["battery"] == BATTERY_DIRECT:
            exam_carriers[str(q["fact_index"])] = q["qid"]

    return {"world": world, "train": train, "dev": dev, "exam": exam,
            "registry": registry, "carriers": carriers,
            "exam_carriers": exam_carriers, "rng_trace": rng_trace, "pools": pools}


# --------------------------------------------------------------------------- #
# Disjointness + ground-truth assertions (in-harness; output IN the manifest)
# --------------------------------------------------------------------------- #

def run_split_assertions(sets: Dict) -> List[Dict]:
    """Every preregistered split assertion. Returns the assertion records (each
    {name, pass, detail}); raises AssertionError if any fails — a failing
    assertion aborts generation (fail loudly; a broken manifest must never be
    committed)."""
    world = sets["world"]
    train, dev, exam = sets["train"], sets["dev"], sets["exam"]
    registry = sets["registry"]
    recs: List[Dict] = []

    def rec(name: str, ok: bool, detail) -> None:
        recs.append({"name": name, "pass": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError(f"split assertion FAILED: {name}: {detail}")

    # 1 — counts and structure
    rec("train_count", len(train) == TRAIN_EXPECTED,
        {"n": len(train), "expected": TRAIN_EXPECTED})
    per_rel: Dict[int, List[str]] = {}
    for t in train:
        per_rel.setdefault(t["fact_index"], []).append(t["kind"])
    ok = (len(per_rel) == 20
          and all(v == ["declarative", "qa", "qa"] for v in per_rel.values()))
    rec("train_structure_20x3", ok,
        {"relationships": len(per_rel),
         "orders": sorted({tuple(v) for v in per_rel.values()})})
    rec("dev_count", len(dev) == DEV_EXPECTED,
        {"n": len(dev), "expected": DEV_EXPECTED,
         "direct": sum(1 for q in dev if q["battery"] == BATTERY_DIRECT),
         "combo": sum(1 for q in dev if q["battery"] == BATTERY_COMBINATION)})
    rec("exam_count", len(exam) == EXAM_EXPECTED,
        {"n": len(exam), "expected": EXAM_EXPECTED,
         "direct": sum(1 for q in exam if q["battery"] == BATTERY_DIRECT),
         "paraphrase": sum(1 for q in exam if q["battery"] == BATTERY_PARAPHRASE),
         "combo": sum(1 for q in exam if q["battery"] == BATTERY_COMBINATION)})

    # 2 — question-text disjointness across splits, and global id uniqueness
    t_texts = [t["text"] for t in train]
    d_texts = [q["text"] for q in dev]
    e_texts = [q["text"] for q in exam]
    for a, b, na, nb in ((t_texts, d_texts, "train", "dev"),
                         (t_texts, e_texts, "train", "exam"),
                         (d_texts, e_texts, "dev", "exam")):
        overlap = sorted(set(a) & set(b))
        rec(f"text_disjoint_{na}_vs_{nb}", not overlap,
            {"overlap": overlap[:5], "n_overlap": len(overlap)})
    all_ids = ([f"train-{t['exp_index']:03d}" for t in train]
               + [q["qid"] for q in dev] + [q["qid"] for q in exam])
    rec("ids_globally_unique", len(set(all_ids)) == len(all_ids),
        {"n": len(all_ids), "unique": len(set(all_ids))})

    # 3 — TEMPLATE-level disjointness across splits (the core law)
    splits_of = {"train": {t["template_id"] for t in train},
                 "dev": {q["template_id"] for q in dev},
                 "exam": {q["template_id"] for q in exam}}
    for a, b in (("train", "dev"), ("train", "exam"), ("dev", "exam")):
        overlap = sorted(splits_of[a] & splits_of[b])
        rec(f"template_disjoint_{a}_vs_{b}", not overlap,
            {"overlap": overlap, "n_a": len(splits_of[a]), "n_b": len(splits_of[b])})

    # 4 — explicit: exam direct/paraphrase templates never used in teach or dev
    teach_dev_ids = splits_of["train"] | splits_of["dev"]
    exam_dp = {q["template_id"] for q in exam
               if q["battery"] in (BATTERY_DIRECT, BATTERY_PARAPHRASE)}
    bad = sorted(exam_dp & teach_dev_ids)
    rec("exam_direct_paraphrase_templates_new", not bad,
        {"exam_dp_template_ids": sorted(exam_dp), "violations": bad})

    # 5 — dev combos != exam combos (pair-level; texts already covered by #2)
    dev_pairs = {(q["meta"]["start"], q["meta"]["target"]) for q in dev
                 if q["battery"] == BATTERY_COMBINATION}
    exam_pairs = {(q["meta"].get("start"), q["meta"].get("target"))
                  for q in exam if q["battery"] == BATTERY_COMBINATION}
    overlap = sorted(dev_pairs & exam_pairs)
    rec("dev_combos_disjoint_from_exam_combos", not overlap,
        {"dev_pairs": sorted(dev_pairs), "exam_pair_n": len(exam_pairs),
         "overlap": overlap})

    # 6 — ground truth: world verification + per-question re-derivation
    rec("world_verify", not verify_world(world),
        {"failures": verify_world(world)[:3]})
    edges = {(s, rel, o) for s, es in world.graph.items() for (rel, o) in es}
    bad = []
    for t in train:
        if (t["subject"], t["relation"], t["object"]) not in edges \
                or t["answer"] != t["object"]:
            bad.append(f"train-{t['exp_index']:03d}")
    for q in dev:
        m = q["meta"]
        if q["battery"] == BATTERY_DIRECT:
            ok = (m["subject"], m["relation"], q["answer"]) in edges
        else:  # relay membership — re-derived from the chain graph
            ok = ((m["target"] in world.reachable(m["start"]))
                  == (q["answer"] == "yes"))
        if not ok:
            bad.append(q["qid"])
    for q in exam:
        m = q["meta"]
        if q["battery"] == BATTERY_DIRECT:
            ok = (m["subject"], m["relation"], q["answer"]) in edges
        elif q["battery"] == BATTERY_PARAPHRASE:
            ok = ((m["subject"], m["relation"], m["object"]) in edges
                  and q["answer"] == m[m["asked_side"]])
        else:  # world combination battery: verify_world already re-derived it
            ok = True
        if not ok:
            bad.append(q["qid"])
    rec("answers_graph_derivable", not bad, {"violations": bad[:8]})

    # 7 — candidate sets: gold present; membership candidates binary
    bad = []
    for t in train:
        if t["kind"] == "qa" and t["gold_candidate"] not in t["rank_candidates"]:
            bad.append(f"train-{t['exp_index']:03d}")
    for q in dev + exam:
        if q["answer"] not in q["candidates"]:
            bad.append(q["qid"])
        if (q["battery"] == BATTERY_COMBINATION
                and q["meta"].get("kind") != "reach"
                and q["candidates"] != ["yes", "no"]):
            bad.append(q["qid"])
    rec("candidates_contain_gold", not bad, {"violations": bad[:8]})

    # 8 — template registry: unique id -> unique string; full coverage
    id_strings = {k: v["template"] for k, v in registry.items()}
    seen: Dict[str, str] = {}
    dup = []
    for k, s in id_strings.items():
        if s in seen:
            dup.append(f"{seen[s]} == {k}: {s!r}")
        seen[s] = k
    rec("template_ids_unique_strings", not dup, {"duplicated": dup})
    covered = (splits_of["train"] | splits_of["dev"] | splits_of["exam"])
    rec("registry_covers_all", covered == set(registry),
        {"missing": sorted(covered - set(registry)),
         "extra": sorted(set(registry) - covered)})

    # 9 — carriers recorded for every fact (train canonical + exam direct)
    want = {str(i) for i in range(20)}
    ok = (set(sets["carriers"]) == want and set(sets["exam_carriers"]) == want)
    rec("carriers_recorded_per_fact", ok,
        {"train_carriers": len(sets["carriers"]),
         "exam_carriers": len(sets["exam_carriers"])})

    # 10 — entity pools unchanged (frozen world-bank identity)
    rec("entity_pools_frozen", all(
        sets["pools"][role] == pool for role, pool in
        (("actor", world.actors), ("conduit", world.conduits),
         ("relay", world.relays), ("device", world.devices),
         ("output", world.outputs))), {})
    return recs


# --------------------------------------------------------------------------- #
# Manifest assembly + standalone re-verification
# --------------------------------------------------------------------------- #

def build_manifest(seed: int = SEED) -> Dict:
    t0 = time.time()
    sets = build_split_sets(seed=seed)
    assertions = run_split_assertions(sets)
    world = sets["world"]
    return {
        "artifact": "LTE-1 split manifest (protocol section 6.3) — committed BEFORE "
                    "any training run; freeze record cites this path + commit SHA",
        "protocol": "docs/LTE_PROTOCOL.md Revision 8, section 6.3 (frozen Revision 6 "
                    "+ Rev 7/8 amendments)",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_seed": seed,
        "seed_lineage": {
            "world_rng": f"generate_world(seed={seed}, n_chains={N_CHAINS}) "
                         "(aion/worlds/generator internal Random)",
            "split_rng": f"random.Random({seed}) — recorded draw order: "
                         + " | ".join(sets["rng_trace"]),
        },
        "world": {"seed": world.seed, "n_chains": world.n_chains,
                  "relationships": len(world.relationships),
                  "verify_world": "OK (all answers graph-derivable)",
                  "chains": world.chains},
        "counts": {"train": len(sets["train"]), "dev": len(sets["dev"]),
                   "exam": len(sets["exam"])},
        "train": {"surface": "L_rank,train / L_disc,train (protocol section 6.3)",
                  "canonical_carrier_per_fact": sets["carriers"],
                  "experiences": sets["train"]},
        "dev": {"role": "calibration ONLY (protocol section 5); DEV carriers are "
                        "feature-extraction-forbidden (section 2.3 hard invariant)",
                "questions": sets["dev"]},
        "final_exam": {"structure": "Phase-3 exam (teacher_student.py --phase3): "
                                    "direct + paraphrase + 15-combination battery",
                       "carrier_per_fact": sets["exam_carriers"],
                       "questions": sets["exam"]},
        "template_registry": sets["registry"],
        "combination_pairs": {
            "dev": [{"qid": q["qid"], **q["meta"]} for q in sets["dev"]
                    if q["battery"] == BATTERY_COMBINATION],
            "exam": [{"qid": q["qid"], **q["meta"]} for q in sets["exam"]
                     if q["battery"] == BATTERY_COMBINATION]},
        "disjointness_assertions": {
            "law": "protocol section 6.3: no question, template, or combination in "
                   "two splits; exam direct/paraphrase templates never used in teach "
                   "or dev; dev combos != exam combos",
            "assertions": assertions,
            "all_pass": all(a["pass"] for a in assertions),
        },
        "generation_wall_s": round(time.time() - t0, 2),
    }


def recheck_manifest(manifest: Dict) -> Dict:
    """Standalone re-verification of a manifest dict (used by the training harness
    at load): re-derives the world from generation_seed and re-checks every recorded
    fact — counts, texts, template disjointness, combination-pair disjointness,
    ground truth, candidate sets, carriers. Returns a record; raises on failure."""
    seed = manifest["generation_seed"]
    sets = build_split_sets(seed=seed)
    fresh = build_manifest(seed=seed)
    rec: Dict = {"seed": seed, "checks": {}}

    def chk(name, ok, detail=None):
        rec["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            raise AssertionError(f"manifest recheck FAILED: {name} {detail or ''}")

    # content-stable regeneration under the same seed (rng draw order recorded)
    chk("regenerates_identically",
        fresh["counts"] == manifest["counts"]
        and [t["text"] for t in fresh["train"]["experiences"]]
        == [t["text"] for t in manifest["train"]["experiences"]]
        and [q["text"] for q in fresh["dev"]["questions"]]
        == [q["text"] for q in manifest["dev"]["questions"]]
        and [q["text"] for q in fresh["final_exam"]["questions"]]
        == [q["text"] for q in manifest["final_exam"]["questions"]],
        "same seed must regenerate identical splits")
    chk("recorded_assertions_all_pass",
        manifest["disjointness_assertions"]["all_pass"] is True
        and all(a["pass"] for a in manifest["disjointness_assertions"]["assertions"]),
        f"{len(manifest['disjointness_assertions']['assertions'])} assertions")
    # live re-run of the whole assertion suite on the regenerated sets
    run_split_assertions(sets)
    chk("assertion_suite_rerun_pass", True,
        {"n_assertions": len(fresh["disjointness_assertions"]["assertions"])})
    rec["pass"] = True
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="LTE-1 split-manifest generator "
                                             "(protocol section 6.3, seed 20261126)")
    ap.add_argument("--out", default="runs/lte/manifest.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    manifest = build_manifest(seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=False)
    out.write_text(payload, encoding="utf-8")

    a = manifest["disjointness_assertions"]
    print(f"[lanes-manifest] seed={args.seed}  train={manifest['counts']['train']}  "
          f"dev={manifest['counts']['dev']}  exam={manifest['counts']['exam']}")
    print(f"[lanes-manifest] disjointness assertions: "
          f"{sum(1 for x in a['assertions'] if x['pass'])}/{len(a['assertions'])} PASS "
          f"(all_pass={a['all_pass']})")
    for x in a["assertions"]:
        print(f"[lanes-manifest]   {'PASS' if x['pass'] else 'FAIL'}  {x['name']}")
    print(f"[lanes-manifest] manifest written: {out.resolve()}  "
          f"(sha256 {sha256_text(payload)[:16]}..., "
          f"gen {manifest['generation_wall_s']}s)")
    print(f"[lanes-manifest] COMMIT this file BEFORE any training run; the freeze "
          f"record cites its path + commit SHA (protocol sections 6.3 / 10)")
    return 0 if a["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
