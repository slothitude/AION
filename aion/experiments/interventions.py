"""AION-1 INTERVENTION EXPERIMENT — can AION predict the consequence of a change to its
learned world?  ("Tav is rerouted to relay Drel — what does Neral reach now?")

The reroute is applied to the WORLD GRAPH (gold re-derived), presented to the plastic
pathway as experience (the PROVEN Phase-1 plastic-override path), and the reach question
is then scored by CONSTRAINED SCORING (Phase-2 pattern). The headline is the
WITH-PLASTIC vs WITHOUT-PLASTIC ablation per reroute: does the plastic state change
predictions in the right direction?

===============================================================================
PREREGISTRATION — FROZEN BEFORE THE RUN (2026-11-25).  Nothing below is chosen
after seeing data.  One configuration, frozen, reported; no mid-run tuning.
===============================================================================

  WORLD          generate_world(seed=0, n_chains=5) — the PROVEN Phase-1 configuration
                 (5 chains / 20 relationships / 65 questions). verify_world() must
                 return [] (every battery answer graph-derivable) or the run aborts
                 before any learning.

  TEACH          all 20 relationships via the PROVEN Phase-1 teach loop:
                 DialogueSession.teach (ScriptedTeacher, reference answers = world gold;
                 lesson delivery is scripted — stated, never hidden; the plastic signal
                 is IDENTICAL to the Phase-1 endpoint run because the extractor consumes
                 lesson_text either way) -> RuleExtractor -> Student.learn_fact ->
                 pathway.nnfs_update per fact (+ gate steps per the Phase-1 config).
                 Knobs match the PROVEN real run (runs/teacher_student/qwen-real-c5)
                 EXACTLY: nnfs learning_rate=0.5, gate_steps=6 per fact, gate_lr=300,
                 answer_tokens=6, correction_repeats=3. Every fact: plastic update +
                 RelationalMemory registration (probe = canonical sentence) + episodic
                 record. Acquisition evidence: all 20 fact plastic strengths > 1e-8
                 and u_fast norms moved from 0.

  BASELINE       epoch 0 (post-teach, pre-intervention): the STABILITY SET — one plain
                 reach question per actor ("What does {actor} reach?", 5 questions) —
                 scored by constrained scoring against G_0 gold.

  REROUTE SET    8 reroutes, frozen, SEQUENTIAL and CUMULATIVE (each statement is
                 reality moving forward; gold is re-derived on the ACCUMULATED graph):
                   reroutes 1..5 : conduits[i] -> relays[(i+1) % 5], i = 0..4
                   reroutes 6..8 : conduits[i] -> relays[(i+3) % 5], i = 0, 2, 4
                 Every reroute's new_relay differs from the conduit's CURRENT relay at
                 that moment (checked at plan time and at apply time). Statement text is
                 exactly "{conduit} is rerouted to relay {new_relay}." — the shape
                 RuleExtractor parses as an intervention.

  INTERVENE      per reroute: fresh DialogueSession.teach(statement) ->
                 CorrectionHandler.apply_reroute -> Student.learn_fact(fact, repeats=3)
                 -> pathway.nnfs_update (+ gate steps) — the EXACT Phase-1 RETAIN
                 correction path (the pathway therefore receives the rendered
                 "Tav routes to Drel." fact, x3; the raw statement lives in the
                 transcript). New fact registered in RelationalMemory (probe = raw
                 statement), old key marked superseded_by, episodic record kind=
                 "correction". G_k = apply_reroute(G_{k-1}, conduit, new_relay).

  QUESTIONS      per reroute k (asked AFTER the plastic update, measurement mode —
                 no plastic writes during any scoring):
                   reach_now : "What does {actor} reach now?"   gold = terminal(A, G_k)
                   route_now : "Which relay does {actor} route through now?"
                                                             gold = the conduit's new relay
                 plus the STABILITY SET re-scored every epoch (0..8) against ITS OWN
                 epoch's gold — did OLD reach answers correctly change?  (affected
                 actor: gold changes -> a correct system should flip its answer;
                 unaffected actors: gold unchanged -> answers should stay.)

  SCORING        CONSTRAINED SCORING (the Phase-2 pattern): P(c|q) ∝ exp(score), score =
                 sum of log-probs of the candidate's tokens at the answer position under
                 the FIXED Phase-2 two-exemplar scaffold (FORMAT_SCAFFOLD, identical text
                 for every mode and question). One forward per candidate. Candidate pool
                 = role-matched world-entity pools (outputs for reach questions, relays
                 for route questions; 5 candidates each -> chance = 0.2).
                 Every question answered once per PROVENANCE BUCKET by ablating one
                 bucket at a time (aion/experiments/retention.py ablated()):
                   WITH plastic    PLASTIC (u_fast alone) and FULL (natural blend)
                   WITHOUT plastic CONSOLIDATED (theta_slow alone — theta stays empty in
                                   this run, so this IS the frozen base) and RETRIEVED
                                   (pathway ablated, episodic lookup only)
                 exact match = argmax candidate == gold; gold_prob and gold_rank recorded.

  THE HEADLINE   per reroute and aggregate: PLASTIC accuracy on intervention questions
                 vs CONSOLIDATED accuracy — the plastic ablation delta. Positive delta =
                 the plastic state (which received the intervention experience) moves
                 predictions toward the new gold; zero delta = the representation reaches
                 u_fast (strengths prove it) but does not discriminate at the answer
                 position.

  CONTROLS       heuristic graph-lookup answers the SAME questions from the CURRENT graph
                 (pure graph logic, no model) — the ceiling, expected 1.0 by construction;
                 frozen-chance = 1/5 = 0.2 reported alongside; CONSOLIDATED accuracy is
                 the frozen-base control.

  STRENGTHS      per reroute: new fact plastic strength (blend=1.0) and old fact strength
                 after; u_fast norms before/after; old-facts-present count and mean/max
                 relative strength change over the other indexed facts.

  HONEST EXPECTATION (pre-registered, from Phase 2): expression through the frozen Base
                 decode may sit near chance on intervention questions (Phase 2: combination
                 acc@1 0.333 vs chance 0.400; free-form exact 0.071; plastic ablation delta
                 0.0). If so, that IS the finding, reported plainly with the same
                 with/without-plastic ablation evidence. No prompt/knob tuning mid-exam.

  IO             runs/interventions/<tag>/: results.jsonl appended INCREMENTALLY (one line
                 per scored question / per reroute — never batch-all-at-end), report.json +
                 world.json at the end, nohup log for the real run.

Run:
  python aion/experiments/interventions.py                      # tiny stand-in (Rog CPU debug)
  python aion/experiments/interventions.py --foundation qwen \
      --local-dir /mnt/seagate/aion-models/Qwen3.5-4B-Base      # REAL run (Lappy, CPU BF16)
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.dialogue.conversation import DialogueSession, default_grader        # noqa: E402
from aion.dialogue.correction import CorrectionHandler                        # noqa: E402
from aion.dialogue.lesson_extractor import RuleExtractor                      # noqa: E402
from aion.dialogue.student import Student, StudentConfig                      # noqa: E402
from aion.dialogue.teacher import ScriptedTeacher                             # noqa: E402
from aion.worlds.generator import (                                           # noqa: E402
    World, apply_reroute, generate_world, verify_world)
from aion.memory.episodic import EpisodicStore                                # noqa: E402
from aion.memory.relational import RelationalMemory, relation_key             # noqa: E402
from aion.experiments.retention import (                                      # noqa: E402
    ablated, CONSOLIDATED, FULL, MODES, PLASTIC, RETRIEVED)
from aion.experiments.teacher_student import (                                # noqa: E402  (read-only patterns)
    FORMAT_SCAFFOLD, build_tiny_foundation, contains_match, lenient_match)
from aion.relational.interface import (                                       # noqa: E402
    InterfaceConfig, RelationalInterface, fingerprints_equal, foundation_fingerprint)
from aion.foundation.pretrained_loader import DEFAULT_LOCAL_DIR, load_frozen_base  # noqa: E402

# --------------------------------------------------------------------------- #
# PREREGISTERED constants (frozen before the run — see module docstring)
# --------------------------------------------------------------------------- #

INTERVENTION_STATEMENT = "{conduit} is rerouted to relay {new_relay}."
SECOND_PASS_INDICES = (0, 2, 4)          # conduits re-rerouted by reroutes 6..8
SECOND_PASS_OFFSET = 3
FIRST_PASS_OFFSET = 1
STRENGTH_THRESHOLD = 1e-8
N_REROUTES = 8


# --------------------------------------------------------------------------- #
# The frozen reroute plan + gold derivation
# --------------------------------------------------------------------------- #

def reroute_plan(world: World, n_reroutes: int = N_REROUTES) -> List[Dict]:
    """The frozen reroute set: first pass offsets every conduit by +1, second pass
    re-reroutes conduits (0, 2, 4) by +3. Deterministic; every new_relay differs from
    the conduit's CURRENT relay at that point in the (cumulative) sequence."""
    n = world.n_chains
    current: Dict[str, str] = {c: world.relays[i] for i, c in enumerate(world.conduits)}
    plan: List[Dict] = []
    order = list(range(n)) + list(SECOND_PASS_INDICES)
    for k, i in enumerate(order[:n_reroutes]):
        offset = FIRST_PASS_OFFSET if k < n else SECOND_PASS_OFFSET
        conduit = world.conduits[i]
        new_relay = world.relays[(i + offset) % n]
        assert new_relay != current[conduit], f"reroute {k} would not change the edge"
        plan.append({"index": k, "conduit": conduit, "new_relay": new_relay,
                     "old_relay": current[conduit],
                     "actor": world.actors[i], "chain_id": i})
        current[conduit] = new_relay
    return plan


def apply_plan(plan: List[Dict], graph: Dict) -> List[Dict]:
    """Walk the plan cumulatively: G_0 -> G_1 -> ... -> G_k (each entry carries its golds)."""
    out = []
    g = graph
    for item in plan:
        old_gold = reach_gold(item["actor"], g)
        g = apply_reroute(g, item["conduit"], item["new_relay"])
        out.append({**item, "gold_reach_pre": old_gold,
                    "gold_reach": reach_gold(item["actor"], g),
                    "gold_route": item["new_relay"], "graph": g})
    return out


def reach_gold(actor: str, graph: Dict) -> str:
    """Ground-truth reach for one actor on one graph (the walk's terminal output)."""
    node, seen = actor, {actor}
    while node in graph:
        _, nxt = graph[node][0]
        if nxt in seen:
            break
        seen.add(nxt)
        node = nxt
    return node          # the chain walk ends on the OUTPUT entity


def route_gold(actor: str, graph: Dict) -> str:
    """Ground-truth relay an actor routes through on one graph (walk position 2)."""
    path = [actor]
    node = actor
    while node in graph and len(path) < 3:
        _, nxt = graph[node][0]
        path.append(nxt)
        node = nxt
    return path[2] if len(path) > 2 else ""


def heuristic_answer(question: str, actors: List[str], graph: Dict) -> str:
    """CONTROL: pure graph logic, no model. reach / route questions answered by walking
    the CURRENT graph; anything else returns '' (never scored)."""
    for a in actors:
        if question in (f"What does {a} reach?", f"What does {a} reach now?"):
            return reach_gold(a, graph)
        if question == f"Which relay does {a} route through now?":
            return route_gold(a, graph)
    return ""


# --------------------------------------------------------------------------- #
# Tiny-standin foundation (Rog CPU debug lane; identical to the Phase-1 stub)
# --------------------------------------------------------------------------- #

def tiny_foundation(seed: int = 0):
    return build_tiny_foundation(seed=seed)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

class InterventionRun:
    """TEACH -> BASELINE -> INTERVENE(score per reroute) -> REPORT, one frozen config."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tok = None
        self.iface: Optional[RelationalInterface] = None
        self.student: Optional[Student] = None
        self.episodic = EpisodicStore(clock=lambda: 0.0)      # deterministic law
        self.rmem: Optional[RelationalMemory] = None
        self.teacher: Optional[ScriptedTeacher] = None
        self.corrections = CorrectionHandler()
        self.world: Optional[World] = None
        self.plan: List[Dict] = []
        self.fingerprint_at_load: Optional[Dict] = None
        self.foundation_label = ""
        self.timings: Dict[str, float] = {}
        self.run_tag = ""
        self.report: Dict = {}
        self._jsonl_path: Optional[str] = None

    # -- plumbing ------------------------------------------------------------- #

    def log(self, msg: str) -> None:
        print(f"[interventions] {msg}", flush=True)

    @property
    def run_dir(self) -> str:
        return self.args.run_dir

    def jsonl(self, row: Dict, stream: str = "results") -> None:
        """INCREMENTAL write: one line, append + close immediately (crash-safe)."""
        assert self._jsonl_path, "run_dir not initialised"
        path = self._jsonl_path if stream == "results" else os.path.join(
            self.run_dir, f"{stream}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # -- setup ------------------------------------------------------------------ #

    def setup(self) -> None:
        t0 = time.perf_counter()
        os.makedirs(self.run_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.run_dir, "results.jsonl")
        self.world = generate_world(seed=self.args.seed, n_chains=self.args.chains)
        failures = verify_world(self.world)
        if failures:
            raise RuntimeError(f"world ground truth inconsistent: {failures[:3]}")
        self.plan = reroute_plan(self.world, self.args.n_reroutes)
        self.world.save_json(os.path.join(self.run_dir, "world.json"))
        self.jsonl({"row": "prereg_plan", "seed": self.args.seed,
                    "chains": self.args.chains, "n_reroutes": self.args.n_reroutes,
                    "statement_template": INTERVENTION_STATEMENT,
                    "scaffold": FORMAT_SCAFFOLD,
                    "knobs": {"learning_rate": self.args.learning_rate,
                              "gate_steps": self.args.gate_steps,
                              "gate_lr": self.args.gate_lr,
                              "answer_tokens": self.args.answer_tokens,
                              "correction_repeats": self.args.correction_repeats},
                    "plan": [{k: v for k, v in p.items() if k != "graph"}
                             for p in apply_plan(self.plan, self.world.graph)]})

        t1 = time.perf_counter()
        if self.args.foundation == "qwen":
            handle = load_frozen_base(local_dir=self.args.local_dir,
                                      device=self.args.device)
            self.model, self.tok = handle.model, handle.tokenizer
            self.fingerprint_at_load = handle.fingerprint
            self.frozen_stats = {"frozen_tensors": handle.n_frozen_tensors,
                                 "frozen_params": handle.n_frozen_params}
            self.foundation_label = (f"Qwen3.5-4B-Base @ {self.args.local_dir} "
                                     f"(CPU BF16, FROZEN)")
        else:
            self.model, self.tok = tiny_foundation(seed=self.args.seed)
            self.fingerprint_at_load = foundation_fingerprint(self.model)
            self.foundation_label = "tiny-standin(Llama-4L-64D, CPU, FROZEN)"
        self.timings["foundation_load_s"] = round(time.perf_counter() - t1, 1)

        self.iface = RelationalInterface(
            self.model, cfg=InterfaceConfig(learning_rate=self.args.learning_rate,
                                            gate_lr=self.args.gate_lr))
        self.student = Student(self.iface, self.tok,
                               StudentConfig(answer_tokens=self.args.answer_tokens,
                                             gate_steps=self.args.gate_steps,
                                             correction_repeats=self.args.correction_repeats))
        self.rmem = RelationalMemory(conns=list(self.iface.conns),
                                     hidden_dim=self.iface.hidden_dim,
                                     episodic=self.episodic,
                                     blend_ratio=self.iface.cfg.blend_ratio,
                                     strength_threshold=STRENGTH_THRESHOLD)
        self.teacher = ScriptedTeacher(
            reference_answers={q.text: q.answer for q in self.world.questions})
        self.timings["setup_s"] = round(time.perf_counter() - t0, 1)
        self.log(f"setup: {self.foundation_label} (load {self.timings['foundation_load_s']}s); "
                 f"world seed={self.args.seed} chains={self.args.chains} "
                 f"relationships={len(self.world.relationships)} verify=OK; "
                 f"reroutes={len(self.plan)} pairs={self.iface.pairs} "
                 f"hidden={self.iface.hidden_dim}")

    # -- probes / registration ---------------------------------------------------- #

    def probe_sentence(self, sentence: str):
        ids = torch.tensor([self.tok.encode(sentence)], dtype=torch.long)
        with torch.no_grad():
            res = self.iface.forward(ids)
        pre_i, post_i = self.iface.pairs[0]
        return (res["hidden"][pre_i].mean(dim=1)[0].float(),
                res["hidden"][post_i].mean(dim=1)[0].float())

    def register_fact(self, key: str, sentence: str, chain_id: int,
                      meta: Optional[Dict] = None) -> None:
        pre, post = self.probe_sentence(sentence)
        self.rmem.register(key, 0, pre, post,
                           meta={"sentence": sentence, "chain": chain_id, **(meta or {})})

    def u_fast_norms(self) -> List[float]:
        return [round(c.plasticity.u_fast.norm().item(), 6) for c in self.iface.conns]

    def gamma_values(self) -> List[float]:
        return [round(float(g.detach()), 6) for g in self.iface.gates]

    # ========================================================================= #
    # PHASE TEACH — the proven Phase-1 teach loop (ScriptedTeacher)
    # ========================================================================= #

    def teach_phase(self) -> None:
        t0 = time.perf_counter()
        u_before = self.u_fast_norms()
        n_facts = 0
        for c in range(self.world.n_chains):
            rels = self.world.relationships_of_chain(c)
            lesson = " ".join(r.sentence for r in rels)
            session = DialogueSession(
                teacher=self.teacher, student=self.student, extractor=RuleExtractor(),
                corrections=self.corrections,       # the run-level ledger (Phase-1 pattern)
                transcript_id=f"teach-chain{c:02d}")
            facts = session.teach(lesson)        # extractor -> student.learn_fact (plastic)
            for fact in facts:
                key = relation_key(fact.subject, fact.relation, fact.object)
                canonical = next(r.sentence for r in rels
                                 if r.subject == fact.subject and r.relation == fact.relation)
                self.register_fact(key, canonical, c)
                self.episodic.record(kind="lesson", text=canonical, channel="transcript",
                                     teacher="ScriptedTeacher (offline stub)",
                                     transcript_id=session.transcript_id,
                                     tags={"answer": fact.object, "fact_key": key,
                                           "kind": "fact", "chain": str(c)})
            n_facts += len(facts)
        u_after = self.u_fast_norms()
        self.timings["teach_s"] = round(time.perf_counter() - t0, 1)
        strengths = {k: (self.rmem.strength(k, blend=1.0) or 0.0)
                     for k in sorted(self.rmem.records)}
        weak = [k for k, s in strengths.items() if s <= STRENGTH_THRESHOLD]
        self.teach_summary = {
            "facts_learned": n_facts,
            "u_fast_norms_before": u_before, "u_fast_norms_after": u_after,
            "gamma_after_teach": self.gamma_values(),
            "theta_slow_empty": all(
                float(c.plasticity.theta_slow.norm().item()) == 0.0 for c in self.iface.conns),
            "n_registered": len(strengths),
            "min_fact_plastic_strength": round(min(strengths.values()), 6) if strengths else None,
            "mean_fact_plastic_strength": (round(sum(strengths.values()) / len(strengths), 6)
                                           if strengths else None),
            "weak_facts": weak,
        }
        self.jsonl({"row": "teach_summary", **self.teach_summary})
        self.log(f"TEACH done: {n_facts} facts plastic ({self.timings['teach_s']}s); "
                 f"u_fast {u_before} -> {u_after}; min strength "
                 f"{self.teach_summary['min_fact_plastic_strength']}; weak={len(weak)}")
        if weak:
            raise RuntimeError(f"acquisition gate failed: {len(weak)} facts below threshold")

    # ========================================================================= #
    # CONSTRAINED SCORING (the Phase-2 pattern) + provenance buckets
    # ========================================================================= #

    def candidates_for(self, qtype: str) -> List[str]:
        if qtype in ("reach_now", "reach_plain"):
            return list(self.world.outputs)
        if qtype == "route_now":
            return list(self.world.relays)
        raise ValueError(f"no candidate pool for qtype {qtype!r}")

    def constrained_scores(self, question: str, candidates: List[str],
                           mode: str) -> Dict[str, float]:
        """P(c|q) ∝ exp(score): candidate log-prob at the answer position under the fixed
        scaffold, with the provenance-bucket ablation active for `mode`."""
        prompt = FORMAT_SCAFFOLD + f"Question: {question}\nAnswer:"
        ids_prompt = self.tok.encode(prompt)
        scores: Dict[str, float] = {}
        for cand in candidates:
            ids_c = self.tok.encode(" " + cand)
            full = torch.tensor([ids_prompt + ids_c], dtype=torch.long)
            with ablated(self.iface.conns, self.iface.gates, mode):
                with torch.no_grad():
                    res = self.iface.forward(full)
            lp = torch.log_softmax(res["logits"][0].float(), dim=-1)
            base = len(ids_prompt) - 1
            scores[cand] = sum(float(lp[base + j, t]) for j, t in enumerate(ids_c))
        return scores

    def retrieved_answer(self, question: str) -> Tuple[str, float]:
        hits = self.episodic.retrieve(question, k=1)
        if not hits or hits[0][1] <= 0.0:
            return "", 0.0
        ev, score = hits[0]
        return ev.tags.get("answer") or ev.text, float(score)

    def score_question(self, epoch: int, qtype: str, question: str, gold: str,
                       graph: Dict) -> Dict:
        """One question, every provenance bucket, scored vs gold. Written to JSONL
        immediately (never batched)."""
        candidates = self.candidates_for(qtype)
        modes: Dict[str, Dict] = {}
        for mode in (FULL, PLASTIC, CONSOLIDATED):
            scores = self.constrained_scores(question, candidates, mode)
            ranked = sorted(candidates, key=lambda c: -scores[c])
            mx = max(scores.values())
            probs = {c: pow(2.718281828, s - mx) / sum(pow(2.718281828, v - mx)
                                                       for v in scores.values())
                     for c, s in scores.items()}
            modes[mode] = {"argmax": ranked[0], "correct": ranked[0] == gold,
                           "gold_rank": ranked.index(gold) + 1,
                           "gold_prob": round(probs[gold], 6),
                           "gold_score": round(scores[gold], 6)}
        retrieved, r_score = self.retrieved_answer(question)
        heuristic = heuristic_answer(question, self.world.actors, graph)
        row = {
            "row": "question", "epoch": epoch, "qtype": qtype, "question": question,
            "gold": gold, "candidates": candidates, "chance": round(1.0 / len(candidates), 4),
            "modes": modes,
            "retrieved_answer": retrieved, "retrieval_score": round(r_score, 4),
            "retrieved_correct": bool(default_grader(retrieved, gold)),
            "heuristic_answer": heuristic,
            "heuristic_correct": bool(heuristic == gold),
            "lenient": {m: bool(lenient_match(modes[m]["argmax"], gold))
                        for m in (FULL, PLASTIC, CONSOLIDATED)},
            "contains": {m: bool(contains_match(modes[m]["argmax"], gold))
                         for m in (FULL, PLASTIC, CONSOLIDATED)},
        }
        self.jsonl(row)
        return row

    # ========================================================================= #
    # PHASE BASELINE (epoch 0) + INTERVENE loop
    # ========================================================================= #

    def stability_questions(self) -> List[Tuple[str, str]]:
        """(qtype, question) pairs of the plain reach set — one per actor, fixed order."""
        return [("reach_plain", f"What does {a} reach?") for a in self.world.actors]

    def score_stability(self, epoch: int, graph: Dict) -> List[Dict]:
        """The plain reach set, scored against THIS epoch's gold (never batched: each
        row is written to JSONL inside score_question)."""
        rows = []
        for qt, q in self.stability_questions():
            actor = self._actor_of(q)
            rows.append(self.score_question(epoch, qt, q, reach_gold(actor, graph), graph))
        return rows

    def _actor_of(self, question: str) -> str:
        for a in self.world.actors:
            if f"What does {a} reach?" in question:
                return a
        raise ValueError(question)

    def present_intervention(self, item: Dict) -> Dict:
        """The PROVEN Phase-1 plastic-override path: DialogueSession.teach(statement) ->
        CorrectionHandler.apply_reroute -> learn_fact(repeats=3)."""
        t0 = time.perf_counter()
        k = item["index"]
        u_before = self.u_fast_norms()
        pre_strengths = {key: (self.rmem.strength(key, blend=1.0) or 0.0)
                         for key in sorted(self.rmem.records)}
        session = DialogueSession(
            teacher=self.teacher, student=self.student, extractor=RuleExtractor(),
            corrections=self.corrections, transcript_id=f"intervene-{k:02d}")
        sentence = INTERVENTION_STATEMENT.format(conduit=item["conduit"],
                                                 new_relay=item["new_relay"])
        session.teach(sentence)
        new_key = relation_key(item["conduit"], "routes-to", item["new_relay"])
        old_key = relation_key(item["conduit"], "routes-to", item["old_relay"])
        # extraction verification: the handler's last event must be the intended reroute
        last = self.corrections.events[-1] if self.corrections.events else None
        extracted_ok = bool(last and last.after.subject == item["conduit"]
                            and last.after.relation == "routes-to"
                            and last.after.object == item["new_relay"]
                            and last.repeats == self.args.correction_repeats)
        self.register_fact(new_key, sentence, item["chain_id"],
                           meta={"correction": True, "reroute": k})
        if old_key in self.rmem.records:
            self.rmem.records[old_key].superseded_by = new_key
        self.episodic.record(kind="correction", text=sentence, channel="transcript",
                             teacher="ScriptedTeacher (offline stub)",
                             transcript_id=session.transcript_id,
                             tags={"answer": item["new_relay"], "fact_key": new_key,
                                   "replaces": old_key, "kind": "reroute", "reroute": str(k)})
        u_after = self.u_fast_norms()
        post_strengths = {key: (self.rmem.strength(key, blend=1.0) or 0.0)
                          for key in sorted(self.rmem.records)}
        changes = []
        for key in pre_strengths:
            if key == new_key:
                continue
            b = pre_strengths[key]
            if b > STRENGTH_THRESHOLD:
                changes.append(abs(post_strengths.get(key, 0.0) - b) / b)
        others = [key for key in pre_strengths if key != new_key]
        present = sum(1 for key in others if post_strengths.get(key, 0.0) > STRENGTH_THRESHOLD)
        out = {
            "row": "reroute", "index": k, "statement": sentence,
            "new_key": new_key, "old_key": old_key,
            "intervention_applied_as_plastic_override": extracted_ok,
            "new_fact_plastic_strength": round(post_strengths.get(new_key, 0.0), 6),
            "old_fact_plastic_strength_after": round(post_strengths.get(old_key, 0.0), 6),
            "old_key_superseded": bool(
                self.rmem.records.get(old_key) and
                self.rmem.records[old_key].superseded_by == new_key),
            "u_fast_norms_before": u_before, "u_fast_norms_after": u_after,
            "old_facts_present_after": f"{present}/{len(others)}",
            "old_facts_all_present": present == len(others),
            "mean_relative_strength_change_old_facts": (round(sum(changes) / len(changes), 6)
                                                        if changes else None),
            "max_relative_strength_change_old_facts": round(max(changes), 6) if changes else None,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        out["new_fact_learned"] = out["new_fact_plastic_strength"] > STRENGTH_THRESHOLD
        self.jsonl(out)
        self.log(f"intervene {k}: {sentence} new_fact_learned={out['new_fact_learned']} "
                 f"(strength {out['new_fact_plastic_strength']}, was "
                 f"{out['old_fact_plastic_strength_after']} for the old edge); "
                 f"u_fast {u_before} -> {u_after} ({out['elapsed_s']}s)")
        return out

    def intervene_phase(self) -> None:
        t0 = time.perf_counter()
        # epoch 0 baseline: the stability set against G_0
        rows0 = self.score_stability(0, self.world.graph)
        acc0 = {m: round(sum(r["modes"][m]["correct"] for r in rows0) / len(rows0), 4)
                for m in (FULL, PLASTIC, CONSOLIDATED)}
        self.jsonl({"row": "epoch_summary", "epoch": 0, "kind": "stability",
                    "accuracy": acc0,
                    "heuristic_correct": sum(r["heuristic_correct"] for r in rows0),
                    "n": len(rows0)})
        self.log(f"epoch 0 baseline: stability reach accuracy {acc0} "
                 f"(heuristic {sum(r['heuristic_correct'] for r in rows0)}/{len(rows0)})")

        g = self.world.graph
        self.epoch_graphs = [g]                     # epoch 0 .. epoch n (epoch = reroutes done)
        self.reroute_reports = []
        self.stability_by_epoch = {0: rows0}
        for item in apply_plan(self.plan, self.world.graph):
            k = item["index"]                       # 0-based reroute id
            epoch = k + 1                           # 1-based epoch (reroutes applied so far)
            rer = self.present_intervention(item)
            g = item["graph"]                       # cumulative gold graph
            self.epoch_graphs.append(g)
            # intervention questions (this reroute's own epoch)
            q_reach = f"What does {item['actor']} reach now?"
            q_route = f"Which relay does {item['actor']} route through now?"
            r1 = self.score_question(epoch, "reach_now", q_reach, item["gold_reach"], g)
            r2 = self.score_question(epoch, "route_now", q_route, item["gold_route"], g)
            # stability set re-scored against THIS epoch's gold
            rows = self.score_stability(epoch, g)
            self.stability_by_epoch[epoch] = rows
            acc = {m: round(sum(r["modes"][m]["correct"] for r in rows) / len(rows), 4)
                   for m in (FULL, PLASTIC, CONSOLIDATED)}
            iacc = {m: round(sum(r["modes"][m]["correct"] for r in (r1, r2)) / 2, 4)
                    for m in (FULL, PLASTIC, CONSOLIDATED)}
            self.jsonl({"row": "epoch_summary", "epoch": epoch, "reroute": k, "kind": "mixed",
                        "stability_accuracy": acc, "intervention_accuracy": iacc,
                        "heuristic_correct": r1["heuristic_correct"] + r2["heuristic_correct"]
                        + sum(r["heuristic_correct"] for r in rows),
                        "heuristic_total": 2 + len(rows),
                        "statement": rer["statement"],
                        "new_fact_plastic_strength": rer["new_fact_plastic_strength"]})
            self.reroute_reports.append({"reroute": rer, "reach_now": r1, "route_now": r2,
                                         "stability_rows": rows})
            self.log(f"epoch {epoch}: intervention acc {iacc} | stability acc {acc} "
                     f"| PLASTIC-vs-CONSOLIDATED delta (intervention) "
                     f"{iacc[PLASTIC] - iacc[CONSOLIDATED]:+.3f}")
        self.timings["intervene_s"] = round(time.perf_counter() - t0, 1)
        self.log(f"INTERVENE done: {len(self.plan)} reroutes "
                 f"({self.timings['intervene_s']}s)")

    # ========================================================================= #
    # PHASE REPORT
    # ========================================================================= #

    def aggregate(self) -> Dict:
        """Headline aggregates over the intervention questions + stability analysis."""
        irows = [(rr["reach_now"], rr["route_now"]) for rr in self.reroute_reports]
        irows = [r for pair in irows for r in pair]
        n = max(1, len(irows))
        per_mode = {}
        for m in (FULL, PLASTIC, CONSOLIDATED):
            per_mode[m] = {
                "acc@1": round(sum(r["modes"][m]["correct"] for r in irows) / n, 4),
                "mean_gold_prob": round(sum(r["modes"][m]["gold_prob"] for r in irows) / n, 5),
                "mean_gold_rank": round(sum(r["modes"][m]["gold_rank"] for r in irows) / n, 3),
                "lenient": round(sum(r["lenient"][m] for r in irows) / n, 4)}
        per_mode[RETRIEVED] = {
            "acc@1": round(sum(r["retrieved_correct"] for r in irows) / n, 4)}
        plastic_only_correct = sum(1 for r in irows if r["modes"][PLASTIC]["correct"]
                                   and not r["modes"][CONSOLIDATED]["correct"])
        plastic_only_wrong = sum(1 for r in irows if r["modes"][CONSOLIDATED]["correct"]
                                 and not r["modes"][PLASTIC]["correct"])
        heuristic_acc = round(sum(r["heuristic_correct"] for r in irows) / n, 4)
        chance = irows[0]["chance"] if irows else 0.0

        # stability analysis: epoch k vs k-1, per actor, per mode (graphs precomputed)
        stable = flip = congruent = unstable = 0
        stability_rows = []
        actors = self.world.actors
        graphs = getattr(self, "epoch_graphs", None) or [self.world.graph]
        for k in range(1, len(self.stability_by_epoch)):
            prev, cur = self.stability_by_epoch[k - 1], self.stability_by_epoch[k]
            affected_actor = self.plan[k - 1]["actor"]
            for a, prow, crow in zip(actors, prev, cur):
                gold_changed = (reach_gold(a, graphs[k]) != reach_gold(a, graphs[k - 1]))
                assert (a == affected_actor) == gold_changed, \
                    f"epoch {k}: affected-actor bookkeeping disagrees with the gold walk"
                for m in (FULL, PLASTIC, CONSOLIDATED):
                    changed = prow["modes"][m]["argmax"] != crow["modes"][m]["argmax"]
                    if gold_changed:
                        flip += 1
                        if changed and crow["modes"][m]["correct"]:
                            congruent += 1
                    else:
                        if changed:
                            unstable += 1
                        else:
                            stable += 1
                    stability_rows.append({"epoch": k, "actor": a, "mode": m,
                                           "gold_changed": gold_changed, "flipped": changed,
                                           "post_correct": crow["modes"][m]["correct"]})
        n_g = max(1, flip)
        n_u = max(1, stable + unstable)
        stability = {
            "affected_answer_flipped": round(flip / n_g, 4),
            "affected_flipped_and_correct": round(congruent / n_g, 4),
            "unaffected_answer_stable": round(stable / n_u, 4),
            "unaffected_answer_moved": round(unstable / n_u, 4),
            "n_affected_comparisons": flip, "n_unaffected_comparisons": stable + unstable,
        }
        return {"intervention_questions": {
                    "n": len(irows), "chance": chance, "heuristic_ceiling": heuristic_acc,
                    "modes": per_mode,
                    "plastic_ablation_delta": round(
                        per_mode[PLASTIC]["acc@1"] - per_mode[CONSOLIDATED]["acc@1"], 4),
                    "full_ablation_delta": round(
                        per_mode[FULL]["acc@1"] - per_mode[CONSOLIDATED]["acc@1"], 4),
                    "plastic_only_correct": plastic_only_correct,
                    "plastic_only_wrong": plastic_only_wrong},
                "stability": stability, "stability_rows": stability_rows}

    def report_phase(self, agg: Dict) -> None:
        fp_ok, fp_diffs = fingerprints_equal(self.fingerprint_at_load,
                                             foundation_fingerprint(self.model))
        strengths = {k: self.rmem.strengths(k) for k in sorted(self.rmem.records)}
        intervened = [rr["reroute"] for rr in self.reroute_reports]
        iq = agg["intervention_questions"]
        acc_p = iq["modes"][PLASTIC]["acc@1"]
        acc_c = iq["modes"][CONSOLIDATED]["acc@1"]
        verdict = {
            "expression_above_chance": bool(acc_p > iq["chance"]),
            "plastic_helps_direction": bool(iq["plastic_ablation_delta"] > 0
                                            or iq["plastic_only_correct"]
                                            > iq["plastic_only_wrong"]),
            "heuristic_ceiling_ok": bool(iq["heuristic_ceiling"] == 1.0),
            "old_reach_stability": agg["stability"]["unaffected_answer_stable"],
            "note": "honest expectation (pre-registered): expression may sit near chance; "
                    "a ~0 ablation delta with strong u_fast evidence IS the finding",
        }
        self.report = {
            "run": {"tag": self.run_tag, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment": "intervention — predict the consequence of a world change",
                    "foundation": self.foundation_label,
                    "teacher": "ScriptedTeacher (offline stub; grading refs = world gold)",
                    "seed": self.args.seed, "chains": self.world.n_chains,
                    "relationships": len(self.world.relationships),
                    "n_reroutes": len(self.plan),
                    "preregistration": "see module docstring — frozen before the run",
                    "knobs": {"learning_rate": self.args.learning_rate,
                              "gate_steps": self.args.gate_steps,
                              "gate_lr": self.args.gate_lr,
                              "answer_tokens": self.args.answer_tokens,
                              "correction_repeats": self.args.correction_repeats},
                    "scaffold": FORMAT_SCAFFOLD,
                    "frozen": getattr(self, "frozen_stats", None),
                    "concurrent_load_note": "real run shares Lappy CPU with the concurrent "
                                            "ownership_scale lane (reported, not tuned around)"},
            "timings_s": self.timings,
            "teach": self.teach_summary,
            "reroutes": intervened,
            "aggregates": agg,
            "fact_strengths_after_all_reroutes": strengths,
            "theta_slow_empty": all(float(c.plasticity.theta_slow.norm().item()) == 0.0
                                    for c in self.iface.conns),
            "foundation_integrity": {
                "bit_identical_after_full_run": bool(fp_ok),
                "differing_params": fp_diffs[:5],
                "n_params_checked": len(self.fingerprint_at_load)},
            "verdict": verdict,
        }
        with open(os.path.join(self.run_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=1, default=str)

        print("\n" + "=" * 88)
        print(f"AION-1 INTERVENTION EXPERIMENT ({self.run_tag})")
        print(f"foundation: {self.foundation_label}")
        print("=" * 88)
        print(f"teach: {self.teach_summary['facts_learned']} facts, u_fast "
              f"{self.teach_summary['u_fast_norms_after']}, min strength "
              f"{self.teach_summary['min_fact_plastic_strength']}")
        print("-" * 88)
        print("PER-REROUTE (reach_now / route_now):")
        for rr in self.reroute_reports:
            rer, r1, r2 = rr["reroute"], rr["reach_now"], rr["route_now"]
            def cell(r, m):
                return f"{r['modes'][m]['argmax']:>8s}{'*' if r['modes'][m]['correct'] else ' '}"
            print(f"  #{rer['index']} {rer['statement']}")
            print(f"      gold reach={r1['gold']:>8s} route={r2['gold']:>8s} | "
                  f"PLASTIC reach={cell(r1, PLASTIC)} route={cell(r2, PLASTIC)} | "
                  f"FULL reach={cell(r1, FULL)} route={cell(r2, FULL)} | "
                  f"CONSOL reach={cell(r1, CONSOLIDATED)} route={cell(r2, CONSOLIDATED)}")
        print("-" * 88)
        print("AGGREGATE (intervention questions, n=%d, chance=%.2f):" % (iq["n"], iq["chance"]))
        for m, s in iq["modes"].items():
            print(f"  {m:13s} acc@1={s['acc@1']:.3f}")
        print(f"  heuristic ceiling = {iq['heuristic_ceiling']:.3f} | "
              f"plastic ablation delta (PLASTIC - CONSOLIDATED) = "
              f"{iq['plastic_ablation_delta']:+.3f} | FULL delta = "
              f"{iq['full_ablation_delta']:+.3f}")
        print(f"  plastic-only correct {iq['plastic_only_correct']} / "
              f"plastic-only wrong {iq['plastic_only_wrong']}")
        s = agg["stability"]
        print("STABILITY (old reach questions re-scored each epoch vs own-epoch gold):")
        print(f"  affected actors flipped {s['affected_answer_flipped']:.2f} "
              f"(flipped-and-correct {s['affected_flipped_and_correct']:.2f}, "
              f"n={s['n_affected_comparisons']}); unaffected stable "
              f"{s['unaffected_answer_stable']:.2f} (moved "
              f"{s['unaffected_answer_moved']:.2f}, n={s['n_unaffected_comparisons']})")
        print("-" * 88)
        print("VERDICT:", json.dumps(verdict, indent=1))
        print(f"report: {os.path.join(self.run_dir, 'report.json')}")

    # --------------------------------------------------------------------------- #

    def run(self) -> int:
        t_start = time.perf_counter()
        slug = f"c{self.args.chains}-r{self.args.n_reroutes}"
        self.run_tag = (f"interventions-{self.args.foundation}-{slug}-s{self.args.seed}-"
                        f"{time.strftime('%H%M%S')}")
        if not self.args.run_dir:
            self.args.run_dir = os.path.join(_REPO_ROOT, "runs", "interventions", self.run_tag)
        self.setup()
        self.teach_phase()
        self.intervene_phase()
        agg = self.aggregate()
        self.report_phase(agg)
        self.timings["total_s"] = round(time.perf_counter() - t_start, 1)
        print(f"TOTAL RUNTIME: {self.timings['total_s']}s "
              f"({self.timings['total_s'] / 60:.1f} min)")
        return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="AION-1 intervention experiment: reroute the learned world, "
                    "score the predicted new reach (constrained scoring + ablation)")
    ap.add_argument("--foundation", default="tiny", choices=("tiny", "qwen"),
                    help="tiny = in-memory stand-in (Rog CPU debug); qwen = real frozen "
                         "Qwen3.5-4B-Base (Lappy)")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chains", type=int, default=5,
                    help="the PROVEN Phase-1 configuration (5 chains / 20 relationships)")
    ap.add_argument("--n-reroutes", type=int, default=N_REROUTES,
                    help="the PREREGISTERED reroute count (8)")
    ap.add_argument("--learning-rate", type=float, default=0.5,
                    help="nnfs plasticity lr — the PROVEN Phase-1 real-run value")
    ap.add_argument("--gate-steps", type=int, default=6,
                    help="residual-gate autograd steps per fact — PROVEN Phase-1 value")
    ap.add_argument("--gate-lr", type=float, default=300.0,
                    help="gate gamma SGD step size — PROVEN Phase-1 value")
    ap.add_argument("--answer-tokens", type=int, default=6,
                    help="PROVEN Phase-1 real-run value (unused by constrained scoring)")
    ap.add_argument("--correction-repeats", type=int, default=3,
                    help="plastic repetition weight for reroute facts — PROVEN value")
    ap.add_argument("--run-dir", default=None)
    return InterventionRun(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
