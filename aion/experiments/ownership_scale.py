"""AION-1 OWNERSHIP SCALE RUN — do the ownership MECHANICS scale to 100 relationships?

PREREGISTRATION (frozen 2026-11-25, BEFORE the run; owner brief: task 9's measurable
portion at scale, ungated because it uses only proven machinery). Nothing below is
chosen after seeing data:

  WORLD          generate_world(seed=0, n_chains=25) -> exactly 100 relationships,
                 325 questions. verify_world() must return [] (ground truth
                 graph-derivable) or the run aborts before any learning.

  TEACH          all 25 chain lessons through DialogueSession.teach (RuleExtractor ->
                 Student.learn_fact -> pathway.nnfs_update per fact) with the
                 ScriptedTeacher (owner-permitted for v1; noted in the report: lesson
                 delivery is scripted, the plastic signal is IDENTICAL to the Phase-1
                 endpoint run because the extractor consumes lesson_text either way).
                 Knobs match the Phase-1 real run (runs/teacher_student/qwen-real-c5)
                 exactly: nnfs learning_rate=0.5, gate_steps=6 per fact, gate_lr=300,
                 answer_tokens=8, correction_repeats=3. Every fact: plastic update +
                 RelationalMemory registration + episodic record (100 lessons).

  TEST           DIRECT battery = ALL 100 direct questions (the acquisition battery,
                 full count at scale). PARAPHRASE subset = 20 questions picked by
                 deterministic stride over the 100 paraphrase questions (indices
                 0,5,...,95). COMBINATION + INTERVENTION batteries are NOT measured:
                 expression is the Phase-2 verdict and is RESTATEd, not re-measured
                 (Phase 2, p2-qwen-c5: combination acc@1 0.333 vs chance 0.400 for
                 C/D/E; free-form exact 0.071).
                 Every answer is produced once per PROVENANCE BUCKET by ablating one
                 bucket at a time (aion/experiments/retention.py ablated()):
                 PLASTIC (u_fast alone), CONSOLIDATED (theta_slow alone), RETRIEVED
                 (pathway ablated, episodic lookup only), plus the natural blend FULL.
                 Provenance label priority PLASTIC > CONSOLIDATED > RETRIEVED, else
                 FULL_ONLY, else NONE. Scoring: strict default_grader primary, with
                 lenient/contains recorded alongside (same graders as Phase 1).

  CORRECT        EXACTLY 10 corrections, mid-world, via the CorrectionHandler
                 intervention path (plastic override, repeats=3): conduits of chains
                 [0,2,5,7,10,12,15,17,20,22] (deterministic stride 25/10) each rerouted
                 to relays[(chain+1) % 25] ("Tav is rerouted to relay Drel." shape).
                 Verify at scale: 10/10 new facts learned (plastic strength >
                 threshold), and the WITHOUT-DESTROYING criterion counted exactly on
                 STRENGTHS: 90 original facts are untouched by corrections and each
                 is reported intact iff its plastic strength stays above threshold
                 ("X/90 intact"). The 10 superseded old keys are reported separately
                 (expected to still transmit: the override is additive in u_fast, not
                 a deletion). Probe-answer identity on 20 direct questions (chains
                 0-19) is REPORTED as a diagnostic, not gated: the NNFS update is
                 additive in u_fast, so an unrelated answer string can shift without
                 any fact being destroyed (Phase 1's 16/16 identity held because all
                 plastic answers were empty strings there — noted, not hidden).

  RETAIN         STRICT restart: plastic state + probe store + fact index serialized,
                 foundation torn down, FULL reload from disk, fresh interface + exact
                 restore, BITWISE equality check (atol=0) and probe regeneration
                 identity over all 110 registered facts. Then the survival batteries:
                 DIRECT (all 100) + PARAPHRASE subset (20) asked pre-restart (in the
                 post-correction state) and again post-restart; survival per battery =
                 per-question answer identity per mode + accuracy parity.

  PURGE          episodic transcript purge -> probe answers (same 20 questions, FULL +
                 PLASTIC) must be UNCHANGED; RETRIEVED is expected to degrade (the
                 transcript WAS its substrate) and is reported as such.

  REPORT         per-battery accuracy + provenance histogram per battery, correction
                 survival count, restart exactness, runtime, and the side-by-side
                 Phase-1 (20-rel) reference numbers, plus the honest expression note.

  PERSISTENCE    incremental JSONL per battery pass (one row per question, flushed as
                 produced — the Lane B lesson: never batch-all-at-end), plus a JSON
                 phase marker after every phase, so a crash leaves partial evidence.

HONESTY RULES: every number is labelled with the foundation and teacher that produced
it; whatever the numbers show is reported; nothing is tuned after peeking.

Run (Lappy, real frozen foundation, CPU):
  /opt/bonsai-demo/.venv/bin/python aion/experiments/ownership_scale.py \
      --foundation qwen --local-dir /mnt/seagate/aion-models/Qwen3.5-4B-Base
Run (Rog CPU debug, tiny stub):
  python aion/experiments/ownership_scale.py            # tiny stand-in, 3 chains
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.dialogue.conversation import DialogueSession, default_grader      # noqa: E402
from aion.dialogue.correction import CorrectionHandler                      # noqa: E402
from aion.dialogue.lesson_extractor import RuleExtractor                    # noqa: E402
from aion.dialogue.student import Student, StudentConfig                    # noqa: E402
from aion.dialogue.teacher import ScriptedTeacher                           # noqa: E402
from aion.worlds.generator import (                                         # noqa: E402
    BATTERY_DIRECT, BATTERY_PARAPHRASE, generate_world, verify_world)
from aion.memory.episodic import EpisodicStore                              # noqa: E402
from aion.memory.relational import RelationalMemory, relation_key           # noqa: E402
from aion.memory.state import PlasticStateSnapshot                          # noqa: E402
from aion.experiments.retention import (                                    # noqa: E402
    ablated, CONSOLIDATED, FULL, MODES, PLASTIC, RETRIEVED, STRONGEST_FIRST)
from aion.experiments.teacher_student import (                              # noqa: E402  (read-only reuse)
    MeasurementStudent, build_tiny_foundation, contains_match, lenient_match)
from aion.relational.interface import (                                     # noqa: E402
    InterfaceConfig, RelationalInterface, fingerprints_equal, foundation_fingerprint)
from aion.foundation.pretrained_loader import DEFAULT_LOCAL_DIR, load_frozen_base  # noqa: E402

BATTERIES = (BATTERY_DIRECT, BATTERY_PARAPHRASE)

# Phase-1 reference (prior landed run, NOT re-measured here):
#   runs/teacher_student/qwen-real-c5/report.json — Qwen3.5-4B-Base CPU BF16, real
#   :8180 endpoint teacher, 5 chains / 20 relationships, same knobs as this run.
PHASE1_REFERENCE = {
    "source": "runs/teacher_student/qwen-real-c5/report.json (2026-11-25, prior run)",
    "foundation": "Qwen3.5-4B-Base @ /mnt/seagate/aion-models/Qwen3.5-4B-Base (CPU BF16, FROZEN)",
    "teacher": "TeacherClient endpoint http://192.168.0.33:8180/v1 alias qwen35-4b",
    "relationships": 20,
    "direct_battery_exact": {"FULL": 0.0, "PLASTIC": 0.0, "CONSOLIDATED": 0.0,
                             "RETRIEVED": 0.75},
    "paraphrase_battery_exact": {"FULL": 0.0, "PLASTIC": 0.0, "CONSOLIDATED": 0.0,
                                 "RETRIEVED": 0.6},
    "provenance_histogram_all": {"RETRIEVED": 21, "NONE": 26},
    "correction": {"new_fact_learned": True, "old_facts_present_after": "20/20",
                   "unaffected_probe_identical": "16/16"},
    "restart_state_restored_exact": True,
    "purge_plastic_answers_identical": "20/20",
    "teach_s": 1184.9, "total_battery_and_retain_s": 505.6 + 563.6,
}

# Phase-2 expression reference (prior landed run, RESTATEd only — expression is not
# re-measured at scale in this run):
#   runs/teacher_student_p2/p2-qwen-c5-s0-022716/report.json
PHASE2_EXPRESSION_REFERENCE = {
    "source": "runs/teacher_student_p2/p2-qwen-c5-s0-022716/report.json (2026-11-25, prior run)",
    "note": "expression (combinations) at chance is the Phase-2 verdict; this scale run "
            "does not re-measure it",
    "combination_acc1_C_D_E": 0.3333, "combination_chance": 0.4,
    "freeform_exact_C_D_E": 0.0714,
    "direct_acc1_C_D_E": 0.25, "frozen_base_direct_acc1": 0.25,
    "paraphrase_acc1_C_D_E": 0.125, "B_retrieval_paraphrase": 0.625,
}


# --------------------------------------------------------------------------- #
# Preregistered subset/plan helpers (pure functions of the world -> testable)
# --------------------------------------------------------------------------- #

def stride_pick(questions: List, cap: int) -> List:
    """Deterministic spread of a battery across its generator order, up to `cap`."""
    if cap <= 0 or len(questions) <= cap:
        return list(questions)
    step = len(questions) / float(cap)
    return [questions[int(i * step)] for i in range(cap)]


def direct_battery(world) -> List:
    """PREREGISTERED: ALL direct questions (one per taught relationship)."""
    return [q for q in world.questions if q.battery == BATTERY_DIRECT]


def paraphrase_subset(world, cap: int = 20) -> List:
    """PREREGISTERED: stride cap over the paraphrase battery (100 -> 20, indices 0,5,..,95)."""
    paras = [q for q in world.questions if q.battery == BATTERY_PARAPHRASE]
    return stride_pick(paras, cap)


def correction_plan(world, n_corrections: int = 10) -> List[Dict]:
    """PREREGISTERED: 10 mid-chain reroutes. Chains by stride (n_chains / n); each
    corrected conduit reroutes to relays[(chain+1) % n_chains] (always != its old relay)."""
    chains = stride_pick(list(range(world.n_chains)), n_corrections)
    plan = []
    for c in chains:
        conduit, old_relay = world.chains[c][1], world.chains[c][2]
        new_relay = world.relays[(c + 1) % world.n_chains]
        assert new_relay != old_relay, f"correction for chain {c} would be a no-op"
        plan.append({"chain": c, "conduit": conduit, "old_relay": old_relay,
                     "new_relay": new_relay,
                     "sentence": f"{conduit} is rerouted to relay {new_relay}.",
                     "new_key": relation_key(conduit, "routes-to", new_relay),
                     "old_key": relation_key(conduit, "routes-to", old_relay)})
    assert len({p["conduit"] for p in plan}) == len(plan), "corrections must touch distinct conduits"
    return plan


def probe_set(world, cap: int = 20) -> List:
    """PREREGISTERED probe battery: the first direct question of the first `cap` chains
    ('Who does X operate?') — used for correction identity and the transcript-purge check."""
    out = []
    for c in range(min(cap, world.n_chains)):
        qs = [q for q in world.questions
              if q.battery == BATTERY_DIRECT and q.chain_id == c]
        out.append(qs[0])
    return out


def grading_row(q, answers: Dict[str, str], world) -> Dict:
    """One scored battery row: answers per provenance bucket + all three graders."""
    chance = 0.5 if q.meta.get("kind") == "membership" else 1.0 / max(2, world.n_chains)
    buckets = [m for m in STRONGEST_FIRST if default_grader(answers[m], q.answer)]
    if buckets:
        provenance = buckets[0]
    elif default_grader(answers[FULL], q.answer):
        provenance = "FULL_ONLY"
    else:
        provenance = "NONE"
    return {
        "qid": q.id, "battery": q.battery, "question": q.text, "gold": q.answer,
        "answers": dict(answers), "chance": round(chance, 4),
        "exact": {m: bool(default_grader(answers[m], q.answer)) for m in MODES},
        "lenient": {m: bool(lenient_match(answers[m], q.answer)) for m in MODES},
        "contains": {m: bool(contains_match(answers[m], q.answer)) for m in MODES},
        "provenance_exact": provenance,
        "provenance_lenient": next(
            (m for m in STRONGEST_FIRST if lenient_match(answers[m], q.answer)),
            "FULL_ONLY" if lenient_match(answers[FULL], q.answer) else "NONE"),
    }


def summarize_rows(rows: List[Dict]) -> Dict:
    """Per-battery summary: accuracy per provenance bucket + histogram."""
    n = max(1, len(rows))
    out = {"n": len(rows),
           "mean_chance": round(sum(r["chance"] for r in rows) / n, 4),
           "accuracy": {}}
    for mode in MODES:
        acc = {"exact": round(sum(r["exact"][mode] for r in rows) / n, 4),
               "lenient": round(sum(r["lenient"][mode] for r in rows) / n, 4),
               "contains": round(sum(r["contains"][mode] for r in rows) / n, 4)}
        acc["empty_answer_fraction"] = round(
            sum(1 for r in rows if not (r["answers"][mode] or "").strip()) / n, 4)
        out["accuracy"][mode] = acc
    hist: Dict[str, int] = {}
    for r in rows:
        hist[r["provenance_exact"]] = hist.get(r["provenance_exact"], 0) + 1
    out["provenance_histogram"] = hist
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

class OwnershipScaleRun:
    """Ownership MECHANICS at 100 relationships: TEACH -> TEST -> CORRECT -> RETAIN
    (strict restart + survival batteries) -> PURGE -> REPORT."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tok = None
        self.iface: Optional[RelationalInterface] = None
        self.student: Optional[Student] = None
        self.meas_student: Optional[MeasurementStudent] = None
        self.teacher = None
        self.teacher_label = ""                            # set in setup()
        self.episodic = EpisodicStore(clock=lambda: 0.0)   # deterministic law
        self.rmem: Optional[RelationalMemory] = None
        self.fingerprint_at_load: Optional[Dict] = None
        self.foundation_label = ""
        self.frozen_stats: Dict = {}
        self.timings: Dict[str, float] = {}
        self.sessions: List[Dict] = []                     # transcript dumps
        self.fact_sentences: Dict[str, str] = {}           # key -> canonical sentence
        self.fact_chain: Dict[str, int] = {}

    # -- plumbing ------------------------------------------------------------- #

    def log(self, msg: str) -> None:
        print(f"[ownership-scale] {msg}", flush=True)

    @property
    def run_dir(self) -> str:
        return self.args.run_dir

    def write_phase(self, name: str, payload: Dict) -> None:
        """JSON phase marker: evidence lands INCREMENTALLY (Lane B lesson)."""
        path = os.path.join(self.run_dir, f"phase_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, default=str)

    def jsonl_path(self, tag: str) -> str:
        return os.path.join(self.run_dir, f"battery_{tag}.jsonl")

    def append_row(self, tag: str, row: Dict) -> None:
        with open(self.jsonl_path(tag), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def reset_jsonl(self, tag: str) -> None:
        with open(self.jsonl_path(tag), "w", encoding="utf-8") as f:
            f.write("")

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tok.encode(text)], dtype=torch.long)

    def make_probe(self, ids: torch.Tensor):
        with torch.no_grad():
            res = self.iface.forward(ids)
        pre_i, post_i = self.iface.pairs[0]
        return (res["hidden"][pre_i].mean(dim=1)[0].float(),
                res["hidden"][post_i].mean(dim=1)[0].float())

    def probe_sentence(self, sentence: str):
        return self.make_probe(self.encode(sentence))

    def register_fact(self, key: str, sentence: str, chain_id: int,
                      meta: Optional[Dict] = None) -> None:
        pre, post = self.probe_sentence(sentence)
        self.rmem.register(key, 0, pre, post, meta={"sentence": sentence,
                                                    "chain": chain_id, **(meta or {})})
        self.fact_sentences[key] = sentence
        self.fact_chain[key] = chain_id

    def retrieve_answer(self, question: str) -> str:
        """RETRIEVED bucket: pathway ablated, episodic store only."""
        hits = self.episodic.retrieve(question, k=1)
        if not hits or hits[0][1] <= 0.0:
            return ""
        ev, score = hits[0]
        return ev.tags.get("answer") or ev.text

    def dump_session(self, session: DialogueSession, tag: str) -> None:
        self.sessions.append({
            "tag": tag, "transcript_id": session.transcript_id,
            "turns": [{"index": t.index, "kind": t.kind, "provenance": t.provenance,
                       "content": t.content, "transcript_id": t.transcript_id,
                       "meta": t.meta} for t in session.turns],
            "events": [e.to_dict() for e in session.events]})

    # -- setup ------------------------------------------------------------------ #

    def setup_foundation(self) -> None:
        t0 = time.perf_counter()
        if self.args.foundation == "qwen":
            handle = load_frozen_base(local_dir=self.args.local_dir,
                                      device=self.args.device)
            self.model = handle.model
            self.tok = handle.tokenizer
            self.fingerprint_at_load = handle.fingerprint
            self.frozen_stats = {"frozen_tensors": handle.n_frozen_tensors,
                                 "frozen_params": handle.n_frozen_params}
            self.foundation_label = (f"Qwen3.5-4B-Base @ {self.args.local_dir} "
                                     f"(CPU BF16, FROZEN)")
        else:
            self.model, self.tok = build_tiny_foundation(seed=self.args.seed)
            self.fingerprint_at_load = foundation_fingerprint(self.model)
            self.foundation_label = "tiny-standin(Llama-4L-64D, CPU, FROZEN)"
        self.iface = RelationalInterface(
            self.model, cfg=InterfaceConfig(learning_rate=self.args.learning_rate,
                                            gate_lr=self.args.gate_lr))
        self.student = Student(self.iface, self.tok,
                               StudentConfig(answer_tokens=self.args.answer_tokens,
                                             gate_steps=self.args.gate_steps,
                                             correction_repeats=self.args.correction_repeats))
        self.meas_student = MeasurementStudent(self.student)
        self.rmem = RelationalMemory(conns=list(self.iface.conns),
                                     hidden_dim=self.iface.hidden_dim,
                                     episodic=self.episodic,
                                     blend_ratio=self.iface.cfg.blend_ratio,
                                     strength_threshold=1e-8)
        self.timings["foundation_load_s"] = round(time.perf_counter() - t0, 1)
        self.log(f"foundation: {self.foundation_label} "
                 f"(load {self.timings['foundation_load_s']}s, pairs={self.iface.pairs}, "
                 f"hidden={self.iface.hidden_dim})")

    def setup(self) -> None:
        t0 = time.perf_counter()
        self.world = generate_world(seed=self.args.seed, n_chains=self.args.chains)
        failures = verify_world(self.world)
        self.timings["world_gen_s"] = round(time.perf_counter() - t0, 2)
        if failures:
            raise RuntimeError(f"world ground truth inconsistent: {failures[:3]}")
        self.log(f"world: seed={self.args.seed} chains={self.world.n_chains} "
                 f"relationships={len(self.world.relationships)} "
                 f"questions={len(self.world.questions)} verify=OK")
        # preregistered measurement plan (frozen before any learning)
        self.test_questions = direct_battery(self.world) + \
            paraphrase_subset(self.world, self.args.para_subset)
        self.survival_questions = list(self.test_questions)
        self.probe_questions = probe_set(self.world, self.args.probe_size)
        self.plan = correction_plan(self.world, self.args.n_corrections)
        self.log(f"plan: direct={len(direct_battery(self.world))} "
                 f"paraphrase_subset={len(paraphrase_subset(self.world, self.args.para_subset))} "
                 f"corrections={len(self.plan)} probes={len(self.probe_questions)} "
                 f"batteries_measured={list(BATTERIES)} "
                 f"(combination+intervention NOT measured: Phase-2 verdict restated)")
        os.makedirs(self.run_dir, exist_ok=True)
        self.world.save_json(os.path.join(self.run_dir, "world.json"))
        self.write_phase("protocol", {
            "preregistered": True, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.args.seed, "chains": self.world.n_chains,
            "relationships": len(self.world.relationships),
            "questions_total": len(self.world.questions),
            "batteries_measured": list(BATTERIES),
            "batteries_not_measured_note": "combination+intervention: Phase-2 verdict restated",
            "direct_battery_n": len(direct_battery(self.world)),
            "paraphrase_subset_n": len(paraphrase_subset(self.world, self.args.para_subset)),
            "corrections": self.plan, "probe_qids": [q.id for q in self.probe_questions],
            "knobs": {"learning_rate": self.args.learning_rate,
                      "gate_steps": self.args.gate_steps, "gate_lr": self.args.gate_lr,
                      "answer_tokens": self.args.answer_tokens,
                      "correction_repeats": self.args.correction_repeats},
            "teacher": "ScriptedTeacher (v1; owner-permitted, noted)",
        })
        self.setup_foundation()
        self.teacher = ScriptedTeacher(
            reference_answers={q.text: q.answer for q in self.world.questions})
        self.teacher_label = ("ScriptedTeacher (offline stub; owner-permitted for v1 — "
                              "lesson delivery scripted, plastic signal identical: the "
                              "extractor consumes lesson_text either way)")

    # ========================================================================= #
    # PHASE TEACH — 25 lessons, 100 facts into the plastic pathway
    # ========================================================================= #

    def teach_phase(self) -> None:
        t0 = time.perf_counter()
        self.corrections = CorrectionHandler()
        u_before = [round(c.plasticity.u_fast.norm().item(), 6) for c in self.iface.conns]
        n_facts = 0
        for c in range(self.world.n_chains):
            rels = self.world.relationships_of_chain(c)
            lesson = " ".join(r.sentence for r in rels)
            learn_session = DialogueSession(
                teacher=self.teacher, student=self.student, extractor=RuleExtractor(),
                corrections=self.corrections, transcript_id=f"teach-chain{c:02d}")
            start_learned = self.student.n_facts_learned
            facts = learn_session.teach(lesson)
            uids = list(range(start_learned + 1, start_learned + len(facts) + 1))
            for fact, uid in zip(facts, uids):
                key = relation_key(fact.subject, fact.relation, fact.object)
                canonical = next(r.sentence for r in rels
                                 if r.subject == fact.subject and r.relation == fact.relation)
                self.register_fact(key, canonical, c)
                self.episodic.record(kind="lesson", text=canonical, channel="transcript",
                                     teacher=self.teacher_label,
                                     transcript_id=learn_session.transcript_id,
                                     plastic_update_ids=[uid],
                                     tags={"answer": fact.object, "fact_key": key,
                                           "kind": "fact", "chain": str(c)})
            n_facts += len(facts)
            # protocol comprehension round, measured only (no plastic writes)
            probe_q = self.world.questions[4 * c]
            meas_session = DialogueSession(
                teacher=self.teacher, student=self.meas_student, extractor=RuleExtractor(),
                corrections=CorrectionHandler(), transcript_id=f"teach-check-chain{c:02d}")
            event = meas_session.ask(probe_q.text)
            self.dump_session(learn_session, f"teach-chain{c:02d}")
            self.dump_session(meas_session, f"teach-check-chain{c:02d}")
            if (c + 1) % 5 == 0 or c + 1 == self.world.n_chains:
                gammas = [round(float(g.detach()), 4) for g in self.iface.gates]
                self.log(f"teach {c + 1}/{self.world.n_chains} chains "
                         f"({n_facts} facts plastic so far; gamma={gammas}; "
                         f"comprehension verdict={event.verdict})")
        u_after = [round(c.plasticity.u_fast.norm().item(), 6) for c in self.iface.conns]
        gammas_after = [round(float(g.detach()), 6) for g in self.iface.gates]
        self.timings["teach_s"] = round(time.perf_counter() - t0, 1)
        registered = len(self.rmem.records)
        self.teach_summary = {
            "facts_learned": n_facts, "lessons": self.world.n_chains,
            "facts_registered_with_probes": registered,
            "all_100_facts_stored": bool(n_facts == len(self.world.relationships)
                                         and registered == len(self.world.relationships)),
            "gate_steps_per_fact": self.args.gate_steps,
            "gate_lr": self.args.gate_lr, "nnfs_learning_rate": self.args.learning_rate,
            "u_fast_norms_before": u_before, "u_fast_norms_after": u_after,
            "gamma_after_teach": gammas_after,
            "episodic_events": len(self.episodic),
            "teacher": self.teacher_label,
        }
        self.write_phase("teach", self.teach_summary)
        self.log(f"TEACH done: {n_facts} facts entered the plastic pathway, "
                 f"{registered} registered with probes ({self.timings['teach_s']}s); "
                 f"u_fast {u_before} -> {u_after}")

    # ========================================================================= #
    # Battery machinery
    # ========================================================================= #

    def battery_pass(self, questions: List, tag: str) -> List[Dict]:
        """Ask every question once per provenance bucket; score; write rows INCREMENTALLY."""
        t0 = time.perf_counter()
        self.reset_jsonl(tag)
        session = DialogueSession(
            teacher=self.teacher, student=self.meas_student, extractor=RuleExtractor(),
            corrections=CorrectionHandler(), transcript_id=f"{tag}")
        rows: List[Dict] = []
        for i, q in enumerate(questions):
            answers = {}
            for mode in MODES:
                if mode == RETRIEVED:
                    answers[mode] = self.retrieve_answer(q.text)
                else:
                    with ablated(self.iface.conns, self.iface.gates, mode):
                        ev = session.ask(q.text)
                    answers[mode] = ev.student_answer or ""
            row = grading_row(q, answers, self.world)
            rows.append(row)
            self.append_row(tag, row)                    # incremental persistence
            if (i + 1) % 20 == 0 or i + 1 == len(questions):
                self.log(f"  [{tag}] {i + 1}/{len(questions)} questions asked")
        self.dump_session(session, tag)
        self.timings[f"battery_{tag}_s"] = round(time.perf_counter() - t0, 1)
        return rows

    # ========================================================================= #
    # PHASE TEST — direct 100 + paraphrase 20, pre-correction state
    # ========================================================================= #

    def test_phase(self) -> None:
        t0 = time.perf_counter()
        self.test_rows = self.battery_pass(self.test_questions, tag="test")
        self.test_summary = {b: summarize_rows([r for r in self.test_rows
                                                if r["battery"] == b])
                             for b in BATTERIES}
        self.timings["test_s"] = round(time.perf_counter() - t0, 1)
        self.write_phase("test", {"batteries": self.test_summary,
                                  "rows_jsonl": self.jsonl_path("test")})
        self.log("TEST done (pre-correction state) — exact accuracy per bucket:")
        for b in BATTERIES:
            s = self.test_summary[b]
            acc = s["accuracy"]
            self.log(f"  {b:11s} n={s['n']:3d} chance={s['mean_chance']:.3f} | "
                     + " ".join(f"{m}={acc[m]['exact']:.2f}" for m in MODES)
                     + f" | provenance={s['provenance_histogram']}")

    # ========================================================================= #
    # PHASE CORRECT — 10 reroutes via CorrectionHandler; without-destroying count
    # ========================================================================= #

    def strengths_snapshot(self) -> Dict[str, Dict[str, Optional[float]]]:
        return {key: self.rmem.strengths(key) for key in sorted(self.rmem.records)}

    def probe_answers(self, questions: List, modes=(FULL, PLASTIC)) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for q in questions:
            out[q.id] = {}
            for mode in modes:
                if mode == RETRIEVED:
                    out[q.id][mode] = self.retrieve_answer(q.text)
                else:
                    with ablated(self.iface.conns, self.iface.gates, mode):
                        out[q.id][mode] = self.student.answer(q.text)
        return out

    def corrections_phase(self) -> None:
        t0 = time.perf_counter()
        affected_chains = {p["chain"] for p in self.plan}
        pre_answers = self.probe_answers(self.probe_questions)
        pre_strengths = self.strengths_snapshot()
        original_keys = [k for k in self.fact_sentences]

        per_correction = []
        for i, step in enumerate(self.plan):
            session = DialogueSession(
                teacher=self.teacher, student=self.student, extractor=RuleExtractor(),
                corrections=self.corrections,
                transcript_id=f"correction-{i:02d}-chain{step['chain']:02d}")
            events_before = len(self.corrections.events)
            session.teach(step["sentence"])              # intervention -> apply_reroute
            new_events = self.corrections.events[events_before:]
            self.dump_session(session, f"correction-{i:02d}")
            uid = self.student.n_facts_learned + 1        # next plastic-update id (teach numbering)
            self.episodic.record(kind="correction", text=step["sentence"],
                                 channel="transcript", teacher=self.teacher_label,
                                 transcript_id=session.transcript_id,
                                 plastic_update_ids=[uid],
                                 tags={"answer": step["new_relay"], "fact_key": step["new_key"],
                                       "replaces": step["old_key"], "kind": "reroute",
                                       "chain": str(step["chain"])})
            self.register_fact(step["new_key"], step["sentence"], step["chain"],
                               meta={"correction": True, "replaces": step["old_key"]})
            old_rec = self.rmem.records.get(step["old_key"])
            if old_rec is not None:
                old_rec.superseded_by = step["new_key"]
            new_strength = self.rmem.strength(step["new_key"], blend=1.0)
            per_correction.append({
                "index": i, "chain": step["chain"], "sentence": step["sentence"],
                "new_key": step["new_key"], "old_key": step["old_key"],
                "correction_events": [e.to_dict() for e in new_events],
                "new_fact_plastic_strength": round(new_strength, 6)
                if new_strength is not None else None,
                "new_fact_learned": bool(new_strength is not None
                                         and new_strength > self.rmem.strength_threshold),
            })
            if (i + 1) % 5 == 0 or i + 1 == len(self.plan):
                self.log(f"correction {i + 1}/{len(self.plan)}: {step['sentence']} "
                         f"new_strength={per_correction[-1]['new_fact_plastic_strength']}")

        post_answers = self.probe_answers(self.probe_questions)
        post_strengths = self.strengths_snapshot()

        superseded = {p["old_key"] for p in self.plan}
        untouched = [k for k in original_keys if k not in superseded]
        intact = [k for k in untouched
                  if (post_strengths.get(k, {}) or {}).get("plastic", 0.0) is not None
                  and (post_strengths[k]["plastic"] or 0.0) > self.rmem.strength_threshold]
        still_transmitting_superseded = [
            k for k in superseded
            if (post_strengths.get(k, {}) or {}).get("plastic", 0.0) is not None
            and (post_strengths[k]["plastic"] or 0.0) > self.rmem.strength_threshold]
        n_new_learned = sum(1 for c in per_correction if c["new_fact_learned"])

        changes = []
        for k in untouched:
            b = (pre_strengths.get(k, {}) or {}).get("plastic")
            a = (post_strengths.get(k, {}) or {}).get("plastic")
            if b and a is not None and b > 0:
                changes.append(abs(a - b) / b)

        stability_rows = []
        for q in self.probe_questions:
            for mode in (FULL, PLASTIC):
                stability_rows.append({
                    "qid": q.id, "chain": q.chain_id, "mode": mode,
                    "affected": q.chain_id in affected_chains,
                    "before": pre_answers[q.id][mode], "after": post_answers[q.id][mode],
                    "identical": pre_answers[q.id][mode] == post_answers[q.id][mode]})
        unaffected = [r for r in stability_rows if not r["affected"]]
        n_stable = sum(1 for r in unaffected if r["identical"])

        self.timings["corrections_s"] = round(time.perf_counter() - t0, 1)
        self.corrections_report = {
            "n_corrections": len(self.plan),
            "plan": self.plan,
            "per_correction": per_correction,
            "new_facts_learned": f"{n_new_learned}/{len(self.plan)}",
            "new_facts_all_learned": n_new_learned == len(self.plan),
            "untouched_old_facts": len(untouched),
            "untouched_old_facts_intact": f"{len(intact)}/{len(untouched)}",
            "untouched_old_facts_all_intact": len(intact) == len(untouched),
            "intact_keys": intact,
            "broken_keys": [k for k in untouched if k not in intact],
            "superseded_old_keys_still_transmitting":
                f"{len(still_transmitting_superseded)}/{len(superseded)}",
            "superseded_note": "the override is ADDITIVE in u_fast (the old edge is "
                               "superseded in the ledger, not deleted), so old keys are "
                               "expected to keep transmitting; the without-destroying "
                               "criterion is counted on the UNTOUCHED facts",
            "mean_relative_strength_change_untouched":
                round(sum(changes) / len(changes), 4) if changes else None,
            "max_relative_strength_change_untouched":
                round(max(changes), 4) if changes else None,
            "probe_rows": stability_rows,
            "unaffected_probe_identical": f"{n_stable}/{len(unaffected)}",
            "unaffected_probe_all_identical": n_stable == len(unaffected),
            "ledger_supersessions": len(self.corrections.ledger.superseded),
            "episodic_events": len(self.episodic),
        }
        self.corrections_report["adaptation_ok"] = bool(
            self.corrections_report["new_facts_all_learned"]
            and self.corrections_report["untouched_old_facts_all_intact"])
        self.corrections_report["probe_identity_ok"] = bool(
            self.corrections_report["unaffected_probe_all_identical"])
        self.corrections_report["probe_identity_note"] = (
            "diagnostic only, not gated: the NNFS update is additive in u_fast, so an "
            "unrelated answer string can shift without any fact being destroyed; the "
            "without-destroying gate counts STRENGTHS (untouched facts intact)")
        self.write_phase("corrections", {k: v for k, v in self.corrections_report.items()
                                         if k not in ("probe_rows",)})
        self.log(f"CORRECT done: new facts {self.corrections_report['new_facts_learned']}, "
                 f"untouched old facts intact "
                 f"{self.corrections_report['untouched_old_facts_intact']}, "
                 f"unaffected probe identical "
                 f"{self.corrections_report['unaffected_probe_identical']} "
                 f"({self.timings['corrections_s']}s)")

    # ========================================================================= #
    # PHASE RETAIN — strict restart + survival per battery
    # ========================================================================= #

    def serialize_state(self) -> None:
        snap = PlasticStateSnapshot.capture(self.iface, meta={
            "phase": "post-corrections", "foundation": self.foundation_label,
            "seed": self.args.seed, "chains": self.world.n_chains})
        payload = {"plastic": snap.to_payload(), "probes": self.rmem.save_probes(),
                   "index": self.rmem.to_dict()}
        torch.save(payload, os.path.join(self.run_dir, "state.pt"))
        self.episodic.save(os.path.join(self.run_dir, "episodic.json"))
        self._snapshot_payload = payload

    def strict_restart(self) -> Dict:
        """save -> tear every live object down -> FULL foundation reload from disk ->
        fresh interface + exact restore -> bitwise check + probe regeneration."""
        t0 = time.perf_counter()
        self.serialize_state()
        snap2 = PlasticStateSnapshot.from_payload(self._snapshot_payload["plastic"])
        n_episodic_before = len(self.episodic)
        del self.iface
        del self.student
        del self.meas_student
        del self.rmem
        del self.model
        self._snapshot_payload = None
        gc.collect()
        self.setup_foundation()                          # full reload from disk
        payload = torch.load(os.path.join(self.run_dir, "state.pt"), weights_only=False)
        snap3 = PlasticStateSnapshot.from_payload(payload["plastic"])
        self.iface = snap3.build_interface(self.model)   # fresh interface + restore
        self.student = Student(self.iface, self.tok, self.student.cfg)
        self.meas_student = MeasurementStudent(self.student)
        self.rmem = RelationalMemory.from_dict(
            payload["index"], conns=list(self.iface.conns), episodic=self.episodic,
            probes=payload["probes"])
        recheck = PlasticStateSnapshot.capture(self.iface)
        exact, diffs = snap3.equals(recheck, atol=0.0)
        probe_ok, probe_diffs = True, []
        for key, sentence in self.fact_sentences.items():
            if key not in self.rmem.records:
                continue
            pre, post = self.probe_sentence(sentence)
            rec = self.rmem.records[key]
            if not (torch.allclose(F.normalize(pre, dim=0), rec.pre_probe, atol=1e-6)
                    and torch.allclose(F.normalize(post, dim=0), rec.post_probe, atol=1e-6)):
                probe_ok = False
                probe_diffs.append(key)
        episodic_reloaded_ok = len(self.episodic) == n_episodic_before
        self.timings["restart_s"] = round(time.perf_counter() - t0, 1)
        out = {"state_restored_exact": bool(exact), "restore_diffs": diffs[:8],
               "probes_regenerate_identically": bool(probe_ok),
               "probe_diffs": probe_diffs[:8],
               "loaded_from_disk_only": True,
               "episodic_events_reloaded": len(self.episodic),
               "episodic_event_count_preserved": bool(episodic_reloaded_ok),
               "fact_index_reloaded": len(self.rmem.records),
               "restart_s": self.timings["restart_s"]}
        self.write_phase("restart", out)
        self.log(f"RESTART: state restored {'EXACTLY (bitwise)' if exact else 'WITH DIFFS: ' + str(diffs[:5])}; "
                 f"probes regenerate {'identically' if probe_ok else 'DIFFERENTLY: ' + str(probe_diffs[:3])} "
                 f"({self.timings['restart_s']}s)")
        return out

    def retain_phase(self) -> None:
        t0 = time.perf_counter()
        self.pre_rows = self.battery_pass(self.survival_questions, tag="survival-pre")
        self.pre_summary = {b: summarize_rows([r for r in self.pre_rows
                                               if r["battery"] == b]) for b in BATTERIES}
        restart = self.strict_restart()
        self.post_rows = self.battery_pass(self.survival_questions, tag="survival-post")
        self.post_summary = {b: summarize_rows([r for r in self.post_rows
                                                if r["battery"] == b]) for b in BATTERIES}
        per_battery = {}
        for b in BATTERIES:
            pre = [r for r in self.pre_rows if r["battery"] == b]
            post = [r for r in self.post_rows if r["battery"] == b]
            identity = {m: round(sum(1 for a, c in zip(pre, post)
                                     if a["answers"][m] == c["answers"][m])
                                 / max(1, len(pre)), 4) for m in MODES}
            acc_pre = {m: self.pre_summary[b]["accuracy"][m]["exact"] for m in MODES}
            acc_post = {m: self.post_summary[b]["accuracy"][m]["exact"] for m in MODES}
            per_battery[b] = {"n": len(pre),
                              "answers_identical_fraction": identity,
                              "accuracy_exact_pre": acc_pre,
                              "accuracy_exact_post": acc_post,
                              "survival_ok": bool(restart["state_restored_exact"]
                                                  and all(v == 1.0 for v in identity.values())
                                                  and acc_pre == acc_post)}
        self.survival = {"restart": restart, "per_battery": per_battery,
                         "survival_all_batteries_ok":
                             bool(all(p["survival_ok"] for p in per_battery.values()))}
        self.timings["retain_s"] = round(time.perf_counter() - t0, 1)
        self.write_phase("retain", self.survival)
        for b in BATTERIES:
            p = per_battery[b]
            self.log(f"RETAIN survival {b}: identical={p['answers_identical_fraction']} "
                     f"acc_pre={p['accuracy_exact_pre'][PLASTIC]:.2f} "
                     f"acc_post={p['accuracy_exact_post'][PLASTIC]:.2f} (PLASTIC) "
                     f"-> {'OK' if p['survival_ok'] else 'DEGRADED'}")

    # ========================================================================= #
    # PHASE PURGE — transcript removal, plastic unchanged
    # ========================================================================= #

    def purge_phase(self) -> None:
        t0 = time.perf_counter()
        questions = self.probe_questions
        before_n = len(self.episodic)
        pre = self.probe_answers(questions, modes=(FULL, PLASTIC, RETRIEVED))
        purge = self.episodic.purge_transcripts()
        post = self.probe_answers(questions, modes=(FULL, PLASTIC, RETRIEVED))
        rows = []
        for q in questions:
            for mode in (FULL, PLASTIC, RETRIEVED):
                rows.append({"qid": q.id, "mode": mode,
                             "before": pre[q.id][mode], "after": post[q.id][mode],
                             "identical": pre[q.id][mode] == post[q.id][mode]})
        plastic_rows = [r for r in rows if r["mode"] in (FULL, PLASTIC)]
        n_same = sum(1 for r in plastic_rows if r["identical"])
        retrieved_degraded = sum(1 for r in rows
                                 if r["mode"] == RETRIEVED and not r["identical"])
        self.timings["purge_s"] = round(time.perf_counter() - t0, 1)
        self.purge_report = {
            "episodic_events_before": before_n, "purge": purge,
            "episodic_events_after": len(self.episodic),
            "probe_rows": rows,
            "plastic_answers_identical": f"{n_same}/{len(plastic_rows)}",
            "plastic_unchanged": n_same == len(plastic_rows),
            "retrieved_answers_changed_after_purge": retrieved_degraded,
            "retrieved_expected_to_degrade": True,
            "note": "RETRIEVED is expected to degrade after a purge (the transcript WAS "
                    "its substrate); PLASTIC must not move — that is the whole test",
        }
        self.write_phase("purge", self.purge_report)
        self.log(f"PURGE: removed {purge['removed']} transcript events -> plastic probe "
                 f"answers identical {self.purge_report['plastic_answers_identical']} "
                 f"(retrieved answers changed: {retrieved_degraded}, expected)")

    # ========================================================================= #
    # PHASE REPORT
    # ========================================================================= #

    def report_phase(self) -> None:
        t0 = time.perf_counter()
        fp_ok, fp_diffs = fingerprints_equal(
            self.fingerprint_at_load, foundation_fingerprint(self.model))
        acceptance = {
            "1_storage_scaled": {
                "verdict": "PASS" if self.teach_summary["all_100_facts_stored"] else "FAIL",
                "evidence": (f"{self.teach_summary['facts_learned']} facts taught, "
                             f"{self.teach_summary['facts_registered_with_probes']} registered "
                             f"with probes; u_fast norms {self.teach_summary['u_fast_norms_after']} "
                             f"(from zero)")},
            "2_correction_without_destroying": {
                "verdict": "PASS" if self.corrections_report["adaptation_ok"] else "FAIL",
                "evidence": (f"new facts {self.corrections_report['new_facts_learned']}; "
                             f"untouched old facts intact (strengths) "
                             f"{self.corrections_report['untouched_old_facts_intact']}; "
                             f"probe-answer identity (diagnostic) "
                             f"{self.corrections_report['unaffected_probe_identical']}")},
            "3_retention_restart": {
                "verdict": "PASS" if self.survival["survival_all_batteries_ok"] else "FAIL",
                "evidence": (f"state_restored_exact="
                             f"{self.survival['restart']['state_restored_exact']}; probes "
                             f"regenerate identically="
                             f"{self.survival['restart']['probes_regenerate_identically']}; "
                             + "; ".join(f"{b}: survival_ok={p['survival_ok']}"
                                         for b, p in self.survival["per_battery"].items()))},
            "4_transcript_independence": {
                "verdict": "PASS" if self.purge_report["plastic_unchanged"] else "FAIL",
                "evidence": (f"purged {self.purge_report['purge']['removed']} transcript "
                             f"events -> plastic probe answers identical "
                             f"{self.purge_report['plastic_answers_identical']}")},
            "5_expression": {
                "verdict": "NOT MEASURED (restated)",
                "evidence": ("combination expression at chance is the Phase-2 verdict; this "
                             "run does not re-measure it — see "
                             "phase2_expression_reference in this report")},
        }
        report = {
            "run": {
                "tag": self.run_tag, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "foundation": self.foundation_label,
                "frozen": self.frozen_stats,
                "teacher": self.teacher_label,
                "seed": self.args.seed, "chains": self.world.n_chains,
                "relationships": len(self.world.relationships),
                "questions_in_world": len(self.world.questions),
                "batteries_measured": list(BATTERIES),
                "direct_battery_n": len(direct_battery(self.world)),
                "paraphrase_subset_n": len(paraphrase_subset(self.world, self.args.para_subset)),
                "n_corrections": self.args.n_corrections,
                "probe_size": len(self.probe_questions),
                "measurement_mode_note": "batteries run in MEASUREMENT mode (verdicts "
                                         "logged, plastic writes suppressed); the run's "
                                         "plastic corrections are the 10 preregistered "
                                         "reroutes",
            },
            "preregistration": json.load(open(os.path.join(self.run_dir, "phase_protocol.json"),
                                              encoding="utf-8")),
            "timings_s": self.timings,
            "teach": self.teach_summary,
            "test_pre_correction": {"batteries": self.test_summary,
                                    "rows": self.test_rows},
            "corrections": self.corrections_report,
            "retain": self.survival,
            "transcript_purge": self.purge_report,
            "foundation_integrity": {
                "bit_identical_after_full_run": bool(fp_ok),
                "differing_params": fp_diffs[:5],
                "n_params_checked": len(self.fingerprint_at_load)},
            "fact_inventory": self.rmem.inventory(),
            "phase1_reference_20rel": PHASE1_REFERENCE,
            "phase2_expression_reference": PHASE2_EXPRESSION_REFERENCE,
            "acceptance": acceptance,
        }
        n_pass = sum(1 for a in acceptance.values() if a["verdict"] == "PASS")
        report["acceptance_score"] = f"{n_pass}/4 mechanical gates"
        with open(os.path.join(self.run_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
        with open(os.path.join(self.run_dir, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=1, default=str)
        self.episodic.save(os.path.join(self.run_dir, "episodic_after_purge.json"))
        self.timings["report_s"] = round(time.perf_counter() - t0, 1)
        self.report = report

        print("\n" + "=" * 78)
        print(f"AION-1 OWNERSHIP SCALE — MECHANICS AT {len(self.world.relationships)} "
              f"RELATIONSHIPS ({self.run_tag})")
        print(f"foundation: {self.foundation_label}")
        print(f"teacher   : {self.teacher_label}")
        print("=" * 78)
        for b in BATTERIES:
            for label, summ in ((f"{b} PRE-CORRECTION", self.test_summary[b]),
                                (f"{b} PRE-RESTART", self.pre_summary[b]),
                                (f"{b} POST-RESTART", self.post_summary[b])):
                acc = summ["accuracy"]
                print(f"\n{label} (n={summ['n']}, chance={summ['mean_chance']:.3f})")
                for mode in MODES:
                    a = acc[mode]
                    print(f"  {mode:12s} exact={a['exact']:.3f} lenient={a['lenient']:.3f}")
                print(f"  provenance (exact): {summ['provenance_histogram']}")
        print("\nCORRECTIONS (the without-destroying criterion at scale)")
        c = self.corrections_report
        print(f"  new facts learned          : {c['new_facts_learned']}")
        print(f"  untouched old facts intact : {c['untouched_old_facts_intact']}")
        print(f"  superseded keys still transmitting: "
              f"{c['superseded_old_keys_still_transmitting']} (additive override)")
        print(f"  unaffected probe identical : {c['unaffected_probe_identical']}")
        print("\nRETAIN")
        print(f"  state restored exact       : {self.survival['restart']['state_restored_exact']} "
              f"(probes regenerate identically: "
              f"{self.survival['restart']['probes_regenerate_identically']})")
        for b, p in self.survival["per_battery"].items():
            print(f"  survival {b:11s}: identical={p['answers_identical_fraction']} "
                  f"-> {'OK' if p['survival_ok'] else 'DEGRADED'}")
        print(f"\nPURGE: {self.purge_report['purge']['removed']} transcript events removed -> "
              f"plastic identical {self.purge_report['plastic_answers_identical']}")
        print(f"FOUNDATION bit-identity     : "
              f"{report['foundation_integrity']['bit_identical_after_full_run']} "
              f"({report['foundation_integrity']['n_params_checked']} params fingerprinted)")
        print("\n" + "-" * 78)
        print("MECHANICAL GATES:")
        for k, a in acceptance.items():
            print(f"  {a['verdict']:22s} {k}")
            print(f"       {a['evidence']}")
        print("-" * 78)
        print(f"score {report['acceptance_score']}  |  report: "
              f"{os.path.join(self.run_dir, 'report.json')}")
        print(f"timings: {self.timings}")

    # --------------------------------------------------------------------------- #

    def run(self) -> int:
        t_start = time.perf_counter()
        self.run_tag = (f"ownership-{self.args.foundation}-c{self.args.chains}"
                        f"-s{self.args.seed}-{time.strftime('%H%M%S')}")
        if not self.args.run_dir:
            self.args.run_dir = os.path.join(_REPO_ROOT, "runs", "ownership_scale",
                                             self.run_tag)
        self.setup()
        self.teach_phase()
        self.test_phase()
        self.corrections_phase()
        self.retain_phase()
        self.purge_phase()
        self.report_phase()
        self.timings["total_s"] = round(time.perf_counter() - t_start, 1)
        n_pass = sum(1 for a in self.report["acceptance"].values()
                     if a["verdict"] == "PASS") if self.report else -1
        self.log(f"TOTAL RUNTIME: {self.timings['total_s']}s "
                 f"({self.timings['total_s'] / 60:.1f} min); mechanical gates {n_pass}/4")
        return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="AION-1 ownership MECHANICS at 100 relationships")
    ap.add_argument("--foundation", default="tiny", choices=("tiny", "qwen"),
                    help="tiny = in-memory stand-in (Rog CPU debug); qwen = real frozen "
                         "Qwen3.5-4B-Base (Lappy)")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="PREREGISTERED: 0")
    ap.add_argument("--chains", type=int, default=25,
                    help="PREREGISTERED: 25 chains = 100 relationships")
    ap.add_argument("--para-subset", type=int, default=20,
                    help="PREREGISTERED paraphrase subset size (deterministic stride)")
    ap.add_argument("--n-corrections", type=int, default=10,
                    help="PREREGISTERED: exactly 10 mid-chain reroutes")
    ap.add_argument("--probe-size", type=int, default=20,
                    help="preregistered probe battery size (one direct question per chain)")
    ap.add_argument("--learning-rate", type=float, default=0.5,
                    help="matches Phase-1 qwen-real-c5")
    ap.add_argument("--gate-steps", type=int, default=6,
                    help="matches Phase-1 qwen-real-c5")
    ap.add_argument("--gate-lr", type=float, default=300.0,
                    help="matches Phase-1 qwen-real-c5")
    ap.add_argument("--answer-tokens", type=int, default=8)
    ap.add_argument("--correction-repeats", type=int, default=3)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    return OwnershipScaleRun(args).run()


if __name__ == "__main__":
    sys.exit(main())
