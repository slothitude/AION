"""AION-1 PHASE 1 — the first real conversational learning run (owner-declared, C-variant).

Four phases, one process:

  TEACH   generate_world(seed) -> N chains (default 5 chains / 20 relationships, the owner's
          original criterion; --full scales to 25 chains / 100) -> chain lessons taught through
          DialogueSession + TeacherClient (structured-JSON protocol) -> lesson_extractor ->
          pathway.nnfs_update per fact (facts enter the PLASTIC system) + gate steps (the
          interface's residual-gate half) -> episodic record for provenance + fact index
          (RelationalMemory) so strengths are measurable.

  TEST    the four batteries (direct / combination / paraphrase / intervention) asked through
          session.ask + grader; every question answered once per PROVENANCE BUCKET by ablating
          one bucket at a time (aion/experiments/retention.py ablated(): PLASTIC = u_fast alone,
          CONSOLIDATED = theta_slow alone, FULL = natural blend, RETRIEVED = pathway ablated,
          episodic store only). Scored per battery against the world graph's gold answers.

  RETAIN  strict restart: plastic state serialized -> foundation torn down -> FULL reload from
          disk -> fresh interface + exact restore -> bitwise equality check -> paraphrase
          battery again (survival). Then EXACTLY ONE correction (a mid-chain reroute through
          the CorrectionHandler's intervention path = plastic override with repeats>1) ->
          verify the new fact learned AND the old facts intact.

  REPORT  per-battery accuracy per bucket vs chance, provenance histogram, restart/correction
          evidence, foundation bit-identity, and the transcript-REMOVAL check (purge episodic
          -> plastic answers unchanged).

Honesty rules this run follows:
  * every number is labelled with which teacher and which foundation produced it;
  * the battery pass runs in MEASUREMENT mode: verdicts and corrections are logged in the
    transcript but NOT applied as plastic updates, so the run contains exactly ONE plastic
    correction (the RETAIN-phase reroute) and the batteries measure one fixed state;
  * grading references are the world graph's gold answers (owner ground truth: every battery
    answer is graph-derivable); the real teacher endpoint delivers the lessons through the
    structured protocol (TeacherClient.teach) — this is stated, never hidden.

Run:
  python aion/experiments/teacher_student.py                     # tiny stand-in + scripted teacher (Rog CPU debug)
  python aion/experiments/teacher_student.py --foundation qwen --teacher auto \
      --local-dir /mnt/seagate/aion-models/Qwen3.5-4B-Base       # REAL run (Lappy)
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.dialogue.conversation import DialogueSession, default_grader          # noqa: E402
from aion.dialogue.correction import CorrectionHandler                          # noqa: E402
from aion.dialogue.lesson_extractor import Fact, RuleExtractor                  # noqa: E402
from aion.dialogue.student import Student, StudentConfig                        # noqa: E402
from aion.dialogue.teacher import (                                             # noqa: E402
    ScriptedTeacher, TeacherClient, TeacherConfig, TeacherTurn, DEFAULT_BASE_URL)
from aion.worlds.generator import (                                             # noqa: E402
    BATTERY_COMBINATION, BATTERY_DIRECT, BATTERY_INTERVENTION, BATTERY_PARAPHRASE,
    DIRECT_ASK, generate_world, verify_world)
from aion.memory.episodic import EpisodicStore                                  # noqa: E402
from aion.memory.relational import RelationalMemory, relation_key               # noqa: E402
from aion.memory.state import PlasticStateSnapshot                              # noqa: E402
from aion.experiments.retention import (                                        # noqa: E402
    ablated, CONSOLIDATED, FULL, MODES, PLASTIC, RETRIEVED, STRONGEST_FIRST)
from aion.relational.interface import (                                         # noqa: E402
    InterfaceConfig, RelationalInterface, fingerprints_equal, foundation_fingerprint)
from aion.foundation.pretrained_loader import DEFAULT_LOCAL_DIR, load_frozen_base  # noqa: E402

BATTERIES = (BATTERY_DIRECT, BATTERY_COMBINATION, BATTERY_PARAPHRASE, BATTERY_INTERVENTION)

# teacher alias discovery: plain HTTP use of the OpenAI-compatible endpoint ONLY.
# NEVER reads llama-swap config, NEVER touches model flags (OWNER LAW).
TEACHER_ALIAS_PREFERENCE = ("qwen35-4b", "qwen38", "lfm3-omni", "inkling2",
                            "tiel-coder", "lfm3", "d1-omni")


# --------------------------------------------------------------------------- #
# Teacher endpoint discovery (GET /v1/models — plain HTTP, nothing else)
# --------------------------------------------------------------------------- #

def discover_teacher(base_url: str, timeout_s: float = 15.0) -> Dict:
    """Discover the model alias at runtime via GET {base_url}/models."""
    import requests
    try:
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as exc:                       # endpoint down -> caller falls back
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}",
                "alias": None, "ids": [], "status": {}}
    ids = [m.get("id") for m in data if m.get("id")]
    status = {m.get("id"): (m.get("status") or {}).get("value") for m in data}
    alias = next((a for a in TEACHER_ALIAS_PREFERENCE if a in ids),
                 ids[0] if ids else None)
    return {"reachable": True, "error": None, "alias": alias, "ids": ids, "status": status}


# --------------------------------------------------------------------------- #
# Teachers: real endpoint for lesson delivery, world graph for grading references
# --------------------------------------------------------------------------- #

class WorldTeacher(TeacherClient):
    """TeacherClient with the world graph as the grading reference.

    teach()/correct() go to the REAL endpoint (the structured protocol channel — the
    conversational lane is genuinely exercised). answer() returns the graph-derivable gold
    as a structured TeacherTurn (the owner's ground truth; a stateless endpoint cannot know
    the invented world, and battery scoring must be against the generator's gold)."""

    def __init__(self, config: TeacherConfig, gold_map: Dict[str, str]):
        super().__init__(config)
        self.gold_map = gold_map
        self.teach_log: List[Dict] = []

    def teach(self, lesson_text: str) -> TeacherTurn:
        turn = super().teach(lesson_text)
        self.teach_log.append({"lesson": lesson_text, "structured": turn.structured,
                               "error": turn.error, "content_chars": len(turn.content)})
        return turn

    def answer(self, question: str) -> TeacherTurn:
        gold = self.gold_map.get(question, "")
        content = json.dumps({"action": "answer", "question": question, "answer": gold})
        return TeacherTurn(action="answer", content=content,
                           parsed={"action": "answer", "question": question, "answer": gold},
                           structured=bool(gold), meta={"reference": "world_graph"})

    def correct(self, mistake: str, correction: str) -> TeacherTurn:
        # battery-verdict bookkeeping only in this harness (the run's single real
        # correction is the RETAIN-phase reroute, which needs no endpoint turn)
        content = json.dumps({"action": "correct", "mistake": mistake, "correction": correction})
        return TeacherTurn(action="correct", content=content,
                           parsed=json.loads(content), structured=True,
                           meta={"note": "logged only (measurement mode)"})


# --------------------------------------------------------------------------- #
# Measurement-mode student proxy: answers are real, plastic writes are suppressed
# --------------------------------------------------------------------------- #

class MeasurementStudent:
    """Battery-pass student proxy.

    answer() delegates to the REAL student (real pathway forward under the ablation
    context). learn_fact() is a NO-OP on plastic state: corrections fired during battery
    verdicts are logged in the transcript but never written, so TEST measures one fixed
    learned state and the run contains exactly ONE plastic correction (RETAIN).
    """

    def __init__(self, real: Student):
        self.real = real
        self.cfg = real.cfg
        self.suppressed: List[str] = []

    def answer(self, question: str) -> str:
        return self.real.answer(question)

    def learn_fact(self, fact: Fact, repeats: int = 1) -> Dict:
        text = self.real.render_fact(fact)
        self.suppressed.append(text)
        return {"text": text, "suppressed": True, "repeats": repeats}


# --------------------------------------------------------------------------- #
# Tiny-standin tokenizer (Rog CPU debug lane; the real run uses the Qwen tokenizer)
# --------------------------------------------------------------------------- #

_CHAR_ALPHABET = "\x00" + "abcdefghijklmnopqrstuvwxyz .,-'"


class CharTokenizer:
    """Deterministic char tokenizer (id 0 = EOS), same contract as aion's stub."""

    eos_id = 0
    vocab_size = len(_CHAR_ALPHABET)

    def encode(self, text):
        return [_CHAR_ALPHABET.find(ch) if ch in _CHAR_ALPHABET else _CHAR_ALPHABET.index(" ")
                for ch in text.lower()]

    def decode(self, ids):
        return "".join(_CHAR_ALPHABET[i] for i in ids)


def build_tiny_foundation(seed: int = 0):
    """4-layer Llama stand-in with the module-tree shape the interface discovers."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=CharTokenizer.vocab_size, hidden_size=64,
                      intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg), CharTokenizer()


# --------------------------------------------------------------------------- #
# Grading: strict (default_grader) + two labelled relaxations
# --------------------------------------------------------------------------- #

_ROLE_WORDS = {"relay", "device", "the", "a", "an"}


def _words(s: str) -> List[str]:
    return [w for w in "".join(c if c.isalnum() or c in "-'" else " " for c in s.lower()).split()
            if w not in _ROLE_WORDS]


def lenient_match(pred: str, gold: str) -> bool:
    """Role-word-stripped whole-answer match (e.g. 'relay Kimo' == gold 'Kimo'); yes/no gold
    matches when the prediction STARTS with it."""
    pw, gw = _words(pred or ""), _words(gold or "")
    if " ".join(gw) in ("yes", "no"):
        return bool(pw) and pw[0] == gw[0]
    return bool(pw) and " ".join(pw) == " ".join(gw)


def contains_match(pred: str, gold: str) -> bool:
    """Every gold word appears in the prediction (weak evidence the right entity is present)."""
    pw, gw = _words(pred or ""), _words(gold or "")
    if " ".join(gw) in ("yes", "no"):
        return lenient_match(pred, gold)
    return bool(gw) and all(w in pw for w in gw)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

class TeacherStudentRun:
    """AION-1 Phase 1 harness: TEACH -> TEST -> RETAIN -> REPORT."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tok = None
        self.iface: Optional[RelationalInterface] = None
        self.student: Optional[Student] = None
        self.fingerprint_at_load: Optional[Dict] = None
        self.teacher = None                            # ScriptedTeacher | WorldTeacher
        self.teacher_label = "scripted"
        self.teacher_discovery: Dict = {"reachable": False, "alias": None, "ids": []}
        self.meas_student: Optional[MeasurementStudent] = None
        self.episodic = EpisodicStore(clock=lambda: 0.0)   # deterministic law
        self.rmem: Optional[RelationalMemory] = None
        self.sessions: List[Dict] = []                 # transcript dump per session
        self.timings: Dict[str, float] = {}
        self.report: Dict = {}
        self.fact_sentences: Dict[str, str] = {}       # key -> canonical sentence
        self.fact_chain: Dict[str, int] = {}

    # -- plumbing ------------------------------------------------------------- #

    def log(self, msg: str) -> None:
        print(f"[teacher-student] {msg}", flush=True)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tok.encode(text)], dtype=torch.long)

    def make_probe(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(pre, post) hidden-state probe for a sentence, from pair 0 (retention-lane pattern)."""
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
        self.rmem.register(key, 0, pre, post, meta={"sentence": sentence, "chain": chain_id,
                                                    **(meta or {})})
        self.fact_sentences[key] = sentence
        self.fact_chain[key] = chain_id

    # -- setup ------------------------------------------------------------------ #

    def setup_foundation(self) -> None:
        t0 = time.perf_counter()
        if self.args.foundation == "qwen":
            # NOTE: aion/foundation/pretrained_loader.load_frozen_base has a latent
            # TypeError (FrozenBase(model_id=...) built before the required `model` field
            # is supplied) — READ-ONLY landed code, reported to the owner, not edited here.
            # This mirrors its proven behaviour exactly (same load, freeze, fingerprint).
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from aion.relational.interface import freeze_base
            src = self.args.local_dir
            model = AutoModelForCausalLM.from_pretrained(
                src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None,
                local_files_only=True).to(self.args.device)
            model.eval()
            tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
            stats = freeze_base(model)                      # asserts requires_grad=False everywhere
            assert not any(p.requires_grad for p in model.parameters())
            self.model, self.tok = model, tok
            self.fingerprint_at_load = foundation_fingerprint(model)
            self.frozen_stats = {"frozen_tensors": stats["frozen_params"],
                                 "frozen_params": sum(p.numel() for p in model.parameters())}
            self.foundation_label = f"Qwen3.5-4B-Base @ {src} (CPU BF16, FROZEN)"
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

    def setup_teacher(self) -> None:
        gold_map = {q.text: q.answer for q in self.world.questions}
        if self.args.teacher == "scripted":
            self.teacher = ScriptedTeacher(reference_answers=gold_map)
            self.teacher_label = "ScriptedTeacher (offline stub)"
            self.teacher_discovery = {"reachable": False, "alias": None,
                                      "note": "scripted by request"}
            return
        disc = discover_teacher(self.args.base_url)
        if not disc["reachable"] or not disc["alias"]:
            self.teacher = ScriptedTeacher(reference_answers=gold_map)
            self.teacher_label = "ScriptedTeacher (FALLBACK: endpoint unreachable)"
            self.teacher_discovery = {**disc, "fallback": "scripted"}
            self.log(f"teacher endpoint UNREACHABLE ({disc.get('error')}) -> ScriptedTeacher "
                     f"fallback (labelled)")
            return
        alias = self.args.teacher_alias or disc["alias"]
        cfg = TeacherConfig(base_url=self.args.base_url, model=alias,
                            temperature=0.0, max_tokens=self.args.teacher_max_tokens)
        self.teacher = WorldTeacher(cfg, gold_map)
        self.teacher_label = f"TeacherClient endpoint={self.args.base_url} alias={alias!r} " \
                             f"(lessons via structured protocol; grading refs = world graph gold)"
        self.teacher_discovery = {**disc, "alias_used": alias}
        self.log(f"teacher endpoint reachable: aliases={disc['ids']} -> using {alias!r} "
                 f"(status={disc['status'].get(alias)})")

    def setup(self) -> None:
        t0 = time.perf_counter()
        n_chains = 25 if self.args.full else self.args.chains
        self.world = generate_world(seed=self.args.seed, n_chains=n_chains)
        failures = verify_world(self.world)
        self.timings["world_gen_s"] = round(time.perf_counter() - t0, 2)
        self.log(f"world: seed={self.args.seed} chains={n_chains} "
                 f"relationships={len(self.world.relationships)} "
                 f"questions={len(self.world.questions)} "
                 f"verify={'OK (all answers graph-derivable)' if not failures else failures[:2]}")
        if failures:
            raise RuntimeError(f"world ground truth inconsistent: {failures[:3]}")
        self.setup_foundation()
        self.setup_teacher()
        os.makedirs(self.run_dir, exist_ok=True)
        self.world.save_json(os.path.join(self.run_dir, "world.json"))

    @property
    def run_dir(self) -> str:
        return self.args.run_dir

    # -- transcript dump ---------------------------------------------------------- #

    def dump_session(self, session: DialogueSession, tag: str) -> None:
        self.sessions.append({
            "tag": tag, "transcript_id": session.transcript_id,
            "turns": [{"index": t.index, "kind": t.kind, "provenance": t.provenance,
                       "content": t.content, "transcript_id": t.transcript_id, "meta": t.meta}
                      for t in session.turns],
            "events": [e.to_dict() for e in session.events]})

    # ========================================================================= #
    # PHASE TEACH
    # ========================================================================= #

    def teach_phase(self) -> None:
        t0 = time.perf_counter()
        self.corrections = CorrectionHandler()
        u_before = [c.plasticity.u_fast.norm().item() for c in self.iface.conns]
        n_facts = 0
        for c in range(self.world.n_chains):
            rels = self.world.relationships_of_chain(c)
            lesson = " ".join(r.sentence for r in rels)
            learn_session = DialogueSession(
                teacher=self.teacher, student=self.student, extractor=RuleExtractor(),
                corrections=self.corrections, transcript_id=f"teach-chain{c:02d}")
            start_learned = self.student.n_facts_learned
            facts = learn_session.teach(lesson)          # teacher.teach -> extractor ->
            uids = list(range(start_learned + 1,         # student.learn_fact (NNFS plasticity)
                              start_learned + len(facts) + 1))
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
            # protocol comprehension round, measured (no plastic writes): teach -> ask -> verdict
            probe_q = self.world.questions[4 * c]        # the chain's first direct question
            meas_session = DialogueSession(
                teacher=self.teacher, student=self.meas_student, extractor=RuleExtractor(),
                corrections=CorrectionHandler(), transcript_id=f"teach-check-chain{c:02d}")
            event = meas_session.ask(probe_q.text)
            self.dump_session(learn_session, f"teach-chain{c:02d}")
            self.dump_session(meas_session, f"teach-check-chain{c:02d}")
            gammas = [round(float(g.detach()), 6) for g in self.iface.gates]
            self.log(f"teach chain {c}: {len(facts)} facts plastic (gamma={gammas}); "
                     f"comprehension '{probe_q.text}' -> student={event.student_answer!r} "
                     f"verdict={event.verdict}")
        u_after = [c.plasticity.u_fast.norm().item() for c in self.iface.conns]
        gammas_after = [round(float(g.detach()), 6) for g in self.iface.gates]
        self.timings["teach_s"] = round(time.perf_counter() - t0, 1)
        self.teach_summary = {
            "facts_learned": n_facts,
            "lessons": self.world.n_chains,
            "gate_steps_per_fact": self.args.gate_steps,
            "gate_lr": self.args.gate_lr,
            "nnfs_learning_rate": self.args.learning_rate,
            "u_fast_norms_before": [round(u, 6) for u in u_before],
            "u_fast_norms_after": [round(u, 6) for u in u_after],
            "gamma_after_teach": gammas_after,
            "teacher_teach_log": getattr(self.teacher, "teach_log", []),
            "n_facts_suppressed_in_checks": len(self.meas_student.suppressed),
        }
        self.log(f"TEACH done: {n_facts} facts entered the plastic pathway "
                 f"({self.timings['teach_s']}s); u_fast norms "
                 f"{[round(u, 4) for u in u_before]} -> {[round(u, 4) for u in u_after]}")

    # ========================================================================= #
    # Battery machinery (TEST + RETAIN reuse)
    # ========================================================================= #

    def retrieve_answer(self, question: str) -> str:
        """RETRIEVED bucket: pathway ablated, episodic store only (variant-B reading)."""
        hits = self.episodic.retrieve(question, k=1)
        if not hits or hits[0][1] <= 0.0:
            return ""
        ev, score = hits[0]
        return ev.tags.get("answer") or ev.text

    def battery_pass(self, questions: List, tag: str) -> List[Dict]:
        """Ask every question once per provenance bucket through session.ask; score vs gold."""
        t0 = time.perf_counter()
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
                        ev = session.ask(q.text)     # student answers through the pathway
                    answers[mode] = ev.student_answer or ""
            n_pool = max(2, self.world.n_chains)
            chance = 0.5 if q.meta.get("kind") == "membership" else 1.0 / n_pool
            buckets = [m for m in STRONGEST_FIRST if default_grader(answers[m], q.answer)]
            if buckets:
                provenance = buckets[0]
            elif default_grader(answers[FULL], q.answer):
                provenance = "FULL_ONLY"
            else:
                provenance = "NONE"
            rows.append({
                "qid": q.id, "battery": q.battery, "question": q.text, "gold": q.answer,
                "answers": answers, "chance": round(chance, 4),
                "exact": {m: bool(default_grader(answers[m], q.answer)) for m in MODES},
                "lenient": {m: bool(lenient_match(answers[m], q.answer)) for m in MODES},
                "contains": {m: bool(contains_match(answers[m], q.answer)) for m in MODES},
                "provenance_exact": provenance,
                "provenance_lenient": next(
                    (m for m in STRONGEST_FIRST if lenient_match(answers[m], q.answer)),
                    "FULL_ONLY" if lenient_match(answers[FULL], q.answer) else "NONE"),
            })
            if (i + 1) % 10 == 0 or i + 1 == len(questions):
                self.log(f"  [{tag}] {i + 1}/{len(questions)} questions asked")
        self.dump_session(session, tag)
        self.timings[f"battery_{tag}_s"] = round(time.perf_counter() - t0, 1)
        return rows

    @staticmethod
    def battery_summary(rows: List[Dict]) -> Dict:
        n = max(1, len(rows))
        out = {"n": len(rows),
               "mean_chance": round(sum(r["chance"] for r in rows) / n, 4),
               "accuracy": {}}
        for mode in MODES:
            out["accuracy"][mode] = {
                "exact": round(sum(r["exact"][mode] for r in rows) / n, 4),
                "lenient": round(sum(r["lenient"][mode] for r in rows) / n, 4),
                "contains": round(sum(r["contains"][mode] for r in rows) / n, 4)}
            out["accuracy"][mode]["empty_answer_fraction"] = round(
                sum(1 for r in rows if not (r["answers"][mode] or "").strip()) / n, 4)
        hist: Dict[str, int] = {}
        for r in rows:
            hist[r["provenance_exact"]] = hist.get(r["provenance_exact"], 0) + 1
        out["provenance_histogram"] = hist
        return out

    # ========================================================================= #
    # PHASE TEST
    # ========================================================================= #

    def test_phase(self) -> None:
        questions = list(self.world.questions)
        if self.args.max_per_battery > 0:
            # direct battery always runs in full (it IS the acquisition criterion, and it is
            # small: one question per taught relationship); the rest are trimmed
            trimmed = []
            for b in BATTERIES:
                qs = [q for q in questions if q.battery == b]
                trimmed += qs if b == BATTERY_DIRECT else qs[:self.args.max_per_battery]
            questions = trimmed
        t0 = time.perf_counter()
        self.test_rows = self.battery_pass(questions, tag="test")
        self.test_summary = {b: self.battery_summary([r for r in self.test_rows
                                                      if r["battery"] == b])
                             for b in BATTERIES}
        hist: Dict[str, int] = {}
        for r in self.test_rows:
            hist[r["provenance_exact"]] = hist.get(r["provenance_exact"], 0) + 1
        self.test_summary["ALL"] = {"n": len(self.test_rows),
                                    "provenance_histogram": hist}
        self.timings["test_s"] = round(time.perf_counter() - t0, 1)
        self.log("TEST done — per-battery accuracy (exact match, per provenance bucket):")
        for b in BATTERIES:
            s = self.test_summary[b]
            acc = s["accuracy"]
            self.log(f"  {b:13s} n={s['n']:3d} chance={s['mean_chance']:.3f} | "
                     + " ".join(f"{m}:exact={acc[m]['exact']:.2f}" for m in MODES))

    # ========================================================================= #
    # PHASE RETAIN — strict restart, then ONE correction
    # ========================================================================= #

    def probe_set(self) -> List:
        """Direct questions used as the plastic-stability probe (chain 0's four + a spread)."""
        k = max(4, self.args.probe_size)
        chain0 = [q for q in self.world.questions
                  if q.battery == BATTERY_DIRECT and q.chain_id == 0]
        rest = [q for q in self.world.questions
                if q.battery == BATTERY_DIRECT and q.chain_id != 0]
        step = max(1, len(rest) // max(1, k - len(chain0)))
        return chain0 + rest[::step][:max(0, k - len(chain0))]

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

    def strengths_snapshot(self) -> Dict[str, Dict[str, Optional[float]]]:
        return {key: self.rmem.strengths(key) for key in sorted(self.rmem.records)}

    def serialize_state(self) -> Dict:
        snap = PlasticStateSnapshot.capture(self.iface, meta={
            "phase": "post-teach", "foundation": self.foundation_label,
            "seed": self.args.seed, "chains": self.world.n_chains})
        payload = {"plastic": snap.to_payload(), "probes": self.rmem.save_probes(),
                   "index": self.rmem.to_dict()}
        torch.save(payload, os.path.join(self.run_dir, "state.pt"))
        self.episodic.save(os.path.join(self.run_dir, "episodic.json"))
        return {"snap": snap, "payload": payload}

    def strict_restart(self) -> Dict:
        """save -> tear the foundation down -> FULL reload from disk -> fresh interface +
        exact restore -> bitwise check. The foundation is frozen weights on disk and simply
        reloads; everything AION-side round-trips through files into brand-new objects."""
        t0 = time.perf_counter()
        saved = self.serialize_state()
        snap: PlasticStateSnapshot = saved["snap"]
        # -- shutdown: drop every live object -- #
        del self.iface
        del self.student
        del self.meas_student
        del self.rmem
        del self.model
        gc.collect()
        # -- restart: foundation reloads from disk (or the seeded stub rebuilds identically) -- #
        self.setup_foundation()
        payload = torch.load(os.path.join(self.run_dir, "state.pt"), weights_only=False)
        snap2 = PlasticStateSnapshot.from_payload(payload["plastic"])
        self.iface = snap2.build_interface(self.model)      # fresh interface + restore
        self.student = Student(self.iface, self.tok, self.student.cfg)
        self.meas_student = MeasurementStudent(self.student)
        self.rmem = RelationalMemory.from_dict(
            payload["index"], conns=list(self.iface.conns), episodic=self.episodic,
            probes=payload["probes"])
        recheck = PlasticStateSnapshot.capture(self.iface)
        exact, diffs = snap2.equals(recheck, atol=0.0)
        # probes are hidden states of the FROZEN foundation -> must regenerate identically
        # (the index stores them unit-normalized, so compare in the same normalized space)
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
        self.timings["restart_s"] = round(time.perf_counter() - t0, 1)
        self.log(f"RESTART: foundation fully reloaded, plastic state restored "
                 f"{'EXACTLY (bitwise)' if exact else 'WITH DIFFS: ' + str(diffs[:5])}; "
                 f"probes regenerate {'identically' if probe_ok else 'DIFFERENTLY: ' + str(probe_diffs[:3])} "
                 f"({self.timings['restart_s']}s)")
        return {"state_restored_exact": bool(exact), "restore_diffs": diffs[:8],
                "probes_regenerate_identically": bool(probe_ok),
                "probe_diffs": probe_diffs[:8],
                "loaded_from_disk_only": True,
                "episodic_events_reloaded": len(self.episodic),
                "fact_index_reloaded": len(self.rmem.records)}

    def correction_step(self) -> Dict:
        """EXACTLY ONE correction: reroute chain 0's conduit to another relay through the
        CorrectionHandler's intervention path (plastic override, repeats>1)."""
        t0 = time.perf_counter()
        chain0 = self.world.chains[0]
        conduit, old_relay = chain0[1], chain0[2]
        new_relay = self.world.relays[1 % self.world.n_chains]
        assert new_relay != old_relay
        questions = self.probe_set()
        route_q = next(q for q in self.world.questions if q.text ==
                       f"Where does {conduit} route?")
        pre_answers = self.probe_answers(questions)
        pre_strengths = self.strengths_snapshot()

        session = DialogueSession(
            teacher=self.teacher, student=self.student, extractor=RuleExtractor(),
            corrections=self.corrections, transcript_id="retain-correction")
        sentence = f"{conduit} is rerouted to relay {new_relay}."
        session.teach(sentence)          # intervention -> CorrectionHandler.apply_reroute
        self.dump_session(session, "retain-correction")
        self.episodic.record(kind="correction", text=sentence, channel="transcript",
                             teacher=self.teacher_label, transcript_id=session.transcript_id,
                             tags={"answer": new_relay,
                                   "fact_key": relation_key(conduit, "routes-to", new_relay),
                                   "replaces": relation_key(conduit, "routes-to", old_relay),
                                   "kind": "reroute"})
        self.register_fact(relation_key(conduit, "routes-to", new_relay), sentence, 0,
                           meta={"correction": True})
        old_key = relation_key(conduit, "routes-to", old_relay)
        if old_key in self.rmem.records:
            self.rmem.records[old_key].superseded_by = relation_key(
                conduit, "routes-to", new_relay)

        post_answers = self.probe_answers(questions)
        post_strengths = self.strengths_snapshot()
        new_plastic = self.rmem.strength(relation_key(conduit, "routes-to", new_relay), blend=1.0)
        old_plastic = self.rmem.strength(old_key, blend=1.0)

        # old-fact intactness: every OTHER registered fact still transmits, and probe answers
        # on questions NOT about the corrected edge are unchanged
        other_keys = [k for k in sorted(self.rmem.records)
                      if k != relation_key(conduit, "routes-to", new_relay)]
        still_present = [k for k in other_keys
                         if (post_strengths.get(k, {}) or {}).get("plastic", 0.0) is not None
                         and (post_strengths[k]["plastic"] or 0.0) > self.rmem.strength_threshold]
        affected_probe_ids = {q.id for q in questions
                              if q.meta.get("subject") in (conduit, chain0[0])}
        stability_rows = []
        for q in questions:
            for mode in (FULL, PLASTIC):
                stability_rows.append({
                    "qid": q.id, "mode": mode, "affected": q.id in affected_probe_ids,
                    "before": pre_answers[q.id][mode], "after": post_answers[q.id][mode],
                    "identical": pre_answers[q.id][mode] == post_answers[q.id][mode]})
        unaffected = [r for r in stability_rows if not r["affected"]]
        n_stable = sum(1 for r in unaffected if r["identical"])
        strength_changes = []
        for k in other_keys:
            b = (pre_strengths.get(k, {}) or {}).get("plastic")
            a = (post_strengths.get(k, {}) or {}).get("plastic")
            if b and a is not None and b > 0:
                strength_changes.append(abs(a - b) / b)
        self.timings["correction_s"] = round(time.perf_counter() - t0, 1)
        out = {
            "correction_sentence": sentence,
            "new_key": relation_key(conduit, "routes-to", new_relay),
            "old_key": old_key,
            "new_fact_plastic_strength": round(new_plastic, 6) if new_plastic is not None else None,
            "old_fact_plastic_strength_after": round(old_plastic, 6) if old_plastic is not None else None,
            "new_fact_learned": bool(new_plastic is not None
                                     and new_plastic > self.rmem.strength_threshold),
            "old_facts_present_after": f"{len(still_present)}/{len(other_keys)}",
            "old_facts_all_present": len(still_present) == len(other_keys),
            "mean_relative_strength_change_old_facts": round(sum(strength_changes) / len(strength_changes), 4) if strength_changes else None,
            "max_relative_strength_change_old_facts": round(max(strength_changes), 4) if strength_changes else None,
            "probe_answers": stability_rows,
            "unaffected_probe_identical": f"{n_stable}/{len(unaffected)}",
            "unaffected_probe_all_identical": n_stable == len(unaffected),
            "route_answer_before": pre_answers.get(route_q.id, {}).get(PLASTIC),
            "route_answer_after": post_answers.get(route_q.id, {}).get(PLASTIC),
            "ledger_supersession": [e.to_dict() for e in self.corrections.ledger.superseded],
        }
        out["adaptation_ok"] = bool(out["new_fact_learned"] and out["old_facts_all_present"]
                                    and out["unaffected_probe_all_identical"])
        self.log(f"CORRECTION: {sentence} new_fact_learned={out['new_fact_learned']} "
                 f"(plastic strength {out['new_fact_plastic_strength']}), "
                 f"old facts present {out['old_facts_present_after']}, "
                 f"unaffected probe identical {out['unaffected_probe_identical']} "
                 f"({self.timings['correction_s']}s)")
        return out

    def retain_phase(self) -> None:
        t0 = time.perf_counter()
        para_qs = [q for q in self.world.questions if q.battery == BATTERY_PARAPHRASE]
        if self.args.max_per_battery > 0:
            para_qs = para_qs[:self.args.max_per_battery]
        pre_rows = self.battery_pass(para_qs, tag="paraphrase-pre-restart")
        self.pre_paraphrase = self.battery_summary(pre_rows)
        restart = self.strict_restart()
        post_rows = self.battery_pass(para_qs, tag="paraphrase-post-restart")
        self.post_paraphrase = self.battery_summary(post_rows)
        identical = {}
        for mode in MODES:
            same = sum(1 for a, b in zip(pre_rows, post_rows)
                       if a["answers"][mode] == b["answers"][mode])
            identical[mode] = round(same / max(1, len(pre_rows)), 4)
        acc_survival = {m: (self.pre_paraphrase["accuracy"][m]["exact"],
                            self.post_paraphrase["accuracy"][m]["exact"]) for m in MODES}
        survival_ok = all(a == b for a, b in acc_survival.values()) and \
            all(v == 1.0 for m, v in identical.items() if m != RETRIEVED) and \
            restart["state_restored_exact"]
        self.paraphrase_survival = {
            "accuracy_exact_pre": {m: v[0] for m, v in acc_survival.items()},
            "accuracy_exact_post": {m: v[1] for m, v in acc_survival.items()},
            "answers_identical_fraction": identical,
            "state_restored_exact": restart["state_restored_exact"],
            "survival_ok": bool(survival_ok)}
        self.log(f"RETAIN paraphrase survival: pre={self.paraphrase_survival['accuracy_exact_pre']} "
                 f"post={self.paraphrase_survival['accuracy_exact_post']} "
                 f"identical={identical} -> {'OK' if survival_ok else 'DEGRADED'}")
        self.correction = self.correction_step()
        self.restart_report = restart
        self.timings["retain_s"] = round(time.perf_counter() - t0, 1)

    # ========================================================================= #
    # PHASE REPORT
    # ========================================================================= #

    def transcript_removal_check(self) -> Dict:
        """Purge the episodic transcript -> plastic answers must be UNCHANGED."""
        before = len(self.episodic)
        questions = self.probe_set()
        pre = self.probe_answers(questions)
        purge = self.episodic.purge_transcripts()
        post = self.probe_answers(questions)
        rows = []
        for q in questions:
            for mode in (FULL, PLASTIC):
                rows.append({"qid": q.id, "mode": mode,
                             "before": pre[q.id][mode], "after": post[q.id][mode],
                             "identical": pre[q.id][mode] == post[q.id][mode]})
        n_same = sum(1 for r in rows if r["identical"])
        retrieved_before = self.retrieve_answer(self.probe_set()[0].text) if rows else ""
        out = {"episodic_events_before": before, "purge": purge,
               "episodic_events_after": len(self.episodic),
               "probe_rows": rows,
               "plastic_answers_identical": f"{n_same}/{len(rows)}",
               "plastic_unchanged": n_same == len(rows),
               "retrieved_sample_after_purge": retrieved_before,
               "note": "RETRIEVED is expected to degrade after a purge (the transcript WAS "
                       "its substrate); PLASTIC must not move — that is the whole test"}
        self.log(f"TRANSCRIPT REMOVAL: purged {purge['removed']} transcript events -> "
                 f"plastic answers identical {out['plastic_answers_identical']}")
        return out

    def evaluate_acceptance(self) -> Dict:
        d = self.test_summary[BATTERY_DIRECT]
        c = self.test_summary[BATTERY_COMBINATION]
        chance_d = d["mean_chance"]
        acc_plastic = d["accuracy"][PLASTIC]["exact"]
        acc_plastic_contains = d["accuracy"][PLASTIC]["contains"]
        baseline = c["accuracy"][CONSOLIDATED]["exact"]   # theta_slow empty -> frozen-base
        acc_combo = c["accuracy"][PLASTIC]["exact"]
        corr = self.correction
        surv = self.paraphrase_survival
        purge = self.transcript_removal
        return {
            "1_acquisition": {
                "verdict": "PASS" if acc_plastic > chance_d else "FAIL",
                "evidence": f"direct exact accuracy PLASTIC={acc_plastic:.3f} vs "
                            f"chance={chance_d:.3f} (contains={acc_plastic_contains:.3f}); "
                            f"teacher={self.teacher_label}; foundation={self.foundation_label}"},
            "2_generalisation": {
                "verdict": "PASS" if acc_combo > baseline else "FAIL",
                "evidence": f"combination exact accuracy PLASTIC={acc_combo:.3f} vs baseline "
                            f"(CONSOLIDATED = pathway transmission off, theta_slow empty) "
                            f"={baseline:.3f}"},
            "3_retention": {
                "verdict": "PASS" if surv["survival_ok"] else "FAIL",
                "evidence": f"restart exact={surv['state_restored_exact']}; paraphrase exact "
                            f"pre={surv['accuracy_exact_pre']} post={surv['accuracy_exact_post']}; "
                            f"answer-identity={surv['answers_identical_fraction']}"},
            "4_correction": {
                "verdict": "PASS" if corr["adaptation_ok"] else "FAIL",
                "evidence": f"'{corr['correction_sentence']}' new_fact_learned="
                            f"{corr['new_fact_learned']} "
                            f"(strength {corr['new_fact_plastic_strength']}); old facts present "
                            f"{corr['old_facts_present_after']}; unaffected probe identical "
                            f"{corr['unaffected_probe_identical']}"},
            "5_transcript_removal": {
                "verdict": "PASS" if purge["plastic_unchanged"] else "FAIL",
                "evidence": f"purged {purge['purge']['removed']} transcript events -> plastic "
                            f"probe answers identical {purge['plastic_answers_identical']}"},
        }

    def report_phase(self) -> None:
        t0 = time.perf_counter()
        self.transcript_removal = self.transcript_removal_check()
        fp_ok, fp_diffs = fingerprints_equal(
            self.fingerprint_at_load, foundation_fingerprint(self.model))
        inventory = self.rmem.inventory()
        theta_norms = [round(float(c.plasticity.theta_slow.norm().item()), 8)
                       for c in self.iface.conns]
        acceptance = self.evaluate_acceptance()
        report = {
            "run": {
                "tag": self.run_tag, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "foundation": self.foundation_label,
                "teacher": self.teacher_label,
                "teacher_discovery": self.teacher_discovery,
                "seed": self.args.seed, "chains": self.world.n_chains,
                "relationships": len(self.world.relationships),
                "questions": len(self.world.questions),
                "gate_steps_per_fact": self.args.gate_steps,
                "answer_tokens": self.args.answer_tokens,
                "mode_note": "batteries run in MEASUREMENT mode (verdicts logged, plastic "
                             "writes suppressed); the run's only plastic correction is the "
                             "RETAIN-phase reroute",
                "frozen": getattr(self, "frozen_stats", None),
            },
            "timings_s": self.timings,
            "world_verify_failures": [],
            "teach": self.teach_summary,
            "theta_slow_norms": theta_norms,
            "theta_slow_empty": all(t == 0.0 for t in theta_norms),
            "test": {"batteries": self.test_summary, "rows": self.test_rows},
            "retain": {"restart": self.restart_report,
                       "paraphrase_survival": self.paraphrase_survival,
                       "correction": self.correction,
                       "pre_paraphrase": self.pre_paraphrase,
                       "post_paraphrase": self.post_paraphrase},
            "transcript_removal": self.transcript_removal,
            "foundation_integrity": {
                "bit_identical_after_full_run": bool(fp_ok),
                "differing_params": fp_diffs[:5], "n_params_checked": len(self.fingerprint_at_load)},
            "fact_inventory": inventory,
            "acceptance": acceptance,
        }
        n_pass = sum(1 for a in acceptance.values() if a["verdict"] == "PASS")
        report["acceptance_score"] = f"{n_pass}/5"
        self.report = report
        with open(os.path.join(self.run_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
        with open(os.path.join(self.run_dir, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=1, default=str)
        self.episodic.save(os.path.join(self.run_dir, "episodic_after_purge.json"))
        self.timings["report_s"] = round(time.perf_counter() - t0, 1)

        # ---- human-readable verdict ---- #
        print("\n" + "=" * 78)
        print(f"AION-1 PHASE 1 — TEACHER-STUDENT RUN  ({self.run_tag})")
        print(f"foundation: {self.foundation_label}")
        print(f"teacher   : {self.teacher_label}")
        if self.args.teacher == "scripted":
            print("endpoint  : not used (scripted teacher by request)")
        else:
            print(f"endpoint  : {self.teacher_discovery.get('ids') or 'UNREACHABLE'} "
                  f"-> alias {self.teacher_discovery.get('alias_used') or self.teacher_discovery.get('alias')}")
        print("=" * 78)
        for b in BATTERIES:
            s = self.test_summary[b]
            acc = s["accuracy"]
            print(f"\n{b.upper()} (n={s['n']}, chance={s['mean_chance']:.3f})")
            for mode in MODES:
                a = acc[mode]
                print(f"  {mode:12s} exact={a['exact']:.3f} lenient={a['lenient']:.3f} "
                      f"contains={a['contains']:.3f}")
            print(f"  provenance (exact): {s['provenance_histogram']}")
        print("\nRETAIN")
        print(f"  restart exact restore      : {self.restart_report['state_restored_exact']} "
              f"(probes regenerate identically: {self.restart_report['probes_regenerate_identically']})")
        print(f"  paraphrase survival (exact): pre={self.paraphrase_survival['accuracy_exact_pre'][PLASTIC]:.3f} "
              f"-> post={self.paraphrase_survival['accuracy_exact_post'][PLASTIC]:.3f} "
              f"(PLASTIC), answers identical "
              f"{self.paraphrase_survival['answers_identical_fraction']}")
        c = self.correction
        print(f"  ONE correction             : {c['correction_sentence']}")
        print(f"     new fact learned        : {c['new_fact_learned']} "
              f"(plastic strength {c['new_fact_plastic_strength']})")
        print(f"     old facts intact        : {c['old_facts_present_after']} present, "
              f"mean strength change {c['mean_relative_strength_change_old_facts']}, "
              f"unaffected probe identical {c['unaffected_probe_identical']}")
        print(f"TRANSCRIPT REMOVAL          : purge {self.transcript_removal['purge']['removed']} "
              f"events -> plastic identical {self.transcript_removal['plastic_answers_identical']}")
        print(f"FOUNDATION bit-identity     : {report['foundation_integrity']['bit_identical_after_full_run']} "
              f"({report['foundation_integrity']['n_params_checked']} params fingerprinted)")
        print("\n" + "-" * 78)
        print("ACCEPTANCE (owner table, honest):")
        for k, a in acceptance.items():
            print(f"  {a['verdict']:4s} {k}")
            print(f"       {a['evidence']}")
        print("-" * 78)
        print(f"score {report['acceptance_score']}  |  report: "
              f"{os.path.join(self.run_dir, 'report.json')}")
        print(f"timings: {self.timings}")

    # --------------------------------------------------------------------------- #

    def run(self) -> int:
        t_start = time.perf_counter()
        slug = "full" if self.args.full else f"c{self.args.chains}"
        self.run_tag = (f"{self.args.foundation}-{self.args.teacher}-{slug}"
                        f"-s{self.args.seed}-{time.strftime('%H%M%S')}")
        self.args.run_dir = os.path.join(_REPO_ROOT, "runs", "teacher_student", self.run_tag) \
            if not self.args.run_dir else self.args.run_dir
        self.student_cfg_public = None
        self.setup()
        self.student_cfg_public = self.student.cfg
        self.teach_phase()
        self.test_phase()
        self.retain_phase()
        self.report_phase()
        self.timings["total_s"] = round(time.perf_counter() - t_start, 1)
        n_pass = sum(1 for a in self.report["acceptance"].values()
                     if a["verdict"] == "PASS") if self.report else -1
        print(f"\nTOTAL RUNTIME: {self.timings['total_s']}s "
              f"({self.timings['total_s'] / 60:.1f} min); acceptance {n_pass}/5")
        return 0


# --------------------------------------------------------------------------- #
# PHASE 2 — the five-variant central experiment (owner spec, 2026-11-25)
#
#   A frozen base          original capability; nothing learned
#   B retrieval            episodic transcript lookup ONLY (the transcript-independent
#                          control C/D/E must beat on transfer + unseen combinations)
#   C plastic              declaratives into u_fast; gamma stays 0 (stored, not expressed)
#   D plastic + question   C + interrogative/expected pairs into u_fast; gamma stays 0
#   E plastic + question   D + gamma gate learning (full expression pathway); the owner's
#     + gamma              SEQUENCING LAW is preserved: ALL plasticity first, gate SECOND
#
# CONSTRAINED SCORING (the key instrumentation): instead of free decode, candidate
# entities are scored at the answer position, P(c_i|q) ∝ exp(score(q,c_i)) with
# score = sum of log-probs of the candidate's tokens given the scaffolded prompt.
# CRITICAL ABLATION: the same scoring runs WITH the plastic pathway active (FULL /
# PLASTIC) and WITHOUT it (CONSOLIDATED: theta_slow empty -> r=0 -> exactly the
# frozen base), so the plastic contribution is decidable per question.
# FREE-FORM GENERATION is kept a SEPARATE evaluation (communication, not
# discrimination). ICL scaffolding: one fixed two-exemplar QA scaffold, IDENTICAL
# text for every variant and every question (no mechanism is favoured).
# --------------------------------------------------------------------------- #

FORMAT_SCAFFOLD = (
    "Question: Which colour does the banner use?\nAnswer: gold.\n\n"
    "Question: Which metal does the coin use?\nAnswer: silver.\n\n"
)

# Training-only second interrogative per relation. DELIBERATELY DISJOINT from the
# world's test paraphrases (PARAPHRASE_TEMPLATES) so the paraphrase battery stays
# UNSEEN wording; the direct battery is the TRAINED format; combinations stay unseen.
TRAIN_Q_TEMPLATES = {
    "operates": "{subject} operates what?",
    "routes-to": "{subject} routes to which relay?",
    "controls": "{subject} controls which device?",
    "produces": "What is {subject}'s output?",
}

RELATION_SUBJECT_ROLE = {"operates": "actor", "routes-to": "conduit",
                         "controls": "relay", "produces": "device"}
RELATION_OBJECT_ROLE = {"operates": "conduit", "routes-to": "relay",
                        "controls": "device", "produces": "output"}

P2_VARIANTS = ("A", "B", "C", "D", "E")
P2_VARIANT_DESCRIPTIONS = {
    "A": "frozen base (original capability; nothing learned)",
    "B": "retrieval (episodic lookup only; transcript-independent control)",
    "C": "plastic (declaratives -> u_fast; gamma=0: stored, not expressed)",
    "D": "plastic + question-format learning (declaratives + interrogative/expected pairs; gamma=0)",
    "E": "plastic + question learning + gamma (full expression pathway; plasticity FIRST, gate SECOND)",
}
P2_VARIANT_MODES = {          # constrained-scoring provenance modes per variant
    "A": (FULL,),
    "B": (),
    "C": (FULL, PLASTIC, CONSOLIDATED),
    "D": (FULL, PLASTIC, CONSOLIDATED),
    "E": (FULL, PLASTIC, CONSOLIDATED),
}


def build_p2_experiences(world) -> List[Dict]:
    """Paired experiences per taught fact (owner Phase-2 spec): the declarative sentence
    plus TWO interrogative->expected pairs in the shared QA format."""
    exps: List[Dict] = []
    for r in world.relationships:
        key = relation_key(r.subject, r.relation, r.object)
        exps.append({"kind": "declarative", "text": r.sentence, "question": None,
                     "answer": r.object, "key": key, "chain": r.chain_id,
                     "fact": Fact(r.subject, r.relation, r.object)})
        for tmpl in (DIRECT_ASK[r.relation], TRAIN_Q_TEMPLATES[r.relation]):
            q = tmpl.format(subject=r.subject)
            exps.append({"kind": "qa", "text": f"Question: {q}\nAnswer: {r.object}.",
                         "question": q, "answer": r.object, "key": key,
                         "chain": r.chain_id, "fact": None})
    return exps


class VariantsRun:
    """Phase 2 harness: five variants, one loaded foundation, identical measurement."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tok = None
        self.world = None
        self.pools: Dict[str, List[str]] = {}
        self.experiences: List[Dict] = []
        self.timings: Dict[str, float] = {}
        self.fingerprint_at_load: Optional[Dict] = None
        self.foundation_label = ""
        self.run_tag = ""
        self.report: Dict = {}

    # -- plumbing ------------------------------------------------------------- #

    def log(self, msg: str) -> None:
        print(f"[phase2] {msg}", flush=True)

    @property
    def run_dir(self) -> str:
        return self.args.run_dir

    def stride(self, qs: List, cap: int) -> List:
        """Spread a battery across chains up to `cap` questions (0 = all)."""
        if cap <= 0 or len(qs) <= cap:
            return list(qs)
        step = len(qs) / float(cap)
        return [qs[int(i * step)] for i in range(cap)]

    def candidates_for(self, q) -> List[str]:
        m = q.meta
        if q.battery == BATTERY_DIRECT:
            return list(self.pools[RELATION_OBJECT_ROLE[m["relation"]]])
        if q.battery == BATTERY_PARAPHRASE:
            role = (RELATION_SUBJECT_ROLE[m["relation"]] if m.get("asked_side") == "subject"
                    else RELATION_OBJECT_ROLE[m["relation"]])
            return list(self.pools[role])
        if q.battery == BATTERY_COMBINATION:
            return ["yes", "no"] if m.get("kind") == "membership" else list(self.pools["output"])
        raise ValueError(f"battery {q.battery} not measured in Phase 2")

    # -- setup ------------------------------------------------------------------ #

    def setup(self) -> None:
        t0 = time.perf_counter()
        self.world = generate_world(seed=self.args.seed, n_chains=self.args.chains)
        failures = verify_world(self.world)
        if failures:
            raise RuntimeError(f"world ground truth inconsistent: {failures[:3]}")
        self.pools = {"actor": self.world.actors, "conduit": self.world.conduits,
                      "relay": self.world.relays, "device": self.world.devices,
                      "output": self.world.outputs}
        self.experiences = build_p2_experiences(self.world)
        os.makedirs(self.run_dir, exist_ok=True)
        self.world.save_json(os.path.join(self.run_dir, "world.json"))

        t1 = time.perf_counter()
        if self.args.foundation == "qwen":
            # PUBLIC loader path (foundation lane owns this file; Phase 2 runs only
            # after the loader fix lands — no mirrored fallback in this phase).
            handle = load_frozen_base(local_dir=self.args.local_dir, device=self.args.device)
            self.model = handle.model
            self.tok = handle.tokenizer
            self.fingerprint_at_load = handle.fingerprint
            self.frozen_stats = {"frozen_tensors": handle.n_frozen_tensors,
                                 "frozen_params": handle.n_frozen_params}
            self.foundation_label = f"Qwen3.5-4B-Base @ {self.args.local_dir} (CPU BF16, FROZEN)"
        else:
            self.model, self.tok = build_tiny_foundation(seed=self.args.seed)
            self.fingerprint_at_load = foundation_fingerprint(self.model)
            self.foundation_label = "tiny-standin(Llama-4L-64D, CPU, FROZEN)"
        self.timings["foundation_load_s"] = round(time.perf_counter() - t1, 1)
        self.timings["setup_s"] = round(time.perf_counter() - t0, 1)

        self.test_questions = (self.stride([q for q in self.world.questions
                                            if q.battery == BATTERY_DIRECT], self.args.p2_direct_cap)
                               + self.stride([q for q in self.world.questions
                                              if q.battery == BATTERY_PARAPHRASE], self.args.p2_para_cap)
                               + self.stride([q for q in self.world.questions
                                              if q.battery == BATTERY_COMBINATION], self.args.p2_combo_cap))
        self.retention_questions = [q for q in self.test_questions
                                    if q.battery == BATTERY_PARAPHRASE]
        self.freeform_questions = (self.stride([q for q in self.world.questions
                                                if q.battery == BATTERY_DIRECT], 6)
                                   + self.stride([q for q in self.world.questions
                                                  if q.battery == BATTERY_PARAPHRASE], 4)
                                   + self.stride([q for q in self.world.questions
                                                  if q.battery == BATTERY_COMBINATION], 4))
        self.log(f"setup: {self.foundation_label} (load {self.timings['foundation_load_s']}s); "
                 f"world chains={self.world.n_chains} relationships={len(self.world.relationships)} "
                 f"experiences={len(self.experiences)}; test={len(self.test_questions)} questions "
                 f"({len(self.retention_questions)} paraphrase for retention), "
                 f"free-form={len(self.freeform_questions)}")

    # -- lanes -------------------------------------------------------------------- #

    def fresh_lane(self, variant: str) -> Dict:
        """A fresh AION-side lane around the SAME loaded foundation: new interface
        (zero-init gates, zero u_fast), new episodic store, new fact index."""
        iface = RelationalInterface(self.model, cfg=InterfaceConfig(
            learning_rate=self.args.learning_rate, gate_lr=self.args.p2_gate_lr))
        student = Student(iface, self.tok,
                          StudentConfig(answer_tokens=self.args.answer_tokens, gate_steps=0))
        episodic = EpisodicStore(clock=lambda: 0.0)
        rmem = RelationalMemory(conns=list(iface.conns), hidden_dim=iface.hidden_dim,
                                episodic=episodic, blend_ratio=iface.cfg.blend_ratio)
        return {"variant": variant, "iface": iface, "student": student,
                "episodic": episodic, "rmem": rmem}

    def teardown_lane(self, lane: Dict) -> None:
        for h in getattr(lane.get("iface"), "_hooks", []) or []:
            h.remove()
        lane.clear()
        gc.collect()

    def teach_variant(self, lane: Dict, variant: str) -> Dict:
        t0 = time.perf_counter()
        episodic, iface, student = lane["episodic"], lane["iface"], lane["student"]
        u_before = [round(c.plasticity.u_fast.norm().item(), 6) for c in iface.conns]
        n_nnfs = gate_steps = 0
        uid = 0
        if variant == "A":
            pass                                          # frozen: nothing is learned
        elif variant == "B":
            for e in self.experiences:                    # transcript lookup only
                episodic.record(kind="lesson" if e["kind"] == "declarative" else "observation",
                                text=e["question"] or e["text"], channel="transcript",
                                teacher="retrieval-control",
                                tags={"answer": e["answer"], "fact_key": e["key"],
                                      "kind": e["kind"]})
        else:
            for e in self.experiences:
                uid += 1
                if e["kind"] == "declarative":
                    if variant in ("C", "D", "E"):        # plasticity FIRST
                        student.learn_fact(e["fact"])
                        n_nnfs += 1
                elif variant in ("D", "E"):               # question-format learning
                    ids = torch.tensor([self.tok.encode(e["text"])])
                    iface.nnfs_update(ids)
                    n_nnfs += 1
                episodic.record(kind="lesson" if e["kind"] == "declarative" else "observation",
                                text=e["question"] or e["text"], channel="transcript",
                                teacher=f"variant-{variant}",
                                plastic_update_ids=[uid],
                                tags={"answer": e["answer"], "fact_key": e["key"],
                                      "kind": e["kind"]})
            for r in self.world.relationships:            # storage-evidence index
                key = relation_key(r.subject, r.relation, r.object)
                pre, post = self.probe_sentence(lane, r.sentence)
                lane["rmem"].register(key, 0, pre, post,
                                      meta={"sentence": r.sentence, "chain": r.chain_id})
            gamma_before_gate = [round(float(g.detach()), 6) for g in iface.gates]
            if variant == "E":
                # SEQUENCING LAW: ALL plasticity has already run; gate learning is SECOND
                for e in self.experiences:
                    if e["kind"] != "qa":
                        continue
                    ids = torch.tensor([self.tok.encode(e["text"])])
                    for _ in range(self.args.p2_gate_steps):
                        iface.gate_step(ids)
                        gate_steps += 1
        u_after = [round(c.plasticity.u_fast.norm().item(), 6) for c in iface.conns]
        gamma_after = [round(float(g.detach()), 6) for g in iface.gates]
        strengths = ([round(lane["rmem"].strength(k, blend=1.0) or 0.0, 6)
                      for k in sorted(lane["rmem"].records)] if variant in ("C", "D", "E") else [])
        out = {"variant": variant, "experiences": len(self.experiences),
               "nnfs_updates": n_nnfs, "gate_steps": gate_steps,
               "gate_lr": self.args.p2_gate_lr if variant == "E" else None,
               "u_fast_norms_before": u_before, "u_fast_norms_after": u_after,
               "gamma_before_gate_training": gamma_before_gate if variant == "E" else gamma_after,
               "gamma_after": gamma_after,
               "fact_plastic_strengths": strengths,
               "mean_fact_plastic_strength": round(sum(strengths) / len(strengths), 6) if strengths else None,
               "episodic_events": len(episodic),
               "teach_s": round(time.perf_counter() - t0, 1)}
        self.log(f"teach {variant}: nnfs={n_nnfs} gate_steps={gate_steps} "
                 f"u_fast={u_after} gamma={gamma_after} "
                 f"({out['teach_s']}s)")
        return out

    def probe_sentence(self, lane: Dict, sentence: str):
        ids = torch.tensor([self.tok.encode(sentence)], dtype=torch.long)
        with torch.no_grad():
            res = lane["iface"].forward(ids)
        pre_i, post_i = lane["iface"].pairs[0]
        return (res["hidden"][pre_i].mean(dim=1)[0].float(),
                res["hidden"][post_i].mean(dim=1)[0].float())

    # -- CONSTRAINED SCORING --------------------------------------------------- #

    def constrained_scores(self, lane: Dict, question: str, candidates: List[str],
                           mode: str) -> Dict[str, float]:
        """P(c_i|q) ∝ exp(score): candidate log-prob at the answer position under the
        scaffolded prompt, with the provenance-bucket ablation active for the mode."""
        prompt = FORMAT_SCAFFOLD + f"Question: {question}\nAnswer:"
        ids_prompt = self.tok.encode(prompt)
        iface = lane["iface"]
        scores: Dict[str, float] = {}
        for cand in candidates:
            ids_c = self.tok.encode(" " + cand)
            full = torch.tensor([ids_prompt + ids_c], dtype=torch.long)
            with ablated(iface.conns, iface.gates, mode):
                with torch.no_grad():
                    res = iface.forward(full)
            lp = torch.log_softmax(res["logits"][0].float(), dim=-1)
            base = len(ids_prompt) - 1                     # position predicting cand token 0
            s = 0.0
            for j, t in enumerate(ids_c):
                s += float(lp[base + j, t])
            scores[cand] = s
        return scores

    def retrieved_answer(self, lane: Dict, question: str) -> Tuple[str, float]:
        hits = lane["episodic"].retrieve(question, k=1)
        if not hits or hits[0][1] <= 0.0:
            return "", 0.0
        ev, score = hits[0]
        return ev.tags.get("answer", ""), float(score)

    def score_row(self, lane: Dict, q, modes: Tuple[str, ...]) -> Dict:
        cands = self.candidates_for(q)
        rows_modes: Dict[str, Dict] = {}
        for mode in modes:
            scores = self.constrained_scores(lane, q.text, cands, mode)
            mx = max(scores.values())
            probs = {c: (pow(2.718281828, s - mx) / sum(pow(2.718281828, v - mx)
                                                        for v in scores.values()))
                     for c, s in scores.items()}
            ranked = sorted(cands, key=lambda c: -scores[c])
            rows_modes[mode] = {"argmax": ranked[0], "correct": ranked[0] == q.answer,
                                "gold_rank": ranked.index(q.answer) + 1,
                                "gold_prob": round(probs[q.answer], 6),
                                "gold_score": round(scores[q.answer], 6)}
        retrieved, r_score = self.retrieved_answer(lane, q.text)
        retrieved_correct = bool(default_grader(retrieved, q.answer))
        correct_buckets = [m for m in STRONGEST_FIRST
                           if (m in rows_modes and rows_modes[m]["correct"])
                           or (m == RETRIEVED and retrieved_correct)]
        if correct_buckets:
            provenance = correct_buckets[0]
        elif rows_modes.get(FULL, {}).get("correct"):
            provenance = "FULL_ONLY"
        else:
            provenance = "NONE"
        return {"qid": q.id, "battery": q.battery, "question": q.text, "gold": q.answer,
                "candidates": cands, "chance": round(1.0 / len(cands), 4),
                "modes": rows_modes, "retrieved_answer": retrieved,
                "retrieval_score": round(r_score, 4),
                "retrieved_correct": retrieved_correct,
                "provenance": provenance}

    def battery_summary(self, rows: List[Dict], modes: Tuple[str, ...]) -> Dict:
        n = max(1, len(rows))
        out = {"n": len(rows),
               "mean_chance": round(sum(r["chance"] for r in rows) / n, 4),
               "retrieval_acc": round(sum(r["retrieved_correct"] for r in rows) / n, 4),
               "modes": {}}
        for mode in modes:
            out["modes"][mode] = {
                "acc@1": round(sum(r["modes"][mode]["correct"] for r in rows) / n, 4),
                "mean_gold_rank": round(sum(r["modes"][mode]["gold_rank"] for r in rows) / n, 3),
                "mean_gold_prob": round(sum(r["modes"][mode]["gold_prob"] for r in rows) / n, 5)}
        hist: Dict[str, int] = {}
        for r in rows:
            hist[r["provenance"]] = hist.get(r["provenance"], 0) + 1
        out["provenance_histogram"] = hist
        return out

    def run_battery(self, lane: Dict, variant: str, questions: List, modes, tag: str):
        rows = []
        t0 = time.perf_counter()
        for i, q in enumerate(questions):
            rows.append(self.score_row(lane, q, modes))
            if (i + 1) % 10 == 0 or i + 1 == len(questions):
                self.log(f"  [{tag}] {i + 1}/{len(questions)} scored")
        self.timings[f"battery_{tag}_s"] = round(time.perf_counter() - t0, 1)
        by_battery = {}
        for b in (BATTERY_DIRECT, BATTERY_PARAPHRASE, BATTERY_COMBINATION):
            bros = [r for r in rows if r["battery"] == b]
            if bros:
                by_battery[b] = self.battery_summary(bros, modes)
        return rows, by_battery

    # -- FREE-FORM GENERATION (separate evaluation) ------------------------------ #

    def freeform_battery(self, lane: Dict, variant: str, questions: List) -> Tuple[List[Dict], Dict]:
        rows = []
        for q in questions:
            prompt = FORMAT_SCAFFOLD + f"Question: {q.text}\nAnswer:"
            text = lane["student"].ask(prompt).text
            rows.append({"qid": q.id, "battery": q.battery, "question": q.text,
                         "gold": q.answer, "generation": text,
                         "exact": bool(default_grader(text, q.answer)),
                         "lenient": bool(lenient_match(text, q.answer)),
                         "contains": bool(contains_match(text, q.answer)),
                         "empty": not bool((text or "").strip())})
        n = max(1, len(rows))
        summary = {"n": n,
                   "exact": round(sum(r["exact"] for r in rows) / n, 4),
                   "lenient": round(sum(r["lenient"] for r in rows) / n, 4),
                   "contains": round(sum(r["contains"] for r in rows) / n, 4),
                   "empty_fraction": round(sum(r["empty"] for r in rows) / n, 4)}
        return rows, summary

    # -- RETENTION (restart per variant) ------------------------------------------- #

    def restart_lane(self, lane: Dict, variant: str) -> Dict:
        """AION-side state round-trips through disk into brand-new objects; the frozen
        foundation reload was proven bit-exact in Phase 1, so the restart here rebuilds
        the interface around the same loaded weights (documented, not hidden)."""
        state_path = os.path.join(self.run_dir, f"state_{variant}.pt")
        episodic_path = os.path.join(self.run_dir, f"episodic_{variant}.json")
        snap = PlasticStateSnapshot.capture(lane["iface"], meta={"phase": "p2", "variant": variant})
        payload = {"plastic": snap.to_payload(), "probes": lane["rmem"].save_probes(),
                   "index": lane["rmem"].to_dict()}
        torch.save(payload, state_path)
        lane["episodic"].save(episodic_path)
        self.teardown_lane(lane)                            # real teardown of live objects
        payload2 = torch.load(state_path, weights_only=False)
        snap2 = PlasticStateSnapshot.from_payload(payload2["plastic"])
        lane["variant"] = variant
        lane["iface"] = snap2.build_interface(self.model)   # fresh interface + restore
        lane["student"] = Student(lane["iface"], self.tok,
                                  StudentConfig(answer_tokens=self.args.answer_tokens, gate_steps=0))
        lane["episodic"] = EpisodicStore.load(episodic_path, clock=lambda: 0.0)
        lane["rmem"] = RelationalMemory.from_dict(payload2["index"],
                                                  conns=list(lane["iface"].conns),
                                                  episodic=lane["episodic"],
                                                  probes=payload2["probes"])
        recheck = PlasticStateSnapshot.capture(lane["iface"])
        exact, diffs = snap2.equals(recheck, atol=0.0)
        os.remove(state_path)                               # transient: retention needs only the round-trip
        return {"state_restored_exact": bool(exact), "restore_diffs": diffs[:5],
                "episodic_events_reloaded": len(lane["episodic"]),
                "fact_index_reloaded": len(lane["rmem"].records)}

    # -- per-variant measurement ----------------------------------------------------- #

    def run_variant(self, variant: str) -> Dict:
        t0 = time.perf_counter()
        modes = P2_VARIANT_MODES[variant]
        lane = self.fresh_lane(variant)
        teach = self.teach_variant(lane, variant)
        test_rows, test_by_battery = self.run_battery(lane, variant, self.test_questions,
                                                      modes, f"{variant}-test")
        ff_rows, ff_summary = self.freeform_battery(lane, variant, self.freeform_questions)
        restart = self.restart_lane(lane, variant)
        post_rows, post_by_battery = self.run_battery(lane, variant, self.retention_questions,
                                                      modes, f"{variant}-post-restart")
        # retention: per-question argmax/retrieval identity pre vs post restart
        pre_para = {r["qid"]: r for r in test_rows if r["battery"] == BATTERY_PARAPHRASE}
        identity = {}
        for mode in modes:
            same = sum(1 for r in post_rows
                       if pre_para[r["qid"]]["modes"][mode]["argmax"] == r["modes"][mode]["argmax"])
            identity[mode] = round(same / max(1, len(post_rows)), 4)
        same_retr = sum(1 for r in post_rows
                        if pre_para[r["qid"]]["retrieved_answer"] == r["retrieved_answer"])
        identity[RETRIEVED] = round(same_retr / max(1, len(post_rows)), 4)
        strengths = teach.pop("fact_plastic_strengths", [])
        out = {"variant": variant, "description": P2_VARIANT_DESCRIPTIONS[variant],
               "teach": teach,
               "fact_plastic_strengths_min_mean": (round(min(strengths), 6),
                                                   round(sum(strengths) / len(strengths), 6))
                                                  if strengths else None,
               "test": {"rows": test_rows, "batteries": test_by_battery},
               "freeform": {"rows": ff_rows, "summary": ff_summary},
               "restart": restart,
               "retention": {"paraphrase_acc_pre": {b: test_by_battery[b] for b in
                                                    (BATTERY_PARAPHRASE,) if b in test_by_battery},
                             "paraphrase_acc_post": post_by_battery.get(BATTERY_PARAPHRASE, {}),
                             "answers_identical_fraction": identity,
                             "survival_ok": bool(restart["state_restored_exact"]
                                                 and all(v == 1.0 for v in identity.values()))},
               "variant_s": round(time.perf_counter() - t0, 1)}
        self.teardown_lane(lane)

        def _fmt(b):
            s = test_by_battery[b]
            full = s.get("modes", {}).get(FULL, {}).get("acc@1")
            head = f"{b}:" + (f"FULL@1={full:.2f}" if full is not None else "no-constrained")
            return f"{head}/retr={s['retrieval_acc']:.2f}"
        self.log(f"VARIANT {variant} done ({out['variant_s']}s): "
                 + " ".join(_fmt(b) for b in test_by_battery)
                 + f" | free-form exact={ff_summary['exact']:.2f} "
                 f"| restart_exact={restart['state_restored_exact']} "
                 f"| retention_ok={out['retention']['survival_ok']}")
        return out

    # -- central comparison + report ------------------------------------------------- #

    def headline(self, reports: Dict[str, Dict]) -> Dict:
        b = reports.get("B")
        if b is None:
            return {"note": "variant B missing"}
        b_para = b["test"]["batteries"].get(BATTERY_PARAPHRASE, {}).get("retrieval_acc", 0.0)
        b_combo = b["test"]["batteries"].get(BATTERY_COMBINATION, {}).get("retrieval_acc", 0.0)
        rows = {}
        for v in ("C", "D", "E"):
            r = reports.get(v)
            if r is None:
                continue
            bats = r["test"]["batteries"]
            p = bats.get(BATTERY_PARAPHRASE, {})
            c = bats.get(BATTERY_COMBINATION, {})
            with_p = p.get("modes", {}).get(FULL, {}).get("acc@1", 0.0)
            without_p = p.get("modes", {}).get(CONSOLIDATED, {}).get("acc@1", 0.0)
            with_c = c.get("modes", {}).get(FULL, {}).get("acc@1", 0.0)
            without_c = c.get("modes", {}).get(CONSOLIDATED, {}).get("acc@1", 0.0)
            rows[v] = {
                "paraphrase_with_plastic": with_p, "paraphrase_without_plastic": without_p,
                "combination_with_plastic": with_c, "combination_without_plastic": without_c,
                "beats_B_paraphrase": bool(with_p > b_para),
                "beats_B_combination": bool(with_c > b_combo),
                "delta_vs_B_paraphrase": round(with_p - b_para, 4),
                "delta_vs_B_combination": round(with_c - b_combo, 4),
                "plastic_ablation_delta_paraphrase": round(with_p - without_p, 4),
                "plastic_ablation_delta_combination": round(with_c - without_c, 4),
            }
        beats = [v for v, r in rows.items()
                 if (r["beats_B_paraphrase"] or r["beats_B_combination"])]
        return {"B_retrieval_paraphrase": b_para, "B_retrieval_combination": b_combo,
                "variants": rows,
                "variants_beating_B_on_transfer_or_combination": beats,
                "headline": (f"C/D/E beat B on knowledge transfer or unseen combinations: "
                             f"{', '.join(beats) if beats else 'NO — retrieval control holds'}")}

    def report_phase(self, reports: Dict[str, Dict]) -> None:
        fp_ok, fp_diffs = fingerprints_equal(
            self.fingerprint_at_load, foundation_fingerprint(self.model))
        self.report = {
            "phase": 2, "tag": self.run_tag, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "foundation": self.foundation_label,
            "teacher": "n/a (Phase 2 teaching is experience-driven; no chat endpoint used)",
            "seed": self.args.seed, "chains": self.world.n_chains,
            "experiences_per_run": len(self.experiences),
            "scaffold": FORMAT_SCAFFOLD,
            "train_q_templates": TRAIN_Q_TEMPLATES,
            "measure_sets": {"test": len(self.test_questions),
                             "retention_paraphrase": len(self.retention_questions),
                             "freeform": len(self.freeform_questions)},
            "gamma_sequencing_law": "plasticity first, gate learning second (E only)",
            "timings_s": self.timings,
            "variants": reports,
            "headline_comparison": self.headline(reports),
            "foundation_integrity": {"bit_identical_after_full_run": bool(fp_ok),
                                     "differing_params": fp_diffs[:5],
                                     "n_params_checked": len(self.fingerprint_at_load)},
        }
        with open(os.path.join(self.run_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=1, default=str)

        h = self.report["headline_comparison"]
        print("\n" + "=" * 88)
        print(f"AION-1 PHASE 2 — FIVE-VARIANT CENTRAL EXPERIMENT ({self.run_tag})")
        print(f"foundation: {self.foundation_label}")
        print("=" * 88)
        print(f"{'variant':8s} {'teach':28s} {'direct@1':>9s} {'para@1':>7s} {'combo@1':>8s} "
              f"{'retr@1':>7s} {'freeform':>9s} {'restart':>8s} {'retain':>7s}")
        for v in P2_VARIANTS:
            r = reports[v]
            bats = r["test"]["batteries"]

            def acc(b, mode=FULL):
                a = bats.get(b, {}).get("modes", {}).get(mode, {}).get("acc@1")
                return "n/a" if a is None else f"{a:.2f}"
            ff = r["freeform"]["summary"]

            def cell(b, mode=FULL, width=9):
                a = bats.get(b, {}).get("modes", {}).get(mode, {}).get("acc@1")
                return ("n/a".rjust(width)) if a is None else f"{a:.2f}".rjust(width)
            print(f"{v:8s} {r['description'][:28]:28s} "
                  f"{cell(BATTERY_DIRECT, width=9)} {cell(BATTERY_PARAPHRASE, width=7)} "
                  f"{cell(BATTERY_COMBINATION, width=8)} "
                  f"{bats.get(BATTERY_DIRECT, {}).get('retrieval_acc', 0.0):7.2f} "
                  f"{ff['exact']:9.2f} "
                  f"{str(r['restart']['state_restored_exact']):>8s} "
                  f"{str(r['retention']['survival_ok']):>7s}")
        print("-" * 88)
        print("WITH vs WITHOUT plastic (constrained scoring, the critical ablation):")
        for v in ("C", "D", "E"):
            for b in (BATTERY_PARAPHRASE, BATTERY_COMBINATION):
                bats = reports[v]["test"]["batteries"].get(b, {})
                w = bats.get("modes", {}).get(FULL, {}).get("acc@1")
                wo = bats.get("modes", {}).get(CONSOLIDATED, {}).get("acc@1")
                if w is not None:
                    print(f"  {v} {b:13s} with={w:.3f} without={wo:.3f} "
                          f"delta={w - wo:+.3f}")
        print("-" * 88)
        print("HEADLINE (central outcome):", h["headline"])
        for v, row in h["variants"].items():
            print(f"  {v}: para {row['paraphrase_with_plastic']:.3f} vs B {h['B_retrieval_paraphrase']:.3f} "
                  f"(delta {row['delta_vs_B_paraphrase']:+.3f}); "
                  f"combo {row['combination_with_plastic']:.3f} vs B {h['B_retrieval_combination']:.3f} "
                  f"(delta {row['delta_vs_B_combination']:+.3f})")
        print(f"foundation bit-identical after all five variants: "
              f"{self.report['foundation_integrity']['bit_identical_after_full_run']}")
        print(f"report: {os.path.join(self.run_dir, 'report.json')}")

    # --------------------------------------------------------------------------- #

    def run(self) -> int:
        t_start = time.perf_counter()
        slug = "full" if self.args.full else f"c{self.args.chains}"
        self.run_tag = (f"p2-{self.args.foundation}-{slug}-s{self.args.seed}-"
                        f"{time.strftime('%H%M%S')}")
        if not self.args.run_dir:
            self.args.run_dir = os.path.join(_REPO_ROOT, "runs", "teacher_student_p2", self.run_tag)
        variants = [v.strip().upper() for v in self.args.p2_variants.split(",") if v.strip()]
        for v in variants:
            if v not in P2_VARIANTS:
                raise ValueError(f"unknown variant {v!r}; choose from {P2_VARIANTS}")
        self.setup()
        reports: Dict[str, Dict] = {}
        for v in variants:
            reports[v] = self.run_variant(v)
        self.report_phase(reports)
        self.timings["total_s"] = round(time.perf_counter() - t_start, 1)
        print(f"TOTAL RUNTIME: {self.timings['total_s']}s "
              f"({self.timings['total_s'] / 60:.1f} min)")
        return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="AION-1 teacher-student runs (Phase 1 / Phase 2)")
    ap.add_argument("--phase2", action="store_true",
                    help="run the PHASE 2 five-variant central experiment (A/B/C/D/E) instead "
                         "of the Phase 1 single-run harness")
    ap.add_argument("--foundation", default="tiny", choices=("tiny", "qwen"),
                    help="tiny = in-memory stand-in (Rog CPU debug); qwen = real frozen "
                         "Qwen3.5-4B-Base (Lappy)")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR,
                    help="checkpoint dir for --foundation qwen")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chains", type=int, default=5,
                    help="chains taught (5 chains = 20 relationships: the owner's original "
                         "criterion)")
    ap.add_argument("--full", action="store_true", help="scale to all 25 chains / 100 relationships")
    ap.add_argument("--teacher", default="scripted", choices=("scripted", "auto", "endpoint"),
                    help="PHASE 1 only: scripted = offline stub; auto = discover endpoint, "
                         "fall back to scripted if unreachable; endpoint = require the endpoint")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--teacher-alias", default=None,
                    help="override the discovered alias (client NEVER reads llama-swap config)")
    ap.add_argument("--teacher-max-tokens", type=int, default=256)
    ap.add_argument("--gate-steps", type=int, default=2,
                    help="PHASE 1: residual-gate autograd steps per fact")
    ap.add_argument("--gate-lr", type=float, default=1e-2,
                    help="PHASE 1: SGD step size for the residual gate gamma")
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--answer-tokens", type=int, default=8)
    ap.add_argument("--correction-repeats", type=int, default=3)
    ap.add_argument("--probe-size", type=int, default=10)
    ap.add_argument("--max-per-battery", type=int, default=0,
                    help="PHASE 1: trim each battery to its first N questions (0 = all)")
    # ---- Phase 2 knobs ---- #
    ap.add_argument("--p2-variants", default="A,B,C,D,E",
                    help="comma list of Phase 2 variants to run (default all five)")
    ap.add_argument("--p2-direct-cap", type=int, default=20)
    ap.add_argument("--p2-para-cap", type=int, default=8)
    ap.add_argument("--p2-combo-cap", type=int, default=6)
    ap.add_argument("--p2-gate-lr", type=float, default=100.0,
                    help="PHASE 2 variant E only: gate gamma SGD step size (modest by law: "
                         "plasticity first, gate second — never maximised indiscriminately)")
    ap.add_argument("--p2-gate-steps", type=int, default=2,
                    help="PHASE 2 variant E only: gate steps per QA experience")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    if args.phase2:
        return VariantsRun(args).run()
    if args.teacher == "endpoint" and args.foundation == "tiny":
        print("note: --teacher endpoint with the tiny foundation still exercises the real "
              "endpoint for lesson delivery (debug lane)")
    return TeacherStudentRun(args).run()


if __name__ == "__main__":
    sys.exit(main())
