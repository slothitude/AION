"""THE STRICT RETENTION TEST for AION-1 (owner spec, verbatim in structure):

    save learned state -> shut down (serialize EVERYTHING AION-side: plastic
    state, memory stores; the foundation is frozen weights on disk and simply
    reloads) -> restart (fresh process) -> re-ask the questions with DIFFERENT
    WORDING (paraphrase battery) -> test NEW COMBINATIONS -> CHANGE ONE
    RELATIONSHIP (feed a correction) and observe adaptation (old relationships
    intact + new one learned — the learn-new-without-destroying-old criterion).

Every answer is emitted with PROVENANCE: which of the three buckets it came
from —

    PLASTIC       answered with a = u_fast alone          (blend forced to 1.0)
    CONSOLIDATED  answered with a = theta_slow alone      (blend forced to 0.0)
    RETRIEVED     pathway fully ablated (gates zeroed); answer located from the
                  episodic store only

— by ABLATING one bucket at a time and asking the same question under each
configuration. The buckets are mutually exclusive (in PLASTIC/CONSOLIDATED the
episodic store is never consulted; in RETRIEVED the pathway contributes
nothing), so each answer's source is decidable. The natural blend is also
evaluated (mode FULL) and correctness there without any single bucket is
labelled FULL_ONLY.

VARIANTS (owner A/B/C/D harness — the decisive comparison, B must lose to C/D
on internalisation tests, is runnable with no new code):
    A frozen                 no further learning after foundation load
    B conversation-retrieval episodic log only, no plastic learning
    C persistent-plastic     NNFS learning + consolidation (+ provenance log)
    D plastic+memory+dream   C + episodic memory + DREAM HOOK STUB ONLY (the
                             dream engine is LOCKED until post-pilot; the hook
                             exists so D wiring needs no new code later)

CLI (the learning-loop lane runs exactly this once AION-1 is declared):
    # single process (probe still rebuilds ALL state from disk; convenient):
    python aion/experiments/retention.py --variant D --phase all \
        --run-dir runs/retention_D --seed 0
    # STRICT fresh-process mode (each phase is its own process; shutdown is real):
    python aion/experiments/retention.py --variant D --phase learn --run-dir runs/retention_D
    python aion/experiments/retention.py --variant D --phase probe --run-dir runs/retention_D
Add --battery aion/memory/fixtures/tiny_battery.json to select a battery; the
worlds lane will deliver the real ones in the same JSON shape (see
aion/experiments/knowledge_discovery.py docstring).

------------------------------------------------------------------------------
EXPECTED-INTERFACE CONTRACT — aion/dialogue (CONCURRENT LANE, NOT IMPORTED HERE)
------------------------------------------------------------------------------
This lane is being built in parallel. retention.py does NOT import it. When it
lands, plug it in by satisfying these two duck-typed protocols; nothing else in
this harness needs to change:

    class AIONAnswerer:            # wraps the dialogue lane's answering path
        def answer(self, prompt_text: str, mode: str) -> dict:
            \"\"\"mode in {'FULL','PLASTIC','CONSOLIDATED','RETRIEVED'}.
            Returns {'text': str, 'token_ids': list[int] | None, 'source': str}.
            The harness decides provenance by calling answer() once per mode.\"\"\"

    class AIONLearner:             # wraps the dialogue lane's teaching path
        def learn(self, exchange_text: str, teacher: str,
                  transcript_id: str | None) -> dict:
            \"\"\"One teaching exchange in. Returns
            {'plastic_update_ids': list[int]} — the NNFS update IDs this
            exchange produced, recorded verbatim into EpisodicStore provenance.\"\"\"

Transcripts are consumed only through EpisodicStore.record(...) — the dialogue
lane hands over (text, teacher, transcript_id); this apparatus owns the rest.
Until it lands, ModelPathwayAnswerer + BatteryLearner (below) stand in on CPU.

------------------------------------------------------------------------------
EXPECTED-INTERFACE CONTRACT — aion/worlds (CONCURRENT LANE, NOT IMPORTED HERE)
------------------------------------------------------------------------------
The question battery JSON shape this harness consumes is documented in
aion/experiments/knowledge_discovery.py (same shape for both harnesses). The
tiny local fixture is aion/memory/fixtures/tiny_battery.json.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.memory.state import PlasticStateSnapshot                      # noqa: E402
from aion.memory.episodic import EpisodicStore                          # noqa: E402
from aion.memory.relational import RelationalMemory, relation_key       # noqa: E402
from aion.memory.consolidation import ConsolidationLedger               # noqa: E402
from aion.memory.cpu_stubs import (                                     # noqa: E402
    build_stub_base, StubVocab, normalize_answer, answer_matches,
)
from aion.relational.interface import RelationalInterface, InterfaceConfig  # noqa: E402

DEFAULT_BATTERY = os.path.join(_REPO_ROOT, "aion", "memory", "fixtures", "tiny_battery.json")

PLASTIC = "PLASTIC"
CONSOLIDATED = "CONSOLIDATED"
RETRIEVED = "RETRIEVED"
FULL = "FULL"
MODES = (FULL, PLASTIC, CONSOLIDATED, RETRIEVED)
STRONGEST_FIRST = (PLASTIC, CONSOLIDATED, RETRIEVED)   # provenance label priority

INTERNALISATION_CAPABILITIES = {"combine", "learn_new", "self_correct"}


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

class Variant:
    A = "A"   # frozen
    B = "B"   # conversation-retrieval
    C = "C"   # persistent plastic
    D = "D"   # plastic + memory + dreaming (dream hook stub)
    ALL = (A, B, C, D)


VARIANT_DESCRIPTIONS = {
    "A": "frozen (no further learning)",
    "B": "conversation-retrieval (episodic store only)",
    "C": "persistent plastic learning (+ consolidation)",
    "D": "plastic + memory + dreaming (dream engine LOCKED: stub hook only)",
}


# --------------------------------------------------------------------------- #
# Bucket ablation (runtime-only knobs on the live objects; values exactly
# restored. The frozen code is never edited — blend_ratio and the residual gate
# are the interface's own evaluation knobs.)
# --------------------------------------------------------------------------- #

@contextmanager
def ablated(conns, gates, mode: str):
    saved_blend = [c.blend_ratio for c in conns]
    saved_gates = [g.detach().clone() for g in gates]
    try:
        if mode == PLASTIC:
            for c in conns:
                c.blend_ratio = 1.0            # a = u_fast alone
        elif mode == CONSOLIDATED:
            for c in conns:
                c.blend_ratio = 0.0            # a = theta_slow alone
        elif mode == RETRIEVED:
            with torch.no_grad():
                for g in gates:
                    g.zero_()                  # pathway contributes nothing
        elif mode != FULL:
            raise ValueError(f"unknown mode {mode!r}")
        yield
    finally:
        for c, b in zip(conns, saved_blend):
            c.blend_ratio = b
        with torch.no_grad():
            for g, s in zip(gates, saved_gates):
                g.copy_(s)


# --------------------------------------------------------------------------- #
# Answerers (the dialogue lane will replace these — see contract above)
# --------------------------------------------------------------------------- #

class ModelPathwayAnswerer:
    """CPU stand-in: answers through the interface under an ablation mode.

    FULL / PLASTIC / CONSOLIDATED read the pathway (interface forward, argmax at
    the last position). RETRIEVED never touches the model: it locates the
    answer in the episodic store only.
    """

    def __init__(self, iface: RelationalInterface, episodic: EpisodicStore, vocab: StubVocab):
        self.iface = iface
        self.episodic = episodic
        self.vocab = vocab

    def answer(self, prompt_text: str, prompt_ids: torch.Tensor, mode: str) -> Dict:
        if mode == RETRIEVED:
            hits = self.episodic.retrieve(prompt_text, k=1)
            if not hits or hits[0][1] <= 0.0:
                return {"text": "", "token_ids": None, "source": "episodic", "hit": False}
            ev, score = hits[0]
            text = ev.tags.get("answer") or ev.text
            return {"text": text, "token_ids": None, "source": "episodic",
                    "hit": True, "retrieval_score": round(float(score), 4)}
        with ablated(self.iface.conns, self.iface.gates, mode):
            with torch.no_grad():
                res = self.iface.forward(prompt_ids)
            token_id = int(res["logits"][0, -1, :].argmax().item())
        return {"text": self.vocab.decode(token_id), "token_ids": [token_id],
                "source": "plastic_pathway", "hit": True}


class ConfiguredAnswerer:
    """Deterministic per-(question, mode) answer table — used ONLY in unit tests
    to make the three provenance buckets decidable without model dynamics."""

    def __init__(self, mapping: Dict[str, Dict[str, str]]):
        self.mapping = mapping                  # mode -> prompt -> answer text

    def answer(self, prompt_text: str, prompt_ids: torch.Tensor, mode: str) -> Dict:
        table = self.mapping.get(mode, {})
        text = table.get(prompt_text, "")
        return {"text": text, "token_ids": None,
                "source": "configured", "hit": bool(text)}


# --------------------------------------------------------------------------- #
# Battery records
# --------------------------------------------------------------------------- #

@dataclass
class Question:
    qid: str
    capability: str
    prompt: str
    paraphrases: List[str]
    gold: str
    tags: Dict[str, str] = field(default_factory=dict)

    def surfaces(self) -> List[Tuple[str, str]]:
        out = [("base", self.prompt)]
        out += [(f"paraphrase{i+1}", p) for i, p in enumerate(self.paraphrases)]
        return out


def load_battery(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def battery_questions(battery: Dict) -> List[Question]:
    return [Question(qid=q["qid"], capability=q["capability"], prompt=q["prompt"],
                     paraphrases=list(q.get("paraphrases", [])), gold=q["gold"],
                     tags=dict(q.get("tags", {})))
            for q in battery.get("questions", [])]


# --------------------------------------------------------------------------- #
# The AION bundle: interface + stores (everything AION-side that is serialized)
# --------------------------------------------------------------------------- #

class AionBundle:
    """RelationalInterface + EpisodicStore + RelationalMemory + ledger."""

    def __init__(self, iface: RelationalInterface, episodic: EpisodicStore,
                 rmem: RelationalMemory, ledger: ConsolidationLedger, vocab: StubVocab):
        self.iface = iface
        self.episodic = episodic
        self.rmem = rmem
        self.ledger = ledger
        self.vocab = vocab

    @classmethod
    def build(cls, seed: int = 0, hidden_size: int = 64, learn_steps_cfg: Optional[Dict] = None):
        torch.manual_seed(seed)
        base = build_stub_base(seed=seed, hidden_size=hidden_size)
        cfg = InterfaceConfig()                       # interface defaults; CPU stub dims
        iface = RelationalInterface(base, cfg=cfg)
        vocab = StubVocab(vocab_size=128)

        episodic = EpisodicStore(clock=lambda: 0.0)   # deterministic law

        def make_probe(prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                res = iface.forward(prompt_ids)
            pre_i, post_i = iface.pairs[0]
            pre = res["hidden"][pre_i].mean(dim=1)[0].float()
            post = res["hidden"][post_i].mean(dim=1)[0].float()
            return pre, post                          # interface already unit-L2 normalizes

        def learn_fn(prompt_ids: torch.Tensor) -> Dict:
            return iface.nnfs_update(prompt_ids)

        rmem = RelationalMemory(conns=list(iface.conns), hidden_dim=iface.hidden_dim,
                                episodic=episodic, blend_ratio=cfg.blend_ratio,
                                strength_threshold=1e-8)
        rmem._host = {"make_probe": make_probe, "learn_fn": learn_fn}  # lane hooks
        ledger = ConsolidationLedger(conns=list(iface.conns), hidden_dim=iface.hidden_dim)
        return cls(iface=iface, episodic=episodic, rmem=rmem, ledger=ledger, vocab=vocab)

    # -- probes ---------------------------------------------------------------- #

    def host_probe(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hooks = getattr(self.rmem, "_host")
        return hooks["make_probe"](prompt_ids)

    def host_learn(self, prompt_ids: torch.Tensor) -> Dict:
        hooks = getattr(self.rmem, "_host")
        return hooks["learn_fn"](prompt_ids)

    def answerer(self) -> ModelPathwayAnswerer:
        return ModelPathwayAnswerer(self.iface, self.episodic, self.vocab)


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #

class RetentionExperiment:
    def __init__(self, variant: str, battery: Dict, run_dir: str, seed: int = 0,
                 n_learn_steps: int = 2, n_gate_steps: int = 0,
                 consolidation_interval: int = 8, retention_fraction: float = 0.5,
                 answerer: Optional[object] = None, verbose: bool = True):
        if variant not in Variant.ALL:
            raise ValueError(f"variant must be one of {Variant.ALL}")
        self.variant = variant
        self.battery = battery
        self.run_dir = run_dir
        self.seed = seed
        self.n_learn_steps = n_learn_steps
        self.n_gate_steps = n_gate_steps
        self.consolidation_interval = consolidation_interval
        self.retention_fraction = retention_fraction
        self.verbose = verbose
        self.questions = battery_questions(battery)
        self.bundle: Optional[AionBundle] = None
        self._answerer = answerer
        self.state_restored_exact: Optional[bool] = None
        self.learn_step = 0

    # -- logging ---------------------------------------------------------------- #

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[retention:{self.variant}] {msg}")

    # -- phase 1: learn ----------------------------------------------------------- #

    def learn_phase(self) -> Dict:
        os.makedirs(self.run_dir, exist_ok=True)
        torch.manual_seed(self.seed)
        self.bundle = AionBundle.build(seed=self.seed)
        b = self.bundle
        self.log(f"variant={VARIANT_DESCRIPTIONS[self.variant]} "
                 f"pairs={b.iface.pairs} hidden={b.iface.hidden_dim}")

        # -- teach the seed relations -- #
        if self.variant == Variant.B:
            # conversation-retrieval: the transcript held Q/A pairs, nothing else.
            # Only BASE prompts were spoken; paraphrases never were.
            for q in self.questions:
                if q.capability in ("recognise", "predict"):
                    b.episodic.record(kind="observation", text=q.prompt,
                                      channel="transcript", teacher="battery",
                                      tags={"answer": q.gold, "kind": "exchange"})
            self.log(f"logged {len(b.episodic.events())} Q/A exchanges to episodic "
                     f"(no plastic learning)")
        for rel in self.battery.get("seed_relations", []):
            key = relation_key(rel["subject"], rel["relation"], rel["object"])
            fact_text = f"{rel['subject']} {rel['relation']} {rel['object']}"
            prompt_ids = b.vocab.encode(fact_text)
            if self.variant == Variant.A:
                with torch.no_grad():                     # frozen: forward only
                    b.iface.forward(prompt_ids)
                self.log(f"frozen pass (no learning): {fact_text}")
            elif self.variant == Variant.B:
                pass                                      # handled above (exchange pass)
            else:                                          # C and D
                rec = b.rmem.learn_relation(
                    key, prompt_ids.flatten().tolist(),
                    learn_fn=b.host_learn,                 # (1,T) tensor in, nnfs pass out
                    teacher="battery", channel="transcript", n_steps=self.n_learn_steps,
                    tags={"kind": "seed"})
                self.log(f"learned {fact_text} (updates {rec.meta['update_ids']}, "
                         f"strengths {b.rmem.strengths(key)})")

        # -- gate steps (C/D; the residual-gate half of interface training) -- #
        if self.variant in (Variant.C, Variant.D) and self.n_gate_steps > 0:
            for rel in self.battery.get("seed_relations", []):
                ids = b.vocab.encode(f"{rel['subject']} {rel['relation']} {rel['object']}")
                for _ in range(self.n_gate_steps):
                    b.iface.gate_step(ids)

        # -- scheduled consolidation (C/D) -- #
        consolidation_report = None
        if self.variant in (Variant.C, Variant.D):
            self.learn_step += max(1, self.n_learn_steps) * max(
                1, len(self.battery.get("seed_relations", [])))
            consolidation_report = b.ledger.maybe_consolidate(
                self.learn_step, self.consolidation_interval) or b.ledger.consolidate_all(
                step=self.learn_step)
            self.log(f"consolidation: invariant_ok="
                     f"{consolidation_report['invariant_ok']}, "
                     f"transferred_pairs={consolidation_report['transferred_pairs']}")

        # -- D only: memory holds the conversation too, then the dream hook
        #    (STUB — dream engine LOCKED until post-pilot) -- #
        dream = None
        if self.variant == Variant.D:
            for q in self.questions:
                if q.capability in ("recognise", "predict"):
                    b.episodic.record(kind="observation", text=q.prompt,
                                      channel="transcript", teacher="battery",
                                      tags={"answer": q.gold, "kind": "exchange"})
            dream = dream_hook(self.run_dir, payload={
                "stage": "learn", "n_lessons": len(b.episodic.events())})

        report = {"variant": self.variant, "phase": "learn",
                  "consolidation": b.ledger.report(),
                  "episodic_events": len(b.episodic.events()),
                  "indexed_relations": len(b.rmem.records),
                  "dream": dream}
        self.serialize()
        return report

    # -- serialization / restart ---------------------------------------------------- #

    def serialize(self) -> None:
        b = self.bundle
        os.makedirs(self.run_dir, exist_ok=True)
        snap = PlasticStateSnapshot.capture(
            b.iface, meta={"variant": self.variant, "seed": self.seed,
                           "battery_id": self.battery.get("battery_id", "?")})
        payload = {"plastic": snap.to_payload(), "probes": b.rmem.save_probes()}
        torch.save(payload, os.path.join(self.run_dir, "state.pt"))
        b.episodic.save(os.path.join(self.run_dir, "episodic.json"))
        with open(os.path.join(self.run_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(b.rmem.to_dict(), f, indent=1, sort_keys=True)
        with open(os.path.join(self.run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": self.variant, "seed": self.seed,
                       "battery_id": self.battery.get("battery_id", "?"),
                       "provenance_buckets": list(MODES)}, f, indent=1)
        self.log(f"serialized AION-side state to {self.run_dir} "
                 f"(foundation NOT serialized: frozen weights simply reload)")

    def restart_from_disk(self) -> None:
        """Fresh objects, loaded ONLY from disk. The strict two-process CLI makes
        the shutdown real (separate processes); in-process this is still a full
        disk round-trip through brand-new objects."""
        payload = torch.load(os.path.join(self.run_dir, "state.pt"), weights_only=False)
        snap = PlasticStateSnapshot.from_payload(payload["plastic"])
        torch.manual_seed(self.seed)
        base = build_stub_base(seed=self.seed, hidden_size=snap.hidden_dim)
        iface = snap.build_interface(base)               # fresh interface + restore
        recheck = PlasticStateSnapshot.capture(iface)
        exact, diffs = snap.equals(recheck, atol=0.0)
        self.state_restored_exact = exact
        self.log(f"restart from disk: plastic state restored "
                 f"{'EXACTLY (bitwise)' if exact else 'WITH DIFFS: ' + str(diffs[:5])}")

        episodic = EpisodicStore.load(os.path.join(self.run_dir, "episodic.json"),
                                      clock=lambda: 0.0)
        with open(os.path.join(self.run_dir, "index.json"), "r", encoding="utf-8") as f:
            index_json = json.load(f)
        vocab = StubVocab(vocab_size=128)

        def make_probe(prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                res = iface.forward(prompt_ids)
            pre_i, post_i = iface.pairs[0]
            return (res["hidden"][pre_i].mean(dim=1)[0].float(),
                    res["hidden"][post_i].mean(dim=1)[0].float())

        def learn_fn(prompt_ids: torch.Tensor) -> Dict:
            return iface.nnfs_update(prompt_ids)

        rmem = RelationalMemory.from_dict(
            index_json, conns=list(iface.conns), episodic=episodic,
            make_probe=make_probe, probes=payload["probes"])
        rmem._host = {"make_probe": make_probe, "learn_fn": learn_fn}
        ledger = ConsolidationLedger(conns=list(iface.conns), hidden_dim=iface.hidden_dim)
        self.bundle = AionBundle(iface=iface, episodic=episodic, rmem=rmem,
                                 ledger=ledger, vocab=vocab)

    # -- phase 2: probe ---------------------------------------------------------------- #

    def _answer(self, q: Question, text: str, mode: str) -> Tuple[Dict, bool]:
        b = self.bundle
        ids = b.vocab.encode(text)
        ans = self._get_answerer().answer(text, ids, mode)
        correct = answer_matches(ans["text"], q.gold)
        return ans, correct

    def _get_answerer(self):
        if self._answerer is not None:
            return self._answerer
        self._answerer = self.bundle.answerer()
        return self._answerer

    def probe_phase(self) -> Dict:
        if self.bundle is None:
            self.restart_from_disk()
        b = self.bundle
        episodic_texts = [normalize_answer(t) for t in b.episodic.texts()]

        # ---- paraphrase battery + provenance accounting ---- #
        q_rows = []
        for q in self.questions:
            for surface, text in q.surfaces():
                modes = {}
                for mode in MODES:
                    ans, correct = self._answer(q, text, mode)
                    modes[mode] = {"answer": ans["text"], "correct": bool(correct),
                                   "source": ans["source"], "hit": ans.get("hit", True)}
                correct_buckets = [m for m in STRONGEST_FIRST if modes[m]["correct"]]
                if correct_buckets:
                    provenance = correct_buckets[0]
                elif modes[FULL]["correct"]:
                    provenance = "FULL_ONLY"
                else:
                    provenance = "NONE"
                internalised = (q.capability in INTERNALISATION_CAPABILITIES) or \
                               (normalize_answer(text) not in episodic_texts)
                q_rows.append({
                    "qid": q.qid, "capability": q.capability, "surface": surface,
                    "prompt": text, "gold": q.gold, "modes": modes,
                    "correct_buckets": correct_buckets, "provenance": provenance,
                    "internalisation_test": bool(internalised),
                    "internalisation_pass": bool(
                        internalised and provenance in (PLASTIC, CONSOLIDATED)),
                })

        # ---- pathway-level NEW COMBINATIONS (the disconnect test's interrogation) ---- #
        combination_checks = []
        for combo in self.battery.get("combinations", []):
            qtext = f"{combo['subject']} {combo['relation']}"
            ids = b.vocab.encode(qtext)
            probe = b.host_probe(ids)
            hits = b.rmem.nearest(probe[0], k=1)
            top = hits[0][0].key if hits else None
            combination_checks.append({
                "query": qtext, "gold": combo["gold"], "nearest_relation": top,
                "reached_gold": bool(top is not None and combo["gold"] in top.split("|")),
            })

        # ---- CHANGE ONE RELATIONSHIP: feed a correction, observe adaptation ---- #
        correction_rows = self._correction_step()

        report = {
            "variant": self.variant,
            "variant_description": VARIANT_DESCRIPTIONS[self.variant],
            "seed": self.seed,
            "battery_id": self.battery.get("battery_id", "?"),
            "restart": {
                "loaded_from_disk_only": True,
                "state_restored_exact": self.state_restored_exact,
            },
            "questions": q_rows,
            "combination_checks": combination_checks,
            "corrections": correction_rows,
            "consolidation": b.ledger.report(),
            "inventory": b.rmem.inventory(),
            "summary": summarize(q_rows, correction_rows, combination_checks),
        }
        with open(os.path.join(self.run_dir, "retention_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
        self._print_summary(report)
        return report

    # -- correction step --------------------------------------------------------------- #

    def _correction_step(self) -> List[Dict]:
        b = self.bundle
        rows = []
        for corr in self.battery.get("corrections", []):
            repl = corr["replaces"]
            old_key = relation_key(repl["subject"], repl["relation"], repl["object"])
            new_key = relation_key(corr["subject"], corr["relation"], corr["object"])
            old_before = b.rmem.strength(old_key)
            old_before_plastic = b.rmem.strength(old_key, blend=1.0)

            if self.variant in (Variant.C, Variant.D):
                prompt_ids = b.vocab.encode(
                    f"{corr['subject']} {corr['relation']} {corr['object']}")
                rec = b.rmem.learn_relation(
                    new_key, prompt_ids.flatten().tolist(),
                    learn_fn=b.host_learn,
                    teacher="reality", channel="transcript", kind="correction",
                    n_steps=self.n_learn_steps,
                    tags={"replaces": old_key})
                if old_key in b.rmem.records:
                    b.rmem.records[old_key].superseded_by = new_key
                self.learn_step += self.n_learn_steps
                b.ledger.maybe_consolidate(self.learn_step, self.consolidation_interval)
            elif self.variant == Variant.B:
                b.episodic.record(kind="correction", text=corr.get(
                    "correction_text", new_key.replace("|", " ")),
                    channel="transcript", teacher="reality",
                    tags={"answer": f"{corr['subject']} {corr['relation']} {corr['object']}",
                          "replaces": old_key})
            # variant A: frozen — nothing at all

            old_after = b.rmem.strength(old_key)
            new_after = b.rmem.strength(new_key)
            new_after_plastic = b.rmem.strength(new_key, blend=1.0)
            old_present_before = old_key in b.rmem.records
            old_intact = bool(
                old_before is not None and old_after is not None
                and old_after >= self.retention_fraction * max(old_before, 1e-12))
            new_learned = bool(new_after is not None
                               and new_after > b.rmem.strength_threshold)
            rows.append({
                "new_key": new_key, "replaces": old_key,
                "old_strength_before": old_before,
                "old_plastic_before": old_before_plastic,
                "old_strength_after": old_after,
                "new_strength_after": new_after,
                "new_plastic_after": new_after_plastic,
                "old_was_indexed": old_present_before,
                "old_intact": old_intact,
                "new_learned": new_learned,
                "adaptation_ok": bool(old_intact and new_learned),
                "learned_by_variant": self.variant in (Variant.C, Variant.D),
            })
            self.log(f"correction: {old_key} -> {new_key} "
                     f"old {old_before} -> {old_after}, new {new_after}, "
                     f"adaptation_ok={rows[-1]['adaptation_ok']}")
        return rows

    # -- summary print ------------------------------------------------------------------ #

    def _print_summary(self, report: Dict) -> None:
        s = report["summary"]
        self.log(f"=== SUMMARY variant {self.variant} ===")
        for mode in MODES:
            self.log(f"  accuracy[{mode:12s}] = {s['accuracy'][mode]:.3f}")
        self.log(f"  internalisation_score = {s['internalisation_score']:.3f} "
                 f"({s['internalisation_passes']}/{s['internalisation_tests']})")
        self.log(f"  provenance histogram: {s['provenance_histogram']}")
        for c in report["corrections"]:
            self.log(f"  correction {c['replaces']} -> {c['new_key']}: "
                     f"old_intact={c['old_intact']} new_learned={c['new_learned']}")
        self.log(f"  consolidation invariant_ok={report['consolidation']['n_violations'] == 0} "
                 f"({report['consolidation']['n_events']} events, "
                 f"{report['consolidation']['total_consolidated_entries']} entries transferred)")


def summarize(q_rows: List[Dict], correction_rows: List[Dict],
              combination_checks: List[Dict]) -> Dict:
    n = max(1, len(q_rows))
    accuracy = {m: sum(1 for r in q_rows if r["modes"][m]["correct"]) / n for m in MODES}
    provenance_histogram: Dict[str, int] = {}
    for r in q_rows:
        provenance_histogram[r["provenance"]] = provenance_histogram.get(r["provenance"], 0) + 1
    itests = [r for r in q_rows if r["internalisation_test"]]
    ipass = [r for r in itests if r["internalisation_pass"]]
    return {
        "n_question_surfaces": len(q_rows),
        "accuracy": accuracy,
        "provenance_histogram": provenance_histogram,
        "internalisation_tests": len(itests),
        "internalisation_passes": len(ipass),
        "internalisation_score": (len(ipass) / len(itests)) if itests else 0.0,
        "correction_adaptation_ok": bool(correction_rows)
        and all(c["adaptation_ok"] for c in correction_rows),
        "combination_reached_gold": (
            sum(1 for c in combination_checks if c["reached_gold"]),
            len(combination_checks)),
    }


# --------------------------------------------------------------------------- #
# DREAM HOOK — STUB ONLY. The dream engine is LOCKED until post-pilot
# (owner law). This hook exists so variant D wiring needs no new code when the
# engine unlocks; it must NEVER import or invoke src/nnfs/dreaming/**.
# --------------------------------------------------------------------------- #

def dream_hook(run_dir: str, payload: Dict) -> Dict:
    marker = os.path.join(run_dir, "dream_hook_stub.json")
    out = {"dream": "stub-locked", "engine": None,
           "note": "dream engine LOCKED until post-pilot; hook wired, engine absent",
           "payload": payload}
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="AION-1 strict retention test (A/B/C/D)")
    ap.add_argument("--variant", default=Variant.C, choices=list(Variant.ALL))
    ap.add_argument("--phase", default="all", choices=("learn", "probe", "all"))
    ap.add_argument("--battery", default=DEFAULT_BATTERY)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn-steps", type=int, default=2)
    ap.add_argument("--gate-steps", type=int, default=0)
    ap.add_argument("--consolidation-interval", type=int, default=8)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on consolidation-invariant violations or "
                         "non-exact state restore")
    args = ap.parse_args()

    battery = load_battery(args.battery)
    run_dir = args.run_dir or os.path.join(_REPO_ROOT, "runs",
                                           f"retention_{args.variant}_seed{args.seed}")
    exp = RetentionExperiment(
        variant=args.variant, battery=battery, run_dir=run_dir, seed=args.seed,
        n_learn_steps=args.learn_steps, n_gate_steps=args.gate_steps,
        consolidation_interval=args.consolidation_interval)

    fatal = []
    if args.phase in ("learn", "all"):
        rep = exp.learn_phase()
        if not rep["consolidation"]["invariant_ok"]:
            fatal.append("consolidation invariant violated during learn phase")
    if args.phase in ("probe", "all"):
        # The probe phase ALWAYS rebuilds from disk only: shutdown is real in the
        # two-process CLI, and --phase all still round-trips everything through
        # state.pt/episodic.json/index.json into brand-new objects.
        exp.bundle = None
        exp.restart_from_disk()
        report = exp.probe_phase()
        if report["restart"]["state_restored_exact"] is False:
            fatal.append("plastic state did NOT restore exactly from disk")
        if not report["consolidation"]["invariant_ok"]:
            fatal.append("consolidation invariant violated during probe phase")

    print(f"\nreport: {os.path.join(run_dir, 'retention_report.json')}"
          if args.phase in ("probe", "all") else f"\nstate: {run_dir}")
    if fatal:
        for f_ in fatal:
            print(f"FATAL: {f_}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
