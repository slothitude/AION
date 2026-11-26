"""LTE-1 STEP 4b — the LTE-1 TRAINING HARNESS (docs/LTE_PROTOCOL.md Revision 8).

Owner's implementation order (protocol section 10): steps 1-3 COMPLETE (gates on the
stand-in + the real host, host immutability demonstrated); step 4 = THIS harness and
the split manifest (aion/experiments/lanes_manifest.py) BEFORE controls/calibration
(step 5) and the final exam (step 6). Steps 5-6 drive this harness.

THE OBJECTIVE (protocol section 2, implemented literally):

  prediction channel — the corrected plasticity law EXACTLY as frozen (baseline v2):
      delta_u_pred = 0.7 * eta_p * delta_pred * psi
      -> aion/relational/interface.RelationalInterface.nnfs_update, untouched landed
      code (plasticity_step, plastic.py:319-369). No harness arithmetic on this path.

  expression channel — gradient DESCENT on the weighted TRAINING objective:
      delta_u_expr = -eta_e * grad_{u_fast}( lambda_r * L_rank,train
                                            + lambda_d * L_disc,train )
      u_fast <- u_fast + delta_u_pred + delta_u_expr   (additive; separate passes,
      see SEQUENCING below)

  L_rank,train (section 2.2): CE over per-candidate frozen-host scores on TRAIN
  questions — each candidate scored on ITS OWN cached prompt+cand context (the
  step-2 literal-scoring structure, aion/experiments/lte_gates_real.py), plastic
  residual injected at the answer positions, score summed over candidate tokens.

  L_disc,train (section 2.3): pairwise cosine hinge over per-fact features from
  TRAIN carriers (fact i's canonical teach question), tau_d = 0.20, float32 cosine,
  zero-norm => cos exactly 0. Masked extraction: fact i's feature reads ONLY fact
  i's connection's residual. DEV carriers are feature-extraction-forbidden
  ABSOLUTELY (harness assertion aborts as an instrument fault). Exam carriers are
  extraction-only, post-freeze, after behavioural scoring, nothing follows them.

JACOBIAN LAW (owner): grad flows through the EFFECTIVE connection
w = blend*u_fast + (1-blend)*theta_slow (the actual forward); the 0.7 blend enters
the gradient ONLY through that forward's Jacobian. The expression path is NEVER
multiplied by another 0.7 — the descent step is literally u -= eta_e * grad.

SEQUENCING (recorded design decision): the untouched gamma-sequencing law
(section 2.4; Phase-2/3 [D2/D3]: plasticity FIRST, gate SECOND, gate_lr 100.0,
2 gate steps per QA experience) is kept in force, so the teach pass is three
recorded sub-passes over the manifest's fixed experience order:
  pass A (plasticity)   every experience -> nnfs_update (delta_u_pred only)
  pass B (gamma)        every QA experience -> gate_step x gate_steps
  pass C (expression)   every QA experience -> one delta_u_expr step on that
                        experience's TRAIN question + the L_disc,train carrier term
The expression pass is the protocol section 5 "full pass" whose train-L_rank curve
drives the instability check. Consolidation is HARD-DISABLED for this run
(plasticity.consolidation_rate = 0.0, plastic.py:403-412 rate<=0 branch).

STRUCTURE: one connection per taught fact (the step-2 gate-fixture precedent —
fact i's per-fact residual block IS connection i's u_fast block, which the swap
test section 6.1 requires to be well-defined). Layer pairs are the interface's own
pick_hook_pairs order extended to the fact count with RealHostFixture._select_pairs'
recorded family-walk pattern. On hosts with fewer hookable pairs than facts (the
tiny stand-in only) facts map round-robin and the swap test is refused.

FROZEN-HOST CACHES: every context's host outputs (answer-position base-logit rows,
seq-mean normalized pre/post states per pair) are cached ONCE — host outputs are
u-independent (the base runs once under no_grad; the residual enters only at the
logits), so the caches are state-independent and SHARED across every calibration
attempt (protocol section 5 retry reset rule: identical inputs per attempt, fresh
plastic lane per attempt). psi is recomputed per context from the cached states
through the REAL RelationshipFunction, detached — a constant in the plastic
objective, exactly as in the step-2 gate fixture.

GATES (section 6.2): already demonstrated (step 1 stand-in `227c638`/`717624f`;
step 2 real host PASS, runs/lte_gates/gates_real_20261126_135538.json). This
harness does NOT re-derive them: pre-flight verifies the recorded step-2 PASS
evidence (default) or re-RUNS the imported lte_gates_real gates on a fresh
RealHostFixture (--rerun-gates, heavy). The run report always carries the
gate-pass reference.

HOST IMMUTABILITY (section 3.1): no optimizer over any foundation parameter
(expression autograd runs on detached u_fast LEAVES only); foundation fingerprints
at load / teach-end / run-end; isfinite(delta_u_expr).all() required on every
expression step (NaN/Inf = instrument fault — instability-ladder retry in
calibration mode, run-abort in train mode); legitimate zero-gradient steps are
RECORDED, never aborted, and counted per run.

CALIBRATION (section 5, --dev): the ONE deterministic nested sequence — configs
A (lambda_d 0.5) / B (lambda_d 0.0) x eta_e grid {1e-4,3e-4,1e-3,3e-3,1e-2} at
lambda_r 1.0, instability ladder lambda_r -> 0.5 -> 0.25 -> ABORTED, every retry
from the SAME frozen initial state + IDENTICAL RNG (fresh seeded lane), selection
on dev combination accuracy@1 (PLASTIC mode, the section 7 behavioural instrument)
with gold-margin then smallest-eta_e tie-breaks, lambda_d decision rule exactly as
preregistered. DEV contributes combination accuracy@1 + gold margin ONLY.

EXAM (--score-exam, step 6): scored EXACTLY ONCE per arm — a marker file guards
re-scoring; after behavioural scoring the state is frozen and ONLY THEN are the
exam diagnostics extracted (section 2.3 exam features -> L_disc,exam / novelty,
section 6.1 swap test) with no update ever following.

CLI:
  python aion/experiments/lte_train.py --dev --manifest runs/lte/manifest.json
  python aion/experiments/lte_train.py --eta-e 3e-4 --lambda-r 1.0 --lambda-d 0.5 \
      --manifest runs/lte/manifest.json --score-exam
  python aion/experiments/lte_train.py --foundation tiny ...   # Rog CPU debug lane
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

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

from aion.experiments.teacher_student import (                            # noqa: E402
    FORMAT_SCAFFOLD, build_tiny_foundation)
from aion.experiments.lanes_manifest import recheck_manifest, sha256_text  # noqa: E402
from aion.experiments.lte_gates_real import (                             # noqa: E402
    MODEL_DIR, RealHostFixture, TAU_D, run_gate1, run_gate2, sha256_tensor)
from aion.relational.interface import (                                   # noqa: E402
    InterfaceConfig, RelationalInterface, discover_decoder_layers,
    fingerprints_equal, foundation_fingerprint, pick_hook_pairs)

# ---- preregistered constants ---------------------------------------------- #

SEED = 20261126
ETA_E_GRID = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)   # section 5 grid order
LAMBDA_R_LADDER = (1.0, 0.5, 0.25)                       # section 5 instability ladder
LAMBDA_D_CONFIGS = (("A", 0.5), ("B", 0.0))              # section 5 configurations
DEFAULT_LAMBDA_R = 1.0
NOVELTY_FLOOR = 0.80                 # section 5 lambda_d decision + section 7 state
COMBO_TIE_TOL = 0.05                 # section 5 lambda_d decision
DEFAULT_GATE_LR = 100.0              # [D2] untouched gamma law
DEFAULT_GATE_STEPS = 2               # [D3] untouched gamma law
DEFAULT_LEARNING_RATE = 0.1          # eta_p, Phase-2/3 default (prediction channel)
MIRROR_FIDELITY_REL = 1e-5           # L1 bound (identical float32 formulas)
MIRROR_L2_BF16_BOUND = 2e-2          # L2 bound when the pathway casts to BF16
FIDELITY_CONTEXTS = 2                # contexts probed for L1/L2
CHUNK_SELFCHECK_SUBSET = 3           # candidates in the chunking self-check


def log(msg: str) -> None:
    print(f"[lte-train] {msg}", flush=True)


def rss_gb() -> Optional[float]:
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     / (1024.0 * 1024.0), 2)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Layer-pair selection — RealHostFixture._select_pairs generalized to n pairs
# --------------------------------------------------------------------------- #

def select_pairs_for_facts(layers, all_pairs, n_facts: int) -> Tuple[list, str]:
    """The step-2 recorded fixture choice, generalized: first the interface's own
    pick_hook_pairs order, then RealHostFixture._select_pairs (imported, reused
    verbatim) for its first-4 extension, then the same family-walk pattern cycled
    until n_facts pairs, then a generic adjacent-pair fallback. Recorded."""
    pairs, note4 = RealHostFixture._select_pairs(layers, all_pairs)
    used = {i for pr in pairs for i in pr}
    added = []
    fams = ("deltanet", "attention")
    fi = 0
    guard = 0
    while len(pairs) < n_facts and guard < 20 * len(layers):
        fam = fams[fi % len(fams)]
        found = None
        for l in layers:
            if (l.family == fam and l.index not in used
                    and (l.index + 1) < len(layers)
                    and (l.index, l.index + 1) not in pairs):
                found = (l.index, l.index + 1)
                break
        if found is None and fi % len(fams) == len(fams) - 1:
            # both families exhausted once -> generic adjacent fallback (the
            # pick_hook_pairs generic-fallback pattern)
            for l in layers:
                if (l.index not in used and (l.index + 1) < len(layers)
                        and (l.index, l.index + 1) not in pairs):
                    found = (l.index, l.index + 1)
                    break
        if found is not None:
            pairs.append(found)
            used.add(found[0])
            added.append(found)
        fi += 1
        guard += 1
    if len(pairs) < n_facts:
        # TINY STAND-IN ONLY (debug lane): fewer hookable adjacent pairs than
        # facts -> use what exists; the harness maps facts round-robin and
        # REFUSES the swap test (no 1:1 per-fact block). The real host asserts
        # 1:1 (foundation == qwen -> LteTrainHost.one_to_one is required).
        note = (f"{note4}; only {len(pairs)} hookable pairs available for "
                f"{n_facts} facts (tiny stand-in) -> facts map round-robin; "
                f"swap test refused")
        return pairs, note
    note = (f"{note4}; extended to {n_facts} pairs by the same family-walk pattern "
            f"(deltanet->attention cycled, then generic adjacent fallback): {added}")
    return pairs[:n_facts], note


# --------------------------------------------------------------------------- #
# The training host: frozen foundation + frozen-host context caches + mirror
# --------------------------------------------------------------------------- #

class LteTrainHost:
    """Frozen foundation + per-context host-output caches + the plastic-objective
    mirror. The MIRROR residual is RealHostFixture's own method, borrowed verbatim
    (the step-2 gate-proven formula); psi comes from the REAL RelationshipFunction
    on cached states, detached — a u-independent constant, exactly as in step 2.
    The plastic LANE (RelationalInterface: conns/gates/traces) is attached fresh
    per training attempt (section 5 retry reset rule)."""

    # step-2 mirror formula, reused VERBATIM from the gate fixture: the blend
    # enters only here, through this forward's Jacobian.
    _residual_for_conn = RealHostFixture._residual_for_conn

    def __init__(self, manifest: Dict, args):
        self.manifest = manifest
        self.args = args
        self.seed = args.seed
        self.t0 = time.time()
        self.forward_times: List[float] = []
        self.logits_dtype_observed = None

        # ---- foundation: EXACTLY RealHostFixture's load --------------------- #
        torch.manual_seed(self.seed)   # lane-agnostic here; re-seeded per lane
        if args.foundation == "qwen":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            src = args.local_dir
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    device_map=None, local_files_only=True).to(args.device)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    src, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    device_map=None, local_files_only=True).to(args.device)
            model.eval()
            tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
            self.foundation_label = f"Qwen3.5-4B-Base @ {src} (CPU BF16, FROZEN)"
        else:
            model, tok = build_tiny_foundation(seed=self.seed)
            self.foundation_label = "tiny-standin(Llama-4L-64D, CPU, FROZEN)"
        self.tok = tok
        self.model = model
        self.device = args.device
        self.exam_scored = False
        self.fingerprint_at_load = foundation_fingerprint(model)
        self.load_s = round(time.time() - self.t0, 1)

        # ---- layer pairs: one connection per taught fact -------------------- #
        self.n_facts = len(self.manifest["train"]["experiences"]) // 3
        layers = discover_decoder_layers(model)
        self.n_layers = len(layers)
        pairs, pair_note = select_pairs_for_facts(layers, pick_hook_pairs(layers),
                                                  self.n_facts)
        self.layer_pairs = [tuple(p) for p in pairs]
        self.pair_note = pair_note
        del layers
        gc.collect()

        # ---- the reference lane (lane 0; fresh lanes attach per attempt) ---- #
        self.cfg = InterfaceConfig(learning_rate=args.learning_rate,
                                   gate_lr=args.gate_lr)
        self._attach_lane()
        self.fidelity_note = f"pairs: {self.layer_pairs} — {pair_note}"

        # ---- mirror head: float32 view of the (possibly BF16-valued) head --- #
        self.head_w = self.model.get_output_embeddings().weight.detach().float()
        self.head_dtype = str(self.model.get_output_embeddings().weight.dtype)

        # ---- frozen-host caches --------------------------------------------- #
        self.ctx_train: Dict[int, List[Dict]] = {}
        self.ctx_dev: Dict[str, List[Dict]] = {}
        self.ctx_exam: Dict[str, List[Dict]] = {}
        self._cache_train()
        self._cache_dev()
        if getattr(args, "score_exam", False):
            self._cache_exam()
        self.caches_s = round(time.time() - self.t0, 1)

        # ---- in-harness verification (step 4's verification duty) ----------- #
        self.fidelity = self.assert_mirror_fidelity()
        self.chunk_selfcheck = self.assert_chunked_gradient_identity()

    # ---- lane lifetime ------------------------------------------------------ #

    def _attach_lane(self) -> Dict:
        """A FRESH plastic lane (section 5 retry reset): identical seed -> identical
        relationship-W draws -> identical initial plastic state. The previous
        lane's hooks are removed first."""
        if getattr(self, "lane", None) is not None:
            self._drop_lane()
        torch.manual_seed(self.seed)     # IDENTICAL RNG state per attempt (law)
        iface = RelationalInterface(
            self.model, cfg=InterfaceConfig(learning_rate=self.args.learning_rate,
                                            gate_lr=self.args.gate_lr),
            layer_pairs=list(self.layer_pairs))
        iface.eval()
        for p in iface.base.parameters():
            p.requires_grad_(False)
        for p in iface.conns.parameters():
            p.requires_grad_(False)      # NNFS owns these (landed law)
        # gates stay TRAINABLE (the untouched gamma law trains them in pass B)
        assert all(bool(g.requires_grad) for g in iface.gates)
        assert not any(p.requires_grad for p in iface.base.parameters())
        self.n_conns = len(iface.conns)
        # 1:1 fact->connection when the host has enough hookable pairs (real host);
        # round-robin only on the tiny stand-in, where the swap test is refused.
        self.conn_of_fact = [i % self.n_conns for i in range(self.n_facts)]
        self.one_to_one = self.n_conns >= self.n_facts
        # CONSOLIDATION HARD-DISABLE (protocol section 2.4, rate<=0 law,
        # plastic.py:403-412) — recorded; the interval alone must not fire sweeps.
        self.consolidation_disable = []
        for conn in iface.conns:
            conn.plasticity.consolidation_rate = 0.0
            self.consolidation_disable.append(
                float(conn.plasticity.consolidation_rate))
        self.lane = iface
        self.lane_seed = self.seed
        self.blend = float(iface.cfg.blend_ratio)   # the borrowed step-2 mirror
        return {"n_conns": self.n_conns, "one_to_one": self.one_to_one,
                "consolidation_rate_set": self.consolidation_disable,
                "seed": self.seed}

    def _drop_lane(self) -> None:
        for h in getattr(self.lane, "_hooks", []) or []:
            h.remove()
        self.lane = None
        gc.collect()

    # ---- encoding + caching ------------------------------------------------- #

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def _cache_context(self, prompt: str, cand_text: str) -> Dict:
        """ONE frozen-host forward per (prompt, candidate) context: answer-position
        base-logit rows (float32) + seq-mean normalized pre/post states per pair.
        Host outputs are u-independent -> cached once, shared by every attempt."""
        ids_prompt = self.encode(prompt)
        ids_c = self.encode(" " + cand_text)
        full = ids_prompt + ids_c
        tf = time.time()
        self.lane._captured.clear()
        with torch.no_grad():
            out = self.lane.base(input_ids=torch.tensor([full], dtype=torch.long))
        self.forward_times.append(round(time.time() - tf, 2))
        self.logits_dtype_observed = str(out.logits.dtype)
        base_logits = out.logits[0].float()              # (T, V)
        hidden = dict(self.lane._captured)
        pres, posts = [], []
        for k, (pre_i, post_i) in enumerate(self.lane.pairs):
            pre_s = hidden[pre_i].mean(dim=1).float()    # (1, D) — interface.py:281-285
            post_s = hidden[post_i].mean(dim=1).float()
            if self.lane.cfg.normalize_states:
                pre_s = torch.nn.functional.normalize(pre_s, dim=-1)
                post_s = torch.nn.functional.normalize(post_s, dim=-1)
            pres.append(pre_s[0])
            posts.append(post_s[0])
        start = len(ids_prompt) - 1                      # rows predicting cand tokens
        rows = base_logits[start:start + len(ids_c)]     # (n_tok, V) float32
        return {"prompt": prompt, "cand_text": cand_text, "ids_c": ids_c,
                "prompt_len": len(ids_prompt), "full_len": len(full),
                "answer_rows": rows, "pres": pres, "posts": posts}

    def _cache_train(self) -> None:
        for e in self.manifest["train"]["experiences"]:
            if e["kind"] != "qa":
                continue
            prompt = FORMAT_SCAFFOLD + f"Question: {e['question']}\nAnswer:"
            self.ctx_train[e["exp_index"]] = [
                self._cache_context(prompt, c) for c in e["rank_candidates"]]

    def _cache_dev(self) -> None:
        for q in self.manifest["dev"]["questions"]:
            prompt = FORMAT_SCAFFOLD + f"Question: {q['text']}\nAnswer:"
            self.ctx_dev[q["qid"]] = [self._cache_context(prompt, c)
                                      for c in q["candidates"]]

    def _cache_exam(self) -> None:
        for q in self.manifest["final_exam"]["questions"]:
            prompt = FORMAT_SCAFFOLD + f"Question: {q['text']}\nAnswer:"
            self.ctx_exam[q["qid"]] = [self._cache_context(prompt, c)
                                       for c in q["candidates"]]

    def cache_stats(self) -> Dict:
        n = (sum(len(v) for v in self.ctx_train.values())
             + sum(len(v) for v in self.ctx_dev.values())
             + sum(len(v) for v in self.ctx_exam.values()))
        rows = 0
        for store in (self.ctx_train, self.ctx_dev, self.ctx_exam):
            for lst in store.values():
                for c in lst:
                    rows += c["answer_rows"].numel()
        return {"contexts": n, "cached_logit_elements": rows,
                "cached_logit_mb": round(rows * 4 / 1e6, 1),
                "host_forwards": len(self.forward_times),
                "mean_forward_s": round(sum(self.forward_times)
                                        / max(1, len(self.forward_times)), 2),
                "logits_dtype": self.logits_dtype_observed}

    # ---- section 2.3 hard invariants as harness assertions ------------------ #

    def _carrier_ctx(self, fact: int, which: str) -> Dict:
        """The carrier context for fact's per-fact feature. DEV carriers are
        FORBIDDEN ABSOLUTELY (section 2.3: the harness assertion aborts as an
        instrument fault). EXAM carriers require the post-freeze flag."""
        if which == "dev":
            raise AssertionError(
                "INSTRUMENT FAULT: feature extraction on a DEV carrier is forbidden "
                "(protocol section 2.3 carrier-source rule — absolute prohibition)")
        if which == "exam":
            assert getattr(self, "exam_scored", False), (
                "INSTRUMENT FAULT: exam-carrier extraction requires the frozen "
                "post-scoring state (protocol section 2.3: extraction ONLY, after "
                "behavioural scoring, nothing follows)")
            qid = self.manifest["final_exam"]["carrier_per_fact"][str(fact)]
            gold = self._exam_gold(fact)
            idx = self.manifest_candidate_index("exam", qid, gold)
            return self.ctx_exam[qid][idx]
        # train: fact i's canonical teach question (manifest-recorded)
        c = self.manifest["train"]["canonical_carrier_per_fact"][str(fact)]
        exp = self.manifest["train"]["experiences"][c["carrier_exp_index"]]
        idx = exp["rank_candidates"].index(c["carrier_candidate"])
        return self.ctx_train[c["carrier_exp_index"]][idx]

    def manifest_candidate_index(self, split: str, qid: str, gold: str) -> int:
        rows = (self.manifest["dev"]["questions"] if split == "dev"
                else self.manifest["final_exam"]["questions"])
        q = next(x for x in rows if x["qid"] == qid)
        return q["candidates"].index(gold)

    def _exam_gold(self, fact: int) -> str:
        qid = self.manifest["final_exam"]["carrier_per_fact"][str(fact)]
        q = next(x for x in self.manifest["final_exam"]["questions"]
                 if x["qid"] == qid)
        return q["answer"]

    # ---- the plastic-objective mirror --------------------------------------- #

    def _gamma(self, k: int) -> float:
        return float(self.lane.gates[k].detach().item())

    def _theta(self, k: int) -> torch.Tensor:
        return self.lane.conns[k].plasticity.theta_slow.detach()

    def _psi(self, ctx: Dict, k: int) -> torch.Tensor:
        """psi for connection k on a cached context: the REAL RelationshipFunction
        on the cached states, detached (u-independent constant; W frozen)."""
        conn = self.lane.conns[k]
        pre, post = ctx["pres"][k], ctx["posts"][k]
        with torch.no_grad():
            psi = conn.relationship(pre.unsqueeze(0), post.unsqueeze(0))
        return psi.detach()                              # (1, D_post, D_pre)

    def _conn_residual(self, u_k, k: int, psi, mode: str = "full"):
        """gamma-scaled residual of connection k. mode 'full' = the REAL forward's
        effective connection (the borrowed step-2 formula — the objective's path);
        'plastic'/'consolidated' are the retention-lane ablation views used ONLY
        for the dev/exam scoring instrument, never inside any gradient."""
        gamma = self._gamma(k)
        if mode == "full":
            return self._residual_for_conn(u_k, self._theta(k), psi, gamma)
        w = u_k if mode == "plastic" else self._theta(k)
        return gamma * (w.unsqueeze(0) * psi).sum(dim=-1).squeeze(0)

    def _context_delta(self, ctx: Dict, u_list, mode: str = "full") -> torch.Tensor:
        """Total injected residual delta on one context = sum over ALL connections
        (the interface broadcasts the total residual)."""
        rs = [self._conn_residual(u_list[k], k, self._psi(ctx, k), mode)
              for k in range(self.n_conns)]
        return torch.stack(rs, dim=0).sum(dim=0)          # (D,)

    def _score_context(self, ctx: Dict, u_list, mode: str = "full") -> torch.Tensor:
        """LITERAL section 2.2 scoring on THIS candidate's own frozen-host context:
        cached answer rows + plastic delta through the float32 head, log-softmax,
        candidate tokens summed."""
        delta = self._context_delta(ctx, u_list, mode)
        lp = torch.log_softmax(ctx["answer_rows"] + delta @ self.head_w.T, dim=-1)
        s = None
        for j, t in enumerate(ctx["ids_c"]):
            term = lp[j, t]
            s = term if s is None else s + term
        return s

    def _fact_feature(self, fact: int, u_list, which: str = "train"):
        """LITERAL section 2.3 per-fact feature: gamma-scaled plastic residual at
        the answer position from FACT'S OWN connection only (masked: every other
        connection contributes EXACTLY ZERO), on the fact's carrier context."""
        k = self.conn_of_fact[fact]
        psi = self._psi(self._carrier_ctx(fact, which), k)
        return self._conn_residual(u_list[k], k, psi, mode="full")

    def _features(self, u_list, which: str = "train"):
        """Per-fact features + guarded L2 normalization (zero-norm => the zero
        vector => every cosine exactly 0 — no NaN; float32 in production)."""
        raw, normalized, norms = [], [], []
        for fact in range(self.n_facts):
            f = self._fact_feature(fact, u_list, which)
            raw.append(f)
            nrm = torch.linalg.vector_norm(f)
            norms.append(float(nrm.detach()))
            normalized.append(torch.zeros_like(f) if float(nrm.detach()) == 0.0
                              else f / nrm)
        return raw, normalized, norms

    # ---- the two TRAIN loss terms ------------------------------------------- #

    def rank_loss(self, exp_entry: Dict, u_list) -> Tuple[torch.Tensor, torch.Tensor]:
        """(L_rank value, scores) for one TRAIN question — CE over per-candidate
        frozen-host scores (section 2.2)."""
        ctxs = self.ctx_train[exp_entry["exp_index"]]
        with torch.no_grad():
            scores = torch.stack([self._score_context(c, u_list) for c in ctxs])
        gold = exp_entry["rank_candidates"].index(exp_entry["gold_candidate"])
        loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0),
                                                 torch.tensor([gold]))
        return loss, scores

    def disc_loss(self, u_list, which: str = "train") -> torch.Tensor:
        """L_disc: pairwise cosine hinge over the per-fact features, tau_d = 0.20
        (TAU_D imported from the step-2 gate fixture — single source)."""
        _raw, normalized, _norms = self._features(u_list, which)
        terms = []
        for i in range(self.n_facts):
            for j in range(self.n_facts):
                if i == j:
                    continue
                cos = torch.dot(normalized[i], normalized[j])
                terms.append(torch.clamp(cos - TAU_D, min=0.0))
        return torch.stack(terms).mean()

    def novelty(self, u_list, which: str = "train") -> float:
        """mean_{i!=j}(1 - cos(f_i, f_j)) on the given carrier context class."""
        _raw, normalized, _norms = self._features(u_list, which)
        ds = [float(1.0 - torch.dot(normalized[i], normalized[j]))
              for i in range(self.n_facts) for j in range(self.n_facts) if i != j]
        return round(sum(ds) / len(ds), 6)

    # ---- the expression gradient (delta_u_expr) ------------------------------ #

    def expression_grad(self, exp_entry: Dict,
                        lambda_r: float, lambda_d: float) -> Tuple[List, Dict]:
        """delta_u_expr = -eta_e * grad( lambda_r*L_rank + lambda_d*L_disc ).
        The gradient runs on DETACHED u_fast LEAVES through the effective
        connection (blend via the borrowed forward's Jacobian — never re-applied).

        Rank chunking (exact by linearity — the step-2 chunking identity):
          CE over candidates: dL/ds_k = p_k - [k=gold], so
          grad L_rank = sum_k (p_k - [k=gold]) * grad s_k, and each candidate's
          (lambda_r * upstream_k * s_k) is backwarded SEPARATELY (bounds the
          retained graph to one candidate's chain). Verified exact by
          assert_chunked_gradient_identity at init.
        Disc: the whole features graph (D-wide per-fact vectors) in one backward.
        """
        leaves = [c.plasticity.u_fast.detach().clone().requires_grad_(True)
                  for c in self.lane.conns]
        acc = [torch.zeros_like(u) for u in leaves]
        rec: Dict = {}

        # phase 1 — candidate scores at the current state (no graph)
        ctxs = self.ctx_train[exp_entry["exp_index"]]
        with torch.no_grad():
            scores = torch.stack([self._score_context(c, leaves) for c in ctxs])
        gold = exp_entry["rank_candidates"].index(exp_entry["gold_candidate"])
        L_rank = float(torch.nn.functional.cross_entropy(
            scores.unsqueeze(0), torch.tensor([gold])))
        p = torch.softmax(scores, dim=-1)
        rec["L_rank"] = L_rank
        rec["scores_mean"] = float(scores.mean())

        # phase 2 — per-candidate chunked backward of lambda_r * L_rank
        for k_c, ctx in enumerate(ctxs):
            up = float(p[k_c].item()) - (1.0 if k_c == gold else 0.0)
            if up == 0.0:
                continue
            s = self._score_context(ctx, leaves)
            (lambda_r * up * s).backward()
            for kk in range(self.n_conns):
                acc[kk] = acc[kk] + leaves[kk].grad.detach()
                leaves[kk].grad = None
            del s
        rec["rank_chunks"] = len(ctxs)

        # disc term — whole graph (section 2.3 train carriers only)
        if lambda_d != 0.0:
            L_disc = self.disc_loss(leaves, which="train")
            rec["L_disc"] = float(L_disc.detach())
            (lambda_d * L_disc).backward()
            for kk in range(self.n_conns):
                acc[kk] = acc[kk] + leaves[kk].grad.detach()
                leaves[kk].grad = None
            del L_disc
        else:
            rec["L_disc"] = None
        for leaf in leaves:
            leaf.grad = None
        grads_finite = bool(all(torch.isfinite(a).all() for a in acc))
        total_norm = float(torch.sqrt(sum((a ** 2).sum() for a in acc)))
        rec["grad_finite"] = grads_finite
        rec["grad_norm"] = total_norm
        rec["zero_grad"] = bool(total_norm == 0.0)
        del leaves
        gc.collect()
        return acc, rec

    def expression_step(self, exp_entry: Dict, cfg: Dict) -> Dict:
        """ONE expression step at the live lane state: gradient -> finiteness gate
        -> u_fast -= eta_e * grad (NO extra blend factor — Jacobian law). Returns
        the step record; raises on non-finite gradients per section 3.1(4)."""
        u_before = [c.plasticity.u_fast.detach().clone() for c in self.lane.conns]
        acc, rec = self.expression_grad(exp_entry, cfg["lambda_r"], cfg["lambda_d"])
        rec["mode"] = cfg.get("mode", "train")
        if not rec["grad_finite"]:
            if cfg["mode"] == "calibrate":
                rec["unstable"] = True     # section 5b: instability -> ladder retry
                return rec
            raise AssertionError(          # section 3.1(4): run-aborting fault
                f"INSTRUMENT FAULT: non-finite expression gradient at experience "
                f"{exp_entry['exp_index']} (train mode aborts per section 3.1(4))")
        if rec["zero_grad"]:
            rec["applied"] = False         # legitimate zero step: recorded, not
            return rec                     # aborted (section 3.1(4))
        with torch.no_grad():
            for conn, a in zip(self.lane.conns, acc):
                conn.plasticity.u_fast.add_(-cfg["eta_e"] * a)
        rec["applied"] = True
        rec["du_expr_norm"] = rec["grad_norm"] * cfg["eta_e"]
        rec["u_fast_norms_after"] = [round(float(c.plasticity.u_fast.norm()), 6)
                                     for c in self.lane.conns]
        rec["u_fast_changed"] = bool(any(
            not torch.equal(c.plasticity.u_fast.detach(), b)
            for c, b in zip(self.lane.conns, u_before)))
        return rec

    # ---- teach passes (A plasticity, B gamma, C expression) ------------------ #

    def teach(self, cfg: Dict) -> Dict:
        exps = self.manifest["train"]["experiences"]
        qa = [e for e in exps if e["kind"] == "qa"]
        t0 = time.time()
        # pass A — delta_u_pred EXACTLY as frozen (nnfs_update; consolidation
        # hard-disabled above; no harness arithmetic on this channel)
        nnfs_stats = []
        for e in exps:
            res = self.lane.nnfs_update(
                torch.tensor([self.encode(e["text"])], dtype=torch.long))
            nnfs_stats.append(res)
        pass_a_s = round(time.time() - t0, 1)
        # pass B — the untouched gamma law ([D2/D3]: gate_lr, 2 steps per QA)
        t1 = time.time()
        gate_curve = []
        for e in qa:
            ids = torch.tensor([self.encode(e["text"])], dtype=torch.long)
            for _ in range(cfg["gate_steps"]):
                res = self.lane.gate_step(ids)
                gate_curve.append(res)
        pass_b_s = round(time.time() - t1, 1)
        # pass C — the expression pass (the section 5 "full pass")
        t2 = time.time()
        curve = []
        zero_steps = 0
        unstable = False
        for e in qa:
            rec = self.expression_step(e, cfg)
            if rec.get("unstable"):
                unstable = True
                break
            zero_steps += int(rec.get("zero_grad", False))
            curve.append({"exp_index": e["exp_index"], "L_rank": rec["L_rank"],
                          "L_disc": rec["L_disc"], "grad_norm": rec["grad_norm"],
                          "applied": rec.get("applied")})
        pass_c_s = round(time.time() - t2, 1)
        lr = [c["L_rank"] for c in curve]
        q1 = lr[:max(1, len(lr) // 4)]
        q4 = lr[-max(1, len(lr) // 4):] if lr else []
        instability = {
            "nonfinite_expression_loss": unstable,
            "L_rank_first_quarter_mean": round(sum(q1) / len(q1), 6) if q1 else None,
            "L_rank_last_quarter_mean": round(sum(q4) / len(q4), 6) if q4 else None,
            "L_rank_rising_over_pass": bool(
                q1 and q4 and sum(q4) / len(q4) > sum(q1) / len(q1)),
        }
        instability["unstable"] = bool(unstable
                                       or instability["L_rank_rising_over_pass"])
        out = {"eta_e": cfg["eta_e"], "lambda_r": cfg["lambda_r"],
               "lambda_d": cfg["lambda_d"], "gate_steps": cfg["gate_steps"],
               "gate_lr": cfg["gate_lr"],
               "pass_a_plasticity": {"experiences": len(exps), "wall_s": pass_a_s},
               "pass_b_gamma": {"qa_experiences": len(qa),
                                "gate_steps": len(gate_curve),
                                "gamma_after": [round(self._gamma(k), 6)
                                                for k in range(self.n_conns)],
                                "wall_s": pass_b_s},
               "pass_c_expression": {"steps": len(curve), "zero_grad_steps":
                                     zero_steps, "wall_s": pass_c_s,
                                     "L_rank_curve": lr,
                                     "curve_head": curve[:4],
                                     "curve_tail": curve[-4:]},
               "instability": instability,
               "u_fast_norms_end": [round(float(c.plasticity.u_fast.norm()), 6)
                                    for c in self.lane.conns],
               "gamma_end": [round(self._gamma(k), 6) for k in range(self.n_conns)],
               "teach_s": round(time.time() - t0, 1)}
        return out

    # ---- scoring instrument (dev / exam) ------------------------------------- #

    def score_question(self, q_entry: Dict, u_list, mode: str) -> Dict:
        ctxs = self.ctx_exam.get(q_entry["qid"]) if q_entry["qid"] in self.ctx_exam \
            else self.ctx_dev[q_entry["qid"]]
        with torch.no_grad():
            per = [float(self._score_context(c, u_list, mode)) for c in ctxs]
        cands = q_entry["candidates"]
        gold = q_entry["answer"]
        ranked = sorted(range(len(cands)), key=lambda i: -per[i])
        distr = [per[i] for i in range(len(cands)) if cands[i] != gold]
        return {"qid": q_entry["qid"], "battery": q_entry["battery"], "mode": mode,
                "argmax": cands[ranked[0]],
                "correct": bool(cands[ranked[0]] == gold),
                "gold_rank": ranked.index(cands.index(gold)) + 1,
                "gold_score": round(per[cands.index(gold)], 6),
                "gold_margin": round(per[cands.index(gold)] - max(distr), 6)
                if distr else None,
                "chance": round(1.0 / len(cands), 4),
                "scores": {c: round(v, 6) for c, v in zip(cands, per)}}

    def score_split(self, split: str, mode: str) -> List[Dict]:
        u_list = [c.plasticity.u_fast.detach() for c in self.lane.conns]
        rows = (self.manifest["dev"]["questions"] if split == "dev"
                else self.manifest["final_exam"]["questions"])
        return [self.score_question(q, u_list, mode) for q in rows]

    def dev_summary(self) -> Dict:
        """Section 5c: dev combination accuracy@1 (PLASTIC — the section 7
        behavioural instrument) + mean gold margin; FULL recorded alongside.
        NOTHING else is read from DEV."""
        out = {}
        for mode in ("plastic", "full"):
            rows = self.score_split("dev", mode)
            combo = [r for r in rows if r["battery"] == "combination"]
            direct = [r for r in rows if r["battery"] == "direct"]
            out[mode] = {
                "combination_acc1": round(sum(r["correct"] for r in combo)
                                          / max(1, len(combo)), 4),
                "combination_mean_gold_margin": round(
                    sum(r["gold_margin"] for r in combo) / max(1, len(combo)), 6),
                "direct_acc1": round(sum(r["correct"] for r in direct)
                                     / max(1, len(direct)), 4),
                "n_combo": len(combo), "n_direct": len(direct)}
        out["selection_metric"] = "plastic.combination_acc1 (section 7 instrument)"
        out["tie_break_1"] = "plastic.combination_mean_gold_margin"
        out["tie_break_2"] = "smallest eta_e"
        return out

    # ---- in-harness verification (step 4) ------------------------------------ #

    def assert_mirror_fidelity(self) -> Dict:
        """L1: the borrowed mirror residual vs the REAL connection forward at the
        same state (identical float32 formulas -> 1e-5 relative). L2: the cached
        scoring path vs the REAL interface.forward pathway on the same ids (bound
        set by the pathway's residual cast, as in the step-2 fixture). Bitwise
        u_fast restore verified."""
        rec: Dict = {"l1_conn_level": {}, "l2_interface_pathway": {}}
        u_state = [c.plasticity.u_fast.detach().clone() for c in self.lane.conns]

        # ---- L1 ------------------------------------------------------------- #
        max_rel = 0.0
        all_restored = True
        checked = 0
        for exp_index, ctxs in list(self.ctx_train.items())[:FIDELITY_CONTEXTS]:
            ctx = ctxs[0]
            for k in range(min(self.n_conns, 3)):
                psi = self._psi(ctx, k)
                mirror = self._residual_for_conn(u_state[k], self._theta(k), psi,
                                                 self._gamma(k))
                conn = self.lane.conns[k]
                saved = conn.plasticity.u_fast.detach().clone()
                conn.plasticity.u_fast.copy_(u_state[k])     # bitwise state in
                with torch.no_grad():
                    r_real, _ = conn(ctx["pres"][k].unsqueeze(0),
                                     ctx["posts"][k].unsqueeze(0), learn=False)
                conn.plasticity.u_fast.copy_(saved)          # bitwise restore
                ok = bool(torch.equal(conn.plasticity.u_fast, saved))
                all_restored = all_restored and ok
                r_real = self._gamma(k) * r_real[0]
                diff = float((mirror - r_real).abs().max())
                scale = max(float(r_real.abs().max()), float(mirror.abs().max()))
                rel = diff / scale if scale > 0.0 else 0.0
                max_rel = max(max_rel, rel)
                checked += 1
        assert max_rel < MIRROR_FIDELITY_REL, (
            f"L1 mirror fidelity FAILED: max_rel={max_rel:.3e}")
        assert all_restored, "L1 bitwise u_fast restore failed"
        rec["l1_conn_level"] = {"max_rel_diff": max_rel, "comparisons": checked,
                                "tolerance_relative": MIRROR_FIDELITY_REL,
                                "bitwise_restore_all": all_restored, "pass": True}
        log(f"mirror fidelity L1 (mirror vs real conn forward, {checked} probes): "
            f"PASS max_rel={max_rel:.3e}")

        # ---- L2 ------------------------------------------------------------- #
        saved_us = [c.plasticity.u_fast.detach().clone() for c in self.lane.conns]
        for conn, s_k in zip(self.lane.conns, u_state):
            conn.plasticity.u_fast.copy_(s_k)
        pathway_dtype = None
        delta_rel_max = 0.0
        base_det_max = 0.0
        score_nat_max = 0.0
        probes = 0
        for exp_index, ctxs in list(self.ctx_train.items())[:FIDELITY_CONTEXTS]:
            ctx = ctxs[0]
            e = next(x for x in self.manifest["train"]["experiences"]
                     if x["exp_index"] == exp_index)
            prompt = FORMAT_SCAFFOLD + f"Question: {e['question']}\nAnswer:"
            full = self.encode(prompt) + ctx["ids_c"]
            with torch.no_grad():
                res = self.lane.forward(torch.tensor([full], dtype=torch.long))
            pathway_dtype = str(res["logits_delta"].dtype)
            real_base_rows = res["base_logits"][0][
                ctx["prompt_len"] - 1: ctx["prompt_len"] - 1 + len(ctx["ids_c"])].float()
            base_det_max = max(base_det_max, float(
                (real_base_rows - ctx["answer_rows"]).abs().max()))
            mirror_ld = self._context_delta(ctx, u_state) @ self.head_w.T
            real_ld = res["logits_delta"][0].float()
            scale = max(float(real_ld.abs().max()), float(mirror_ld.abs().max()))
            drel = (float((mirror_ld - real_ld).abs().max()) / scale
                    if scale > 0.0 else 0.0)
            delta_rel_max = max(delta_rel_max, drel)
            bp = ctx["prompt_len"] - 1
            lp_real = torch.log_softmax(
                res["logits"][0].float()[bp:bp + len(ctx["ids_c"])], dim=-1)
            lp_mir = torch.log_softmax(
                (ctx["answer_rows"] + mirror_ld.unsqueeze(0)), dim=-1)
            s_real = s_mir = None
            for j, t in enumerate(ctx["ids_c"]):
                a, b = float(lp_real[j, t]), float(lp_mir[j, t])
                s_real = a if s_real is None else s_real + a
                s_mir = b if s_mir is None else s_mir + b
            score_nat_max = max(score_nat_max, abs(s_mir - s_real))
            probes += 1
        for conn, saved in zip(self.lane.conns, saved_us):
            conn.plasticity.u_fast.copy_(saved)
        l2_restored = all(torch.equal(c.plasticity.u_fast, s)
                          for c, s in zip(self.lane.conns, saved_us))
        assert l2_restored, "L2 bitwise u_fast restore failed"
        if pathway_dtype == "torch.float32":
            bound, bnote = MIRROR_FIDELITY_REL, "float32 pathway: 1e-5 applies"
            ok = delta_rel_max < bound
        else:
            bound = MIRROR_L2_BF16_BOUND
            bnote = ("pathway casts the residual to the BF16 logit dtype before the "
                     "head matmul (interface.py:292-300); bound is the "
                     "cast-consistent 2e-2 as in the step-2 fixture; the exact "
                     "float32 mirror math is verified at L1")
            ok = delta_rel_max < bound
        assert ok, (f"L2 fidelity FAILED: rel={delta_rel_max:.3e} vs {bound} "
                    f"(pathway {pathway_dtype})")
        rec["l2_interface_pathway"] = {
            "pathway_residual_dtype": pathway_dtype, "probes": probes,
            "base_rows_forward_determinism_max_abs": base_det_max,
            "logits_delta_max_rel_diff": delta_rel_max,
            "score_max_abs_diff_nats": score_nat_max,
            "assertion_bound_relative": bound, "bound_note": bnote,
            "bitwise_restore_all": l2_restored, "pass": True}
        log(f"mirror fidelity L2 (mirror scoring vs interface.forward, {probes} "
            f"probes): PASS pathway={pathway_dtype} delta_rel={delta_rel_max:.3e} "
            f"score_diff={score_nat_max:.3e} nats")
        return rec

    def assert_chunked_gradient_identity(self) -> Dict:
        """The chunked expression gradient must be IDENTICAL to one whole-graph
        backward of the same loss on the same subset (the step-2 chunking
        identity, verified here for THIS harness's rank chunking)."""
        exp_entry = next(e for e in self.manifest["train"]["experiences"]
                         if e["kind"] == "qa")
        ctxs = self.ctx_train[exp_entry["exp_index"]][:CHUNK_SELFCHECK_SUBSET]
        gold = exp_entry["rank_candidates"].index(exp_entry["gold_candidate"])
        if gold >= len(ctxs):
            ctxs = self.ctx_train[exp_entry["exp_index"]][:gold + 1]
            gold = gold

        def whole():
            leaves = [c.plasticity.u_fast.detach().clone().requires_grad_(True)
                      for c in self.lane.conns]
            scores = torch.stack([self._score_context(c, leaves) for c in ctxs])
            loss = torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.tensor([gold]))
            loss.backward()
            g = [t.grad.detach().clone() for t in leaves]
            return g, float(loss.detach())

        def chunked():
            leaves = [c.plasticity.u_fast.detach().clone().requires_grad_(True)
                      for c in self.lane.conns]
            with torch.no_grad():
                scores = torch.stack([self._score_context(c, leaves)
                                      for c in ctxs])
            p = torch.softmax(scores, dim=-1)
            acc = [torch.zeros_like(u) for u in leaves]
            for k_c, ctx in enumerate(ctxs):
                up = float(p[k_c].item()) - (1.0 if k_c == gold else 0.0)
                if up == 0.0:
                    continue
                s = self._score_context(ctx, leaves)
                (up * s).backward()
                for kk in range(self.n_conns):
                    acc[kk] = acc[kk] + leaves[kk].grad.detach()
                    leaves[kk].grad = None
            return acc

        g_whole, loss_v = whole()
        g_chunk = chunked()
        diffs = [float((a - b).abs().max()) for a, b in zip(g_whole, g_chunk)]
        scale = max(float(a.abs().max()) for a in g_whole)
        rel = (max(diffs) / scale) if scale > 0.0 else 0.0
        assert rel < MIRROR_FIDELITY_REL, (
            f"chunked-gradient self-check FAILED: rel={rel:.3e}")
        out = {"max_abs_diff": max(diffs), "max_rel_diff": rel,
               "tolerance_relative": MIRROR_FIDELITY_REL,
               "subset_candidates": len(ctxs), "loss_value": loss_v, "pass": True,
               "note": "chunked rank gradient verified IDENTICAL to the whole-graph "
                       "backward on the same candidate subset"}
        log(f"chunked-gradient self-check: PASS rel={rel:.3e} "
            f"({len(ctxs)} candidates)")
        return out

    # ---- host immutability (section 3.1) ------------------------------------- #

    def immutability_check(self, label: str) -> Dict:
        fp_ok, fp_diffs = fingerprints_equal(self.fingerprint_at_load,
                                             foundation_fingerprint(self.model))
        host_grad_count = sum(1 for p in self.model.parameters()
                              if p.grad is not None)
        any_base_requires_grad = any(p.requires_grad for p in self.model.parameters())
        ok = bool(fp_ok and host_grad_count == 0 and not any_base_requires_grad)
        return {"point": label, "fingerprints_equal": bool(fp_ok),
                "differing_tensors": fp_diffs[:5],
                "n_tensors_checked": len(self.fingerprint_at_load),
                "host_grad_count": host_grad_count,
                "any_base_param_requires_grad": bool(any_base_requires_grad),
                "optimizer_over_foundation": False,
                "optimizer_note": ("no torch.optim object exists anywhere in this "
                                   "harness; expression autograd runs on detached "
                                   "u_fast leaves only"),
                "pass": ok}

    # ---- exam phase (step 6 machinery; scored exactly once) ------------------ #

    def score_exam_once(self, marker_dir: str, force_void: bool = False) -> Dict:
        marker = Path(marker_dir) / "exam_scored.marker"
        if marker.exists():
            if not force_void:
                raise AssertionError(
                    f"INSTRUMENT FAULT: the final exam was already scored "
                    f"({marker}) — protocol section 10: the final exam is scored "
                    f"exactly once per arm; any exam-adjacent adjustment voids "
                    f"the run. Remediation requires the owner's explicit "
                    f"--score-exam-force-void, which records the void.")
            prev = marker.read_text(encoding="utf-8")
            log(f"EXAM VOID recorded (owner --score-exam-force-void); previous "
                f"marker: {prev.strip()[:120]}")
        exam_rows = {m: self.score_split("exam", m) for m in ("plastic", "full")}
        self.exam_scored = True          # freeze: exam carriers unlock BELOW this
        payload = json.dumps({"scored_at": datetime.now().isoformat()},
                             default=str)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(payload, encoding="utf-8")
        out = {"exam_scored_once": True, "marker": str(marker),
               "rows": exam_rows,
               "summary": {m: {
                   "direct_acc1": round(sum(r["correct"] for r in exam_rows[m]
                                            if r["battery"] == "direct")
                                        / max(1, sum(1 for r in exam_rows[m]
                                                     if r["battery"] == "direct")), 4),
                   "paraphrase_acc1": round(sum(r["correct"] for r in exam_rows[m]
                                                if r["battery"] == "paraphrase")
                                           / max(1, sum(1 for r in exam_rows[m]
                                                        if r["battery"] == "paraphrase")), 4),
                   "combination_acc1": round(sum(r["correct"] for r in exam_rows[m]
                                                 if r["battery"] == "combination")
                                            / max(1, sum(1 for r in exam_rows[m]
                                                         if r["battery"] == "combination")), 4),
                   "combination_mean_gold_margin": round(
                       sum(r["gold_margin"] for r in exam_rows[m]
                           if r["battery"] == "combination")
                       / max(1, sum(1 for r in exam_rows[m]
                                    if r["battery"] == "combination")), 6)}
                   for m in exam_rows}}
        return out

    def exam_diagnostics(self) -> Dict:
        """Post-freeze, post-scoring ONLY (asserted by _carrier_ctx): section 2.3
        exam features (L_disc,exam + novelty) and the section 6.1 swap test. No
        update ever follows these numbers."""
        assert getattr(self, "exam_scored", False), (
            "exam diagnostics require the post-scoring frozen state")
        u_list = [c.plasticity.u_fast.detach() for c in self.lane.conns]
        with torch.no_grad():
            L_disc_exam = float(self.disc_loss(u_list, which="exam"))
            nov_exam = self.novelty(u_list, which="exam")
        diag = {"L_disc_exam": L_disc_exam,
                "L_disc_exam_is_zero": bool(L_disc_exam == 0.0),
                "novelty_exam": nov_exam,
                "novelty_ge_floor": bool(nov_exam >= NOVELTY_FLOOR),
                "criterion": "L_disc,exam = 0 (preregistered); novelty >= 0.80 is "
                             "its exact mean-form consequence",
                "note": "extraction ONLY from the frozen end-state, after "
                        "behavioural scoring; no update follows (section 2.3)"}
        if self.one_to_one:
            diag["swap_test"] = self.swap_test(u_list)
        else:
            diag["swap_test"] = {"skipped": True,
                                 "reason": "no 1:1 fact->connection mapping "
                                           "(tiny stand-in)"}
        return diag

    def swap_test(self, u_list) -> Dict:
        """Section 6.1, EXAM carriers: for each ordered fact pair (i, j) sharing a
        relation template: delta_swap = [s_fj(a_j) - s_fj(a_i)]
                                     - [s_fi(a_j) - s_fi(a_i)],
        s_fj = fact i's per-fact residual block (its connection's u_fast) REPLACED
        by fact j's; all other state identical (masked substitution, no retraining).
        Criterion: mean over ordered pairs >= +0.10 log-prob."""
        assert self.one_to_one
        facts = sorted({q["fact_index"] for q in
                        self.manifest["final_exam"]["questions"]
                        if q["battery"] == "direct"})
        relation_of = {}
        for q in self.manifest["final_exam"]["questions"]:
            if q["battery"] == "direct":
                relation_of[q["fact_index"]] = q["meta"]["relation"]
        deltas = []
        for i in facts:
            for j in facts:
                if i == j or relation_of[i] != relation_of[j]:
                    continue
                a_i, a_j = self._exam_gold(i), self._exam_gold(j)
                qid_i = self.manifest["final_exam"]["carrier_per_fact"][str(i)]
                ci = self.manifest_candidate_index("exam", qid_i, a_i)
                cj = self.manifest_candidate_index("exam", qid_i, a_j)
                ctx_gold, ctx_alt = self.ctx_exam[qid_i][ci], self.ctx_exam[qid_i][cj]
                with torch.no_grad():
                    s_fi_ai = float(self._score_context(ctx_gold, u_list))
                    s_fi_aj = float(self._score_context(ctx_alt, u_list))
                    swapped = list(u_list)
                    swapped[self.conn_of_fact[i]] = \
                        u_list[self.conn_of_fact[j]]
                    s_fj_ai = float(self._score_context(ctx_gold, swapped))
                    s_fj_aj = float(self._score_context(ctx_alt, swapped))
                deltas.append({"i": i, "j": j,
                               "delta_swap": round((s_fj_aj - s_fj_ai)
                                                   - (s_fi_aj - s_fi_ai), 6)})
        mean_d = round(sum(d["delta_swap"] for d in deltas) / max(1, len(deltas)), 6)
        return {"n_ordered_pairs": len(deltas), "mean_delta_swap": mean_d,
                "criterion": "mean >= +0.10 log-prob",
                "pass": bool(mean_d >= 0.10),
                "per_pair": deltas}

    # ---- misc ---------------------------------------------------------------- #

    def state_checksums(self) -> Dict:
        return {f"u_conn{k}": sha256_tensor(c.plasticity.u_fast)
                for k, c in enumerate(self.lane.conns)}


# --------------------------------------------------------------------------- #
# Gate pre-flight (section 6.2 — already demonstrated; referenced or re-run)
# --------------------------------------------------------------------------- #

def gate_reference(runs_dir: str = "runs/lte_gates", rerun: bool = False) -> Dict:
    if not rerun:
        cands = sorted(Path(runs_dir).glob("gates_real_*.json"))
        if not cands:
            raise RuntimeError(
                f"no step-2 real-host gate evidence found in {runs_dir}/ — pass "
                f"--rerun-gates to demonstrate the gates in-session")
        path = cands[-1]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        verdict = evidence.get("verdict")
        g1 = bool(evidence.get("gate1", {}).get("pass"))
        g2 = bool(evidence.get("gate2", {}).get("pass"))
        if not (verdict == "PASS" and g1 and g2):
            raise AssertionError(
                f"gate evidence {path} is not a PASS record "
                f"(verdict={verdict}, gate1={g1}, gate2={g2}) — training may not "
                f"proceed (protocol sections 6.2/10)")
        out = {"mode": "recorded_step2_evidence", "path": str(path),
               "sha256": sha256_text(path.read_bytes().decode("utf-8", "replace")),
               "verdict": verdict, "gate1_pass": g1, "gate2_pass": g2,
               "protocol": evidence.get("protocol", "")[:160]}
        log(f"gate pre-flight: recorded step-2 evidence VERIFIED "
            f"({path}, verdict PASS, gate1 PASS, gate2 PASS)")
        return out
    log("gate pre-flight: RE-RUNNING the imported lte_gates_real gates on a fresh "
        "RealHostFixture (heavy — this is the step-2 procedure)")
    fx = RealHostFixture()
    fx.build_gate_state()
    fx.assert_mirror_fidelity()
    fx.free_host()
    g1 = run_gate1(fx)
    g2 = run_gate2(fx)
    ok = bool(g1["pass"] and g2["pass"])
    out = {"mode": "rerun_in_session", "gate1_pass": bool(g1["pass"]),
           "gate2_pass": bool(g2["pass"]), "verdict": "PASS" if ok else "FAIL",
           "gate1": g1, "gate2": g2}
    log(f"gate pre-flight rerun: gate1={g1['pass']} gate2={g2['pass']}")
    assert ok, "in-session gate rerun FAILED — training may not proceed"
    return out


# --------------------------------------------------------------------------- #
# One training attempt (fresh lane per attempt — section 5 retry reset rule)
# --------------------------------------------------------------------------- #

def run_attempt(host: LteTrainHost, cfg: Dict, args) -> Dict:
    lane_rec = host._attach_lane()
    t0 = time.time()
    teach = host.teach(cfg)
    imm_teach = host.immutability_check("teach-end")
    assert imm_teach["pass"], f"host immutability FAILED at teach-end: {imm_teach}"
    dev = host.dev_summary()
    with torch.no_grad():
        u_list = [c.plasticity.u_fast.detach() for c in host.lane.conns]
        nov_train = host.novelty(u_list, which="train")
    rec = {"config": dict(cfg), "lane": lane_rec, "teach": teach,
           "dev": dev, "novelty_train_carriers": nov_train,
           "imm_teach_end": imm_teach,
           "u_checksums_end": host.state_checksums(),
           "attempt_s": round(time.time() - t0, 1)}
    stable = not teach["instability"]["unstable"]
    rec["stable"] = stable
    return rec


# --------------------------------------------------------------------------- #
# Calibration — the ONE deterministic nested sequence (protocol section 5)
# --------------------------------------------------------------------------- #

def run_calibration(host: LteTrainHost, args) -> Dict:
    t0 = time.time()
    log("CALIBRATION (section 5): configs A(lambda_d=0.5)/B(lambda_d=0.0) x "
        f"eta_e {ETA_E_GRID} at lambda_r 1.0, ladder {LAMBDA_R_LADDER} on "
        "instability; every attempt from the same frozen initial state + "
        "identical RNG (fresh seeded lane); selection on dev combination@1")
    grid: Dict = {}
    selected: Dict = {}
    for cfg_name, lambda_d in LAMBDA_D_CONFIGS:
        cfg_grid = {}
        for eta_e in ETA_E_GRID:
            attempts = []
            point = {"eta_e": eta_e, "attempts": attempts, "aborted": False}
            for lambda_r in LAMBDA_R_LADDER:
                cfg = {"eta_e": eta_e, "lambda_r": lambda_r, "lambda_d": lambda_d,
                       "gate_lr": args.gate_lr, "gate_steps": args.gate_steps,
                       "mode": "calibrate"}
                log(f"  [{cfg_name}] eta_e={eta_e:.0e} lambda_r={lambda_r} "
                    f"lambda_d={lambda_d} — attempt {len(attempts) + 1}")
                att = run_attempt(host, cfg, args)
                att["lambda_r"] = lambda_r
                attempts.append(att)
                log(f"    stable={att['stable']} "
                    f"dev.combo@1={att['dev']['plastic']['combination_acc1']} "
                    f"({att['attempt_s']}s)")
                if att["stable"]:
                    break
                if lambda_r == LAMBDA_R_LADDER[-1]:
                    point["aborted"] = True       # section 5b: recorded, continue
                    log(f"    UNSTABLE through the whole ladder -> grid point "
                        f"ABORTED (recorded, not tuned further)")
            survived = [a for a in attempts if a["stable"]]
            point["survived_lambda_r"] = (survived[-1]["lambda_r"]
                                          if survived else None)
            point["dev"] = survived[-1]["dev"] if survived else None
            point["novelty_train_carriers"] = (survived[-1]["novelty_train_carriers"]
                                               if survived else None)
            cfg_grid[str(eta_e)] = point
        # selection (section 5 step 2): highest dev combo@1; tie 1 gold margin;
        # tie 2 smallest eta_e. All aborted -> configuration VOID (instrument fault)
        live = [(float(e), pt) for e, pt in cfg_grid.items() if not pt["aborted"]]
        if not live:
            selected[cfg_name] = {"void": True,
                                  "reason": "all five eta_e points aborted "
                                            "(section 5: instrument fault)"}
            grid[cfg_name] = cfg_grid
            continue
        best = sorted(live, key=lambda kv: (
            -kv[1]["dev"]["plastic"]["combination_acc1"],
            -kv[1]["dev"]["plastic"]["combination_mean_gold_margin"],
            kv[0]))[0]
        selected[cfg_name] = {
            "void": False, "eta_e": best[0],
            "lambda_r": best[1]["survived_lambda_r"],
            "dev": best[1]["dev"], "novelty_train_carriers":
                best[1]["novelty_train_carriers"]}
        grid[cfg_name] = cfg_grid

    # lambda_d decision (section 5, after both configurations)
    a, b = selected.get("A", {}), selected.get("B", {})
    if a.get("void") or b.get("void"):
        decision = {"lambda_d": None,
                    "void": True,
                    "reason": "a configuration is void — the lambda_d decision "
                              "cannot be taken (instrument fault)"}
    else:
        nov_a = a["novelty_train_carriers"]
        combo_a = a["dev"]["plastic"]["combination_acc1"]
        combo_b = b["dev"]["plastic"]["combination_acc1"]
        retain = bool(nov_a >= NOVELTY_FLOOR
                      and combo_a >= combo_b - COMBO_TIE_TOL)
        decision = {"lambda_d": 0.5 if retain else 0.0,
                    "novelty_A": nov_a, "dev_combo_A": combo_a,
                    "dev_combo_B": combo_b,
                    "rule": f"retain 0.5 iff novelty_A >= {NOVELTY_FLOOR} AND "
                            f"combo(A) >= combo(B) - {COMBO_TIE_TOL}; ties -> 0.5",
                    "tie_note": ("exact tie resolves to 0.5 (the retain condition "
                                 "is already satisfied by an exact tie)")}
    out = {"grid": grid, "selected": selected, "lambda_d_decision": decision,
           "final_freeze": ("the winning configuration (lambda_r, lambda_d, eta_e) "
                            "is committed here; the primary LTE-1 run trains with "
                            "exactly it; the final exam is scored exactly once "
                            "per arm (section 5)"),
           "calibration_s": round(time.time() - t0, 1)}
    return out


# --------------------------------------------------------------------------- #
# Report + CLI
# --------------------------------------------------------------------------- #

def build_report(args, host: LteTrainHost, mode: str, extra: Dict) -> Dict:
    imm_end = host.immutability_check("run-end")
    report = {
        "step": "LTE-1 step 4 — training harness (consumed by steps 5-6)",
        "protocol": "docs/LTE_PROTOCOL.md Revision 8 (frozen Revision 6 + Rev 7/8)",
        "mode": mode,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gate_reference": extra["gates"],
        "manifest": {"path": args.manifest,
                     "sha256": sha256_text(Path(args.manifest).read_text("utf-8")),
                     "generation_seed": host.manifest["generation_seed"],
                     "counts": host.manifest["counts"],
                     "disjointness_all_pass":
                         host.manifest["disjointness_assertions"]["all_pass"],
                     "recheck": extra["recheck"],
                     "note": "committed to the repository BEFORE any training run; "
                             "the freeze record cites this path + commit SHA"},
        "config": {"eta_e": args.eta_e, "lambda_r": args.lambda_r,
                   "lambda_d": args.lambda_d, "tau_d": TAU_D,
                   "eta_p": args.learning_rate, "gate_lr": args.gate_lr,
                   "gate_steps": args.gate_steps, "seed": args.seed,
                   "foundation": host.foundation_label,
                   "n_facts": host.n_facts, "n_conns": host.n_conns,
                   "one_to_one_fact_conn": host.one_to_one,
                   "conn_of_fact": host.conn_of_fact,
                   "layer_pairs": [list(p) for p in host.layer_pairs],
                   "pair_selection": host.pair_note},
        "sequencing_decision": (
            "pass A plasticity (delta_u_pred, nnfs_update verbatim) -> pass B gamma "
            "(the untouched gamma-sequencing law [D2/D3]) -> pass C expression "
            "(delta_u_expr per QA experience; the section 5 full pass). The "
            "expression channel is ADDITIVE (u_fast <- u_fast + delta_u_pred + "
            "delta_u_expr); the 0.7 blend enters the gradient only through the "
            "effective connection's Jacobian (never re-applied)"),
        "consolidation": {"disabled": True,
                          "mechanism": "plasticity.consolidation_rate = 0.0 "
                                       "(rate<=0 hard-disable, plastic.py:403)"},
        "host_immutability": {
            "fingerprint_at_load": len(host.fingerprint_at_load),
            "points": [extra["imm_load"], imm_end],
            "per_attempt_teach_end_checks": "in results (every attempt re-verifies "
                                            "the fingerprints at teach-end, "
                                            "section 3.1(3))",
            "section_3_1_four_point": {
                "no_optimizer_over_foundation": True,
                "fingerprints_identical": bool(extra["imm_load"]["pass"]
                                               and imm_end["pass"]),
                "finite_gradients": "isfinite(delta_u_expr).all() asserted every "
                                    "expression step; zero-gradient steps recorded "
                                    "not aborted",
                "host_grad_count_zero": "verified at every immutability point"},
        },
        "mirror_verification": {"fidelity": host.fidelity,
                                "chunked_gradient_selfcheck": host.chunk_selfcheck},
        "cache": host.cache_stats(),
        "results": extra["results"],
        "timings_s": {"load": host.load_s, "caches": host.caches_s,
                      "total": round(time.time() - host.t0, 1)},
        "peak_rss_gb": rss_gb(),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
        "not_implemented_here": [
            "controls A/B/C re-measurement (protocol section 4) — step 5 of the "
            "owner's order; this harness provides the scoring instrument they use",
            "retention battery (section 7 state criteria) — retention.py owns it; "
            "not wired into this harness",
            "the real-model run itself — the model lives on Lappy; local "
            "validation is the tiny-lane structural smoke + the manifest run",
            "protocol section 3's proposed '--lte mode inside teacher_student.py' "
            "shape — the owner's step-4 order made this a separate harness "
            "(aion/experiments/lte_train.py); Phase-1/2/3 paths untouched"],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="LTE-1 training harness (protocol "
                                             "Revision 8; steps 5-6 run this)")
    ap.add_argument("--dev", action="store_true",
                    help="calibration mode: the section 5 dev trainings (shorter "
                         "budgets — no exam, no exam diagnostics; the section 5 "
                         "deterministic nested grid)")
    ap.add_argument("--mode", default=None, choices=("calibrate", "train"),
                    help="overrides --dev")
    ap.add_argument("--eta-e", type=float, default=1.0e-3,
                    help="expression learning rate (train mode: the frozen "
                         "calibration value)")
    ap.add_argument("--lambda-r", type=float, default=DEFAULT_LAMBDA_R)
    ap.add_argument("--lambda-d", type=float, default=0.5)
    ap.add_argument("--manifest", default="runs/lte/manifest.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--foundation", default="qwen", choices=("tiny", "qwen"),
                    help="tiny = in-memory stand-in (Rog CPU debug); qwen = the "
                         "real frozen Qwen3.5-4B-Base (Lappy)")
    ap.add_argument("--local-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                    help="eta_p — the prediction channel's frozen plasticity lr")
    ap.add_argument("--gate-lr", type=float, default=DEFAULT_GATE_LR,
                    help="[D2] untouched gamma law")
    ap.add_argument("--gate-steps", type=int, default=DEFAULT_GATE_STEPS,
                    help="[D3] untouched gamma law: gate steps per QA experience")
    ap.add_argument("--rerun-gates", action="store_true",
                    help="re-run the imported lte_gates_real gates in-session "
                         "(heavy) instead of verifying the recorded step-2 PASS")
    ap.add_argument("--score-exam", action="store_true",
                    help="score the final exam (train mode only; EXACTLY ONCE — "
                         "marker-guarded per protocol section 10)")
    ap.add_argument("--score-exam-force-void", action="store_true",
                    help="record an exam VOID and re-score (owner action only)")
    ap.add_argument("--marker-dir", default="runs/lte")
    args = ap.parse_args()

    mode = args.mode or ("calibrate" if args.dev else "train")
    if args.score_exam and mode != "train":
        ap.error("--score-exam requires train mode (the exam follows the frozen "
                 "configuration; calibration reads DEV only)")
    if mode == "calibrate" and (args.eta_e, args.lambda_r, args.lambda_d) != (
            1.0e-3, DEFAULT_LAMBDA_R, 0.5):
        log("note: --eta-e/--lambda-r/--lambda-d are ignored in calibration mode "
            "(the section 5 grid fixes lambda_r=1.0 and lambda_d per config)")

    t0 = time.time()
    log(f"LTE-1 training harness — mode={mode} seed={args.seed} "
        f"foundation={args.foundation}")
    gates = gate_reference(rerun=args.rerun_gates)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["generation_seed"] != args.seed:
        raise AssertionError(
            f"manifest generation_seed {manifest['generation_seed']} != --seed "
            f"{args.seed} — the split manifest is preregistered; refusing")
    recheck = recheck_manifest(manifest)
    log(f"manifest {args.manifest} re-verified: counts={manifest['counts']} "
        f"disjointness all_pass="
        f"{manifest['disjointness_assertions']['all_pass']} recheck_pass="
        f"{recheck['pass']}")

    host = LteTrainHost(manifest, args)
    if args.foundation == "qwen":
        # the real host MUST give one connection per taught fact (the section 6.1
        # swap test needs a well-defined per-fact residual block)
        assert host.one_to_one, (
            f"real host yielded only {host.n_conns} hookable pairs for "
            f"{host.n_facts} facts — 1:1 fact->connection is required (swap-test "
            f"law); pair selection note: {host.pair_note}")
    log(f"host ready: {host.foundation_label}; facts={host.n_facts} "
        f"conns={host.n_conns} (1:1={host.one_to_one}); caches={host.cache_stats()}"
        f"; load {host.load_s}s + caches {host.caches_s}s")

    imm_load = host.immutability_check("load")
    assert imm_load["pass"], f"host immutability FAILED at load: {imm_load}"

    if mode == "calibrate":
        cal = run_calibration(host, args)
        host._drop_lane()
        results = {"calibration": cal}
        report = build_report(args, host, mode, {
            "gates": gates, "recheck": recheck, "results": results,
            "imm_load": imm_load})
        out = Path(args.out or f"runs/lte/calibration_"
                               f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        cfg = {"eta_e": args.eta_e, "lambda_r": args.lambda_r,
               "lambda_d": args.lambda_d, "gate_lr": args.gate_lr,
               "gate_steps": args.gate_steps, "mode": "train"}
        att = run_attempt(host, cfg, args)
        results = {"attempt": att}
        if att["teach"]["instability"]["unstable"]:
            # the section 5 instability ladder belongs to calibration; a frozen
            # primary configuration arriving unstable means the calibration
            # selection failed its job — NO exam may follow this run; evidence is
            # written and the run is flagged
            host._drop_lane()
            report = build_report(args, host, mode, {
                "gates": gates, "recheck": recheck, "results": results,
                "imm_load": imm_load})
            report["verdict"] = (
                "UNSTABLE PRIMARY RUN — the frozen configuration showed section 5b "
                "instability (see results.attempt.teach.instability); calibration "
                "selection must be re-examined before any exam")
            out = Path(args.out or f"runs/lte/lte_train_"
                                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")
            log(f"TRAIN RUN UNSTABLE (section 5b criteria at the frozen config) — "
                f"evidence {out}; NO exam may follow this run")
            return 2
        if args.score_exam or args.score_exam_force_void:
            if not host.ctx_exam:              # --score-exam-force-void alone
                host._cache_exam()
            exam = host.score_exam_once(args.marker_dir,
                                        force_void=args.score_exam_force_void)
            exam["diagnostics"] = host.exam_diagnostics()
            results["exam"] = exam
        host._drop_lane()
        report = build_report(args, host, mode, {
            "gates": gates, "recheck": recheck, "results": results,
            "imm_load": imm_load})
        out = Path(args.out or f"runs/lte/lte_train_"
                               f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"report: {out.resolve()}  ({report['timings_s']['total']}s, "
        f"peak RSS {report['peak_rss_gb']} GiB)")
    if mode == "calibrate":
        d = report["results"]["calibration"]["lambda_d_decision"]
        log(f"calibration complete: lambda_d decision -> {d.get('lambda_d')} "
            f"(void={d.get('void', False)})")
    else:
        log(f"train run complete: dev.combo@1="
            f"{report['results']['attempt']['dev']['plastic']['combination_acc1']}"
            f" novelty(train)="
            f"{report['results']['attempt']['novelty_train_carriers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
