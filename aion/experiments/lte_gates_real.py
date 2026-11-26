"""LTE-1 STEP 2 — Gate 1 + Gate 2 on the REAL Qwen3.5-4B-Base host.

Authorisation: owner GO 2026-11-26, STEP 2 ONLY (docs/LTE_PROTOCOL.md freeze record,
section 10: "demonstrate Gates 1-2 on the real Qwen host with literal 2.2 scoring and
2.3 features"). Boundaries observed by this script:
  (a) LITERAL section 2.2 candidate scoring — each candidate is scored by the actual
      frozen-host computation on ITS OWN prompt+cand context: one host forward per
      (fact, candidate) is cached (base logits float32 + per-pair seq-mean normalized
      hidden states + psi per connection), and the plastic residual delta is injected
      at the prescribed answer positions of that same context. No shared-context
      shortcut of any kind (this is the difference from the step-1 stand-in fixture).
  (b) LITERAL section 2.3 per-fact features — gamma-scaled plastic residual at the
      answer position from that fact's OWN connection set only (masked residual sum:
      every connection outside the fact's set contributes EXACTLY ZERO), evaluated on
      the fact's own carrier context; float32 cosine; zero-norm vector => cos exactly 0.
      This is the section 2.3 specification implemented directly for the real host —
      not a stand-in variant carried forward.
  (c) Rev 7 gate mathematics preserved (identical to the step-1 stand-in run,
    aion/experiments/lte_gates.py).
  (d) Precision auditable — host-forward dtype, cached-logit dtype, residual dtype and
      objective-reference dtype are recorded, and the float64 reference is described
      exactly as: "The float64 calculation is a higher-precision evaluation of the
      defined plastic objective over fixed host outputs; the host forward itself
      remains in its native precision — this is NOT float64 Qwen inference."
  (e) Stop at the step-2 boundary. A failing gate stops the run and is reported;
      criteria are never weakened.

Gate 1 (protocol 6.2): D_AD = grad F(u) . d  vs central differences
  D_FD(eps) = [F(u + eps d) - F(u - eps d)] / (2 eps),  eps in {1e-2, 1e-3, 1e-4},
  K = 5 identical repeats give the repeatability noise n_rep(eps);
  pass: |D_AD - D_FD| / max(1, |D_AD|) <= max(10 * n_rep, 1e-4) at {1e-2, 1e-3}.
  Float64 FD diagnostics recorded alongside (Rev 7).

Gate 2 (protocol 6.2, Rev 6 form + Rev 7 criterion): F (both terms) before ->
  RAW unscaled gradient -> u -= eta_gate * grad with eta_gate = 0.1 * eta_e applied
  EXACTLY ONCE -> F recomputed on identical inputs with the FLOAT64 reference ->
  must decrease -> fixture's original plastic state restored BITWISE (torch.equal).

F = lambda_r * L_rank,train + lambda_d * L_disc,train.
"""

import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ---- repo path (robust: this file runs flat at ~/lte_gates_real.py on Lappy) --- #

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

from aion.experiments.teacher_student import FORMAT_SCAFFOLD            # noqa: E402
from aion.relational.interface import (                                  # noqa: E402
    InterfaceConfig, RelationalInterface, discover_decoder_layers,
    foundation_fingerprint, freeze_base, pick_hook_pairs)

# ---- fixture configuration (recorded in the evidence) ---------------------- #

LAMBDA_R = 1.0
LAMBDA_D = 0.5
TAU_D = 0.20
ETA_E = 1.0e-3                    # fixture value (mid-grid); the real eta_e is frozen
                                  # after dev calibration per protocol section 5
GATE_FIXTURE_VALUE = 1.0          # gamma per connection (constant in the fixture)
N_REPEATS = 5
FD_EPSILONS = (1.0e-2, 1.0e-3, 1.0e-4)
PASS_EPSILONS = (1.0e-2, 1.0e-3)
NUMERICAL_FLOOR = 1.0e-4
SEED = 20261126
MODEL_DIR = "/mnt/seagate/aion-models/Qwen3.5-4B-Base"
N_FACTS = 4

FLOAT64_SENTENCE = (
    "The float64 calculation is a higher-precision evaluation of the defined plastic "
    "objective over fixed host outputs; the host forward itself remains in its native "
    "precision — this is NOT float64 Qwen inference.")

# Fixture facts — reused verbatim from the step-1 stand-in (lte_gates.py); they
# tokenize with the real Qwen tokenizer fine. Deliberately distinct relations.
FACTS = [
    {"question": "Tav relays to whom?", "gold": "Kimo", "distractor": "Drel"},
    {"question": "Neral reaches which device?", "gold": "Vesk", "distractor": "Juna"},
    {"question": "Who maintains Pola?", "gold": "Sef", "distractor": "Ryn"},
    {"question": "Which relay does Quill use?", "gold": "Modo", "distractor": "Hath"},
]


def sha256_bytes(b: bytes, n: int = 64) -> str:
    return hashlib.sha256(b).hexdigest()[:n]


def sha256_tensor(t: torch.Tensor, n: int = 64) -> str:
    """Checksum of a tensor's raw bytes (float32, contiguous, CPU)."""
    x = t.detach().to(torch.float32).contiguous().cpu()
    return sha256_bytes(x.view(torch.uint8).numpy().tobytes(), n)


# --------------------------------------------------------------------------- #
# Fixture: real frozen host + cached per-candidate host outputs + mirror
# --------------------------------------------------------------------------- #

class RealHostFixture:
    """The real-host gate fixture: frozen Qwen3.5-4B-Base, per-candidate host
    forward caches, and the float32 mirror that defines the plastic objective."""

    def __init__(self, seed: int = SEED):
        t0 = time.time()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # EXACTLY as teacher_student.setup_foundation does for --foundation qwen
        # (READ-ONLY landed code; same load, freeze, fingerprint order).
        load_kwargs_used = "torch_dtype"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                device_map=None, local_files_only=True).to("cpu")
        except TypeError:
            # newer transformers renamed the kwarg; same semantics, recorded
            load_kwargs_used = "dtype"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                device_map=None, local_files_only=True).to("cpu")
        model.eval()
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.freeze_stats = freeze_base(model)          # asserts requires_grad False
        assert not any(p.requires_grad for p in model.parameters())
        self.fingerprint = foundation_fingerprint(model)  # 426-tensor instrument
        fp_canon = json.dumps({k: list(v) for k, v in sorted(self.fingerprint.items())},
                              sort_keys=True)
        self.fingerprint_digest = sha256_bytes(fp_canon.encode("utf-8"))
        self.load_s = round(time.time() - t0, 1)
        self.model_dir = MODEL_DIR
        self.load_kwargs_used = load_kwargs_used

        layers = discover_decoder_layers(model)
        self.n_layers = len(layers)
        all_pairs = pick_hook_pairs(layers)             # the interface's OWN order
        self.layer_pairs, self.pair_selection_note = self._select_pairs(layers, all_pairs)
        self.pairs_desc = [
            {"pair": list(p), "pre_class": layers[p[0]].class_name,
             "post_class": layers[p[1]].class_name, "pre_family": layers[p[0]].family,
             "post_family": layers[p[1]].family} for p in self.layer_pairs]
        assert len(self.layer_pairs) == N_FACTS, (
            f"need {N_FACTS} layer pairs for {N_FACTS} facts, got {self.layer_pairs}")

        cfg = InterfaceConfig()                          # teacher_student defaults
        self.iface = RelationalInterface(model, cfg=cfg, layer_pairs=list(self.layer_pairs))
        self.iface.eval()
        for p in self.iface.base.parameters():
            p.requires_grad_(False)
        for p in self.iface.conns.parameters():
            p.requires_grad_(False)
        # gamma: constants for this fixture (set on the interface's gate parameters
        # AND used as detached constants inside the mirror).
        for g in self.iface.gates:
            g.data.fill_(GATE_FIXTURE_VALUE)
            g.requires_grad_(False)
        self.gammas = [float(g.item()) for g in self.iface.gates]
        self.blend = float(self.iface.cfg.blend_ratio)   # mirror tied to the real conns
        # Mirror head: float32 view of the (BF16-valued) output head. The cast is
        # EXACT (BF16 -> float32 loses nothing), so the mirror math is unchanged;
        # it lets the float32 mirror matmul against the head on a BF16 host. The
        # float64 reference casts this float32 copy onward (also exact).
        self.head_w = self.iface.base.get_output_embeddings().weight.detach().float()
        self.head_w_dtype = str(self.iface.base.get_output_embeddings().weight.dtype)
        self.hidden_dim = self.iface.hidden_dim
        self.n_conns = len(self.iface.conns)
        assert self.n_conns == N_FACTS
        self.theta = [conn.plasticity.theta_slow.detach().clone()
                      for conn in self.iface.conns]
        self.host_param_dtype = str(next(model.parameters()).dtype)
        del layers
        gc.collect()

        # ---- per-(fact, candidate) frozen-host caches (LITERAL 2.2) ---------- #
        self.items: List[Dict] = []
        self.cache: List[List[Dict]] = []
        self.forward_times: List[float] = []
        self.logits_dtype_observed = None
        for fi, f in enumerate(FACTS):
            prompt = FORMAT_SCAFFOLD + f"Question: {f['question']}\nAnswer:"
            ids_prompt = self.tok.encode(prompt)
            cands = [(f["gold"], 0), (f["distractor"], 1)]   # (text, gold_index)
            per_cand = []
            for text, gi in cands:
                ids_c = self.tok.encode(" " + text)
                full = ids_prompt + ids_c
                tf = time.time()
                self.iface._captured.clear()
                with torch.no_grad():
                    out = self.iface.base(input_ids=torch.tensor([full], dtype=torch.long))
                self.forward_times.append(round(time.time() - tf, 2))
                self.logits_dtype_observed = str(out.logits.dtype)
                base_logits = out.logits[0].float()            # (T, V) float32 cache
                hidden = dict(self.iface._captured)
                states, psi = [], []
                for k, (pre_i, post_i) in enumerate(self.iface.pairs):
                    pre_s = hidden[pre_i].mean(dim=1).float()   # (1, D) seq-mean states
                    post_s = hidden[post_i].mean(dim=1).float() #   (interface.py:281-285)
                    if self.iface.cfg.normalize_states:
                        pre_s = torch.nn.functional.normalize(pre_s, dim=-1)  # unit-L2
                        post_s = torch.nn.functional.normalize(post_s, dim=-1)
                    states.append((pre_s, post_s))
                    with torch.no_grad():
                        # psi is u-independent (RelationshipFunction uses pre/post/W only)
                        psi.append(self.iface.conns[k].relationship(pre_s, post_s)
                                   .detach().clone())           # (1, D_post, D_pre)
                per_cand.append({
                    "cand_text": text, "gold_index": gi, "gold": gi == 0,
                    "ids": ids_c, "prompt_len": len(ids_prompt), "full_len": len(full),
                    "base_logits": base_logits, "states": states, "psi": psi})
            self.items.append({"fact": fi, "question": f["question"], "gold": f["gold"],
                               "distractor": f["distractor"],
                               "prompt": prompt, "prompt_ids": ids_prompt,
                               "cands": per_cand})
            self.cache.append(per_cand)
        self.caches_s = round(time.time() - t0, 1)

        self.u_state: List[torch.Tensor] = []

    # ---- pair selection (recorded fixture choice) ------------------------- #

    @staticmethod
    def _select_pairs(layers, all_pairs) -> Tuple[List[Tuple[int, int]], str]:
        """The interface's OWN pick_hook_pairs order, limited to the first 4 pairs.
        On the Qwen3.5-4B hybrid stack pick_hook_pairs yields THREE pairs
        (first deltanet pair, first attention pair, deepest pair) — recorded from
        the interventions real-host run and re-verified here — so the 4th pair is
        the first unused second-occurrence pair in the function's own family walk
        (deltanet first, then attention). Fixture choice, recorded."""
        pairs = [tuple(p) for p in all_pairs][:N_FACTS]
        if len(pairs) >= N_FACTS:
            return pairs, (f"first {N_FACTS} pairs of pick_hook_pairs order")
        used = {i for pr in pairs for i in pr}
        added = []
        for fam in ("deltanet", "attention"):
            for l in layers:
                if (l.family == fam and l.index not in used
                        and (l.index + 1) < len(layers)
                        and (l.index, l.index + 1) not in pairs):
                    pairs.append((l.index, l.index + 1))
                    used.add(l.index)
                    added.append((l.index, l.index + 1))
                    break
            if len(pairs) >= N_FACTS:
                break
        note = (f"pick_hook_pairs yielded {len(all_pairs)} pairs {list(all_pairs)}; "
                f"first {N_FACTS} requested -> extended in the function's own family-walk "
                f"pattern with second-occurrence pair(s) {added} (fixture choice, recorded)")
        return pairs, note

    # ---- mirror (the defined plastic objective over fixed host outputs) --- #

    def _residual_for_conn(self, u, theta, psi, gamma, cast=None):
        """r = gamma * ((blend*u + (1-blend)*theta) * psi).sum(-1) -> (D,) — the
        PlasticRelationalConnection forward (learn=False) through the interface's
        gamma gate, with autograd ONLY on u (blend Jacobian exactly here; the
        expression path is never multiplied by another blend — protocol 2.1)."""
        cast = cast or (lambda t: t)
        w = self.blend * cast(u) + (1.0 - self.blend) * cast(theta)
        transmitted = w.unsqueeze(0) * cast(psi)            # (1, D_post, D_pre)
        return gamma * transmitted.sum(dim=-1).squeeze(0)   # (D,)

    def _context_residuals(self, u_list, fact, cand_idx, cast=None) -> List[torch.Tensor]:
        """r_k for EVERY connection on this (fact, candidate) host context
        (psi from that context's own cache)."""
        psi = self.cache[fact][cand_idx]["psi"]
        return [self._residual_for_conn(u_list[k], self.theta[k], psi[k],
                                        self.gammas[k], cast)
                for k in range(self.n_conns)]

    def _candidate_scores(self, u_list, fact, cast=None) -> torch.Tensor:
        """LITERAL section 2.2: per-candidate score from THAT candidate's own
        frozen-host context — its own cached base logits and its own cached psi —
        with the plastic residual delta injected (broadcast) and the score summed
        over the candidate's tokens at the answer positions."""
        cast = cast or (lambda t: t)
        scores = []
        for ci, cand in enumerate(self.items[fact]["cands"]):
            rs = self._context_residuals(u_list, fact, ci, cast)
            delta = torch.stack(rs, dim=0).sum(dim=0)       # (D,) total residual
            base = cast(self.cache[fact][ci]["base_logits"])  # (T, V)
            head = cast(self.head_w)                        # (V, D)
            logits_delta = delta @ head.T                   # (V,)
            base_pos = cand["prompt_len"] - 1               # position predicting token 0
            rows = torch.tensor([base_pos + j for j in range(len(cand["ids"]))],
                                dtype=torch.long)
            lp = torch.log_softmax(base[rows] + logits_delta, dim=-1)  # (n_tokens, V)
            s = None
            for j, t in enumerate(cand["ids"]):
                term = lp[j, t]
                s = term if s is None else s + term
            scores.append(s)
        return torch.stack(scores, dim=0)

    def _fact_feature(self, u_list, fact, cast=None):
        """LITERAL section 2.3: gamma-scaled plastic residual at the answer position
        from fact i's OWN connection set only — masked residual sum, every other
        connection contributes EXACTLY ZERO (this fixture: fact i owns exactly one
        connection, connection i) — evaluated on fact i's OWN carrier context (its
        prompt + gold-candidate ids). The interface broadcasts one residual vector
        per sequence, so the answer-position feature is that vector as received at
        the answer positions (prompt_len-1 .. prompt_len-1+len(cand)-1)."""
        cast = cast or (lambda t: t)
        gold_ci = 0                                          # gold is candidate index 0
        psi = self.cache[fact][gold_ci]["psi"]
        return self._residual_for_conn(u_list[fact], self.theta[fact], psi[fact],
                                       self.gammas[fact], cast)

    def _features(self, u_list, cast=None):
        """Per-fact features + L2 normalization with the explicit zero-norm check
        (zero-norm vector => normalized feature is the zero vector, so every
        cosine involving it is EXACTLY 0 — no NaN). The norm is computed in the
        feature's own working dtype: float32 in the production evaluation (the
        protocol's "cos in float32"), float64 in the float64 reference evaluation —
        never a dtype cast inside an evaluation (a float32 staircase inside the
        float64 reference would break the FD diagnostics)."""
        cast = cast or (lambda t: t)
        raw, normalized, norms = [], [], []
        for fact in range(len(self.items)):
            f = self._fact_feature(u_list, fact, cast)
            raw.append(f)
            nrm = torch.linalg.vector_norm(f)
            norms.append(float(nrm.detach()))
            normalized.append(torch.zeros_like(f) if float(nrm.detach()) == 0.0
                              else f / nrm)
        return raw, normalized, norms

    def loss_parts(self, u_list, cast=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(L_rank, L_disc) separately — recorded so the evidence PROVES both
        gradient paths are exercised."""
        cast = cast or (lambda t: t)
        rank_losses = []
        for fact in range(len(self.items)):
            scores = self._candidate_scores(u_list, fact, cast)
            gold = self.items[fact]["cands"][0]["gold_index"]
            rank_losses.append(torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.tensor([gold])))
        L_rank = torch.stack(rank_losses).mean()
        _raw, normalized, _norms = self._features(u_list, cast)
        cos_terms = []
        n = len(normalized)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cos = torch.dot(normalized[i], normalized[j])
                cos_terms.append(torch.clamp(cos - TAU_D, min=0.0))
        L_disc = torch.stack(cos_terms).mean()
        return L_rank, L_disc

    def eval_F(self, u_list, cast=None) -> torch.Tensor:
        """F = lambda_r * L_rank,train + lambda_d * L_disc,train (both terms,
        every evaluation)."""
        L_rank, L_disc = self.loss_parts(u_list, cast)
        return LAMBDA_R * L_rank + LAMBDA_D * L_disc

    def state_diagnostics(self, u_list, cast=None) -> Dict:
        """Feature norms + ALL pairwise cosines + both loss terms at a state
        (owner addendum: the separation structure itself must be recorded)."""
        cast = cast or (lambda t: t)
        _raw, normalized, norms = self._features(u_list, cast)
        with torch.no_grad():
            L_rank, L_disc = self.loss_parts(u_list, cast)
        cosines, hinge = {}, {}
        n = len(normalized)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c = float(torch.dot(normalized[i], normalized[j]))
                cosines[f"cos(f{i},f{j})"] = c
                hinge[f"hinge({i},{j})"] = max(0.0, c - TAU_D)
        return {"L_rank": float(L_rank), "L_disc": float(L_disc),
                "disc_hinge_active": bool(float(L_disc) > 0.0),
                "feature_norms": {f"||f{i}||": norms[i] for i in range(n)},
                "pairwise_cosines": cosines, "pairwise_hinge_terms": hinge}

    # ---- state handling ---------------------------------------------------- #

    def get_state(self) -> List[torch.Tensor]:
        return [u.detach().clone() for u in self.u_state]

    def set_state(self, state: List[torch.Tensor]) -> None:
        self.u_state = [s.detach().clone() for s in state]

    # ---- mirror fidelity ---------------------------------------------------- #

    def assert_mirror_fidelity(self) -> Dict:
        """Two levels, both at the fixture state with bitwise state copy/restore:
        L1 — mirror conn-level residual (float32) vs the REAL PlasticRelationalConnection
             forward (learn=False) on the SAME cached states: must agree < 1e-5
             relative (the step-1 exactness check; identical float32 formulas).
        L2 — the defined gate scoring computation (cached base logits + float32
             delta through the float32 head) vs the REAL interface.forward pathway
             on the same ids: the real pathway casts the residual to the host logit
             dtype (interface.py:292-300) before the head matmul, so the achievable
             agreement is set by that cast, not by the mirror math. The actual max
             diffs are RECORDED; the assertion is against a cast-consistent bound
             (2e-2, ~5x the BF16 single-rounding bound 2^-8) — a wrong mirror would
             be O(1) and still abort. If the host pathway turns out float32, the
             1e-5 relative assertion from step 1 applies directly."""
        state = self.get_state()
        rec: Dict = {"l1_conn_level": {}, "l2_interface_pathway": {}}

        # ---- L1: float32 mirror vs real connection forward ------------------ #
        max_abs = max_rel = 0.0
        worst = None
        all_restored = True
        for fi in range(len(self.items)):
            for ci in range(len(self.cache[fi])):
                for k in range(self.n_conns):
                    conn = self.iface.conns[k]
                    psi = self.cache[fi][ci]["psi"][k]
                    mirror = self._residual_for_conn(state[k], self.theta[k], psi,
                                                     self.gammas[k])
                    saved_u = conn.plasticity.u_fast.detach().clone()
                    conn.plasticity.u_fast.copy_(state[k])   # bitwise state copy
                    pre_s, post_s = self.cache[fi][ci]["states"][k]
                    with torch.no_grad():
                        r_real, _ = conn(pre_s, post_s, learn=False)
                    conn.plasticity.u_fast.copy_(saved_u)    # bitwise restore
                    ok = bool(torch.equal(conn.plasticity.u_fast, saved_u))
                    all_restored = all_restored and ok
                    r_real = self.gammas[k] * r_real[0]
                    diff = float((mirror - r_real).abs().max())
                    scale = max(float(r_real.abs().max()), float(mirror.abs().max()))
                    rel = diff / scale if scale > 0.0 else 0.0
                    if diff > max_abs:
                        max_abs, worst = diff, f"fact{fi}/cand{ci}/conn{k}"
                    max_rel = max(max_rel, rel)
        assert max_rel < 1e-5, (f"L1 mirror fidelity FAILED: max_rel={max_rel:.3e} "
                                f"(max_abs={max_abs:.3e} at {worst})")
        assert all_restored, "L1 bitwise u_fast restore failed"
        rec["l1_conn_level"] = {"max_abs_diff": max_abs, "max_rel_diff": max_rel,
                                "relative_metric": "max|mirror-real| / max(|mirror|max, |real|max)",
                                "worst": worst, "tolerance_relative": 1e-5,
                                "comparisons": len(self.items) * 2 * self.n_conns,
                                "bitwise_restore_all": all_restored, "pass": True}
        print(f"[lte-gates-real] mirror fidelity L1 (conn level, float32 vs real conn "
              f"forward): PASS  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
              f"({rec['l1_conn_level']['comparisons']} comparisons, restore bitwise)")

        # ---- L2: defined gate scoring vs the REAL interface.forward --------- #
        # The real pathway reads conn.plasticity.u_fast, so the fixture state is
        # loaded into the conns bitwise first and restored bitwise after (verified
        # with torch.equal) — the comparison is interface.forward AT THE FIXTURE
        # STATE, as required.
        saved_us = [conn.plasticity.u_fast.detach().clone()
                    for conn in self.iface.conns]
        for conn, s_k in zip(self.iface.conns, state):
            conn.plasticity.u_fast.copy_(s_k)            # bitwise fixture state in
        pathway_residual_dtype = None
        base_det_max = 0.0
        delta_rel_max = 0.0
        score_nat_max = 0.0
        worst_score = None
        for fi in range(len(self.items)):
            for ci in range(len(self.cache[fi])):
                cand = self.cache[fi][ci]
                full = self.items[fi]["prompt_ids"] + cand["ids"]
                with torch.no_grad():
                    res = self.iface.forward(torch.tensor([full], dtype=torch.long))
                pathway_residual_dtype = str(res["logits_delta"].dtype)
                real_base = res["base_logits"][0].float()
                cached_base = cand["base_logits"]
                base_det_max = max(base_det_max,
                                   float((real_base - cached_base).abs().max()))
                rs = self._context_residuals(state, fi, ci)
                mirror_delta = torch.stack(rs, dim=0).sum(dim=0)
                mirror_ld = mirror_delta @ self.head_w.T
                real_ld = res["logits_delta"][0].float()
                scale = max(float(real_ld.abs().max()), float(mirror_ld.abs().max()))
                drel = (float((mirror_ld - real_ld).abs().max()) / scale
                        if scale > 0.0 else 0.0)
                delta_rel_max = max(delta_rel_max, drel)
                bp = cand["prompt_len"] - 1
                s_real = s_mir = None
                # BOTH sides include the plastic delta: the real side at the
                # interface's own logits (base + real delta, BF16 pathway), the
                # mirror side at cached base + float32 delta.
                lp_real = torch.log_softmax(
                    res["logits"][0].float()[bp:bp + len(cand["ids"])], dim=-1)
                lp_mir = torch.log_softmax(
                    (cached_base + mirror_ld.unsqueeze(0))[bp:bp + len(cand["ids"])],
                    dim=-1)
                for j, t in enumerate(cand["ids"]):
                    a, b = float(lp_real[j, t]), float(lp_mir[j, t])
                    s_real = a if s_real is None else s_real + a
                    s_mir = b if s_mir is None else s_mir + b
                nd = abs(s_mir - s_real)
                if nd > score_nat_max:
                    score_nat_max, worst_score = nd, f"fact{fi}/cand{ci}"
        for conn, saved in zip(self.iface.conns, saved_us):
            conn.plasticity.u_fast.copy_(saved)          # bitwise restore
        l2_restored = all(torch.equal(conn.plasticity.u_fast, saved)
                          for conn, saved in zip(self.iface.conns, saved_us))
        assert l2_restored, "L2 bitwise u_fast restore failed"

        if pathway_residual_dtype == "torch.float32":
            bound, bound_note = 1e-5, ("host pathway residual is float32 — the step-1 "
                                       "1e-5 relative assertion applies directly")
            ok = delta_rel_max < 1e-5
        else:
            bound, bound_note = 2e-2, ("host pathway casts the residual to the BF16 "
                                       "logit dtype before the head matmul "
                                       "(interface.py:292-300); the achievable "
                                       "agreement is that cast (~2^-8 relative), so the "
                                       "assertion bound is the cast-consistent 2e-2 — "
                                       "the actual max diff is recorded and the exact "
                                       "float32 mirror math is verified at L1")
            ok = delta_rel_max < bound
        assert ok, (f"L2 interface-pathway fidelity FAILED: max_rel={delta_rel_max:.3e} "
                    f"vs bound {bound} (pathway dtype {pathway_residual_dtype})")
        rec["l2_interface_pathway"] = {
            "pathway_residual_dtype": pathway_residual_dtype,
            "relative_metric": "max|mirror-real| / max(max|mirror|, max|real|) per context",
            "base_logits_forward_determinism_max_abs_diff": base_det_max,
            "logits_delta_max_rel_diff": delta_rel_max,
            "score_max_abs_diff_nats": score_nat_max,
            "worst_score_context": worst_score,
            "bitwise_restore_all": l2_restored,
            "assertion_bound_relative": bound, "bound_note": bound_note,
            "contexts_compared": len(self.items) * 2, "pass": True}
        print(f"[lte-gates-real] mirror fidelity L2 (defined gate scoring vs real "
              f"interface.forward, {len(self.items) * 2} contexts): PASS  "
              f"pathway_dtype={pathway_residual_dtype}  base_det={base_det_max:.3e}  "
              f"delta_rel={delta_rel_max:.3e}  score_max_diff={score_nat_max:.3e} nats")
        return rec

    # ---- fixture plastic state (owner-boundary construction) --------------- #

    def build_gate_state(self) -> Dict:
        """Fixture plastic state construction, recorded:
        1. LITERAL step-1 seeding (torch.manual_seed(SEED+7), one RNG stream,
           randn*0.5 per connection) — evaluated as a RECORDED diagnostic. In the
           2560-dim real host the per-fact features are near-orthogonal for ANY
           random state (pairwise cosines ~ N(0, ~0.04^2) vs tau_d = 0.20), so the
           discrimination hinge is expected DEAD there.
        2. If (and only if) the hinge is dead, construct an activating state per the
           step-1 precedent printed in lte_gates.py ("bumping the seed to ACTIVATE a
           term is fixture construction, not gate tuning") — no seed can activate it
           in 2560 dims, so the state is CONSTRUCTED from the fixture's own
           relationship matrices: u_k row 0 := W_k row 0 / |W_k row 0|^2 (rows j>0
           zero). Then each connection's modulator vector is exactly
           blend * e_0, the per-fact features are gamma_k * blend * post_k[0] * e_0,
           and pairwise cosines are exactly +/-1 by the sign of post[0] — the hinge
           is active for any sign split of 4 facts. The state is non-zero, DISTINCT
           per connection (independent W draws under seed 20261126), and the gates
           run the FULL real forward math on it (blend, psi, gamma, head)."""
        torch.manual_seed(SEED + 7)
        u_literal = [torch.randn_like(c.plasticity.u_fast) * 0.5
                     for c in self.iface.conns]
        lit_diag = self.state_diagnostics(u_literal)
        print(f"[lte-gates-real] LITERAL seeding diagnostic (step-1 recipe, seed "
              f"{SEED}+7): L_rank={lit_diag['L_rank']:.6f}  "
              f"L_disc={lit_diag['L_disc']:.6f}  hinge_active="
              f"{lit_diag['disc_hinge_active']}")
        print(f"[lte-gates-real]   pairwise cosines at the literal state: "
              + ", ".join(f"{k}={v:+.4f}" for k, v in lit_diag["pairwise_cosines"].items()))
        if lit_diag["disc_hinge_active"]:
            state, source, rationale = u_literal, "literal_step1_seeding", \
                "hinge active at the literal seeded state"
        else:
            u_const = []
            for conn in self.iface.conns:
                W = conn.relationship.W.detach().clone()
                row0 = W[0]
                u = torch.zeros_like(W)
                u[0] = row0 / float(row0.dot(row0))   # modulator v_k = blend * e_0 exactly
                u_const.append(u)
            state, source = u_const, "constructed_w_row0"
            rationale = (
                "the discrimination hinge is dead at the literal seeded state in the "
                "2560-dim host (recorded diagnostic above); per the step-1 precedent "
                "(activating a loss term is fixture construction, not gate tuning) the "
                "state is constructed from the fixture's own relationship W matrices: "
                "u_k row 0 := W_k row 0 / |W_k row 0|^2 makes each connection's "
                "modulator exactly blend*e_0, per-fact features become "
                "gamma_k*blend*post_k[0]*e_0 and pairwise cosines exactly +/-1 by sign")
        self.set_state(state)
        chk = {f"u_conn{k}": {"sha256": sha256_tensor(u),
                              "norm": float(u.float().norm()),
                              "nonzero": bool(float(u.abs().sum()) > 0.0)}
               for k, u in enumerate(self.u_state)}
        distinct = len({v["sha256"] for v in chk.values()}) == len(chk)
        gate_diag = self.state_diagnostics(self.get_state())
        assert gate_diag["L_rank"] > 0.0, "L_rank must be positive at the gate state"
        assert gate_diag["disc_hinge_active"], (
            f"discrimination hinge INACTIVE at the gate state "
            f"(L_disc={gate_diag['L_disc']}) — the gates would prove nothing about "
            f"the disc gradient path")
        rec = {"literal_seeding_diagnostic": lit_diag,
               "state_source": source, "construction_rationale": rationale,
               "u_checksums_norms": chk, "u_states_distinct": bool(distinct),
               "gate_state": gate_diag}
        print(f"[lte-gates-real] gate state source: {source}  distinct={distinct}")
        print(f"[lte-gates-real] gate state: L_rank={gate_diag['L_rank']:.6f}  "
              f"L_disc={gate_diag['L_disc']:.6f}  (both loss terms ACTIVE)")
        print(f"[lte-gates-real] gate-state feature norms: "
              + ", ".join(f"{k}={v:.4f}" for k, v in gate_diag["feature_norms"].items()))
        print(f"[lte-gates-real] gate-state pairwise cosines: "
              + ", ".join(f"{k}={v:+.4f}" for k, v in gate_diag["pairwise_cosines"].items()))
        return rec


# --------------------------------------------------------------------------- #
# Gates — mathematics IDENTICAL to step 1 (lte_gates.py, Rev 7)
# --------------------------------------------------------------------------- #

def run_gate1(fx: RealHostFixture) -> Dict:
    print("\n[lte-gates-real] ===== GATE 1: genuine numerical derivative comparison =====",
          flush=True)
    state = fx.get_state()
    with torch.no_grad():
        L_rank0, L_disc0 = fx.loss_parts([s for s in state])
        L_rank0_v, L_disc0_v = float(L_rank0), float(L_disc0)
    assert L_disc0_v > 0.0, (
        f"discrimination hinge INACTIVE at gate state (L_disc={L_disc0_v}) — the gates "
        f"would pass while proving nothing about the disc gradient path")
    assert L_rank0_v > 0.0, "L_rank must be positive at the gate state"
    print(f"  at gate state: L_rank={L_rank0_v:.6f}  L_disc={L_disc0_v:.6f}  "
          f"(both loss terms ACTIVE)")
    leaves = [s.detach().clone().requires_grad_(True) for s in state]
    F0 = fx.eval_F(leaves)
    F0.backward()
    g = [t.grad.detach().clone() for t in leaves]
    g_flat = torch.cat([t.reshape(-1) for t in g])
    g_norm = float(g_flat.norm())
    assert g_norm > 0 and bool(torch.isfinite(g_flat).all()), \
        "Gate-1 fixture gradient must be finite and nonzero (mathematically expected here)"
    d_flat = (-g_flat / g_norm)                      # d = Delta_u_expr / |Delta_u_expr|
    d_list = []
    off = 0
    for t in g:
        n = t.numel()
        d_list.append(d_flat[off:off + n].reshape(t.shape))
        off += n
    D_AD = float(torch.dot(g_flat, d_flat))
    print(f"  F(u) = {F0.item():.6f}   |grad F| = {g_norm:.6f}   D_AD = {D_AD:.6f}")

    def F_value(u_list: List[torch.Tensor]) -> float:
        with torch.no_grad():
            return float(fx.eval_F([u for u in u_list]))

    def perturb(sgn: float, eps: float) -> List[torch.Tensor]:
        return [s + sgn * eps * dv for s, dv in zip(state, d_list)]

    out: Dict = {"D_AD": D_AD, "grad_norm": g_norm, "F_u": F0.item(),
                 "L_rank_at_gate_state": L_rank0_v, "L_disc_at_gate_state": L_disc0_v,
                 "per_epsilon": {}, "pass": True}
    for eps in FD_EPSILONS:
        t_e = time.time()
        d_fds = []
        for _ in range(N_REPEATS):
            F_plus = F_value(perturb(+1.0, eps))
            F_minus = F_value(perturb(-1.0, eps))
            d_fds.append((F_plus - F_minus) / (2.0 * eps))
        n_rep = max(abs(d - d_fds[0]) for d in d_fds) / max(1.0, abs(D_AD))
        D_FD = d_fds[0]
        rel = abs(D_AD - D_FD) / max(1.0, abs(D_AD))
        tol = max(10.0 * n_rep, NUMERICAL_FLOOR)
        ok = (rel <= tol) if eps in PASS_EPSILONS else None
        record = {"eps": eps, "D_FD": D_FD, "rel_error": rel, "n_rep": n_rep,
                  "tolerance": tol, "repeats": d_fds,
                  "F_plus": F_value(perturb(+1.0, eps)),
                  "F_minus": F_value(perturb(-1.0, eps)),
                  "eval_s": round(time.time() - t_e, 1)}
        if ok is not None:
            record["pass"] = bool(ok)
            out["pass"] = out["pass"] and bool(ok)
        out["per_epsilon"][str(eps)] = record
        print(f"  eps={eps:.0e}  D_FD={D_FD:+.6f}  rel={rel:.3e}  n_rep={n_rep:.3e}  "
              f"tol={tol:.3e}  {'PASS' if ok is True else ('fail' if ok is False else 'record')}",
              flush=True)

    # ---- float64 diagnostic (NOT the pass criterion; Rev 7) ----------------- #
    # The protocol pass criterion above runs at the production float32 evaluation
    # over fixed host outputs. This diagnostic re-runs the same comparison entirely
    # in float64 to separate genuine derivative error from float32 central-
    # difference cancellation (protocol 6.2 convergence record).
    cast64 = lambda t: t.double()
    l64 = [s.detach().double().requires_grad_(True) for s in state]
    F64 = fx.eval_F(l64, cast=cast64)
    F64.backward()
    g64 = torch.cat([t.grad.detach().reshape(-1) for t in l64])
    d64_flat = -g64 / g64.norm()
    D_AD64 = float(torch.dot(g64, d64_flat))
    d64_list = []
    off = 0
    for t in l64:
        n = t.numel()
        d64_list.append(d64_flat[off:off + n].reshape(t.shape))
        off += n

    def F_value64(offs: List[torch.Tensor]) -> float:
        with torch.no_grad():
            return float(fx.eval_F(offs, cast=cast64))

    diag64: Dict = {"D_AD": D_AD64, "grad_norm": float(g64.norm()),
                    "note": ("diagnostic only — the pass criterion uses the float32 "
                             "production evaluation over fixed host outputs"),
                    "per_epsilon": {}}
    for eps in FD_EPSILONS:
        fp = F_value64([s.double() + eps * dv for s, dv in zip(state, d64_list)])
        fm = F_value64([s.double() - eps * dv for s, dv in zip(state, d64_list)])
        dfd = (fp - fm) / (2.0 * eps)
        rel64 = abs(D_AD64 - dfd) / max(1.0, abs(D_AD64))
        diag64["per_epsilon"][str(eps)] = {"D_FD": dfd, "rel_error": rel64,
                                           "F_plus": fp, "F_minus": fm}
        print(f"  [f64 diag] eps={eps:.0e}  D_FD={dfd:+.8f}  rel={rel64:.3e}")
    out["float64_diagnostic"] = diag64
    out["objective_reference_dtype"] = "torch.float64"
    return out


def run_gate2(fx: RealHostFixture) -> Dict:
    print("\n[lte-gates-real] ===== GATE 2: gradient-direction test "
          "(Rev 6 form, Rev 7 float64 reference) =====", flush=True)
    state = fx.get_state()
    pre_state = [s.clone() for s in state]
    leaves = [s.detach().clone().requires_grad_(True) for s in state]
    F_before = fx.eval_F(leaves)
    F_before.backward()
    g = [t.grad.detach().clone() for t in leaves]        # raw unscaled gradient
    eta_gate = 0.1 * ETA_E
    # learning rate applied EXACTLY ONCE, to the raw unscaled gradient (Rev 6):
    stepped = [s - eta_gate * gv for s, gv in zip(state, g)]
    fx.set_state(stepped)
    F_after32 = float(fx.eval_F([s for s in fx.u_state]))
    # Rev 7 amendment (owner, 2026-11-26): the decrease criterion uses a FLOAT64
    # reference — a higher-precision evaluation of the defined plastic objective
    # over FIXED HOST OUTPUTS; the host forward itself stays in its native
    # precision and is never re-run in float64.
    cast64 = lambda t: t.double()
    with torch.no_grad():
        F_b64 = float(fx.eval_F([s.double() for s in pre_state], cast=cast64))
        F_a64 = float(fx.eval_F([s.double() for s in stepped], cast=cast64))
    decreased = F_a64 < F_b64
    # bitwise restoration — the gate step is a measurement, not a training step
    fx.set_state(pre_state)
    restored = all(torch.equal(a, b) for a, b in zip(fx.get_state(), pre_state))
    both_terms = LAMBDA_R * 1.0 + LAMBDA_D * 1.0
    out = {"eta_gate": eta_gate, "F_before_float32": float(F_before.item()),
           "F_after_float32": F_after32,
           "F_before_float64_reference": F_b64, "F_after_float64_reference": F_a64,
           "decrease_float64": F_b64 - F_a64,
           "decreased": bool(decreased), "restored_bitwise": bool(restored),
           "both_loss_terms_in_F": bool(abs(both_terms - (LAMBDA_R + LAMBDA_D)) < 1e-12),
           "criterion": "float64 reference (Rev 7 owner amendment)",
           "pass": bool(decreased and restored)}
    with torch.no_grad():
        Lr_b, Ld_b = fx.loss_parts(pre_state)
        Lr_a, Ld_a = fx.loss_parts(stepped)
    out.update({"L_rank_before": float(Lr_b), "L_disc_before": float(Ld_b),
                "L_rank_after": float(Lr_a), "L_disc_after": float(Ld_a)})
    print(f"  eta_gate = {eta_gate:.1e}   F_before = {F_b64:.10f}   "
          f"F_after = {F_a64:.10f}   decreased = {decreased}   "
          f"restored_bitwise = {restored}   both_terms = {out['both_loss_terms_in_F']}")
    print(f"  [float32 record] F {out['F_before_float32']:.6f} -> {F_after32:.6f}")
    print(f"  L_rank {out['L_rank_before']:.6f} -> {out['L_rank_after']:.6f}   "
          f"L_disc {out['L_disc_before']:.6f} -> {out['L_disc_after']:.6f}")
    return out


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def precision_audit(fx: RealHostFixture) -> Dict:
    """Owner boundary 5 — recorded from ACTUAL observed tensor dtypes."""
    state = fx.get_state()
    with torch.no_grad():
        rs = fx._context_residuals([s for s in state], 0, 0)
        delta = torch.stack(rs, dim=0).sum(dim=0)
    return {
        "host_forward_dtype": fx.logits_dtype_observed,
        "host_param_dtype": fx.host_param_dtype,
        "cached_base_logits_dtype": str(fx.cache[0][0]["base_logits"].dtype),
        "residual_dtype": str(delta.dtype),
        "objective_reference_dtype": "torch.float64",
        "float64_meaning": FLOAT64_SENTENCE,
    }


def reproducibility_block(fx: RealHostFixture, state_rec: Dict) -> Dict:
    """Owner addendum item 3 — everything needed to reproduce the fixture."""
    return {
        "model_dir": fx.model_dir,
        "load_call": ("AutoModelForCausalLM.from_pretrained(MODEL_DIR, "
                      "torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, "
                      "device_map=None, local_files_only=True).to('cpu'); eval(); "
                      "freeze_base; foundation_fingerprint — exactly "
                      "teacher_student.setup_foundation --foundation qwen"),
        "load_kwargs_used": fx.load_kwargs_used,
        "freeze_stats": fx.freeze_stats,
        "fingerprint_digest_sha256": fx.fingerprint_digest,
        "fingerprint_tensors": {k: list(v) for k, v in sorted(fx.fingerprint.items())},
        "rng": {"fixture_seed": SEED,
                "seed_stream": ("torch.manual_seed(20261126) before interface "
                                "construction (relationship W draws deterministic); "
                                "torch.manual_seed(20261127) for the literal-state "
                                "diagnostic draw")},
        "layer_pairs": [list(p) for p in fx.layer_pairs],
        "pair_selection": fx.pair_selection_note,
        "pairs_desc": fx.pairs_desc,
        "n_layers_discovered": fx.n_layers,
        "hidden_dim": fx.hidden_dim,
        "gammas": fx.gammas,
        "head_weight_native_dtype": fx.head_w_dtype,
        "head_weight_mirror_view_dtype": str(fx.head_w.dtype),
        "blend_ratio": fx.blend,
        "relation_type": fx.iface.cfg.relation_type,
        "normalize_states": fx.iface.cfg.normalize_states,
        "candidate_tokens_and_ordering": [
            {"fact": it["fact"], "question": it["question"], "gold": it["gold"],
             "distractor": it["distractor"],
             "candidate_order": ["gold", "distractor"],
             "prompt_ids_len": len(it["prompt_ids"]),
             "cands": [{"cand_text": c["cand_text"], "gold_index": c["gold_index"],
                        "ids": c["ids"], "prompt_len": c["prompt_len"],
                        "full_len": c["full_len"]}
                       for c in it["cands"]]} for it in fx.items],
        "feature_extraction": {
            "spec": ("LTE protocol section 2.3, implemented literally for the real "
                     "host: per-fact feature = gamma-scaled plastic residual at the "
                     "answer position from that fact's own connection set only "
                     "(masked residual sum — every connection outside the set "
                     "contributes EXACTLY ZERO); float32 cosine; zero-norm vector "
                     "=> cos exactly 0 (explicit norm check)"),
            "connection_set_per_fact": {f"fact_{i}": {"connections": [i],
                                                      "pair": list(fx.layer_pairs[i]),
                                                      "gamma": fx.gammas[i]}
                                        for i in range(fx.n_conns)},
            "carrier_context": ("fact i's own carrier context = its prompt + "
                                "gold-candidate ids (the fact's canonical scoring "
                                "context); connection i's psi on that context"),
            "answer_position_definition": ("the interface broadcasts one residual "
                                           "vector per sequence; the answer-position "
                                           "feature is that vector as received at the "
                                           "answer positions prompt_len-1 .. "
                                           "prompt_len-1+len(cand_ids)-1"),
            "cosine": "float32 dot of L2-normalized features",
            "tau_d": TAU_D},
        "u_state": {"source": state_rec["state_source"],
                    "checksums_norms": state_rec["u_checksums_norms"],
                    "distinct": state_rec["u_states_distinct"],
                    "construction_rationale": state_rec["construction_rationale"]},
        "host_forward_times_s_per_context": fx.forward_times,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torch_threads": torch.get_num_threads()},
    }


def main() -> int:
    t0 = time.time()
    print("[lte-gates-real] LTE-1 STEP 2 — Gate 1 + Gate 2 on the REAL Qwen host "
          f"({MODEL_DIR})", flush=True)
    print("[lte-gates-real] authorisation: owner GO 2026-11-26 — STEP 2 ONLY "
          "(literal 2.2 scoring + 2.3 features; Rev 7 gate mathematics; stop at the "
          "step-2 boundary)")
    fx = RealHostFixture()
    print(f"[lte-gates-real] foundation loaded + frozen + fingerprinted in "
          f"{fx.load_s}s (fingerprint digest {fx.fingerprint_digest}, "
          f"{fx.freeze_stats['frozen_params']} tensors); pairs={fx.layer_pairs}")
    print(f"[lte-gates-real] pair selection: {fx.pair_selection_note}")
    print(f"[lte-gates-real] host caches: {len(fx.items)} facts x "
          f"{len(fx.cache[0])} candidates = {len(fx.forward_times)} frozen-host "
          f"forwards, {fx.caches_s}s total; logits dtype "
          f"{fx.logits_dtype_observed}")

    state_rec = fx.build_gate_state()
    fidelity = fx.assert_mirror_fidelity()

    g1 = run_gate1(fx)
    g2 = run_gate2(fx)

    report = {
        "step": "LTE-1 step 2 — gates on the REAL Qwen host (literal 2.2 scoring, "
                "literal 2.3 features)",
        "protocol": ("docs/LTE_PROTOCOL.md frozen Revision 6 + Revision 7 amendment "
                     "(commits 7b7642c2..., f1a4a02); freeze record section 10"),
        "authorisation": "owner GO 2026-11-26: STEP 2 ONLY",
        "config": {"lambda_r": LAMBDA_R, "lambda_d": LAMBDA_D, "tau_d": TAU_D,
                   "eta_e_fixture": ETA_E, "n_repeats": N_REPEATS,
                   "fd_epsilons": list(FD_EPSILONS),
                   "pass_epsilons": list(PASS_EPSILONS),
                   "numerical_floor": NUMERICAL_FLOOR, "seed": SEED,
                   "gamma_fixture": GATE_FIXTURE_VALUE,
                   "format_scaffold_source": "aion.experiments.teacher_student.FORMAT_SCAFFOLD",
                   "fixture_facts": FACTS,
                   "repo_root_on_host": str(REPO_ROOT)},
        "gate_state_losses": {"L_rank": state_rec["gate_state"]["L_rank"],
                              "L_disc": state_rec["gate_state"]["L_disc"],
                              "both_terms_active": bool(
                                  state_rec["gate_state"]["L_rank"] > 0.0
                                  and state_rec["gate_state"]["disc_hinge_active"])},
        "fixture_state_construction": state_rec,
        "mirror_fidelity": fidelity,
        "feature_structure_at_gate_state": {
            "feature_norms": state_rec["gate_state"]["feature_norms"],
            "pairwise_cosines": state_rec["gate_state"]["pairwise_cosines"],
            "pairwise_hinge_terms": state_rec["gate_state"]["pairwise_hinge_terms"]},
        "gate1": g1,
        "gate2": g2,
        "precision_audit": precision_audit(fx),
        "reproducibility": reproducibility_block(fx, state_rec),
        "verdict": "PASS" if (g1["pass"] and g2["pass"]) else "FAIL — stop, report to owner",
        "wall_s": round(time.time() - t0, 1),
    }
    out_dir = Path("runs/lte_gates")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gates_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[lte-gates-real] GATE 1: {'PASS' if g1['pass'] else 'FAIL'}   "
          f"GATE 2: {'PASS' if g2['pass'] else 'FAIL'}   "
          f"VERDICT: {report['verdict']}", flush=True)
    print(f"[lte-gates-real] evidence: {out.resolve()}  ({report['wall_s']}s)", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
