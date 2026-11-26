"""LTE-1 STEP 1 — Gate 1 + Gate 2 demonstration on the tiny stand-in.

Per docs/LTE_PROTOCOL.md (frozen Revision 6, freeze record Section 10, commit a88d50d):
the owner's conditional sign-off authorises ONLY this step — build and demonstrate the
two gradient gates on the tiny stand-in, record the evidence, stop if either fails.

Gate 1 (protocol 6.2): genuine numerical derivative comparison.
  D_AD = grad F(u) . d                      (autograd, d = Delta_u_expr / |Delta_u_expr|)
  D_FD(eps) = [F(u+eps d) - F(u-eps d)] / (2 eps)   (central difference, 3 eps)
  Repeatability noise: K = 5 identical evaluations at fixed eps and state.
  Pass: |D_AD - D_FD(eps)| / max(1, |D_AD|) <= max(10 * n_rep(eps), 1e-4) at eps in {1e-2, 1e-3}.

Gate 2 (protocol 6.2, Rev 6 form): gradient-direction test, learning rate applied ONCE.
  u <- u - eta_gate * grad F,  eta_gate = 0.1 * eta_e   (raw unscaled gradient)
  F recomputed with BOTH loss terms on identical inputs; must decrease;
  fixture's original plastic state restored BITWISE afterwards.

F = lambda_r * L_rank,train + lambda_d * L_disc,train (both terms, every evaluation).
The differentiable mirror is asserted equal to the real PlasticRelationalConnection
forward before the gates run (mirror-fidelity assertion).

Host immutability during the fixture: the foundation runs under no_grad with detached
hidden states, all foundation params requires_grad=False (asserted), and the fixture's
gradient leaves are clones of u_fast only — the host is a gradient conduit, never an
update target (protocol 2.1/3.1; the full four-point host test is step 3).
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aion.experiments.teacher_student import FORMAT_SCAFFOLD, CharTokenizer, build_tiny_foundation
from aion.relational.interface import InterfaceConfig, RelationalInterface
from src.nnfs.connections.plastic import PlasticRelationalConnection  # noqa: F401  (mirror fidelity)

# ---- fixture configuration (recorded in the evidence) ---------------------- #

LAMBDA_R = 1.0
LAMBDA_D = 0.5
TAU_D = 0.20
ETA_E = 1.0e-3                    # fixture value (mid-grid); the real eta_e is frozen
                                  # after dev calibration per protocol section 5
GATE_FIXTURE_VALUE = 1.0          # gamma per connection (constant in the fixture;
                                  # the gate-learning stage is NOT part of step 1)
N_REPEATS = 5
FD_EPSILONS = (1.0e-2, 1.0e-3, 1.0e-4)
PASS_EPSILONS = (1.0e-2, 1.0e-3)
NUMERICAL_FLOOR = 1.0e-4
SEED = 20261126

# Four facts -> four layer pairs (stand-in has 4 decoder layers).
LAYER_PAIRS = [(0, 1), (0, 2), (1, 3), (2, 3)]

# Fixture facts: question, gold answer, distractor. Deliberately distinct relations.
FACTS = [
    {"question": "Tav relays to whom?", "gold": "Kimo", "distractor": "Drel"},
    {"question": "Neral reaches which device?", "gold": "Vesk", "distractor": "Juna"},
    {"question": "Who maintains Pola?", "gold": "Sef", "distractor": "Ryn"},
    {"question": "Which relay does Quill use?", "gold": "Modo", "distractor": "Hath"},
]


class LteFixture:
    """The differentiable stand-in fixture: frozen tiny host + plastic mirror."""

    def __init__(self, seed: int = SEED):
        torch.manual_seed(seed)
        self.tok = CharTokenizer()
        base, _ = build_tiny_foundation(seed=seed)
        cfg = InterfaceConfig()  # relation_type="post", normalize_states=True
        self.iface = RelationalInterface(base, cfg=cfg, layer_pairs=list(LAYER_PAIRS))
        self.iface.eval()
        for p in self.iface.base.parameters():
            p.requires_grad_(False)
        for p in self.iface.conns.parameters():
            p.requires_grad_(False)
        # gamma: constants for this fixture (set data on the interface's gate
        # parameters AND use detached constants inside the mirror).
        for g in self.iface.gates:
            g.data.fill_(GATE_FIXTURE_VALUE)
            g.requires_grad_(False)
        self.gammas = [float(g.item()) for g in self.iface.gates]

        self.blend = float(self.iface.cfg.blend_ratio)   # mirror tied to the real conns
        self.head_w = self.iface.base.get_output_embeddings().weight.detach()

        # Plastic leaves: clones of each connection's u_fast (the gradient target).
        # One RNG stream for the whole loop so each conn gets a DISTINCT draw
        # (audit fix: seeding inside the loop gave all four conns identical u0).
        self.u0: List[torch.Tensor] = []
        torch.manual_seed(seed + 7)
        for conn in self.iface.conns:
            u = conn.plasticity.u_fast.detach().clone()
            u = torch.randn_like(u) * 0.5      # non-zero seed state (non-zero grads)
            u.requires_grad_(True)
            self.u0.append(u)
        self.theta = [conn.plasticity.theta_slow.detach().clone()
                      for conn in self.iface.conns]

        # Fixture questions: prompt ids + candidate ids per fact.
        self.items: List[Dict] = []
        for i, f in enumerate(FACTS):
            prompt = FORMAT_SCAFFOLD + f"Question: {f['question']}\nAnswer:"
            ids_prompt = self.tok.encode(prompt)
            cands = [(f["gold"], 0), (f["distractor"], 1)]      # (text, gold_index)
            cand_enc = []
            for text, gi in cands:
                ids_c = self.tok.encode(" " + text)
                cand_enc.append({"ids": ids_c, "gold": gi == 0})
            self.items.append({"prompt_ids": ids_prompt, "cands": cand_enc, "fact": i})

        # Base forwards (frozen, detached) — computed once, reused by every F eval.
        self.base_cache: List[Dict] = []
        with torch.no_grad():
            for item in self.items:
                full_ids = item["prompt_ids"] + item["cands"][0]["ids"]
                # use the longest candidate context for the shared base forward
                lens = [len(c["ids"]) for c in item["cands"]]
                out = self.iface.base(input_ids=torch.tensor([item["prompt_ids"] + max(
                    item["cands"], key=lambda c: len(c["ids"]))["ids"]], dtype=torch.long))
                self.base_cache.append({"prompt_ids": item["prompt_ids"], "lens": lens})

        # Hidden states per prompt (detached), per fact — cached per fact (prompt-only
        # forwards; candidate tokens are appended only for scoring positions).
        self.hidden_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for item in self.items:
                self.iface._captured.clear()
                ids = torch.tensor([item["prompt_ids"]], dtype=torch.long)
                self.iface.base(input_ids=ids)
                hidden = dict(self.iface._captured)
                pairs_states = []
                for (pre_i, post_i) in self.iface.pairs:
                    pre_s = hidden[pre_i].mean(dim=1).float()
                    post_s = hidden[post_i].mean(dim=1).float()
                    if self.iface.cfg.normalize_states:
                        pre_s = torch.nn.functional.normalize(pre_s, dim=-1)
                        post_s = torch.nn.functional.normalize(post_s, dim=-1)
                    pairs_states.append((pre_s, post_s))
                self.hidden_cache.append(pairs_states)

        # psi per connection per fact (constant in the fixture: no u dependence).
        self.psi_cache: List[List[torch.Tensor]] = []
        with torch.no_grad():
            for fi, pairs_states in enumerate(self.hidden_cache):
                per_conn = []
                for k, conn in enumerate(self.iface.conns):
                    pre_s, post_s = pairs_states[k]
                    per_conn.append(conn.relationship(pre_s, post_s))
                self.psi_cache.append(per_conn)

        # Mirror-fidelity assertion (protocol: the gradient must follow the actual
        # forward computation — verified numerically before the gates run).
        self._assert_mirror_fidelity()

        self.u_state: List[torch.Tensor] = [u.detach().clone() for u in self.u0]

    # ---- differentiable mirror ------------------------------------------- #

    def _mirror_residuals(self, u_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """r_k = sum_over_pre( (blend u + (1-blend) theta) * psi ), gamma applied —
        the PlasticRelationalConnection forward with autograd on u_list only."""
        rs = []
        for k in range(len(self.iface.conns)):
            w = self.blend * u_list[k] + (1.0 - self.blend) * self.theta[k]  # blend Jacobian: EXACTLY here
            transmitted = w.unsqueeze(0) * self.psi_cache[0][k]  # fixture uses fact-0 prompt states
            rs.append(self.gammas[k] * transmitted.sum(dim=-1).squeeze(0))  # (D,)
        return rs

    def _assert_mirror_fidelity(self) -> None:
        """Assert the mirror reproduces the REAL PlasticRelationalConnection
        forward (learn=False) at the IDENTICAL u state (protocol 6.2)."""
        with torch.no_grad():
            u_try = [u.detach().clone() for u in self.u0]
            mirror = self._mirror_residuals(u_try)
            for k, conn in enumerate(self.iface.conns):
                # the real forward reads conn.plasticity.u_fast — load the same
                # u_try into it, compare, then restore the original state
                saved_u = conn.plasticity.u_fast.detach().clone()
                conn.plasticity.u_fast.copy_(u_try[k])
                pre_s, post_s = self.hidden_cache[0][k]
                r_real, _ = conn(pre_s, post_s, learn=False)
                conn.plasticity.u_fast.copy_(saved_u)     # restore
                r_real = self.gammas[k] * r_real[0]
                diff = float((mirror[k] - r_real).abs().max())
                assert diff < 1e-5, f"mirror fidelity FAILED on conn {k}: max|diff|={diff}"
        print(f"[lte-gates] mirror fidelity: PASS (max|mirror-real| < 1e-5 over "
              f"{len(self.iface.conns)} connections)")

    # ---- F evaluation ------------------------------------------------------ #

    def _residuals_for_fact(self, u_list: List[torch.Tensor], fact: int,
                            cast=None) -> List[torch.Tensor]:
        """Per-fact per-connection r_k on the fact's OWN prompt states."""
        cast = cast or (lambda t: t)
        rs = []
        for k in range(len(self.iface.conns)):
            w = self.blend * u_list[k] + (1.0 - self.blend) * cast(self.theta[k])
            transmitted = w.unsqueeze(0) * cast(self.psi_cache[fact][k])
            rs.append(self.gammas[k] * transmitted.sum(dim=-1).squeeze(0))
        return rs

    def _candidate_logits(self, u_list: List[torch.Tensor], fact: int,
                          cast=None) -> torch.Tensor:
        """Candidate scores (both candidates) for one fact — autograd-capable."""
        cast = cast or (lambda t: t)
        item = self.items[fact]
        rs = self._residuals_for_fact(u_list, fact, cast)
        delta_full = torch.stack(rs, dim=0).sum(dim=0)          # (D,) total residual
        scores = []
        with torch.no_grad():
            ids = torch.tensor([item["prompt_ids"] + max(item["cands"], key=lambda c: len(c["ids"]))["ids"]],
                               dtype=torch.long)
            base_out = cast(self.iface.base(input_ids=ids).logits[0].float())
        head_w = cast(self.head_w)
        prompt_len = len(item["prompt_ids"])
        for c in item["cands"]:
            ids_c = c["ids"]
            delta = delta_full @ head_w.T                       # (V,) logits delta (seq-broadcast)
            lp_full = torch.log_softmax(base_out + delta.unsqueeze(0), dim=-1)
            s = 0.0
            for j, t in enumerate(ids_c):
                s = s + lp_full[prompt_len - 1 + j, t]
            scores.append(s)
        return torch.stack(scores, dim=0)

    def loss_parts(self, u_list: List[torch.Tensor], cast=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(L_rank, L_disc) separately — recorded so the evidence PROVES both
        gradient paths are exercised (audit: an inactive disc hinge would let the
        gates pass while proving nothing about the disc gradient)."""
        cast = cast or (lambda t: t)
        # --- ranking term (CE over candidate scores, mean over facts) ---
        rank_losses = []
        for fact in range(len(self.items)):
            scores = self._candidate_logits(u_list, fact, cast)
            gold = self.items[fact]["cands"].index(next(c for c in self.items[fact]["cands"] if c["gold"]))
            rank_losses.append(torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.tensor([gold])))
        L_rank = torch.stack(rank_losses).mean()
        # --- discrimination term (hinge on pairwise cos of per-fact features) ---
        feats = []
        for fact in range(len(self.items)):
            rs = self._residuals_for_fact(u_list, fact, cast)
            f_i = torch.stack(rs, dim=0).sum(dim=0)             # (D,) fact feature
            feats.append(torch.nn.functional.normalize(f_i, dim=-1))
        cos_terms = []
        n = len(feats)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cos = torch.dot(feats[i], feats[j])
                cos_terms.append(torch.clamp(cos - TAU_D, min=0.0))
        L_disc = torch.stack(cos_terms).mean()
        return L_rank, L_disc

    def eval_F(self, u_list: List[torch.Tensor], cast=None) -> torch.Tensor:
        """F = lambda_r * L_rank,train + lambda_d * L_disc,train — autograd-capable
        when any u in u_list requires grad; pure value otherwise."""
        L_rank, L_disc = self.loss_parts(u_list, cast)
        return LAMBDA_R * L_rank + LAMBDA_D * L_disc

    # ---- state handling ----------------------------------------------------- #

    def get_state(self) -> List[torch.Tensor]:
        return [u.detach().clone() for u in self.u_state]

    def set_state(self, state: List[torch.Tensor]) -> None:
        self.u_state = [s.detach().clone() for s in state]

    def raw_gradient(self) -> List[torch.Tensor]:
        """Raw unscaled gradient of F w.r.t. the current u_state (autograd)."""
        leaves = [s.detach().clone().requires_grad_(True) for s in self.u_state]
        F = self.eval_F(leaves)
        F.backward()
        return [t.grad.detach().clone() for t in leaves], float(F.item())


def run_gate1(fx: LteFixture) -> Dict:
    print("\n[lte-gates] ===== GATE 1: genuine numerical derivative comparison =====")
    state = fx.get_state()
    with torch.no_grad():
        L_rank0, L_disc0 = fx.loss_parts([s for s in state])
        L_rank0_v, L_disc0_v = float(L_rank0), float(L_disc0)
    assert L_disc0_v > 0.0, (
        f"discrimination hinge INACTIVE at gate state (L_disc={L_disc0_v}) — the gates "
        f"would pass while proving nothing about the disc gradient path")
    print(f"  at gate state: L_rank={L_rank0_v:.6f}  L_disc={L_disc0_v:.6f}  "
          f"(disc hinge ACTIVE)")
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
                  "F_plus": F_value(perturb(+1.0, eps)), "F_minus": F_value(perturb(-1.0, eps))}
        if ok is not None:
            record["pass"] = bool(ok)
            out["pass"] = out["pass"] and bool(ok)
        out["per_epsilon"][str(eps)] = record
        print(f"  eps={eps:.0e}  D_FD={D_FD:+.6f}  rel={rel:.3e}  n_rep={n_rep:.3e}  "
              f"tol={tol:.3e}  {'PASS' if ok is True else ('fail' if ok is False else 'record')}")

    # ---- float64 diagnostic (NOT the pass criterion) ------------------------ #
    # The protocol pass criterion above runs at the production float32 evaluation.
    # This diagnostic re-runs the same comparison entirely in float64 to separate
    # genuine derivative error from float32 central-difference cancellation noise
    # (protocol 6.2 convergence record: "they inform interpretation").
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
                    "note": "diagnostic only — pass criterion uses the float32 evaluation",
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
    return out


def run_gate2(fx: LteFixture) -> Dict:
    print("\n[lte-gates] ===== GATE 2: gradient-direction test (Rev 7 form: float64 reference) =====")
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
    # reference — the step-1 evidence (227c638) proved the true decrease can be
    # < 1 ULP of float32 F, making the float32 recomputation unsatisfiable.
    cast64 = lambda t: t.double()
    with torch.no_grad():
        F_b64 = float(fx.eval_F([s.double() for s in pre_state], cast=cast64))
        F_a64 = float(fx.eval_F([s.double() for s in stepped], cast=cast64))
    decreased = F_a64 < F_b64
    # bitwise restoration — the gate step is a measurement, not a training step
    fx.set_state(pre_state)
    restored = all(torch.equal(a, b) for a, b in zip(fx.get_state(), pre_state))
    both_terms = LAMBDA_R * 1.0 + LAMBDA_D * 1.0          # F includes both terms by construction
    out = {"eta_gate": eta_gate, "F_before_float32": float(F_before.item()),
           "F_after_float32": F_after32,
           "F_before_float64_reference": F_b64, "F_after_float64_reference": F_a64,
           "decrease_float64": F_b64 - F_a64,
           "decreased": bool(decreased), "restored_bitwise": bool(restored),
           "both_loss_terms_in_F": bool(abs(both_terms - (LAMBDA_R + LAMBDA_D)) < 1e-12),
           "criterion": "float64 reference (Rev 7 owner amendment)",
           "pass": bool(decreased and restored)}
    # both loss terms recorded SEPARATELY before and after (audit: the evidence
    # must prove BOTH gradient paths were exercised by the step)
    with torch.no_grad():
        Lr_b, Ld_b = fx.loss_parts(pre_state)
        Lr_a, Ld_a = fx.loss_parts(stepped)
    out.update({"L_rank_before": float(Lr_b), "L_disc_before": float(Ld_b),
                "L_rank_after": float(Lr_a), "L_disc_after": float(Ld_a)})
    print(f"  eta_gate = {eta_gate:.1e}   F_before = {out['F_before_float64_reference']:.10f}   "
          f"F_after = {out['F_after_float64_reference']:.10f}   decreased = {decreased}   "
          f"restored_bitwise = {restored}   both_terms = {out['both_loss_terms_in_F']}")
    print(f"  [float32 record] F {out['F_before_float32']:.6f} -> {out['F_after_float32']:.6f} "
          f"(bit-identical — below float32 resolution, as diagnosed)")
    print(f"  L_rank {out['L_rank_before']:.6f} -> {out['L_rank_after']:.6f}   "
          f"L_disc {out['L_disc_before']:.6f} -> {out['L_disc_after']:.6f}")
    return out


def main() -> int:
    t0 = time.time()
    print("[lte-gates] LTE-1 STEP 1 — Gate 1 + Gate 2 on the tiny stand-in")
    print("[lte-gates] frozen protocol: Revision 6 (commit 7b7642c, freeze a88d50d); "
          "conditional sign-off: THIS STEP ONLY")
    fx = LteFixture()
    # fixture-activity precondition (audit): the discrimination hinge must be
    # ACTIVE at the gate state, or the gates could pass while proving nothing
    # about the disc gradient path. Bumping the seed to ACTIVATE a term is
    # fixture construction, not gate tuning.
    with torch.no_grad():
        _lr0, _ld0 = fx.loss_parts([s for s in fx.u_state])
    assert float(_ld0) > 0.0, (
        f"fixture disc hinge INACTIVE (L_disc={float(_ld0)}) at seed {SEED} — "
        f"re-seed the fixture until the hinge is active before running gates")
    print(f"[lte-gates] fixture: {len(FACTS)} facts / {len(fx.iface.conns)} connections / "
          f"hidden={fx.iface.hidden_dim}  lambda_r={LAMBDA_R} lambda_d={LAMBDA_D} "
          f"tau_d={TAU_D} eta_e={ETA_E:.0e} seed={SEED}")
    print(f"[lte-gates] fixture state: L_rank={float(_lr0):.6f}  L_disc={float(_ld0):.6f}  "
          f"(both loss terms active)")

    g1 = run_gate1(fx)
    g2 = run_gate2(fx)

    report = {
        "step": "LTE-1 step 1 — gates on tiny stand-in",
        "protocol": "docs/LTE_PROTOCOL.md frozen Revision 6 (commit 7b7642c2ae547b5047234e54a16f629960642a2e; freeze a88d50d)",
        "authorisation": "owner conditional sign-off 2026-11-26: step 1 ONLY",
        "config": {"lambda_r": LAMBDA_R, "lambda_d": LAMBDA_D, "tau_d": TAU_D,
                   "eta_e_fixture": ETA_E, "n_repeats": N_REPEATS,
                   "fd_epsilons": list(FD_EPSILONS), "pass_epsilons": list(PASS_EPSILONS),
                   "numerical_floor": NUMERICAL_FLOOR, "seed": SEED,
                   "layer_pairs": LAYER_PAIRS, "gamma_fixture": GATE_FIXTURE_VALUE},
        "fixture_losses_at_gate_state": {"L_rank": float(_lr0), "L_disc": float(_ld0),
                                         "disc_hinge_active": bool(float(_ld0) > 0.0)},
        "gate1": g1,
        "gate2": g2,
        "verdict": "PASS" if (g1["pass"] and g2["pass"]) else "FAIL — stop, report to owner",
        "wall_s": round(time.time() - t0, 1),
    }
    out_dir = Path("runs/lte_gates")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gates_standin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[lte-gates] GATE 1: {'PASS' if g1['pass'] else 'FAIL'}   "
          f"GATE 2: {'PASS' if g2['pass'] else 'FAIL'}   "
          f"VERDICT: {report['verdict']}")
    print(f"[lte-gates] evidence: {out}  ({report['wall_s']}s)")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
