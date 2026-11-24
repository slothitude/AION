"""AION LEARNING DIAGNOSTIC - Gate-4 acceptance (owner directive).

Run:  PYTHONPATH=src C:/Python313/python.exe scripts/diagnostic_learning.py

Sections:
  [MEASURE]  psi RMS / distribution / |du| per update / prediction change per update
  [G2 LAW]   calibrated (gain, eta) pairs - invariance check + stability boundary
  [GATE-4]   7 acceptance criteria with numbers
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from nnfs.connections.plastic import PlasticRelationalConnection
from nnfs.connections.network import RelationalNetwork

X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
Y = (2.0 * X[:, 0] + 3.0 * X[:, 1]).unsqueeze(1)
POST_ONES = torch.ones(4, 1)
POST_SIGNED = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
U_CLAMP = 10.0


def make_conn(lr=0.05, **kw):
    conn = PlasticRelationalConnection(
        pre_idx=0, post_idx=1, pre_dim=2, post_dim=1,
        context_dim=0, learning_rate=lr, weight_decay=1e-5, **kw)
    with torch.no_grad():
        conn.relationship.W.fill_(1.0)
    return conn


# ---------------------------------------------------------------------------
# Gate 4 acceptance, criterion 5: coefficient recovery [2, 3]
# ---------------------------------------------------------------------------
def criterion5(steps=800, lr=0.05):
    print("-" * 70)
    print("CRITERION 5: coefficient recovery y = 2*x1 + 3*x2")
    torch.manual_seed(0)
    conn = make_conn(lr=lr, consolidation_interval=0)
    traj = []
    for t in range(steps):
        y_hat = conn(X, POST_ONES, context=None, target=None, learn=False)[0]
        if t % 100 == 0:
            traj.append(round(((y_hat - Y) ** 2).mean().item(), 5))
        conn(X, POST_ONES, context=None, target=Y, learn=True)
    a = conn.plasticity.get_combined_weight(conn.blend_ratio).detach().flatten()
    y_hat = conn(X, POST_ONES, context=None, target=None, learn=False)[0]
    mse = ((y_hat - Y) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(a, torch.tensor([2.0, 3.0]), dim=0).item()
    ok = cos > 0.999 and mse < 1e-3
    print(f"  effective a = {a.tolist()} (target [2, 3])  cos={cos:.6f}")
    print(f"  predictions {[round(v, 4) for v in y_hat.flatten().tolist()]} vs {Y.flatten().tolist()}")
    print(f"  train MSE {traj} -> {mse:.2e}   PASS={ok}")
    return ok, mse


# ---------------------------------------------------------------------------
# Gate 4 acceptance, criterion 6: negative / zero-mean stability
# ---------------------------------------------------------------------------
def criterion6(steps=800, lr=0.05):
    print("-" * 70)
    print("CRITERION 6: stable behaviour with negative and zero-mean inputs")
    results = {}
    for label, x, y, post in (
            ("all-positive", X, Y, POST_ONES),
            ("negative", -X, -Y, -POST_ONES),
            ("zero-mean-post", X, Y, POST_SIGNED)):
        torch.manual_seed(1)
        conn = make_conn(lr=lr)
        u_max, l0, traj = 0.0, None, []
        for t in range(steps):
            y_hat = conn(x, post, context=None, target=None, learn=False)[0]
            l = ((y_hat - y) ** 2).mean().item()
            if l0 is None:
                l0 = l
            if t % 200 == 0:
                traj.append(round(l, 4))
            u_max = max(u_max, conn.plasticity.u_fast.abs().max().item())
            conn(x, post, context=None, target=y, learn=True)
        y_hat = conn(x, post, context=None, target=None, learn=False)[0]
        lf = ((y_hat - y) ** 2).mean().item()
        stable = np.isfinite(lf) and lf <= traj[0] and u_max < U_CLAMP
        results[label] = stable
        print(f"  {label:<15}: loss {traj} -> {round(lf, 4)} (init {round(l0, 4)})  "
              f"max|u|={u_max:.3f}  {'STABLE' if stable else 'UNSTABLE'}")
    return all(results.values())


# ---------------------------------------------------------------------------
# Gate 4 acceptance, criterion 2: finite-difference gradient agreement
# ---------------------------------------------------------------------------
def criterion2(lr=0.05, eps=1e-3):
    print("-" * 70)
    print("CRITERION 2: finite-difference gradient agreement (one update)")
    torch.manual_seed(0)
    conn = make_conn(lr=lr)
    with torch.no_grad():
        conn.plasticity.u_fast.copy_(torch.tensor([[0.5, -0.3]]))
    conn(X, POST_ONES, context=None, target=None, learn=True)  # activate mask

    def loss():
        with torch.no_grad():
            y_hat = conn(X, POST_ONES, context=None, target=None, learn=False)[0]
            return 0.5 * ((y_hat - Y) ** 2).mean().item()

    u0 = conn.plasticity.u_fast.detach().clone()
    grad = torch.zeros_like(u0)
    for k in range(u0.numel()):
        with torch.no_grad():
            conn.plasticity.u_fast.copy_(u0)
            conn.plasticity.u_fast.flatten()[k] += eps
            lp = loss()
            conn.plasticity.u_fast.flatten()[k] -= 2 * eps
            lm = loss()
            conn.plasticity.u_fast.copy_(u0)
        grad.flatten()[k] = (lp - lm) / (2 * eps)
    conn(X, POST_ONES, context=None, target=Y, learn=True)
    du = conn.plasticity.u_fast.detach() - u0
    expected = -lr * grad
    rel = ((du - expected).abs() / (expected.abs() + 1e-9)).max().item()
    ok = torch.allclose(du, expected, rtol=1e-2, atol=1e-6)
    print(f"  implemented du = {du.flatten().tolist()}")
    print(f"  FD gradient    = {expected.flatten().tolist()}")
    print(f"  max rel error = {rel:.2e}   PASS={ok}")
    return ok


# ---------------------------------------------------------------------------
# MEASURE protocol + G2 law + criteria 1, 3, 4, 7 (live network)
# ---------------------------------------------------------------------------
def gate4_live(gain=1.0, lr=0.05, leak=0.5, steps=600, seed=0, hold=10, measure=True):
    print("-" * 70)
    print(f"GATE 4 LIVE: RelationalNetwork g={gain} eta={lr} leak={leak} "
          f"({steps} steps, fixed-pattern persistence)")
    torch.manual_seed(seed)
    net = RelationalNetwork(num_neurons=8, neuron_dim=8, sparsity=0.15,
                            relation_type="post", learning_rate=lr,
                            trace_decay=0.95, consolidation_rate=0.0,
                            blend_ratio=1.0, state_leak_rate=leak)
    for c in net.connections.values():
        torch.nn.init.xavier_uniform_(c.relationship.W, gain=gain)
        c.plasticity.learning_rate = lr
    net.reset_states(batch_size=4)
    pat = torch.randn(4, 8, 8)
    ho = torch.randn(4, 8, 8)

    def evaluate(pats):
        net.reset_states(batch_size=4)
        vals = []
        with torch.no_grad():
            for p in pats:
                inp = p.unsqueeze(0).expand(4, -1, -1).contiguous()
                net.neuron_states.zero_()
                for c in net.connections.values():
                    c.eligibility_trace.reset()
                vs = []
                for _ in range(hold):
                    out = net.step(inp, target_next=inp, learn=False)
                    vs.append(out["connection_prediction_mse"])
                vals.append(float(np.mean(vs[hold // 2:])))
        return float(np.mean(vals))

    ho_before, tr_before = evaluate(ho), evaluate(pat)
    curve, umax, psi_rms, du_norm, dtr_norm = [], 0.0, [], [], []
    psi_final = None
    for t in range(steps):
        p = pat[(t // hold) % len(pat)]
        inp = p.unsqueeze(0).expand(4, -1, -1).contiguous()
        u_before = [c.plasticity.u_fast.detach().clone() for c in net.connections.values()]
        with torch.no_grad():
            trans_before = net.step(inp, learn=False)["transmissions"].clone()
        out = net.step(inp, target_next=inp, learn=True)
        curve.append(out["connection_prediction_mse"])
        with torch.no_grad():
            umax = max(umax, max(c.plasticity.u_fast.abs().max().item()
                                 for c in net.connections.values()))
            if measure:
                du_norm.append(np.mean([(c.plasticity.u_fast.detach() - u0).norm().item()
                                        for c, u0 in zip(net.connections.values(), u_before)]))
                psis = []
                for c in net.connections.values():
                    pre = net.neuron_states[:, int(c.pre_idx)]
                    post = net.neuron_states[:, int(c.post_idx)]
                    w = c.relationship.W.unsqueeze(0).expand(pre.shape[0], -1, -1)
                    psis.append((post.unsqueeze(-1) * w).flatten())
                psi_all = torch.cat(psis)
                psi_rms.append(psi_all.pow(2).mean().sqrt().item())
                if t == steps - 1:
                    psi_final = (psi_all.min().item(),
                                 torch.quantile(psi_all, torch.tensor([0.05, 0.5, 0.95])).tolist(),
                                 psi_all.max().item())
                trans_after = net.step(inp, learn=False)["transmissions"]
                dtr_norm.append((trans_after - trans_before).abs().mean().item())
    curve = np.array(curve)
    finite = bool(np.isfinite(curve).all())
    ho_after, tr_after = evaluate(ho), evaluate(pat)
    imp = curve[:50].mean() - curve[-50:].mean()

    if measure and psi_final:
        print(f"  [MEASURE 1-2] psi RMS={psi_rms[-1]:.4f} (mean over run {np.mean(psi_rms):.4f}); "
              f"distribution min={psi_final[0]:.3f} p5/p50/p95="
              f"{[round(v, 3) for v in psi_final[1]]} max={psi_final[2]:.3f}")
        print(f"  [MEASURE 3-4] ||du_fast|| per update = {np.mean(du_norm):.5f} "
              f"(blend=1.0 so ||da|| == ||du||)")
        print(f"  [MEASURE 5]   prediction change per update |d_trans| = "
              f"{np.mean(dtr_norm):.5f}")
    print(f"  [CRIT 3] train MSE: first50={curve[:50].mean():.4f} mid={np.mean(curve[steps // 2:3 * steps // 4]):.4f} "
          f"last50={curve[-50:].mean():.4f} falling={curve[-50:].mean() < curve[:50].mean()} "
          f"finite={finite}")
    print(f"  [CRIT 4] HELD-OUT connection_prediction_mse: {ho_before:.4f} -> {ho_after:.4f} "
          f"({100 * (1 - ho_after / max(ho_before, 1e-12)):+.1f}%)  "
          f"train-patterns {tr_before:.4f} -> {tr_after:.4f}")
    print(f"  [CRIT 7] max|u_fast|={umax:.3f} vs clamp {U_CLAMP:.0f} "
          f"(margin {100 * (1 - umax / U_CLAMP):.1f}%)  all-finite={finite}")
    return {"psi_rms": psi_rms[-1], "ho_before": ho_before, "ho_after": ho_after,
            "falling": finite and curve[-50:].mean() < curve[:50].mean(),
            "finite": finite, "umax": umax}


def g2_law():
    print("-" * 70)
    print("G2 LAW: single-step output-side invariance for (g, eta/g^2)")
    print("  eta_new = eta_old*(g_old/g_new)^2 keeps the per-update change in")
    print("  y_hat invariant to first order; da scales 1/g, psi scales g.")
    results = {}
    for g, eta in ((1.0, 0.05), (2.0, 0.0125), (4.0, 0.003125), (4.0, 0.05)):
        torch.manual_seed(0)
        net = RelationalNetwork(num_neurons=8, neuron_dim=8, sparsity=0.15,
                                relation_type="post", learning_rate=eta,
                                trace_decay=0.95, consolidation_rate=0.0,
                                blend_ratio=1.0, state_leak_rate=0.5)
        for c in net.connections.values():
            torch.nn.init.xavier_uniform_(c.relationship.W, gain=1.0)
            with torch.no_grad():
                c.relationship.W.mul_(g)   # same draw, scaled: psi_g = g * psi_1
            c.plasticity.learning_rate = eta
        net.reset_states(batch_size=4)
        inp = torch.randn(4, 8, 8)
        tgt = torch.randn(4, 8, 8)
        with torch.no_grad():
            trans0 = net.step(inp, learn=False)["transmissions"].clone()
        u0 = [c.plasticity.u_fast.detach().clone() for c in net.connections.values()]
        net.step(inp, target_next=tgt, learn=True)
        with torch.no_grad():
            trans1 = net.step(inp, learn=False)["transmissions"]
            du = np.mean([(c.plasticity.u_fast.detach() - u0).norm().item()
                          for c, u0 in zip(net.connections.values(), u0)])
        dtrans = (trans1 - trans0).abs().mean().item()
        results[(g, eta)] = (dtrans, du)
        print(f"  g={g:<4} eta={eta:<9}: |d_yhat| per update={dtrans:.6f}  "
              f"||du||={du:.6f}")
    d0 = results[(1.0, 0.05)][0]
    spread = max(abs(results[(g, e)][0] - d0) / d0 for g, e in
                 ((2.0, 0.0125), (4.0, 0.003125)))
    inv = spread < 0.05
    print(f"  output-side invariance spread vs g=1: {spread:.1e} "
          f"-> {'CONFIRMED' if inv else 'FAILED'}")
    print(f"  da scales ~1/g: ||du|| ratio g=4/g=1 = "
          f"{results[(4.0, 0.003125)][1] / results[(1.0, 0.05)][1]:.3f} (expect ~0.25)")
    print("  trajectory-level invariance does NOT hold once psi drives the state")
    print("  feedback loop (psi scales the forward output too) - the law is")
    print("  first-order in the update, which is what calibration controls.")

    # overshoot on the live loop: uncalibrated gain at fixed eta
    torch.manual_seed(0)
    net = RelationalNetwork(num_neurons=8, neuron_dim=8, sparsity=0.15,
                            relation_type="post", learning_rate=0.05,
                            trace_decay=0.95, consolidation_rate=0.0,
                            blend_ratio=1.0, state_leak_rate=0.5)
    for c in net.connections.values():
        torch.nn.init.xavier_uniform_(c.relationship.W, gain=4.0)
        c.plasticity.learning_rate = 0.05
    net.reset_states(batch_size=4)
    pat = torch.randn(4, 8, 8)
    umax, finite = 0.0, True
    for t in range(300):
        inp = pat[t % 4].unsqueeze(0).expand(4, -1, -1).contiguous()
        out = net.step(inp, target_next=inp, learn=True)
        finite = finite and np.isfinite(out["connection_prediction_mse"])
        with torch.no_grad():
            umax = max(umax, max(c.plasticity.u_fast.abs().max().item()
                                 for c in net.connections.values()))
    print(f"  UNCALIBRATED live loop (g=4, eta=0.05, 300 steps): finite={finite} "
          f"max|u|={umax:.2f} <- owner's predicted overshoot (16x effective step)")
    return inv


if __name__ == "__main__":
    print("=" * 70)
    print("AION LEARNING DIAGNOSTIC - Gate-4 acceptance")
    print("=" * 70)
    ok2 = criterion2()
    ok5, mse5 = criterion5()
    ok6 = criterion6()
    inv = g2_law()
    res = gate4_live()
    print("=" * 70)
    print("GATE-4 SUMMARY")
    print(f"  1. psi RMS (documented): {res['psi_rms']:.4f} at g=1, leak=0.5")
    print(f"  2. FD gradient agreement: {'PASS' if ok2 else 'FAIL'}")
    print(f"  3. falling training MSE:  {'PASS' if res['falling'] else 'FAIL'}")
    print(f"  4. falling held-out MSE:  {res['ho_before']:.4f} -> {res['ho_after']:.4f} "
          f"({'PASS' if res['ho_after'] < res['ho_before'] else 'FAIL'})")
    print(f"  5. coefficient recovery:  {'PASS' if ok5 else 'FAIL'} (final MSE {mse5:.2e})")
    print(f"  6. negative/zero-mean:    {'PASS' if ok6 else 'FAIL'}")
    print(f"  7. |u| bounded:           max|u|={res['umax']:.2f} vs clamp 10 "
          f"({'PASS' if res['umax'] < 5 else 'FAIL'})")
    print(f"  g2 invariance:            {'CONFIRMED' if inv else 'FAILED'}")
    print("=" * 70)
