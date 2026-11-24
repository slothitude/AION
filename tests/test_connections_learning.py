"""Regression tests for NNFS connection learning (owner gates 2/3).

Fix 2: corrected plasticity  u_fast <- u_fast + blend * lr * (y - y_hat) * psi
       with y_hat = sum_i a_ij * psi_ij (the ACTUAL forward output).
Fix 3: output-preserving consolidation  u <- u - d ; theta += (blend/(1-blend))*d
       keeps a = blend*u_fast + (1-blend)*theta (and y_hat) invariant.

Run: PYTHONPATH=src python -m pytest tests/test_connections_learning.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pytest
import torch

from nnfs.connections.plastic import PlasticRelationalConnection
from nnfs.connections.network import RelationalNetwork

# Canonical task: y = 2*x1 + 3*x2 on inputs spanning R^2
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
Y = (2.0 * X[:, 0] + 3.0 * X[:, 1]).unsqueeze(1)          # [[2],[3],[5],[7]]
POST_ONES = torch.ones(4, 1)
POST_SIGNED = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])  # zero-mean post


def make_conn(post_dim=1, lr=0.1, weight_decay=0.0, **kwargs):
    """Unit test connection: W_rel = ones so psi = post * pre and the
    forward map is exactly y_hat = a @ x (a = blend*u_fast + (1-blend)*theta)."""
    conn = PlasticRelationalConnection(
        pre_idx=0, post_idx=1, pre_dim=2, post_dim=post_dim,
        context_dim=0, learning_rate=lr, trace_decay=0.95,
        weight_decay=weight_decay, **kwargs
    )
    with torch.no_grad():
        conn.relationship.W.fill_(1.0)
    return conn


def forward_loss(conn, x, y, post):
    """Loss of the ACTUAL forward output (no learning side effects)."""
    with torch.no_grad():
        y_hat = conn(x, post, context=None, target=None, learn=False)[0]
        return 0.5 * ((y_hat - y) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Test 1: forward consistency - the update is the finite-difference gradient
# ---------------------------------------------------------------------------
def test_forward_consistency_finite_difference():
    torch.manual_seed(0)
    conn = make_conn(lr=0.05)
    # Seed a nonzero fast state and activate the trace mask
    with torch.no_grad():
        conn.plasticity.u_fast.normal_(0.0, 0.5)
    conn(X, POST_ONES, context=None, target=None, learn=True)  # trace update only

    u_before = conn.plasticity.u_fast.detach().clone()
    l0 = forward_loss(conn, X, Y, POST_ONES)

    eps = 1e-3
    grad = torch.zeros_like(u_before)
    for k in range(u_before.numel()):
        with torch.no_grad():
            conn.plasticity.u_fast.copy_(u_before)
            conn.plasticity.u_fast.flatten()[k] += eps
            lp = forward_loss(conn, X, Y, POST_ONES)
            conn.plasticity.u_fast.flatten()[k] -= 2 * eps
            lm = forward_loss(conn, X, Y, POST_ONES)
            conn.plasticity.u_fast.copy_(u_before)
        grad.flatten()[k] = (lp - lm) / (2 * eps)

    # One real learning step (weight_decay=0, clamp inactive at these magnitudes)
    conn(X, POST_ONES, context=None, target=Y, learn=True)
    delta_u = conn.plasticity.u_fast.detach() - u_before

    # u <- u + blend*lr*mean(delta*psi); for L = 0.5*mean((y-y_hat)^2),
    # dL/du = -blend*mean(delta*psi), so delta_u == -lr * grad exactly.
    assert torch.allclose(delta_u, -0.05 * grad, rtol=1e-3, atol=1e-6), (
        f"update {delta_u.flatten().tolist()} vs finite-difference "
        f"{(-0.05 * grad).flatten().tolist()} (L0={l0:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 2: gradient direction - one update reduces loss in BOTH post regimes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("post,label", [(POST_ONES, "all-positive"), (POST_SIGNED, "signed/zero-mean")])
def test_gradient_direction_one_step(post, label):
    torch.manual_seed(1)
    conn = make_conn(lr=0.1)
    l_before = forward_loss(conn, X, Y, post)
    conn(X, post, context=None, target=Y, learn=True)
    l_after = forward_loss(conn, X, Y, post)
    assert torch.isfinite(torch.tensor([l_before, l_after])).all()
    assert l_after < l_before, f"{label}: loss rose {l_before:.4f} -> {l_after:.4f}"
    # and the fast state stays far from the old +-10 runaway clamp
    assert conn.plasticity.u_fast.abs().max().item() < 5.0


# ---------------------------------------------------------------------------
# Test 3: independent inputs - learns BOTH coefficients 2 and 3
# ---------------------------------------------------------------------------
def test_independent_inputs_coefficient_recovery():
    torch.manual_seed(2)
    conn = make_conn(lr=0.1, consolidation_interval=0)  # theta stays 0
    for _ in range(800):
        conn(X, POST_ONES, context=None, target=Y, learn=True)

    a = conn.plasticity.get_combined_weight(conn.blend_ratio).detach().flatten()
    true_w = torch.tensor([2.0, 3.0])
    y_hat = conn(X, POST_ONES, context=None, target=None, learn=False)[0]

    mse = ((y_hat - Y) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(a, true_w, dim=0).item()
    # Effective learned mapping approaches [2, 3] (owner check a)
    assert cos > 0.999, f"effective a={a.tolist()} not aligned with [2,3], cos={cos:.5f}"
    assert (a - true_w).abs().max().item() < 0.05, f"effective a={a.tolist()} vs [2,3]"
    # Actual predictions match targets
    assert mse < 1e-3, f"MSE={mse:.6f}"
    assert (y_hat.flatten() - Y.flatten()).abs().max().item() < 0.05
    # Raw fast state is a/[blend] with theta=0: both coefficients independent
    u = conn.plasticity.u_fast.detach().flatten()
    assert torch.allclose(u, a / 0.7, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: trace separation - coactivity trace vs error-driven eligibility
# ---------------------------------------------------------------------------
def test_trace_separation():
    torch.manual_seed(3)

    # (a) coactivity trace evolves identically with or without an error target;
    #     only the error target moves u_fast.
    conn_err = make_conn(lr=0.1)
    conn_noerr = make_conn(lr=0.1)
    for t in range(5):
        conn_err(X, POST_ONES, context=None, target=Y, learn=True)
        conn_noerr(X, POST_ONES, context=None, target=None, learn=True)
    assert torch.equal(conn_err.eligibility_trace.trace,
                       conn_noerr.eligibility_trace.trace), \
        "coactivity trace diverged between error/no-error paths"
    assert conn_err.plasticity.u_fast.abs().sum().item() > 0, "error path did not learn"
    assert conn_noerr.plasticity.u_fast.abs().sum().item() == 0, \
        "u_fast moved without a target - error eligibility leaked into no-error path"

    # (b) the error-driven update is independent of the trace VALUE (it only
    #     gates through the active mask): scaling the trace does not change du.
    def one_step(trace_scale):
        c = make_conn(lr=0.1)
        with torch.no_grad():
            c.plasticity.u_fast.copy_(torch.tensor([[0.4, -0.3]]))
        c(X, POST_ONES, context=None, target=None, learn=True)  # activate trace
        with torch.no_grad():
            c.eligibility_trace.trace.mul_(trace_scale)
        u0 = c.plasticity.u_fast.detach().clone()
        c(X, POST_ONES, context=None, target=Y, learn=True)
        return c.plasticity.u_fast.detach() - u0

    du_ref = one_step(1.0)
    for scale in (0.5, 1000.0):
        du = one_step(scale)
        assert torch.equal(du, du_ref), (
            f"trace scaled by {scale} changed the update: error eligibility "
            f"is contaminated by the coactivity trace"
        )


# ---------------------------------------------------------------------------
# Test 5: consolidation is output-preserving (a and y_hat invariant)
# ---------------------------------------------------------------------------
def test_consolidation_output_preserving():
    torch.manual_seed(4)
    conn = make_conn(lr=0.1, consolidation_interval=0)
    with torch.no_grad():
        conn.plasticity.u_fast.copy_(torch.tensor([[3.0, -2.0]]))
    conn(X, POST_ONES, context=None, target=Y, learn=True)  # activate trace mask

    a_before = conn.plasticity.get_combined_weight(conn.blend_ratio).detach().clone()
    y_hat_before = conn(X, POST_ONES, context=None, target=None, learn=False)[0].detach().clone()
    u_before = conn.plasticity.u_fast.detach().clone()
    theta_before = conn.plasticity.theta_slow.detach().clone()

    stats = conn.consolidate()
    assert stats.get("consolidated", 0) > 0, f"nothing consolidated: {stats}"

    a_after = conn.plasticity.get_combined_weight(conn.blend_ratio).detach()
    y_hat_after = conn(X, POST_ONES, context=None, target=None, learn=False)[0].detach()

    # Effective connection and predictions invariant (float-tolerant, not bitwise)
    assert torch.allclose(a_before, a_after, atol=1e-6, rtol=1e-6), (
        f"effective a moved: max|da|={(a_after - a_before).abs().max().item():.3e}"
    )
    assert torch.allclose(y_hat_before, y_hat_after, atol=1e-6, rtol=1e-6), (
        f"prediction moved after consolidation: "
        f"max|dy|={(y_hat_after - y_hat_before).abs().max().item():.3e}"
    )
    # Both sides of the transfer actually moved
    assert not torch.equal(u_before, conn.plasticity.u_fast.detach()), "u_fast did not move"
    assert not torch.equal(theta_before, conn.plasticity.theta_slow.detach()), "theta_slow did not move"

    # Configurable consolidation interval: interval=10 fires automatically
    # (the check runs on the step count BEFORE it is incremented, so the
    # 11th update carries count==10)
    conn2 = make_conn(lr=0.1, consolidation_interval=10)
    for _ in range(11):
        conn2(X, POST_ONES, context=None, target=Y, learn=True)
    assert conn2.plasticity.theta_slow.abs().sum().item() > 0, \
        "consolidation_interval=10 did not fire within 11 updates"


# ---------------------------------------------------------------------------
# Test 6: metric contract on RelationalNetwork.step
# ---------------------------------------------------------------------------
def test_metric_contract_keys():
    torch.manual_seed(5)
    net = RelationalNetwork(num_neurons=4, neuron_dim=4, sparsity=0.5, context_dim=4)
    inp = torch.randn(2, 4, 4)
    target = torch.randn(2, 4, 4)

    out = net.step(inp, target_next=target, learn=True)
    for key in ("connection_prediction_mse", "state_prediction_mse", "readout_mse"):
        assert key in out, f"missing metric key {key}"
        assert isinstance(out[key], float) and math.isfinite(out[key]), f"{key}={out[key]}"
    assert out["mse"] == out["readout_mse"], "legacy mse is not the readout alias"
    # primary metric must exist even before any learning
    assert out["connection_prediction_mse"] >= 0.0

    out_none = net.step(inp, learn=True)  # no target
    for key in ("connection_prediction_mse", "state_prediction_mse", "readout_mse"):
        assert key in out_none and out_none[key] is None, f"{key} should be None without target"


# ---------------------------------------------------------------------------
# Test 7: evidence gates open in realistic O(1)-error regimes (deadlock fix a)
# ---------------------------------------------------------------------------
def test_evidence_gate_reachable_at_o1_error():
    """The gate must encode STABILITY, not absolute error level: with O(1)
    prediction errors and exercised coactivity the gate opens; a connection
    with no coactivity (never exercised) stays closed. Note the 0.95-decay
    eligibility trace smooths intermittent exercise by design, so
    exercised-vs-absent is the discriminable pair at the trace level;
    noise/dead entries are additionally excluded by the activity mask."""
    torch.manual_seed(6)
    conn_ok = make_conn(lr=0.1)
    conn_silent = make_conn(lr=0.1)

    for t in range(300):
        conn_ok(X, POST_ONES, context=None, target=Y, learn=True)
        # silent: zero coactivity every step (no exercise evidence)
        conn_silent(torch.zeros_like(X), torch.zeros_like(POST_ONES),
                    context=None, target=torch.zeros_like(Y), learn=True)

    gate_ok = torch.sigmoid(conn_ok.plasticity.consolidation_gate)
    gate_silent = torch.sigmoid(conn_silent.plasticity.consolidation_gate)
    # stability statistic is scale-free: O(1) error did not pin it near 0
    assert gate_ok.mean().item() > 0.8, (
        f"gate did not open under O(1) error: mean={gate_ok.mean().item():.3f}"
    )
    assert gate_silent.mean().item() < 0.3, (
        f"gate opened with no coactivity: mean={gate_silent.mean().item():.3f}"
    )

    # end-to-end: consolidation now actually fires and transfers
    th_before = conn_ok.plasticity.theta_slow.detach().clone()
    stats = conn_ok.consolidate()
    th_after = conn_ok.plasticity.theta_slow.detach()
    assert stats.get("gated_consolidated", 0) > 0, f"gates blocked transfer: {stats}"
    assert (th_after - th_before).abs().max().item() > 0, "consolidation did not transfer"


# ---------------------------------------------------------------------------
# Test 8: blend=1.0 deadlock broken by blend annealing (deadlock fix b)
# ---------------------------------------------------------------------------
def test_consolidation_blend_1_anneals_and_transfers():
    torch.manual_seed(7)
    conn = make_conn(lr=0.1, blend_ratio=1.0)
    with torch.no_grad():
        conn.plasticity.u_fast.copy_(torch.tensor([[3.0, -2.0]]))
    conn(X, POST_ONES, context=None, target=Y, learn=True)  # activate mask

    a_before = conn.plasticity.get_combined_weight(conn.blend_ratio).detach().clone()
    y_before = conn(X, POST_ONES, context=None, target=None, learn=False)[0].detach().clone()
    th_before = conn.plasticity.theta_slow.detach().clone()
    assert conn.blend_ratio == 1.0

    stats = conn.consolidate()
    a_after = conn.plasticity.get_combined_weight(conn.blend_ratio).detach()
    y_after = conn(X, POST_ONES, context=None, target=None, learn=False)[0].detach()

    # blend annealed away from 1.0 and written back on the connection
    assert stats["blend_before"] == 1.0
    assert conn.blend_ratio == stats["blend_after"] == 0.95, (
        f"blend did not anneal: after={stats.get('blend_after')}"
    )
    # slow weights now carry consolidated knowledge
    assert (conn.plasticity.theta_slow - th_before).abs().max().item() > 0, "no transfer"
    # effective connection and predictions preserved through transfer+anneal
    assert torch.allclose(a_before, a_after, atol=1e-6, rtol=1e-6), (
        f"a moved under anneal: max|da|={(a_after - a_before).abs().max().item():.3e}"
    )
    assert torch.allclose(y_before, y_after, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 9: preservation invariant under repeated events / arbitrary blends
# ---------------------------------------------------------------------------
def test_consolidation_preservation_invariant_random():
    torch.manual_seed(8)
    for blend in (1.0, 0.7, 0.3):
        conn = make_conn(lr=0.1, blend_ratio=blend)
        with torch.no_grad():
            conn.plasticity.u_fast.normal_(0.0, 1.5)
            conn.plasticity.theta_slow.normal_(0.0, 0.5)
        # repeated consolidation events: anneal walks blend toward its floor
        for _ in range(6):
            conn(X, POST_ONES, context=None, target=Y, learn=True)  # keep mask alive
            a_ref = conn.plasticity.get_combined_weight(conn.blend_ratio).detach().clone()
            stats = conn.consolidate()
            if stats.get("consolidated", 0) == 0:
                break
            a_now = conn.plasticity.get_combined_weight(conn.blend_ratio).detach()
            assert torch.allclose(a_ref, a_now, atol=1e-6, rtol=1e-6), (
                f"blend={blend}: a drifted over events, max|da|="
                f"{(a_now - a_ref).abs().max().item():.3e}"
            )
        # blend may only walk DOWN toward the floor, never past it or up
        assert conn.plasticity.blend_floor - 1e-9 <= conn.blend_ratio or blend < conn.plasticity.blend_floor
        assert min(blend, max(conn.plasticity.blend_floor, blend)) + 1e-9 >= conn.blend_ratio >= min(blend, conn.plasticity.blend_floor) - 1e-9
        # predictions preserved end-to-end at the final blend
        y1 = conn(X, POST_ONES, context=None, target=None, learn=False)[0]
        w = conn.plasticity.get_combined_weight(conn.blend_ratio)
        y2 = (w @ X.T).T
        assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 10: consolidation_rate=0 disables consolidation (valid controls)
# ---------------------------------------------------------------------------
def test_consolidation_rate_zero_disables():
    """Plastic-D0-style controls set consolidation_rate=0.0; the interval
    alone must never fire sweeps for them (gate-6 control validity)."""
    torch.manual_seed(9)
    conn = make_conn(lr=0.1, consolidation_rate=0.0)
    with torch.no_grad():
        conn.plasticity.u_fast.copy_(torch.tensor([[3.0, -2.0]]))
    conn(X, POST_ONES, context=None, target=Y, learn=True)

    th_before = conn.plasticity.theta_slow.detach().clone()
    blend_before = conn.blend_ratio
    stats = conn.consolidate()
    assert stats.get("consolidation_disabled") == 1, f"rate=0 still consolidated: {stats}"
    assert stats.get("consolidated", 0) == 0
    assert torch.equal(th_before, conn.plasticity.theta_slow.detach()), "theta moved at rate=0"
    assert conn.blend_ratio == blend_before, "blend annealed at rate=0"

    # and the periodic path honours it too
    conn2 = make_conn(lr=0.1, consolidation_rate=0.0, consolidation_interval=10)
    for _ in range(12):
        conn2(X, POST_ONES, context=None, target=Y, learn=True)
    assert conn2.plasticity.theta_slow.abs().sum().item() == 0.0, (
        "periodic consolidation fired at rate=0"
    )
