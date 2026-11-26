# LTE-1 gradient-gate verification question (adversarial consult)

## THE QUESTION (verbatim, from the owner's LTE-1 gradient-gate work)

"A preregistered gradient gate verifies Delta_u = -eta_gate * grad F by checking F decreases after the step, using a float64 reference evaluation of the objective F (the host forward itself stays in its native precision — only the plastic-objective math over cached fixed host outputs is evaluated in float64). Gate 1 separately verifies autograd vs central-difference agreement with a repeatability-based tolerance. Question: does the float64-reference decrease check + autograd/central-difference agreement together constitute a sound verification that the implemented gradient is connected to and descends the actual candidate-scoring loss, or is there a remaining failure mode where both checks pass while the gradient is wrong? Identify any such mode or state clearly that none exists."

## ACTUAL IMPLEMENTATION (excerpts from aion/experiments/lte_gates.py, the artifact in question)

Fixture: frozen tiny host ("the host is a gradient conduit, never an update target").
The plastic path is evaluated through a differentiable MIRROR of the real
PlasticRelationalConnection forward; a mirror-fidelity assertion compares mirror
vs real forward at ONE state (fact-0 prompt states, u = u0) with max|diff| < 1e-5.
psi (per-connection relationship features) is precomputed once per fact with
no_grad from frozen host hidden states (comment: "constant in the fixture: no u
dependence"). Host hidden states cached detached; base logits recomputed
per-evaluation under the same frozen weights in eval() mode (deterministic).

The mirror (autograd path over u):
    w = blend * u_list[k] + (1 - blend) * theta[k]     # blend Jacobian: EXACTLY here
    transmitted = w.unsqueeze(0) * psi_cache[fact][k]
    r_k = gamma_k * transmitted.sum(dim=-1)

Candidate scores per fact:
    delta_full = sum_k r_k                    # (D,)
    base_out  = host logits (no_grad, native precision, frozen)
    lp_full   = log_softmax(base_out + (delta_full @ head_w.T).unsqueeze(0), -1)
    score_c   = sum_j lp_full[prompt_len - 1 + j, cand_c_ids[j]]

F = lambda_r * L_rank + lambda_d * L_disc, with
    L_rank = CE over the two candidate scores, mean over 4 facts
    L_disc = mean over ordered pairs of clamp(cos(f_i, f_j) - tau_d, min=0)
    where f_i = sum_k r_k(fact_i) normalized.

GATE 1 (numerical derivative comparison):
    g = autograd grad F(u_state); d = -g/|g|
    D_AD = g . d
    D_FD(eps) = [F(u+eps d) - F(u-eps d)] / (2 eps), central difference,
      K=5 identical evaluations at fixed eps/state -> repeatability noise n_rep
    Pass: |D_AD - D_FD(eps)| / max(1, |D_AD|) <= max(10*n_rep, 1e-4) at eps in {1e-2, 1e-3}
    (eps=1e-4 recorded, not a pass gate). A float64 diagnostic re-run of the same
    comparison exists but is explicitly NOT the pass criterion.
    Precondition asserted: L_disc > 0 at the gate state (both hinge terms active).

GATE 2 (Rev 7, owner amendment): gradient-direction test.
    g = autograd grad F(u_state) (raw unscaled gradient)
    step once: stepped = u_state - eta_gate * g,  eta_gate = 0.1 * eta_e
    decreased = F_float64(stepped) < F_float64(u_state)   # float64 REFERENCE eval
      — the float64 eval casts only the plastic-objective math (psi cache, theta,
      head_w, mirror math) to double over the SAME cached fixed host outputs; the
      host forward itself stays native precision (its outputs are fixed anyway)
    pass = decreased AND bitwise state restoration.

Recorded evidence also separates L_rank and L_disc before/after the step, to prove
both gradient paths were exercised by the step.

## REVIEW FOCUS

Adversarial verification-theory review. Answer the QUESTION above: is the check
pair sound evidence that the implemented gradient is (a) connected to and (b)
descends the actual candidate-scoring loss? For every failure mode you claim,
state it concretely in terms of THIS implementation. Do not invent facts.
