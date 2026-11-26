# LTE-1 Step-5 Diagnosis — why the calibration found no learning

**Date:** 2026-11-27. **Evidence:** `runs/lte/calibration_20261127_073033.json` (362 KB, all 30 attempts, 16.65 h) + the step-2 gate evidence. **Status:** diagnosis from existing evidence — no new runs.

---

## The finding

**The expression-channel updates are numerically real but ~1,000,000× too small to influence the scored loss.**

Three independent lines of evidence from the calibration record:

### 1. Identical L_rank curves across a 100× learning-rate range

All 30 attempts — both λ_d configs, all five η_e values (1e-4 → 1e-2), all three λ_r rungs — have **identical per-question L_rank curves**: min 0.009, max 8.742, first-quarter mean 2.741, last-quarter mean 4.341, to three decimal places. If the expression updates influenced the scored loss, curves would diverge across a 100× η_e range. They do not.

### 2. The per-step descent is real but ~10⁻⁶ on the loss scale

Gate 2 (step 2, real host, float64): F_before 1.5707771360 → F_after 1.5707758223, decrease = 1.31e-06, matching the analytic η_gate·‖∇F‖² = 1.32e-06 exactly. The gradient is correct and descends — by a **millionth** of the loss scale per step.

### 3. The γ-gates trained 5 orders of magnitude harder than the expression channel

Pass B (γ training): gate_lr = 100, 80 steps — γ landed at 0.85–5.12. Pass C (expression): the u_fast updates move the objective by ~1e-6/step. The readout is therefore dominated by γ + the frozen host; the u_fast expression contribution is invisible in the recorded losses — hence identical curves across all hyperparameters.

---

## What this means

- **The step-2 gates are correct and did their job**: the gradient exists, descends, and the host is untouched. The instrument is honest.
- **The failure is a magnitude problem, not a sign or connectivity problem**: the expression signal is ~10⁻⁶ per step against losses of order 1. No configuration within the frozen grid could learn, because no η_e in the grid (max 1e-2, capped by §5's preregistered ladder) closes a 10⁶ gap.
- **The earlier Phase-2 collinearity finding stands unchanged** — it is the *deeper* representation problem. But the *immediate* calibration failure has a simpler, quantified cause: signal magnitude.

---

## What would close the gap (owner decision — none of this is authorised)

1. **Raise η_e far beyond the frozen grid** (the gap needs η_e ~ 1e0–1e2, or a normalised/adaptive update rule) — a protocol revision, since the grid was preregistered.
2. **Normalise the expression update** (e.g., update magnitude fixed per step, direction from the gradient) — also a protocol revision.
3. **Train γ and u_fast jointly in the expression pass** rather than separating γ (pass B) from expression (pass C) — changes the §2.1 sequencing law, owner-gated.
4. **Accept the magnitude finding and step back** to diagnose the representation (the Phase-2 collinearity) before any further expression training.

Every option starts from the same recorded evidence — 30 attempts, complete per-question curves, and a quantified 10⁶ gap.

---

*Diagnosis written 2026-11-27 by the nnfs lane from the recorded evidence. No new runs performed.*
