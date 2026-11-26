# LTE-1 — Learn-Toward-Expression Protocol (DRAFT for owner review — NOT AUTHORIZED TO RUN)

**Status:** DESIGN + PREREGISTRATION DRAFT only. Per the owner's decision 2026-11-26: design and preregistration are authorised; **execution is NOT**. No mechanism changes, no runs, no tuning until this document is reviewed and the owner lifts the hold.

**Central requirement (owner verbatim):** *Learning a fact must create a distinguishable representation that can subsequently move the correct answer ahead of its distractors on unseen questions.*

**Design law (owner verbatim):** *The learning objective and the expression objective must describe the same computation. We do not bolt another readout onto a representation that has never learned to carry the information that readout needs.*

---

## 1. What the four experiments measured, and what this protocol changes

| Established (Phase 1/2, interventions, Phase-3) | Consequence for design |
|---|---|
| Plastic state acquires, stores, corrects, retains, restarts bitwise | Keep the entire ownership/retention apparatus unchanged |
| Transmitted residual is large (‖γ·r‖ 7.2–10.4) but answer-uncorrelated (gold helped 44–61 % ≈ coin-flip; margins negative; rank→correct 0/0/0) | Signal magnitude was never the deficit; the deficit is *what the signal encodes*. The learning signal must therefore be derived at the point of answer selection |
| Per-fact features nearly collinear (novelty 0.0013–0.0071); density multiplied signal without decorrelating | Discrimination must be a property of the objective, not a hope |
| The current objective (`connection_prediction_mse`) never references the decoder | The expression objective enters the update law itself |

**The change, stated plainly:** the plastic update law gains an **expression term** whose gradient is computed **at the frozen host's candidate-selection**, and flows back through the frozen host (gradients only — zero host weight updates) into the plastic state. The existing prediction term is untouched.

---

## 2. Objective and equations

### 2.1 The two channels of the update

The corrected plasticity law (owner canon, unchanged):

```
u_fast ← u_fast + 0.7·η_p·δ_pred·ψ          (prediction channel — EXACTLY as frozen in baseline v2)
```

gains a second channel:

```
u_fast ← u_fast + η_e·g_expr                (expression channel — NEW)
```

with

```
g_expr = ∂L_rank/∂u_fast
```

computed by automatic differentiation **through the plastic forward (ψ-path) and the frozen host** (foundation parameters frozen; `requires_grad` set on the injected residual only). The host is a gradient conduit, never an update target.

### 2.2 The ranking loss (answer alignment — owner constraint 2)

At the answer position, for a training question with candidate set {c_1..c_K} (gold = g):

```
L_rank = −log softmax_g( s_1 .. s_K )
s_k    = candidate logit from the frozen host's scoring of c_k with the plastic residual injected
```

CE-over-candidates is chosen because it is the *same computation* as the Phase-2/3 scoring instrument — the loss and the exam measure the same quantity by construction.

### 2.3 The discrimination term (fact discrimination — owner constraint 1)

For the per-fact transmitted features f_i (i = 1..N facts):

```
L_disc = mean_{i≠j} max(0, cos(f_i, f_j) − τ_d)        τ_d = 0.20 (fixed at prereg)
```

Hinge form: no penalty below τ_d, linear penalty above — pressure against collapse without fighting legitimate similarity. Total objective:

```
L = L_pred + λ_r·L_rank + λ_d·L_disc
```

λ_r, λ_d **fixed at preregistration from dev-split evidence only** (see §5). Defaults proposed: λ_r = 1.0, λ_d = 0.5. If the dev split shows λ_d unnecessary (novelty already ≥ gate), λ_d = 0 is chosen *before* the final exam and recorded — that is the only tuning allowed, and it never touches exam questions.

### 2.4 What does NOT change

- ψ, the forward map, dual-timescale structure, the 0.7 factor — untouched
- Consolidation: exact-solve blend-anneal law unchanged; it acts on u_fast however reached
- γ-sequencing law: plasticity first, gate learning second — preserved
- rate ≤ 0 hard-disable; HostMode authority; zero-direct-write gym law — all preserved
- Foundation: frozen, fingerprint-checked before and after (Phase-3 instrument)

**Protected-property note (listed because any change here is an owner gate):** the update law is extended by an ADDITIVE term with its own coefficient; the existing term is byte-identical to baseline v2. The ψ-independence scope condition is unaffected (ψ not modified). This extension is declared as an amendment to the corrected plasticity law and stands or falls by this protocol's own gates.

---

## 3. Proposed implementation (no code written yet)

| Piece | Source | Change |
|---|---|---|
| Harness | `aion/experiments/teacher_student.py` | new `--lte` mode; Phase-1/2/3 paths untouched |
| Candidate scoring | exists (Phase-2/3 scoring) | batched (K candidates per forward) |
| Backward through frozen host | new | autograd pass, residual-only `requires_grad`; no optimizer over host params; assertion `host_grad_count == 0` in every step |
| Novelty measurement | exists (Lane G decomposition) | becomes a live gate input, not just a post-hoc diagnostic |
| Swap test | new (see §6) | fact-feature substitution at exam time |
| Questions | existing 20-rel world bank | split: train / dev / final-exam, fixed at preregistration (final exam = the Phase-3 43-question structure: direct 28 / paraphrase / 15-question combination battery; held-out splits [5,5,5] re-derived per LTE seed and frozen before teach) |

**Budget estimate (honest, CPU BF16, Lappy):** expression channel adds forward+backward per training question (~8–10 s per 6-candidate batch): teach ≈ 3.0–3.6 ks/config; exam incl. ablation rerun + swap tests ≈ 2.5–3 h. Per-config ≈ 6 h; full LTE-1 block (controls + arm + analyses) ≈ 12–14 h. Wall uncertainty ±30 % (co-tenant load), as in Phase-3.

---

## 4. Controls (constraint 6: frozen evaluation)

| Control | Purpose |
|---|---|
| A — frozen floor | the Phase-3 floor, re-measured in-session |
| B — retrieval | the structural ceiling for transfer; blind to composition (0.000) |
| C — old objective, same harness | **the null that must replicate the Phase-3 floor** — if C moves, the instrument changed, and the comparison is void (instruments-before-science law) |

---

## 5. Tuning law (no tuning against the final exam)

Dev questions (held out from teach, distinct from final exam) are the ONLY surface where λ_r, λ_d, η_e may be adjusted. One dev pass, choices recorded with values and the dev numbers that justified them, then frozen. The final exam is scored exactly once per arm. Any exam-adjacent adjustment voids the run and is recorded as such.

---

## 6. The differentiator test (owner's explicit ask)

*A test that can distinguish genuine fact-specific learning from another large but answer-uncorrelated residual.*

Three preregistered discriminating measurements, all per-question, all preserved in the record:

1. **Swap test (fact-specificity):** for each exam question about fact i, re-score with fact j's features substituted for fact i's. Genuine fact-specific state ⇒ candidate scores shift **toward fact j's answers** (measurable Δ toward the j-consistent candidate). Preregistered criterion: mean swap-shift ≥ +0.10 log-prob toward the swapped fact's answer. Phase-3's collinear state predicts ≈ 0 shift — the test is discriminative by construction against the measured null.
2. **Margin decomposition:** mean (gold − best-distractor) Δscore ≥ +0.10 AND gold-helped fraction ≥ 2/3 (Phase-3's null: negative margins, 44–61 %).
3. **Rank conversions:** count of rank-gain→correct-answer conversions > 0 on combination questions (Phase-3's null: 0/0/0).

A large residual with any of these failing is the Phase-3 signature and is recorded as the same negative, by name.

---

## 7. Success / failure criteria (fixed now)

**Behavioural success (all required):**
- combination@1 ≥ 0.50 (untaught combinations — constraint 3)
- ablation delta ≥ +0.10 (plastic-state zeroed rerun drops by ≥ this — constraint 4)
- ≥ 2/3 held-out splits pass (constraint 6)

**State criteria (diagnostic, reported regardless):**
- novelty (mean pairwise cosine distance) ≥ 0.10 — owner constraint 1's measurement
- differentiator tests §6 all pass
- retention battery: corrections, 20-old-fact intactness, transcript purge, bitwise restart — all PASS (constraint 5)
- foundation fingerprint bit-identical after the run

**Any behavioural criterion failing = the experiment's negative result, recorded in RESULTS.md §2 by name, with the decomposition data.** A pass on state criteria with a behavioural fail is the Phase-3 signature — recorded as such, not upgraded.

---

## 8. What is NOT authorised under this protocol

| Direction | Status |
|---|---|
| Learn-toward-expression | **design + prereg only (this document); execution awaits owner review** |
| Free-decode pathway | HOLD (owner: risks bypassing rather than solving ownership) |
| Dreaming six-config rerun | HOLD (does not answer the expression failure) |
| Compliance ruling | separate owner gate, untouched |

---

## 9. Preregistration mechanics

This document is frozen (git + SHA-256 noted at commit) before any LTE code executes. The preregistered values (λ_r, λ_d, τ_d, η_e, splits, thresholds) may be amended ONLY before the dev pass, with the amendment committed and the reason recorded. After the final exam is scored: no amendments, no reinterpretations — negatives land in the register by name.

*Open items for owner review:* (a) λ/τ/η defaults; (b) dev-split size; (c) swap-test threshold; (d) whether the expression channel also accrues consolidation eligibility (default proposed: yes — u_fast is u_fast; consolidation law is entry-agnostic); (e) anything the owner wants added or struck before freeze.
