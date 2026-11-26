# LTE-1 — Learn-Toward-Expression Protocol (REVISION 2 — post owner review, execution still HELD)

**Status:** REVISION 2, incorporating the owner's review of 2026-11-26 in full. Design direction: ACCEPTED. **Execution remains explicitly held — no implementation, no training, no run until the owner reviews this corrected protocol.**

**Revision 2 changes:** (1) gradient-sign correction — expression update is DESCENT on the full objective, with the 0.7 blend Jacobian entering only through the actual forward computation; (2) L_disc gradient included; (3) finite-difference verification of the new gradient required; (4) five open items decided by the owner and recorded; (5) novelty/cosine-hinge relation made exact; (6) swap test given an exact score, candidate construction and sign convention; (7) dev split fully specified; (8) host-immutability strengthened to the four-point test; (9) gradient-direction test added as a run gate; (10) consolidation DISABLED in LTE-1 (owner decision).

**Central requirement (owner verbatim):** *Learning a fact must create a distinguishable representation that can subsequently move the correct answer ahead of its distractors on unseen questions.*

**Design law (owner verbatim):** *The learning objective and the expression objective must describe the same computation. We do not bolt another readout onto a representation that has never learned to carry the information that readout needs.*

---

## 1. What the four experiments measured, and what this protocol changes

| Established (Phase 1/2, interventions, Phase-3) | Consequence for design |
|---|---|
| Plastic state acquires, stores, corrects, retains, restarts bitwise | Keep the entire ownership/retention apparatus unchanged |
| Transmitted residual is large (‖γ·r‖ 7.2–10.4) but answer-uncorrelated (gold helped 44–61 % ≈ coin-flip; margins negative; rank→correct 0/0/0) | The deficit is *what the signal encodes*; the learning signal must be derived at the point of answer selection |
| Per-fact features nearly collinear (novelty 0.0013–0.0071); density multiplied signal without decorrelating | Discrimination must be a property of the objective |
| The current objective (`connection_prediction_mse`) never references the decoder | The expression objective enters the update law itself |

**The change, stated plainly:** the plastic update law gains an **expression-descent term** whose gradient is computed at the frozen host's candidate selection and flows back through the frozen host (gradients only — zero host weight updates) into the plastic state. The existing prediction term is untouched.

---

## 2. Objective and equations (REVISED — sign corrected)

### 2.1 The two channels of the update

Prediction channel — the corrected plasticity law, **exactly as frozen in baseline v2, unchanged**:

```
Δu_pred = 0.7 · η_p · δ_pred · ψ
```

Expression channel — **gradient DESCENT** on the weighted expression objective:

```
Δu_expr = − η_e · ∇_{u_fast} ( λ_r·L_rank + λ_d·L_disc )
```

Combined update:

```
u_fast ← u_fast + Δu_pred + Δu_expr
```

**Jacobian note (owner):** the gradient ∇_{u_fast} is taken through the *effective connection* — the actual forward computation, in which the effective strength is `a = 0.7·u_fast + 0.3·θ_slow`. The 0.7 blend therefore enters the gradient **naturally, through the forward's Jacobian**. The expression path must NOT multiply the gradient by another 0.7 factor.

### 2.2 The ranking loss (answer alignment — constraint 2)

At the answer position, for a training question with candidate set {c_1..c_K} (gold = g):

```
L_rank = − log softmax_g( s_1 .. s_K )
s_k    = candidate logit from the frozen host's scoring of c_k with the plastic residual injected
```

CE-over-candidates is chosen because it is the *same computation* as the Phase-2/3 scoring instrument — the loss and the exam measure the same quantity by construction.

### 2.3 The discrimination term (fact discrimination — constraint 1)

Definitions first, so the two discrimination measures cannot disagree:

- `cos(f_i, f_j) = ⟨f_i, f_j⟩ / (‖f_i‖·‖f_j‖)`, computed in float32 from BF16 features; zero-norm features give cos = 0 (distance 1).
- **Per-fact feature** `f_i`: the γ-scaled plastic residual contributed at the answer position by fact i's connection set only — extracted by a masked forward in which every connection not belonging to fact i contributes zero (masking, not retraining), on fact i's canonical dev-bank question. Identical construction for all facts.
- **Novelty** (the Lane G diagnostic, exact definition): `novelty = mean_{i≠j} ( 1 − cos(f_i, f_j) )`.

The training pressure is a pairwise hinge:

```
L_disc = mean_{i≠j} max( 0, cos(f_i, f_j) − τ_d )        τ_d = 0.20 (owner: provisionally fixed)
```

**Exact relation between the two measures (owner item A):** with `cos̄ = mean_{i≠j} cos(f_i,f_j)`, novelty `= 1 − cos̄` identically. Since `(·)⁺` is convex, `L_disc = mean (cos − τ_d)⁺ ≥ (cos̄ − τ_d)⁺`. Therefore:

- `L_disc = 0` ⟺ every pair has cos ≤ τ_d ⟹ `novelty ≥ 1 − τ_d = 0.80`. *(one-way implication — a high novelty does NOT imply every pair is separated)*
- The preregistered **discrimination criterion is `L_disc = 0` on exam-extracted features** (the trained quantity, pairwise-strict); **novelty ≥ 0.80 is recorded alongside it as the exact mean-form consequence**. One mechanism, two readings, relation proven — they cannot disagree silently.

### 2.4 Total objective and what does NOT change

```
L = L_pred + λ_r·L_rank + λ_d·L_disc
```

Untouched: ψ, the forward map, dual-timescale structure, the 0.7 factor inside Δu_pred, the consolidation law itself, the γ-sequencing law (plasticity first, gate learning second), rate ≤ 0 hard-disable, HostMode authority, the gym zero-direct-write law, and the frozen-fingerprint foundation instrument.

**Consolidation (owner decision):** consolidation is **DISABLED during LTE-1** (rate 0 hard-disable path). The existing law stays intact everywhere else; its interaction with LTE is a LATER, separately preregistered experiment. LTE-1 isolates one variable: the learning objective.

**Protected-property note:** the update law is extended by an ADDITIVE descent term with its own coefficient; Δu_pred is byte-identical to baseline v2. ψ is unmodified, so the ψ-independence scope condition is unaffected. This amendment stands or falls by LTE-1's own gates.

---

## 3. Proposed implementation (no code written yet)

| Piece | Source | Change |
|---|---|---|
| Harness | `aion/experiments/teacher_student.py` | new `--lte` mode; Phase-1/2/3 paths untouched |
| Candidate scoring | exists (Phase-2/3) | batched (K candidates per forward) |
| Backward through frozen host | new | autograd pass; `requires_grad` on the injected residual only |
| Host-immutability guard | strengthened | four-point test, §3.1 |
| Novelty / L_disc extraction | new (extends Lane G machinery) | masked forwards, §2.3 definitions |
| Swap test | new | exact score, §6.1 |
| Gradient verification | new | finite-difference + direction tests, §6.2 |
| Question splits | existing world bank | train/dev/exam, §6.3 |

**Budget estimate (honest, CPU BF16, Lappy):** expression channel adds forward+backward per training question (~8–10 s per 6-candidate batch): teach ≈ 3.0–3.6 ks/config; exam incl. ablation rerun, swap tests, gradient tests ≈ 2.5–3 h. Per-config ≈ 6 h; the full LTE-1 block (controls + arm + analyses) ≈ 12–14 h, wall uncertainty ±30 % (co-tenant load), as in Phase-3.

### 3.1 Host immutability — the four-point test (owner item D; `host_grad_count == 0` alone is NOT sufficient)

A frozen host legitimately transmits gradients to the injected residual while its parameters receive no updates. Required, all four:

1. **Host parameters have no optimizer.** No optimizer object is constructed over any foundation parameter.
2. **Host parameter fingerprints are identical before and after** the run (the existing 426-tensor fingerprint instrument).
3. **No host parameter receives an update** — per-phase check that each foundation tensor's data is unchanged (fingerprint re-verified at teach-end and exam-end, not just run-end).
4. **Gradients to the plastic state are finite and nonzero where expected** — assert `isfinite(Δu_expr).all()` and `‖Δu_expr‖ > 0` on expression steps; a zero or NaN expression gradient is a run-aborting instrument fault, not a silent skip.

`host_grad_count == 0` is retained as an additional check, never as the definition of immutability.

---

## 4. Controls (constraint 6: frozen evaluation)

| Control | Purpose |
|---|---|
| A — frozen floor | the Phase-3 floor, re-measured in-session |
| B — retrieval | the structural ceiling for transfer; blind to composition (0.000) |
| C — old objective, same harness | **the null that must replicate the Phase-3 floor** — if C moves, the instrument changed, and the comparison is void (instruments-before-science law) |

---

## 5. Tuning law — owner decisions recorded (no tuning against the final exam)

| Item | Owner decision |
|---|---|
| λ_r | **1.0 starting value; dev-only adjustment** |
| λ_d | **0.5 starting value; retained only if dev evidence justifies it** |
| τ_d | **0.20 cosine-similarity threshold, provisionally fixed** |
| η_e | **specified and frozen after dev calibration** |
| Dev split | **explicitly defined before execution** (§6.3) |
| Swap threshold | **+0.10 retained, with the precise scoring definition of §6.1** |
| Consolidation eligibility | **NO — excluded from LTE-1** (owner: isolating one variable; consolidation interaction is a later, separately preregistered experiment) |

Dev questions are the ONLY surface where λ_r, λ_d, η_e may be adjusted. One dev pass, choices recorded with the values and the dev numbers that justified them, then frozen. The final exam is scored exactly once per arm. Any exam-adjacent adjustment voids the run and is recorded as such.

---

## 6. Tests and splits (owner items B and C; owner §4)

### 6.1 Swap test — exact definition (owner item B)

For each ordered fact pair (i, j), i ≠ j, sharing a relation template, with gold answers a_i and a_j:

- Carrier question: Q_i (the question whose gold is a_i, from the DEV bank).
- Candidates scored: the pair {a_i, a_j}.
- **Native scoring:** `s_{f_i}(c)` = candidate score with the full native plastic state.
- **Swapped scoring:** `s_{f_j}(c)` = candidate score with fact i's per-fact residual block REPLACED by fact j's (masked-forward machinery as §2.3 — swap is a state substitution at exam time, no retraining).

```
Δ_swap(i, j) = [ s_{f_j}(a_j) − s_{f_j}(a_i) ]  −  [ s_{f_i}(a_j) − s_{f_i}(a_i) ]
```

**Sign convention:** Δ_swap > 0 means j's features favour j's answer over i's answer, *relative to* how i's features score that pair — i.e. the state carries fact-specific information. The collinear Phase-3 null predicts Δ_swap ≈ 0 (features-i ≈ features-j ⇒ the two brackets cancel).

**Preregistered criterion:** `mean over ordered pairs Δ_swap ≥ +0.10` log-prob (owner: threshold retained). The score s is the candidate's log-probability at the answer position under the standard scoring instrument. All swap measurements are preserved per-pair in results.jsonl.

### 6.2 Gradient verification — two run gates (owner §1 and §4)

**Gate 1 — finite-difference check of the new gradient.** On the tiny standin AND once on the real host before training: compute Δu_expr at the current state; numerically verify the directional derivative agreement

```
⟨ ∇L, Δu_expr ⟩  ≈  −(1/η_e) · ‖Δu_expr‖²     (relative error ≤ 1e-4, float32 reference)
```

and that `L(u_fast + ε·Δu_expr) < L(u_fast)` for ε ∈ {1e-2, 1e-3}. This verifies BOTH the sign (descent, not ascent) and the connection to the candidate-scoring computation, numerically — not by asserting autograd ran.

**Gate 2 — gradient-direction test (owner §4, verbatim procedure).**
1. Capture the expression loss before an update.
2. Calculate the proposed plastic update.
3. Apply a small update using the expression gradient alone.
4. Recalculate the same loss with everything else fixed.
5. Verify that the loss decreases.

Failure of either gate aborts the run before any training. Gate results are recorded in the run report.

### 6.3 Question splits (owner item C — question-level distinctness)

One RNG seed fixed at preregistration: **seed 20261126**. The split is generated by the world bank's template allocator with disjoint allocation, verified by an in-harness assertion (no question, template, or combination appears in two splits), and committed as a manifest before any run:

| Split | Count | Content |
|---|---|---|
| TRAIN (teach) | 60 experiences | 20 relationships × 3 teach facts — the existing teach set |
| DEV | 12 questions | 8 direct (untaught question-templates) + 4 untaught combinations — the ONLY tuning surface (§5) |
| FINAL EXAM | 43 questions | the Phase-3 structure: direct + paraphrase + 15-question combination battery; all unseen; held-out splits re-derived per the LTE seed and frozen before teach; combination questions are unseen combinations of individually taught facts |

Rules: (a) every exam direct/paraphrase question uses a question-template never used in teach or dev; (b) exam combinations and dev combinations are disjoint pairs; (c) the split manifest (question ids, templates, seeds) is committed alongside this protocol's freeze; (d) the final exam is scored exactly once per arm.

---

## 7. Success / failure criteria (fixed now — unchanged from draft except where owner-tightened)

**Behavioural success (all required):**
- combination@1 ≥ 0.50 (untaught combinations — constraint 3)
- ablation delta ≥ +0.10 (plastic-state zeroed rerun drops by ≥ this — constraint 4)
- ≥ 2/3 held-out splits pass (constraint 6)

**State criteria (diagnostic, reported regardless):**
- discrimination: **L_disc = 0 on exam-extracted features** under τ_d = 0.20; **novelty ≥ 0.80** recorded as its exact mean-form consequence (§2.3)
- differentiator tests: swap ≥ +0.10 (§6.1); margin decomposition gold Δmargin ≥ +0.10 with gold-helped ≥ 2/3; rank-gain→correct conversions > 0
- retention battery: corrections, 20-old-fact intactness, transcript purge, bitwise restart — all PASS (constraint 5)
- host immutability four-point test: PASS (§3.1)
- gradient gates 1 and 2: PASS before training (§6.2)

**Any behavioural criterion failing = the experiment's negative result, recorded in RESULTS.md §2 by name, with the full decomposition data.** A pass on state criteria with a behavioural fail is the Phase-3 signature — recorded as such, never upgraded.

---

## 8. What is NOT authorised under this protocol

| Direction | Status |
|---|---|
| Learn-toward-expression | **design + prereg (this revision); execution awaits owner review of THIS corrected protocol** |
| Free-decode pathway | HOLD (owner: risks bypassing rather than solving ownership) |
| Dreaming six-config rerun | HOLD (does not answer the expression failure) |
| Compliance ruling | separate owner gate, untouched |
| Consolidation interaction with LTE | later, separately preregistered experiment (owner decision) |

---

## 9. Review ledger (owner review 2026-11-26 → this revision)

| Owner review item | Disposition in Revision 2 |
|---|---|
| Gradient sign (ascent error) | **FIXED** — Δu_expr = −η_e·∇(λ_r·L_rank + λ_d·L_disc), §2.1 |
| L_disc gradient missing | **FIXED** — included in the same descent step, §2.1 |
| No extra 0.7 on the expression gradient; blend Jacobian via the forward | **RECORDED as a design law**, §2.1 |
| Finite-difference verification required | **ADDED** as run Gate 1, §6.2 |
| λ_r = 1.0 / λ_d = 0.5 / τ_d = 0.20 / η_e post-dev | **RECORDED**, §5 |
| Dev split defined before execution | **SPECIFIED**, §6.3 |
| Swap threshold +0.10 with precise definition | **SPECIFIED** — exact score, candidates, sign, §6.1 |
| Consolidation eligibility: NO | **APPLIED** — consolidation disabled in LTE-1, §2.4 |
| Novelty ↔ hinge relation | **PROVEN** — novelty ≡ 1 − cos̄; L_disc = 0 ⟹ novelty ≥ 1 − τ_d; criterion restated as L_disc = 0 with novelty ≥ 0.80 recorded, §2.3 |
| Swap test precision | **SPECIFIED**, §6.1 |
| Host immutability four-point test | **ADOPTED**, §3.1 |
| Gradient-direction test | **ADDED** as run Gate 2, §6.2 |

*Remaining open item: none — all five draft open items are owner-decided. This revision awaits the owner's final review before freeze and any implementation.*
