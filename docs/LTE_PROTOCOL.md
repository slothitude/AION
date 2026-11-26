# LTE-1 — Learn-Toward-Expression Protocol (REVISION 3 — execution still HELD)

**Status:** REVISION 3, resolving the owner's five review findings on Revision 2. **Not yet approved for implementation. Execution remains held — no implementation, no training, no run until the owner's final review.** On sign-off, implementation proceeds in the owner's order: (1) gradient and host-immutability gates → (2) harness and split-manifest verification → (3) controls and calibration → (4) final exam exactly once.

**Revision 3 changes:** (1) DEV-leakage eliminated — training-time discrimination uses TRAIN carriers only; DEV is calibration-only; exam features are extraction-only at final evaluation; (2) Gate 1 replaced with a genuine numerical central-difference comparison against autograd, tolerance justified by measured noise floor; (3) swap-test criterion moved to held-out EXAM carriers, post-freeze; (4) calibration made deterministic (grid, metric, tie-breaks); (5) split-manifest freeze mechanics specified (path + commit SHA recorded at implementation step 2, before any run).

---

## 1. What the four experiments measured, and what this protocol changes

| Established (Phase 1/2, interventions, Phase-3) | Consequence for design |
|---|---|
| Plastic state acquires, stores, corrects, retains, restarts bitwise | Keep the entire ownership/retention apparatus unchanged |
| Transmitted residual is large (‖γ·r‖ 7.2–10.4) but answer-uncorrelated (gold helped 44–61 % ≈ coin-flip; margins negative; rank→correct 0/0/0) | The deficit is *what the signal encodes*; the learning signal must be derived at the point of answer selection |
| Per-fact features nearly collinear (novelty 0.0013–0.0071); density multiplied signal without decorrelating | Discrimination must be a property of the objective |
| The current objective (`connection_prediction_mse`) never references the decoder | The expression objective enters the update law itself |

---

## 2. Objective and equations

### 2.1 The two channels of the update

Prediction channel — the corrected plasticity law, **exactly as frozen in baseline v2, unchanged**:

```
Δu_pred = 0.7 · η_p · δ_pred · ψ
```

Expression channel — **gradient DESCENT** on the weighted TRAINING expression objective:

```
Δu_expr = − η_e · ∇_{u_fast} ( λ_r · L_rank,train + λ_d · L_disc,train )
u_fast  ← u_fast + Δu_pred + Δu_expr
```

Both loss terms in the gradient are computed on **TRAIN data only** (§6.3): L_rank,train on training questions and their candidate sets; L_disc,train on per-fact features extracted from TRAIN carrier questions. The gradient flows through the plastic forward and the frozen host (gradients only — zero host weight updates).

**Jacobian law (owner):** ∇_{u_fast} is taken through the *effective connection* — the actual forward, in which `a = 0.7·u_fast + 0.3·θ_slow`. The 0.7 blend enters the gradient naturally through the forward's Jacobian. The expression path is NEVER multiplied by another 0.7.

### 2.2 The ranking loss (answer alignment — constraint 2)

At the answer position, for a training question with candidate set {c_1..c_K} (gold = g):

```
L_rank,train = − log softmax_g( s_1 .. s_K )
s_k          = candidate logit from the frozen host's scoring of c_k with the plastic residual injected
```

CE-over-candidates: the loss and the exam's scoring instrument are the same computation by construction.

### 2.3 Discrimination — two strictly separated contexts (REVISED per owner finding 1)

Definitions (unchanged): `cos(f_i,f_j)` in float32 (zero-norm ⇒ cos = 0); per-fact feature = γ-scaled plastic residual at the answer position from fact i's connection set only (masked forward); `novelty = mean_{i≠j}(1 − cos(f_i,f_j))`; hinge `L_disc = mean_{i≠j} max(0, cos(f_i,f_j) − τ_d)`, τ_d = 0.20.

**The two contexts are now explicitly distinct and never mixed:**

| Context | Carrier questions | When | Role |
|---|---|---|---|
| `f_i^train` / `L_disc,train` | **TRAIN carriers** (fact i's canonical teach-question) | during training, inside the gradient | learning pressure |
| `f_i^exam` / `L_disc,exam` | fact i's **exam** direct question | **final evaluation only**, state frozen, **no update taken** | diagnostic criterion |

DEV carriers are used for **calibration only** (§5) — never inside any gradient, never as a diagnostic source. The exam diagnostic is extracted after the final exam is scored, from the frozen end-state, and no update follows it.

**Exact relation between the measures (owner item A, Rev 2, unchanged):** novelty ≡ 1 − cos̄; since `(·)⁺` is convex, `L_disc ≥ (cos̄ − τ_d)⁺`; therefore `L_disc = 0` ⟺ every pair cos ≤ τ_d ⟹ novelty ≥ 1 − τ_d = 0.80 (one-way implication). **Preregistered discrimination criterion: `L_disc,exam = 0`; `novelty(exam features) ≥ 0.80` recorded as its exact mean-form consequence.** The training-time analogue (`L_disc,train → 0` on train carriers) is reported as the learning curve, never as the criterion.

### 2.4 Total objective and what does NOT change

```
L = L_pred + λ_r·L_rank,train + λ_d·L_disc,train
```

Untouched: ψ, the forward map, dual-timescale structure, the 0.7 factor inside Δu_pred, the consolidation LAW (disabled for this run only), the γ-sequencing law, rate ≤ 0 hard-disable, HostMode authority, gym zero-direct-write, the frozen-fingerprint instrument.

**Consolidation (owner decision, Rev 2, confirmed):** DISABLED during LTE-1 (rate 0 hard-disable). Interaction with LTE is a later, separately preregistered experiment.

**Protected-property note:** the update law gains an ADDITIVE descent term; Δu_pred is byte-identical to baseline v2; ψ unmodified (ψ-independence scope condition unaffected). The amendment stands or falls by LTE-1's own gates.

---

## 3. Proposed implementation (no code written yet)

| Piece | Source | Change |
|---|---|---|
| Harness | `aion/experiments/teacher_student.py` | new `--lte` mode; Phase-1/2/3 paths untouched |
| Candidate scoring | exists (Phase-2/3) | batched (K candidates per forward) |
| Backward through frozen host | new | autograd; `requires_grad` on injected residual only |
| Host-immutability guard | strengthened | four-point test, §3.1 |
| Feature extraction (train/exam) | new | masked forwards per §2.3, per-context |
| Swap test | new | §6.1, exam carriers, post-freeze |
| Gradient gates | new | §6.2, genuine FD comparison |
| Split generator + manifest | new | §6.3, committed before any run |

**Budget estimate (honest, CPU BF16, Lappy):** teach ≈ 3.0–3.6 ks/config; exam + ablation + swap + gates ≈ 2.5–3 h per config. The deterministic λ_d decision (§5) requires TWO dev-configured trainings (λ_d = 0.5 vs λ_d = 0): add ≈ 6 h. Full LTE-1 block ≈ 18–20 h wall, uncertainty ±30 % (co-tenant load), as in Phase-3.

### 3.1 Host immutability — four-point test (owner item D; unchanged from Rev 2)

1. Host parameters have no optimizer (no optimizer object constructed over any foundation parameter).
2. Host parameter fingerprints identical before and after the run (426-tensor instrument).
3. No host parameter receives an update — fingerprints re-verified at teach-end AND exam-end.
4. Gradients to the plastic state are finite and nonzero where expected: `isfinite(Δu_expr).all()` and `‖Δu_expr‖ > 0` on every expression step; zero/NaN = run-aborting instrument fault.

`host_grad_count == 0` retained as an additional check, never the definition.

---

## 4. Controls (constraint 6)

| Control | Purpose |
|---|---|
| A — frozen floor | the Phase-3 floor, re-measured in-session |
| B — retrieval | the structural transfer ceiling; blind to composition (0.000) |
| C — old objective, same harness | **must replicate the Phase-3 floor** — if C moves, the instrument changed and the comparison is void (instruments-before-science law) |

---

## 5. Calibration — DETERMINISTIC procedure (owner finding 4; replaces all discretionary adjustment)

**η_e selection (single pass over a preregistered grid, in order):**
- Grid: `η_e ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`.
- Selection metric: highest **dev combination accuracy@1** on the 4 dev-combination questions.
- Tie-break 1: highest dev mean gold-margin. Tie-break 2: smallest η_e.
- All five results recorded in the calibration report; η_e frozen at the winner; **no other values considered**.

**λ_r rule:** start 1.0. Adjusted only on measured training instability, defined as: train `L_rank` increasing over a full training pass, or any NaN in the expression loss. If triggered: λ_r ← 0.5, retry; still unstable → λ_r ← 0.25, retry; still unstable → run aborts as an instrument fault (recorded, not tuned further).

**λ_d rule (deterministic two-config dev comparison, cost noted in §3):**
1. Train two dev-configured models, identical in everything except λ_d: config A (λ_d = 0.5), config B (λ_d = 0).
2. Measure on DEV: novelty of train-carrier features post-teach (for B this is the un-pressured baseline) and dev combination@1.
3. **Retain λ_d = 0.5 iff novelty_A ≥ 0.80 AND dev-combo@1(A) ≥ dev-combo@1(B) − 0.05.** Otherwise λ_d = 0. Ties → 0.5.
4. Both configs' numbers recorded; decision and justification frozen before the final exam.

**Freeze:** after calibration, η_e, λ_r, λ_d, and the chosen config are committed to the calibration report; the final exam is scored exactly once per arm; any exam-adjacent adjustment voids the run and is recorded as such.

---

## 6. Tests and splits

### 6.1 Swap test — EXAM carriers, post-freeze (owner finding 3)

For each ordered fact pair (i, j), i ≠ j, sharing a relation template, golds a_i, a_j:

- **Carrier: the held-out EXAM direct question for fact i** — measured during final evaluation only, after calibration is frozen. **Never used for tuning.** (Rev-2's dev-bank carrier wording is withdrawn.)
- Candidates scored: the pair {a_i, a_j} — the same two candidates under both states.
- Native scoring `s_{f_i}`: full native plastic state.
- Swapped scoring `s_{f_j}`: fact i's per-fact residual block REPLACED by fact j's; **all other state identical** (masked-forward substitution, no retraining).

```
Δ_swap(i, j) = [ s_{f_j}(a_j) − s_{f_j}(a_i) ] − [ s_{f_i}(a_j) − s_{f_i}(a_i) ]
```

Sign: Δ_swap > 0 ⇒ j's features favour j's answer relative to how i's features do ⇒ fact-specific state. The collinear Phase-3 null predicts ≈ 0. **Criterion: mean over ordered pairs ≥ +0.10 log-prob**, score = candidate log-probability at the answer position under the standard instrument, all per-pair values preserved in results.jsonl. A train-carrier swap diagnostic may additionally be logged during development, but carries NO criterion weight.

### 6.2 Gradient verification — two run gates (owner finding 2 + §4 of prior review)

**Gate 1 — GENUINE numerical derivative comparison (Rev 3: no longer an algebraic self-check).**
Let `F = λ_r·L_rank,train + λ_d·L_disc,train` at the current state, and `d = Δu_expr / ‖Δu_expr‖` (the actual planned step direction). Autograd gives `D_AD = ∇F(u)·d`. Numerical central differences give, for ε ∈ {1e-2, 1e-3, 1e-4}:

```
D_FD(ε) = [ F(u + ε·d) − F(u − ε·d) ] / (2ε)
```

**Tolerance is measured, not assumed:** before training, a tolerance-calibration run on the real host measures the FD noise floor `n(ε) = |D_FD(ε) − D_FD(ε')| / max(1, |D_AD|)` (ε' = next-larger grid ε). **Pass criterion: `|D_AD − D_FD(ε)| / max(1, |D_AD|) ≤ max(10·n(ε), 1e-4)` at ε = 1e-2 and ε = 1e-3.** D_AD, every D_FD(ε), every relative error, and the loss values at each ε are recorded in the run report. Gate 1 runs on the tiny standin AND once on the real host, before any training.

**Gate 2 — gradient-direction test (independent step size and isolation, per owner):**
1. Capture F before the update.
2. Compute Δu_expr.
3. Apply `u_fast ← u_fast + η_gate · Δu_expr` with **η_gate = 0.1·η_e** (recorded), **expression gradient alone** — prediction channel, gates, and all other state frozen for the duration of the check (state isolation).
4. Recompute F with the identical candidate batch in identical order, float32 reference.
5. **Verify F decreased.**
Failure of Gate 1 or Gate 2 aborts the run before training; gate results are recorded.

### 6.3 Question splits (owner item C, Rev 2; manifest mechanics per finding 5)

One RNG seed fixed at preregistration: **seed 20261126**. Template-disjoint allocation (no question, template, or combination in two splits), verified by an in-harness assertion whose output is part of the manifest:

| Split | Count | Content |
|---|---|---|
| TRAIN (teach) | 60 experiences | 20 relationships × 3 teach facts — also the L_rank,train / L_disc,train surface |
| DEV | 12 questions | 8 direct + 4 untaught combinations — calibration ONLY (§5) |
| FINAL EXAM | 43 questions | Phase-3 structure: direct + paraphrase + 15-combination battery; all unseen; held-out splits re-derived per the LTE seed and frozen before teach; combinations are unseen combinations of individually taught facts |

**Split-manifest freeze mechanics (owner finding 5):** the manifest does not yet exist — it is produced by the split generator at implementation **step 2** (owner's order), and is **committed to the repository BEFORE any training run**. It contains: question ids, template ids, combination pairs, the generation seed, and the disjointness-assertion output. This protocol's freeze record (a final commit on owner sign-off) will cite the manifest's repository path and exact commit SHA; a manifest not demonstrably asserting disjointness voids the run.

---

## 7. Success / failure criteria (fixed)

**Behavioural success (all required):** combination@1 ≥ 0.50 · ablation delta ≥ +0.10 · ≥ 2/3 held-out splits pass.

**State criteria (diagnostic, reported regardless):** `L_disc,exam = 0` with novelty(exam) ≥ 0.80 (§2.3) · swap ≥ +0.10 on exam carriers (§6.1) · margin decomposition (gold Δmargin ≥ +0.10, gold-helped ≥ 2/3) · rank conversions > 0 · retention battery all-PASS (corrections, 20-old-fact intactness, transcript purge, bitwise restart) · host four-point test PASS · gradient Gates 1–2 PASS before training.

**Any behavioural criterion failing = the experiment's negative result, recorded in RESULTS.md §2 by name with full decomposition data.** State-pass + behavioural-fail is the Phase-3 signature — recorded as such, never upgraded.

---

## 8. What is NOT authorised

| Direction | Status |
|---|---|
| LTE-1 | **design + prereg (this revision); implementation awaits the owner's final review of Revision 3** |
| Free-decode pathway | HOLD |
| Dreaming six-config rerun | HOLD |
| Compliance ruling | separate owner gate |
| Consolidation × LTE interaction | later, separately preregistered |

---

## 9. Review ledger

| Owner review item (Rev-2 review) | Disposition in Revision 3 |
|---|---|
| 1. L_disc consumed DEV (critical) | **FIXED** — training features from TRAIN carriers; DEV calibration-only; exam features extraction-only post-scoring; L_disc,train vs L_disc,exam distinguished, §2.3 |
| 2. Gate 1 was an algebraic self-check (critical) | **FIXED** — genuine central-difference D_FD vs autograd D_AD at three ε; both values, relative errors, loss values recorded; tolerance from measured noise floor; loss-decrease retained; §6.2 |
| 3. Swap carrier wording (DEV → criterion contamination) | **FIXED** — exam carriers, post-freeze, never tuned on; same carrier + same candidates under both states; only fact i's block replaced, §6.1 |
| 4. Calibration rule not deterministic | **FIXED** — η_e grid + metric + tie-breaks; λ_r instability rule with abort; λ_d two-config rule with thresholds; §5 |
| 5. Split manifest unverifiable | **SPECIFIED** — manifest born at step 2, committed before any run, freeze record cites path + SHA; §6.3 |

**Implementation order on owner sign-off (owner's order, recorded):** (1) gradient and host-immutability gates → (2) harness and split-manifest verification → (3) controls and calibration → (4) final exam exactly once.
