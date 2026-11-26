# LTE-1 — Learn-Toward-Expression Protocol (REVISION 4 — execution still HELD)

**Status:** REVISION 4, resolving the owner's three final findings on Revision 3. **Not yet approved for implementation. Execution remains held — no implementation, no training, no run until the owner signs off.** On sign-off: freeze the protocol, then implement in the owner's order (1) gradient and host-immutability gates → (2) harness and split-manifest verification → (3) controls and calibration → (4) final exam exactly once.

**Revision 4 changes:** (1) Gate 1's noise floor redefined — repeatability noise from repeated IDENTICAL evaluations at fixed ε and state (K = 5), separate from convergence (central differences at all three ε kept for the convergence record); the undefined ε′ relation withdrawn; (2) the §5/§2.3 contradiction removed — novelty is computed from TRAIN-carrier features ONLY, ever; DEV contributes combination accuracy alone (generalisation), never features; an in-harness assertion forbids feature extraction on DEV carriers; (3) calibration rewritten as ONE deterministic nested sequence — same procedure applied identically to both λ_d configurations (fair comparison), λ_r retry handled per grid point inside each configuration, final selection and freeze explicit.

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

**Carrier-source rule (owner, Rev 4 — stated as a hard invariant):** novelty and L_disc are computed from **TRAIN-carrier features only, always** — during training, during calibration, and at exam. **DEV contributes exactly one quantity: combination accuracy@1, as a generalisation evaluation. Feature extraction on DEV carriers is forbidden**, and the harness enforces it with an assertion (any masked-feature-extraction call on a DEV carrier aborts the run as an instrument fault). DEV can evaluate generalisation; it can never become a hidden source of feature diagnostics.

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

## 5. Calibration — ONE deterministic nested sequence (owner finding on Rev 3: ordering ambiguity resolved)

Everything below runs AFTER the gradient gates and host-immutability gates pass (§6.2, §3.1). The identical procedure is applied to BOTH λ_d configurations, so the comparison is fair. Feature extraction never touches DEV carriers (§2.3 hard invariant); the only DEV quantity read anywhere in calibration is dev combination accuracy@1 (plus dev mean gold-margin for tie-breaking).

**Configuration grid:** config A (λ_d = 0.5) and config B (λ_d = 0), identical in everything else.

**Per-configuration procedure (identical for A and B):**
1. For each η_e in grid order {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, with λ_r = 1.0:
   a. Train the dev-configured model (TRAIN data only; consolidation disabled).
   b. Instability check per grid point: train `L_rank` rising over a full pass, or any NaN in the expression loss → retry with λ_r ← 0.5; still unstable → λ_r ← 0.25; still unstable → mark this grid point ABORTED and continue to the next η_e (the point's data is recorded; it is not tuned further).
   c. Score DEV: combination@1 and mean gold-margin. Nothing else is read from DEV.
2. Select η_e*(c) for the configuration: highest dev combination@1; tie-break 1: highest dev mean gold-margin; tie-break 2: smallest η_e. (If all five points aborted: the configuration is void and recorded as an instrument fault.)
3. Record for the selected point: λ_r used, dev combo@1, dev gold-margin, and novelty from TRAIN-carrier features post-teach.

**λ_d decision (after both configurations complete):** retain λ_d = 0.5 (config A) iff `novelty_A ≥ 0.80` AND `dev-combo@1(A) ≥ dev-combo@1(B) − 0.05`; otherwise λ_d = 0. Ties → 0.5.

**Final freeze:** the winning configuration (λ_r, λ_d, η_e) is committed to the calibration report with every grid point's numbers; the primary LTE-1 run then trains with exactly that configuration. **The final exam is scored exactly once per arm.** Any exam-adjacent adjustment voids the run and is recorded as such.

**Budget note:** 2 configurations × 5 η_e points = up to 10 dev trainings (plus λ_r retries). Full LTE-1 block ≈ 24 h wall (uncertainty ±30 %, co-tenant load), as in Phase-3.

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

**Gate 1 — GENUINE numerical derivative comparison (REVISED Rev 4: noise floor from repeatability, not cross-ε differences).**
Let `F = λ_r·L_rank,train + λ_d·L_disc,train` at the current state, and `d = Δu_expr / ‖Δu_expr‖` (the actual planned step direction). Autograd gives `D_AD = ∇F(u)·d`. Numerical central differences give, for ε ∈ {1e-2, 1e-3, 1e-4}:

```
D_FD(ε) = [ F(u + ε·d) − F(u − ε·d) ] / (2ε)
```

**Repeatability noise (the tolerance's basis — owner correction):** at each ε, repeat the IDENTICAL central-difference evaluation at the IDENTICAL state K = 5 times; `n_rep(ε)` = the maximum absolute deviation among the 5 `D_FD(ε)` values, normalised: `n_rep(ε) = max_k |D_FD^{(k)}(ε) − D_FD^{(1)}(ε)| / max(1, |D_AD|)`. This measures evaluation noise, not step-size sensitivity.

**Convergence record (separate from the pass criterion):** central differences at all three ε are kept and reported as the convergence pattern of D_FD across step sizes; they inform interpretation and are recorded, but the pass criterion rests on repeatability-normalised agreement.

**Pass criterion:** `|D_AD − D_FD(ε)| / max(1, |D_AD|) ≤ max(10 · n_rep(ε), 1e-4)` at ε = 1e-2 and ε = 1e-3, where 1e-4 is the separately specified numerical floor. D_AD, every D_FD(ε) (all repeats), every relative error, the per-ε loss values, and the measured n_rep are recorded in the run report. Gate 1 runs on the tiny standin AND once on the real host, before any training.

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

**Revision-2 review (resolved in Rev 3):** DEV-leakage in L_disc carriers · algebraic self-check Gate 1 · swap on DEV carriers · non-deterministic calibration · unverifiable split manifest — all FIXED in Revision 3, confirmed by the owner's Rev-3 review.

**Revision-3 review (resolved in THIS revision):**

| Owner finding | Disposition in Revision 4 |
|---|---|
| 1. n(ε) undefined at grid top; cross-ε differences measure step-size sensitivity, not noise | **FIXED** — repeatability noise from K = 5 identical evaluations at fixed ε and state; central differences at all three ε retained for the convergence record; tolerance = max(10·n_rep(ε), 1e-4 floor), §6.2 |
| 2. §5 "novelty on DEV" contradicted §2.3's no-DEV-diagnostics rule | **FIXED** — novelty/L_disc from TRAIN carriers only, always; DEV contributes combination accuracy@1 (+ gold-margin tie-break) alone; harness assertion aborts any DEV feature extraction, §2.3, §5 |
| 3. Calibration composition ambiguous (η_e × λ_d nesting, λ_r independence, final selection) | **FIXED** — one deterministic nested sequence: identical per-configuration procedure for A and B, λ_r retry ladder per grid point inside each configuration, η_e*(c) selection rule, explicit λ_d decision, final freeze, §5 |

**Implementation order on owner sign-off (owner's order, recorded):** (1) gradient and host-immutability gates → (2) harness and split-manifest verification → (3) controls and calibration → (4) final exam exactly once.
