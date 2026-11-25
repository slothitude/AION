# Phase 3 Protocol — Closing the Expression Gap (DRAFT FOR OWNER REVIEW)

**Status:** DRAFT protocol for owner review. Nothing in this document has been run. Every owner-decision point is marked **[D1]–[D9]** and is OPEN. No code is changed by this document; the code citations below describe where a knob *would* go if the owner approves.
**Sources (read-only, all numbers copied from them):**
- Phase-2 measured results: `runs/teacher_student_p2/p2-qwen-c5-s0-022716/report.json` (Qwen3.5-4B-Base @ `/mnt/seagate/aion-models/Qwen3.5-4B-Base`, CPU BF16 frozen, seed 0, 5 chains / 20 relationships / 60 experiences; foundation bit-identical after the full run, 426 params fingerprinted)
- Phase-2 harness (the file a Phase-3 knob would extend): `aion/experiments/teacher_student.py`
- γ-residual pathway: `aion/relational/interface.py`
- Register: `docs/RESULTS.md` §2/§3, §5; spec: `docs/APC.md` §6 (learning contract), §9 (change control)

---

## 1. The question

**Can the expression gap be closed by training density, by residual↔candidate alignment, or by both — measured under the same preregistered A–E instrument on real Qwen?**

Phase 2's verdict (owner-corrected): **negative for expression, positive for localisation.** The plastic knowledge exists and the γ pathway is the right place to look, but the knowledge is logit-invisible or sub-threshold:

- **C** (plastic, γ=0) and **D** (plastic + question format, γ=0) are **bitwise identical to A** on every scored row in every provenance mode — knowledge stored (`u_fast` norms after: C `[0.768956, 0.560119, 0.171069]`, D `[1.184052, 1.123644, 0.528366]`; mean fact plastic strength C 0.683863, D 1.047001) with **zero** effect on any score (`plastic_ablation_delta` = 0.0 for C/D/E on both paraphrase and combination, `report.json` `headline_comparison`).
- **E** (plastic + question + γ, gate_lr 100.0, 80 gate steps) trained the gate to **γ = [10.371428, 11.180095, 0.522552]** — the pathway is alive — but expression is sub-threshold: mean gold rank moved at most **0.1** (direct 3.15 CONSOLIDATED → 3.05 FULL / 3.10 PLASTIC) and gold probability moved by ±0.007 (combination 0.35213 CONSOLIDATED vs 0.34747 PLASTIC; paraphrase 0.17455 CONSOLIDATED vs 0.18656 PLASTIC, mixed direction). Combination acc@1 stayed 0.3333 in every mode — exactly A's frozen number (the frozen base itself; combination mean_chance is 0.4, so 0.3333 is *below* chance, not knowledge).

The Phase-2 negative result identifies exactly two levers: **(a) TRAINING DENSITY** — plasticity gets 1 pass per experience in the dialogue teach path; **(b) RESIDUAL↔CANDIDATE ALIGNMENT** — at γ≈10 the residual moves candidate rankings ~0.1 rank; it may not reach the candidate-token embeddings at the answer position usefully.

**THE SUCCESS CONDITION (owner law for Phase 3): BEHAVIOURAL.** AION answers correctly **from its own plastic state on unseen tasks**. NOT `u_fast` growth, NOT γ magnitude, NOT logits moving. If those grow without behavioural accuracy, Phase 3 fails its own test. The decisive measurement: plastic knowledge beats retrieval on unseen composition questions — B's 0.00 combination score is the target, but beating zero alone is NOT enough; improvement must repeat across held-out combinations, with paraphrase and direct reported as **safeguards**. Additionally measured in every arm: the **relationship between residual magnitude, candidate-rank movement, and actual answer accuracy** — this distinguishes alignment that improves useful discrimination from alignment that merely moves logits around.

---

## 2. LEVER A — DENSITY: the k-passes knob

### 2.1 The knob spec

Today plasticity gets exactly **one** NNFS pass per experience in the dialogue teach loop (`aion/experiments/teacher_student.py`, `VariantsRun.teach_variant`, lines 1157–1221):

- Declarative experiences (20): `student.learn_fact(e["fact"])` (lines 1176–1178) → `Student.learn_fact` runs `nnfs_update` once (`repeats` defaults to 1; the `repeats` parameter **already exists** in `aion/dialogue/student.py` lines 81–94 and is never passed).
- QA experiences (40, variants D/E): `iface.nnfs_update(ids)` once per experience (lines 1179–1182).

Measured cost: D ran 60 nnfs updates in `teach_s` 314.6 s ≈ **5.2 s/update** on CPU BF16 (Lappy-class). C ran 20 updates in 134.8 s — consistent.

**Where the knob goes (spec only — no edit now):** a `--p2-k-passes k` CLI argument beside the existing Phase-2 knobs (`--p2-gate-lr`, `--p2-gate-steps`, lines 1604–1613), consumed in `teach_variant`'s experience loop: the QA branch repeats `iface.nnfs_update(ids)` k times; the declarative branch calls `student.learn_fact(fact, repeats=k)`. **[D3]** decides whether the per-experience gate budget also scales with k or stays fixed (proposal: stays fixed at the Phase-2 value, 2 gate steps per QA experience, so density changes *only* the plasticity rate and the γ-sequencing law — plasticity first, gate second — is untouched).

### 2.2 The k sweep

| Config | k | Everything else |
|---|---|---|
| k1 (baseline) | 1 | identical to measured Phase-2 E (no rerun needed for the baseline numbers; the measured E row is the k=1 reference) |
| k4 | 4 | γ pathway, gate_lr 100.0, gate_steps 2/expr, sequencing law: unchanged |
| k16 | 16 | same |

Derived cost estimate (labelled: arithmetic from the measured 5.2 s/update rate; batteries unchanged at the measured E pace, `battery_E-test` 6110.1 s + post-restart 1335.4 s): k4 ≈ 240 updates ≈ +21 min teach → **≈ 2.3–2.5 h total**; k16 ≈ 960 updates ≈ +84 min teach → **≈ 3.2–3.5 h total**. Phase-2 variants measured 1.7–2.6 h wall each (E `variant_s` 9459.8 s).

### 2.3 Predicted effects

- **u_fast norms:** the update is exact gradient descent on the per-experience error (APC §3) with weight decay λ=1e-5 and the ±10 clamp. k>1 re-descends on the same experience before decay can pull it back, so norms should rise above the measured D/E levels (≈1.18 / 1.12 / 0.53) roughly with the number of *effective* visits; the clamp gives large margin (Phase-2 freeze evidence: 93.7 % margin at max‖u‖=0.63 on the pilot scale). Prediction: k4 roughly doubles the D-level norms; k16 approaches saturation where repeated passes on an already-fit experience produce shrinking per-pass error.
- **Candidate rankings:** if the gap is a *quantity* problem, the γ-expressed residual grows with the stored signal and rank movement should exceed the measured 0.1-rank / ±0.007-prob band. If k16 still leaves ablation delta at 0.0, density alone is falsified as the lever.

### 2.4 Risks and the measure that detects each

| Risk | Signature | Measure |
|---|---|---|
| **Overfitting to taught phrasing** — k passes on the same 60 experiences binds knowledge to the trained surface forms | direct (trained format) climbs while paraphrase (unseen wording, 8 questions) stays flat or drops; combination (unseen composition) stays flat | the preregistered safeguards: direct / paraphrase / combination all reported per arm; paraphrase-vs-direct divergence is the phrase-binding signature |
| **u_fast drift past the interface's effective range** — norms grow toward the ±10 clamp; states are unit-L2 normalised (`normalize_states=True`, `interface.py` line 196) so ψ magnitudes are bounded, but the effective blend a = 0.7u + 0.3θ (θ empty in these runs — `theta_slow_empty` true in the report) is carried entirely by u | per-pass error shrinking to ~0 while norms still climb (memorisation); clamp margin falling | report `u_fast_norms_after` per arm (existing field) plus clamp-margin (max abs u vs 10); fact plastic strength distribution (`fact_plastic_strengths_min_mean`) |
| **Gate budget interaction** — if gate steps scale with k (rejected by default, **[D3]**), γ could ride density rather than expression quality | γ grows proportionally to k with no behavioural change | γ values reported per arm (existing `gamma_after` field); γ magnitude is explicitly NOT a success criterion |

---

## 3. LEVER B — ALIGNMENT: bringing the residual to the candidate-token embeddings

**Why alignment is suspect, in code.** The residual is built from **sequence-mean-pooled** hidden states (`hidden[i].mean(dim=1)`, `interface.py` lines 281–285) and lands as one vector broadcast to **every** position identically: `logits = base_logits + logits_delta.unsqueeze(1)` (line 301), where `logits_delta = residual @ W_head.T` (line 300). The pathway never varies by position, so the contribution at the answer position is the same whole-sequence-mean feature projected through the frozen head. It reaches candidate tokens only through that one dot product — which may not separate the candidates.

### Option 1 — Candidate-scoring projection (read the residual's per-candidate contribution directly)

The constrained-scoring path (`constrained_scores`, `teacher_student.py` lines 1233–1253) already computes per-candidate scores as summed candidate-token log-probs under the scaffolded prompt, per provenance mode. The forward already returns **both** `base_logits` and `logits_delta` (`interface.py` lines 300–311), so the residual's per-candidate contribution is isolable **with zero extra foundation forwards**:

- per candidate c: Δscore(c) = Σ_j [ log_softmax(base + δ)_tj − log_softmax(base)_tj ] over the candidate's token positions, computed from the returned tensors.
- record per question: per-pair residual norms ‖γ_k r_k‖, per-candidate Δscore, gold-rank movement (CONSOLIDATED → PLASTIC/FULL), and correct/incorrect. The **margin** score(gold) − max_distractor score(d) is reported under base and under base+δ. This is the owner's residual↔rank↔accuracy relationship measurement, and it runs as instrumentation in **every** arm.
- optionally (a code change, owner-approved only): position-gate the broadcast so the residual lands only at the answer position instead of `unsqueeze(1)` everywhere.

*Risk:* the decomposition may show the residual adds a near-constant shift to all candidates (no discrimination) — that is a real finding, not a failure of the instrument. *Failure measure:* |Δscore| correlated with rank movement but margin unchanged, and acc@1 unmoved on all splits.

### Option 2 — Residual norm scaling per layer

The interface hooks **three** layer pairs (`pick_hook_pairs`, `interface.py` lines 156–179: the first DeltaNet-family pair, the first attention-family pair, and the deepest pair L−2→L−1), each with its own gate — matching the three γ values in the report. Sub-options:

- **(a) inference-time per-pair scale s_k** on γ_k·r_k at scoring, preregistered (e.g. s ∈ {1, 10} per pair), no retraining;
- **(b) per-pair γ training targets** during gate learning (per-pair gate_lr or per-pair loss weighting), sequencing law preserved (gate still second).

*Risk:* scaling is a global logit-temperature move — it can flip argmax without improving discrimination; unbounded scaling can distort the whole distribution. *Failure measure:* the Option-1 decomposition shows a uniform shift across candidates (Δscore roughly equal for gold and distractors) while acc@1 is unchanged; softmax over candidates stays degenerate.

### Option 3 — Per-candidate γ

A γ per candidate-token cluster: cluster the candidate tokens by frozen-head embedding similarity over the world's entity vocabulary, and learn one gate per cluster.

*Risk (the serious one):* **per-candidate γ is a per-class prior.** Fitted to the exam it is answer-key leakage; fitted to training only it is a prior that may not transfer. If permitted at all: clusters AND γ values must be **frozen before the exam**, from training experiences only, preregistered in the run report. *Failure measure:* direct (trained format) improves while combination (unseen composition) does not — the prior signature; plus a leave-one-cluster-out check on held-out combos.

---

## 4. The preregistered variant table

### 4.1 Reference matrix (Phase 2, measured — kept as the comparison floor)

| Variant | Description | direct acc@1 | paraphrase acc@1 | combination acc@1 | retrieval (dir/para/combo) | Expression evidence |
|---|---|---|---|---|---|---|
| A | frozen base | 0.25 | 0.125 | 0.3333 | 0.0 / 0.0 / 0.0 | n/a |
| B | retrieval control | — | — | — | 1.0 / 0.625 / **0.0** | n/a |
| C | plastic, γ=0 | 0.25 | 0.125 | 0.3333 | 1.0 / 0.625 / 0.0 | bitwise identical to A |
| D | plastic + question, γ=0 | 0.25 | 0.125 | 0.3333 | 1.0 / 0.625 / 0.0 | bitwise identical to A |
| E | plastic + question + γ | 0.25 | 0.125 | 0.3333 | 1.0 / 0.625 / 0.0 | sub-threshold (γ≈10.4/11.2, ±0.1 rank, ablation delta 0.0) |

### 4.2 Phase-3 arms (tested independently; combined last)

Frozen across every arm: Qwen foundation frozen + identical ICL scaffold; same fact set, question batteries, held-out splits; bitwise restart + foundation fingerprint checks; raw AND adjusted scores reported together; no tuning on the final evaluation questions; γ sequencing retained (plasticity first, γ second); thresholds and stopping rules fixed **before** execution.

| Arm | Variant id | k | Alignment | γ | Tests |
|---|---|---|---|---|---|
| ARM-1a | E-k4 | 4 | none | trained as E | density dose-response |
| ARM-1b | E-k16 | 16 | none | trained as E | density saturation point |
| ARM-2a | E-align-1 | 1 | Option 1 (candidate-scoring projection + position-gated readout) | trained as E | alignment via per-candidate isolation |
| ARM-2b | E-align-2 | 1 | Option 2 (per-layer scaling, sub-option **[D4]**) | trained as E | alignment via layer weighting |
| ARM-2c (optional) | E-align-3 | 1 | Option 3 (per-candidate γ, frozen pre-exam) | per-cluster | the prior-lever — run only if **[D5]** permits |
| ARM-3 | combined | best of ARM-1 | best of ARM-2 | trained as E | both levers together |
| controls | A, B | — | — | — | frozen floor + retrieval control, re-measured in-session (**[D7]**) |

### 4.3 Success criteria — stated BEFORE any run (the DRAFT's proposal; the owner's numbers decide, **[D6]**)

**PRIMARY (decisive, behavioural):** on the unseen combination battery, PLASTIC-provenance constrained acc@1 (and FULL with its ablation delta) satisfies BOTH:
1. **> frozen-chance performance** — above the frozen base's 0.3333 and above the battery's mean chance 0.4; proposal: **≥ 0.50**;
2. **with/without-plastic ablation delta ≥ +0.10** (FULL or PLASTIC vs CONSOLIDATED on the same rows) — the plastic state itself, not the frozen base, carries the answer.

**REPEATABILITY:** the improvement holds on at least 2 of 3 held-out combination splits (splits fixed before the first arm runs; no question in any final evaluation split is ever used for tuning). Beating B's 0.00 alone is explicitly NOT sufficient.

**SAFEGUARDS (reported every arm, never traded away):** direct PLASTIC acc@1 within −0.05 of A's 0.25; paraphrase acc@1 not below A's 0.125; restart exact-restore true; foundation bit-identity true; retrieval column reported alongside.

**FAILURE IS A RESULT:** any arm that grows `u_fast` or γ without behavioural movement **fails its own test** (owner law) and is recorded as measured-negative. Mechanistic growth is reported as diagnostics, never counted as success.

**Stopping rules (preregistered):**
- Density: if at k16 the combination ablation delta is still 0.0 on every held-out split, ARM-1 is closed as failed-alone and ARM-3 falls back to alignment-only settings.
- Alignment: if an option moves gold rank on ≥50 % of questions but converts to acc@1 on no split, it is recorded as "logits move, behaviour doesn't" and closed.
- Budget: one full run per arm; no re-tuning after seeing the exam (a changed config = a new preregistration).

---

## 5. Decision points for the owner

1. **[D1] Arm order.** Proposal: ARM-1 (density) first — one knob, cheapest to spec, and Phase-2 E shows the γ pathway alive but under-fed — then ARM-2, then ARM-3. Alternative: alignment first (cheaper: no teach-loop cost increase; Option 1's instrumentation is nearly free).
2. **[D2] What "γ fixed" means in ARM-1.** Proposal: γ *training procedure* identical to E (gate_lr 100.0, 2 steps/QA, sequencing law) so k is the only difference. Alternative: γ numerically frozen at E's measured values [10.371428, 11.180095, 0.522552] during the density sweep.
3. **[D3] Gate budget under k.** Proposal: gate steps stay fixed per experience (density changes only the plasticity rate). Alternative: scale gate steps ×k.
4. **[D4] Alignment sub-option for ARM-2b.** Inference-time fixed per-pair scale (preregistered s values) vs per-pair γ training targets.
5. **[D5] Whether Option 3 (per-candidate γ) is permitted at all**, given the per-class-prior leakage risk; if yes, its freeze-before-exam rule applies.
6. **[D6] The final success numbers.** Draft proposal: combination ≥ 0.50 acc@1 AND ablation delta ≥ +0.10 AND 2-of-3 splits. The owner sets the pass line; once set it is not moved after results.
7. **[D7] Controls per session.** Re-run A and B in every Phase-3 session for in-session comparability (A ≈ 46 min, B ≈ 10 min measured) vs reuse the measured Phase-2 A/B rows.
8. **[D8] Budget and scope.** Estimated per arm (derived from measured Phase-2 rates): ARM-1a ≈ 2.3–2.5 h, ARM-1b ≈ 3.2–3.5 h, ARM-2a/b ≈ E's 2.6 h, ARM-3 ≈ 3–4 h. Which arms are approved, in what order, and whether Option 1's position-gated readout (a code change) is approved as the sole code delta vs a new Phase-3 experiment file.
9. **[D9] Failure reporting.** Proposal: every arm reports a full `report.json` under `runs/teacher_student_p3/`, a headline block in the Phase-2 format, and a measured-negative row in `docs/RESULTS.md` §2; no re-run without a new preregistration.

---

## 6. Register compliance

- **Corrected wordings.** No new-model-type claims, no "organism learns language" claims. Phase 3 tests whether a plastic-relational pathway can express stored knowledge on a frozen foundation — the register stays "a promising mechanism under test" (`docs/RESULTS.md` preamble) whatever the outcome.
- **Results reported whatever they show.** Failures are recorded, not tuned away (`docs/APC.md` §9.4; `docs/RESULTS.md` §5 life-gates). The Phase-2 bitwise-A result for C/D and the sub-threshold γ for E are preserved and cited as the baseline this protocol extends.
- **Freeze discipline.** Any code change that touches the mechanism (position-gated readout, per-layer scaling, per-candidate γ) requires a freeze manifest with SHA-256 before its results are citable (`docs/APC.md` §9.2).
- **Numbers.** Every figure above is copied from `runs/teacher_student_p2/p2-qwen-c5-s0-022716/report.json` or the cited source files; cost estimates explicitly labelled "derived" are arithmetic on measured rates, not measurements.
- **No runs.** This document changes no code and starts no lane. Execution begins only after the owner answers **[D1]–[D9]** and the success numbers in **[D6]** are final.
