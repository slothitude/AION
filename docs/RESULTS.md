# AION — Consolidated Results (2026-11-25)

AION (Adaptive Intelligence Organism Network) is an experimental research project — code package `nnfs` — built around one question: can a system whose defining ability is *learning* be raised on a borrowed substrate and survive its loss? The mechanism under test is a plastic-relational substrate (APC, `docs/APC.md`) with dual-timescale state and an error-driven learning law that follows the actual forward computation; it is examined against four controls — a static (no-learning) control, a retrieval-only control, a rule-based heuristic, and the frozen foundation itself. **The project register is: a promising mechanism under test.** Nothing in this document is a claim of a new model type, a new intelligence, or a solved problem.

**Honesty commitment.** Every number below is copied from a named result file in this repository; the file, not this document, is authoritative. Negative results are records, not embarrassments — they are preserved in `pilot_results/` and cited here, and erasing them is a spec violation (`docs/APC.md` §9.4). Claims are classified three ways: **demonstrated** (measured on a frozen baseline with a named source), **measured-negative** (measured and failed, stated plain), and **in flight** (running; no verdict claimed). Where a comparison is *unanswered* rather than negative, it is labelled unanswered. Held-out claims are process claims — they rest on the documented dataset split and the freeze dates of the baseline they were measured against, and they do not extend beyond those runs.

---

## 1. What is demonstrated

### 1.1 Connection-level learning (frozen v2 baseline)

| Measure | Result | Source |
|---|---|---|
| Held-out connection prediction MSE | 1.026 → 0.577 (**−43.8 %**); per-seed −51.6 %, −49.2 % | `freeze/AION_LEARNING_BASELINE_20261124.md` (gate-4 evidence table); `docs/APC.md` §6, §8 |
| Coefficient recovery via the effective mapping | a = [2.0003, 2.9989] vs truth [2, 3]; abs err **[0.0003, 0.0011]**; cosine **0.999998** | same |
| Predictions on y = 2x₁ + 3x₂ | [2.003, 2.993, 4.997, 7.000], MSE 1.7e-5 | same |
| Finite-difference gradient agreement | max rel err **1.95e-4** | same |
| Gain-calibration law (g²η invariance) | ‖Δŷ‖ = 0.000946 identical across (g,η) = (1, 0.05)/(2, 0.0125)/(4, 0.003125); spread 0.0 | same |
| Stability | no divergence under negative/zero-mean inputs; max‖u‖ = 0.63, 93.7 % margin from the ±10 clamp | same |

Citable point: v2 freeze `freeze/aion_learning_baseline_v2_20261125.tar.gz`, SHA-256 `ea802b96…f38734af` (registry in `docs/APC.md` §6). Per the register, these are process claims bounded by the documented split and the 2026-11-24/25 freeze dates.

### 1.2 Network-level pilot: Plastic beats Static on the corrected instrument

Six-configuration pilot, mean of 3 seeds, rotation benchmark (lag-1 targets, phase-block resets; `pilot_results/pilot_results.json`):

| Measure | Plastic D0 | Static D0 | Δ | Source |
|---|---|---|---|---|
| Rotation phase A MSE | 0.7508 | 1.0008 | **−25.0 %** | `scripts/eval_fix_report.md`, research row 1; per-seed 0.7558/0.7393/0.7572 vs 1.0017/0.9992/1.0015 in `pilot_results/pilot_results.json` |
| Rotation phase C MSE | 0.7529 | 1.0043 | −25.0 % | same |
| Held-out discovery error | 0.9696 | 1.0032 | improvement | same |

Static-vs-Plastic-vs-Consolidating differentiation holds on the corrected instrument. The untrained readout moves independently of connection learning (readout MSE ~1.12–1.23 in every arm), confirming it is not the learning signal (`scripts/eval_fix_report.md`, [VALIDATION] block). Adaptation-step criteria are honestly `None` for every variant at 900 steps: the model improves ~25 % on phase error but never halves switch-time error within a 300-step phase.

### 1.3 Dual-timescale consolidation: the mechanism now fires and preserves output

The original gate-6 pilot recorded consolidation *unable to fire*: selective gates held at `gate_mean` **0.062** with **0 elements transferred** in every config (`pilot_results/pilot_results.json`, `consolidating_d0_none`, seed 42: gate_mean_avg 0.06204, consolidated_total 0, theta 0 → 0). Reported as a FAIL per owner directive, then repaired at the unit level (stability gate, blend anneal — `docs/APC.md` §4, tests 7–10). The gate-6 rerun demonstrates the corrected mechanism:

| Measure (consolidating D0, 3 seeds) | Rotation | Compositional | Source |
|---|---|---|---|
| Gate mean | 0.4438 / 0.4370 / 0.4346 | 0.5077 / 0.5097 / 0.5066 | `pilot_results/pilot_results_gate6_consolidation.json` |
| Elements transferred | 53,506 / 51,119 / 45,924 | 85,496 / 88,801 / 80,314 | same |
| θ_slow norm (0 → final) | 109.28 / 112.81 / 105.30 | 32.17 / 30.99 / 32.39 | same |
| Output preservation (max ‖Δŷ‖) | ≤ **2.4e-7** (1.19e-7 / 2.38e-7 / 1.19e-7) | ≤ 6.0e-8 | same |

What this demonstrates: consolidation transfers fast state into slow state at scale while preserving the transmitted output to float precision — the persistence property (`docs/APC.md` §4, protected property 4; unit-level preservation asserted at 1e-6 in tests 5, 8, 9). What it does **not** demonstrate is a retention benefit — see §2.3. Rate > 0 consolidation has still not been exercised in a live continual run (`docs/APC.md` §8).

### 1.4 The Disconnect Experiment: capabilities survive host removal

One run, seed 0, battery `disconnect-tiny`, 12 seed relations (`runs/disconnect_real_seed0/disconnect_report.json`). The disconnected lane learned the same seed facts through direct observation only; the WITH-host control ran on `Qwen/Qwen3.5-4B-Base` and scored 1.0 on all five capabilities (the denominator). Five capabilities, both runs reported, ratio = without_host/with_host on the same battery:

| Capability | WITH host | WITHOUT, post-restart | Raw survival ratio | Verdict |
|---|---|---|---|---|
| combine | 1.0 | 1.0 | 1.0 | fully-survives |
| learn_new | 1.0 | 1.0 | 1.0 | fully-survives |
| predict | 1.0 | 0.75 | 0.75 | degrades |
| recognise | 1.0 | 0.75 | 0.75 | degrades |
| self_correct | 1.0 | 1.0 | 1.0 | fully-survives |

**Epoch-adjusted reading: 1.0 on all five.** The 0.75 rows trace entirely to RETRIEVED rows (6/24 predict, 3/12 recognise) that were **never owned** — their substrate is the transcript, not the plastic state. When the WITH-host lane is re-scored on the symmetric epoch (post-intervention), it degrades identically to 0.75 on the same rows (`runs/disconnect_real_seed0/host_rescored_post.json`: "post-intervention (symmetric with the disconnected lane's post-restart scoring)"); on the symmetric epoch the survival ratio is 1.0 across all five capabilities. Every owned row is PLASTIC-provenance (2/2, 4/4, 18/18, 9/9, 3/3).

**Structural proof of disconnection.** Across every disconnected phase (teach onward) the counters read foundation_forward **0**, foundation_load **0**, teacher_call **0**, with `iface_absent: true`, `monitor_closed: true`, `poison_loader_intact: true` — including a lane lifetime with zero teacher calls. The WITH-host teach phase, measured in the same report, used **92** foundation forwards and **1** foundation load. During disconnected observation, 4 new relations entered via the direct-feed encoder only, all PLASTIC provenance, consolidation invariant ok.

**Restart.** `bitwise_tensors: "ALL-BITWISE-EQUAL"`, `state_restored_exact: true`, `tensor_diffs: []`, probes regenerate exactly; post-restart capability scores reproduce (1.0 / 1.0 / 0.75 / 0.75 / 1.0). Corrections and interventions applied while disconnected were re-scored 3/3 self-correct rows passing (1 correction, 2 interventions), with old facts intact.

This is one seed of one small battery. It demonstrates the *architecture* of independence — owned knowledge outlives the host, restart is bitwise — not yet breadth.

### 1.5 Host mode: adaptation under an authority-gated runtime

20 episodes per arm, exam seed 20261125 (`aion/hostmode/results/hostmode_results.json`). Headline battery (needs-grant class), plastic arm adaptation curve — mean trials to grant by episode block:

| Episode block | 1–5 | 6–10 | 11–15 | 16–20 |
|---|---|---|---|---|
| Plastic mean trials to grant | **16.25** | **9.0** | **8.8** | 10.8 |

Headline summary: plastic success 0.95 (mean 13.35 trials, 0.95 failed writes/episode) vs fixed-random control 0.9 (25.9 trials, 1.85 failed writes/episode); heuristic control 1.0 (6.0 trials). **Authority violations: 0** in every arm, every episode. On the mixed control battery (6 fits-default / 6 impossible / 8 needs-grant episodes), **correct stop on impossible tasks: plastic 1.0 (6/6) vs fixed-random 0.0 (0/6)** — the plastic arm stops with `insufficient_capacity` in 3–9 trials where the fixed-random arm exhausts 40 (honest note: the rule-based heuristic also stops correctly at 1.0, so plastic's measured deltas over baselines are efficiency — trials, failed writes, exhaustion rate 0.0 vs 0.1 — not capability). Post-grant success is 1.0 in all arms (8 grants, 0 unneeded). The demonstration trace records the full arc in one episode: WRITE denied (4,498 B > 1,024 B cap) → REQUEST_CAPACITY granted on demonstrated need (1,024 → 65,536 B) → WRITE succeeds → host verifier confirms → goal achieved (episode 1, trial 17–18).

### 1.6 D1 replay validation: replay is a real, measured intervention

The D1-only Dream Engine re-implementation (owner decisions D1–D6, `docs/DREAM_REIMPL_SPEC.md` §6a; validation plan `docs/D1_REPLAY_VALIDATION_PLAN.md`) passed its validation gate: **18 runs** (3 arms × 2 benchmarks × 3 seeds, ~60 min), **54/54 replay cycles** produced measured state deltas ‖Δu_fast‖ 0.47–2.15 against the D0 control's measured **exactly 0.0**; **1,728 decision records** all `replayed=true` with (cycle_id, step, effective_lr); budget honoured exactly (`min(32, retrieved)`, 1 step each); the θ-write bypass is **eliminated** (instrumentation: all θ changes occur inside `consolidate()` windows, outside exactly 0.000e+00; source scan clean); the norm guard is calibrated both ways (3/3 valid accepted, 3/3 pathological rejected, margin 65.2×). Storage was rebuilt per owner D2 (fixed-shape windows; flattened records archived, never replayed). Frozen baseline v2 byte-identical throughout. Results: `pilot_results/d1_validation_results.json`.

**Honest scope:** replay is a real intervention on learning *state*; whether it improves downstream learning remains the six-configuration question (§2.1) — at this dose the held-out batteries barely moved, stated plain.

---

## 2. What is measured-negative (stated plain)

### 2.1 D2 vs D1 — **unanswered; the reimplementation is built and gate-passed, the preregistered re-run awaits owner approval**

This question — does imagination add anything over replay? — is **not answered**. What is on record:

- **Replay was inert in the pilot.** D1 is bit-identical to D0 on every field, every seed (e.g. rotation final MSE 0.7580715894699097 in both; 0 hypotheses; theta 0 → 0; topology unchanged) — the pilot's replay was a stub for blend-1.0 plastic variants, partly by design (`consolidate()` no-ops at blend ≥ 1.0; `scripts/eval_fix_report.md`, finding 2). With a broken replay lane, D1 is not a valid baseline, so no D2-vs-D1 conclusion could be drawn.
- **D2's writes were artifacts.** D2 accepted 60/60 hypotheses (precision 1.0) with no held-out effect (A→C 0.9694 vs 0.9696) and hurt rotation retention (1.0810 vs 1.0029); its dream-side consolidation moved theta to norm 9.81 while the connection counters recorded 0 created as topology grew 36 → 61/72 — the engine's creation path bypassed the instrumentation (`scripts/eval_fix_report.md`, research row 4 and finding 3). Under equal compute (D1x2, 2× dream budget), D2 ties exactly (0.9696 vs 0.9696) at ~2× the dream compute (0.33 s vs 0.17 s per the report table).
- **Status update (2026-11-25): the reimplementation is BUILT, D1-gate-validated, AND the D2 imagination arm has now MEASURED the question at validation dose** (`docs/D1_REPLAY_VALIDATION_PLAN.md` + stage-2: 360 hypotheses → 167 accepted / 193 rejected by reality, acceptance rate 0.464, every acceptance beyond-noise evidenced; θ only via the output-preserving law; retention damage gone). **D2-vs-D1 verdict at this dose: imagination adds no demonstrated held-out benefit over replay** — rotation ~0.3% better, compositional ~0.05% worse, within 3-seed noise, at ~20× dream compute. The full-scale six-configuration re-run — which would answer the question at production dose — is **drafted and awaiting the collected-report review and explicit owner approval**. Until that re-run, D2-vs-D1 at full scale remains unanswered — never as "reported either way".

### 2.2 Phase-1 conversational acquisition — **FAIL: a negative result for the mechanism**

AION-1 Phase 1 (real Qwen teacher + frozen `Qwen3.5-4B-Base` foundation, CPU BF16, 426 tensors / 4.2 B params frozen, 5 chains × 20 relationships × 65 questions; `runs/teacher_student/qwen-real-c5/report.json`): **acceptance 3/5** — retention, correction, transcript-removal PASS; acquisition and generalisation honest FAIL.

| Battery (exact) | PLASTIC | RETRIEVED (transcript) | Chance |
|---|---|---|---|
| direct (n=20) | **0.00** | **0.75** | 0.20 |
| combination (n=10) | **0.00** | 0.00 | 0.20–0.50 |
| paraphrase (n=10) | **0.00** | 0.60 | 0.20 |
| intervention (n=10) | **0.00** | 0.30 | 0.20 |

The plastic channel answered **empty on 100 % of questions** (`empty_answer_fraction: 1.0` across PLASTIC/FULL/CONSOLIDATED in every battery); the only working channel was transcript retrieval. That is a **negative result for the plasticity mechanism** on conversational acquisition in this configuration — stated as the current state of the mechanism, not a verdict on the register. It also means the disconnect experiment's headline rests on a lane that learned by direct observation, while the conversational path is the open failure.

What did pass, honestly recorded: bitwise restart (`state_restored_exact: true`, probes regenerate identically); correction (new fact strength 1.307077 learned, old facts 20/20 present, unaffected probes 16/16 identical, mean relative strength change on old facts 0.0079); transcript-removal invariance (21 events purged, plastic answers 20/20 identical); foundation integrity bit-identical after the full run (426 params checked).

### 2.3 Consolidation benefit and the gate-6 criterion — **no retention improvement measured**

On the corrected instrument, consolidating vs plastic is within noise on retention (1.0013 vs 1.0029) and slightly worse on discovery (0.9791 vs 0.9696) — **gate 6 as originally posed FAILS** (`scripts/eval_fix_report.md`, research row 2). The rerun (`pilot_results/pilot_results_gate6_consolidation.json`, §1.3) shows the corrected mechanism fires and preserves output; it does not yet show consolidation beating plastic on any outcome measure. Network-level consolidation behaviour in a live continual run remains unproven (`docs/APC.md` §8).

### 2.4 NAV exam — decision heads do not transfer

Preregistered `nav-exam-1.0` (preregistered 2026-11-25, world seed 20261125, 30 scenarios, 6 per gold class, closed enum QUERY/INFER/ANSWER/ASK_TEACHER/STOP). Heuristic floor (ceiling): **1.0** (30/30).

| Backend | Accuracy | Per class (correct/6) | Source |
|---|---|---|---|
| larql head (classifier) | **0.20** (6/30) | QUERY 6/6; ANSWER 0/6, INFER 0/6, STOP 0/6, ASK_TEACHER 0/6 — **the head does not transfer** | `aion/nav/results/nav_exam_20261125_045356.json` (Rog); same 0.20 in the first partial run `nav_exam_20261125_045147.json`; replicated on Lappy in `lappy_d1_exam_20261125_054944.json` |
| d1-2B | **0.533** (16/30) | ANSWER 6/6, INFER 4/6, STOP 5/6, ASK_TEACHER 1/6, **QUERY 0/6 — the QUERY blind spot** | `aion/nav/results/lappy_d1_exam_20261125_054944.json` (Linux/Lappy) |
| Tev1 0.8B | **no score** — 30/30 LICENSE-BLOCKED | accuracy null | same file (`license_blocked: 30`; repo untagged and card states the release license is being finalised) |

The larql head was trained on larql M6 states and probed on NAV chain-world states — the distribution shift is recorded in the exam records themselves; the 0.20 is a transfer failure, stated plainly. d1-2B is partial: it answers and infers but cannot decide to query.

### 2.5 Phase-3 expression matrix — **all three levers fail; every arm at the frozen floor** (matrix closed 2026-11-26)

Protocol `docs/PHASE3_PROTOCOL.md`, decisions D1–D9 locked, in-session controls: A frozen floor direct@1=0.250 paraphrase@1=0.125 combination@1=0.400; B retrieval direct=1.000 para=0.625 combo=0.000 splits=[0.0, 0.0, 0.0] (retrieval remains structurally blind to composition — the gap plastic must clear).

| Arm | Mechanism | direct@1 | paraphrase@1 | combination@1 | D6_success | restart |
|---|---|---|---|---|---|---|
| ARM-1a | density k=4 (9749 s) | 0.200 | 0.125 | 0.400 | False | bitwise |
| ARM-1b | density k=16 (15306 s; 960 connections) | 0.200 | 0.125 | 0.400 | False | bitwise |
| ARM-2a | alignment k=1 (16396 s) | 0.200 | 0.125 | 0.400 | False | bitwise |
| ARM-3 | combined | **NOT RUN** — locked D8 rule: both mechanisms failed independently, never combine two failed mechanisms | | | | |

Foundation bit-identical after the matrix: True; matrix errors: 0. Total runtime 725.8 min on real Qwen3.5-4B-Base (CPU BF16, frozen), 20-rel world, 43 questions, held-out splits [5, 5, 5]. Sources: Lappy `p3run.log` and `runs/teacher_student_p3/p3-c5-s0-200451/report.json` (+ `results.jsonl`).

**Stated plain:** density (4 or 16 connections per relationship) and alignment (k=1) leave expression **exactly at the frozen floor** — no accuracy metric moved in any arm. This is consistent with Phase 1 (empty answers), Phase 2 (bitwise-A variants; γ real but sub-threshold), and the interventions run (degenerate argmax). Four experiments now measure the same boundary: **the plastic state acquires, stores, corrects, retains and survives — and none of it reaches the frozen host's answer-selection through the pathways tested.**

**Measured why (per-question decomposition, preserved in `results.jsonl`):** the transmitted residual is real and large (mean ‖γ·r‖ = 9.30 / 10.42 / 7.22 across the arms) but **uncorrelated with the answer**: gold Δscore −0.126 / −0.120 / −0.070 log-probs, gold helped on only 58% / 44% / 61% of questions (coin-flip); the gold-vs-best-distractor margin moved **negative** in all arms; rank-gain-to-correct conversions **0 / 0 / 0**. This is the precise "merely moving logits" signature the protocol's risk line predicted — candidate-specific, answer-uncorrelated, margin-negative.

**Mechanistic lead (diagnostic, not yet a causal verdict):** novelty score — mean pairwise cosine distance between the 20 per-fact transmitted features — is **0.0037 / 0.0071 / 0.0013**: the plastic state's fact representations are nearly collinear. Density multiplied the signal (u_fast [1.49, 1.70, 1.74] → [2.00, 2.38, 4.21]; 240 → 960 updates) without decorrelating it: the fast weights store one shared task direction, not distinct per-fact features. Chain-0 retention 1.0 throughout (nothing forgotten).

The next central question — whether the learning objective itself must change to target expression (learn-toward-expression, now with a measured decorrelation target), rather than bolting expression onto a prediction-trained state — belongs to the owner's review of this report, per the no-automatic-progression law. Local artifacts: `runs/teacher_student_p3/p3-c5-s0-200451/{report.json, results.jsonl, world.json}` (Lappy mirror identical).

---

## 3. What is in flight (no verdicts claimed)

**Phase-3 expression matrix (Lane G, Lappy — RUNNING).** Protocol: `docs/PHASE3_PROTOCOL.md`, decisions D1–D9 locked; in-session controls on the 20-rel world / 43 questions: A frozen floor direct@1=0.250 para@1=0.125 combo@1=0.400; B retrieval direct=1.000 para=0.625 combo=0.000. **Phase-3 expression matrix — CLOSED 2026-11-26; verdict recorded in §2.5. Owner decision 2026-11-26: LEARN-TOWARD-EXPRESSION.** The owner reviewed §2.5 and authorised the design and preregistration of the Learn-Toward-Expression experiment (`docs/LTE_PROTOCOL.md`) — NOT its execution. Free-decode pathway and the dreaming six-config rerun are HELD; the compliance ruling remains a separate gate. No mechanism changes or runs until the LTE protocol is reviewed.

**Pending, explicitly:** the substrate-cost baseline (task 27) and the substrate ruling itself are **pending** the compliance ruling. No substrate-level claim rides on an unresolved ruling.

**Pending, explicitly:** the substrate-cost baseline (task 27) and the substrate ruling itself are **pending** the compliance ruling. No substrate-level claim rides on an unresolved ruling.

---

## 4. Known limits

- **Scale.** The pilot networks are 16-neuron relational networks; the conversational worlds are 5 chains / 20 relationships; the disconnect battery is 12 seed relations, one seed. All demonstrations are small-world demonstrations.
- **20-relationship world.** The Phase-1 conversational result (§2.2) is measured on exactly this world; nothing generalises beyond it until a larger world is run.
- **CPU speed.** All compute is CPU: the pilot rotation runs took ~120–190 s per seed; the Phase-1 run took teach 1,184.9 s + test 505.6 s with the 4.2 B-parameter foundation frozen on CPU BF16. Wall-clock is a real constraint on iteration count and world size.
- **Adaptation criterion never met.** At 900 pilot steps no variant halves switch-time error within a 300-step phase — the honest `None` (`scripts/eval_fix_report.md`).
- **Unlearnable-instrument null preserved.** The first pilot run tied every target to the *current* input (`y = R@x_t`), making the task informationally unlearnable from state history. All six configurations recorded connection MSE ≈ 1.0048 — the level of E[target²] — with adaptation `None` everywhere (`pilot_results/pilot_results_unlearnable_instrument.json`; `scripts/eval_fix_report.md` Part 1). This is kept as the preregistration record of the instrument defect; the corrected instrument proved alive in the single-cell gate before the burn (rotation plastic D0: phase A 0.7558 / C 0.7581 vs B 0.8192 — learned **and** retained; compositional 0.9694 < 1.0).
- **Property 3 (contextual plasticity) is code-level-proven only.** The frozen calibration and the unit suite run `context_dim = 0`; there is no dedicated numerical test yet (`docs/APC.md` §5, property 3 honest note).
- **Consolidation's live continual behaviour** at rate > 0 has not been exercised in a continual run (§1.3, §2.3).

---

## 5. Instruments and preregistration — how the project prevents self-deception

- **Freeze manifests.** Baselines are SHA-tagged archives with per-file hashes (`freeze/`; v1 2026-11-24 `a39013c9…`, v2 2026-11-25 `ea802b96…`, registry in `docs/APC.md` §6). Results quoted without a freeze pointer are anecdotes (`docs/APC.md` §9.2). Mechanism changes require a new manifest before the new point is citable.
- **Interpretation matrix.** Each measured question is read under pre-declared interpretation readings, and disconfirming readings are reported alongside favourable ones — e.g. the pilot's row 4 reports interpretation reading #4 for D2 (accepted-precision 1.0) next to the retention damage (1.0810 vs 1.0029) and the dual-budget tie; the disconnect register string itself is owner-corrected and mandates **both** runs be reported with RETRIEVED never counted as owned (`runs/disconnect_real_seed0/disconnect_report.json`, register field).
- **Life-gates.** Acceptance criteria are evaluated and *reported as FAIL when they fail*, not tuned away — gate 6's 0-transfer failure, Phase 1's 2/5 FAIL verdicts and the 3/5 score, and the adaptation-`None` honesty are all on record in their result files.
- **Redo-question rule.** A question that cannot be validly measured on the current instrument is re-posed with a corrected instrument, and the failed instrument run is preserved as the record of why — the unlearnable-instrument null (§4) and the lag-1 instrument fix that produced run-2 are the worked example.
- **Honest-null preservation.** Negative records live in the repository and are cited: the superseded handover is kept and marked superseded rather than deleted (`HANDOVER_SUMMARY.md`), the pre-fix and unlearnable run records remain in `pilot_results/`, and erasing them is a spec violation (`docs/APC.md` §9.4).
- **Preregistration headers.** The NAV exam carries its preregistration date, world seed, gold counts and closed enum in the result file itself; the host-mode exam carries its seed and caps; the disconnect report carries its battery id and registration string.

---

*Consolidated 2026-11-25 from the source files named throughout. Docs-only: this document changes no code. Where any figure here and its source file disagree, the source wins.*
