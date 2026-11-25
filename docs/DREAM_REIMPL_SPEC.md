# Dream Engine Reimplementation Specification — Real Replay + Reality-Grounded Evaluation

**Status:** Decision document for owner review. This document prepares a decision; it decides nothing and implements nothing.
**Date:** 2026-11-25. **Scope:** docs-only — zero code changes, zero runs.
**Derives from:** the owner-deferred reimplementation (options 2 + 3), recorded as deferred in `README.md` stage-04 row: *"Diagnosed + bookkeeping repaired; real replay + reality-grounded evaluation deferred (owner decision); dream-side consolidation routing unblocked."*
**Register context:** `docs/RESULTS.md` §2.1 records D2-vs-D1 as *unanswered, with a reimplementation pending*. Until the owner decides, this document does not change that record.

---

## 1. What the diagnosis established

The Dream Engine's pilot failures trace to three root causes, each verified at the file and line level. All numbers below are copied from the named sources; the sources, not this document, are authoritative.

### 1.1 Root cause 1 — Replay is a stub: D1 is bit-identical to D0

`src/nnfs/dreaming/replay.py` never touches the network:

- `replay_batch` (lines 62–84) loops over retrieved experiences and its body ends in `pass` (line 82), with the in-file admission `seq_len = 10  # Default, would be stored with experience` (line 78).
- `replay_episodic` (lines 86–112) retrieves experiences and returns placeholder statistics — `{"replayed": len(experiences), "avg_mse": 0.0, "consolidations": 0}` (lines 108–112) — without a single forward pass or learning step.
- Verified by search: **zero `network.step` calls** across `engine.py`, `replay.py`, `simulator.py`, `evaluator.py`, `consolidator.py`. (The only `network.step` calls under `src/nnfs/dreaming/` live in `benchmarks.py` and `experiments.py`, which are separate experiment scripts, not the dream path.)

Measured consequence (`scripts/eval_fix_report.md`, research row 3): **D1 = D0 on every field, every seed** — e.g. rotation final MSE `0.7580715894699097` in both; 0 hypotheses; theta 0 → 0; topology unchanged. D1 is therefore not a valid baseline, which is why `docs/RESULTS.md` §2.1 records D2-vs-D1 as unanswered rather than negative. A secondary contributor is documented: the live calibration runs `blend_ratio = 1.0`, and the connection's own `consolidate()` no-ops at blend ≥ 1.0 absent annealing (`docs/APC.md` §4); the replay lane could not consolidate for the blend-1.0 plastic variants even if it had run (`scripts/eval_fix_report.md`, finding 2).

### 1.2 Root cause 2 — Dream-side θ-writes bypassed every gate, preservation law and clamp

`src/nnfs/dreaming/consolidator.py` writes slow weights directly instead of routing through the connection's output-preserving `consolidate()` (`src/nnfs/connections/plastic.py`, the annealing exact-solve, `docs/APC.md` §4):

- `_consolidate_composition`, new-connection path: after creating the connection, `conn.plasticity.theta_slow.data = hypothesis.hypothesis_weight.data.clone()` (line 222). This is a raw θ-write — no activity mask, no selective gate, no a-preservation solve, no `±10` clamp discipline, no output-preservation check.
- `_consolidate_active_connections` (lines 305–337): raw `slow.add_(self.consolidation_rate * delta)` on every connection with trace strength > 0.1 — again outside `consolidate()` (it carries the repaired 50 %-norm guard, owner fix 2, but is still not the connection's own law).
- The composed hypothesis weight is `w2 @ w1` of the two source connections' combined weights (`recombiner.py` lines 166–170, comment *"Compose: w_composed = w2 @ w1 (for linear)"*), and that product was cloned wholesale into θ of brand-new connections.
- At pilot time the new connections were built with **hardcoded `learning_rate=0.01` / `blend_ratio=0.7`** against the live calibration `η = 0.05` / `blend = 1.0` (`docs/APC.md` §6). Under blend 0.7, a = 0.7·u + 0.3·θ — so 0.3 × the composed weight entered the transmitted signal immediately. **This is the 0.3× artifact injection.** Owner fix 4 (2026-11-25, `consolidator.py` lines 29–42 and 203–219) has since changed creation to inherit the live network's lr/blend — that repair is landed; the direct θ-write at line 222 and the raw sweep at lines 305–337 are **not** repaired and remain outside `consolidate()`.
- Bookkeeping was also wrong: connection counters recorded **0 created** while topology grew **36 → 61** (D2; +25 new connections) and 36 → 72 (D3) (`scripts/eval_fix_report.md`, finding 3). Owner fix 1 (`engine.py` lines 300–308) repairs the counting; the pilot numbers predate it.

Measured consequence (`scripts/eval_fix_report.md`, gate-6 row 2 and research rows 4–5): D2/D3 θ moved **0 → 9.81 / 6.66** "via the dream engine's own consolidation path"; D2 accepted 60/60 hypotheses yet left held-out A→C unchanged (0.9694 vs 0.9696) and **actively hurt rotation retention** (1.0810 vs 1.0029; phase C 0.9326 vs 0.7529, worse in all 3 seeds). D3's rotation benefit (A 0.7213, C 0.7210, retention 0.9996 — the only variant ≤ 1.0) is recorded as-is; this spec does not claim to explain it, and the re-run will re-measure it.

### 1.3 Root cause 3 — The simulator returns hardcoded constants; acceptance is guaranteed

`src/nnfs/dreaming/simulator.py` never runs the network. `simulate_hypothesis` receives `initial_state` and ignores it; `predicted_outcomes` and `prediction_errors` are `torch.zeros(1)` placeholders (lines 109–112), and `engine.py` passes `initial_state=torch.zeros(1)  # Would use actual state` (line 262). Every score is a table lookup:

| Score | Code | Composition value | Source |
|---|---|---|---|
| consistency | lines 130–142 | **0.8** (composition) / 0.6 (analogy) / 0.5 (context_transfer) | hardcoded per `relationship_type` |
| novelty | lines 144–150 | **0.7** | hardcoded, unconditionally |
| utility | lines 152–162 | **0.8 × confidence** (composition base) | table × `hypothesis.confidence` |
| stability | lines 164–167 | confidence × 0.8 | formula, not measured |

Acceptance arithmetic (simulator lines 94–107, weights 0.4/0.3/0.3, accept above 0.65): for a composition hypothesis with confidence c, total = 0.4·0.8 + 0.3·0.7 + 0.3·0.8·c = **0.53 + 0.24·c**. At c = 0.7 this is **0.698 > 0.65 → "accept"**; at c = 1.0 it is 0.77. The evaluator then always accepts composition as well: overall = 0.3·0.8 + 0.2·0.7 + 0.2·0.8 + 0.3·(1 − 0.1) = **0.810 ≥ 0.6 → accept** (evaluator lines 91–101; forgetting risk 0.1 because consistency 0.8 > 0.7, lines 145–153; efficiency 0.8 for composition, lines 155–162). The gate-6 pilot's **60/60 acceptance (precision 1.0)** is this arithmetic, not evidence.

Two further facts complete the picture: the evaluator's `overall_score` (lines 91–96) sums consistency, novelty, efficiency and (1 − forgetting) — **utility is not in the sum** (it appears only in the feedback dict, line 116); and `engine.py` lines 276–280 evaluate *every* simulation result regardless of the simulator's recommendation, so the simulator's own accept/reject never gates anything. **"Reality decides" is nowhere implemented** — nothing in the dream path ever measures the network's behaviour with the hypothesis installed.

---

## 2. REPLAY SPEC (option 2)

**Goal:** replay_episodic actually rehearses stored experiences through the network's real learning law, so D1 becomes a valid baseline and replay becomes a measurable intervention.

**R1. Storage must preserve shapes.** `MemoryRetriever` currently flattens each experience, averages over `seq_len`, and pads/truncates to `experience_dim` (`retriever.py` lines 91–108) — the reconstruction information `replay_batch` says it lacks (line 78) was destroyed at store time. Store per experience: `seq_len`, `num_neurons`, `neuron_dim`, the un-averaged (or last-step) `(input, target)` tensors, `context`, `context_id`, significance and uncertainty. Backfill policy (owner point): the 10 k-slot ring may be rebuilt to hold small fixed-shape windows instead of flattened vectors; the pilot world is 16 neurons × `neuron_dim`, so memory cost is bounded.

**R2. Replay executes the real learning law.** For each retrieved experience: reconstruct `(input, target)` at stored shape → `network.step(input, target_next=target, learn=True)` (signature, `src/nnfs/connections/network.py` line 157) — the same law the live path runs (`docs/APC.md` §3), with learning scaled by `learning_rate_scale` (default **0.1**, i.e. effective per-step η = 0.005 against live 0.05; both numbers already exist in `replay.py` lines 46/54 and `engine.py` line 119). `interleave_prob` (0.1) may later mix replay steps into the live stream; not required for the first re-run.

**R3. Integration point with the landed consolidation.** Replay must not invent its own consolidation. The LANDED path is the connection's output-preserving annealing exact-solve — `plastic.py consolidate()`: `a_before = b·u + (1−b)·θ`, `u' = u − d`, `θ' = (a_before − b'·u')/(1 − b')`, blend anneal 0.05 toward floor 0.5, `transfer_fraction` 0.1, rate-0 disable, output preserved to 1e-6 (tests 5, 8, 9; `docs/APC.md` §4). Replay simply drives `network.step(..., learn=True)`; any periodic consolidation then flows through the connection's own `consolidate()` with its gates, mask and preservation invariant — the same machinery the gate-6 rerun already proved fires at scale (gate means 0.4438/0.4370/0.4346; 45,924–53,506 elements transferred; output preservation ≤ 2.4e-7; `pilot_results/pilot_results_gate6_consolidation.json` via `docs/RESULTS.md` §1.3). Dream-side consolidation routes through **nothing else** (see §5).

**R4. Replay budget per dream cycle.** Bounded and equal-compute-auditable: one cycle replays `min(32, retrieved)` experiences (engine lines 241–247 already pass 32), 1 step each, at the scaled rate — the D1x2 equal-compute control (2× dream budget, `scripts/run_pilot.py --gate5`) is retained so equal-observation and equal-compute readings both exist. Every replayed step is counted in `total_dream_steps` (engine line 247 already does this; the count becomes real).

**R5. Provenance.** Every experience touched by replay is tagged `replayed=true` with (cycle_id, step, effective lr) in decision records, distinct from live-observation provenance. This continues the project's provenance separation law (PLASTIC vs RETRIEVED never conflated — `docs/RESULTS.md` §1.4); imagined-vs-actual gets the same treatment in §3.

**R6. Honest failure modes.** If replay produces no measurable change on a valid instrument, that is the result — reported, not tuned away (`docs/RESULTS.md` §5 life-gates; `docs/APC.md` §9.4). A replay lane that cannot move the connection state at scaled lr on the corrected instrument would be a finding about the instrument or the rate, reported as such.

---

## 3. REALITY-GROUNDED EVALUATOR SPEC (option 3)

**Goal:** replace the constant tables with measurement. "Reality decides" means: the hypothesis is installed, the network is run on stored experiences, and the scores are *measured behaviour deltas*, not type lookups.

**E1. The measurement loop.** For each hypothesis: run the network on a batch of stored replay experiences **with the hypothesis installed** (lightweight deep-copy of affected state, or install-then-restore around the measurement — implementation choice is an owner point, §6), with `learn=False` during scoring so measurement does not itself train. Baseline: the same experiences on the unmodified network. The scores become:

- **consistency** := measured prediction degradation on stored experiences the hypothesis was *not* built from (conflict measure), 1 − normalised error increase.
- **utility** := **measured prediction improvement** vs the unmodified network on the experiences the hypothesis was built from — the delta the current code never computes (simulator lines 109–112 are `torch.zeros(1)` placeholders).
- **novelty** := max cosine distance between `hypothesis.hypothesis_weight` and existing connection weights (the current `_evaluate_novelty`, simulator lines 144–150, returns the constant 0.7 without checking the network it names in its comment).
- **stability** := repeated-measurement variance of the utility delta across held-out replay batches (low variance = stable).

**E2. utility_weight enters the evaluator sum.** The evaluator's `overall_score` (lines 91–96) currently omits utility. With measured utility in hand it enters the sum: overall = w_c·consistency + w_n·novelty + w_u·utility + w_e·efficiency + w_f·(1 − forgetting_risk), with default weights to be fixed at re-run time (owner point; the existing 0.3/0.2/—/0.2/0.3 layout is the starting proposal, with utility taking a slot rather than being appended silently). Forgetting risk itself becomes measurable (error increase on consolidated, high-gate connections after install) instead of a consistency-derived constant (lines 145–153).

**E3. Minimum-improvement ε threshold.** Acceptance requires **utility ≥ ε** — a minimum *measured* improvement, not a score above 0.6. Proposed default ε = 0.0 strictly (any real improvement required) with a reported utility magnitude; an owner may set ε > 0 to demand a margin. The 0.65/0.6 constant thresholds (simulator line 100, evaluator line 47) are retired from the acceptance path; they may remain as reporting bins but cannot consolidate anything.

**E4. Hallucination guard — composed-weight norm sanity.** Reject any hypothesis whose composed weight is structurally implausible: if ‖w2@w1‖ ≫ ‖w1‖·‖w2‖ sanity ratio — concretely, if the composed norm exceeds the product-of-norms reference by more than a configured factor (matrix products can inflate norms when the factors are misaligned; a product far outside the sane band is a composition artifact, not a discovery), reject before it ever reaches consolidation. This guard runs **before** measurement, is reported per rejection, and is the cheap first filter so the expensive measurement loop is not spent on garbage.

**E5. Acceptance then means:** hallucination guard passed ∧ ε satisfied on measured utility ∧ forgetting risk within bound ∧ (owner-optional) stability within bound. Only then does consolidation run — through the connection's own `consolidate()` (§5), never through a direct θ-write. The consolidator's remaining bypasses (line 222, lines 305–337) are **inside the scope of this reimplementation**: the re-run spec requires dream-side writes to route through the connection's law, and the retired paths are removed or disabled behind a flag when the new path lands.

**E6. Statistics must reflect truth.** Simulator/evaluator counters (accepted/rejected/deferred) and engine cycle counts continue as today (owner fix 1 counting already landed); any hypothesis rejected by E4 is counted in its own bucket so the acceptance rate means something.

---

## 4. What changes in gate-5's meaning

**Today, gate 5 is unanswerable.** Its question — *does imagination enable new discoveries over replay?* — needs a working D1 and a non-artifact D2. The pilot could not supply either: D1 is bit-identical to D0 (§1.1), so the replay baseline does not exist; D2's 60/60 acceptance was constant-table arithmetic (§1.3) and its writes were artifacts (§1.2). That is why `docs/RESULTS.md` §2.1 records the row as **unanswered** — never as reported either way.

**With a real dreamer, D2-vs-D1 becomes answerable**, and the re-run plan is deliberately boring: **the same preregistered six-configuration matrix, the same corrected instrument, the same holdout** (`scripts/run_pilot.py --full` + `--gate5` equal-compute mode; lag-1 targets, phase-block resets, oracle lagged identically — `scripts/eval_fix_report.md` Part 1, run-2 protocol), 3 seeds, same reporting law (absolute errors primary, similarity alongside).

**Prediction (falsifiable, stated before the run):** the current numbers would be **REPLACED**, specifically:

- The identity **C ≡ D ≡ E ≡ A on acc@1** (composition-accept collapse: every composition accepted at the 0.698/0.810 arithmetic) disappears — under E1–E5, acceptance becomes a measured property and the 60/60 precision-1.0 row cannot recur.
- The **D2/D3 artifact rows** (θ 0 → 9.81/6.66; topology 36 → 61/72 with 0 counted creations; retention 1.0810) are replaced by measurements from a path that routes through the output-preserving solve — θ can only move via a law whose output preservation is asserted at 1e-6, so the artifact signature is structurally impossible to reproduce.
- D1 stops being bit-identical to D0; whether replay then *helps* retention is an open empirical question this spec deliberately does not predict.
- D2-vs-D1 gets a real verdict for the first time; D3's rotation numbers (A 0.7213 / retention 0.9996) are re-measured on the new path, not carried forward.

If the re-run instead reproduces artifact signatures, that falsifies the diagnosis and is reported as such.

---

## 5. Safeguards preserved (owner laws)

1. **Dreams cannot validate themselves — and neither can a teacher.** README law: *"A dream cannot validate itself — and neither can a teacher. Reality (executable environments, held-out observations) is the referee"* (`README.md` line 108). Codified here as: acceptance requires **measured improvement on stored replay experiences**, never the evaluator's own constants (§3 E1–E3); the pilot's held-out A→C battery remains the external referee the dream loop cannot touch.
2. **Dream-side consolidation routes through the connection's output-preserving `consolidate()` only.** The annealing exact-solve, selective gates, rate-0 disable and 1e-6 preservation invariant are protected property 4 (`docs/APC.md` §4–§5); no dream path writes θ directly. The reimplementation's acceptance criteria include retiring the two bypass paths (§3 E5).
3. **Provenance separation (imagined vs actual).** Replayed and imagined contributions are tagged separately in decision records (§2 R5, §3 E6); dream-derived connections are identifiable after the fact. This extends the PLASTIC-vs-RETRIEVED rule the disconnect experiment already enforces (`docs/RESULTS.md` §1.4).
4. **Control validity.** Plastic-D0 and rate-0 variants must not consolidate via the dream path any more than via the live path — `consolidation_rate = 0` disables outright (Test 10; `docs/APC.md` §4.1). The dream engine respects the same disable.
5. **Freeze discipline.** Any mechanism change here — replay path, evaluator sum, acceptance law — requires a new freeze manifest in `freeze/` before its results are citable (`docs/APC.md` §9.2). The v2 freeze covers `src/nnfs/connections/**` only; dreaming is explicitly outside it, so the dreaming work does not disturb the frozen plasticity baseline.
6. **Negative results are records.** The stub-replay D1≡D0 row and the artifact D2 rows stay in `pilot_results/` and the report regardless of what the re-run shows (`docs/APC.md` §9.4; `docs/RESULTS.md` §5).

---

## 6. Decision points for the owner

| # | Decision | Options | Default if deferred |
|---|---|---|---|
| D1 | Approve this spec as the reimplementation of record | approve / amend (name the section) / defer again | stays deferred; `docs/RESULTS.md` §2.1 unchanged |
| D2 | Storage rebuild (§2 R1): ring rebuilt for fixed-shape windows vs compatibility shim over flattened store | rebuild (clean, discards stored flat vectors) / shim (keeps both) | — |
| D3 | Measurement isolation (§3 E1): lightweight copy of affected state vs install-then-restore on the live network | copy (safer, more memory) / install-restore (cheaper, needs strict restore proof) | — |
| D4 | Evaluator weight vector and ε (§3 E2–E3): confirm 0.3/0.2/utility-slot/0.2/0.3 layout and ε default | confirm / amend values | — |
| D5 | Effort approval (estimates below are engineering estimates, not measurements; CPU-only machine, `docs/RESULTS.md` §4) | — | — |
| D6 | What the first re-run measures: full six-config matrix now, or D1-only first (replay validity alone) as a staged gate | full / staged | — |

**Effort estimate per section** (rough, CPU-bound, single-machine):

| Section | Work | Estimate |
|---|---|---|
| §2 R1 storage | retriever shape-preserving store + tests | small–medium |
| §2 R2–R4 replay | replay_batch/episodic real implementation + budget accounting | medium |
| §3 E1 measurement loop | install/measure/restore harness + scoring | medium–large (the bulk) |
| §3 E2–E6 evaluator | new sum, ε gate, hallucination guard, stats | small–medium |
| §3 E5 consolidator retirement | remove/disable the two θ-write bypasses + tests | small |
| Re-run | six-config matrix, 3 seeds, on CPU (pilot runs ~120–190 s/seed live; dream compute grows with the real loop) | run-time, not build time |

**What the first re-run measures**, in one line: whether replay (D1) is a real intervention, whether imagination (D2) measured against it adds held-out discovery, and whether the artifact signatures are gone — on the same instrument, the same holdout, under the same reporting law.

---

## 7. Unknowns (stated as unknown)

- The exact confidence values the recombiner produced in the pilot (the 0.698 point in §1.3 assumes c = 0.7; the code path supports any c). Not measured in the report; not load-bearing for the spec.
- Whether the blend-1.0 consolidation no-op (§1.1, secondary contributor) remains a live-path factor after the v2 anneal fix — the gate-6 rerun says the mechanism now fires (`docs/RESULTS.md` §1.3); its live continual behaviour at rate > 0 is still unproven (`docs/APC.md` §8).
- D3's rotation advantage (§1.2): real effect vs artifact residue — unexplained here by design; the re-run re-measures it.
- Replay compute cost on CPU at real budget: unmeasured until run.

---

*Docs-only: this document changes no code and runs nothing. Where any figure here and its source file disagree, the source wins (`docs/RESULTS.md` closing rule). The owner's decision on §6 D1–D6 gates all implementation; nothing proceeds without it.*
