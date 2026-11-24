# AION Evaluation Fix Report — metric repairs + gate-5/6 pilot evidence

**Project:** AION — Adaptive Intelligence Organism Network (code package still `nnfs`; renames deferred until validation gates pass)
**Date:** 2026-11-24 · **Author:** eval-metrics lane · **Scope:** benchmarks/, experiments/, scripts/, tests/test_eval_metrics.py
**Connections/dreaming untouched** (Lane A owns them; freeze law honoured — zero mechanism changes).

---

## PART 1 — Gate-5/6 pilot evidence (six-configuration pilot, owner gate 5)

Two pilot runs are on record:

- **Run-1** (`pilot_results_unlearnable_instrument.json`, `pilot_results_d1x2_unlearnable_instrument.json`) — honest NULL. The original benchmark tied every target to the CURRENT input (`y = R@x_t`, fresh random x each step): informationally unlearnable from state history (Lane A's gate-4 diagnosis). All six configs bit-identical, conn-MSE ~ E[target^2] ~ 1.00 everywhere. Preserved as the preregistration record of the instrument defect.
- **Run-2** (`pilot_results.json`, `pilot_results_d1x2.json`) — after documented protocol evolution (lag-1 targets in all three datasets with phase-block resets; oracle lagged identically; lr 0.05; mean-vs-sum fix in relative_to_oracle/random). The instrument proved ALIVE in the required single-cell gate before the burn (rotation Plastic D0: phase A 0.7558 / C 0.7581 vs B 0.8192 — learned AND retained; compositional 0.9694 < 1.0 baseline).

### Research rows (run-2, mean of 3 seeds)

| # | Question | Verdict | Evidence |
|---|---|---|---|
| 1 | Plastic > Static (does online learning improve adaptation?) | **YES, clearly** | Rotation phase A 0.7508 vs 1.0008 static (-25.0%); phase C 0.7529 vs 1.0043. Discovery error 0.9696 vs 1.0032. The static control no longer learns (its rate-0 is actually applied now). |
| 2 | Consolidating > Plastic (does consolidation improve retention?) | **NO — within noise** | Rotation A 0.7501 vs 0.7508; retention 1.0013 vs 1.0029; discovery 0.9791 (slightly WORSE). **Gate 6 FAILS** — see below. |
| 3 | D1 > D0 (does replay prevent forgetting?) | **NO — bit-identical** | D1 = D0 on every field, every seed (0 hypotheses, theta 0 -> 0, topology unchanged). Replay is currently inert for blend-1.0 plastic variants. Dreaming-lane finding; frozen, reported only. |
| 4 | D2 > D1 (does imagination enable new discoveries?) | **NO — interpretation reading #4** | D2 accepted 60/60 hypotheses (precision 1.0) but held-out A->C unchanged: 0.9694 vs 0.9696. D2 actively HURT rotation retention (1.0810 vs 1.0029; phase C 0.9326 vs 0.7529, worse in all 3 seeds). Imagination generates ideas, not useful discoveries, at 5.5x D1's dream compute. |
| 5 | D3 > D2 (does structural dreaming add value?) | **MIXED** | Rotation: D3 is the BEST config overall (A 0.7213, C 0.7210, retention 0.9996 — the only variant <= 1.0). Compositional: slightly worse (0.9767 vs 0.9694). Structural dreaming helps adaptation/retention, not discovery. |

### Gate 5 measurement table (compositional discovery, mean of 3 seeds)

| Variant | Held-out discovery error | Discovery precision | Retention (rotation) | Dream compute (s) | Accepted hypotheses |
|---|---|---|---|---|---|
| D1 (equal-observation) | 0.9696 | None (no hypotheses generated) | 1.0029 | 0.06 | 0 |
| D2 (replay + imagination) | 0.9694 | 1.00 (60/60) | 1.0810 (worse) | 0.33 | 60 |
| D1x2 (equal-compute, 2x dream budget) | 0.9696 | None | 1.0029 | 0.17 | 0 |

**Dual-budget reading:** D2 wins equal-observation by 0.0002 (noise) and ties equal-compute exactly. D2 does not beat MORE REPLAY; the added compute bought nothing measurable on held-out A->C, and degraded rotation retention.

### Gate 6 — consolidation acceptance: **FAILS** (independent of gate 5)

| Criterion | Result |
|---|---|
| 1. Consolidation actually fires | 17 sweep calls per run (interval 50) — yes, but **0 elements transferred in every config** |
| 2. theta_slow demonstrably changes | **NO** for Consolidating D0: 0.0000 -> 0.0000. Selective gates stay shut (gate_mean 0.062 vs threshold 0.5; they close further under O(1) prediction error). (D2/D3 theta moved 0 -> 9.81/6.66 via the dream engine's own consolidation path — outside my instrumentation, dreaming-lane domain.) |
| 3. Effective-connection output preserved during transfer | YES — max diff 0.0 (tolerance 1e-6); trivially, since nothing transferred |
| 4. Held-out retention improves vs Plastic D0 | 1.0013 vs 1.0029 — within noise |

Reported as a FAIL per owner directive, not tuned away. Observation for the owner (no change proposed, freeze law): the evidence-driven gates need LOW prediction error to open, but at this data scale error stays O(1), so consolidation can never trigger — a design deadlock, not an experiment knob.

### [VALIDATION] block (run-2, seed 42, full-size phases)

```
[VALIDATION] model=static seed=42 phase_length=300
Phase A MSE: 1.0017
Phase B MSE: 1.0018
Phase C MSE: 1.0044
Adaptation steps: B=None, C=None
Discovery error: 1.0030

[VALIDATION] model=plastic seed=42 phase_length=300
Phase A MSE: 0.7558
Phase B MSE: 0.8192
Phase C MSE: 0.7581
Adaptation steps: B=None, C=None
Discovery error: 0.9694

[VALIDATION] model=consolidating seed=42 phase_length=300
Phase A MSE: 0.7549
Phase B MSE: 0.8203
Phase C MSE: 0.7561
Adaptation steps: B=None, C=None
Discovery error: 0.9782
```

Three-metric contract, seed 42 rotation training (last 100 steps): static conn 1.0044 / state 0.9107 / readout 1.1154; plastic conn 0.7588 / state 1.0034 / readout 1.2258; consolidating conn 0.7545 / state 0.9950 / readout 1.2220. The untrained readout moves INDEPENDENTLY of learning — confirming it must never be the learning signal.

**Reporting law (absolute errors alongside similarity):** discovery errors above are absolute MSE. Similarity-style diagnostics (relative_to_oracle / relative_to_random) are per seed in both JSONs; run-2 plastic D0: relOracle 387.5, relRand 0.4844 (corrected mean-vs-sum computation).

### Differentiation verdict

Static != Plastic != Consolidating holds on the corrected instrument (rotation phase A 1.0008 / 0.7508 / 0.7501; discovery 1.0032 / 0.9696 / 0.9791), and the six dream levels separate (D3 best on rotation; D2 worst on retention). Adaptation steps remain None for every variant at 900 steps: the model improves ~25% on phase error but never halves the switch-time error within a 300-step phase — the None is honest, not a metric bug.

---

## PART 2 — The four blocker root causes and metric repairs (earlier phase)

1. **Phase B MSE = 0.0000** — `evaluate_adaptation_retention` (src/nnfs/benchmarks/validation.py) never returned a "phase_b_mse" key; `pilot_experiment.py:_evaluate_changing_rotation` read `results.get("phase_b_mse", 0)` -> default 0. Missing key, not divide-by-zero. Fixed: phase B computed (tail-50 mean; the -45 deg phase transform verified by test).
2. **Adaptation steps = 4500 (trivial max)** — three compounding bugs in the same function: scalar batch MSE appended once PER BATCH ELEMENT (15 test sequences x 300 steps = 4500); eval ran learn=False so a frozen network's per-phase MSE is flat (adaptation unmeasurable by construction); the no-crossing branch returned len(mses). Fixed: one entry per network step; eval learns (learn=True); steps 1-indexed from phase start; None when the criterion is never reached.
3. **Discovery score = 1.0 everywhere** — (a) the pilot built a SECOND CompositionalDiscoveryDataset for ground truth; its random T_AC differed from the data's T_AC (torch RNG had advanced) -> oracle_mse ~2.17, worse than random (~2.00); (b) `1 - (test-oracle)/max(random-oracle, 1e-8)` with a negative denominator forced denom to 1e-8 -> quotient exploded -> min(1.0, ...) pinned everything to 1.0. Fixed: one dataset instance shared with the loaders; score = raw unclamped held-out error.
4. **Static = Plastic = Consolidating identical / MSE flat ~1.02** — the measured out["mse"] came from RelationalNetwork.prediction_head, a random never-trained linear head, blind to plastic learning; plus RelationalNetwork.__init__ (pre-gate-4 line 85) hardcoded connection learning_rate=0.01, so the "static" control (rate 0) was secretly learning. Fixed harness-side: metric contract (connection_prediction_mse primary; state/readout separate; loud KeyError, no legacy fallback), per-variant rates applied to connections (network now propagates them post-gate-4; belt-and-braces retained).

### Edits made (all in my turf)

- `src/nnfs/benchmarks/validation.py` — metric contract helpers (REQUIRED_METRIC_KEYS, benchmark_metrics loud-error, primary_mse, tensor_pair_mse shape assert); phase_length params on datasets; lag-1 targets in all three datasets with phase-block resets (compositional) and stream lag (rotation, context); single-dataset loaders; evaluate_compositional_discovery per-sequence lag-consistent oracle + unclamped discovery + mean-based ratios; evaluate_adaptation_retention learn=True, per-step tracking, real phase_b_mse, 1-indexed phase-relative adaptation, None on no-criterion; evaluate_context_switching same contract.
- `src/nnfs/experiments/pilot_experiment.py` — shared ground-truth dataset; PilotNetwork rate application; relation_type="post" + state_leak_rate 0.5 (gate-4 values); consolidation_interval wiring + CONSOLIDATION_INTERVALS=[10,25,50,100] + gate-6 instrumentation (calls/counts/gate stats/theta norms/output-preservation check); three metric curves; Optional None semantics (adaptation, D0 dream precision); incremental JSONL persistence + resume-skip + progress lines in run_pilot_matrix; configs/experiment_kwargs/output_name params.
- `configs/default.yaml` — pilot.consolidation_interval: 50.
- `scripts/run_pilot.py` — matrix gated behind --full; --gate5 equal-compute D1x2 mode; None-safe prints; AION banner.
- `scripts/run_validation.py` — default = small diagnostic printing the [VALIDATION] block; full validation blocked (SystemExit).
- `tests/test_eval_metrics.py` — 9 tests (see below).

### Test status

`python -m pytest tests/ -q`: **97 passed, 3 failed (pre-existing), 0 xfail.**
- Mine (`tests/test_eval_metrics.py`): **9/9 PASS** — missing-keys loud error / no legacy fallback; shape mismatch fails loudly; readout cannot mask connection learning; live RelationalNetwork learning test (UN-XFAILED after Lane A's gate-4 fix: lag-1 recipe, relation_type="post", state_leak_rate=0.5, lr=0.05 — asserts >=5% primary-metric improvement, corr(transmitted, target) growth > 0.05, readout flat); adaptation 1-indexed phase-relative (8/6 on scripted curves) + None when flat; discovery unclamped; D0 dream precision null; Phase B uses -45 deg and is measured; no oracle/target leakage into the prediction path.
- Pre-existing failures, NOT mine (`tests/test_training.py` imports only nnfs.training.*): test_mnist_loaders, test_compute_metrics_perfect (float epsilon in training/metrics.py), test_analyze_model_complexity ('ReLU' missing in training/trainer.py layer_type_counts).

### Findings outside my turf (observations only, freeze law)

1. `connections/network.py` (pre-gate-4 line 85) hardcoded connection learning_rate — Lane A has since fixed propagation; my belt-and-braces rate setter remains harmless.
2. `connections/plastic.py` consolidate(): blend_ratio >= 1.0 no-ops consolidation BY DESIGN — so PLASTIC (blend 1.0) and any replay variant on it can never consolidate; owner may want to know D1's inertness traces partly to this.
3. Dreaming-lane: D1 replay produced zero observable effect (bit-identical to D0, all seeds); D2's dream-side consolidation moved theta to norm 9.81 and DEGRADED rotation retention; my connection counters recorded 0 created while topology grew 36 -> 61/72 — the engine's creation path bypasses the dream_stats counters I read.
4. `training/metrics.py` / `trainer.py`: the 3 pre-existing test failures above (owner action).

### Reproduce

```
C:/Python313/python.exe scripts/run_validation.py          # small diagnostic + [VALIDATION] block
C:/Python313/python.exe -m pytest tests/test_eval_metrics.py -v
C:/Python313/python.exe scripts/run_pilot.py --full        # six-config matrix (incremental + resumable)
C:/Python313/python.exe scripts/run_pilot.py --gate5       # equal-compute D1x2 supplementary
```

Result files: pilot_results/pilot_results.json (run-2), pilot_results_d1x2.json, *_unlearnable_instrument.json (run-1 record), pilot_results_prefix.json (pre-fix record), pilot_results_incremental.jsonl (durable per-run log).
