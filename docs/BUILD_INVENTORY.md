# BUILD INVENTORY — what this repository actually contains

**Date:** 2026-11-25 · **Method:** docs-only survey grounded in the tree, the tests, the git log and the result files. Every number below was read from a repo source (file, JSON, `git log`, or a live pytest run on this machine); nothing is assumed. Where this document and a source disagree, the source wins.

---

## 1. The organism in one paragraph

AION (Adaptive Intelligence Organism Network) is an experimental research project — code package `nnfs` — built around one question: can a system whose defining ability is *learning* be raised on a borrowed substrate and survive its loss? The substrate is the plastic-relational core (APC, `docs/APC.md`): per-connection dual-timescale state (`u_fast` learns error-driven from the actual forward output, `theta_slow` is written only by an output-preserving annealing consolidation) that attaches to a frozen host brain — real Qwen3.5-4B-Base on CPU BF16, or the LittleBit-derived Pith-E2B artifact through a Rust bridge — via hidden-state hooks and a zero-initialised γ-residual pathway. Around that core sits the host lane: synthetic dialogue worlds, a teacher/student protocol, provenance-tracked memory (episodic / relational / consolidation / developmental-history ledger), the NOMEN naming system, the host-mode authority runtime, the NAV decision head, and the Disconnect Experiment that severs the host and proves owned knowledge survives restart bitwise. The register is deliberately honest: promising mechanism, several capabilities demonstrated, several honest negatives preserved (conversational acquisition FAIL, dreaming inert, consolidation no retention benefit yet) — `docs/RESULTS.md` is authoritative for claims.

**Repo scale (measured 2026-11-25):** 317 tracked+present files (excluding `.git`, `__pycache__`, pytest caches, `*.tar.gz`). Python: 138 files / **35,652 LOC** (`src/` 54 files, `aion/` 70, `tests/` 8, `scripts/` 6), plus **365 lines of Rust** (`aion/relational/pith_bridge/`). Git history: 16 commits, all 2026-11-25 (initial public release → owner decision-docs pack). Large data: `data/` ≈486 MB legacy MNIST/FashionMNIST/CIFAR-10 downloads; `aion/experiments/data/` ≈300 MB TinyStories corpus + BPE tokenizer + `train.bin` (deprecated from-scratch lane; gitignored). Root `output.txt` is an empty 0-byte gitignored artifact. Test suites: **247 pytest tests passed + 1 skipped** on this machine (104 in `tests/`, 143 across the aion suites), all green.

---

## 2. Component map

### 2.1 `src/nnfs/` — the NNFS learning mechanism (54 files, 13,813 LOC)

| Component | Files | LOC | Tests | Status | Evidence pointer |
|---|---|---|---|---|---|
| `connections/` — plastic relational core | 4 | 1,433 | 11 (`tests/test_connections_learning.py`) | **Frozen** (v1 + v2 SHA-tagged) | `freeze/AION_LEARNING_BASELINE_20261124.md`; `docs/APC.md` §6 registry; −43.8% held-out + coefficient recovery cos 0.999998 (`docs/RESULTS.md` §1.1) |
| — `plastic.py` | 1 | 853 | — | The mathematics: `EligibilityTrace` (23–131), `RelationshipFunction` (134–261, families bilinear/post/mlp/attention), `FastSlowPlasticity` learning law + annealing exact-solve consolidation (264–524), `PlasticRelationalConnection` (644–853) | `docs/APC.md` §7 line table |
| — `network.py` | 1 | ~200s | — | `RelationalNetwork`: composition + in-degree credit split (`target / in_degree`) | `network.py:202–206` (APC §5 P2) |
| — `consolidation.py` | 1 | — | — | Consolidation management (scheduling) | APC §4.1 |
| `dreaming/` — Dream Engine | 13 | 2,924 | 0 (covered by pilot evidence only) | **Diagnosed; re-implementation approved-staged, not built** | `docs/DREAM_REIMPL_SPEC.md` §1 (three root causes, file+line); `docs/D1_REPLAY_VALIDATION_PLAN.md`; D1 bit-identical to D0 (`docs/RESULTS.md` §2.1) |
| — subsystems | retriever, replay, recombiner, simulator, evaluator, consolidator, structural, engine | — | — | `replay.py` is a **stub** (body ends `pass`; zero `network.step` calls in the dream path) | spec §1.1 |
| — `benchmarks.py`, `experiments.py`, `experiments/run_*.py` | 4 | — | — | separate dream benchmark/experiment scripts (the only `network.step` callers under `dreaming/`) | spec §1.1 |
| `benchmarks/` — evaluation tasks | 7 | 2,306 | 9 (`test_eval_metrics.py`) | Corrected instrument live (`validation.py` lag-1); legacy rotated-MNIST / continual / sequential / xor_parity / circular | `scripts/eval_fix_report.md` Part 1 (unlearnable-instrument null → run-2 fix) |
| `experiments/` — pilot & validation runners | 8 | 3,537 | — | Six-config pilot harness + ablation/comparison/continual runners | `pilot_results/pilot_results.json`; `scripts/run_pilot.py` |
| `training/` — trainer + metrics | 3 | 542 | 10 (`test_training.py`) | Green (3 env failures fixed at root cause, commit `3ca4f3c`) | git `3ca4f3c` |
| `models/` — standard / multiplicative / geometric MLPs + specialized net | 5 | 792 | 21 (`test_models.py`) | Legacy architecture baselines | `tests/test_models.py` |
| `neurons/` — standard / multiplicative / geometric neurons | 4 | 560 | 18 (`test_neurons.py`) | Legacy alternative-neuron math | `tests/test_neurons.py` |
| `layers/` — 5 specialized layers (detection, interaction, competition, integration, prediction) | 6 | 1,153 | 22 (`test_layers.py`) | Legacy specialized-network lane | `tests/test_layers.py` |
| `memory/` — fast/slow multiscale memory | 2 | 387 | 13 (`test_memory.py`) | Legacy multi-timescale memory | `tests/test_memory.py` |
| `config.py`, `__init__.py` | 2 | — | — | Config dataclasses + package doc | `configs/default.yaml` |

### 2.2 `aion/` — the host lane (70 py files, 19,721 LOC + 365 Rust)

| Component | Files | LOC | Tests | Status | Evidence pointer |
|---|---|---|---|---|---|
| `foundation/` — frozen host brain | 5 | 620 | 1 passed + 1 skipped (`test_pretrained_loader.py`; skip = real checkpoint only on Lappy) | Qwen3.5-4B-Base loader (frozen, fingerprinted); `model.py` AION-Seed-Base tiny transformer; `tokenizer.py` + `train.py` **DEPRECATED** 11-24 with the from-scratch lane | `aion/experiments/test_relational_smoke.py` docstring; Phase-1 foundation bit-identity (426 params, `runs/teacher_student/qwen-real-c5/report.json`) |
| `relational/` — γ-residual interface + Pith bridge | 2 py (359) + 365 Rust | — | script smoke (`test_relational_smoke.py`) | Interface proven on BOTH substrates: Qwen dense 4/4 + Pith factor-executed 4/4, bit-exact identity at γ=0 | `aion/experiments/pith_smoke_results.json` (`verdict` all four criteria true; checkpoint `pith-e2b-1p5-dualsvid`, 315 factor tensors) |
| — `pith_bridge/` (Rust) | 2 | 365 | R3b-gated externally | READ-ONLY consumer of the validated R3b crate; protocol `bf16-forward-v1` by construction | `pith_bridge/src/main.rs` header |
| `dialogue/` — teacher/student apparatus | 7 | 1,270 | 26 (`test_dialogue.py`) | Session state machine, teacher client (OpenAI-compatible endpoint), lesson extractor, student, correction-as-plastic-update | `runs/teacher_student/qwen-real-c5/` (P1 executed on real endpoint) |
| `worlds/` — synthetic worlds | 2 | 316 | — | 100 invented chain relationships + 4 question batteries (direct/combination/paraphrase/intervention); names held-out so pretrained knowledge cannot answer | `aion/worlds/generator.py` docstring; `verify_world` gate in every run report |
| `memory/` — provenance apparatus | 9 | 2,138 | 10 (`test_apparatus.py`) + 13 (`test_devhistory.py`) | Episodic / relational (owned-knowledge index) / consolidation ledger / exact state serialization + `devhistory.py` (the individual's biography ledger) + CPU stubs + fixture battery | restart bitwise-exact: `state_restored_exact: true`, `bitwise_tensors: "ALL-BITWISE-EQUAL"` (`runs/disconnect_real_seed0/disconnect_report.json`) |
| `nomen/` — NOMEN naming system | 10 | 2,173 | 37 (`test_nomen.py`) | Entity registry, name generation (names emerge from observation), collision detector, identity resolver, lineage graph, evolution tracker, self-naming, exact JSON snapshot | `aion/nomen/test_nomen.py` (37/37); README NOMEN section |
| `nav/` — NAV decision lane | 12 | 3,417 | 15 (`test_nav_head.py`) | Deterministic graph-derived dataset (13 features, frozen split), trained logistic head (800-row holdout 1.000), d1-as-is arm, Tev1 backend, exam harnesses (original + amended A1/A2), post-hoc roster diagnostic, step-7 adaptation seam | `aion/nav/artifacts/nav_decision_head_meta.json` (holdout 1.000, n=800); `nav_head_exam_20261125_073813.json` (trained_head accuracy 1.0, 30/30); transfer-failure record `results/nav_exam_20261125_045356.json` (larql head 0.20) |
| `hostmode/` — authority-gated runtime | 10 | 2,557 | 18 (`test_hostmode.py`) | THE RUNTIME is the authority (proposers only propose); closed op enum + caps; capacity grants; own-state root; two exams + demonstration trace; sandbox demo artifacts | `results/hostmode_results.json` (20 episodes/arm, seed 20261125); `results/demonstration_trace.json` |
| `experiments/` — the AION-1 experiment lane | 12 | 6,870 | 14 (`test_disconnect.py`) + 9 (`test_ownership_scale.py`) + script smokes | `teacher_student.py` (Phase 1 + Phase 2 A–E harness), `disconnect.py` (the defining milestone), `knowledge_discovery.py` (capability battery + survival metric), `retention.py` (strict retention test + provenance ablation), `ownership_scale.py` (100-relationship prereg; untracked, run not yet stored), `pith_interface_smoke.py`; `prepare_data.py` DEPRECATED | §4 of this document |

### 2.3 `tests/` + `scripts/` (14 files, 2,108 LOC)

- `tests/` — 8 files, 1,516 LOC, **104/104 passed** (measured: 11.6 s). Per file: `test_connections_learning.py` 11, `test_eval_metrics.py` 9, `test_layers.py` 22, `test_memory.py` 13, `test_models.py` 21, `test_neurons.py` 18, `test_training.py` 10 (+`__init__.py`).
- `scripts/` — 6 py files, 592 LOC: `diagnostic_learning.py` (gate-4 acceptance runner, ~30 s rerun), `run_pilot.py`, `run_validation.py`, `run_dreaming.py`, `run_continual_benchmark.py`, `run_experiment.py`; plus `scripts/eval_fix_report.md` (120 lines — the full experimental record).
- Script-form acceptance smokes (NOT pytest-collected, run directly): `aion/experiments/test_smoke.py` (tiny transformer forward/backward/loss-decreases), `aion/experiments/test_relational_smoke.py` (stage-1/2 four-criterion acceptance).
- aion pytest suites (all measured green): dialogue 26, nomen 37, memory apparatus 10, devhistory 13, hostmode 18, nav 15, disconnect 14, ownership-scale 9, pretrained-loader 1 passed + 1 skipped → **143**. Grand total **247 passed + 1 skipped**.

---

## 3. The documentation set

| Document | Lines | Purpose |
|---|---|---|
| `README.md` | 162 | Project identity ("Raise your own AION") + current stage table (00 Seed → Maturation, honestly marked), design laws, APC summary, repo layout, environment. The canonical current-state doc. |
| `docs/APC.md` | 295 | **The founding math document**: forward map, learning law (exact-gradient derivation + scope condition J_ψ), annealing exact-solve consolidation, the five protected properties with their proving tests, the substrate-agnostic learning contract, versioned freeze registry, change control. |
| `docs/RESULTS.md` | 155 | The consolidated honest public results register: demonstrated / measured-negative / in-flight, every number copied from a named source file. Authoritative for claims. |
| `docs/PHASE3_PROTOCOL.md` | 167 | Phase-3 three-arm expression protocol (density knob × residual-alignment), owner decisions D1–D9 locked 2026-11-25; behavioural success condition is owner law. *(Modified in working tree at survey time.)* |
| `docs/DREAM_REIMPL_SPEC.md` | 164 | Approved staged Dream Engine re-implementation spec: the three diagnosed root causes (file+line), requirements R1–R6, owner decisions D1–D6 recorded. |
| `docs/D1_REPLAY_VALIDATION_PLAN.md` | 63 | The staged gate before any dreaming claim: prerequisites P1–P6, validation measures V1–V4, pass/fail, and what the gate does NOT authorise. |
| `docs/COMPLIANCE_REQUEST_DRAFT.md` | 105 | Draft organiser-ruling request for substrate (Qwen-derived) use — HOLD per owner decision S1; nothing sent from this project; no claim rides on an unresolved ruling. |
| `scripts/eval_fix_report.md` | 120 | The experimental record of the metric repairs: run-1 unlearnable-instrument null, run-2 corrected instrument, research rows 1–5, gate-5 table, gate-6 FAIL, [VALIDATION] block, reporting law. |
| `HANDOVER_SUMMARY.md` | 161 | **SUPERSEDED** (banner, 2026-11-25) — the 11-24 handover preserved as history; its flagship claim ("network isn't learning") traced to an untrained-head metric, not dead plasticity. Honest-null preservation per APC §9.4. |
| `README_DREAMING.md` | 291 | Legacy pre-diagnosis Dream Engine overview (architecture, math framework, research directions). Predates the diagnosis; its claims are superseded by `docs/RESULTS.md` §2.1. Note: cites a `LICENSE` file that is **not present** in the repo. |
| `freeze/AION_LEARNING_BASELINE_20261124.md` | 48 | v1 freeze manifest: scope, SHA-256 + per-file hashes, chosen calibration point, gate-4 evidence table, known limits. |

---

## 4. The experimental record (runs + artifacts)

### 4.1 Freeze manifests (`freeze/` — hashes verified in this survey)

| Version | Archive | SHA-256 (verified matches docs) | Contents |
|---|---|---|---|
| v1, 2026-11-24 | `aion_learning_baseline_20261124.tar.gz` | `a39013c90fb5f36118a76c35d3a6d5dde01d8c903a576a892d89a75218132daf` | Gate-4 baseline; suite 7/7 |
| v2, 2026-11-25 | `aion_learning_baseline_v2_20261125.tar.gz` | `ea802b9603aaf4a9e91816a8d8b59ddabeedb30a2ca36906762084d9f38734af` | Deadlock-fix baseline; suite grows to 10 functions (tests 7–10) |

### 4.2 `pilot_results/` (12 files) — the six-configuration pilot

- `pilot_results_unlearnable_instrument.json` + `pilot_results_d1x2_unlearnable_instrument.json` — **run-1 honest NULL** preserved: targets tied to current input, all six configs ≈1.0048 (the preregistration record of the instrument defect).
- `pilot_results.json` — run-2 six-config, mean of 3 seeds (rotation + compositional): Plastic D0 vs Static D0 phase A 0.7508 vs 1.0008 (**−25.0%**), phase C 0.7529 vs 1.0043, discovery 0.9696 vs 1.0032; consolidating within noise; readout independent of connection learning.
- `pilot_results_d1x2.json` + `pilot_results_d1x2_incremental.jsonl` — equal-compute control: D2 ties D1 exactly (0.9696 vs 0.9696) at ~2× dream compute (0.33 s vs 0.17 s).
- `pilot_results_gate6_consolidation.json` + `..._incremental.jsonl` + `gate6_consolidation_run.log` — the repaired-consolidation rerun: gate mean ≈0.44/0.51, 45,924–88,801 elements transferred, θ_slow norms 30.99–112.81, output preservation max ‖Δŷ‖ ≤ **2.4e-7**.
- `pilot_results_prefix.json`, `pilot_full.log/.err`, `pilot_gate5.log/.err`, `validation_diagnostic.json` — run logs and the pre-burn instrument-alive diagnostic (single-cell gate).

### 4.3 `runs/` — the AION-1 conversational + disconnect record

- `disconnect_real_seed0/` (13 files) — **the Disconnect Experiment on real Qwen3.5-4B-Base**, seed 0, battery `disconnect-tiny`, 12 seed relations: 9 phase files (`teach` → `observe` → `register` → `disconnect_host` → `disconnect_teacher` → `simulate` → `restart` → `retest` → `phase_*`), `disconnect_report.json` (register: survival 1.0/1.0/**0.75**/0.75/1.0 raw; every owned row PLASTIC; epoch-adjusted 1.0 via `host_rescored_post.json`), `state_disc.pt`/`state_host.pt`. Structural proof measured: `foundation_forward 0`, `foundation_load 0`, `teacher_call 0`, `iface_absent`, `monitor_closed`, `poison_loader_intact`; restart `bitwise_tensors: "ALL-BITWISE-EQUAL"`, `state_restored_exact: true`.
- `disconnect_stub_final/` + `dis_split_dbg/` (untracked) — harness-development runs on `stub-tiny-llama` (same 9-phase shape).
- `teacher_student/qwen-real-c5/` — **Phase 1 on the real endpoint** (`runs/teacher_student/qwen-real-c5/report.json`): acceptance **3/5** — acquisition FAIL (PLASTIC exact 0.00 everywhere, `empty_answer_fraction 1.0`; RETRIEVED direct 0.75), generalisation FAIL, retention/correction/transcript-removal PASS; timings teach 1,184.9 s + test 505.6 s. Companion: `episodic.json`, `episodic_after_purge.json` (21 events purged), `transcript.json`, `world.json`. Plus 4 `tiny-auto/scripted-c3` development runs (stub-teacher harness iterations).
- `teacher_student_p2/` — **Phase 2 A–E**: `p2-qwen-c5-s0-022716/report.json` (real 4B, 5 chains/60 experiences, A–E complete): verdict negative for expression, positive for localisation — C/D bitwise identical to A on scored rows (`plastic_ablation_delta 0.0`), E's γ pathway alive (`gamma_after [10.371428, 11.180095, 0.522552]`) but sub-threshold (gold rank moved ≤0.1; combination 0.3333 in every mode = A's frozen number). Dry-run validation: `p2-tiny-c3-s0-022348` (complete A–E) and `p2-tiny-c3-s0-022448` (A/B partial at time of RESULTS.md), plus two aborted tiny runs.

### 4.4 Per-lane results directories

- `aion/nav/results/` — `nav_exam_20261125_045147.json` + `_045356.json` (Rog: heuristic floor 1.0, larql head **0.20**, Tev1 0.8B **30/30 license-blocked**, accuracy null), `lappy_d1_exam_20261125_054944.json` (d1-2B **0.5333**; QUERY blind spot 0/6).
- `aion/nav/artifacts/` — `nav_decision_head.pkl` + `nav_decision_head_meta.json` (logistic, 13 features, val 1.000, holdout **1.000** n=800, 2,720 train / 480 val, per-class-bias perceptron adaptation seam), `nav_dataset.json` + `nav_dataset_manifest.json` (schema nav-dataset-1.0, frozen split), `nav_head_exam_20261125_073813.json` (amended protocol A1/A2: **trained_head 30/30 = 1.000**, 0 enum violations, d1-as-is arms 0.5333), `lappy_d1asis_exam_20261125.json` (0.5333 both arms), `d1_roster_diag.json` (post-hoc, not an exam row: original 0.70 / fresh 0.7333), `adapt_demo.json` (step-7 seam: degraded-prior arm recovers 0.87 → 0.99 → 0.98 → 0.98 over 4×100).
- `aion/hostmode/results/` — `hostmode_results.json` (20 episodes/arm, exam seed 20261125: plastic 0.95 success / 13.35 trials / 0.95 failed-writes vs fixed-random 0.9 / 25.9 / 1.85; heuristic 1.0 / 6.0; **0 authority violations in every arm**; impossible-task correct-stop plastic **1.0 (6/6)** vs fixed-random **0.0**; grants 8, 0 unneeded, post-grant success 1.0) and `demonstration_trace.json` (episode 1 arc: WRITE denied → REQUEST_CAPACITY granted on demonstrated need → WRITE succeeds → host verifier confirms → goal achieved).
- `aion/experiments/pith_smoke_results.json` — Pith-E2B factor-executed interface smoke: verdict `{1_frozen: true, 2_hooks: true, 3_identity: true, 4_nnfs: true}` (γ=0 bit-exact, factor fingerprint unchanged, artifact SHA unchanged).
- `aion/memory/fixtures/tiny_battery.json` — local CPU fixture battery for the retention/provenance unit tests.
- `aion/hostmode/sandbox/` — demo sandbox inputs (gitignored): `task_script.py`, `dataset.txt`, `report.txt`.

### 4.5 Legacy records (kept, honestly marked)

- `ablation_results/xor_test/` — legacy architecture-ablation study (6 configs × 2 seeds, `ablation_results.json` + `best_model.pt` checkpoints): all arms reach 1.0 on XOR — a legacy sanity study, superseded by the relational line.
- `continual_results/changing_rotation/comparison.json` — **a broken legacy record**: all five arms errored with a tensor-size mismatch. Preserved, not cited as evidence.
- Top-level `nnfs_*.tar.gz` handover archives (5) — gitignored packaging artifacts referenced by the superseded handover.

---

## 5. Git history — what each commit delivered (oldest → newest, all 2026-11-25)

| Commit | Delivered |
|---|---|
| `0a37dad` | AION: initial public release (the whole tree) |
| `047861b` | README: AION Plasticity Core (APC) — central scientific foundation section |
| `3ca4f3c` | Fix 3 pre-existing training test failures at root cause — suite fully green (104/0) |
| `79e91d4` | APC specification: `docs/APC.md` — the founding math document |
| `7a15bb1` | Developmental-history ledger: `aion/memory/devhistory.py` — the individual's biography |
| `5267331` | Mark 11-24 handover superseded — preserve history, point to current state |
| `92a50d4` | HOST MODE: AION's first measured host-manipulation capability |
| `dd55c37` | gitignore: hostmode sandbox runtime outputs (demo evidence already captured in `92a50d4`) |
| `6aac6e2` | DISCONNECT EXPERIMENT executed on real 4B — all five capabilities survive |
| `9a4b792` | README: gestation stage table refreshed — disconnect executed, host mode demonstrated, Phase 2 running |
| `640c67c` | `docs/RESULTS.md` — the consolidated honest public results document |
| `b295f57` | NAV step 6+7: trained NAV-domain decision head — 1.000 on all exams (preregistered) |
| `9386859` | AION-1 Phase 2: A–E executed on real 4B — honest negative on expression |
| `4aeffa5` | Phase-3 protocol draft: three-arm expression experiment (owner review) |
| `181363b` | Decision-docs pack: dream reimplementation spec + compliance request draft |
| `4253bb0` | Owner decisions recorded: Dream D1–D6 approved (staged) + compliance S1–S5 (hold/reconcile/verify) + D1 replay validation plan |

---

## 6. The evidence table (every number copied from its named source)

| Capability | Measured result | Source file |
|---|---|---|
| Connection-level learning | Held-out connection-prediction MSE 1.026 → 0.577 (**−43.8%**); per-seed −51.6%, −49.2% | `freeze/AION_LEARNING_BASELINE_20261124.md`; `docs/APC.md` §6, §8 |
| Coefficient recovery (effective mapping) | a = [2.0003, 2.9989] vs truth [2, 3]; abs err [0.0003, 0.0011]; cosine **0.999998**; predictions MSE 1.7e-5 | same |
| g²η calibration law | ‖Δŷ‖ = 0.000946 identical across (g,η) = (1,0.05)/(2,0.0125)/(4,0.003125); spread 0.0 | same |
| Finite-difference gradient agreement | max rel err **1.95e-4** | same |
| Stability | no divergence; max‖u‖ = 0.63 — 93.7% margin from the ±10 clamp | same |
| Network pilot: Plastic > Static | Rotation phase A **0.7508 vs 1.0008 (−25.0%)**, phase C 0.7529 vs 1.0043; discovery 0.9696 vs 1.0032 | `pilot_results/pilot_results.json`; `scripts/eval_fix_report.md` row 1 |
| Adaptation criterion (honest) | `None` for every variant at 900 steps — switch-time error never halves within a 300-step phase | `scripts/eval_fix_report.md` |
| Dual-timescale consolidation | Gate-6 rerun: 45,924–88,801 elements transferred, θ norms 30.99–112.81, output preserved to ≤**2.4e-7**; **no retention benefit yet** (1.0013 vs 1.0029) | `pilot_results/pilot_results_gate6_consolidation.json`; `docs/RESULTS.md` §1.3, §2.3 |
| Dreaming D1 (replay) | **Bit-identical to D0** on every field, every seed (e.g. rotation final MSE 0.7580715894699097 both) — replay stub | `scripts/eval_fix_report.md` row 3; `docs/DREAM_REIMPL_SPEC.md` §1.1 |
| Dreaming D2 (imagination) | Accepted 60/60 hypotheses, no held-out effect (0.9694 vs 0.9696), hurt rotation retention (1.0810 vs 1.0029); ties D1x2 equal-compute; **D2-vs-D1 = unanswered** | `pilot_results/pilot_results_d1x2.json`; `docs/RESULTS.md` §2.1 |
| Structural dreaming D3 | Best rotation config (A 0.7213, C 0.7210, retention 0.9996) but not discovery — MIXED | `scripts/eval_fix_report.md` row 5 |
| Conversational acquisition (Phase 1) | **3/5** — acquisition + generalisation honest FAIL (PLASTIC exact 0.00 everywhere, empty-answer fraction 1.0; RETRIEVED direct 0.75); retention, correction, transcript-removal PASS; bitwise restart | `runs/teacher_student/qwen-real-c5/report.json` |
| Disconnect Experiment (real 4B, seed 0) | Raw survival 1.0 / 1.0 / **0.75** / 0.75 / 1.0 (combine / learn_new / predict / recognise / self_correct); **epoch-adjusted 1.0 on all five** (0.75 rows are never-owned RETRIEVED rows); structural proof: zero foundation forwards/loads/teacher calls in the disconnected lane; restart bitwise-equal | `runs/disconnect_real_seed0/disconnect_report.json`; `host_rescored_post.json`; `docs/RESULTS.md` §1.4 |
| Host mode | Plastic 0.95 success (13.35 trials) vs fixed-random 0.9 (25.9 trials); heuristic 1.0 (6.0); adaptation curve 16.25 → 9.0 → 8.8 → 10.8 trials-to-grant by episode block; **0 authority violations in every arm**; impossible-task correct-stop plastic **1.0 (6/6)** vs fixed-random **0.0** (heuristic also 1.0 — plastic's delta is efficiency, not capability); 8 grants, 0 unneeded | `aion/hostmode/results/hostmode_results.json`; `docs/RESULTS.md` §1.5 |
| NAV: larql head transfer | **0.20** (6/30) — QUERY 6/6, others 0/6; distribution-shift failure stated plainly; replicated on Lappy | `aion/nav/results/nav_exam_20261125_045356.json`; `lappy_d1_exam_20261125_054944.json` |
| NAV: d1-2B as-is | **0.5333** — ANSWER 6/6, INFER 4/6, STOP 5/6, ASK_TEACHER 1/6, QUERY 0/6 (the blind spot) | `lappy_d1_exam_20261125_054944.json` |
| NAV: Tev1 0.8B | **No score — 30/30 LICENSE-BLOCKED** (accuracy null) | same file |
| NAV: NAV-domain trained head (post-RESULTS.md) | **1.000** — 30/30, 0 enum violations, holdout 1.000 (n=800); degraded-prior adaptation recovers 0.87 → 0.99 → 0.98 → 0.98 | `aion/nav/artifacts/nav_head_exam_20261125_073813.json`; `nav_decision_head_meta.json`; `adapt_demo.json` |
| NAV: post-hoc roster diagnostic | original 0.70 / fresh 0.7333 — not a preregistered exam row | `aion/nav/artifacts/d1_roster_diag.json` |
| Phase 2 (real 4B, A–E) | **Negative for expression, positive for localisation**: C/D bitwise identical to A (`plastic_ablation_delta 0.0`); E's γ pathway alive ([10.371428, 11.180095, 0.522552]) but sub-threshold; combination 0.3333 in every mode | `runs/teacher_student_p2/p2-qwen-c5-s0-022716/report.json`; `docs/PHASE3_PROTOCOL.md` §1 |
| Substrate interface (Qwen + Pith) | Qwen dense 4/4 + Pith factor-executed **4/4**, γ=0 bit-exact identity on both; foundation bit-identical after runs (426 params fingerprinted) | `aion/experiments/pith_smoke_results.json`; `test_relational_smoke.py`; Phase-1 `foundation_integrity` |
| Ownership mechanics (per-run) | Transcript-removal invariance (21 events purged, plastic answers 20/20 identical); correction: new fact strength 1.307077, old facts 20/20 intact (mean relative strength change 0.0079) | `runs/teacher_student/qwen-real-c5/{episodic_after_purge,report}.json` |

---

## 7. What is NOT built yet — the honest frontier

1. **Dreaming, real implementation.** The Dream Engine exists but its replay lane is a stub and its dream-side θ-writes bypassed the gates. The re-implementation is **approved staged work, not built**: prerequisites P1 (storage rebuild) through P6 (provenance) of `docs/D1_REPLAY_VALIDATION_PLAN.md` are unimplemented; the D1 validation gate (V1–V4: replay must measurably move the network) has not been run.
2. **D2 imagination evaluation.** Explicitly not authorised until the D1 gate passes (owner D6). D2-vs-D1 remains *unanswered, with a reimplementation pending*.
3. **The six-configuration re-run.** Requires its own frozen preregistration (owner D6: no automatic progression). Nothing re-run.
4. **Substrate-cost baseline (task 27) and the substrate ruling.** Pending the compliance ruling; `docs/COMPLIANCE_REQUEST_DRAFT.md` is on HOLD per S1/S2/S4. No substrate-level claim rides on it.
5. **The ownership scale run.** `aion/experiments/ownership_scale.py` carries a frozen preregistration (100 relationships, 325 questions) and 9 passing unit tests, but the run itself is **not executed** — no results directory exists yet (the file is also untracked in git).
6. **Phase 3 execution.** `docs/PHASE3_PROTOCOL.md` is approved with owner decisions D1–D9 locked, but no Phase-3 run exists.
7. **Growth / DreamGrowthManager (stage 05).** README records it as *designed* ("need-based growth law locked") — **no growth code exists in the repository**; a repo-wide search for a growth manager class returns nothing.
8. **NCM.** No NCM component exists anywhere in this repository — searched, absent.
9. **Consolidation's live continual behaviour at rate > 0**, and **property 3 (contextual plasticity) as a dedicated numerical test** (currently code-level-proven only, `context_dim = 0` in the frozen calibration) — recorded gaps per APC §8 and RESULTS §4.
10. **A LICENSE file.** `README_DREAMING.md` cites MIT "see LICENSE file" — no LICENSE file is present in the tree.
11. Everything at scale. All demonstrations are small-world (16-neuron pilot networks; 5 chains / 20 relationships conversational worlds; 12-relation disconnect battery, one seed; CPU-only compute) — the results register's own "Known limits" section.

---

*Survey ends. Docs-only: this document changes no code, touches no run, and disturbs no lane.*
