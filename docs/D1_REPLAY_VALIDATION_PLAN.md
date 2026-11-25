# D1 Replay Validation Plan — the staged gate before any dreaming claim

**Status:** Plan of record per owner decisions D1–D6 (2026-11-25, recorded in `docs/DREAM_REIMPL_SPEC.md` §6a). Implementation of this plan is **approved staged engineering** (owner D5). Nothing runs until the implementation passes its own tests and the prerequisites below are met.
**Date:** 2026-11-25. **Gate:** D1-only replay validation. **Not authorised by this plan:** D2 imagination evaluation, the full six-configuration re-run, or any dreaming-benefit claim.

---

## 1. The question this gate answers

**Is replay a real intervention?** The pilot's D1 was bit-identical to D0 because `replay_episodic` was a stub — retrieval with placeholder statistics, zero network passes (`docs/DREAM_REIMPL_SPEC.md` §1.1). The re-implementation (owner-approved spec §2 R1–R6) must demonstrate that replay, executed at the scaled learning rate, **actually changes the network** — with correct provenance, bounded compute, and measured state deltas. Until it does, D2-vs-D1 remains *unanswered, with a reimplementation pending*, and nothing may claim otherwise.

## 2. Implementation prerequisites (each blocks the gate)

| # | Prerequisite | Source |
|---|---|---|
| P1 | **Storage rebuild** (owner D2): the experience ring rebuilt around fixed-shape windows; old flattened records archived, never replayed | spec §2 R1 |
| P2 | **θ-write bypass eliminated** (owner condition 1): the two bypass paths (`consolidator.py` line 222 direct θ-write; lines 305–337 raw sweep) retired or disabled behind a flag; ALL dream-side slow-weight changes route through the connection's output-preserving `consolidate()` | spec §3 E5 + owner condition 1 |
| P3 | **Norm guard calibrated** (owner condition 2): the composed-weight guard (spec §3 E4) calibrated against **valid** compositions as well as pathological ones — a guard that rejects legitimate discoveries is another failure | spec §3 E4 + owner condition 2 |
| P4 | **Replay executes the real learning law**: reconstruct `(input, target)` at stored shape → `network.step(learn=True)` at `learning_rate_scale` 0.1 | spec §2 R2 |
| P5 | **Budget accounting**: one cycle replays `min(32, retrieved)`, 1 step each, every step counted in `total_dream_steps` | spec §2 R4 |
| P6 | **Provenance**: every replayed experience tagged `replayed=true` with (cycle_id, step, effective lr) | spec §2 R5 |

## 3. The validation run

**Setup:** the frozen plasticity baseline v2 (`freeze/aion_learning_baseline_v2_20261125.tar.gz` — the dreaming work does not disturb it, spec §5.5) + the rebuilt storage + real replay. Instrument: the six-configuration matrix's corrected instrument (lag-1 targets, run-2 protocol), D1 arm only, 3 seeds.

**What D1 must demonstrate (owner condition 3), each with its measure:**

| # | Demonstration | Measure |
|---|---|---|
| V1 | **Actual network updates** | per-replay-cycle state deltas: ‖Δu_fast‖ and ‖Δθ‖ > 0, logged per cycle — vs D0's measured zero |
| V2 | **Correct provenance** | every replayed step tagged `replayed=true`; zero untagged dream-side state changes |
| V3 | **Bounded compute** | replay steps counted in `total_dream_steps`; wall-clock per cycle reported; the D1×2 equal-compute control retained so equal-compute readings exist |
| V4 | **Measurable state changes** | replay-driven deltas distinguishable from measurement noise (delta > noise floor, established from a no-replay D0 control run) |

**The validity proof:** **D1 stops being bit-identical to D0** — on state deltas, on downstream battery behaviour, or both. If replay at scaled lr still cannot move the network, that is the finding: reported, not tuned away (spec §2 R6).

## 4. Pass / fail

**PASS:** V1–V4 all demonstrated + the θ-bypass retirement verified + D1 measurably ≠ D0 on at least the state-delta measures. The gate then opens: D2 imagination may be designed and evaluated (its own preregistration required).

**FAIL:** replay cannot move the network at scaled lr on the corrected instrument, or any artifact signature from the pilot era (θ jumps outside the preservation law, counter bypasses) reappears. Reported as a measured-negative in `docs/RESULTS.md`; D2-vs-D1 remains *unanswered*; the re-implementation is re-examined before any retry.

## 5. What this gate does NOT authorise

- **No D2 imagination evaluation** until this gate passes (owner D6).
- **No six-configuration re-run** until D2 evaluation has its own frozen preregistration (owner D6: "no automatic progression").
- **No dreaming-benefit claim** of any kind — D1 validation establishes only that replay is a real intervention.
- **No freeze-manifest disturbance**: the dreaming work stays outside the v2 frozen plasticity baseline (spec §5.5).

## 6. Effort (engineering estimates, CPU-bound; owner D5 approved staged work)

| Stage | Estimate |
|---|---|
| P1 storage rebuild | small–medium |
| P2 bypass retirement | small |
| P3 guard calibration | small–medium |
| P4–P6 replay implementation + budget + provenance | medium |
| Validation run (D1 arm, 3 seeds) | run-time (dream compute grows with the real loop; unmeasured until P4 lands — spec §7) |

---

*Plan ends. Implementation is approved staged work; the gate itself opens only on the evidence in §3. Where this plan and the code disagree, the code wins — and the disagreement is reported.*
