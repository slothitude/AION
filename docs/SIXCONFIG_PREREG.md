# Six-Configuration Preregistration — DRAFT for owner review

**Status:** DRAFT — not to be run until the owner explicitly approves, and not until the collected four-report review (A–E, ownership-scale, NAV training, build inventory) is complete. Per the development order: no automatic progression from a completed experiment to the next major experiment without reviewing its report.
**Date drafted:** 2026-11-25. **Review criteria source:** the owner's pre-preregistration scrutiny checklist (seven checks, below).

---

## 0. The one clean question

**With a real (validated) dreamer, does dreaming — imagination with measured-acceptance evaluation on top of replay — produce better learning than ordinary experience replay alone?**

D2 > D1 on unseen composition, beyond noise, is the question. A null is an answer.

---

## 1. Configuration definitions (check 1: all six arms precisely specified, no ambiguous combinations)

| Arm | Configuration | Dream state | Notes |
|---|---|---|---|
| 1 | Static D0 | none | no-learning control floor |
| 2 | Plastic D0 | none | the learning mechanism, no dreams |
| 3 | Consolidating D0 | none + consolidation on (rate > 0, annealing law) | tests the persistence property live |
| 4 | Plastic D1 | **real replay** (frozen DREAM_D1_BASELINE implementation) | replay = stored experiences re-run through the network law at scaled lr |
| 5 | Plastic D2 | replay + **imagination** (reality-grounded evaluator, owner D4 weights) | **requires:** imagination implementation (built separately) + explicit owner authorisation of this arm — implementation approval ≠ experiment approval |
| 6 | Plastic D3 | D2 + structural (create/prune via the routed consolidation law) | requires D2's dependencies + the structural path through the same law |

**No ambiguous combinations:** each arm is a closed definition; no arm mixes undeclared mechanisms; D2/D3 arms do not run unless their prerequisites (imagination implementation + explicit authorisation) are separately satisfied, and the preregistration states this in the run header.

## 2. D1 baseline (check 2: frozen implementation, exact recorded settings)

Arm 4 uses **exactly** `freeze/DREAM_D1_BASELINE_20261125.tar.gz` (SHA `ae1ed7da…3204f`): storage fixed-shape ring (archive-never-replayed), replay at scaled lr 0.1, budget min(32, retrieved), provenance replayed=true, θ routed through the connection's annealing `consolidate()` only. Settings recorded in the manifest; deviating settings = new preregistration.

## 3. D2/D3 dependencies (check 3: explicitly authorised before those arms run)

- Imagination implementation per `docs/DREAM_REIMPL_SPEC.md` §3 (E1–E6) — **not yet built**; building it is owner-approved staged work (D5), but *running* D2/D3 arms is not.
- Explicit owner authorisation of arms 5–6 in this preregistration's approval decision.
- If neither is authorised at run time: the matrix runs as five arms (1–4 + Plastic D1), and the D2/D3 rows are recorded NOT-RUN.

## 4. Fair comparison (check 4: identical holdouts, seeds, sources, compute)

- **Identical holdouts:** the corrected instrument's lag-1 holdout, phase-block resets, same batteries, same candidate pools — one instrument spec, frozen in the run header.
- **Identical seeds:** 3 seeds (0, 1, 2), same worlds, same teach order, same question order.
- **Identical experience sources:** every arm receives the identical live experience stream; only the dream configuration differs.
- **Compute budgets properly accounted:** equal-observation and equal-compute both reported (the D1×2 control retained); dream compute counted in `total_dream_steps`; wall-clock reported per arm.

## 5. Evaluation (check 5: the five scores measured independently AND aggregate)

- **consistency** (degradation on non-source experiences), **novelty** (cosine distance), **utility** (measured improvement), **efficiency** (capacity cost), **forgetting** (error increase on consolidated paths) — each reported independently with its own number, then the aggregate score with the owner D4 weights (0.25/0.15/0.25/0.10/0.25).
- Acceptance arithmetic per the frozen evaluator law (ε = 0.0, beyond-noise requirement, hallucination guard before measurement).
- Live-continual retention measured per arm (consolidation rate > 0 exercised live for the first time — recorded as a first, not a claim).

## 6. Acceptance + failure + stopping (check 6: beyond noise, explicit rules)

- **Acceptance:** any claimed win requires improvement beyond the measured noise floor (established from a D0 no-replay control run before the matrix), on 3-seed means, in the preregistered direction.
- **Failure:** any arm that degrades retention or discovery is reported as measured-negative in `docs/RESULTS.md` §2.
- **Stopping rules:** per-arm compute cap (wall-clock, preregistered); matrix stop if two arms hit irreversible errors (reported, not retried silently); no re-run without a new preregistration.

## 7. Reporting (check 7: every arm, including failed interventions + guard rejections)

- Full `report.json` per arm; the Phase-2-format headline; guard-rejection bucket counts; measured-negative rows in `docs/RESULTS.md` §2; the raw ratios beside epoch-adjusted ones; no row omitted.

---

## 8. The seven scrutiny checks — how this draft answers them

| Owner check | How this draft satisfies it |
|---|---|
| Configuration definitions precise | §1 closed table; D2/D3 conditional-authorisation stated in the run header |
| D1 baseline = frozen implementation, exact settings | §2: manifest + SHA + recorded settings, no deviation |
| D2/D3 dependencies explicitly authorised | §3: implementation status + separate authorisation, stated as a run precondition |
| Fair comparison | §4: identical holdouts/seeds/sources; equal-observation + equal-compute both reported |
| Evaluation independent + aggregate | §5: five scores independently, then the weighted aggregate |
| Acceptance beyond noise + explicit failure/stopping | §6: noise floor from a control run, preregistered direction, caps, stop rules |
| Reporting every arm | §7: full report.json per arm, guard-rejection buckets, measured-negative rows |

**Plus the preserved distinction:** approving this preregistration is approving an *experiment*. It is not approval of D2 imagination evaluation as a capability claim, not permission for the full six-config to substitute for the collected-report review, and not authorisation of any dreaming-benefit language in RESULTS.md before the data exists.

---

## 9. Sequencing

1. Lane Q (ownership-scale) and Lane INT (interventions) land → their results join the Phase-2 review.
2. The collected four-report review is presented to the owner.
3. The owner approves, revises or holds this preregistration.
4. Only then does the matrix run.

*Draft ends. Nothing in this document authorises a run.*
