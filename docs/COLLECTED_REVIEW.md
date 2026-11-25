# AION — Collected Report Review (the four closing reports)

**Date:** 2026-11-25. **Order executed:** the owner's adopted development order, step 1 — close the running experiments, collect and review ALL reports before Phase-3 continuation.
**Reports collected:** (1) Lane G — Phase-2 A–E expression results (real Qwen 4B, 7.8 h); (2) Lane Q — 100-relationship ownership-scale (real Qwen 4B, 418.8 min); (3) Lane T — NAV step 6+7 trained head (3 exams); (4) Build inventory (ground-truth survey, 317 files).
**Register law:** corrected wordings throughout; negative results preserved; no claims beyond evidence.

---

## The four reports, reviewed

### Report 1 — Phase-2 A–E expression experiment (the decisive one)

**Question:** can AION's plastic knowledge, learned through conversation, express through the frozen host's answer-selection?

**Verdict: NO — an honest negative, cleanly measured.**

| Variant | direct@1 | paraphrase@1 | combination@1 |
|---|---|---|---|
| A frozen | 0.250 | 0.125 | 0.333 (= frozen-chance) |
| B retrieval | 1.00 | 0.62 | 0.000 |
| C plastic | 0.250 | 0.125 | 0.333 |
| D plastic+Q | 0.250 | 0.125 | 0.333 |
| E +γ | 0.250 | 0.125 | 0.333 |

- C/D are **bitwise A** — the γ=0 law holds exactly; the plastic base stores everything but never reaches logits.
- E's γ trained to [10.37, 11.18, 0.53] and transmits — **real but sub-threshold**: gold rank 3.15 → 3.05, gold prob 0.175 → 0.187, one question flipped the wrong way. acc@1 unmoved.
- B retrieval wins transfer 0.625 vs 0.125 and is structurally blind to composition (0.000) — the gap plastic must clear.

**What the run proves positive:** all 20 facts stored in plastic state (strengths measured); question-format learning 2×'d the u_fast norms (D vs C); γ sequencing law evidenced live (plasticity first, gate second); bitwise restarts ×5; foundation fingerprint untouched through 7.8 h.

**Localisation:** the expression gap is real, measured with the clean instrument, and consistent with Phase 1 (empty answers) and the interventions run (degenerate argmax). Three experiments, one boundary.

### Report 2 — Ownership-scale (100 relationships)

**Verdict: the ownership MECHANICS scale — 4/4 mechanical gates PASS.**

| Measure | 20-rel (Phase 1) | 100-rel (this run) |
|---|---|---|
| Corrections learned | 1/1 (1.307) | **10/10** (1.309–1.334 — same magnitude) |
| Untouched old facts intact | 20/20 | **90/90** (drift mean 0.37%, max 0.48%) |
| Superseded keys still transmitting | — | 10/10 (additive override) |
| Strict restart | bitwise | **bitwise** (`restore_diffs: []`), 110 probes identical |
| Transcript purge | — | plastic **40/40 unchanged**; RETRIEVED degraded as expected |
| Plastic free-decode | 0.00 | 0.00 (Phase-2 verdict restated) |

Foundation bit-identical after 418.8 min. Honest caveats: concurrent CPU job inflated wall-clock ~2–3× (cannot affect deterministic numbers); the 552 MB state.pt lives on Lappy (bitwise round-trip recorded).

### Report 3 — NAV training (step 6+7)

**Verdict: the trained NAV-domain head scores 1.000 everywhere** — original exam, fresh-seed holdout, and 800-example unseen-world holdout (vs the 0.200 untrained-transfer floor); 0.1 ms/decision. **d1-as-is (zero training): 0.533** — decomposition per card improved ASK_TEACHER (1/6→4/6); half the QUERY failure is state-exactness (roster re-presentation → 0.700/0.733 post-hoc), half genuine zero-training limitation. Step-7 adaptation: outcome feedback recovers a degraded prior [0.87 → 0.99] with zero spurious updates. The owner's question — can d1 work without training? — answered: **partially (0.533); the trained head is the fix (1.000).**

### Report 4 — Build inventory (ground truth)

317 files · 35,652 LOC Python + 365 Rust · **247 tests + 1 skip, all green** · 16 commits, all 2026-11-25. Freeze hashes re-verified from inside the archives (v1 `a39013c9…`, v2 `ea802b96…`). Honest frontier confirmed: dreaming real implementation now BUILT and validated (D1 gate PASS, D2 arm measured), six-config rerun pending, substrate baseline compliance-gated, NCM absent, DreamGrowthManager designed-not-built.

---

## The convergent finding

**Three independent experiments measure the same boundary:** Phase 1 (empty answers), Phase 2 (bitwise-A variants, sub-threshold γ), interventions (degenerate argmax). AION's plastic state **acquires, stores, corrects, retains and survives** — and does not yet **express** through the frozen Base decode. The boundary is localised to the expression pathway, with two named levers (density, alignment) carrying an approved, locked protocol.

## Verdicts for the record

1. **Ownership mechanics: SCALE-PROVEN.** Phase 3's expression arms proceed.
2. **D2-vs-D1: measured at validation dose — no demonstrated held-out benefit over replay at ~20× compute.** Honest null, reported. The full-scale six-config re-run with the real dreamer awaits owner-approved preregistration.
3. **NAV: the trained head is the fix** (1.000 vs the 0.533 zero-training partial).
4. **Inventory: matches the register.** Two hygiene notes logged (README_DREAMING stale LICENSE cite; stage-05 wording "designed" → "designed, unbuilt" for register cleanliness).

---

*Review closes the collected-report phase. Phase-3 continuation proceeds under the locked D1–D9 protocol. No automatic progression beyond it without reviewing its report.*
