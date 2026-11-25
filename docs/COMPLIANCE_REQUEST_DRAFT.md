# Compliance Request — DRAFT for organiser ruling (owner edits and sends)

**Status: DRAFT — HOLD per owner decision S1 (2026-11-25).** Hold briefly pending factual reconciliation (S2) and recipient verification (S4); then send for an organiser ruling. The **owner** edits it, decides whether to send it, and sends it. **Nothing is sent by anyone else, and no submission channel is contacted from this project.** Nothing in this document is a claim; it prepares a question. Silence is not permission: the project treats the absence of a ruling as *no ruling*.

**Date drafted:** 2026-11-25. **Project:** AION (code package `nnfs`, `C:\Users\aaron\Desktop\nnfs`).
**Why now:** `docs/RESULTS.md` §3 records the state plainly: *"the substrate-cost baseline (task 27) and the substrate ruling itself are **pending** the compliance ruling. No substrate-level claim rides on an unresolved ruling."* This draft turns that pending item into a sendable question.

---

## 1. What is known about the compliance requirement

**`COMPLIANCE.md` is not present in this repository** (verified by search over the whole tree, 2026-11-25). The requirement is therefore **stated as reported** — it comes from the project's own record, not from an in-repo document:

- As reported in the project record: the competition (Gemma-4 Kaggle entry; the SCULLM build is a separate project) requires that **use of competition-relevant weights** — including derived or downstream artifacts of them — be disclosed and, where required, covered by a **written organiser ruling** before the use informs any submitted or claimed result.
- What this repo independently shows is consistent with that posture: the learning contract states that every substrate AION attaches to "must pass the same contract … before any claim is made on it" (`README.md` line 132), and the results register refuses to let any substrate-level claim ride on the unresolved ruling (`docs/RESULTS.md` §3).

**Marked unknown:** the exact wording, scope and trigger conditions of the organiser's rules. That is precisely what this request asks them to rule on. No clause of any compliance document is quoted or paraphrased as fact in this draft.

---

## 2. Context — what would be measured or run (facts, each sourced)

The three scopes below sit on the same substrate facts, all from this repository:

| Fact | Source |
|---|---|
| Pith-E2B is the seed substrate of the project: "Pith-E2B artifact exists (post-QAT behaviour gate pending); Qwen3.5-4B-Base hosted on Lappy" (stage 00) | `README.md` line 94 |
| Interface proven on BOTH substrates: Qwen dense (4/4 smoke, bit-exact identity) + **Pith factor-executed (4/4, bit-exact under bf16-forward-v1)** (stage 01) | `README.md` line 95 |
| Pith-E2B v1 = dual-svid, 1.5 BPW + PLE rowwise8; **FACTOR-EXECUTED**. **Two facts, kept distinct:** (i) the interface smoke (4/4 bit-exact) ran against THIS v1 artifact as a **plumbing test only** — artifact identity: `pith-e2b-1p5-dualsvid` at Lappy `/home/aaron/pith-pilot/runs/`; (ii) v1's **behaviour gate FAILS** (hidden cosine 0.118, argmax 0/53; logits cosine median 0.14) — the v1 brain is degraded, and **behaviour testing and any learning on v1 are FORBIDDEN** | `aion/experiments/pith_interface_smoke.py` lines 1–6, 22, 288; behaviour numbers from the R3b dense-baseline comparison |
| Hidden states extract from the **REAL Gemma-4 tree** via the runner's `run_layer` isolation; "Gemma-4-E2B measured topology"; smoke scope is "Interface-level plumbing; **no Pith learning**" | same file, lines 12, 50, 22 |
| AION's own evaluation batteries (independent of any substrate): disconnect battery (12 seed relations, one seed), NAV exam (30 scenarios, preregistered), hostmode exam (20 episodes/arm) | `docs/RESULTS.md` §1.4, §2.4, §1.5 |
| The proposed substrate-cost baseline is task 27 in the project register and is explicitly pending this ruling | `docs/RESULTS.md` §3 |

**Relationship between the artefacts (stated so the organiser sees it plainly):** Pith-E2B is a Gemma-4-derived, factor-executed compressed substrate (the upstream artefact identity `togethercomputer/pith` is **reported, not verified in this repo** — marked unknown). **What the 4/4 interface result establishes and what it does not:** it establishes that AION's interface attaches to the factor-executed substrate bit-exactly under the specified execution contract — plumbing. It does NOT establish that the substrate is behaviourally valid: Pith-E2B v1 currently **fails its own behaviour gate**, and no behavioural, learning, or capability claim is made against it. The exact artefact behind every result in this request is the dual-svid v1 specimen identified above; if any result is later obtained against a different artefact or execution target, it will be named separately. **Bit-exact plumbing does not establish a usable or healthy substrate.**

---

## 3. The three scopes and the ruling question for each

The question is deliberately asked **three times, separately**, so the ruling can differ per scope.

### Scope A — Substrate-cost baseline measurement (proposed; task 27)

**What it is:** a measurement, not a model. Dense Gemma-4 E2B vs Pith-E2B, **no plasticity, no learning, no training** — both substrates evaluated read-only on **AION's own held-out batteries** (the same batteries the project uses for its controls: e.g. the disconnect battery and exam batteries, `docs/RESULTS.md` §1.4/§1.5/§2.4). Output: a cost/quality comparison for the project's internal register.

**The exact question:**
> Does running the proposed substrate-cost baseline measurement — dense Gemma-4 E2B versus Pith-E2B, both unmodified and untrained, evaluated on AION's own held-out batteries with no competition data involved — constitute **use of competition-relevant weights requiring a written organiser ruling** under the compliance rules, given that one side of the comparison (Pith-E2B) is a compressed factor-executed derivative of the competition weights?

### Scope B — AION attach (interface work; already partially done)

**What it is:** interface-level attachment of AION to the Pith substrate — the bit-exact smoke and bridge plumbing ("Interface-level plumbing; no Pith learning", `pith_interface_smoke.py` line 22). No weights are trained, no competition task is attempted; the substrate is exercised as a computation substrate only.

**The exact question:**
> Does AION's interface-level attachment to Pith-E2B — loading and executing the factor-executed substrate for bit-exact interface verification, with no learning, no fine-tuning, and no competition-task use — constitute **use of competition-relevant weights requiring a written organiser ruling** under the compliance rules, including **retroactively** for the interface work already recorded?

### Scope C — Any future AION learning involving Pith-derived weights (not started; scope drawn WIDE per owner decision S3)

**What it is:** **any future learning conducted on, through, or materially involving Pith-derived weights** — Pith-seeded learning, substrate-hosted plasticity, learning on derived artifacts, or any downstream learning that materially involves Pith-derived weights. Nothing of this kind has been run (the smoke explicitly excludes learning; no Pith-learning code exists in this repo). The boundary of "materially involving" is deliberately **asked of the organiser rather than assumed by us**.

**The exact question:**
> Does **any** future AION learning work conducted on, through, or materially involving Pith-E2B or any Pith-derived weights — including derived artifacts, downstream learning, and any pipeline in which Pith-derived weights participate in, host, or receive AION's plastic learning process — constitute **use of competition-relevant weights requiring a written organiser ruling** under the compliance rules; and if yes, **what disclosure, segregation, or provenance conditions** would such a ruling require for results derived from that work? We ask the organiser to **define the boundary** of this scope rather than leaving it to our interpretation.

---

## 4. The ask

One request, three rulings:

1. A **written ruling per scope** (A / B / C), each answering the same form: does it constitute use of competition-relevant weights requiring a ruling — yes or no — and if yes, under what conditions.
2. For any scope ruled **yes**: the disclosure or documentation the organiser wants attached to any future result that touched that scope.
3. For any scope ruled **no**: a one-line written confirmation is requested so the project's register can cite it.

**What we commit to pending the ruling** (already the project's posture, restated as commitment): no substrate-cost baseline is run, no substrate-level claim is made, and no competition submission references Pith work until the written ruling arrives (`docs/RESULTS.md` §3: "No substrate-level claim rides on an unresolved ruling"). The interface record (Scope B) is disclosed here rather than withheld.

---

## 5. Owner send-decision points — DECIDED 2026-11-25 (recorded, binding)

| # | Decision | OWNER DECISION (2026-11-25) |
|---|---|---|
| S1 | Send or hold | **HOLD briefly** pending factual reconciliation (S2) and recipient verification (S4). **Then send** for an organiser ruling. |
| S2 | Reconcile the two substrate facts | **Amend the wording** to distinguish interface plumbing from behavioural validity, and identify the exact artifact/version behind each result. (Applied above: §2 table rewritten; the 4/4 result is named to the dualsvid v1 artifact as a plumbing-only test.) |
| S3 | Scope C wording | **WIDE** — any future learning on, through, or materially involving Pith-derived weights, including derived artifacts and downstream learning. The organiser is asked to define the boundary (§3 Scope C). |
| S4 | Channel and recipient | **Official competition organiser compliance/contact channel; verify the actual recipient before sending.** Owner-only action; not specified in this draft. |
| S5 | Attachments | **Attach** the interface smoke report + concise results-register excerpts, with artifact identity and hashes where available. **Do NOT attach model weights.** |

---

## 6. Verification status of every fact used

| Claim | In-repo? | Source |
|---|---|---|
| Pith-E2B seed substrate, behaviour gate pending | Yes | `README.md` line 94 |
| Interface smoke 4/4 bit-exact under bf16-forward-v1 (both substrates) | Yes | `README.md` line 95 |
| Pith-E2B v1 fails behaviour gate; smoke FORBIDDEN against v1 | Yes | `aion/experiments/pith_interface_smoke.py` lines 6, 288 |
| Hidden states from the REAL Gemma-4 tree; no Pith learning in smoke | Yes | same file, lines 12, 22 |
| Substrate ruling pending; no claim rides on it | Yes | `docs/RESULTS.md` §3 |
| Learning contract applies to every substrate | Yes | `README.md` line 132; `docs/APC.md` §6 |
| Compliance requirement exists and its trigger conditions | **No — reported only** | stated as reported (§1); `COMPLIANCE.md` absent from this repo (verified 2026-11-25) |
| Upstream artefact identity `togethercomputer/pith` | **No — reported only** | owner brief; not verified in-repo |
| Gemma-4 competition rules text | **No** | not in repo; exactly what the ruling must supply |

---

*Draft ends. The owner edits, decides, and sends — or does not. Nothing in this project contacts any organiser, and no fact above is asserted beyond its listed source. Where any figure here and its source file disagree, the source wins.*
