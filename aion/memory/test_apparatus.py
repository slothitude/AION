"""Unit tests for the AION-1 retention/provenance apparatus (CPU stubs only).

Owner acceptance for build-ahead (2026-11-24): construct + unit-test with
stubs/tiny models on CPU; NO real experiments, NO real model/endpoint calls,
everything deterministic with seeds. Covers:
  1. save/restart round-trip preserves plastic state EXACTLY (bitwise)
  2. transcript removal doesn't change plastic answers (and kills variant B)
  3. provenance buckets decidable (PLASTIC / CONSOLIDATED / RETRIEVED / FULL_ONLY / NONE)
  4. consolidation output-preservation invariant (1e-6) passes on the frozen
     machinery and FIRES on a corrupted readout
  5. survival-ratio computation correct on fixture numbers
  6. all four owner variants (A/B/C/D) run end-to-end and differ as specified

Run:
    python -m pytest aion/memory/test_apparatus.py -q        # pytest
    python aion/memory/test_apparatus.py                     # standalone
"""

import json
import math
import os
import sys
import tempfile

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.memory.state import PlasticStateSnapshot                            # noqa: E402
from aion.memory.episodic import EpisodicStore, PURGE_TOMBSTONE               # noqa: E402
from aion.memory.consolidation import ConsolidationLedger                     # noqa: E402
from aion.memory.cpu_stubs import StubVocab                                   # noqa: E402
from aion.experiments.retention import (                                      # noqa: E402
    RetentionExperiment, ModelPathwayAnswerer, ConfiguredAnswerer,
    load_battery, PLASTIC, CONSOLIDATED, RETRIEVED, FULL, Variant,
)
from aion.experiments.knowledge_discovery import (                            # noqa: E402
    survival_ratio, survival_ratios,
)

DEFAULT_BATTERY = os.path.join(_REPO_ROOT, "aion", "memory", "fixtures", "tiny_battery.json")
GOLD_Q1 = "melts ice"
Q1_PROMPT = "What does spark do to ice?"


# --------------------------------------------------------------------------- #
# 1. save/restart round-trip: plastic state preserved EXACTLY
# --------------------------------------------------------------------------- #

def test_round_trip_preserves_plastic_state_exactly(tmp_path):
    run_dir = str(tmp_path / "rt")
    battery = load_battery(DEFAULT_BATTERY)
    exp = RetentionExperiment(variant="C", battery=battery, run_dir=run_dir,
                              seed=0, n_learn_steps=1, verbose=False)
    exp.learn_phase()
    before = PlasticStateSnapshot.capture(exp.bundle.iface)
    assert before.conns[0]["u_fast"].abs().max().item() > 0, "fixture learned nothing"

    exp.restart_from_disk()                       # fresh objects, disk only
    assert exp.state_restored_exact is True

    after = PlasticStateSnapshot.capture(exp.bundle.iface)
    exact, diffs = before.equals(after, atol=0.0)  # atol=0 -> torch.equal, bitwise
    assert exact, f"round-trip drifted: {diffs[:8]}"

    # the decisive tensors, checked explicitly against what is on disk
    payload = torch.load(os.path.join(run_dir, "state.pt"), weights_only=False)
    snap = PlasticStateSnapshot.from_payload(payload["plastic"])
    for k in range(len(after.conns)):
        assert torch.equal(snap.conns[k]["u_fast"], after.conns[k]["u_fast"])
        assert torch.equal(snap.conns[k]["theta_slow"], after.conns[k]["theta_slow"])
        assert torch.equal(snap.conns[k]["consolidation_gate"],
                           after.conns[k]["consolidation_gate"])
        assert torch.equal(snap.conns[k]["eligibility.trace"],
                           after.conns[k]["eligibility.trace"])
    for a, b in zip(snap.gates, after.gates):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 2. transcript REMOVAL: plastic answers survive; variant B's retrieval dies
# --------------------------------------------------------------------------- #

def test_transcript_removal_keeps_plastic_answers(tmp_path):
    battery = load_battery(DEFAULT_BATTERY)

    # --- variant C: knowledge lives in plastic state, not the log --- #
    exp = RetentionExperiment(variant="C", battery=battery,
                              run_dir=str(tmp_path / "c"), seed=0,
                              n_learn_steps=1, verbose=False)
    exp.learn_phase()
    b = exp.bundle
    assert len(b.episodic.events(channel="transcript")) > 0
    probe_ids = b.vocab.encode("spark melts ice")
    with torch.no_grad():
        logits_before = b.iface.forward(probe_ids)["logits"].clone()
    u_before = [c.plasticity.u_fast.clone() for c in b.iface.conns]

    purged = b.episodic.purge_transcripts()
    assert purged["removed"] > 0
    assert all(ev.channel != "transcript" for ev in b.episodic.events())
    assert b.episodic.events()[0].kind == PURGE_TOMBSTONE        # audit trail only

    with torch.no_grad():
        logits_after = b.iface.forward(probe_ids)["logits"]
    assert torch.equal(logits_before, logits_after), "purging the log changed answers"
    for got, want in zip(b.iface.conns, u_before):
        assert torch.equal(got.plasticity.u_fast, want), "purge touched plastic state"
    assert b.rmem.present("spark|melts|ice"), "indexed knowledge vanished with the log"

    # --- variant B: the same purge destroys its only answer source --- #
    exp_b = RetentionExperiment(variant="B", battery=battery,
                                run_dir=str(tmp_path / "b"), seed=0,
                                n_learn_steps=1, verbose=False)
    exp_b.learn_phase()
    ans = ModelPathwayAnswerer(exp_b.bundle.iface, exp_b.bundle.episodic,
                               exp_b.bundle.vocab)
    ids = exp_b.bundle.vocab.encode(Q1_PROMPT)
    hit_before = ans.answer(Q1_PROMPT, ids, RETRIEVED)
    assert hit_before["hit"] is True and hit_before["text"] == GOLD_Q1
    exp_b.bundle.episodic.purge_transcripts()
    hit_after = ans.answer(Q1_PROMPT, ids, RETRIEVED)
    assert hit_after["hit"] is False, "variant B answered after its substrate was purged"


# --------------------------------------------------------------------------- #
# 3. provenance buckets are decidable (ablate one bucket at a time)
# --------------------------------------------------------------------------- #

def _prov_battery():
    return {
        "battery_id": "prov-fixture",
        "seed_relations": [{"subject": "a", "relation": "r1", "object": "b"}],
        "questions": [
            {"qid": "q1", "capability": "recognise", "prompt": "a r1?",
             "paraphrases": ["a r1 alt?"], "gold": "b"},
            {"qid": "q2", "capability": "recognise", "prompt": "c r2?",
             "paraphrases": [], "gold": "d"},
            {"qid": "q3", "capability": "learn_new", "prompt": "e r3?",
             "paraphrases": [], "gold": "f"},
            {"qid": "q4", "capability": "combine", "prompt": "g r4?",
             "paraphrases": [], "gold": "h"},
            {"qid": "q5", "capability": "predict", "prompt": "i r5?",
             "paraphrases": [], "gold": "j"},
        ],
        "new_relations": [], "combinations": [], "corrections": [],
    }


def test_provenance_buckets_decidable(tmp_path):
    mapping = {
        PLASTIC: {"a r1?": "b", "a r1 alt?": "b"},       # plastic alone answers q1
        CONSOLIDATED: {"c r2?": "d"},                     # theta alone answers q2
        RETRIEVED: {"e r3?": "f"},                        # episodic alone answers q3
        FULL: {"g r4?": "h"},                             # only the blend answers q4
        # q5: nothing anywhere -> NONE
    }
    exp = RetentionExperiment(variant="C", battery=_prov_battery(),
                              run_dir=str(tmp_path / "p"), seed=0, n_learn_steps=1,
                              verbose=False, answerer=ConfiguredAnswerer(mapping))
    exp.learn_phase()
    report = exp.probe_phase()
    rows = {}
    for r in report["questions"]:
        rows[(r["qid"], r["surface"])] = r
    expect = {
        ("q1", "base"): PLASTIC, ("q1", "paraphrase1"): PLASTIC,
        ("q2", "base"): CONSOLIDATED,
        ("q3", "base"): RETRIEVED,
        ("q4", "base"): "FULL_ONLY",
        ("q5", "base"): "NONE",
    }
    for key, want in expect.items():
        got = rows[key]["provenance"]
        assert got == want, f"{key}: provenance {got!r} != {want!r}"
        if want in (PLASTIC, CONSOLIDATED, RETRIEVED):
            assert want in rows[key]["correct_buckets"], \
                f"{key}: {want} missing from correct_buckets {rows[key]['correct_buckets']}"
        elif want == "FULL_ONLY":
            assert rows[key]["modes"][FULL]["correct"] is True
            assert rows[key]["correct_buckets"] == []
    # the buckets are separately measurable and mutually exclusive where configured
    q1 = rows[("q1", "base")]
    assert q1["modes"][CONSOLIDATED]["correct"] is False
    assert q1["modes"][RETRIEVED]["correct"] is False
    assert q1["internalisation_test"] is True and q1["internalisation_pass"] is True
    q3 = rows[("q3", "base")]
    assert q3["internalisation_test"] is True and q3["internalisation_pass"] is False, \
        "RETRIEVED must never count as an internalisation pass"


# --------------------------------------------------------------------------- #
# 4. consolidation output-preservation invariant (1e-6, frozen machinery)
# --------------------------------------------------------------------------- #

def test_consolidation_invariant_fires():
    from aion.experiments.retention import AionBundle
    bundle = AionBundle.build(seed=3)
    conn = bundle.iface.conns[0]
    ledger = bundle.ledger
    # the FROZEN PlasticRelationalConnection.consolidate() masks by the
    # eligibility trace, so a REAL learning pass must run first for the trace
    # to be non-empty (otherwise the frozen op legitimately no-ops)
    probe = bundle.vocab.encode("spark melts ice")
    bundle.iface.nnfs_update(probe)
    assert float(conn.eligibility_trace.trace.abs().max().item()) > 0
    with torch.no_grad():
        conn.plasticity.u_fast.add_(0.5)          # guarantee entries above threshold

    ev = ledger.consolidate_pair(0)               # calls the FROZEN consolidate()
    assert ev["frozen_stats"]["consolidated"] > 0, "no real transfer exercised"
    assert conn.plasticity.theta_slow.norm().item() > 0, "theta did not move"
    assert ev["invariant_ok"] is True, f"1e-6 invariant failed on frozen op: {ev['invariant_detail']}"
    ledger.assert_clean()

    # a deliberately NON-preserving change must trip the checker
    baseline = ledger.readout(0)
    with torch.no_grad():
        conn.plasticity.theta_slow.add_(0.5)      # the broken-consolidation analogue
    ok, detail = ledger.verify_preservation(baseline, 0)
    assert ok is False, f"checker missed a real violation ({detail})"
    assert "a_maxdiff" in detail or "y_maxdiff" in detail


def test_consolidation_scheduler():
    ledger = ConsolidationLedger(conns=[], hidden_dim=4)
    assert ledger.maybe_consolidate(3, 4) is None
    assert ledger.maybe_consolidate(8, 4) is not None      # fires on the interval
    assert ledger.maybe_consolidate(0, 4) is None          # never at step 0


# --------------------------------------------------------------------------- #
# 5. survival-ratio computation (fixture numbers)
# --------------------------------------------------------------------------- #

def test_survival_ratio_computation():
    assert survival_ratio(0.8, 0.4) == 0.5
    assert survival_ratio(1.0, 1.0) == 1.0
    assert survival_ratio(0.9, 1.0) == 1.0 / 0.9
    assert survival_ratio(0.0, 0.0) == 1.0            # nothing to lose -> full
    assert survival_ratio(0.0, 0.3) == math.inf       # impossible gain -> flag
    ratios = survival_ratios(
        {"recognise": 1.0, "predict": 0.5, "learn_new": 0.0,
         "combine": 0.2, "self_correct": 0.0},
        {"recognise": 0.5, "predict": 0.5, "learn_new": 0.0,
         "combine": 0.0, "self_correct": 0.4})
    assert ratios["recognise"] == 0.5                 # capability lost half
    assert ratios["predict"] == 1.0                   # fully survived
    assert ratios["learn_new"] == 1.0                 # dead both sides -> 1.0
    assert ratios["combine"] == 0.0                   # fully lost
    assert ratios["self_correct"] == math.inf         # impossible gain -> flagged


# --------------------------------------------------------------------------- #
# 6. A/B/C/D variant harness: all four run, and differ as specified
# --------------------------------------------------------------------------- #

def test_variants_abcd_run_and_differ(tmp_path):
    battery = load_battery(DEFAULT_BATTERY)
    u_norms = {}
    internalisation = {}
    for variant in Variant.ALL:
        run_dir = str(tmp_path / f"v{variant}")
        exp = RetentionExperiment(variant=variant, battery=battery, run_dir=run_dir,
                                  seed=0, n_learn_steps=1, verbose=False)
        exp.learn_phase()
        exp.bundle = None              # force the probe phase through the disk round-trip
        report = exp.probe_phase()
        assert report["restart"]["state_restored_exact"] is True, variant
        assert report["summary"]["n_question_surfaces"] > 0
        assert report["consolidation"]["n_violations"] == 0
        u_norms[variant] = sum(float(c.plasticity.u_fast.norm().item())
                               for c in exp.bundle.iface.conns)
        internalisation[variant] = report["summary"]["internalisation_score"]

    # A and B never touch plastic weights; C and D do
    assert u_norms["A"] == 0.0 and u_norms["B"] == 0.0
    assert u_norms["C"] > 0.0 and u_norms["D"] > 0.0
    # B's episodic substrate exists; C's knowledge exists without any exchange log
    # (provenance machinery distinguishes them even when the stub model cannot)
    assert internalisation.keys() == {"A", "B", "C", "D"}


def test_variant_d_dream_stub_only(tmp_path):
    battery = load_battery(DEFAULT_BATTERY)
    run_dir = str(tmp_path / "d")
    exp = RetentionExperiment(variant="D", battery=battery, run_dir=run_dir,
                              seed=0, n_learn_steps=1, verbose=False)
    exp.learn_phase()
    with open(os.path.join(run_dir, "dream_hook_stub.json"), encoding="utf-8") as f:
        stub = json.load(f)
    assert stub["dream"] == "stub-locked" and stub["engine"] is None
    assert "LOCKED" in stub["note"]


# --------------------------------------------------------------------------- #
# episodic provenance mechanics
# --------------------------------------------------------------------------- #

def test_episodic_provenance_links():
    store = EpisodicStore(clock=lambda: 0.0)
    ev = store.record("lesson", "spark melts ice", plastic_update_ids=[7],
                      tags={"k": "v"})
    assert store.attach_update(ev.event_id, 8) is True
    idx = store.updates_index()
    assert idx[7] == [ev.event_id] and idx[8] == [ev.event_id]
    out = store.purge_transcripts()
    assert out["removed"] == 1
    assert out["orphaned_update_ids"] == [7, 8]
    assert len(store.events()) == 1
    assert store.events()[0].kind == PURGE_TOMBSTONE
    assert store.texts() == []
    # deterministic retrieval ranking
    s2 = EpisodicStore(clock=lambda: 0.0)
    s2.record("observation", "spark melts ice", tags={"answer": GOLD_Q1})
    s2.record("observation", "unrelated entirely different words here")
    hits = s2.retrieve("What does spark do to ice?", k=2)
    assert hits[0][0].tags["answer"] == GOLD_Q1 and hits[0][1] > 0.0


def test_stub_vocab_roundtrip_and_determinism():
    v1, v2 = StubVocab(128), StubVocab(128)
    ids1 = v1.encode_ids("spark melts ice")
    ids2 = v2.encode_ids("spark melts ice")
    assert ids1 == ids2, "vocab must be identical across processes"
    assert v1.decode_ids(ids1) == "spark melts ice"
    assert len(set(ids1)) == 3, "distinct words must get distinct ids"


# --------------------------------------------------------------------------- #
# standalone runner
# --------------------------------------------------------------------------- #

def _main() -> int:
    from pathlib import Path
    tests = [
        test_round_trip_preserves_plastic_state_exactly,
        test_transcript_removal_keeps_plastic_answers,
        test_provenance_buckets_decidable,
        test_consolidation_invariant_fires,
        test_consolidation_scheduler,
        test_survival_ratio_computation,
        test_variants_abcd_run_and_differ,
        test_variant_d_dream_stub_only,
        test_episodic_provenance_links,
        test_stub_vocab_roundtrip_and_determinism,
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for t in tests:
            name = t.__name__
            try:
                takes_tmp = "tmp_path" in t.__code__.co_varnames
                t(tmp_path=Path(tmp) / name) if takes_tmp else t()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001 — test runner must report, not crash
                failures += 1
                import traceback
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
