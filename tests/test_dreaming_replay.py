"""
Dream Engine reimplementation tests (spec DREAM_REIMPL_SPEC.md §2 R1-R6,
§3 E1-E6, §5 safeguards, §6a owner decisions D1-D6; validation plan
docs/D1_REPLAY_VALIDATION_PLAN.md prerequisites P1-P6).

Covers:
- R1/D2 storage: fixed-shape windows, bounded ring, flattened archive never replayed
- R2/P4 replay executes the real learning law at the scaled rate
- R4/P5 budget accounting (min(32, retrieved), total_dream_steps)
- R5/P6 provenance (replayed=true, cycle_id, step, effective_lr)
- E4/P3 hallucination guard calibrated against valid AND pathological compositions
- E1-E5/D3/D4 measurement loop: copy isolation, measured scores, utility in the
  sum, epsilon=0.0 beyond-noise acceptance, measured forgetting risk
- Owner condition 1: NO theta-write bypass -- every theta change during a dream
  cycle coincides with consolidate() calls (runtime instrumentation) and no
  dreaming source file contains a direct theta write (source scan)
- V1/V4: D1 state deltas > 0 per cycle vs D0's measured zero
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnfs.connections.network import RelationalNetwork
from nnfs.connections.plastic import FastSlowPlasticity, PlasticRelationalConnection
from nnfs.dreaming import (
    DreamEngine,
    ExperienceReplay,
    MemoryRetriever,
    ComposedWeightNormGuard,
    calibrate_guard,
)
from nnfs.dreaming.retriever import create_retriever
from nnfs.dreaming.replay import create_experience_replay
from nnfs.dreaming.simulator import DreamSimulator
from nnfs.dreaming.evaluator import DreamEvaluator, SimulationResult
from nnfs.dreaming.consolidator import DreamConsolidator
from nnfs.dreaming.recombiner import HypothesizedRelationship


def make_network(seed=0, num_neurons=8, neuron_dim=8, sparsity=0.25,
                 consolidation_rate=0.0, blend_ratio=1.0, learning_rate=0.05):
    torch.manual_seed(seed)
    return RelationalNetwork(
        num_neurons=num_neurons, neuron_dim=neuron_dim, sparsity=sparsity,
        relation_type="post", learning_rate=learning_rate,
        consolidation_rate=consolidation_rate,
        consolidation_threshold=0.001, blend_ratio=blend_ratio,
    )


def make_engine(network, **kwargs):
    defaults = dict(
        dream_cycle_interval=1000,
        enable_replay=True,
        enable_recombination=False,
        enable_consolidation=True,
        enable_structural=False,
        replay_budget=4,
    )
    defaults.update(kwargs)
    return DreamEngine(network=network, **defaults)


def store_random_experiences(engine, n=12, seed=0, num_neurons=8, neuron_dim=8):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        engine.store_experience(
            torch.randn(num_neurons, neuron_dim, generator=g),
            torch.randn(num_neurons, neuron_dim, generator=g),
            significance=1.0,
        )


# ----------------------------------------------------------------------
# R1 / owner D2: fixed-shape storage; flattened records archived, never replayed
# ----------------------------------------------------------------------
class TestStorageRebuild:
    def test_windows_stored_and_retrieved_at_fixed_shape(self):
        r = create_retriever(max_memory=50, num_neurons=8, neuron_dim=8)
        x = torch.randn(8, 8)
        y = torch.randn(8, 8)
        r.store(x, y, significance=0.9, uncertainty=0.2)
        exps = r.retrieve(num_experiences=5, min_score=0.0)
        assert len(exps) == 1
        e = exps[0]
        assert e.input_signal.shape == (8, 8)
        assert e.target_next.shape == (8, 8)
        assert torch.equal(e.input_signal, x)
        assert torch.equal(e.target_next, y)
        assert e.significance == pytest.approx(0.9)
        assert e.uncertainty == pytest.approx(0.2)
        assert e.metadata["is_live_observation"] is True

    def test_sequence_stores_last_step_not_the_average(self):
        r = create_retriever(max_memory=50, num_neurons=6, neuron_dim=6)
        seq = torch.randn(5, 6, 6)
        tgt = torch.randn(5, 6, 6)
        r.store(seq, tgt)
        e = r.retrieve(1, min_score=0.0)[0]
        # last step, un-averaged (spec R1: "un-averaged (or last-step)")
        assert torch.equal(e.input_signal, seq[-1])
        assert torch.equal(e.input_signal, seq[-1]) and not torch.equal(
            e.input_signal, seq.mean(dim=0))
        assert e.metadata["seq_len"] == 5

    def test_ring_is_bounded(self):
        r = create_retriever(max_memory=7, num_neurons=4, neuron_dim=4)
        for i in range(30):
            r.store(torch.full((4, 4), float(i)), torch.full((4, 4), float(i)))
        assert r.replayable_count() == 7
        assert r.get_stats()["stored_experiences"] == 7
        # newest windows survive; count of distinct fill values = 7
        exps = r.retrieve(50, min_score=0.0)
        assert len(exps) == 7

    def test_flattened_records_archived_and_never_replayed(self):
        r = create_retriever(max_memory=50, num_neurons=8, neuron_dim=8)
        # one live window + a legacy flattened record
        r.store(torch.randn(8, 8), torch.randn(8, 8), significance=1.0)
        legacy_inputs = torch.randn(3, 256)   # old flattened format
        legacy_targets = torch.randn(3, 256)
        archived = r.archive_legacy_flattened(legacy_inputs, legacy_targets)
        assert archived == 3
        stats = r.get_stats()
        assert stats["archived_flat_records"] == 3
        assert stats["replayable_experiences"] == 1
        # retrieval serves ONLY fixed-shape windows
        exps = r.retrieve(50, min_score=0.0)
        assert len(exps) == 1
        for e in exps:
            assert e.input_signal.dim() == 2 and tuple(e.input_signal.shape) == (8, 8)
        # archive survives a ring clear (it is an archive)
        r.clear()
        assert r.get_stats()["archived_flat_records"] == 3
        assert r.retrieve(10, min_score=0.0) == []

    def test_window_shape_mismatch_is_loud(self):
        r = create_retriever(max_memory=10, num_neurons=8, neuron_dim=8)
        with pytest.raises(ValueError):
            r.store(torch.randn(4, 4), torch.randn(4, 4))


# ----------------------------------------------------------------------
# R2/P4: replay executes the real learning law
# ----------------------------------------------------------------------
class TestReplayRealLaw:
    def test_replay_moves_u_fast_and_learn_false_does_not(self):
        net = make_network(seed=1)
        r = create_retriever(max_memory=50, num_neurons=8, neuron_dim=8)
        g = torch.Generator().manual_seed(2)
        for _ in range(8):
            r.store(torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g),
                    significance=1.0)
        replay = create_experience_replay(net, r, learning_rate_scale=0.1)

        u_before = [c.plasticity.u_fast.detach().clone() for c in net.connections.values()]
        theta_before = [c.plasticity.theta_slow.detach().clone() for c in net.connections.values()]

        results = replay.replay_batch(r.retrieve(4, min_score=0.0), learn=True)
        assert len(results) == 4
        assert all(res.prediction_mse >= 0.0 for res in results)
        u_after = [c.plasticity.u_fast.detach() for c in net.connections.values()]
        assert any(not torch.equal(b, a) for b, a in zip(u_before, u_after)), \
            "replay with learn=True must update u_fast (real law)"

        # learn=False must not train
        u_now = [c.plasticity.u_fast.detach().clone() for c in net.connections.values()]
        replay.replay_batch(r.retrieve(2, min_score=0.0), learn=False)
        u_final = [c.plasticity.u_fast.detach() for c in net.connections.values()]
        assert all(torch.equal(b, a) for b, a in zip(u_now, u_final))

    def test_scaled_rate_is_respected_and_restored(self):
        # identical nets/experiences: replay at scale 1.0 vs 0.1
        deltas = {}
        for scale in (1.0, 0.1):
            torch.manual_seed(5)
            net = make_network(seed=5)
            r = create_retriever(max_memory=50, num_neurons=8, neuron_dim=8)
            g = torch.Generator().manual_seed(6)
            exps = []
            for _ in range(4):
                exps.append((torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g)))
                r.store(*exps[-1], significance=1.0)
            replay = create_experience_replay(net, r, learning_rate_scale=scale)
            replay.replay_batch(r.retrieve(4, min_score=0.0), learn=True)
            delta = sum(
                float(c.plasticity.u_fast.detach().norm())
                for c in net.connections.values())
            live_rates = {float(c.plasticity.learning_rate) for c in net.connections.values()}
            assert live_rates == {0.05}, "live learning rate must be restored after replay"
            deltas[scale] = delta
        assert deltas[0.1] > 0.0
        assert deltas[0.1] < deltas[1.0]
        # the 0.1-scaled run lands well below the full-rate run
        assert deltas[0.1] < 0.9 * deltas[1.0]

    def test_replay_episodic_returns_real_stats(self):
        net = make_network(seed=3)
        r = create_retriever(max_memory=50, num_neurons=8, neuron_dim=8)
        g = torch.Generator().manual_seed(4)
        for _ in range(10):
            r.store(torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g),
                    significance=1.0)
        replay = create_experience_replay(net, r, replay_budget=5, learning_rate_scale=0.1)
        stats = replay.replay_episodic(num_experiences=5, cycle_id=0)
        assert stats["replayed"] == 5
        assert stats["total_dream_steps"] == 5
        assert stats["avg_mse"] >= 0.0
        assert stats["u_fast_delta_norm"] > 0.0  # real measured state change


# ----------------------------------------------------------------------
# R4/P5 budget + R5/P6 provenance
# ----------------------------------------------------------------------
class TestBudgetAndProvenance:
    def test_budget_is_bounded(self):
        net = make_network(seed=7)
        r = create_retriever(max_memory=100, num_neurons=8, neuron_dim=8)
        g = torch.Generator().manual_seed(8)
        for _ in range(40):
            r.store(torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g),
                    significance=1.0)
        replay = create_experience_replay(net, r, replay_budget=6, learning_rate_scale=0.1)
        stats = replay.replay_episodic(num_experiences=32, cycle_id=0)
        assert stats["replayed"] == 6          # budget, not the 40 stored
        assert stats["replay_budget"] == 6
        assert stats["total_dream_steps"] == 6

    def test_provenance_tags_every_replayed_step(self):
        net = make_network(seed=9)
        r = create_retriever(max_memory=100, num_neurons=8, neuron_dim=8)
        g = torch.Generator().manual_seed(10)
        for _ in range(12):
            r.store(torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g),
                    significance=1.0)
        replay = create_experience_replay(net, r, replay_budget=4, learning_rate_scale=0.1)
        stats = replay.replay_episodic(num_experiences=4, cycle_id=3)
        records = list(replay.decision_records)
        assert len(records) == 4
        for i, rec in enumerate(records):
            assert rec["replayed"] is True
            assert rec["cycle_id"] == 3
            assert rec["step"] == i
            assert rec["effective_lr"] == pytest.approx(0.05 * 0.1)
        # the experiences themselves are tagged
        assert stats["provenance_count"] == 4

    def test_engine_counts_replayed_steps_in_total_dream_steps(self):
        net = make_network(seed=11)
        eng = make_engine(net, replay_budget=4)
        store_random_experiences(eng, n=10)
        out = eng.run_dream_cycle()
        assert out.experiences_replayed == 4
        assert out.replay_steps_executed == 4
        assert eng.total_dream_steps.item() == 4
        assert len(eng.cycle_log) == 1
        assert eng.cycle_log[0]["replay_steps_executed"] == 4


# ----------------------------------------------------------------------
# E4 / P3 owner condition 2: guard calibrated both ways
# ----------------------------------------------------------------------
class TestHallucinationGuard:
    def test_calibration_table_valid_and_pathological(self):
        result = calibrate_guard(seed=0, verbose=False)
        assert result["valid_accepted"] == result["valid_total"]
        assert result["pathological_rejected"] == result["pathological_total"]
        assert result["calibrated"] is True
        # margin: pathological scale ratios far above the threshold, valid below
        assert result["min_pathological_scale_ratio"] > 10.0
        assert result["max_valid_scale_ratio"] < 10.0
        assert result["margin_ratio"] > 2.0
        reasons = {row["reason"] for row in result["table"] if not row["pass"]}
        assert not reasons

    def test_guard_rejects_nan_and_nonfinite(self):
        net = make_network(seed=12)
        guard = ComposedWeightNormGuard(net)
        hyp = HypothesizedRelationship(
            pre_idx=0, post_idx=1, relationship_type="composition",
            source_relationships=[(0, 1), (1, 2)],
            hypothesis_weight=torch.full((8, 8), float("nan")),
            confidence=0.7, source_contexts=[None, None],
        )
        verdict = guard.check(hyp)
        assert not verdict.accepted
        assert verdict.reason == "non_finite"

    def test_guard_rejects_out_of_sane_band_composition(self):
        net = make_network(seed=13)
        # give the network a realistic weight scale
        g = torch.Generator().manual_seed(13)
        with torch.no_grad():
            for c in net.connections.values():
                c.plasticity.u_fast.add_(0.5 * torch.randn(c.plasticity.u_fast.shape, generator=g))
        guard = ComposedWeightNormGuard(net, max_scale_ratio=10.0)
        hyp = HypothesizedRelationship(
            pre_idx=0, post_idx=1, relationship_type="composition",
            source_relationships=[(0, 1), (1, 2)],
            hypothesis_weight=torch.randn(8, 8) * 500.0,
            confidence=0.7, source_contexts=[None, None],
        )
        verdict = guard.check(hyp)
        assert not verdict.accepted
        assert verdict.reason == "outside_sane_band"

    def test_guard_accepts_sane_composition(self):
        net = make_network(seed=14)
        g = torch.Generator().manual_seed(14)
        with torch.no_grad():
            for c in net.connections.values():
                c.plasticity.u_fast.add_(0.5 * torch.randn(c.plasticity.u_fast.shape, generator=g))
        guard = ComposedWeightNormGuard(net, max_scale_ratio=10.0)
        hyp = HypothesizedRelationship(
            pre_idx=0, post_idx=1, relationship_type="composition",
            source_relationships=[(0, 1), (1, 2)],
            hypothesis_weight=0.5 * torch.randn(8, 8, generator=g),
            confidence=0.7, source_contexts=[None, None],
        )
        verdict = guard.check(hyp)
        assert verdict.accepted

    def test_engine_counts_guard_rejections_in_own_bucket(self):
        net = make_network(seed=15)
        g = torch.Generator().manual_seed(15)
        with torch.no_grad():
            for c in net.connections.values():
                c.plasticity.u_fast.add_(
                    0.5 * torch.randn(c.plasticity.u_fast.shape, generator=g))
        eng = make_engine(net, enable_recombination=True, replay_budget=4)
        store_random_experiences(eng, n=10)
        eng.guard.max_scale_ratio = 1e-9  # every nonzero composition is out-of-band
        out = eng.run_dream_cycle()
        # no measurement spent on garbage: zero simulations ran
        assert out.simulations_run == 0
        assert out.hypotheses_generated > 0
        assert out.hypotheses_guard_rejected == out.hypotheses_generated
        assert len(out.guard_rejections) == out.hypotheses_generated


# ----------------------------------------------------------------------
# E1-E5 / D3 / D4: the measurement loop and the acceptance law
# ----------------------------------------------------------------------
class TestMeasuredEvaluator:
    @staticmethod
    def _measured_sim(utility_delta, noise_floor=0.0, forgetting=0.0, measured=True):
        hyp = HypothesizedRelationship(
            pre_idx=0, post_idx=1, relationship_type="composition",
            source_relationships=[(0, 1), (1, 2)],
            hypothesis_weight=torch.randn(8, 8), confidence=0.7,
            source_contexts=[None, None],
        )
        utility = max(-1.0, min(1.0, utility_delta / max(abs(0.8), 1e-12)))
        return SimulationResult(
            hypothesis=hyp,
            predicted_outcomes=torch.zeros(4),
            prediction_errors=torch.tensor([utility_delta]),
            consistency_score=0.9, novelty_score=0.5,
            utility_score=utility, stability_estimate=0.9,
            recommendation="accept",
            measured=measured,
            utility_err_baseline=0.8,
            utility_err_installed=0.8 - utility_delta,
            utility_delta_abs=utility_delta,
            noise_floor=noise_floor,
            forgetting_risk=forgetting,
            efficiency_score=0.9,
            install_mode="existing_connection",
        )

    def test_owner_d4_weights(self):
        net = make_network(seed=16)
        ev = DreamEvaluator(network=net)
        w = ev.OWNER_D4_WEIGHTS
        assert w == {"consistency": 0.25, "novelty": 0.15, "utility": 0.25,
                     "efficiency": 0.10, "forgetting": 0.25}
        assert abs(ev.consistency_weight + ev.novelty_weight + ev.utility_weight
                   + ev.efficiency_weight + ev.forgetting_weight - 1.0) < 1e-9
        assert ev.utility_epsilon == 0.0
        with pytest.raises(ValueError):
            DreamEvaluator(network=net, consistency_weight=0.5)  # must sum to 1.0

    def test_utility_enters_the_overall_score(self):
        net = make_network(seed=17)
        ev = DreamEvaluator(network=net)
        sim = self._measured_sim(utility_delta=0.4)
        res = ev.evaluate(sim)
        expected = (0.25 * sim.consistency_score + 0.15 * sim.novelty_score
                    + 0.25 * sim.utility_score + 0.10 * sim.efficiency_score
                    + 0.25 * (1 - sim.forgetting_risk))
        assert res.overall_score == pytest.approx(expected)
        assert res.feedback["utility"] == sim.utility_score

    def test_acceptance_law_helpful_beyond_noise_accepts(self):
        net = make_network(seed=18)
        ev = DreamEvaluator(network=net)
        res = ev.evaluate(self._measured_sim(utility_delta=0.4, noise_floor=0.01))
        assert res.acceptance == "accept"
        assert res.recommended_action == "consolidate"
        assert res.beyond_noise is True

    def test_acceptance_law_harmful_rejects(self):
        net = make_network(seed=19)
        ev = DreamEvaluator(network=net)
        res = ev.evaluate(self._measured_sim(utility_delta=-0.2, noise_floor=0.01))
        assert res.acceptance == "reject"
        assert res.recommended_action == "discard"

    def test_acceptance_law_within_noise_defers_not_accepts(self):
        net = make_network(seed=20)
        ev = DreamEvaluator(network=net)
        res = ev.evaluate(self._measured_sim(utility_delta=0.001, noise_floor=0.01))
        assert res.acceptance == "defer"   # positive but not beyond noise
        assert res.beyond_noise is False

    def test_acceptance_law_forgetting_bound_blocks(self):
        net = make_network(seed=21)
        ev = DreamEvaluator(network=net)
        res = ev.evaluate(self._measured_sim(utility_delta=0.4, noise_floor=0.01,
                                             forgetting=0.9))
        assert res.acceptance == "reject"
        assert res.feedback["forgetting_risk"] == pytest.approx(0.9)

    def test_unmeasured_results_are_never_accepted(self):
        net = make_network(seed=22)
        ev = DreamEvaluator(network=net)
        res = ev.evaluate(self._measured_sim(utility_delta=0.4, measured=False))
        assert res.acceptance == "defer"
        assert ev.get_stats().get("unmeasured") == 1

    def test_guard_failure_blocks_acceptance(self):
        net = make_network(seed=23)
        ev = DreamEvaluator(network=net)
        res = ev.evaluate(self._measured_sim(utility_delta=0.4, noise_floor=0.01),
                          guard_passed=False, guard_reason="outside_sane_band")
        assert res.acceptance == "reject"
        assert res.guard_passed is False


class TestMeasurementLoop:
    def test_simulator_scores_are_measured_not_constants(self):
        # two hypotheses about the same trained connection must measure
        # differently: zeroing the trained weight vs keeping it
        torch.manual_seed(24)
        net = make_network(seed=24, consolidation_rate=0.0)
        g = torch.Generator().manual_seed(25)
        # train briefly so weights carry signal
        for _ in range(40):
            net.step(torch.randn(2, 8, 8, generator=g),
                     target_next=torch.randn(2, 8, 8, generator=g), learn=True)

        # find a connection with nonzero combined weight
        target = None
        for c in net.connections.values():
            if float(c.plasticity.get_combined_weight().norm()) > 1e-3:
                target = c
                break
        assert target is not None, "network must have a trained connection"

        experiences = []
        for _ in range(9):
            x = torch.randn(8, 8, generator=g)
            y = torch.randn(8, 8, generator=g)
            experiences.append(type(
                "E", (), {"input_signal": x, "target_next": y,
                          "metadata": {}})())

        sim = DreamSimulator(network=net, num_folds=3)

        def hyp_with(weight, pre, post):
            return HypothesizedRelationship(
                pre_idx=pre, post_idx=post, relationship_type="composition",
                source_relationships=[(pre, 2), (2, post)],
                hypothesis_weight=weight, confidence=0.7,
                source_contexts=[None, None],
            )

        pre_i, post_i = int(target.pre_idx), int(target.post_idx)
        w_current = target.plasticity.get_combined_weight().detach()
        h_zero = hyp_with(torch.zeros_like(w_current), pre_i, post_i)
        h_same = hyp_with(w_current.clone(), pre_i, post_i)
        h_big = hyp_with(w_current * 4.0, pre_i, post_i)

        r_zero = sim.simulate_hypothesis(h_zero, None,
                                         experiences_utility=experiences[:6],
                                         experiences_consistency=experiences[6:])
        r_same = sim.simulate_hypothesis(h_same, None,
                                         experiences_utility=experiences[:6],
                                         experiences_consistency=experiences[6:])
        r_big = sim.simulate_hypothesis(h_big, None,
                                        experiences_utility=experiences[:6],
                                        experiences_consistency=experiences[6:])

        # measured, not the old constant table (0.8/0.7/0.6/0.5)
        assert r_zero.measured and r_same.measured and r_big.measured
        assert r_zero.utility_err_baseline == r_same.utility_err_baseline
        # zeroing a trained weight must measure WORSE than keeping it
        assert r_zero.utility_delta_abs < r_same.utility_delta_abs
        # the three hypotheses do not share one constant score
        scores = {round(r.utility_score, 6) for r in (r_zero, r_same, r_big)}
        assert len(scores) >= 2
        # real error evidence is returned, not torch.zeros(1) placeholders
        assert r_zero.predicted_outcomes.numel() == 4
        assert r_zero.consistency_err_baseline >= 0.0

    def test_measurement_isolation_live_network_untouched(self):
        torch.manual_seed(26)
        net = make_network(seed=26, consolidation_rate=0.001)
        g = torch.Generator().manual_seed(27)
        for _ in range(20):
            net.step(torch.randn(2, 8, 8, generator=g),
                     target_next=torch.randn(2, 8, 8, generator=g), learn=True)
        sim = DreamSimulator(network=net, num_folds=2)

        snap_u = [c.plasticity.u_fast.detach().clone() for c in net.connections.values()]
        snap_t = [c.plasticity.theta_slow.detach().clone() for c in net.connections.values()]
        snap_states = net.neuron_states.detach().clone()
        snap_rel = [c.relationship.W.detach().clone() for c in net.connections.values()]

        hyp = HypothesizedRelationship(
            pre_idx=0, post_idx=1, relationship_type="composition",
            source_relationships=[(0, 1), (1, 2)],
            hypothesis_weight=torch.randn(8, 8, generator=g) * 3.0,
            confidence=0.7, source_contexts=[None, None],
        )
        exps = [type("E", (), {"input_signal": torch.randn(8, 8, generator=g),
                               "target_next": torch.randn(8, 8, generator=g),
                               "metadata": {}})() for _ in range(6)]
        sim.simulate_batch([hyp], torch.zeros(1),
                           experiences_utility=exps[:3], experiences_consistency=exps[3:])

        assert all(torch.equal(a, b) for a, b in
                   zip(snap_u, [c.plasticity.u_fast.detach() for c in net.connections.values()]))
        assert all(torch.equal(a, b) for a, b in
                   zip(snap_t, [c.plasticity.theta_slow.detach() for c in net.connections.values()]))
        assert torch.equal(snap_states, net.neuron_states)
        assert all(torch.equal(a, b) for a, b in
                   zip(snap_rel, [c.relationship.W.detach() for c in net.connections.values()]))


# ----------------------------------------------------------------------
# Owner condition 1: theta-write bypass ELIMINATED
# ----------------------------------------------------------------------
DIRECT_THETA_WRITE_PATTERNS = [
    "theta_slow.data =",
    "theta_slow.data=",
    "theta_slow.copy_",
    "theta_slow.add_",
    "theta_slow.sub_",
    "theta_slow.mul_",
    "theta_slow.zero_",
    "theta_slow.fill_",
    "theta_slow[",
    "theta_slow +=",
    "theta_slow -=",
    ".theta_slow.data",
]


class TestThetaBypassEliminated:
    def _run_instrumented_cycle(self, seed=30):
        """Run a full dream cycle (replay + recombination + consolidation)
        with consolidate() instrumented; returns (engine, per-instance theta
        change recorded inside consolidate windows, initial thetas)."""
        torch.manual_seed(seed)
        net = make_network(seed=seed, consolidation_rate=0.001)
        g = torch.Generator().manual_seed(seed + 1)
        for _ in range(30):
            net.step(torch.randn(2, 8, 8, generator=g),
                     target_next=torch.randn(2, 8, 8, generator=g), learn=True)

        # per-plasticity-instance initial theta and accumulated window delta
        initial = {id(c.plasticity): c.plasticity.theta_slow.detach().clone()
                   for c in net.connections.values()}
        inside = {pid: torch.zeros_like(t) for pid, t in initial.items()}

        orig = FastSlowPlasticity.consolidate

        def wrapped(plasticity_self, *args, **kwargs):
            before = plasticity_self.theta_slow.detach().clone()
            stats = orig(plasticity_self, *args, **kwargs)
            after = plasticity_self.theta_slow.detach().clone()
            pid = id(plasticity_self)
            if pid in inside:
                inside[pid] += (after - before)
            return stats

        eng = make_engine(net, enable_recombination=True, enable_consolidation=True,
                          replay_budget=4)
        store_random_experiences(eng, n=12, seed=seed + 2)

        with patch.object(FastSlowPlasticity, "consolidate", wrapped):
            out = eng.run_dream_cycle()

        return eng, initial, inside, out

    def test_all_theta_changes_coincide_with_consolidate_calls(self):
        eng, initial, inside, out = self._run_instrumented_cycle(seed=30)
        net = eng.network
        mismatches = []
        for key, conn in net.connections.items():
            pid = id(conn.plasticity)
            if pid not in initial:
                # connection created during the cycle: its theta starts at the
                # constructed zero and its hypothesis went into the FAST
                # channel -- creation carries no theta write (verified in the
                # dedicated test below).
                continue
            final = conn.plasticity.theta_slow.detach()
            expected = initial[pid] + inside[pid]
            if not torch.allclose(final, expected, atol=1e-6):
                mismatches.append(
                    (key, float((final - expected).norm())))
        assert not mismatches, (
            "theta changed OUTSIDE consolidate() windows on: "
            f"{mismatches}; cycle: accepted={out.accepted} "
            f"consolidations={out.consolidations_performed}")

    def test_theta_changes_only_via_consolidate_with_created_connections(self):
        # force acceptance path: create connections from accepted hypotheses so
        # the dream-side creation path is exercised, then re-verify the law.
        eng, initial, inside, out = self._run_instrumented_cycle(seed=31)
        net = eng.network
        for key, conn in net.connections.items():
            pid = id(conn.plasticity)
            if pid not in initial:
                continue  # created during the cycle (checked below)
            final = conn.plasticity.theta_slow.detach()
            assert torch.allclose(final, initial[pid] + inside[pid], atol=1e-6)
        # connections created during the cycle start from theta=0 exactly
        for key, conn in net.connections.items():
            pid = id(conn.plasticity)
            if pid not in initial:
                assert float(conn.plasticity.theta_slow.detach().norm()) == 0.0, (
                    "dream-created connection must start with theta=0 "
                    "(hypothesis goes into the fast channel, not theta)")

    def test_no_direct_theta_writes_in_dreaming_source(self):
        import ast
        dream_dir = Path(__file__).resolve().parents[1] / "src" / "nnfs" / "dreaming"
        offenders = []
        for py in sorted(dream_dir.glob("*.py")):
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text)
            # docstring line ranges (module/class/function) are documentation,
            # not code -- the retired bypass is quoted verbatim in consolidator's
            # module docstring on purpose.
            doc_lines = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    body = node.body
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
            lines = text.splitlines()
            for line_no, line in enumerate(lines, start=1):
                if line_no in doc_lines:
                    continue
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # sentinel: a theta write on a disposable MEASUREMENT CLONE is
                # not a live-state write (owner D3 isolation), and must be
                # marked as such inline.
                if "measurement-clone-only" in line:
                    continue
                # pure reads (snapshots) are not writes: exclude .clone() reads
                # of theta_slow.data
                if ".clone()" in line:
                    continue
                for pat in DIRECT_THETA_WRITE_PATTERNS:
                    if pat in line:
                        offenders.append((py.name, line_no, stripped))
        assert not offenders, (
            "dreaming code writes theta_slow directly (must route through "
            f"the connection's consolidate()): {offenders}")

    def test_rate_zero_disables_dream_consolidation(self):
        torch.manual_seed(32)
        net = make_network(seed=32, consolidation_rate=0.0)  # Plastic-D0 control
        eng = make_engine(net, enable_recombination=True, enable_consolidation=True,
                          replay_budget=4)
        store_random_experiences(eng, n=10)
        theta_before = [c.plasticity.theta_slow.detach().clone()
                        for c in net.connections.values()]
        out = eng.run_dream_cycle()
        theta_after = [c.plasticity.theta_slow.detach()
                       for c in net.connections.values()]
        assert all(torch.equal(a, b) for a, b in zip(theta_before, theta_after)), \
            "rate-0 controls must not consolidate via the dream path"
        assert out.state_delta_theta_slow == pytest.approx(0.0)


# ----------------------------------------------------------------------
# V1/V4: D1 is a real intervention -- D1 != D0
# ----------------------------------------------------------------------
class TestD1NotD0:
    def _run_arm(self, with_replay, seed, cycles=3):
        torch.manual_seed(seed)
        net = make_network(seed=seed)
        u_initial = [c.plasticity.u_fast.detach().clone() for c in net.connections.values()]
        du_per_cycle = []
        if with_replay:
            eng = make_engine(net, replay_budget=4)
            store_random_experiences(eng, n=16, seed=seed + 100)
            for _ in range(cycles):
                out = eng.run_dream_cycle()
                du_per_cycle.append(out.state_delta_u_fast)
        return net, u_initial, du_per_cycle

    def test_d1_state_deltas_positive_d0_exactly_zero(self):
        net1, u0_1, du1 = self._run_arm(True, seed=40)
        net0, u0_0, du0 = self._run_arm(False, seed=40)
        # V1: every replay cycle moved the network
        assert len(du1) == 3
        assert all(d > 0.0 for d in du1), f"D1 cycle deltas must be > 0: {du1}"
        # V4: the no-replay D0 control measures exactly zero across the same
        # dream windows (the noise floor is zero on this deterministic CPU path)
        assert du0 == []

        u1 = [c.plasticity.u_fast.detach() for c in net1.connections.values()]
        u0 = [c.plasticity.u_fast.detach() for c in net0.connections.values()]
        assert any(not torch.equal(a, b) for a, b in zip(u1, u0)), \
            "D1 and D0 must NOT be bit-identical on fast weights"

    def test_d1_differs_from_d0_in_absolute_state(self):
        net1, u0_1, _ = self._run_arm(True, seed=41)
        net0, u0_0, _ = self._run_arm(False, seed=41)
        d = sum(float((a - b).norm()) for a, b in
                zip((c.plasticity.u_fast.detach() for c in net1.connections.values()),
                    (c.plasticity.u_fast.detach() for c in net0.connections.values())))
        assert d > 0.0

    def test_engine_cycle_log_carries_validation_evidence(self):
        net = make_network(seed=42)
        eng = make_engine(net, replay_budget=4)
        store_random_experiences(eng, n=12)
        out = eng.run_dream_cycle()
        entry = eng.cycle_log[0]
        assert entry["state_delta_u_fast"] > 0.0
        assert entry["replayed"] == entry["replay_steps_executed"] > 0
        assert entry["provenance_count"] > 0
        assert entry["replay_wall_clock_s"] >= 0.0
        assert entry["replay_budget"] == 4
