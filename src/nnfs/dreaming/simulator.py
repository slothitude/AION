"""
Dream Simulator - Evaluates hypothesized relationships by MEASUREMENT.

Reality-grounded per spec DREAM_REIMPL_SPEC.md §3 E1 (owner-approved,
2026-11-25). The previous version never ran the network: every score was a
hardcoded per-relationship-type table lookup and predicted_outcomes /
prediction_errors were torch.zeros(1) placeholders, so acceptance was
guaranteed arithmetic (60/60 at precision 1.0) rather than evidence.

Now: "reality decides" is implemented. For each hypothesis the network is
run on stored replay experiences WITH the hypothesis installed and the
scores are measured behaviour deltas, not type lookups:

- consistency := 1 - normalised prediction degradation on stored experiences
  the hypothesis was NOT built from (conflict measure).
- utility := measured prediction improvement vs the unmodified network on the
  experiences the hypothesis was built from.
- novelty := max cosine distance between the hypothesis weight and existing
  connection weights (actually checked against the network).
- stability := repeated-measurement spread of the utility delta across
  held-out replay folds (low spread = stable).

Measurement isolation (owner decision D3, COPY): the hypothesis is installed
on a DEEP COPY of the network and every measurement runs on that copy. The
live network is never mutated by measurement -- no install-then-restore
risk. Baselines are measured on their own copy (reused across hypotheses
within one measurement session), so the live network is untouched by the
whole loop.

learn=False during scoring: measurement never trains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import copy
import pickle

from nnfs.dreaming.recombiner import HypothesizedRelationship
from nnfs.connections.plastic import PlasticRelationalConnection


@dataclass
class SimulationResult:
    """Result of MEASURING a hypothesized relationship (spec §3 E1)."""
    hypothesis: 'HypothesizedRelationship'
    predicted_outcomes: torch.Tensor  # measured per-fold prediction errors
    prediction_errors: torch.Tensor   # measured per-fold error deltas
    consistency_score: float          # MEASURED: 1 - normalised degradation (non-source)
    novelty_score: float              # MEASURED: max cosine distance to existing weights
    utility_score: float              # MEASURED: normalised improvement (source experiences)
    stability_estimate: float         # MEASURED: 1/(1+spread) across repeated folds
    recommendation: str               # reporting bins only (spec E3) - never gates
    # --- measurement evidence ---
    measured: bool = False            # False when no experiences were available
    utility_err_baseline: float = 0.0     # source-fold error, unmodified network
    utility_err_installed: float = 0.0    # source-fold error, hypothesis installed
    utility_delta_abs: float = 0.0        # absolute error improvement (baseline - installed)
    consistency_err_baseline: float = 0.0
    consistency_err_installed: float = 0.0
    utility_delta_per_fold: List[float] = field(default_factory=list)
    noise_floor: float = 0.0          # across-fold spread of the utility delta
    forgetting_risk: float = 0.0      # MEASURED error increase on consolidated conn paths
    efficiency_score: float = 0.0     # MEASURED capacity cost of the hypothesis
    install_mode: str = "none"        # "existing_connection" | "new_connection" | "none"


def _connection_key(network, pre_idx: int, post_idx: int):
    """Find an existing connection by (pre, post) indices. Returns key or None."""
    for key, conn in network.connections.items():
        pre = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
        post = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
        if pre == pre_idx and post == post_idx:
            return key
    return None


def _live_connection_params(network) -> Dict:
    """Dream-created connections inherit the live network's plasticity
    parameters (owner fix 4)."""
    existing = next(iter(network.connections.values()), None)
    if existing is not None:
        learning_rate = float(existing.plasticity.learning_rate)
        blend_ratio = float(getattr(existing, "blend_ratio",
                                    getattr(network, "blend_ratio", 0.7)))
    else:
        learning_rate = 0.01
        blend_ratio = float(getattr(network, "blend_ratio", 0.7))
    return {"learning_rate": learning_rate, "blend_ratio": blend_ratio}


class DreamSimulator(nn.Module):
    """
    Measures hypothesized relationships against the network's real behaviour.

    The measurement loop (E1): a deep COPY of the network carries the
    hypothesis; the copy runs the stored replay experiences with
    learn=False; the same experiences run on a baseline copy of the
    unmodified network; the scores are the measured error deltas.
    """

    def __init__(
        self,
        network: 'RelationalNetwork',
        simulation_steps: int = 5,
        consistency_weight: float = 0.4,
        novelty_weight: float = 0.3,
        utility_weight: float = 0.3,
        num_folds: int = 3,
    ):
        super().__init__()
        self.network = network
        self.simulation_steps = simulation_steps
        self.consistency_weight = consistency_weight
        self.novelty_weight = novelty_weight
        self.utility_weight = utility_weight
        self.num_folds = max(1, int(num_folds))

        # Statistics
        self.register_buffer("simulations_run", torch.tensor(0))
        self.register_buffer("hypotheses_accepted", torch.tensor(0))
        self.register_buffer("hypotheses_rejected", torch.tensor(0))
        self.register_buffer("hypotheses_deferred", torch.tensor(0))

        # Reused baseline copy (D3 isolation: the live network is never touched)
        self._baseline_clone = None
        self._baseline_errors = None

    # ------------------------------------------------------------------
    # Isolation helpers (owner D3: COPY affected state)
    # ------------------------------------------------------------------
    def _make_clone(self) -> 'RelationalNetwork':
        """Deep copy of the live network (owner D3: COPY isolation).

        pickle round-trip rather than copy.deepcopy: the live network's
        dynamic state (neuron_states) can carry an autograd graph after live
        learn=True steps, and deepcopy refuses non-leaf tensors. Pickling
        copies the data as leaves and never touches the live network.
        All measurement stepping runs under torch.no_grad() so the clone
        builds no graph either.
        """
        return pickle.loads(pickle.dumps(self.network))

    def _install_hypothesis(self, clone: 'RelationalNetwork',
                            hypothesis: 'HypothesizedRelationship') -> str:
        """Install the hypothesis weight on the CLONE. Returns install mode.

        The hypothesis weight w is installed as the connection's effective
        weight a = blend*u + (1-blend)*theta exactly: u_fast = w / blend,
        theta_slow = 0. On the clone only -- the live network is untouched.
        """
        blend = float(_live_connection_params(clone)["blend_ratio"])
        blend = blend if blend > 0 else 1.0
        w = hypothesis.hypothesis_weight.detach().clone()

        key = _connection_key(clone, hypothesis.pre_idx, hypothesis.post_idx)
        if key is not None:
            conn = clone.connections[key]
            with torch.no_grad():
                conn.plasticity.u_fast.copy_(w / blend)
                conn.plasticity.theta_slow.zero_()  # measurement-clone-only (owner D3: disposable copy, never live state)
            return "existing_connection"

        # New connection: created on the clone only, with live params.
        params = _live_connection_params(clone)
        conn = PlasticRelationalConnection(
            pre_idx=hypothesis.pre_idx,
            post_idx=hypothesis.post_idx,
            pre_dim=clone.neuron_dim,
            post_dim=clone.neuron_dim,
            context_dim=clone.context_dim,
            relation_type=clone.relation_type,
            learning_rate=params["learning_rate"],
            trace_decay=0.95,
            consolidation_rate=0.001,
            blend_ratio=params["blend_ratio"],
        )
        with torch.no_grad():
            conn.plasticity.u_fast.copy_(w / params["blend_ratio"])
        new_key = f"conn_{len(clone.connections)}"
        clone.connections[new_key] = conn
        clone.rebuild_connection_indices()
        return "new_connection"

    @staticmethod
    def _measure_errors(net: 'RelationalNetwork',
                        fold_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Mean connection_prediction_mse over the fold batches, learn=False.

        Each fold batch carries the stored (input, target) window pair, so
        the measurement scores prediction of the STORED target.
        """
        total = 0.0
        counted = 0
        net.eval()
        with torch.no_grad():
            for input_batch, target_batch in fold_batches:
                out = net.step(input_batch, target_next=target_batch, learn=False)
                mse = out.get("connection_prediction_mse")
                if mse is not None:
                    total += float(mse)
                    counted += 1
        return total / max(counted, 1)

    @staticmethod
    def _split_folds(experiences: List, num_folds: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Split experiences into disjoint folds of stored (input, target)
        window batches, each (B, num_neurons, neuron_dim)."""
        if not experiences:
            return []
        inputs = torch.stack([
            e.input_signal if e.input_signal.dim() == 2 else e.input_signal[-1]
            for e in experiences
        ])
        targets = torch.stack([
            e.target_next if e.target_next.dim() == 2 else e.target_next[-1]
            for e in experiences
        ])
        folds = max(1, min(num_folds, inputs.shape[0]))
        per_fold = max(1, inputs.shape[0] // folds)
        batches = []
        for i in range(folds):
            inp = inputs[i * per_fold:(i + 1) * per_fold]
            tgt = targets[i * per_fold:(i + 1) * per_fold]
            if inp.shape[0] > 0:
                batches.append((inp, tgt))
        return batches

    # ------------------------------------------------------------------
    # Measured scores
    # ------------------------------------------------------------------
    def baseline_session(self, experiences_utility: List, experiences_consistency: List) -> Dict:
        """Measure the unmodified network once per dream cycle.

        Runs on its own deep copy (D3). Reused by every hypothesis in the
        cycle -- the baseline is identical for all of them.
        """
        utility_folds = self._split_folds(experiences_utility, self.num_folds)
        consistency_folds = self._split_folds(experiences_consistency, self.num_folds)
        clone = self._make_clone()
        return {
            "utility_folds": utility_folds,
            "consistency_folds": consistency_folds,
            "utility_err_baseline": self._measure_errors(clone, utility_folds),
            "consistency_err_baseline": self._measure_errors(clone, consistency_folds),
        }

    def _measured_consistency(self, baseline_err: float, installed_err: float) -> float:
        """1 - normalised error increase on non-source experiences."""
        increase = (installed_err - baseline_err) / max(abs(baseline_err), 1e-12)
        return float(max(0.0, min(1.0, 1.0 - increase)))

    def _measured_utility(self, baseline_err: float, installed_err: float) -> Tuple[float, float]:
        """Normalised prediction improvement on source experiences.

        Returns (clamped score, absolute delta). Absolute delta = baseline
        error - installed error (positive = the hypothesis helped).
        """
        delta_abs = baseline_err - installed_err
        norm = delta_abs / max(abs(baseline_err), 1e-12)
        return float(max(-1.0, min(1.0, norm))), float(delta_abs)

    def _measured_novelty(self, hypothesis: 'HypothesizedRelationship') -> float:
        """Max cosine distance between the hypothesis weight and existing
        connection weights (actually checked against the network)."""
        w = hypothesis.hypothesis_weight.detach().flatten()
        w_norm = w.norm()
        if w_norm < 1e-12:
            return 0.0
        max_dist = 0.0
        for conn in self.network.connections.values():
            cw = conn.plasticity.get_combined_weight().detach().flatten()
            if cw.shape != w.shape:
                continue
            denom = w_norm * cw.norm()
            if denom < 1e-12:
                dist = 1.0
            else:
                cos = float(torch.dot(w, cw) / denom)
                dist = 1.0 - cos
            max_dist = max(max_dist, dist)
        return float(max(0.0, min(1.0, max_dist)))

    def _measured_forgetting_risk(self, hypothesis: 'HypothesizedRelationship',
                                  consistency_experiences: List) -> float:
        """Measured error increase on consolidated, high-gate connections.

        Restricted to the held-out (non-source) experiences that exercise at
        least one consolidated connection (consolidation_rate > 0 with an
        open selective gate). If no such connections exist, there is no
        theta channel to overwrite and the risk is 0 by measurement of the
        network, not by assumption.
        """
        high_gate = []
        for conn in self.network.connections.values():
            if conn.plasticity.consolidation_rate <= 0:
                continue
            gate = getattr(conn.plasticity, "consolidation_gate", None)
            if gate is None:
                continue
            open_frac = float((torch.sigmoid(gate.detach()) > 0.5).float().mean())
            if open_frac > 0.5:
                high_gate.append(conn)

        if not high_gate:
            return 0.0

        # Experiences whose windows exercise a high-gate connection.
        exercised = []
        for e in consistency_experiences:
            window = e.input_signal if e.input_signal.dim() == 2 else e.input_signal[-1]
            for conn in high_gate:
                pre = window[conn.pre_idx]
                post = window[conn.post_idx]
                if float((pre.abs() * post.abs()).mean()) > 1e-4:
                    exercised.append(e)
                    break

        if not exercised:
            return 0.0

        folds = self._split_folds(exercised, self.num_folds)
        base_clone = self._make_clone()
        inst_clone = self._make_clone()
        self._install_hypothesis(inst_clone, hypothesis)
        err_base = self._measure_errors(base_clone, folds)
        err_inst = self._measure_errors(inst_clone, folds)
        increase = (err_inst - err_base) / max(abs(err_base), 1e-12)
        return float(max(0.0, min(1.0, increase)))

    def _measured_efficiency(self, hypothesis: 'HypothesizedRelationship',
                             install_mode: str) -> float:
        """Measured capacity cost: fraction of network capacity the
        hypothesis adds (1.0 when it updates an existing connection and adds
        no capacity)."""
        if install_mode == "existing_connection":
            return 1.0
        numel_hyp = int(hypothesis.hypothesis_weight.numel())
        numel_net = sum(
            int(conn.plasticity.get_combined_weight().numel())
            for conn in self.network.connections.values())
        return float(max(0.0, 1.0 - numel_hyp / max(numel_net + numel_hyp, 1)))

    # ------------------------------------------------------------------
    # The measurement loop
    # ------------------------------------------------------------------
    def simulate_hypothesis(
        self,
        hypothesis: 'HypothesizedRelationship',
        initial_state: torch.Tensor,  # kept for signature compatibility; measurement
                                      # uses the stored replay experiences instead
        context: Optional[torch.Tensor] = None,
        target_sequence: Optional[torch.Tensor] = None,
        experiences_utility: Optional[List] = None,
        experiences_consistency: Optional[List] = None,
        baseline: Optional[Dict] = None,
    ) -> SimulationResult:
        """Measure one hypothesis (spec §3 E1).

        The hypothesis is installed on a deep COPY of the network (owner D3);
        the copy runs the stored experiences with learn=False; the scores are
        the measured deltas against the baseline copy of the unmodified
        network. The live network is never mutated by this loop.
        """
        experiences_utility = experiences_utility if experiences_utility is not None else []
        experiences_consistency = experiences_consistency if experiences_consistency is not None else []

        if not experiences_utility and not experiences_consistency:
            # No stored experiences to measure against: nothing is accepted on
            # constants. Report unmeasured; the evaluator defers.
            self.simulations_run.add_(1)
            self.hypotheses_deferred.add_(1)
            return SimulationResult(
                hypothesis=hypothesis,
                predicted_outcomes=torch.zeros(1),
                prediction_errors=torch.zeros(1),
                consistency_score=0.0, novelty_score=self._measured_novelty(hypothesis),
                utility_score=0.0, stability_estimate=0.0,
                recommendation="defer", measured=False,
                install_mode="none",
            )

        if baseline is None:
            baseline = self.baseline_session(experiences_utility, experiences_consistency)

        utility_folds = baseline["utility_folds"]
        consistency_folds = baseline["consistency_folds"]

        # Install on a COPY (D3) and measure.
        inst_clone = self._make_clone()
        install_mode = self._install_hypothesis(inst_clone, hypothesis)

        utility_err_installed = self._measure_errors(inst_clone, utility_folds)
        consistency_err_installed = self._measure_errors(inst_clone, consistency_folds)

        consistency = self._measured_consistency(
            baseline["consistency_err_baseline"], consistency_err_installed)
        utility, utility_delta_abs = self._measured_utility(
            baseline["utility_err_baseline"], utility_err_installed)
        novelty = self._measured_novelty(hypothesis)

        # Repeated held-out measurements (E1 stability + D4 noise floor):
        # per-fold utility deltas on the source experiences. Fresh clones per
        # fold so baseline and installed start each fold from identical states.
        per_fold_deltas = []
        if utility_folds:
            for fold in utility_folds:
                base_fold = self._make_clone()
                inst_fold = self._make_clone()
                self._install_hypothesis(inst_fold, hypothesis)
                err_base = self._measure_errors(base_fold, [fold])
                err_inst = self._measure_errors(inst_fold, [fold])
                per_fold_deltas.append(err_base - err_inst)
        if len(per_fold_deltas) > 1:
            deltas = torch.tensor(per_fold_deltas, dtype=torch.float64)
            noise_floor = float(deltas.std(unbiased=False))
            # stability: low spread = stable
            stability = 1.0 / (1.0 + noise_floor / max(abs(utility_delta_abs), 1e-12))
        else:
            noise_floor = 0.0
            stability = 1.0

        forgetting_risk = self._measured_forgetting_risk(
            hypothesis, experiences_consistency)
        efficiency = self._measured_efficiency(hypothesis, install_mode)

        # Recommendation bins are REPORTING ONLY (spec E3): the 0.65/0.6
        # constant thresholds are retired from the acceptance path; the
        # evaluator decides acceptance from measured evidence.
        if utility > 0.0:
            recommendation = "accept"
        elif utility > -0.1:
            recommendation = "test_more"
        else:
            recommendation = "reject"

        result = SimulationResult(
            hypothesis=hypothesis,
            predicted_outcomes=torch.tensor([
                baseline["utility_err_baseline"], utility_err_installed,
                baseline["consistency_err_baseline"], consistency_err_installed,
            ]),
            prediction_errors=torch.tensor(per_fold_deltas) if per_fold_deltas else torch.zeros(1),
            consistency_score=consistency,
            novelty_score=novelty,
            utility_score=utility,
            stability_estimate=stability,
            recommendation=recommendation,
            measured=True,
            utility_err_baseline=baseline["utility_err_baseline"],
            utility_err_installed=utility_err_installed,
            utility_delta_abs=utility_delta_abs,
            consistency_err_baseline=baseline["consistency_err_baseline"],
            consistency_err_installed=consistency_err_installed,
            utility_delta_per_fold=per_fold_deltas,
            noise_floor=noise_floor,
            forgetting_risk=forgetting_risk,
            efficiency_score=efficiency,
            install_mode=install_mode,
        )

        self.simulations_run.add_(1)
        if recommendation == "accept":
            self.hypotheses_accepted.add_(1)
        elif recommendation == "reject":
            self.hypotheses_rejected.add_(1)
        else:
            self.hypotheses_deferred.add_(1)

        return result

    def simulate_batch(
        self,
        hypotheses: List['HypothesizedRelationship'],
        initial_state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        experiences_utility: Optional[List] = None,
        experiences_consistency: Optional[List] = None,
    ) -> List[SimulationResult]:
        """Measure multiple hypotheses against one shared baseline session."""
        results = []
        baseline = None
        if experiences_utility or experiences_consistency:
            baseline = self.baseline_session(experiences_utility, experiences_consistency)
        for hyp in hypotheses:
            result = self.simulate_hypothesis(
                hyp, initial_state, context, None,
                experiences_utility=experiences_utility,
                experiences_consistency=experiences_consistency,
                baseline=baseline,
            )
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        return {
            "simulations_run": self.simulations_run.item(),
            "accepted": self.hypotheses_accepted.item(),
            "rejected": self.hypotheses_rejected.item(),
            "deferred": self.hypotheses_deferred.item(),
            "acceptance_rate": (
                self.hypotheses_accepted.item() / max(1, self.simulations_run.item())
            ),
        }


def create_dream_simulator(
    network: 'RelationalNetwork',
    simulation_steps: int = 5,
    consistency_weight: float = 0.4,
    novelty_weight: float = 0.3,
    utility_weight: float = 0.3,
) -> DreamSimulator:
    """Factory for dream simulator."""
    return DreamSimulator(
        network=network,
        simulation_steps=simulation_steps,
        consistency_weight=consistency_weight,
        novelty_weight=novelty_weight,
        utility_weight=utility_weight,
    )
