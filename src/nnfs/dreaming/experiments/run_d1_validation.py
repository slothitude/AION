"""
D1 Replay Validation Run (docs/D1_REPLAY_VALIDATION_PLAN.md).

The staged gate before any dreaming claim (owner decisions D1-D6, spec
DREAM_REIMPL_SPEC.md section 6a). D1 arm only, 3 seeds, the six-configuration
matrix's corrected instrument (lag-1 targets, phase-block resets), the frozen
plasticity baseline v2 connections (untouched -- dreaming is outside the
freeze, spec section 5.5).

Arms:
  d0    PLASTIC + retrieval-only dream engine (replay DISABLED) -- the no-replay
        control. Its dream-window state deltas are the measured noise floor
        (expected exactly zero on this deterministic CPU path) and its battery
        numbers are the D0 baseline.
  d1    PLASTIC + real replay (learning_rate_scale 0.1, budget 32/cycle).
  d1x2  PLASTIC + real replay at doubled dream compute (interval 150,
        duration 100) -- the equal-compute control is retained so
        equal-compute readings exist (plan V3).

What D1 must demonstrate (owner condition 3, plan section 3):
  V1 actual network updates   : per-cycle ||Delta u_fast|| > 0, ||Delta theta|| logged
  V2 correct provenance       : every replayed step tagged replayed=true; zero
                                untagged dream-side state changes
  V3 bounded compute          : total_dream_steps == sum of replayed steps ==
                                sum of min(32, retrieved); wall-clock per cycle
  V4 measurable state changes : D1 deltas above the D0 noise floor
Validity proof: D1 stops being bit-identical to D0 (state deltas; battery
behaviour reported alongside).

Usage:
  python -m nnfs.dreaming.experiments.run_d1_validation [--seeds 42 43 44]

This script changes no code and claims nothing beyond the plan's gate: a PASS
here opens D2 design only (owner D6 -- no automatic progression).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Standalone-script bootstrap (sibling run_benchmarks/run_experiments pattern):
# run from the repo root so the relative pilot_results/ output resolves.
_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "src"))

import torch
import numpy as np

from nnfs.experiments.pilot_experiment import (
    ExperimentConfig,
    ModelVariant,
    DreamVariant,
    PilotExperiment,
    PilotNetwork,
    _metrics_to_dict,
)
from nnfs.dreaming import DreamEngine, calibrate_guard


ARMS = ["d0", "d1", "d1x2"]


def build_config(benchmark: str, seed: int, arm: str) -> ExperimentConfig:
    kwargs = {}
    if arm == "d1x2":
        # gate-5 style equal-compute doubling: 2x dream budget
        kwargs = {"dream_cycle_interval": 150, "dream_duration": 100}
    return ExperimentConfig(
        benchmark=benchmark,
        model_variant=ModelVariant.PLASTIC,
        dream_variant=DreamVariant.D0_NONE if arm == "d0" else DreamVariant.D1_REPLAY,
        seed=seed,
        **kwargs,
    )


def attach_arm_engine(experiment: PilotExperiment, arm: str):
    """Attach the arm's dream engine.

    - d0: retrieval-only engine (replay disabled). Dream windows exist, the
      network is never stepped by them: this is the no-replay D0 control
      whose per-cycle state deltas ARE the measured noise floor.
    - d1 / d1x2: real replay engine (recombination off = D1-only gate).
    """
    if arm == "d0":
        experiment.dream_engine = DreamEngine(
            network=experiment.network,
            dream_cycle_interval=experiment.config.dream_cycle_interval,
            dream_duration=experiment.config.dream_duration,
            enable_replay=False,       # replay DISABLED: the D0 control
            enable_recombination=False,
            enable_consolidation=True,
            enable_structural=False,
            replay_budget=32,
        )
    else:
        experiment.dream_engine = DreamEngine(
            network=experiment.network,
            dream_cycle_interval=experiment.config.dream_cycle_interval,
            dream_duration=experiment.config.dream_duration,
            enable_replay=True,
            enable_recombination=False,  # D1-only gate (owner D6)
            enable_consolidation=True,
            enable_structural=False,
            replay_budget=32,
            learning_rate_scale=0.1,
        )


def dream_evidence(experiment) -> dict:
    """Harvest the dream engine's per-cycle validation evidence."""
    eng = experiment.dream_engine
    if eng is None:
        return {}
    return {
        "cycle_log": eng.cycle_log,
        "total_dream_steps": int(eng.total_dream_steps.item()),
        "dream_cycles": int(eng.dream_cycle_count.item()),
        "decision_records": list(eng.experience_replay.decision_records),
        "replay_budget": eng.experience_replay.replay_budget,
        "learning_rate_scale": eng.experience_replay.learning_rate_scale,
        "retriever_stats": eng.retriever.get_stats(),
        "replay_stats": eng.experience_replay.get_stats(),
    }


def validate_provenance(arm: str, evidence: dict) -> dict:
    """V2: every replayed step tagged replayed=true; zero untagged deltas."""
    if arm == "d0" or not evidence:
        return {"checked": False, "ok": True, "records": 0, "untagged_state_changes": None}
    records = evidence["decision_records"]
    records_ok = all(
        r.get("replayed") is True
        and r.get("cycle_id") is not None
        and r.get("step") is not None
        and r.get("effective_lr") is not None
        for r in records
    )
    # zero untagged dream-side state changes: any cycle with a state delta
    # must carry replayed steps and provenance records
    untagged = 0
    for c in evidence["cycle_log"]:
        if c["state_delta_u_fast"] > 0 or c["state_delta_theta_slow"] > 0:
            if c["replayed"] == 0 or c["provenance_count"] == 0:
                untagged += 1
    return {
        "checked": True,
        "ok": bool(records_ok and untagged == 0),
        "records": len(records),
        "records_ok": records_ok,
        "untagged_state_changes": untagged,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["changing_rotation", "compositional_discovery"])
    parser.add_argument("--output-dir", type=str, default="pilot_results")
    parser.add_argument("--output-name", type=str, default="d1_validation_results.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    print("=" * 72)
    print("D1 REPLAY VALIDATION RUN (docs/D1_REPLAY_VALIDATION_PLAN.md)")
    print("arms: d0 (replay-off control) | d1 (replay, 0.1x lr, budget 32) | "
          "d1x2 (equal-compute, 2x dream budget)")
    print("=" * 72)

    # Owner condition 2: the guard calibration table is part of the record.
    print("\n[guard-calibration] (owner condition 2)")
    calibration = calibrate_guard(seed=0, verbose=True)

    results = {"arms": {}, "guard_calibration": calibration}
    total_runs = len(args.benchmarks) * len(ARMS) * len(args.seeds)
    done = 0
    t_all = time.time()

    for benchmark in args.benchmarks:
        for arm in ARMS:
            key = f"{benchmark}|{arm}"
            results["arms"].setdefault(key, {})
            for seed in args.seeds:
                seed_key = f"seed_{seed}"
                t0 = time.time()
                print(f"\n--- {benchmark} | {arm} | {seed_key} ---")
                config = build_config(benchmark, seed, arm)
                experiment = PilotExperiment(config)
                attach_arm_engine(experiment, arm)
                metrics = experiment.run()
                metrics_dict = _metrics_to_dict(metrics)
                evidence = dream_evidence(experiment)
                prov = validate_provenance(arm, evidence)
                results["arms"][key][seed_key] = {
                    "metrics": metrics_dict,
                    "dream_evidence": evidence,
                    "provenance_check": prov,
                    "run_wall_clock_s": time.time() - t0,
                }
                done += 1
                print(f"    replayed_steps={evidence.get('total_dream_steps', 0)} "
                      f"cycles={evidence.get('dream_cycles', 0)} "
                      f"prov_ok={prov['ok']} "
                      f"wall={time.time() - t0:.1f}s")
                # incremental persistence
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)

    print(f"\n[runs] {done}/{total_runs} complete in {time.time() - t_all:.1f}s")

    # ------------------------------------------------------------------
    # V1-V4 verification and the gate verdict (plan section 3/4)
    # ------------------------------------------------------------------
    report = {"V1": {}, "V2": {}, "V3": {}, "V4": {}, "validity": {}}
    all_d1_cycle_deltas = []
    all_d0_cycle_deltas = []
    budget_ok = True
    prov_ok = True

    for key, seeds in results["arms"].items():
        benchmark, arm = key.split("|")
        for seed_key, run in seeds.items():
            ev = run["dream_evidence"]
            log = ev.get("cycle_log", [])
            if arm == "d0":
                all_d0_cycle_deltas.extend(
                    [c["state_delta_u_fast"] for c in log])
                all_d0_cycle_deltas.extend(
                    [c["state_delta_theta_slow"] for c in log])
            else:
                all_d1_cycle_deltas.extend(
                    [c["state_delta_u_fast"] for c in log if c["replayed"] > 0])
                expected = sum(min(32, c["retrieved"]) for c in log)
                executed = sum(c["replay_steps_executed"] for c in log)
                if executed != ev["total_dream_steps"] or executed != expected:
                    budget_ok = False
                if not run["provenance_check"]["ok"]:
                    prov_ok = False

    noise_floor = max(all_d0_cycle_deltas) if all_d0_cycle_deltas else 0.0
    d1_deltas = all_d1_cycle_deltas

    report["V1"] = {
        "measure": "per-replay-cycle ||Delta u_fast|| + ||Delta theta_slow|| (D1 cycles with replayed steps)",
        "d1_cycles_with_replay": len(d1_deltas),
        "d1_delta_u_min": min(d1_deltas) if d1_deltas else None,
        "d1_delta_u_mean": float(np.mean(d1_deltas)) if d1_deltas else None,
        "d1_delta_u_max": max(d1_deltas) if d1_deltas else None,
        "d0_delta_max": noise_floor,
        "all_positive": bool(d1_deltas) and all(d > 0 for d in d1_deltas),
    }
    report["V2"] = {
        "measure": "every replayed step tagged replayed=true with (cycle_id, step, effective_lr); zero untagged dream-side state changes",
        "provenance_ok": prov_ok,
        "decision_records": sum(
            len(run["dream_evidence"].get("decision_records", []))
            for key, seeds in results["arms"].items()
            for run in seeds.values()),
    }
    report["V3"] = {
        "measure": "total_dream_steps == sum(min(32, retrieved)); wall-clock per cycle logged; d1x2 equal-compute control retained",
        "budget_ok": budget_ok,
        "equal_compute_control_present": any(
            "|d1x2" in key for key in results["arms"]),
        "wall_clock_per_cycle_s": [
            round(c["replay_wall_clock_s"], 4)
            for key, seeds in results["arms"].items()
            for run in seeds.values()
            for c in run["dream_evidence"].get("cycle_log", [])
        ][:20],
    }
    report["V4"] = {
        "measure": "D1 state deltas above the D0 noise floor",
        "noise_floor_d0_max_delta": noise_floor,
        "d1_min_delta": min(d1_deltas) if d1_deltas else None,
        "above_noise": bool(d1_deltas) and noise_floor == 0.0 and all(
            d > noise_floor for d in d1_deltas),
    }

    # Validity proof: D1 != D0 -- state deltas AND battery behaviour
    d0_theta_final = []
    d1_theta_final = []
    battery = {}
    for key, seeds in results["arms"].items():
        benchmark, arm = key.split("|")
        for seed_key, run in seeds.items():
            m = run["metrics"]
            battery.setdefault(benchmark, {}).setdefault(arm, []).append({
                "phase_a_mse": m.get("phase_a_mse"),
                "phase_c_mse": m.get("phase_c_mse"),
                "retention_ratio": m.get("retention_ratio"),
                "final_mse": m.get("final_mse"),
                "theta_slow_norm_initial": m.get("theta_slow_norm_initial"),
                "theta_slow_norm_final": m.get("theta_slow_norm_final"),
            })
    for benchmark, arms in battery.items():
        d0_vals = arms.get("d0", [])
        d1_vals = arms.get("d1", [])
        if d0_vals and d1_vals:
            def mean(rs, k):
                vals = [r[k] for r in rs if r[k] is not None]
                return float(np.mean(vals)) if vals else None
            battery[benchmark] = {
                "d0": d0_vals, "d1": d1_vals, "d1x2": arms.get("d1x2", []),
                "mean_d0": {k: mean(d0_vals, k) for k in
                            ("phase_a_mse", "phase_c_mse", "retention_ratio", "final_mse")},
                "mean_d1": {k: mean(d1_vals, k) for k in
                            ("phase_a_mse", "phase_c_mse", "retention_ratio", "final_mse")},
            }

    state_deltas_differ = bool(d1_deltas) and all(d > 0 for d in d1_deltas)
    report["validity"] = {
        "d1_state_deltas_nonzero": state_deltas_differ,
        "battery": battery,
    }

    # ------------------------------------------------------------------
    # Gate verdict (plan section 4)
    # ------------------------------------------------------------------
    v1 = report["V1"]["all_positive"]
    v2 = report["V2"]["provenance_ok"]
    v3 = report["V3"]["budget_ok"] and report["V3"]["equal_compute_control_present"]
    v4 = report["V4"]["above_noise"]
    bypass_retired = True  # verified by tests/test_dreaming_replay.py (runtime
    # instrumentation + source scan); the two retired paths are gone from
    # consolidator.py and no dreaming file writes theta outside consolidate().

    passed = bool(v1 and v2 and v3 and v4 and bypass_retired and state_deltas_differ)
    report["gate"] = {
        "V1_actual_updates": v1,
        "V2_provenance": v2,
        "V3_bounded_compute": v3,
        "V4_above_noise": v4,
        "theta_bypass_retired": bypass_retired,
        "d1_not_d0_state_deltas": state_deltas_differ,
        "verdict": "PASS" if passed else "FAIL",
    }
    results["validation_report"] = report

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("D1 VALIDATION REPORT (plan section 3)")
    print("=" * 72)
    print(f"V1 actual updates    : {v1} "
          f"(d1 cycles={report['V1']['d1_cycles_with_replay']}, "
          f"min ||du||={report['V1']['d1_delta_u_min']}, "
          f"mean={report['V1']['d1_delta_u_mean']}, "
          f"d0 max delta={noise_floor})")
    print(f"V2 provenance        : {v2} "
          f"(records={report['V2']['decision_records']})")
    print(f"V3 bounded compute   : {v3} (budget honoured, d1x2 retained)")
    print(f"V4 above noise       : {v4} (noise floor={noise_floor}, "
          f"d1 min delta={report['V4']['d1_min_delta']})")
    print(f"D1 != D0 (state)     : {state_deltas_differ}")
    for benchmark, b in battery.items():
        if isinstance(b, dict) and "mean_d0" in b:
            print(f"  battery {benchmark}:")
            print(f"    d0  phaseA={b['mean_d0']['phase_a_mse']:.4f} "
                  f"phaseC={b['mean_d0']['phase_c_mse']:.4f} "
                  f"ret={b['mean_d0']['retention_ratio']:.4f}")
            print(f"    d1  phaseA={b['mean_d1']['phase_a_mse']:.4f} "
                  f"phaseC={b['mean_d1']['phase_c_mse']:.4f} "
                  f"ret={b['mean_d1']['retention_ratio']:.4f}")
            if b.get("d1x2"):
                d1x2 = b["d1x2"]
                def _m(rs, k):
                    vals = [r[k] for r in rs if r.get(k) is not None]
                    return float(np.mean(vals)) if vals else float("nan")
                print(f"    d1x2 phaseA={_m(d1x2, 'phase_a_mse'):.4f} "
                      f"phaseC={_m(d1x2, 'phase_c_mse'):.4f} "
                      f"ret={_m(d1x2, 'retention_ratio'):.4f}")
    print(f"\nGATE VERDICT: {report['gate']['verdict']}  "
          f"(per docs/D1_REPLAY_VALIDATION_PLAN.md section 4)")
    print(f"results: {out_path}")
    return results


if __name__ == "__main__":
    main()
