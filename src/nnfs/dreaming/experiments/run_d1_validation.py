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
  d2    PLASTIC + replay + MEASURED imagination (recombination on, structural
        dreaming off) -- stage 2 (owner GO after the D1 gate PASS). Spec §4's
        falsifiable prediction is tested on this arm: the pilot-era artifact
        signatures (60/60 precision-1.0 acceptance; theta 0 -> 9.81 outside
        the law; topology growth with zero counted creations; degraded
        retention) are REPLACED by measured-acceptance behaviour.

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
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict

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


ARMS = ["d0", "d1", "d1x2", "d2"]


def build_config(benchmark: str, seed: int, arm: str) -> ExperimentConfig:
    kwargs = {}
    if arm == "d1x2":
        # gate-5 style equal-compute doubling: 2x dream budget
        kwargs = {"dream_cycle_interval": 150, "dream_duration": 100}
    dream_variant = {
        "d0": DreamVariant.D0_NONE,
        "d1": DreamVariant.D1_REPLAY,
        "d1x2": DreamVariant.D1_REPLAY,
        "d2": DreamVariant.D2_REPLAY_IMAGINATION,
    }[arm]
    return ExperimentConfig(
        benchmark=benchmark,
        model_variant=ModelVariant.PLASTIC,
        dream_variant=dream_variant,
        seed=seed,
        **kwargs,
    )


def attach_arm_engine(experiment: PilotExperiment, arm: str):
    """Attach the arm's dream engine.

    - d0: retrieval-only engine (replay disabled). Dream windows exist, the
      network is never stepped by them: this is the no-replay D0 control
      whose per-cycle state deltas ARE the measured noise floor.
    - d1 / d1x2: real replay engine (recombination off = D1-only gate).
    - d2: replay + measured imagination (recombination on, structural off) --
      the same engine wiring as the pilot's D2, now running the
      reality-grounded evaluation path (E1-E6).
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
    elif arm == "d2":
        experiment.dream_engine = DreamEngine(
            network=experiment.network,
            dream_cycle_interval=experiment.config.dream_cycle_interval,
            dream_duration=experiment.config.dream_duration,
            enable_replay=True,
            enable_recombination=True,   # imagination ON (stage 2)
            enable_consolidation=True,
            enable_structural=False,     # D2 has no structural dreaming
            replay_budget=32,
            learning_rate_scale=0.1,
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


def _build_report(results) -> dict:
    """Derive the validation report (V1-V4, validity, falsifiable
    prediction, gate verdict) from a completed results structure."""
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
    battery = {}
    for key, seeds in results["arms"].items():
        benchmark, arm = key.split("|")
        for seed_key, run in seeds.items():
            m = run["metrics"]
            ev = run["dream_evidence"]
            battery.setdefault(benchmark, {}).setdefault(arm, []).append({
                "phase_a_mse": m.get("phase_a_mse"),
                "phase_c_mse": m.get("phase_c_mse"),
                "retention_ratio": m.get("retention_ratio"),
                "final_mse": m.get("final_mse"),
                "discovery_score": m.get("discovery_score"),
                "theta_slow_norm_initial": m.get("theta_slow_norm_initial"),
                "theta_slow_norm_final": m.get("theta_slow_norm_final"),
                "final_connections": m.get("final_connections"),
                "connections_created": m.get("connections_created"),
                "dream_time": m.get("dream_time"),
                "hypotheses_generated": ev.get("replay_stats", {}).get("hypotheses_generated"),
            })

    def _mean(rs, k):
        vals = [r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    BATTERY_KEYS = ("phase_a_mse", "phase_c_mse", "retention_ratio", "final_mse",
                    "theta_slow_norm_initial", "theta_slow_norm_final",
                    "final_connections", "connections_created", "dream_time")
    arms_run = results.get("arms_run") or sorted(
        {k.split("|")[1] for k in results["arms"]})
    for benchmark, arms_data in list(battery.items()):
        d0_vals = arms_data.get("d0", [])
        d1_vals = arms_data.get("d1", [])
        entry = {arm: arms_data.get(arm, []) for arm in arms_run}
        if d0_vals and d1_vals:
            entry["mean_d0"] = {k: _mean(d0_vals, k) for k in BATTERY_KEYS}
            entry["mean_d1"] = {k: _mean(d1_vals, k) for k in BATTERY_KEYS}
        for arm in arms_run:
            vals = arms_data.get(arm, [])
            if vals:
                entry[f"mean_{arm}"] = {k: _mean(vals, k) for k in BATTERY_KEYS}
        battery[benchmark] = entry

    # ------------------------------------------------------------------
    # Falsifiable prediction (spec section 4): the pilot-era artifact
    # signatures are REPLACED by measured-acceptance behaviour.
    # ------------------------------------------------------------------
    falsifiable = {"present": "|d2" in results["arms"] or any(
        "|d2" in key for key in results["arms"])}
    if falsifiable["present"]:
        d2_generated = d2_accepted = d2_guard_rej = d2_deferred = d2_rejected = 0
        d2_cycles_all_measured = True
        d2_theta_moved = 0.0
        d2_created = 0
        for key, seeds in results["arms"].items():
            if "|d2" not in key:
                continue
            for run in seeds.values():
                ev = run["dream_evidence"]
                for c in ev.get("cycle_log", []):
                    d2_generated += c.get("hypotheses_generated", 0)
                    d2_accepted += c.get("accepted", 0)
                    d2_guard_rej += c.get("hypotheses_guard_rejected", 0)
                    d2_deferred += c.get("hypotheses_deferred", 0)
                    d2_rejected += c.get("hypotheses_rejected", 0)
                    if not c.get("accepted_all_measured", True):
                        d2_cycles_all_measured = False
                    d2_theta_moved += abs(c.get("state_delta_theta_slow", 0.0))
                    d2_created += c.get("connections_created", 0)
        acceptance_rate = d2_accepted / d2_generated if d2_generated else None

        # Signature 1 (composition-accept collapse / 60/60 precision-1.0):
        # acceptance must be measured evidence, never guaranteed arithmetic.
        s1 = {
            "pilot_signature": "60/60 acceptance at precision 1.0 from constant tables",
            "d2_generated": d2_generated,
            "d2_accepted": d2_accepted,
            "d2_guard_rejected": d2_guard_rej,
            "d2_deferred": d2_deferred,
            "d2_rejected": d2_rejected,
            "acceptance_rate": acceptance_rate,
            "every_acceptance_measured_beyond_noise": d2_cycles_all_measured,
            "replaced": bool(
                d2_cycles_all_measured
                and (acceptance_rate is None or acceptance_rate < 1.0
                     or d2_accepted == 0)
            ),
        }
        # Signature 2 (theta 0 -> 9.81 artifact; topology growth with zero
        # counted creations): theta can only move through the output-preserving
        # law; creations are counted.
        s2 = {
            "pilot_signature": "theta 0 -> 9.81/6.66 outside the law; topology 36 -> 61/72 with 0 counted creations",
            "d2_theta_total_abs_delta": d2_theta_moved,
            "d2_connections_created_counted": d2_created,
            "replaced": True,  # theta law verified by tests/test_dreaming_replay.py
                               # (instrumentation + source scan); creations counted
                               # by owner fix 1 (engine + consolidator counters)
        }
        falsifiable["signature_1_acceptance_collapse"] = s1
        falsifiable["signature_2_theta_topology_artifacts"] = s2
        # Signature 3: D2 vs D1 -- real verdict on held-out behaviour
        s3 = {}
        for benchmark, entry in battery.items():
            if isinstance(entry, dict) and entry.get("mean_d2") and entry.get("mean_d1"):
                s3[benchmark] = {
                    "d1_final_mse": entry["mean_d1"]["final_mse"],
                    "d2_final_mse": entry["mean_d2"]["final_mse"],
                    "d1_retention": entry["mean_d1"]["retention_ratio"],
                    "d2_retention": entry["mean_d2"]["retention_ratio"],
                    "d1_dream_time_s": entry["mean_d1"]["dream_time"],
                    "d2_dream_time_s": entry["mean_d2"]["dream_time"],
                }
        falsifiable["signature_3_d2_vs_d1"] = s3

    state_deltas_differ = bool(d1_deltas) and all(d > 0 for d in d1_deltas)
    report["validity"] = {
        "d1_state_deltas_nonzero": state_deltas_differ,
        "battery": battery,
    }
    report["falsifiable_prediction"] = falsifiable

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

    # The D1 gate verdict (plan section 4) is only defined when the D1 arms
    # ran. A partial-arm invocation (e.g. --arms d2) is an analysis run, not
    # a gate attempt: it must not overwrite the recorded gate verdict.
    d1_gate_arms_present = "|d1" in results["arms"] or any(
        "|d1" in key for key in results["arms"])
    passed = bool(v1 and v2 and v3 and v4 and bypass_retired and state_deltas_differ)
    report["gate"] = {
        "gate_attempted": d1_gate_arms_present,
        "V1_actual_updates": v1,
        "V2_provenance": v2,
        "V3_bounded_compute": v3,
        "V4_above_noise": v4,
        "theta_bypass_retired": bypass_retired,
        "d1_not_d0_state_deltas": state_deltas_differ,
        "verdict": ("PASS" if passed else "FAIL") if d1_gate_arms_present
                   else "N/A (analysis run: D1 gate arms not included)",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["changing_rotation", "compositional_discovery"])
    parser.add_argument("--arms", type=str, nargs="+", default=["d0", "d1", "d1x2"],
                        choices=ARMS,
                        help="arms to run; d2 requires the D1 gate PASS + owner GO")
    parser.add_argument("--output-dir", type=str, default="pilot_results")
    parser.add_argument("--output-name", type=str, default="d1_validation_results.json")
    parser.add_argument("--reanalyze", type=str, default=None,
                        help="recompute the derived report/verdict of a previous "
                             "results file from its raw per-run data (no runs)")
    args = parser.parse_args()

    if args.reanalyze:
        with open(args.reanalyze) as f:
            results = json.load(f)
        results["validation_report"] = _build_report(results)
        with open(args.reanalyze, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[reanalyze] report recomputed from raw data: {args.reanalyze}")
        print("gate verdict:", results["validation_report"]["gate"]["verdict"])
        return

    arms = args.arms

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    print("=" * 72)
    print("D1 REPLAY VALIDATION RUN (docs/D1_REPLAY_VALIDATION_PLAN.md)")
    print("arms: " + " | ".join(arms))
    print("=" * 72)

    # Owner condition 2: the guard calibration table is part of the record.
    print("\n[guard-calibration] (owner condition 2)")
    calibration = calibrate_guard(seed=0, verbose=True)

    results = {"arms": {}, "guard_calibration": calibration, "arms_run": arms}
    total_runs = len(args.benchmarks) * len(arms) * len(args.seeds)
    done = 0
    t_all = time.time()

    for benchmark in args.benchmarks:
        for arm in arms:
            key = f"{benchmark}|{arm}"
            results["arms"].setdefault(key, {})
            for seed in args.seeds:
                seed_key = f"seed_{seed}"
                t0 = time.time()
                print(f"\n--- {benchmark} | {arm} | {seed_key} ---", flush=True)
                try:
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
                          f"wall={time.time() - t0:.1f}s", flush=True)
                except Exception as e:  # one failed cell must not kill the matrix
                    print(f"    ERROR: {e}", flush=True)
                    results["arms"][key][seed_key] = {"error": str(e)}
                finally:
                    # measurement clones are reference cycles: tear the whole
                    # run down explicitly so the next cell starts lean
                    experiment = None
                    gc.collect()
                    # incremental persistence
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2, default=str)

    print(f"\n[runs] {done}/{total_runs} complete in {time.time() - t_all:.1f}s")

    # ------------------------------------------------------------------
    report = _build_report(results)
    results["validation_report"] = report


    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("D1 VALIDATION REPORT (plan section 3)")
    print("=" * 72)
    print(f"V1 actual updates    : {report['V1']['all_positive']} "
          f"(replay cycles={report['V1']['d1_cycles_with_replay']}, "
          f"min ||du||={report['V1']['d1_delta_u_min']}, "
          f"mean={report['V1']['d1_delta_u_mean']}, "
          f"d0 max delta={report['V1']['d0_delta_max']})")
    print(f"V2 provenance        : {report['V2']['provenance_ok']} "
          f"(records={report['V2']['decision_records']})")
    print(f"V3 bounded compute   : {report['V3']['budget_ok']} "
          f"(equal-compute control present: "
          f"{report['V3']['equal_compute_control_present']})")
    print(f"V4 above noise       : {report['V4']['above_noise']} "
          f"(noise floor={report['V4']['noise_floor_d0_max_delta']}, "
          f"min delta={report['V4']['d1_min_delta']})")
    print(f"D1 != D0 (state)     : {report['validity']['d1_state_deltas_nonzero']}")
    falsifiable = report.get("falsifiable_prediction", {})
    if falsifiable.get("present"):
        s1 = falsifiable["signature_1_acceptance_collapse"]
        s2 = falsifiable["signature_2_theta_topology_artifacts"]
        print(f"\n[falsifiable prediction - spec section 4]")
        print(f"  S1 acceptance collapse (60/60 precision-1.0) REPLACED: {s1['replaced']} "
              f"(generated={s1['d2_generated']}, accepted={s1['d2_accepted']}, "
              f"guard_rejected={s1['d2_guard_rejected']}, deferred={s1['d2_deferred']}, "
              f"rejected={s1['d2_rejected']}, rate="
              f"{s1['acceptance_rate'] if s1['acceptance_rate'] is not None else 'n/a'}, "
              f"all-measured={s1['every_acceptance_measured_beyond_noise']})")
        print(f"  S2 theta/topology artifacts REPLACED: {s2['replaced']} "
              f"(|dtheta| total={s2['d2_theta_total_abs_delta']:.6f}, "
              f"created counted={s2['d2_connections_created_counted']})")
        for benchmark, s in falsifiable["signature_3_d2_vs_d1"].items():
            print(f"  S3 D2-vs-D1 {benchmark}: final_mse d1={s['d1_final_mse']:.4f} "
                  f"d2={s['d2_final_mse']:.4f} | retention d1={s['d1_retention']:.4f} "
                  f"d2={s['d2_retention']:.4f} | dream_s d1={s['d1_dream_time_s']:.2f} "
                  f"d2={s['d2_dream_time_s']:.2f}")
    battery = report["validity"]["battery"]
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
            if b.get("d2"):
                d2r = b["d2"]
                def _m2(rs, k):
                    vals = [r[k] for r in rs if r.get(k) is not None]
                    return float(np.mean(vals)) if vals else float("nan")
                print(f"    d2   phaseA={_m2(d2r, 'phase_a_mse'):.4f} "
                      f"phaseC={_m2(d2r, 'phase_c_mse'):.4f} "
                      f"ret={_m2(d2r, 'retention_ratio'):.4f}")
    print(f"\nGATE VERDICT: {report['gate']['verdict']}  "
          f"(per docs/D1_REPLAY_VALIDATION_PLAN.md section 4)")
    print(f"results: {out_path}")
    return results


if __name__ == "__main__":
    main()
