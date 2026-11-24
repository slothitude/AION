#!/usr/bin/env python
"""Run NNFS Pilot Experiments (Phase 5)."""

import sys
sys.path.insert(0, "src")

from nnfs.experiments.pilot_experiment import run_pilot_matrix, analyze_pilot_results

if __name__ == "__main__":
    if "--gate5" in sys.argv:
        # Gate 5 safeguard: EQUAL-COMPUTE D1 — replay-only given ~2x the dream
        # compute (double duration, half the cycle interval) so the D2-vs-D1
        # headline can separate "imagination helps" from "more compute helps".
        from nnfs.experiments.pilot_experiment import (
            ModelVariant, DreamVariant, run_pilot_matrix,
        )
        print("Running AION Phase 5 - Gate 5 equal-compute D1 (2x dream compute)")
        print("=" * 60)
        run_pilot_matrix(
            benchmarks=["changing_rotation", "compositional_discovery"],
            seeds=[42, 43, 44],
            configs=[(ModelVariant.PLASTIC, DreamVariant.D1_REPLAY)],
            experiment_kwargs={"dream_cycle_interval": 150, "dream_duration": 100},
            output_dir="pilot_results",
            output_name="pilot_results_d1x2.json",
        )
        sys.exit(0)

    if "--full" not in sys.argv:
        # Owner directive: full benchmark matrix stays STOPPED.
        print("STOPPED: full pilot matrix requires --full (owner directive: no matrix runs).")
        print("Small controlled runs: use scripts/run_validation.py (diagnostic).")
        sys.exit(1)

    print("Running AION Phase 5 - Pilot Experiments")
    print("=" * 60)
    print("6 Configurations × 2 Benchmarks × 3 Seeds")
    print("=" * 60)
    
    results = run_pilot_matrix(
        benchmarks=["changing_rotation", "compositional_discovery"],
        seeds=[42, 43, 44],
        output_dir="pilot_results"
    )
    
    analysis = analyze_pilot_results(results)
    
    print("\n=== PILOT ANALYSIS ===")
    for benchmark, benchmark_analysis in analysis.items():
        print(f"\n{benchmark}:")
        for config_key, metrics in benchmark_analysis.items():
            print(f"  {config_key}:")
            print(f"    MSE={metrics['final_mse']:.4f}")
            if benchmark == "changing_rotation":
                print(f"    Phase A={metrics['phase_a_mse']:.4f}, B={metrics['phase_b_mse']:.4f}, C={metrics['phase_c_mse']:.4f}")
                print(f"    Adapt B={metrics['adaptation_steps_b']}, C={metrics['adaptation_steps_c']}, Ret={metrics['retention_ratio']:.4f}")
            elif benchmark == "compositional_discovery":
                print(f"    Disc={metrics['discovery_score']:.4f}, vsOracle={metrics['relative_to_oracle']:.4f}")
            dream_prec = "None" if metrics['dream_precision'] is None else f"{metrics['dream_precision']:.4f}"
            print(f"    Conn={metrics['final_connections']:.1f}, DreamPrec={dream_prec}")
    
    print("\nResults saved to pilot_results/pilot_results.json")