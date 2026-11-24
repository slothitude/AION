#!/usr/bin/env python
"""Run continual learning benchmark experiments."""

import sys
sys.path.insert(0, "src")

from nnfs.experiments.continual_benchmark import run_full_matrix, analyze_results

if __name__ == "__main__":
    print("Running NNFS Continual Learning Benchmark")
    print("=" * 50)
    print("5 Model Types × 4 Dream Levels × 5 Seeds × 3 Benchmarks")
    print("=" * 50)
    
    # Run with fewer seeds for quick testing
    results = run_full_matrix(
        benchmarks=["changing_rotation", "context_switching", "sequential_assoc"],
        seeds=[42, 43, 44],  # 3 seeds for speed
        output_dir="continual_results"
    )
    
    analysis = analyze_results(results)
    
    print("\n=== ANALYSIS ===")
    for benchmark, benchmark_analysis in analysis.items():
        print(f"\n{benchmark}:")
        for model_type, model_analysis in benchmark_analysis.items():
            print(f"  {model_type}:")
            for dream_level, metrics in model_analysis.items():
                print(f"    {dream_level}: MSE imp={metrics['mse_improvement']:.3f}, "
                      f"Ret imp={metrics['retention_improvement']:.3f}, "
                      f"Disc={metrics['discovery_precision']:.3f}, "
                      f"Eff={metrics['structural_efficiency']:.3f}")
    
    print("\nResults saved to continual_results/full_matrix_results.json")