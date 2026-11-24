#!/usr/bin/env python
"""Run AION validation diagnostics (Phase 5).

Default behaviour is a SMALL controlled diagnostic run (owner directive:
full matrix stays STOPPED). It runs the 3 D0 model variants
(static / plastic / consolidating) at reduced scale:
2 benchmarks × 1 seed × phase_length=100 (300 steps per run)
and prints the [VALIDATION] metric block per model.
"""

import sys
import json
sys.path.insert(0, "src")

from pathlib import Path

from nnfs.experiments.pilot_experiment import (
    ExperimentConfig,
    ModelVariant,
    DreamVariant,
    run_pilot_experiment,
)

DIAGNOSTIC_MODELS = [ModelVariant.STATIC, ModelVariant.PLASTIC, ModelVariant.CONSOLIDATING]


def _fmt(value):
    if value is None:
        return "None"
    try:
        return f"{value:.4f}"
    except TypeError:
        return str(value)


def run_diagnostic(
    seed: int = 42,
    phase_length: int = 100,
    num_sequences: int = 8,
    models=None,
    consolidation_interval: int = 50,
):
    models = models if models is not None else DIAGNOSTIC_MODELS
    output_path = Path("pilot_results")
    output_path.mkdir(parents=True, exist_ok=True)

    diagnostic = {}

    for model in models:
        metrics = {}
        for benchmark in ["changing_rotation", "compositional_discovery"]:
            config = ExperimentConfig(
                benchmark=benchmark,
                model_variant=model,
                dream_variant=DreamVariant.D0_NONE,
                seed=seed,
                phase_length=phase_length,
                total_steps=3 * phase_length,
                eval_interval=phase_length,
                num_sequences=num_sequences,
                consolidation_interval=consolidation_interval,
            )
            print(f"[diagnostic] running {benchmark} | {model.value} | seed={seed} | "
                  f"{3 * phase_length} steps")
            result = run_pilot_experiment(config)
            metrics[benchmark] = {
                "phase_a_mse": result.phase_a_mse,
                "phase_b_mse": result.phase_b_mse,
                "phase_c_mse": result.phase_c_mse,
                "adaptation_steps_b": result.adaptation_steps_b,
                "adaptation_steps_c": result.adaptation_steps_c,
                "retention_ratio": result.retention_ratio,
                "final_mse": result.final_mse,
                "discovery_score": result.discovery_score,
                "oracle_mse": result.oracle_mse,
                "random_mse": result.random_mse,
                "online_mse_final": (
                    sum(result.online_mse_curve[-100:]) / len(result.online_mse_curve[-100:])
                    if result.online_mse_curve else 0.0
                ),
                "connection_mse_final": (
                    sum(result.online_mse_curve[-100:]) / len(result.online_mse_curve[-100:])
                    if result.online_mse_curve else 0.0
                ),
                "state_mse_final": (
                    sum(result.state_mse_curve[-100:]) / len(result.state_mse_curve[-100:])
                    if result.state_mse_curve else 0.0
                ),
                "readout_mse_final": (
                    sum(result.readout_mse_curve[-100:]) / len(result.readout_mse_curve[-100:])
                    if result.readout_mse_curve else 0.0
                ),
                "dream_precision": result.dream_precision,
                "final_connections": result.final_connections,
            }

        rotation = metrics["changing_rotation"]
        discovery = metrics["compositional_discovery"]

        print()
        print(f"[VALIDATION] model={model.value} seed={seed} phase_length={phase_length}")
        print(f"Phase A MSE: {_fmt(rotation['phase_a_mse'])}")
        print(f"Phase B MSE: {_fmt(rotation['phase_b_mse'])}")
        print(f"Phase C MSE: {_fmt(rotation['phase_c_mse'])}")
        print(f"Adaptation steps: B={_fmt(rotation['adaptation_steps_b'])}, "
              f"C={_fmt(rotation['adaptation_steps_c'])}")
        print(f"Discovery error: {_fmt(discovery['discovery_score'])}")
        print(f"[diagnostic] connection_mse={_fmt(rotation['connection_mse_final'])} "
              f"state_mse={_fmt(rotation['state_mse_final'])} "
              f"readout_mse={_fmt(rotation['readout_mse_final'])} "
              f"(last 100 training steps, primary = connection)")
        print()

        diagnostic[model.value] = metrics

    with open(output_path / "validation_diagnostic.json", "w") as f:
        json.dump(diagnostic, f, indent=2, default=str)
    print(f"Diagnostic results saved to {output_path / 'validation_diagnostic.json'}")
    return diagnostic


def run_full_validation(*args, **kwargs):
    """Blocked by owner directive: full benchmark matrix stays STOPPED."""
    raise SystemExit(
        "STOPPED: full validation matrix is blocked by owner directive. "
        "Use the default small diagnostic (run_validation.py)."
    )


if __name__ == "__main__":
    print("Running AION Phase 5 - Small Validation Diagnostic")
    print("=" * 60)
    print("3 D0 model variants x 2 benchmarks x 1 seed x 300 steps (reduced phases)")
    print("=" * 60)

    run_diagnostic()
