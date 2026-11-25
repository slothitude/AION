"""Dream Engine - Offline relational learning through dreaming."""

from nnfs.dreaming.retriever import MemoryRetriever, RetrievedExperience, create_retriever
from nnfs.dreaming.replay import ExperienceReplay, create_experience_replay
from nnfs.dreaming.guard import ComposedWeightNormGuard, GuardVerdict, calibrate_guard
from nnfs.dreaming.recombiner import PatternRecombiner, HypothesizedRelationship, create_pattern_recombiner
from nnfs.dreaming.simulator import DreamSimulator, SimulationResult, create_dream_simulator
from nnfs.dreaming.evaluator import DreamEvaluator, EvaluationResult, create_dream_evaluator
from nnfs.dreaming.consolidator import DreamConsolidator, ConsolidationResult, create_dream_consolidator
from nnfs.dreaming.structural import StructuralDreamer, StructuralChangeResult, create_structural_dreamer
from nnfs.dreaming.engine import DreamEngine, DreamCycleResult, create_dream_engine
from nnfs.dreaming.benchmarks import DreamingBenchmarkConfig, DreamingBenchmark, run_dreaming_benchmark
from nnfs.dreaming.experiments import (
    ExperimentConfig,
    run_concept_reversal_experiment,
    run_context_switching_experiment,
    run_structural_efficiency_experiment,
    run_all_experiments,
)

__all__ = [
    "MemoryRetriever",
    "RetrievedExperience",
    "create_retriever",
    "ExperienceReplay",
    "create_experience_replay",
    "ComposedWeightNormGuard",
    "GuardVerdict",
    "calibrate_guard",
    "PatternRecombiner",
    "HypothesizedRelationship",
    "create_pattern_recombiner",
    "DreamSimulator",
    "SimulationResult",
    "create_dream_simulator",
    "DreamEvaluator",
    "EvaluationResult",
    "create_dream_evaluator",
    "DreamConsolidator",
    "ConsolidationResult",
    "create_dream_consolidator",
    "StructuralDreamer",
    "StructuralChangeResult",
    "create_structural_dreamer",
    "DreamEngine",
    "DreamCycleResult",
    "create_dream_engine",
    "DreamingBenchmarkConfig",
    "DreamingBenchmark",
    "run_dreaming_benchmark",
    "ExperimentConfig",
    "run_concept_reversal_experiment",
    "run_context_switching_experiment",
    "run_structural_efficiency_experiment",
    "run_all_experiments",
]