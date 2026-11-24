"""
NNFS - Neural Network Fundamental Structures
Experimental architectures with plastic relational connections, specialized layers, and dreaming.

Phase 1: Alternative neuron mathematics (multiplicative, geometric)
Phase 2: Specialized layer types (Detection, Interaction, Competition, Integration, Prediction)
Phase 3: Plastic Relational Connections with dual-timescale plasticity
Phase 4: Dream Engine - Offline Relational Learning
"""

__version__ = "0.4.0"

from nnfs.config import ExperimentConfig, ModelConfig, TrainingConfig
from nnfs.neurons import StandardNeuron, MultiplicativeNeuron, GeometricNeuron
from nnfs.layers import (
    DetectionLayer,
    InteractionLayer,
    CompetitionLayer,
    IntegrationLayer,
    PredictionLayer,
)
from nnfs.memory import MultiScaleMemory, FastSlowMemory
from nnfs.models import StandardMLP, MultiplicativeMLP, GeometricMLP, SpecializedNet
from nnfs.training import Trainer, create_data_loaders, measure_inference_time, analyze_model_complexity
from nnfs.training.metrics import MetricsTracker, compute_metrics, compute_parameter_efficiency

# Phase 3: Plastic Relational Connections
from nnfs.connections import (
    PlasticRelationalConnection,
    RelationalNetwork,
    EligibilityTrace,
    FastSlowPlasticity,
    RelationshipFunction,
    ContextualRelation,
    StructuralPlasticity,
)

# Phase 4: Dream Engine
from nnfs.dreaming import (
    MemoryRetriever,
    ExperienceReplay,
    PatternRecombiner,
    DreamSimulator,
    DreamEvaluator,
    DreamConsolidator,
    StructuralDreamer,
    DreamEngine,
    create_dream_engine,
)

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "StandardNeuron",
    "MultiplicativeNeuron",
    "GeometricNeuron",
    "DetectionLayer",
    "InteractionLayer",
    "CompetitionLayer",
    "IntegrationLayer",
    "PredictionLayer",
    "MultiScaleMemory",
    "FastSlowMemory",
    "StandardMLP",
    "MultiplicativeMLP",
    "GeometricMLP",
    "SpecializedNet",
    "Trainer",
    "create_data_loaders",
    "measure_inference_time",
    "analyze_model_complexity",
    "MetricsTracker",
    "compute_metrics",
    "compute_parameter_efficiency",
    "PlasticRelationalConnection",
    "RelationalNetwork",
    "EligibilityTrace",
    "FastSlowPlasticity",
    "RelationshipFunction",
    "ContextualRelation",
    "StructuralPlasticity",
    "MemoryRetriever",
    "ExperienceReplay",
    "PatternRecombiner",
    "DreamSimulator",
    "DreamEvaluator",
    "DreamConsolidator",
    "StructuralDreamer",
    "DreamEngine",
    "create_dream_engine",
]