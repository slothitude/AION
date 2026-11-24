"""Benchmark tasks for evaluating neural architectures."""

from nnfs.benchmarks.xor_parity import XORDataset, ParityDataset
from nnfs.benchmarks.rotated_mnist import RotatedMNIST
from nnfs.benchmarks.sequential import SequentialPrediction, NoisyPatternCompletion
from nnfs.benchmarks.circular import CircularPatterns, InterferencePatterns
from nnfs.benchmarks.continual import (
    ChangingRelationshipDataset,
    SequentialAssociationDataset,
    ContextDependentMappingDataset,
    ContextSwitchingDataset,
    get_changing_relationship_loaders,
    get_context_switching_loaders,
    get_sequential_association_loaders,
)

__all__ = [
    "XORDataset",
    "ParityDataset", 
    "RotatedMNIST",
    "SequentialPrediction",
    "NoisyPatternCompletion",
    "CircularPatterns",
    "InterferencePatterns",
    "ChangingRelationshipDataset",
    "SequentialAssociationDataset",
    "ContextDependentMappingDataset",
    "ContextSwitchingDataset",
    "get_changing_relationship_loaders",
    "get_context_switching_loaders",
    "get_sequential_association_loaders",
]