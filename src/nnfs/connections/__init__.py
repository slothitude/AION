"""Plastic relational connections and networks."""

from nnfs.connections.plastic import (
    PlasticRelationalConnection,
    EligibilityTrace,
    FastSlowPlasticity,
    RelationshipFunction,
    ContextualRelation,
    StructuralPlasticity,
)
from nnfs.connections.network import RelationalNetwork
from nnfs.connections.consolidation import ConsolidationManager, ConsolidationConfig

__all__ = [
    "PlasticRelationalConnection",
    "RelationalNetwork",
    "EligibilityTrace",
    "FastSlowPlasticity",
    "RelationshipFunction",
    "ContextualRelation",
    "StructuralPlasticity",
    "ConsolidationManager",
    "ConsolidationConfig",
]