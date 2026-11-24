"""Experimental model architectures."""

from nnfs.models.standard_mlp import StandardMLP
from nnfs.models.multiplicative_mlp import MultiplicativeMLP
from nnfs.models.geometric_mlp import GeometricMLP
from nnfs.models.specialized_net import SpecializedNet

__all__ = [
    "StandardMLP",
    "MultiplicativeMLP",
    "GeometricMLP",
    "SpecializedNet",
]