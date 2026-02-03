"""
SAES: Scene-Adaptive Early-Stopping Dataflow

A hardware simulator for tile-based feedback processing in generalizable 3DGS encoders.
"""

from .types import (
    TileConfig,
    GaussianSimilarityMetrics,
    TileProcessingResult,
    SAESProfilingResult,
    Gaussian,
)
from .similarity_evaluator import GaussianSimilarityEvaluator
from .decision_controller import DecisionController
from .gaussian_merger import GaussianMerger
from .profiler import SAESProfiler
from .tile_processor import TileProcessor

__version__ = "0.1.0"

__all__ = [
    "TileConfig",
    "GaussianSimilarityMetrics",
    "TileProcessingResult",
    "SAESProfilingResult",
    "Gaussian",
    "GaussianSimilarityEvaluator",
    "DecisionController",
    "GaussianMerger",
    "SAESProfiler",
    "TileProcessor",
]
