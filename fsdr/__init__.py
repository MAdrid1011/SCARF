"""
FSDR (Feature Similarity Depth Reuse) Module

Hardware simulator for narrowed depth search based on 2D feature space similarity.

Key Components:
- LSHHasher: Locality-Sensitive Hashing for feature signatures
- CacheTable: Semantic-indexed cache with Hamming distance lookup
- FSDRSimulator: Narrowed Depth Search ASIC model

Example:
    from fsdr import FSDRSimulator

    simulator = FSDRSimulator()
    result = simulator.process_frame(gaussians, features, depths, ...)
"""

from .types import (
    FSDRConfig,
    CacheEntry,
    FSDRResult,
)
from .lsh_hasher import LSHHasher
from .cache_table import CacheTable
from .narrowed_search_simulator import FSDRSimulator

__all__ = [
    # Configuration
    'FSDRConfig',

    # Data structures
    'CacheEntry',
    'FSDRResult',

    # Components
    'LSHHasher',
    'CacheTable',

    # Narrowed Depth Search Simulator (ASIC model)
    'FSDRSimulator',
]
