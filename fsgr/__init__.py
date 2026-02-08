"""
FSGR (Feature-Similarity Gaussian Reuse) Module

Hardware simulator for narrowed depth search based on 2D feature space similarity.

Key Components:
- LSHHasher: Locality-Sensitive Hashing for feature signatures
- CacheTable: Semantic-indexed cache with Hamming distance lookup
- FSGRSimulator: Narrowed Depth Search ASIC model

Example:
    from fsgr import FSGRSimulator

    simulator = FSGRSimulator()
    result = simulator.process_frame(gaussians, features, depths, ...)
"""

from .types import (
    FSGRConfig,
    CacheEntry,
    FSGRResult,
)
from .lsh_hasher import LSHHasher
from .cache_table import CacheTable
from .narrowed_search_simulator import FSGRSimulator

__all__ = [
    # Configuration
    'FSGRConfig',

    # Data structures
    'CacheEntry',
    'FSGRResult',

    # Components
    'LSHHasher',
    'CacheTable',

    # Narrowed Depth Search Simulator (ASIC model)
    'FSGRSimulator',
]
