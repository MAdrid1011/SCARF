"""
FSDR (Feature-Similarity Depth Reuse) Module

Hardware simulator for semantic-indexed cache that enables depth reuse
based on 2D feature space similarity.

Key Components:
- LSHHasher: Locality-Sensitive Hashing for feature signatures
- CacheTable: Semantic-indexed cache with Hamming distance lookup
- DepthCorrector: Three-level correction strategy
- LightVerifier: Local depth search verification
- FSDRProcessor: Main processing engine
- FSDRProfiler: Performance metrics collection

Example:
    from fsdr import FSDRProcessor, FSDRConfig
    
    config = FSDRConfig()
    depth_candidates = torch.linspace(0.5, 10.0, 32)
    
    processor = FSDRProcessor(config, depth_candidates)
    result = processor.process_pixel(feature, position, cost_fn, prob_fn)
"""

from .types import (
    FSDRConfig,
    CacheEntry,
    FSDRResult,
)
from .lsh_hasher import LSHHasher
from .cache_table import CacheTable
from .depth_corrector import DepthCorrector
from .light_verifier import LightVerifier
from .fsdr_processor import FSDRProcessor
from .profiler import FSDRProfiler

__all__ = [
    # Configuration
    'FSDRConfig',
    
    # Data structures
    'CacheEntry',
    'FSDRResult',
    
    # Components
    'LSHHasher',
    'CacheTable',
    'DepthCorrector',
    'LightVerifier',
    'FSDRProcessor',
    'FSDRProfiler',
]
