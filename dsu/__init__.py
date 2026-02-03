"""
DSU (Depth Search Unit) Module

Hardware simulator for depth prediction through cost volume computation.

Key Components:
- DepthSampler: Projection and bilinear sampling
- CostVolume: Feature matching cost computation
- SoftmaxAggregator: Probability distribution and depth estimation
- DSUProcessor: Main processing engine

Example:
    from dsu import DSUProcessor, DSUConfig
    
    config = DSUConfig()
    processor = DSUProcessor(config)
    
    depth, probs = processor.search_depth(
        ref_feature, target_feature_map,
        ref_coord, depth_candidates, projection_params
    )
"""

from .types import DSUConfig, DSUResult
from .depth_sampler import DepthSampler
from .cost_volume import CostVolume
from .softmax_aggregator import SoftmaxAggregator
from .dsu_processor import DSUProcessor

__all__ = [
    'DSUConfig',
    'DSUResult',
    'DepthSampler',
    'CostVolume',
    'SoftmaxAggregator',
    'DSUProcessor',
]
