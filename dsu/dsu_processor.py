"""
DSU Processor

Main processing engine for Depth Search Unit.
"""
import torch
from typing import Tuple, Callable, Optional, TYPE_CHECKING

from .types import DSUConfig, DSUResult
from .depth_sampler import DepthSampler
from .cost_volume import CostVolume
from .softmax_aggregator import SoftmaxAggregator

if TYPE_CHECKING:
    from benchmark.cycle_counter import CycleCounter

# Hardware cycle constants for DSU operations
DSU_CYCLE_CONSTANTS = {
    'project': 8,    # 3D-2D projection
    'sample': 4,     # Bilinear sampling
    'cost': 2,       # Cost computation
    'softmax': 6,    # Softmax aggregation
}


class DSUProcessor:
    """
    Main DSU processing engine.
    
    Orchestrates:
        - Projection and feature sampling
        - Cost volume computation
        - Softmax aggregation
        - Statistics extraction
    
    Hardware Mapping:
        - 4 parallel DSU units for throughput
        - Shared memory interface
        - Total: ~5,600 LUTs, 184 DSPs (×4), ~20 cycles
    
    Example:
        processor = DSUProcessor(config)
        
        # Full depth search
        result = processor.search_depth(
            ref_feature, target_feature_map,
            ref_coord, depth_candidates, projection_params
        )
        
        # Single depth cost (for FSDR light verify)
        cost = processor.compute_single_cost(
            ref_feature, target_feature_map,
            ref_coord, depth_idx, depth_candidates, projection_params
        )
    """
    
    def __init__(
        self,
        config: DSUConfig,
        enable_cycle_counting: bool = False,
        cycle_counter: Optional['CycleCounter'] = None,
    ):
        """
        Initialize DSU processor.
        
        Args:
            config: DSU configuration
            enable_cycle_counting: Whether to count hardware cycles
            cycle_counter: Optional external CycleCounter instance
        """
        self.config = config
        self.enable_cycle_counting = enable_cycle_counting
        self.cycle_counter = cycle_counter
        
        self.depth_sampler = DepthSampler(config)
        self.cost_volume = CostVolume(config)
        self.aggregator = SoftmaxAggregator(config)
    
    def _record_cycles(self, operation: str, cycles: int, memory_accesses: int = 0):
        """Record cycles if cycle counting is enabled."""
        if self.enable_cycle_counting and self.cycle_counter is not None:
            self.cycle_counter.record('dsu', operation, cycles, memory_accesses)
    
    def search_depth(
        self,
        ref_feature: torch.Tensor,
        target_feature_map: torch.Tensor,
        ref_coord: torch.Tensor,
        depth_candidates: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> DSUResult:
        """
        Full depth search for a single pixel.
        
        Args:
            ref_feature: [C] reference feature vector
            target_feature_map: [C, H, W] target features
            ref_coord: [2] reference pixel coordinates
            depth_candidates: [D] depth values
            *_intrinsics, *_extrinsics: Camera parameters
        
        Returns:
            DSUResult with depth and statistics
        """
        num_depths = len(depth_candidates)
        
        # Sample target features at all depths
        target_features = self.depth_sampler.sample_target_features(
            target_feature_map, ref_coord, depth_candidates,
            ref_intrinsics, ref_extrinsics,
            tgt_intrinsics, tgt_extrinsics,
        )
        # Record cycles: project + sample per depth
        self._record_cycles('project', DSU_CYCLE_CONSTANTS['project'] * num_depths)
        self._record_cycles('sample', DSU_CYCLE_CONSTANTS['sample'] * num_depths, memory_accesses=num_depths)
        
        # Compute costs
        costs = self.cost_volume.compute_costs(ref_feature, target_features)
        self._record_cycles('cost', DSU_CYCLE_CONSTANTS['cost'] * num_depths)
        
        # Aggregate to depth
        depth, probs = self.aggregator.aggregate(
            costs, depth_candidates, return_distribution=True
        )
        self._record_cycles('softmax', DSU_CYCLE_CONSTANTS['softmax'])
        
        # Extract statistics
        best_idx, peak_prob, second_idx, spread = self.aggregator.extract_statistics(
            probs, depth_candidates
        )
        
        return DSUResult(
            depth=depth,
            probabilities=probs,
            best_idx=best_idx,
            peak_prob=peak_prob,
            second_idx=second_idx,
            spread=spread,
            num_candidates=len(depth_candidates),
        )
    
    def compute_single_cost(
        self,
        ref_feature: torch.Tensor,
        target_feature_map: torch.Tensor,
        ref_coord: torch.Tensor,
        depth_idx: int,
        depth_candidates: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> float:
        """
        Compute single depth cost (for FSDR light verify).
        
        Args:
            ref_feature: [C] reference feature
            target_feature_map: [C, H, W]
            ref_coord: [2]
            depth_idx: Index into depth_candidates
            depth_candidates: [D]
            *_intrinsics, *_extrinsics: Camera parameters
        
        Returns:
            Cost value (higher = better for correlation)
        """
        depth = float(depth_candidates[depth_idx])
        
        # Project
        tgt_coord = self.depth_sampler.project_to_target(
            ref_coord, depth,
            ref_intrinsics, ref_extrinsics,
            tgt_intrinsics, tgt_extrinsics,
        )
        
        # Sample
        tgt_feature = self.depth_sampler.bilinear_sample(
            target_feature_map, tgt_coord
        )
        
        # Compute cost
        return self.cost_volume.compute_single_cost(ref_feature, tgt_feature)
    
    def create_fsdr_cost_fn(
        self,
        ref_feature: torch.Tensor,
        target_feature_map: torch.Tensor,
        ref_coord: torch.Tensor,
        depth_candidates: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> Callable[[torch.Tensor, int], float]:
        """
        Create cost function for FSDR light verify.
        
        Returns:
            fn(feature, depth_idx) -> cost
        """
        def cost_fn(feature: torch.Tensor, depth_idx: int) -> float:
            # For FSDR, return negative cost (so lower is better)
            cost = self.compute_single_cost(
                feature, target_feature_map, ref_coord, depth_idx,
                depth_candidates, ref_intrinsics, ref_extrinsics,
                tgt_intrinsics, tgt_extrinsics,
            )
            # Return negative if correlation (higher=better) so FSDR can use min
            if self.config.cost_type == 'correlation':
                return -cost
            return cost
        
        return cost_fn
    
    def create_fsdr_prob_fn(
        self,
        target_feature_map: torch.Tensor,
        ref_coord: torch.Tensor,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Create probability function for FSDR cache miss.
        
        Returns:
            fn(feature, depth_candidates) -> probabilities
        """
        def prob_fn(feature: torch.Tensor, depth_candidates: torch.Tensor) -> torch.Tensor:
            result = self.search_depth(
                feature, target_feature_map, ref_coord, depth_candidates,
                ref_intrinsics, ref_extrinsics,
                tgt_intrinsics, tgt_extrinsics,
            )
            return result.probabilities
        
        return prob_fn
