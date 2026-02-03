"""
Light Verifier

Local depth search verification for low-confidence cache hits.
"""
import torch
from typing import Callable, Tuple

from .types import FSDRConfig, CacheEntry


class LightVerifier:
    """
    Light verification through local depth search.
    
    Instead of searching all D depth candidates, searches only
    a small range around the cached best depth based on spread.
    
    Search Range Calculation:
        spread_idx = min(max(1, spread * D), max_radius)
        range = [best_idx - spread_idx, best_idx + spread_idx]
        
    Typical search: 3-7 candidates (vs 32 for full search)
    Memory savings: 78-91%
    
    Hardware Mapping:
        - Range calculation: 2 multiplies, 2 clamps (~50 LUTs)
        - Local search: reuse DSU cost calculator
        - Min selector: same as full search but smaller
        - Total: ~100 LUTs, 2 DSPs (shared with DSU)
    
    Example:
        verifier = LightVerifier(config)
        depth, num_searches = verifier.verify(entry, feature, cost_fn)
    """
    
    def __init__(self, config: FSDRConfig):
        """
        Initialize light verifier.
        
        Args:
            config: FSDR configuration
        """
        self.config = config
        self.max_radius = config.max_verify_radius
    
    def calculate_search_range(
        self, 
        entry: CacheEntry, 
        num_depths: int
    ) -> Tuple[int, int]:
        """
        Calculate search range based on cached spread.
        
        Args:
            entry: Cache entry with spread information
            num_depths: Total number of depth candidates
        
        Returns:
            (start_idx, end_idx) - exclusive end
        
        Hardware:
            - spread * num_depths (1 multiply)
            - Clamp to [1, max_radius] (2 comparators)
            - best_idx ± spread_idx with boundary clamp
            - Combinational, ~0 cycles
        """
        # Calculate spread-based radius
        # spread is in [0, 1], multiply by num_depths
        spread_idx = max(1, int(entry.spread * num_depths))
        spread_idx = min(spread_idx, self.max_radius)
        
        # Calculate range with boundary clamping
        start_idx = max(0, entry.best_idx - spread_idx)
        end_idx = min(num_depths, entry.best_idx + spread_idx + 1)
        
        return start_idx, end_idx
    
    def verify(
        self,
        entry: CacheEntry,
        feature: torch.Tensor,
        cost_fn: Callable[[torch.Tensor, int], float],
        depth_candidates: torch.Tensor,
    ) -> Tuple[float, int]:
        """
        Perform local depth search for verification.
        
        Args:
            entry: Cache entry
            feature: Query feature vector
            cost_fn: fn(feature, depth_idx) -> cost (lower is better)
            depth_candidates: [D] depth values
        
        Returns:
            (best_depth, num_searches)
        
        Hardware:
            - Calculate search range (~0 cycles)
            - Search loop: 3-7 iterations
            - Each iteration: 1 cost computation (~20 cycles for DSU)
            - Min tracking: parallel comparators
            - Total: varies, typically 60-140 cycles
        """
        num_depths = len(depth_candidates)
        
        # Calculate search range
        start_idx, end_idx = self.calculate_search_range(entry, num_depths)
        num_searches = end_idx - start_idx
        
        # Local search within range
        min_cost = float('inf')
        best_idx = entry.best_idx  # Default to cached
        
        for idx in range(start_idx, end_idx):
            cost = cost_fn(feature, idx)
            if cost < min_cost:
                min_cost = cost
                best_idx = idx
        
        best_depth = float(depth_candidates[best_idx])
        
        return best_depth, num_searches
    
    def verify_with_correlation(
        self,
        entry: CacheEntry,
        feature: torch.Tensor,
        correlation_fn: Callable[[torch.Tensor, int], float],
        depth_candidates: torch.Tensor,
    ) -> Tuple[float, int]:
        """
        Verify using correlation (higher is better, for MVSplat).
        
        Args:
            entry: Cache entry
            feature: Query feature vector
            correlation_fn: fn(feature, depth_idx) -> correlation (higher is better)
            depth_candidates: [D] depth values
        
        Returns:
            (best_depth, num_searches)
        """
        num_depths = len(depth_candidates)
        
        start_idx, end_idx = self.calculate_search_range(entry, num_depths)
        num_searches = end_idx - start_idx
        
        max_corr = float('-inf')
        best_idx = entry.best_idx
        
        for idx in range(start_idx, end_idx):
            corr = correlation_fn(feature, idx)
            if corr > max_corr:
                max_corr = corr
                best_idx = idx
        
        best_depth = float(depth_candidates[best_idx])
        
        return best_depth, num_searches
