"""
Depth Corrector

Three-level correction strategy for FSDR.
"""
import torch
from typing import Tuple

from .types import FSDRConfig, CacheEntry


class DepthCorrector:
    """
    Three-level depth correction for cache hits.
    
    Strategies:
        1. Direct Reuse: Return cached depth unchanged
           - Condition: peak_prob > 0.8 AND hamming <= 2
           - Memory saved: 100% (0 searches)
        
        2. Interpolation: Blend best and second-best depths
           - Condition: peak_prob > 0.5 OR hamming <= 3
           - Memory saved: 100% (0 searches)
        
        3. Light Verify: Delegated to LightVerifier
           - Condition: Other hits
           - Memory saved: 78-91% (3-7 searches)
    
    Hardware Mapping:
        - Strategy decision: comparators (~50 LUTs)
        - Direct reuse: pass-through (0 cycles)
        - Interpolation: 2 multiplies, 1 add (~100 LUTs, 4 DSPs)
        - Total: ~150 LUTs, 4 DSPs, 1 cycle
    
    Example:
        corrector = DepthCorrector(config, depth_candidates)
        strategy = corrector.decide_strategy(entry, hamming_dist)
        if strategy == 'direct_reuse':
            depth = corrector.direct_reuse(entry)
        elif strategy == 'interpolation':
            depth = corrector.interpolate(entry, hamming_dist)
    """
    
    def __init__(self, config: FSDRConfig, depth_candidates: torch.Tensor):
        """
        Initialize depth corrector.
        
        Args:
            config: FSDR configuration
            depth_candidates: [D] depth candidate values
        """
        self.config = config
        self.depth_candidates = depth_candidates
        self.num_depths = len(depth_candidates)
    
    def decide_strategy(self, entry: CacheEntry, hamming_dist: int) -> str:
        """
        Decide correction strategy based on confidence metrics.
        
        Args:
            entry: Cache entry
            hamming_dist: Hamming distance from query signature
        
        Returns:
            Strategy name: 'direct_reuse', 'interpolation', or 'light_verify'
        
        Hardware:
            - 4 comparators
            - Combinational logic
            - 0 cycles (parallel with lookup)
        """
        peak_prob = entry.peak_prob
        
        # Level 1: High confidence direct reuse
        if (peak_prob > self.config.high_confidence_threshold and 
            hamming_dist <= self.config.hamming_direct_reuse):
            return 'direct_reuse'
        
        # Level 2: Medium confidence interpolation
        if (peak_prob > self.config.medium_confidence_threshold or 
            hamming_dist <= self.config.hamming_interpolate):
            return 'interpolation'
        
        # Level 3: Low confidence light verification
        return 'light_verify'
    
    def direct_reuse(self, entry: CacheEntry) -> float:
        """
        Directly reuse cached depth (highest confidence).
        
        Args:
            entry: Cache entry
        
        Returns:
            Cached best_depth unchanged
        
        Hardware:
            - Pass-through (no computation)
            - 0 cycles
        """
        return entry.best_depth
    
    def interpolate(self, entry: CacheEntry, hamming_dist: int) -> float:
        """
        Interpolate between best and second-best depths.
        
        Formula:
            λ = min(hamming_dist / 4, 0.5)
            depth = (1 - λ) * best_depth + λ * second_depth
        
        Rationale:
            - Higher Hamming → features differ more → more weight on alternative
            - Cap at 0.5 to always favor primary estimate
        
        Args:
            entry: Cache entry
            hamming_dist: Hamming distance from query
        
        Returns:
            Interpolated depth
        
        Hardware:
            - Compute second_depth from offset
            - 1 divide (or shift for powers of 2)
            - 2 multiplies, 1 add
            - ~1 cycle
        """
        # Compute second depth index
        second_idx = entry.best_idx + entry.second_offset
        second_idx = max(0, min(self.num_depths - 1, second_idx))
        second_depth = float(self.depth_candidates[second_idx])
        
        # Interpolation coefficient based on Hamming distance
        # Higher distance → more weight on alternative
        lambda_coeff = hamming_dist / 4.0
        lambda_coeff = min(lambda_coeff, 0.5)  # Cap at 50%
        
        # Linear interpolation
        depth = (1 - lambda_coeff) * entry.best_depth + lambda_coeff * second_depth
        
        return depth
    
    def get_second_depth(self, entry: CacheEntry) -> Tuple[int, float]:
        """
        Get second-best depth index and value.
        
        Args:
            entry: Cache entry
        
        Returns:
            (second_idx, second_depth)
        """
        second_idx = entry.best_idx + entry.second_offset
        second_idx = max(0, min(self.num_depths - 1, second_idx))
        second_depth = float(self.depth_candidates[second_idx])
        return second_idx, second_depth
