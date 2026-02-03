"""
Softmax Aggregator

Probability distribution and depth estimation.
"""
import torch
import torch.nn.functional as F
from typing import Tuple

from .types import DSUConfig


class SoftmaxAggregator:
    """
    Softmax aggregation for depth estimation.
    
    Converts cost/correlation values to probability distribution
    and computes expected depth through soft argmax.
    
    Hardware Mapping:
        - Softmax: exp + sum + divide
        - Expected depth: weighted sum
        - Statistics: argmax + variance
        - Total: ~300 LUTs, 4 DSPs, 6 cycles
    
    Example:
        aggregator = SoftmaxAggregator(config)
        depth = aggregator.aggregate(costs, depth_candidates)
        depth, probs = aggregator.aggregate(costs, candidates, return_distribution=True)
    """
    
    def __init__(self, config: DSUConfig):
        """Initialize softmax aggregator."""
        self.config = config
        self.temperature = config.temperature
        self.cost_type = config.cost_type
    
    def aggregate(
        self,
        costs: torch.Tensor,
        depth_candidates: torch.Tensor,
        return_distribution: bool = False,
    ) -> Tuple[float, torch.Tensor]:
        """
        Aggregate costs to depth estimate via soft argmax.
        
        Args:
            costs: [D] cost/correlation values
            depth_candidates: [D] depth values
            return_distribution: Whether to return probabilities
        
        Returns:
            depth: Expected depth
            probs: (if return_distribution) probability distribution
        """
        # Apply softmax
        if self.cost_type == 'correlation':
            # Higher is better
            probs = F.softmax(costs / self.temperature, dim=0)
        else:
            # Lower is better (negate)
            probs = F.softmax(-costs / self.temperature, dim=0)
        
        # Expected depth
        depth = float((probs * depth_candidates).sum())
        
        if return_distribution:
            return depth, probs
        return depth, probs
    
    def extract_statistics(
        self,
        probs: torch.Tensor,
        depth_candidates: torch.Tensor,
    ) -> Tuple[int, float, int, float]:
        """
        Extract statistics for FSDR cache entry.
        
        Args:
            probs: [D] probability distribution
            depth_candidates: [D] depth values
        
        Returns:
            (best_idx, peak_prob, second_idx, spread)
        """
        D = len(probs)
        
        # Best depth
        best_idx = int(torch.argmax(probs).item())
        peak_prob = float(probs[best_idx])
        
        # Second best
        probs_copy = probs.clone()
        probs_copy[best_idx] = -float('inf')
        second_idx = int(torch.argmax(probs_copy).item())
        
        # Spread (normalized standard deviation)
        mean_depth = (probs * depth_candidates).sum()
        variance = (probs * (depth_candidates - mean_depth) ** 2).sum()
        std_dev = torch.sqrt(variance)
        
        depth_range = depth_candidates[-1] - depth_candidates[0]
        spread = float(std_dev / depth_range) if depth_range > 0 else 0.0
        spread = min(1.0, spread)
        
        return best_idx, peak_prob, second_idx, spread
    
    def set_temperature(self, temperature: float):
        """Set softmax temperature."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature
