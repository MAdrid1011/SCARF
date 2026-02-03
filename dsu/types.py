"""
DSU Data Types

Core data structures for the DSU module.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch


@dataclass
class DSUConfig:
    """
    DSU configuration parameters.
    
    Attributes:
        num_depth_candidates: Number of depth candidates (default: 32)
        feature_dim: Feature vector dimension (default: 128)
        cost_type: 'correlation' (higher=better) or 'cost' (lower=better)
        temperature: Softmax temperature (default: 1.0)
        use_inverse_depth: Whether to use inverse depth spacing
    """
    num_depth_candidates: int = 32
    feature_dim: int = 128
    cost_type: str = 'correlation'
    temperature: float = 1.0
    use_inverse_depth: bool = False
    
    def __post_init__(self):
        if self.cost_type not in ('correlation', 'cost'):
            raise ValueError(f"cost_type must be 'correlation' or 'cost', got {self.cost_type}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


@dataclass
class DSUResult:
    """
    Result of DSU depth search.
    
    Attributes:
        depth: Estimated depth value
        probabilities: [D] probability distribution
        best_idx: Index of best depth candidate
        peak_prob: Peak probability value
        second_idx: Second-best index
        spread: Distribution spread
        num_candidates: Number of candidates searched
    """
    depth: float
    probabilities: Optional[torch.Tensor] = None
    best_idx: int = 0
    peak_prob: float = 0.0
    second_idx: int = 0
    spread: float = 0.0
    num_candidates: int = 32
    
    def get_fsdr_stats(self) -> Dict:
        """Get statistics for FSDR cache entry."""
        return {
            'best_idx': self.best_idx,
            'peak_prob': self.peak_prob,
            'second_offset': self.second_idx - self.best_idx,
            'spread': self.spread,
        }
