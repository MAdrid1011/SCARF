"""
Decision Controller

Selects processing path based on similarity score and generates
remaining pixel indices for sparse/full-continue paths.
"""
from typing import List

from .types import TileConfig


class DecisionController:
    """
    Path selection controller based on similarity thresholds
    
    Decision logic:
    - similarity ≥ high_threshold (0.85) → early_stop
    - low_threshold ≤ similarity < high_threshold → sparse_continue  
    - similarity < low_threshold (0.60) → full_continue
    """
    
    def __init__(self, config: TileConfig):
        """
        Initialize controller with configuration
        
        Args:
            config: TileConfig with thresholds and sparse indices
        
        Raises:
            ValueError: If thresholds are invalid
        """
        self.config = config
        self._validate_thresholds()
    
    def decide_path(self, similarity_score: float) -> str:
        """
        Select processing path based on similarity score
        
        Args:
            similarity_score: Similarity in [0, 1] from probe evaluation
        
        Returns:
            'early_stop', 'sparse_continue', or 'full_continue'
        
        Examples:
            - score = 0.90 → 'early_stop' (high similarity)
            - score = 0.70 → 'sparse_continue' (medium similarity)
            - score = 0.40 → 'full_continue' (low similarity)
        """
        if similarity_score >= self.config.high_similarity_threshold:
            return "early_stop"
        elif similarity_score >= self.config.low_similarity_threshold:
            return "sparse_continue"
        else:
            return "full_continue"
    
    def get_remaining_indices(
        self,
        path: str,
        tile_size: int,
        probe_size: int,
    ) -> List[int]:
        """
        Get pixel indices to process based on path decision
        
        Args:
            path: 'early_stop', 'sparse_continue', or 'full_continue'
            tile_size: Pixels per tile edge
            probe_size: Number of probe pixels already processed
        
        Returns:
            List of pixel indices to process
        
        Examples:
            For 4×4 tile (16 pixels) with probe_size=4:
            - early_stop: [] (no more pixels)
            - sparse_continue: [4, 7, 11, 15] (4 sparse pixels)
            - full_continue: [4, 5, ..., 15] (12 remaining pixels)
        
        Raises:
            ValueError: If path is invalid
        """
        if path not in {"early_stop", "sparse_continue", "full_continue"}:
            raise ValueError(
                f"Invalid path: '{path}'. Must be one of: "
                f"'early_stop', 'sparse_continue', 'full_continue'"
            )
        
        # Early-stop: no more pixels to process
        if path == "early_stop":
            return []
        
        # Sparse-continue: use configured sparse indices
        if path == "sparse_continue":
            return self.config.sparse_indices.copy()
        
        # Full-continue: all remaining pixels
        if path == "full_continue":
            total_pixels = tile_size ** 2
            return list(range(probe_size, total_pixels))
        
        # Should never reach here
        raise RuntimeError(f"Unexpected path: {path}")
    
    def _validate_thresholds(self) -> None:
        """
        Validate threshold configuration
        
        Raises:
            ValueError: If high_threshold ≤ low_threshold
        """
        high = self.config.high_similarity_threshold
        low = self.config.low_similarity_threshold
        
        if high <= low:
            raise ValueError(
                f"high_similarity_threshold ({high}) must be strictly greater than "
                f"low_similarity_threshold ({low})"
            )
        
        # Also validate range [0, 1]
        if not (0.0 <= low <= 1.0):
            raise ValueError(f"low_similarity_threshold ({low}) must be in [0, 1]")
        
        if not (0.0 <= high <= 1.0):
            raise ValueError(f"high_similarity_threshold ({high}) must be in [0, 1]")
    
    @property
    def summary(self) -> str:
        """Human-readable decision summary"""
        return (
            f"DecisionController Configuration:\n"
            f"  High threshold (early-stop): ≥ {self.config.high_similarity_threshold}\n"
            f"  Low threshold (full-continue): < {self.config.low_similarity_threshold}\n"
            f"  Medium range (sparse-continue): [{self.config.low_similarity_threshold}, "
            f"{self.config.high_similarity_threshold})\n"
            f"  Sparse indices: {self.config.sparse_indices}"
        )
