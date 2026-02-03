"""
SAES Type Definitions

Data structures for SAES tile processing, profiling, and Gaussian representation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch
from torch import Tensor


@dataclass
class Gaussian:
    """
    3D Gaussian primitive representation (transplat-decoupled)
    
    Attributes:
        mean: [3] - 3D position (x, y, z)
        cov: [3, 3] - Covariance matrix (must be positive definite)
        opacity: float - Transparency in [0, 1]
        harmonics: [C, D_sh] - Spherical harmonics coefficients for color
    """
    mean: Tensor  # [3]
    cov: Tensor   # [3, 3]
    opacity: float
    harmonics: Tensor  # [C, D_sh]
    
    def to(self, device):
        """Move Gaussian to device"""
        return Gaussian(
            mean=self.mean.to(device),
            cov=self.cov.to(device),
            opacity=self.opacity,
            harmonics=self.harmonics.to(device),
        )
    
    def clone(self):
        """Deep copy of Gaussian"""
        return Gaussian(
            mean=self.mean.clone(),
            cov=self.cov.clone(),
            opacity=self.opacity,
            harmonics=self.harmonics.clone(),
        )


@dataclass
class TileConfig:
    """
    SAES tile processing configuration
    
    Attributes:
        tile_size: Pixels per tile edge (e.g., 4 means 4×4 = 16 pixels per tile)
        probe_size: Number of pixels to process in probe phase (must be ≤ tile_size²)
        sparse_indices: Pixel indices to process in sparse-continue path
        high_similarity_threshold: Similarity above this triggers early-stop (range: [0.80, 0.90])
        low_similarity_threshold: Similarity below this triggers full-continue (range: [0.55, 0.65])
    
    Validation:
        - high_similarity_threshold > low_similarity_threshold (strict inequality)
        - probe_size ≤ tile_size²
        - All sparse_indices < tile_size²
    """
    tile_size: int = 4
    probe_size: int = 4
    sparse_indices: List[int] = field(default_factory=lambda: [4, 7, 11, 15])
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.60
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.high_similarity_threshold <= self.low_similarity_threshold:
            raise ValueError(
                f"high_similarity_threshold ({self.high_similarity_threshold}) must be greater than "
                f"low_similarity_threshold ({self.low_similarity_threshold})"
            )
        
        max_pixels = self.tile_size ** 2
        if self.probe_size > max_pixels:
            raise ValueError(
                f"probe_size ({self.probe_size}) cannot exceed tile_size² ({max_pixels})"
            )
        
        for idx in self.sparse_indices:
            if idx >= max_pixels:
                raise ValueError(
                    f"sparse_indices contains {idx} which exceeds tile_size² ({max_pixels})"
                )


@dataclass
class GaussianSimilarityMetrics:
    """
    3D Gaussian similarity evaluation results
    
    All dispersion metrics are normalized to be comparable:
    - position_dispersion: max pairwise distance / scene_scale
    - covariance_dispersion: max pairwise Frobenius norm / avg_cov_norm
    - color_dispersion: max pairwise SH DC component distance
    - opacity_dispersion: max - min opacity
    - similarity_score: exp(-weighted_dispersion / 0.1) in [0, 1]
    
    Weighted dispersion = 0.4*pos + 0.3*cov + 0.15*color + 0.15*opacity
    """
    position_dispersion: float
    covariance_dispersion: float
    color_dispersion: float
    opacity_dispersion: float
    similarity_score: float
    
    def __post_init__(self):
        """Validate similarity score range"""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError(
                f"similarity_score ({self.similarity_score}) must be in [0, 1]"
            )


@dataclass
class TileProcessingResult:
    """
    Result from processing a single tile through 3-phase workflow
    
    Attributes:
        tile_id: (row, col) tile coordinates in grid
        path_taken: 'early_stop', 'sparse_continue', or 'full_continue'
        pixels_processed: Number of pixels that underwent depth search + Gaussian generation
        gaussians_generated: Number of Gaussians produced for this tile
        similarity_score: Similarity score from probe phase evaluation
        timing_ns: Phase timing breakdown {'probe': ns, 'evaluate': ns, 'decide': ns}
    """
    tile_id: Tuple[int, int]
    path_taken: str
    pixels_processed: int
    gaussians_generated: int
    similarity_score: float
    timing_ns: Dict[str, int]
    
    def __post_init__(self):
        """Validate path name"""
        valid_paths = {"early_stop", "sparse_continue", "full_continue"}
        if self.path_taken not in valid_paths:
            raise ValueError(
                f"path_taken must be one of {valid_paths}, got '{self.path_taken}'"
            )


@dataclass
class SAESProfilingResult:
    """
    Scene-level SAES profiling statistics
    
    Aggregates results from all tiles to provide overall performance metrics:
    - Path distribution (how many tiles took each path)
    - Computation savings (pixels that skipped depth search)
    - Overhead (time spent in similarity evaluation + decision)
    - Net speedup (accounting for overhead)
    
    Attributes:
        scene_name: Identifier for the scene
        total_tiles: Total number of tiles processed
        early_stop_tiles: Tiles that took early-stop path
        sparse_continue_tiles: Tiles that took sparse-continue path
        full_continue_tiles: Tiles that took full-continue path
        total_pixels_saved: Pixels that skipped depth search
        computation_saving_ratio: Fraction of pixels saved (in [0, 1])
        overhead_ns: Total SAES overhead (evaluate + decide phases)
        net_speedup: Overall speedup considering overhead
        tile_results: List of per-tile results
    """
    scene_name: str
    total_tiles: int
    early_stop_tiles: int
    sparse_continue_tiles: int
    full_continue_tiles: int
    total_pixels_saved: int
    computation_saving_ratio: float
    overhead_ns: int
    net_speedup: float
    tile_results: List[TileProcessingResult] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate consistency"""
        # Path counts should sum to total
        path_sum = self.early_stop_tiles + self.sparse_continue_tiles + self.full_continue_tiles
        if path_sum != self.total_tiles:
            raise ValueError(
                f"Path counts ({path_sum}) don't match total_tiles ({self.total_tiles})"
            )
        
        # Saving ratio should be in [0, 1]
        if not (0.0 <= self.computation_saving_ratio <= 1.0):
            raise ValueError(
                f"computation_saving_ratio ({self.computation_saving_ratio}) must be in [0, 1]"
            )
    
    def summary_str(self) -> str:
        """Generate human-readable summary"""
        return (
            f"SAES Profiling for {self.scene_name}:\n"
            f"  Total tiles: {self.total_tiles}\n"
            f"  Early-stop: {self.early_stop_tiles} ({self.early_stop_tiles/self.total_tiles:.1%})\n"
            f"  Sparse: {self.sparse_continue_tiles} ({self.sparse_continue_tiles/self.total_tiles:.1%})\n"
            f"  Full: {self.full_continue_tiles} ({self.full_continue_tiles/self.total_tiles:.1%})\n"
            f"  Pixels saved: {self.total_pixels_saved}\n"
            f"  Computation saving: {self.computation_saving_ratio:.2%}\n"
            f"  SAES overhead: {self.overhead_ns / 1e6:.2f} ms\n"
            f"  Net speedup: {self.net_speedup:.2f}×"
        )
