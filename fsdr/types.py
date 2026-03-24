"""
FSDR Data Types

Core data structures for the FSDR module.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import torch


@dataclass
class FSDRConfig:
    """
    FSDR configuration parameters.
    
    Attributes:
        cache_size: Number of cache entries (default: 128)
        lsh_dim: LSH signature dimension in bits (default: 16)
        feature_dim: Input feature vector dimension (default: 128)
        hamming_threshold: Maximum Hamming distance for cache hit (default: 4)
        high_confidence_threshold: Peak prob threshold for direct reuse (default: 0.8)
        medium_confidence_threshold: Peak prob threshold for interpolation (default: 0.5)
        hamming_direct_reuse: Max Hamming for direct reuse (default: 2)
        hamming_interpolate: Max Hamming for interpolation (default: 3)
        max_verify_radius: Maximum radius for light verification (default: 3)
        depth_update_alpha: EMA coefficient for depth updates (default: 0.1)
        seed: Random seed for LSH projection matrix (default: None)
    
    Hardware Mapping:
        - cache_size × 70 bits = SRAM requirement
        - lsh_dim × feature_dim = ROM for projection matrix
    """
    cache_size: int = 128
    lsh_dim: int = 16
    feature_dim: int = 128
    hamming_threshold: int = 4
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
    hamming_direct_reuse: int = 2
    hamming_interpolate: int = 3
    max_verify_radius: int = 3
    depth_update_alpha: float = 0.1
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {self.cache_size}")
        if self.lsh_dim <= 0 or self.lsh_dim > 32:
            raise ValueError(f"lsh_dim must be in [1, 32], got {self.lsh_dim}")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.hamming_threshold < 0:
            raise ValueError(f"hamming_threshold must be non-negative, got {self.hamming_threshold}")
        if not 0 <= self.high_confidence_threshold <= 1:
            raise ValueError(f"high_confidence_threshold must be in [0, 1]")
        if not 0 <= self.medium_confidence_threshold <= 1:
            raise ValueError(f"medium_confidence_threshold must be in [0, 1]")
        if self.hamming_direct_reuse > self.hamming_interpolate:
            raise ValueError("hamming_direct_reuse must be <= hamming_interpolate")


@dataclass
class CacheEntry:
    """
    Cache entry structure (70 bits total in hardware).
    
    Attributes:
        signature: 16-bit LSH signature
        position: (u, v) pixel coordinates (8 bits each)
        best_depth: Optimal depth value (FP16)
        best_idx: Index of best depth candidate (5 bits, 0-31)
        peak_prob: Peak probability (8 bits, quantized to [0, 1])
        second_offset: Offset to second-best index (signed 5 bits)
        spread: Distribution spread (8 bits, quantized)
        valid: Entry validity flag (1 bit)
        last_access: Access timestamp for LRU (not stored in hardware)
    
    Hardware Layout:
        [69:54] signature (16 bits)
        [53:46] position_u (8 bits)
        [45:38] position_v (8 bits)
        [37:22] best_depth (16 bits, FP16)
        [21:17] best_idx (5 bits)
        [16:9]  peak_prob (8 bits)
        [8:4]   second_offset (5 bits, signed)
        [3:1]   spread (3 bits for quick access, full 8 bits in extended)
        [0]     valid (1 bit)
    """
    signature: int
    position: Tuple[int, int]
    best_depth: float
    best_idx: int
    peak_prob: float
    second_offset: int
    spread: float
    valid: bool = True
    last_access: int = 0
    
    def __post_init__(self):
        """Validate entry fields."""
        if not 0 <= self.signature < 2**16:
            raise ValueError(f"signature must be 16-bit, got {self.signature}")
        if not 0 <= self.best_idx < 32:
            raise ValueError(f"best_idx must be in [0, 31], got {self.best_idx}")
        if not -16 <= self.second_offset < 16:
            raise ValueError(f"second_offset must be in [-16, 15], got {self.second_offset}")
        if not 0 <= self.peak_prob <= 1:
            raise ValueError(f"peak_prob must be in [0, 1], got {self.peak_prob}")
        if not 0 <= self.spread <= 1:
            raise ValueError(f"spread must be in [0, 1], got {self.spread}")
    
    def to_bits(self) -> int:
        """
        Serialize entry to 70-bit integer (for hardware simulation).
        
        Returns:
            70-bit integer representation
        """
        bits = 0
        bits |= (self.signature & 0xFFFF) << 54
        bits |= (self.position[0] & 0xFF) << 46
        bits |= (self.position[1] & 0xFF) << 38
        # best_depth: convert to FP16 bits
        depth_bits = int(self.best_depth * 1000) & 0xFFFF  # Simplified quantization
        bits |= depth_bits << 22
        bits |= (self.best_idx & 0x1F) << 17
        peak_quant = int(self.peak_prob * 255) & 0xFF
        bits |= peak_quant << 9
        offset_bits = (self.second_offset + 16) & 0x1F  # Bias to unsigned
        bits |= offset_bits << 4
        spread_quant = int(self.spread * 7) & 0x07  # 3 bits
        bits |= spread_quant << 1
        bits |= 1 if self.valid else 0
        return bits
    
    @classmethod
    def from_bits(cls, bits: int) -> 'CacheEntry':
        """
        Deserialize entry from 70-bit integer.
        
        Args:
            bits: 70-bit integer representation
        
        Returns:
            CacheEntry instance
        """
        signature = (bits >> 54) & 0xFFFF
        pos_u = (bits >> 46) & 0xFF
        pos_v = (bits >> 38) & 0xFF
        depth_bits = (bits >> 22) & 0xFFFF
        best_depth = depth_bits / 1000.0
        best_idx = (bits >> 17) & 0x1F
        peak_quant = (bits >> 9) & 0xFF
        peak_prob = peak_quant / 255.0
        offset_bits = (bits >> 4) & 0x1F
        second_offset = offset_bits - 16  # Remove bias
        spread_quant = (bits >> 1) & 0x07
        spread = spread_quant / 7.0
        valid = bool(bits & 0x01)
        
        return cls(
            signature=signature,
            position=(pos_u, pos_v),
            best_depth=best_depth,
            best_idx=best_idx,
            peak_prob=peak_prob,
            second_offset=second_offset,
            spread=spread,
            valid=valid,
        )


@dataclass
class FSDRResult:
    """
    Result of FSDR processing for a single pixel.
    
    Attributes:
        depth: Estimated depth value
        source: Processing path ('direct_reuse', 'interpolation', 
                'light_verify', 'full_search')
        cache_hit: Whether cache was hit
        hamming_distance: Hamming distance if hit, None if miss
        num_searches: Number of depth candidates searched
        timing_ns: Timing breakdown in nanoseconds
    """
    depth: float
    source: str
    cache_hit: bool
    hamming_distance: Optional[int] = None
    num_searches: int = 0
    timing_ns: Dict[str, int] = field(default_factory=dict)
    
    @property
    def memory_saved(self) -> int:
        """
        Number of depth candidate accesses saved vs full search.
        
        Assumes full search requires 32 accesses.
        """
        return 32 - self.num_searches


@dataclass
class FSDRProfilingResult:
    """
    Aggregated profiling results for FSDR processing.
    
    Attributes:
        total_pixels: Total pixels processed
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        path_counts: Count per processing path
        total_searches: Total depth candidate accesses
        baseline_searches: Searches without FSDR (all full search)
        total_time_ns: Total processing time
        path_time_ns: Time per processing path
    """
    total_pixels: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    path_counts: Dict[str, int] = field(default_factory=lambda: {
        'direct_reuse': 0,
        'interpolation': 0,
        'light_verify': 0,
        'full_search': 0,
    })
    total_searches: int = 0
    baseline_searches: int = 0
    total_time_ns: int = 0
    path_time_ns: Dict[str, int] = field(default_factory=lambda: {
        'direct_reuse': 0,
        'interpolation': 0,
        'light_verify': 0,
        'full_search': 0,
    })
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def memory_reduction(self) -> float:
        """Memory access reduction ratio."""
        if self.baseline_searches == 0:
            return 0.0
        return 1.0 - (self.total_searches / self.baseline_searches)
    
    @property
    def direct_reuse_rate(self) -> float:
        """Fraction of pixels using direct reuse."""
        return self.path_counts['direct_reuse'] / self.total_pixels if self.total_pixels > 0 else 0.0
    
    @property
    def avg_time_per_pixel_ns(self) -> float:
        """Average processing time per pixel."""
        return self.total_time_ns / self.total_pixels if self.total_pixels > 0 else 0.0
