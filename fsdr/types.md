# FSDR Types

Data structures for Feature-Similarity Depth Reuse.

## External Interface

### FSDRConfig

Configuration dataclass for FSDR processor.

```python
@dataclass
class FSDRConfig:
    feature_dim: int = 128
    num_hash_bits: int = 16
    cache_size: int = 256
    hamming_threshold: int = 4
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
```

**Fields:**
- `feature_dim`: Feature vector dimension (must match model)
- `num_hash_bits`: LSH signature bit width
- `cache_size`: Maximum cache entries
- `hamming_threshold`: Max Hamming distance for cache hit
- `high_confidence_threshold`: Peak probability for direct reuse
- `medium_confidence_threshold`: Peak probability for interpolation

### CacheEntry

70-bit cache entry for depth reuse.

```python
@dataclass
class CacheEntry:
    signature: int          # 16-bit LSH signature
    best_depth_idx: int     # 5-bit depth index
    best_depth: float       # Actual depth value
    second_depth_idx: int   # 5-bit second-best index
    peak_prob: float        # 8-bit quantized probability
    spread: float           # 4-bit quantized spread
    confidence: float       # Confidence score
```

**Methods:**
- `to_bits() -> int`: Serialize to 70-bit integer
- `from_bits(bits: int, depth_candidates: Tensor) -> CacheEntry`: Deserialize

### FSDRResult

Result from FSDR pixel processing.

```python
@dataclass
class FSDRResult:
    depth: float
    source: str             # 'direct_reuse', 'interpolation', 'light_verify', 'full_search'
    cache_hit: bool
    num_searches: int       # Memory accesses performed
```

### FSDRProfilingResult

Aggregated profiling statistics.

```python
@dataclass
class FSDRProfilingResult:
    total_pixels: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    path_distribution: Dict[str, int]
    total_searches: int
    baseline_searches: int
    memory_reduction: float
```

## Hardware Mapping

- CacheEntry: 70 bits packed for SRAM storage
- Depth quantization: 16-bit fixed-point (×1000)
- Probability quantization: 8-bit (0-255 → 0.0-1.0)
- Spread quantization: 4-bit index into lookup table
