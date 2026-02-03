# Types Interface Documentation

## External Interface

### Gaussian Class

**Purpose**: Represents a single 3D Gaussian primitive (decoupled from transplat).

```python
@dataclass
class Gaussian:
    mean: Tensor        # [3] - 3D position (x, y, z)
    cov: Tensor         # [3, 3] - Covariance matrix (must be positive definite)
    opacity: float      # Transparency in [0, 1]
    harmonics: Tensor   # [C, D_sh] - Spherical harmonics coefficients
```

**Methods**:
- `to(device)` → Gaussian: Move to CUDA/CPU device
- `clone()` → Gaussian: Deep copy

**Validation**: None at construction (for performance). Covariance should be symmetric positive definite.

**Usage**:
```python
g = Gaussian(
    mean=torch.tensor([1.0, 2.0, 3.0]),
    cov=torch.eye(3) * 0.5,
    opacity=0.8,
    harmonics=torch.randn(3, 16)
)
g_gpu = g.to('cuda')
g_copy = g.clone()
```

---

### TileConfig Class

**Purpose**: SAES configuration with automatic validation.

```python
@dataclass
class TileConfig:
    tile_size: int = 4
    probe_size: int = 4
    sparse_indices: List[int] = field(default_factory=lambda: [4, 7, 11, 15])
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.60
```

**Parameters**:
- `tile_size`: Pixels per tile edge (e.g., 4 → 4×4 = 16 pixels)
  - Range: [2, 16]
  - Hardware: Configurable register
  
- `probe_size`: Pixels to process in probe phase
  - Range: [2, tile_size²]
  - Must be ≤ tile_size²
  
- `sparse_indices`: Indices for sparse-continue path
  - Default: [4, 7, 11, 15] (4 corners)
  - All indices must be < tile_size²
  
- `high_similarity_threshold`: Early-stop trigger
  - Range: [0.0, 1.0]
  - Must be > low_similarity_threshold
  - Recommended: [0.80, 0.90]
  
- `low_similarity_threshold`: Full-continue trigger
  - Range: [0.0, 1.0]
  - Must be < high_similarity_threshold
  - Recommended: [0.55, 0.65]

**Validation**: Performed in `__post_init__`:
1. high_threshold > low_threshold (strict inequality)
2. probe_size ≤ tile_size²
3. All sparse_indices < tile_size²

**Raises**: `ValueError` if validation fails

**Usage**:
```python
# Default config
config = TileConfig()

# Custom config
config = TileConfig(
    tile_size=8,
    high_similarity_threshold=0.90,
)

# Invalid config (raises ValueError)
config = TileConfig(
    high_similarity_threshold=0.60,
    low_similarity_threshold=0.85,  # ERROR: high < low
)
```

---

### GaussianSimilarityMetrics Class

**Purpose**: Results from 3D Gaussian similarity evaluation.

```python
@dataclass
class GaussianSimilarityMetrics:
    position_dispersion: float
    covariance_dispersion: float
    color_dispersion: float
    opacity_dispersion: float
    similarity_score: float  # in [0, 1]
```

**Fields**:
- `position_dispersion`: Normalized spatial spread
  - Formula: max_pairwise_dist(means) / scene_scale
  - Higher = more spread out
  
- `covariance_dispersion`: Normalized shape variation
  - Formula: max_pairwise_frobenius_norm(covs) / avg_cov_norm
  - Higher = more shape variation
  
- `color_dispersion`: SH DC component variation
  - Formula: max_pairwise_dist(sh_dc[:3])
  - Higher = more color difference
  
- `opacity_dispersion`: Transparency variation
  - Formula: max(opacities) - min(opacities)
  - Range: [0, 1]
  
- `similarity_score`: Weighted aggregation
  - Formula: exp(-(0.4×pos + 0.3×cov + 0.15×color + 0.15×opacity) / 0.1)
  - Range: [0, 1] where 1.0 = perfect similarity

**Validation**: `similarity_score` must be in [0, 1] (checked in `__post_init__`)

**Usage**:
```python
metrics = evaluator.evaluate(probe_gaussians)
if metrics.similarity_score > 0.85:
    print("High similarity detected")
print(f"Position dispersion: {metrics.position_dispersion:.3f}")
```

---

### TileProcessingResult Class

**Purpose**: Records processing result for a single tile.

```python
@dataclass
class TileProcessingResult:
    tile_id: Tuple[int, int]        # (row, col) in tile grid
    path_taken: str                 # 'early_stop', 'sparse_continue', 'full_continue'
    pixels_processed: int           # Number of pixels that underwent depth search
    gaussians_generated: int        # Number of Gaussians produced
    similarity_score: float         # From probe evaluation
    timing_ns: Dict[str, int]       # Phase timing {'probe': ns, 'evaluate': ns, ...}
```

**Validation**: `path_taken` must be one of valid path names (checked in `__post_init__`)

**Usage**:
```python
result = TileProcessingResult(
    tile_id=(2, 5),
    path_taken="early_stop",
    pixels_processed=4,
    gaussians_generated=1,
    similarity_score=0.92,
    timing_ns={'probe': 100000, 'evaluate': 16000, 'decide': 500}
)
```

---

### SAESProfilingResult Class

**Purpose**: Scene-level aggregated performance statistics.

```python
@dataclass
class SAESProfilingResult:
    scene_name: str
    total_tiles: int
    early_stop_tiles: int
    sparse_continue_tiles: int
    full_continue_tiles: int
    total_pixels_saved: int
    computation_saving_ratio: float  # in [0, 1]
    overhead_ns: int
    net_speedup: float
    tile_results: List[TileProcessingResult] = field(default_factory=list)
```

**Validation**: 
- Path counts must sum to total_tiles
- computation_saving_ratio must be in [0, 1]

**Methods**:
- `summary_str()` → str: Human-readable summary

**Usage**:
```python
profiling = processor.get_scene_summary()
print(profiling.summary_str())

# Detailed analysis
for tile_result in profiling.tile_results:
    if tile_result.path_taken == "early_stop":
        print(f"Tile {tile_result.tile_id}: Early-stop, similarity={tile_result.similarity_score:.3f}")
```

---

## Internal Helpers

None. This module contains only data structures (dataclasses). No helper functions.

---

## Hardware Mapping

### Gaussian → Hardware Register Layout

```
Offset | Size | Field       | Format
-------|------|-------------|--------
0x00   | 12B  | mean        | 3× float32 (or 3× int16 fixed-point)
0x0C   | 36B  | cov         | 9× float32 (or 9× int16 fixed-point)
0x30   | 4B   | opacity     | float32 (or uint8 fixed-point)
0x34   | 76B  | harmonics   | 3×16× float16 (or int8 fixed-point)
-------|------|-------------|--------
Total  | 128B per Gaussian
```

### TileConfig → Configuration Registers

```
Register | Bits | Field                  | Default
---------|------|------------------------|--------
CFG0     | 8    | tile_size              | 4
CFG1     | 8    | probe_size             | 4
CFG2     | 8    | high_threshold         | 217 (0.85×255)
CFG3     | 8    | low_threshold          | 153 (0.60×255)
CFG4-7   | 4×16 | sparse_indices[0..15]  | [4,7,11,15,...]
```

---

## Design Rationale

**Why dataclasses?**
- Explicit attribute declaration (vs. dict or dynamic attributes)
- Type safety and IDE autocomplete
- Auto-generated `__init__`, `__repr__`, `__eq__`
- Validation via `__post_init__`

**Why separate Gaussian from transplat's Gaussians?**
- Decoupling: SAES doesn't depend on transplat installation
- Simplicity: Only essential attributes (mean, cov, opacity, harmonics)
- Portability: Works with any model that can convert to this format

**Why validation in __post_init__?**
- Fail fast: Catch invalid configs at construction, not during processing
- Clear error messages: User knows immediately what's wrong
- Zero runtime overhead: Validation once at initialization

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03
