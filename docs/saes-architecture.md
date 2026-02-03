# SAES Architecture: Scene-Adaptive Early-Stopping Dataflow

## 1. Overview

SAES (Scene-Adaptive Early-Stopping Dataflow) is a hardware-software co-design component that reduces redundant computation in generalizable 3D Gaussian Splatting (3DGS) encoders by exploiting 3D geometric continuity. This document describes the software simulator architecture.

### 1.1 Design Goals

1. **Feedback-driven early stopping**: Detect 3D Gaussian similarity during generation to terminate redundant computation
2. **Cross-stage feedback**: Establish control path from Gaussian generation back to depth prediction
3. **Adaptive granularity**: Dynamically adjust computation based on 3D geometric local continuity
4. **Hardware-realizable**: Tile-based processing with pipeline-friendly design
5. **Transplat-decoupled**: Core SAES logic is independent of transplat implementation

### 1.2 Problem Statement

Traditional generalizable 3DGS encoders use a two-stage serial architecture:
```
Feature Extraction → Depth Prediction (all pixels) → Gaussian Generation (all pixels)
```

**Problems:**
- Depth prediction cannot anticipate which pixels will generate similar Gaussians
- Even when Gaussians are highly similar (smooth 3D surfaces), all computations must complete
- Lacks cross-stage feedback mechanism

### 1.3 SAES Solution

SAES introduces feedback-driven tile processing:
```
For each tile:
  Phase 1 (Probe):    Process first 4 pixels → generate probe Gaussians
  Phase 2 (Evaluate): Compute 3D similarity among probe Gaussians
  Phase 3 (Decide):   Based on similarity score, choose path:
                      - High (>0.85):   Early stop, use enlarged probes
                      - Medium (0.6-0.85): Sparse continue, process 4 more
                      - Low (<0.6):    Full continue, process all 16
```

**Key Innovation:** Decision is based on **already-generated Gaussians' 3D properties**, not on 2D features.

## 2. System Architecture

### 2.1 Component Hierarchy

```
TileProcessor (main engine)
├── GaussianSimilarityEvaluator (Phase 2)
│   ├── Position dispersion calculator
│   ├── Covariance dispersion calculator
│   ├── Color dispersion calculator
│   ├── Opacity dispersion calculator
│   └── Weighted similarity aggregator
├── DecisionController (Phase 3)
│   ├── Threshold-based path selector
│   └── Remaining pixel index generator
├── GaussianMerger (early-stop path)
│   ├── Covariance enlargement
│   └── Gaussian averaging
└── SAESProfiler (performance tracking)
    ├── Per-tile timing recorder
    ├── Computation savings calculator
    └── Statistics aggregator
```

### 2.2 Data Flow

```
Input: Feature map [B, V, C, H, W]

TileProcessor.process_scene():
  ├─► Compute tile grid (e.g., 16×16 tiles for 64×64 feature map)
  │
  └─► For each tile (4×4 pixels):
      │
      ├─► Phase 1: Probe
      │   ├─ Call depth_predictor_fn(tile[:4])  → probe_depths
      │   └─ Call gaussian_adapter_fn(probe_depths) → probe_gaussians
      │
      ├─► Phase 2: Evaluate
      │   └─ similarity_evaluator.evaluate(probe_gaussians) → similarity_score
      │
      ├─► Phase 3: Decide
      │   ├─ decision_controller.decide_path(similarity_score) → path
      │   │
      │   ├─► [Early Stop Path]
      │   │   └─ gaussian_merger.enlarge_and_merge(probe_gaussians) → tile_gaussians
      │   │
      │   ├─► [Sparse Continue Path]
      │   │   ├─ remaining_indices = [4, 7, 11, 15]
      │   │   ├─ depth_predictor_fn(tile[remaining_indices]) → remaining_depths
      │   │   ├─ gaussian_adapter_fn(remaining_depths) → remaining_gaussians
      │   │   └─ tile_gaussians = concat(probe, remaining)
      │   │
      │   └─► [Full Continue Path]
      │       ├─ remaining_indices = [4, 5, ..., 15]
      │       ├─ depth_predictor_fn(tile[remaining_indices]) → remaining_depths
      │       ├─ gaussian_adapter_fn(remaining_depths) → remaining_gaussians
      │       └─ tile_gaussians = concat(probe, remaining)
      │
      └─► profiler.finish_tile(tile_result)

Output: Gaussians [B, N, ...], SAESProfilingResult
```

## 3. Core Components

### 3.1 TileProcessor

**Purpose**: Main orchestrator coordinating tile-based processing.

**Key Methods:**
- `process_scene(features, depth_predictor_fn, gaussian_adapter_fn, context)` - Entry point
- `process_tile(tile_coords, features, ...)` - Single tile 3-phase workflow
- `_extract_tile_features(features, tile_coords)` - Slice feature tensor for tile
- `_compute_tile_grid(H, W, tile_size)` - Determine tile layout

**Design Decisions:**
- **Tile size = 4×4 pixels**: Balances probe overhead vs. early-stop benefit
- **Probe size = 4 pixels**: Minimal sample for reliable similarity estimation
- **Callbacks for depth/Gaussian generation**: Maintains decoupling from transplat

**Interface:**
```python
def process_scene(
    self,
    features: Tensor,              # [B, V, C, H, W] - from backbone
    depth_predictor_fn: Callable,  # fn(features, indices) -> depths
    gaussian_adapter_fn: Callable, # fn(depths, ...) -> Gaussians
    context: Dict,                 # Camera parameters
) -> Tuple[Gaussians, SAESProfilingResult]:
    """
    Process entire scene with tile-based SAES.
    
    Returns:
        - Gaussians: Final Gaussians (some from early-stop, some from full)
        - SAESProfilingResult: Performance metrics
    """
```

### 3.2 GaussianSimilarityEvaluator

**Purpose**: Compute 3D similarity among probe Gaussians.

**Similarity Metrics** (from Design2 Section 5.1):

1. **Position Dispersion**: 
   ```
   pos_disp = max_pairwise_dist(positions) / scene_scale
   ```
   - Measures spatial spread of Gaussians
   - Normalized by scene scale (auto-estimated from position std)

2. **Covariance Dispersion**:
   ```
   cov_disp = max_pairwise_dist(covariances) / avg_cov_norm
   ```
   - Measures shape variation (using Frobenius norm)
   - Normalized by average covariance magnitude

3. **Color Dispersion**:
   ```
   color_disp = max_pairwise_dist(sh_dc_components)
   ```
   - Measures color variation (SH DC coefficients)
   - RGB color encoded in first 3 SH coefficients

4. **Opacity Dispersion**:
   ```
   opacity_disp = max(opacities) - min(opacities)
   ```
   - Measures transparency variation

**Weighted Aggregation** (Design2 Eq. in Section 5.1):
```python
dispersion = (
    0.4 * pos_disp +      # Position most important
    0.3 * cov_disp +      # Shape second
    0.15 * color_disp +   # Color tertiary
    0.15 * opacity_disp   # Opacity tertiary
)

similarity_score = exp(-dispersion / 0.1)  # Maps to [0, 1]
```

**Design Rationale:**
- Position weighted highest (0.4) because spatial continuity is strongest signal
- Exponential mapping ensures smooth degradation with increasing dispersion
- Temperature parameter (0.1) controls sensitivity

**Interface:**
```python
def evaluate(
    self,
    gaussians: List[Gaussian],  # Probe Gaussians (typically 4)
) -> GaussianSimilarityMetrics:
    """
    Compute 3D similarity metrics.
    
    Returns:
        GaussianSimilarityMetrics with:
        - position_dispersion: float
        - covariance_dispersion: float
        - color_dispersion: float
        - opacity_dispersion: float
        - similarity_score: float in [0, 1]
    """
```

### 3.3 DecisionController

**Purpose**: Select processing path based on similarity score.

**Decision Logic** (Design2 Section 4.2):
```python
if similarity_score > 0.85:
    return "early_stop"     # High similarity → 75% computation saved
elif similarity_score > 0.60:
    return "sparse_continue" # Medium similarity → 50% computation saved
else:
    return "full_continue"   # Low similarity → 0% saved
```

**Threshold Rationale:**
- `0.85`: Empirically, sim > 0.85 indicates high geometric continuity (smooth walls, floors)
- `0.60`: Mid-range captures moderate variation (textured surfaces)
- Below 0.60: Complex geometry (edges, occlusions) requires full processing

**Interface:**
```python
def decide_path(self, similarity_score: float) -> str:
    """
    Returns: 'early_stop', 'sparse_continue', or 'full_continue'
    """

def get_remaining_indices(self, path: str, tile_size: int, probe_size: int) -> List[int]:
    """
    Returns pixel indices to process based on path.
    
    Examples:
    - early_stop: []
    - sparse_continue: [4, 7, 11, 15]  # 4 corners
    - full_continue: [4, 5, 6, ..., 15]  # all remaining
    """
```

### 3.4 GaussianMerger

**Purpose**: Enlarge probe Gaussians to cover entire tile for early-stop path.

**Enlargement Strategy** (Design2 Section 5.1):

1. **Position**: Weighted average of probe positions
   ```python
   merged_position = mean([g.mean for g in probes])
   ```

2. **Covariance**: Scale up to cover tile area
   ```python
   tile_area = tile_size² * pixel_size²
   scale_factor = sqrt(tile_area / probe_area)
   merged_cov = avg([g.cov for g in probes]) * scale_factor
   ```

3. **Color**: Average SH coefficients
   ```python
   merged_sh = mean([g.harmonics for g in probes])
   ```

4. **Opacity**: Average opacity
   ```python
   merged_opacity = mean([g.opacity for g in probes])
   ```

**Design Rationale:**
- Covariance enlargement ensures coverage without gaps
- Averaging maintains appearance consistency
- Trade-off: Slight quality loss for large computation savings

**Interface:**
```python
def enlarge_and_merge(
    self,
    probe_gaussians: Gaussians,
    tile_coverage: Tuple[int, int, int, int],  # (row_start, row_end, col_start, col_end)
) -> Gaussians:
    """
    Enlarge probe Gaussians to represent entire tile.
    
    Returns:
        Gaussians: Merged/enlarged Gaussians covering tile area
    """
```

### 3.5 SAESProfiler

**Purpose**: Track performance metrics for analysis.

**Metrics Collected:**
- Per-tile timing (probe/evaluate/decide phases)
- Path distribution (early-stop / sparse / full counts)
- Pixels processed vs. skipped
- Computation saving ratio
- SAES overhead

**Key Calculations:**
```python
pixels_saved = sum(pixels_skipped per tile)
total_pixels = num_tiles * tile_size²
computation_saving_ratio = pixels_saved / total_pixels

saes_overhead = sum(evaluate_time + decide_time per tile)
net_speedup = (baseline_time - saved_time) / (baseline_time + saes_overhead)
```

**Interface:**
```python
def start_tile(self, tile_id: Tuple[int, int]) -> None:
    """Begin timing for a tile"""

def record_phase(self, phase: str, duration_ns: int) -> None:
    """Record phase timing (probe/evaluate/decide)"""

def finish_tile(self, result: TileProcessingResult) -> None:
    """Complete tile and store result"""

def get_scene_summary(self) -> SAESProfilingResult:
    """Aggregate all tiles into scene-level statistics"""
```

## 4. Data Structures

### 4.1 Core Types

```python
@dataclass
class TileConfig:
    """SAES configuration"""
    tile_size: int = 4  # Pixels per tile edge
    probe_size: int = 4  # Pixels in probe phase
    sparse_indices: List[int] = field(default_factory=lambda: [4, 7, 11, 15])
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.60

@dataclass
class GaussianSimilarityMetrics:
    """3D Gaussian similarity evaluation results"""
    position_dispersion: float
    covariance_dispersion: float
    color_dispersion: float
    opacity_dispersion: float
    similarity_score: float  # in [0, 1]

@dataclass
class TileProcessingResult:
    """Result from processing a single tile"""
    tile_id: Tuple[int, int]  # (row, col)
    path_taken: str  # 'early_stop', 'sparse_continue', 'full_continue'
    pixels_processed: int
    gaussians_generated: int
    similarity_score: float
    timing_ns: Dict[str, int]  # {'probe': ..., 'evaluate': ..., 'decide': ...}

@dataclass
class SAESProfilingResult:
    """Scene-level SAES statistics"""
    scene_name: str
    total_tiles: int
    early_stop_tiles: int
    sparse_continue_tiles: int
    full_continue_tiles: int
    total_pixels_saved: int
    computation_saving_ratio: float
    overhead_ns: int
    net_speedup: float
```

### 4.2 Gaussian Representation

SAES operates on Gaussians with standard 3DGS representation:
```python
@dataclass
class Gaussian:
    """Single 3D Gaussian primitive"""
    mean: Tensor  # [3] - 3D position
    cov: Tensor   # [3, 3] - covariance matrix
    opacity: float  # [0, 1] - transparency
    harmonics: Tensor  # [C, D_sh] - spherical harmonics (color)
```

## 5. Algorithm Details

### 5.1 Tile Grid Computation

```python
def _compute_tile_grid(H: int, W: int, tile_size: int) -> Tuple[int, int]:
    """
    Compute number of tiles.
    
    Args:
        H, W: Feature map height/width
        tile_size: Pixels per tile edge
    
    Returns:
        (n_tiles_h, n_tiles_w)
    
    Example:
        H=64, W=64, tile_size=4 → (16, 16) = 256 tiles
    """
    n_tiles_h = (H + tile_size - 1) // tile_size  # Ceiling division
    n_tiles_w = (W + tile_size - 1) // tile_size
    return n_tiles_h, n_tiles_w
```

### 5.2 Pairwise Distance Computation

Used in similarity evaluator for efficient distance calculations:
```python
def _pairwise_distances(vectors: Tensor) -> Tensor:
    """
    Compute all pairwise L2 distances.
    
    Args:
        vectors: [N, D] tensor
    
    Returns:
        distances: [N, N] tensor where distances[i,j] = ||vectors[i] - vectors[j]||
    
    Implementation:
        ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
    """
    # Efficient vectorized computation
    dots = vectors @ vectors.T  # [N, N]
    norms = torch.diag(dots)  # [N]
    distances_sq = norms[:, None] + norms[None, :] - 2 * dots
    return torch.sqrt(torch.clamp(distances_sq, min=0))
```

### 5.3 Scene Scale Estimation

```python
def _estimate_scene_scale(positions: Tensor) -> float:
    """
    Auto-estimate scene scale from Gaussian positions.
    
    Args:
        positions: [N, 3] - all Gaussian positions in scene
    
    Returns:
        scene_scale: float - standard deviation of positions
    
    Rationale:
        Scene scale normalizes position dispersion to be comparable
        across different scene sizes (small objects vs. large rooms)
    """
    return positions.std().item()
```

## 6. Integration Points

### 6.1 Decoupling from Transplat

SAES core logic is **independent** of transplat implementation:

**Abstraction via Callbacks:**
```python
# SAES doesn't know about transplat's DepthPredictorTrans or GaussianAdapter
# Instead, it receives callable functions

depth_predictor_fn: Callable[[Tensor, List[int]], Tensor]
# Input: (features, pixel_indices) → Output: depths for those pixels

gaussian_adapter_fn: Callable[[Tensor, Dict], Gaussians]
# Input: (depths, context) → Output: Gaussians
```

**Benefits:**
- SAES can work with any depth predictor / Gaussian generator
- Testing doesn't require transplat installation
- Future models can use SAES without modification

### 6.2 Integration with Transplat (for evaluation)

Integration happens in `transplat/src/model/encoder/encoder_trans.py`:

```python
# Original code:
depths, densities, raw_gaussians = self.depth_predictor(in_feats, ...)
gaussians = self.gaussian_adapter(depths, ...)

# With SAES:
if enable_saes:
    # Wrap depth_predictor and gaussian_adapter as callbacks
    def depth_fn(feats, indices):
        return self.depth_predictor(feats, indices=indices, ...)
    
    def gaussian_fn(depths, ctx):
        return self.gaussian_adapter(depths, ctx, ...)
    
    # SAES takes over
    gaussians, saes_result = saes_processor.process_scene(
        in_feats, depth_fn, gaussian_fn, context
    )
else:
    # Original path
    depths, densities, raw_gaussians = self.depth_predictor(in_feats, ...)
    gaussians = self.gaussian_adapter(depths, ...)
```

## 7. Performance Characteristics

### 7.1 Expected Savings

**Computation (from Design2 Section 10.1):**
- Early-stop tiles: 75% pixels skipped
- Sparse tiles: 50% pixels skipped
- Expected distribution: ~30% early / ~40% sparse / ~30% full
- **Overall saving: ~42.5%**

**Overhead:**
- Similarity evaluation: ~16 cycles per tile (Design2 Section 7.2)
- Decision logic: <1 cycle
- Target: <10% of baseline time

**Net Speedup:**
- Computation savings: 42.5%
- Overhead: ~8%
- **Net speedup: ~1.35×**

### 7.2 Quality Impact

**Expected degradation (from Design2 Section 10.3):**
- PSNR: -0.15 dB average (max -0.2 dB)
- SSIM: -0.007 average (max -0.01)

**Rationale:**
- Early-stop only on high-similarity regions (already smooth)
- Gaussian enlargement preserves appearance
- Quality loss concentrated in already-smooth areas (visually imperceptible)

## 8. Testing Strategy

### 8.1 Unit Tests

Test each component in isolation with synthetic data:
- **Similarity evaluator**: Identical Gaussians → score ≈ 1.0
- **Decision controller**: Threshold boundary cases
- **Gaussian merger**: Covariance scale verification
- **Tile processor**: Mock depth_predictor_fn

### 8.2 Integration Tests

Test end-to-end with real RE10K scenes:
- Compare SAES vs. baseline output
- Verify PSNR/SSIM within acceptable degradation
- Validate computation savings in expected range
- Check path distribution aligns with expectations

## 9. Future Enhancements

### 9.1 Adaptive Thresholds

Current thresholds (0.85, 0.60) are fixed. Future work:
- Learn thresholds per-scene from statistics
- Adjust based on desired quality-speed tradeoff

### 9.2 FSDR Integration

Combine with FSDR (Feature-Similarity Depth Reuse):
- FSDR optimizes Phase 1 (probe) by reducing depth search memory access
- SAES optimizes Phases 2-3 (evaluate-decide) by skipping computation
- Combined benefit: ~60% memory + ~40% computation savings

### 9.3 Variable Tile Sizes

Current tile size (4×4) is fixed. Future work:
- Larger tiles (8×8) for very smooth regions
- Smaller tiles (2×2) for complex regions
- Adaptive tile sizing based on local statistics

## References

- Design2.md: Complete SAES hardware design specification
- Challenge.md: Problem statement and motivation
- `transplat/scripts/analyze_gaussian_smoothness.py`: Validation of 3D continuity assumption
