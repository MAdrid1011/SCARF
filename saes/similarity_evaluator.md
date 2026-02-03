# GaussianSimilarityEvaluator Interface Documentation

## External Interface

### GaussianSimilarityEvaluator Class

**Purpose**: Compute 3D similarity among probe Gaussians based on position, covariance, color, and opacity dispersions.

**Initialization**:
```python
def __init__(self, scene_scale: Optional[float] = None)
```
- **Args**: `scene_scale` - Optional normalization factor for position dispersion
  - If None: Auto-estimated from Gaussian positions as std(positions)
  - If provided: Used for consistent normalization across multiple evaluations
- **Side Effects**: `scene_scale` may be set during first `evaluate()` call if initially None

---

### Primary Method: evaluate

```python
def evaluate(self, gaussians: List[Gaussian]) -> GaussianSimilarityMetrics
```

**Purpose**: Compute multi-dimensional similarity metrics.

**Parameters**:
- `gaussians`: List of probe Gaussians (typically 4)
  - Must be non-empty
  - Single Gaussian returns perfect similarity (score=1.0)

**Returns**: `GaussianSimilarityMetrics` with:
- `position_dispersion`: float - Spatial spread
- `covariance_dispersion`: float - Shape variation
- `color_dispersion`: float - Color difference
- `opacity_dispersion`: float - Transparency variation
- `similarity_score`: float in [0, 1] - Weighted aggregation

**Algorithm**:
1. Extract attributes from Gaussians
2. Compute 4 dispersion metrics independently
3. Weighted aggregation: `dispersion = 0.4×pos + 0.3×cov + 0.15×color + 0.15×opacity`
4. Map to similarity: `score = exp(-dispersion / 0.1)`

**Raises**:
- `ValueError`: If gaussians list is empty

**Performance**:
- For 4 Gaussians: 6 pairwise comparisons per metric
- Complexity: O(N²) where N = len(gaussians)
- Typical: <1ms for N=4 (software), 16 cycles (hardware)

**Example**:
```python
evaluator = GaussianSimilarityEvaluator(scene_scale=5.0)
metrics = evaluator.evaluate(probe_gaussians)

if metrics.similarity_score > 0.85:
    # High similarity → early-stop
    print("Smooth region detected")
print(f"Position dispersion: {metrics.position_dispersion:.3f}")
```

---

## Internal Helpers

### _compute_position_dispersion

```python
def _compute_position_dispersion(self, positions: Tensor) -> float
```

**Purpose**: Compute normalized spatial spread.

**Algorithm**:
```python
distances = _pairwise_distances(positions)  # [N, N]
max_dist = distances.max()
return max_dist / scene_scale
```

**Scene Scale**:
- If None: Auto-estimate as `positions.std()`
- Prevents division by zero: Uses max(scene_scale, 1e-6)

---

### _compute_covariance_dispersion

```python
def _compute_covariance_dispersion(self, covariances: Tensor) -> float
```

**Purpose**: Compute normalized shape variation.

**Algorithm**:
```python
cov_flat = covariances.reshape(N, -1)  # [N, 9]
distances = _pairwise_distances(cov_flat)
max_dist = distances.max()
avg_cov_norm = torch.norm(covariances, p='fro', dim=(-2,-1)).mean()
return max_dist / avg_cov_norm
```

**Normalization**: Uses Frobenius norm to handle different covariance scales.

---

### _compute_color_dispersion

```python
def _compute_color_dispersion(self, harmonics: Tensor) -> float
```

**Purpose**: Compute color (SH) variation.

**Algorithm**:
```python
sh_dc = harmonics[:, :3, 0]  # Extract RGB DC component [N, 3]
distances = _pairwise_distances(sh_dc)
return distances.max()
```

**Note**: Uses first 3 SH channels as RGB. Adapts if harmonics.shape[1] < 3.

---

### _compute_opacity_dispersion

```python
def _compute_opacity_dispersion(self, opacities: Tensor) -> float
```

**Purpose**: Compute transparency variation.

**Algorithm**: `return opacities.max() - opacities.min()`

---

### _compute_similarity_score

```python
def _compute_similarity_score(
    self, pos_disp: float, cov_disp: float, 
    color_disp: float, opacity_disp: float
) -> float
```

**Purpose**: Aggregate dispersions into similarity score.

**Formula** (from Design2):
```python
weighted_dispersion = (
    0.4 * pos_disp +      # Position most important
    0.3 * cov_disp +      # Shape second
    0.15 * color_disp +   # Color tertiary
    0.15 * opacity_disp   # Opacity tertiary
)
similarity = exp(-weighted_dispersion / 0.1)  # Temperature = 0.1
```

**Output**: Clamped to [0, 1] for numerical stability

---

### _pairwise_distances

```python
def _pairwise_distances(self, vectors: Tensor) -> Tensor
```

**Purpose**: Efficiently compute all pairwise L2 distances.

**Algorithm**:
```python
||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩

dots = vectors @ vectors.T  # [N, N]
norms_sq = torch.diag(dots)  # [N]
distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
distances = torch.sqrt(torch.clamp(distances_sq, min=0))
```

**Complexity**: O(N²D) where D = vector dimension

**Hardware Optimization**: Matrix multiply parallelizable

---

### _estimate_scene_scale

```python
def _estimate_scene_scale(self, positions: Tensor) -> float
```

**Purpose**: Auto-estimate scene scale from Gaussian positions.

**Algorithm**: `return positions.std().item()`

**Rationale**: Standard deviation represents typical spatial extent, enabling position dispersion to be comparable across scenes of different sizes (small objects vs. large rooms).

---

## Hardware Mapping

### Position Comparator → Hardware Module

```verilog
module position_comparator (
    input  wire [15:0] mean_0 [0:2],  // Position 0 (3×int16)
    input  wire [15:0] mean_1 [0:2],  // Position 1
    input  wire [15:0] mean_2 [0:2],  // Position 2
    input  wire [15:0] mean_3 [0:2],  // Position 3
    input  wire [15:0] scene_scale,   // Normalization factor
    output reg  [11:0] position_dispersion  // 12-bit fixed-point
);

// Compute 6 pairwise distances in parallel
wire [15:0] dist_01, dist_02, dist_03, dist_12, dist_13, dist_23;
wire [15:0] max_dist;

distance_3d u_dist01 (.p1(mean_0), .p2(mean_1), .dist(dist_01));
distance_3d u_dist02 (.p1(mean_0), .p2(mean_2), .dist(dist_02));
// ... (6 distance units total)

max6 u_max (.in({dist_01, dist_02, dist_03, dist_12, dist_13, dist_23}), 
            .out(max_dist));

divider u_norm (.num(max_dist), .den(scene_scale), .quot(position_dispersion));

endmodule
```

**Resources**: ~300 LUTs, 18 DSPs (6 distance units)  
**Latency**: 4 cycles (parallel distance + max + divide)

---

### Weighted Aggregator → Hardware Module

```verilog
module weighted_aggregator (
    input  wire [11:0] pos_disp,
    input  wire [11:0] cov_disp,
    input  wire [11:0] color_disp,
    input  wire [11:0] opacity_disp,
    input  wire [7:0]  weight_pos,      // 0.4 * 255 = 102
    input  wire [7:0]  weight_cov,      // 0.3 * 255 = 77
    input  wire [7:0]  weight_color,    // 0.15 * 255 = 38
    input  wire [7:0]  weight_opacity,  // 0.15 * 255 = 38
    output reg  [11:0] weighted_dispersion
);

// Stage 1: Multiply (parallel)
wire [19:0] term_pos = pos_disp * weight_pos;        // 12+8=20 bits
wire [19:0] term_cov = cov_disp * weight_cov;
wire [19:0] term_color = color_disp * weight_color;
wire [19:0] term_opacity = opacity_disp * weight_opacity;

// Stage 2: Sum
wire [21:0] sum = term_pos + term_cov + term_color + term_opacity;

// Stage 3: Normalize (divide by 255 to get back to original scale)
always @(*) begin
    weighted_dispersion = sum[19:8];  // Right-shift 8 bits ≈ divide by 256
end

endmodule
```

**Resources**: ~150 LUTs, 4 DSPs (4 multipliers, 3 adders)  
**Latency**: 2 cycles

---

## Design Rationale

**Why position weighted 0.4?**
- Spatial continuity is strongest geometric signal
- Neighboring pixels in smooth regions have nearby 3D positions
- Empirical validation from transplat smoothness analysis

**Why exponential mapping?**
- Smooth degradation: Small dispersion → high similarity
- Temperature (0.1) controls sensitivity
- Hardware-friendly: LUT implementation

**Why auto scene scale?**
- Adaptivity: Works across different scene sizes
- No manual tuning: Automatic normalization
- Robustness: Prevents scale-dependent thresholds

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03
