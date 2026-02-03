# TileProcessor Interface Documentation

## External Interface

### TileProcessor Class

**Purpose**: Main SAES orchestration engine coordinating 3-phase tile processing workflow.

**Initialization**:
```python
def __init__(self, config: TileConfig)
```
- **Args**: `config` - TileConfig with tile size, thresholds, sparse indices
- **Creates**: similarity_evaluator, decision_controller, gaussian_merger (initialized internally)
- **Thread Safety**: Not thread-safe (create separate instances per thread)

---

### Primary Method: process_scene

```python
def process_scene(
    self,
    features: Tensor,              # [B, V, C, H, W]
    depth_predictor_fn: Callable,  # fn(features, indices) -> depths
    gaussian_adapter_fn: Callable, # fn(depths, context) -> List[Gaussian]
    context: Dict,
) -> Tuple[List[Gaussian], SAESProfilingResult]:
```

**Purpose**: Process entire scene with tile-based SAES.

**Parameters**:
- `features`: Feature map tensor `[B, V, C, H, W]`
  - B: Batch size (typically 1)
  - V: Number of views (e.g., 2 for Transplat, 3 for DepthSplat)
  - C: Feature channels (e.g., 128)
  - H, W: Spatial dimensions (e.g., 64×64)
  
- `depth_predictor_fn`: Callback for depth prediction
  - **Signature**: `(features: Tensor, indices: List[int]) -> Tensor`
  - **Contract**: 
    - Input: Tile feature slice and pixel indices
    - Output: Depths for requested indices
    - Example: For indices=[0,1,2,3], returns depths tensor of shape [4, ...]
  
- `gaussian_adapter_fn`: Callback for Gaussian generation
  - **Signature**: `(depths: Tensor, context: Dict) -> List[Gaussian]`
  - **Contract**:
    - Input: Depths and camera context
    - Output: List of Gaussian primitives
    - Must return Gaussian with {mean, cov, opacity, harmonics}
  
- `context`: Dictionary with scene-specific parameters
  - **Required**: None (implementation-defined)
  - **Optional**: `scene_name` (str) - for profiling identification
  - **Model-specific**: intrinsics, extrinsics, near, far (passed to callbacks)

**Returns**:
- `List[Gaussian]`: Final Gaussians for entire scene
  - Length varies: early-stop reduces count via merging
  - Order: Row-major tile order
  
- `SAESProfilingResult`: Performance metrics
  - Fields: total_tiles, path distribution, computation_saving_ratio, overhead_ns, net_speedup
  - See `types.md` for complete structure

**Raises**:
- `ValueError`: If feature map size is invalid (H=0 or W=0)

**Side Effects**:
- Modifies `self.profiler` (created per scene)
- Modifies `self.similarity_evaluator.scene_scale` (auto-estimated)

**Thread Safety**: Not thread-safe. Create separate TileProcessor instances for parallel processing.

**Example**:
```python
processor = TileProcessor(config)

# Process scene
gaussians, profiling = processor.process_scene(
    features=torch.randn(1, 2, 128, 64, 64),
    depth_predictor_fn=my_depth_fn,
    gaussian_adapter_fn=my_gaussian_fn,
    context={"scene_name": "scene_001"},
)

# Inspect results
print(f"Generated {len(gaussians)} Gaussians")
print(f"Computation saving: {profiling.computation_saving_ratio:.2%}")
print(f"Net speedup: {profiling.net_speedup:.2f}×")
```

---

## Internal Helpers

### _process_tile

```python
def _process_tile(
    self,
    tile_id: Tuple[int, int],
    tile_coords: Tuple[int, int, int, int],
    features: Tensor,
    depth_predictor_fn: Callable,
    gaussian_adapter_fn: Callable,
    context: Dict,
) -> List[Gaussian]:
```

**Purpose**: Execute 3-phase workflow for single tile.

**Algorithm**:
1. **Phase 1 (Probe)**: Process first N pixels (typically 4)
   - Extract tile features
   - Call depth_predictor_fn for probe indices [0..3]
   - Call gaussian_adapter_fn to generate probe Gaussians
   
2. **Phase 2 (Evaluate)**: Compute 3D similarity
   - Call similarity_evaluator.evaluate(probe_gaussians)
   - Get similarity_score in [0, 1]
   
3. **Phase 3 (Decide & Execute)**: Select and execute path
   - Call decision_controller.decide_path(similarity_score)
   - If early_stop: Merge and enlarge probes → skip remaining
   - If sparse/full: Process remaining indices → concatenate

**Returns**: List of Gaussians for this tile

---

### _process_pixels

```python
def _process_pixels(
    self,
    tile_features: Tensor,
    indices: List[int],
    depth_predictor_fn: Callable,
    gaussian_adapter_fn: Callable,
    context: Dict,
) -> List[Gaussian]:
```

**Purpose**: Process specified pixels through depth + Gaussian generation.

**Algorithm**:
1. Call depth_predictor_fn(tile_features, indices)
2. Call gaussian_adapter_fn(depths, context)
3. Return generated Gaussians

**Handles**: Empty indices list (returns [])

---

### _compute_tile_grid

```python
def _compute_tile_grid(self, H: int, W: int, tile_size: int) -> Tuple[int, int]:
```

**Purpose**: Compute tile grid dimensions.

**Algorithm**: Ceiling division
```python
n_tiles_h = (H + tile_size - 1) // tile_size
n_tiles_w = (W + tile_size - 1) // tile_size
```

**Example**:
- H=64, W=64, tile_size=4 → (16, 16)
- H=65, W=65, tile_size=4 → (17, 17)

---

### _tile_coordinates

```python
def _tile_coordinates(
    self,
    tile_id: Tuple[int, int],
    H: int,
    W: int,
    tile_size: int,
) -> Tuple[int, int, int, int]:
```

**Purpose**: Convert tile ID to pixel coordinates.

**Returns**: (row_start, row_end, col_start, col_end)

**Example**: tile_id=(1, 2), tile_size=4 → (4, 8, 8, 12)

**Edge Handling**: Clamps at feature map boundaries
```python
row_end = min(row_start + tile_size, H)
col_end = min(col_start + tile_size, W)
```

---

### _extract_tile_features

```python
def _extract_tile_features(
    self,
    features: Tensor,
    tile_coords: Tuple[int, int, int, int],
) -> Tensor:
```

**Purpose**: Slice feature tensor for tile region.

**Implementation**: Standard PyTorch slicing
```python
row_start, row_end, col_start, col_end = tile_coords
return features[:, :, :, row_start:row_end, col_start:col_end]
```

---

## Hardware Mapping

### TileProcessor → SAES_Controller FSM

**States**:
1. IDLE: Waiting for new scene
2. TILE_SETUP: Compute tile grid and coordinates
3. PROBE: Execute Phase 1 (call DSU + Gaussian Gen)
4. EVALUATE: Execute Phase 2 (call Similarity Evaluator)
5. DECIDE: Execute Phase 3 (call Decision Logic)
6. EXECUTE: Process remaining pixels or merge probes
7. OUTPUT: Write tile Gaussians to buffer
8. DONE: All tiles complete

**Control Signals**:
- `probe_start`, `probe_done`
- `eval_start`, `eval_done`
- `decide_start`, `decide_done`
- `execute_start`, `execute_done`

**Cycle Count**:
- TILE_SETUP: 1 cycle
- PROBE: 100 cycles
- EVALUATE: 16 cycles
- DECIDE: <1 cycle
- EXECUTE: 5-300 cycles (path-dependent)
- OUTPUT: 5 cycles

**Total per tile**: 121-417 cycles (vs. 400 baseline)

---

## Error Handling

| Error Condition | Detection | Behavior |
|----------------|-----------|----------|
| Empty feature map (H=0 or W=0) | `process_scene` entry | Raise ValueError |
| Empty Gaussian list | `similarity_evaluator.evaluate` | Raise ValueError |
| Invalid path name | `decision_controller.get_remaining_indices` | Raise ValueError |
| Invalid threshold config | `TileConfig.__post_init__` | Raise ValueError at construction |

**Philosophy**: Fail fast at entry point, not during processing loop.

---

## Performance Characteristics

### Expected Metrics (RE10K dataset)

| Metric | Value | Range |
|--------|-------|-------|
| **Computation Saving** | 42% | [30%, 50%] |
| **Early-stop Ratio** | 30% | [20%, 40%] |
| **Sparse Ratio** | 40% | [30%, 50%] |
| **Full Ratio** | 30% | [20%, 40%] |
| **SAES Overhead** | 8% | [5%, 10%] |
| **Net Speedup** | 1.31× | [1.2×, 1.4×] |

### Scaling Analysis

| Feature Map | Tiles | Baseline Cycles | SAES Cycles (avg) | Speedup |
|-------------|-------|-----------------|-------------------|---------|
| 16×16 | 16 | 6.4K | 4.8K | 1.33× |
| 32×32 | 64 | 25.6K | 19.2K | 1.33× |
| 64×64 | 256 | 102.4K | 76.8K | 1.33× |
| 128×128 | 1024 | 409.6K | 307.2K | 1.33× |

**Insight**: Speedup scales consistently regardless of resolution.

---

## Multi-Model Adaptation Guide

### Transplat Integration

```python
# In encoder_trans.py
def depth_fn(features, indices):
    return self.depth_predictor(features, indices=indices, ...)

def gaussian_fn(depths, context):
    return self.gaussian_adapter(depths, context, ...)

gaussians, saes_result = processor.process_scene(
    trans_features, depth_fn, gaussian_fn, context
)
```

### DepthSplat Integration (3-view)

```python
# DepthSplat: [B, 3, C, H, W] (3 views)
def depth_fn(features, indices):
    # DepthSplat may need all 3 views for depth
    return depthsplat_model.predict_depth_multi_view(
        features, pixel_indices=indices
    )

def gaussian_fn(depths, context):
    # Convert DepthSplat format to SAES Gaussian
    ds_gauss = depthsplat_model.splat(depths, context)
    return [to_saes_gaussian(g) for g in ds_gauss]
```

### MVSPlat Integration (multi-scale)

```python
# MVSPlat: Multi-scale features
def depth_fn(features, indices):
    # Select appropriate scale
    scale = context.get('saes_scale', 0)
    return mvsplat_model.depth_net[scale](features, indices)

def gaussian_fn(depths, context):
    # MVSPlat may fuse across scales
    return mvsplat_model.gaussian_net(depths, scale=context['saes_scale'])
```

**Key**: All model specifics isolated in adapter callbacks, SAES core unchanged.

---

## Future Enhancements

### Planned

1. **Protocol Interfaces** (Type Safety):
   - Define `DepthPredictorProtocol`, `GaussianAdapterProtocol`
   - Enable type checking for callbacks

2. **Configurable Similarity Weights**:
   - Extract (0.4, 0.3, 0.15, 0.15) to `SimilarityWeights` dataclass
   - Enable per-model tuning

3. **Adaptive Thresholding**:
   - Learn optimal thresholds per scene
   - Adjust based on quality-speed tradeoff

### Under Consideration

1. **Variable Tile Sizes**: Adapt tile size based on scene complexity
2. **Hierarchical Tiling**: Multi-level tile organization
3. **Online Learning**: Update thresholds during inference

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03
