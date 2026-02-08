# types.py

Type definitions for depth prediction hardware simulators.

## External Interface

### `CostVolumeType` (Enum)
Cost volume construction method.
- `CORRELATION`: Direct correlation (MVSplat, DepthSplat)
- `TRANSFORMER`: Transformer matching (Transplat)

### `DepthRegressionType` (Enum)
Depth regression method.
- `SOFT_ARGMAX`: Softmax + weighted sum
- `ARGMAX`: Hard argmax

### `DepthPredictorConfig` (dataclass)
Configuration for depth predictor simulator.

**Key fields:**
- `num_depth_candidates: int = 32` — Number of depth bins
- `depth_range: Tuple[float, float] = (0.5, 100.0)` — Near/far planes
- `use_inverse_depth: bool = True` — Use inverse depth spacing
- `cost_volume_type: CostVolumeType` — Cost volume construction method
- `regression_type: DepthRegressionType` — Depth regression method
- `softmax_temperature: float = 1.0` — Softmax temperature

**Presets:**
- `transplat_preset()` — Transplat defaults (32 candidates, transformer matching)
- `mvsplat_preset()` — MVSplat defaults (32 candidates, correlation)
- `depthsplat_preset()` — DepthSplat defaults (32 candidates, multi-scale)

### `DepthPredictorOutput` (dataclass)
Output container from depth predictor simulators.

**Fields:**
- `depths: torch.Tensor` — Predicted depths `[B, V, H*W, 1, 1]`
- `raw_gaussians: Optional[torch.Tensor]` — Raw Gaussian parameters
- `densities: Optional[torch.Tensor]` — Per-pixel density
- `cycle_breakdown: CycleBreakdown` — Per-stage hardware cycle breakdown
- `total_cycles: int` — Total hardware cycles

### `CycleBreakdown` (dataclass)
Cycle breakdown for depth predictor stages.

**Fields:**
- `feature_extraction: int` — Feature extraction cycles
- `cost_volume: int` — Cost volume construction cycles
- `unet_refinement: int` — U-Net refinement cycles
- `depth_head: int` — Depth head + regression cycles
- `gaussian_head: int` — Gaussian head cycles
- `total: int` — Sum of all stages

### `UNetConfig`, `UNetLayerConfig` (dataclasses)
Configuration for U-Net architecture used in cost volume refinement.

## Internal Helpers

None — this module contains only data definitions.
