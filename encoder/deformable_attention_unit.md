# deformable_attention_unit.py

Hardware simulator for Multi-Scale Deformable Attention (Transplat UVTransformer).

## External Interface

### `DeformableAttentionConfig` (dataclass)
Configuration for deformable attention hardware.

**Fields:**
- `parallel_heads: int = 4` — Heads processed in parallel
- `parallel_points: int = 4` — Sampling points processed in parallel
- `parallel_channels: int = 32` — Channels processed in parallel
- `coord_bits: int = 16` — Coordinate fixed-point precision
- `weight_bits: int = 16` — Attention weight precision
- `memory_bandwidth_bytes: int = 64` — Memory bandwidth (bytes/cycle)

### `DeformableAttentionUnit`
Multi-scale deformable attention hardware simulator.

Architecture:
1. Coordinate Unit: fixed-point multiply-add for sampling grid transform
2. Bilinear Sampler: 4-point sampling with interpolation
3. Accumulator: MAC operations for weighted sum across heads and points

**Constructor:**
```python
DeformableAttentionUnit(config: Optional[DeformableAttentionConfig] = None)
```

**Methods:**
- `forward(value, spatial_shapes, sampling_locations, attention_weights) -> Tuple[Tensor, CycleStats]` — Multi-scale deformable attention.
  - `value: [B, total_hw, num_heads, head_dim]`
  - `spatial_shapes: [num_levels, 2]` (H, W per level)
  - `sampling_locations: [B, N, num_heads, num_levels, num_points, 2]`
  - `attention_weights: [B, N, num_heads, num_levels, num_points]`
- `get_total_cycles() -> int` — Accumulated cycles.
- `reset_cycles() -> None` — Reset cycle counter.

### `create_deformable_attention_unit(**kwargs) -> DeformableAttentionUnit`
Factory function with custom config.

## Internal Helpers

### `_sample_single_level(value_level, sampling_locs, H, W) -> Tensor`
Bilinear grid sampling at a single feature level.

### `_compute_cycles(value, spatial_shapes, sampling_locations, attention_weights) -> CycleStats`
Cycle model: coord transform (2 cycles/point) + bilinear sample (4 cycles/point) + weighted accumulate (1 cycle/point) per head.
