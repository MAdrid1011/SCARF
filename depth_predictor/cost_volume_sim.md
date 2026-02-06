# cost_volume_sim.py

Cost volume construction simulator for plane-sweep stereo matching.

## External Interface

### `CostVolumeSimulator`
Builds cost volumes by warping source features to reference view at multiple depth planes.

**Constructor:**
```python
CostVolumeSimulator(config: DepthPredictorConfig, device: torch.device)
```

**Key methods:**
- `build_cost_volume(ref_features, src_features, intrinsics, extrinsics, depth_candidates) -> Tuple[Tensor, int]` — Build cost volume via plane-sweep warping. Returns `(cost_volume [B, D, H, W], cycles)`.
- `warp_features(src_features, src_intrinsics, src_extrinsics, ref_intrinsics, ref_extrinsics, depth) -> Tensor` — Warp source features to reference frame at a given depth.

## Internal Helpers

### `_compute_homography(K_ref, K_src, R_rel, t_rel, depth) -> Tensor`
Compute 3×3 homography matrix for a given depth plane.

### `_correlation(ref_feat, warped_feat) -> Tensor`
Compute per-pixel correlation (dot product) between reference and warped features.
