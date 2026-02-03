# Depth Sampler

3D-2D projection and bilinear feature sampling.

## External Interface

### DepthSampler

```python
class DepthSampler:
    def __init__(self, config: DSUConfig)
    
    def project(
        self,
        pixel: Tuple[float, float],
        depth: float,
        ref_intrinsics: torch.Tensor,
        ref_extrinsics: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_extrinsics: torch.Tensor
    ) -> Tuple[float, float]
    
    def sample_bilinear(
        self,
        feature_map: torch.Tensor,
        u: float,
        v: float
    ) -> torch.Tensor
```

**Methods:**

#### project(pixel, depth, ...) -> (u, v)
Project reference pixel to target view at given depth.

- **Input**:
  - `pixel` - (u, v) reference coordinates
  - `depth` - depth value
  - `ref_intrinsics` - [3, 3] reference camera matrix
  - `ref_extrinsics` - [4, 4] reference camera-to-world
  - `tgt_intrinsics` - [3, 3] target camera matrix
  - `tgt_extrinsics` - [4, 4] target camera-to-world
- **Output**: (u, v) target coordinates

**Pipeline:**
```
1. Unproject: ray = K_ref^{-1} @ [u, v, 1]^T
2. Scale: point_cam = ray × depth
3. To world: point_world = R_ref @ point_cam + t_ref
4. To target: point_tgt = R_tgt^{-1} @ (point_world - t_tgt)
5. Project: [u', v', w] = K_tgt @ point_tgt
6. Normalize: (u'/w, v'/w)
```

#### sample_bilinear(feature_map, u, v) -> Tensor
Sample feature at sub-pixel location.

- **Input**:
  - `feature_map` - [C, H, W] feature tensor
  - `u, v` - continuous coordinates
- **Output**: [C] interpolated feature

**Interpolation:**
```
f = (1-dx)(1-dy)f[y0,x0] + dx(1-dy)f[y0,x1]
  + (1-dx)dy f[y1,x0] + dx dy f[y1,x1]
```

## Hardware Mapping

- Projection: ~400 LUTs, 8 DSPs (matrix multiply)
- Bilinear: ~400 LUTs, 8 DSPs (4 weighted adds)
- Total: ~800 LUTs, 16 DSPs
