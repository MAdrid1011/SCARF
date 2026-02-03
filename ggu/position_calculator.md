# Position Calculator

3D position from pixel coordinates and depth.

## External Interface

### PositionCalculator

```python
class PositionCalculator:
    def __init__(self, config: GGUConfig)
    
    def calculate(
        self,
        pixel: Tuple[float, float],
        depth: float,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> torch.Tensor
```

**Methods:**

#### calculate(pixel, depth, intrinsics, extrinsics) -> Tensor
Calculate 3D world position from pixel and depth.

- **Input**:
  - `pixel` - (u, v) image coordinates
  - `depth` - depth value
  - `intrinsics` - [3, 3] camera intrinsic matrix
  - `extrinsics` - [4, 4] camera-to-world transform
- **Output**: [3] world position (x, y, z)

**Formula:**
```
ray = K^{-1} @ [u, v, 1]^T
point_cam = ray × depth / ray[2]  # Normalize z=1
point_world = R @ point_cam + t
```

## Internal Helpers

### _unproject_pixel(pixel, K_inv) -> Tensor
Convert pixel to normalized ray direction.

### _transform_to_world(point_cam, extrinsics) -> Tensor
Apply camera-to-world transformation.

## Hardware Mapping

- K inverse: Precomputed, stored as 9 constants
- Ray calculation: 3 multiplies, 2 adds
- Depth scaling: 3 multiplies
- World transform: 9 multiplies, 6 adds
- Total: ~200 LUTs, 4 DSPs, 2 cycles
