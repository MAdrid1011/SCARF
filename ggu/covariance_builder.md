# Covariance Builder

Covariance matrix from scales and rotation.

## External Interface

### CovarianceBuilder

```python
class CovarianceBuilder:
    def __init__(self, config: GGUConfig)
    
    def map_scales(
        self,
        raw_scales: torch.Tensor,
        depth: float
    ) -> torch.Tensor
    
    def quaternion_to_rotation(
        self,
        quaternion: torch.Tensor
    ) -> torch.Tensor
    
    def build_covariance(
        self,
        scales: torch.Tensor,
        quaternion: torch.Tensor
    ) -> torch.Tensor
    
    def transform_to_world(
        self,
        cov_local: torch.Tensor,
        c2w_rotation: torch.Tensor
    ) -> torch.Tensor
```

**Methods:**

#### map_scales(raw_scales, depth) -> Tensor
Map network outputs to valid scale range.

- **Input**: `raw_scales` - [3] unbounded values, `depth` - for adaptive scaling
- **Output**: [3] scales in [scale_min, scale_max]
- **Formula**: `scales = scale_min + sigmoid(raw) × (scale_max - scale_min) × depth`

#### quaternion_to_rotation(quaternion) -> Tensor
Convert unit quaternion to rotation matrix.

- **Input**: `quaternion` - [4] (w, x, y, z), normalized
- **Output**: [3, 3] rotation matrix

**Formula:**
```
R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
     [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
     [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
```

#### build_covariance(scales, quaternion) -> Tensor
Build covariance matrix.

- **Input**: `scales` - [3], `quaternion` - [4]
- **Output**: [3, 3] symmetric positive-definite matrix
- **Formula**: `Σ = R @ diag(s²) @ R^T`

#### transform_to_world(cov_local, c2w_rotation) -> Tensor
Transform covariance to world space.

- **Formula**: `Σ_world = R_c2w @ Σ_local @ R_c2w^T`

## Hardware Mapping

- Sigmoid: 256-entry LUT
- Quaternion to R: 12 multiplies, 12 adds
- Covariance: 27 multiplies, 18 adds
- Total: ~400 LUTs, 12 DSPs, 5 cycles
