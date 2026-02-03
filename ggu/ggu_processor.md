# GGU Processor

Main processor for Gaussian generation.

## External Interface

### GGUProcessor

```python
class GGUProcessor:
    def __init__(self, config: GGUConfig)
    
    def generate_gaussian(
        self,
        pixel_coord: Tuple[float, float],
        depth: float,
        raw_gaussian: torch.Tensor,
        density: float,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> GaussianOutput
    
    def generate_batch(
        self,
        pixel_coords: torch.Tensor,
        depths: torch.Tensor,
        raw_gaussians: torch.Tensor,
        densities: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> List[GaussianOutput]
```

**Methods:**

#### generate_gaussian(...) -> GaussianOutput
Generate single 3D Gaussian from network outputs.

- **Input**:
  - `pixel_coord` - (u, v) image coordinates
  - `depth` - estimated depth
  - `raw_gaussian` - [14] network outputs:
    - [0:3]: raw scales
    - [3:7]: quaternion (wxyz)
    - [7:16]: SH coefficients (degree 2)
  - `density` - raw opacity value
  - `intrinsics` - [3, 3] camera matrix
  - `extrinsics` - [4, 4] camera-to-world
- **Output**: GaussianOutput with all parameters

**Pipeline:**
```
1. PositionCalculator: pixel + depth → world position
2. CovarianceBuilder:
   a. Map raw scales → valid scales
   b. Quaternion → rotation matrix
   c. Build local covariance
   d. Transform to world space
3. SHRotator: Rotate SH coefficients
4. Opacity: sigmoid(density)
5. Assemble GaussianOutput
```

#### generate_batch(...) -> List[GaussianOutput]
Generate multiple Gaussians (for vectorization).

## Internal Helpers

### _parse_raw_gaussian(raw) -> Tuple
Unpack raw network output into components.

### _activate_opacity(density) -> float
Apply sigmoid activation to opacity.

## Hardware Mapping

- Position: ~200 LUTs, 4 DSPs
- Covariance: ~400 LUTs, 12 DSPs
- SH Rotation: ~600 LUTs, 16 DSPs
- Opacity sigmoid: LUT
- Total: ~1,200 LUTs, 32 DSPs, 8 cycles
