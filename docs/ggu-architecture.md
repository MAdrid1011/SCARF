# GGU Architecture: Gaussian Generation Unit

## 1. Overview

The Gaussian Generation Unit (GGU) converts depth estimates and raw network outputs into 3D Gaussian primitives suitable for splatting-based rendering.

### 1.1 Design Goals

1. **Complete Gaussian generation**: Produce all Gaussian attributes (position, covariance, color, opacity)
2. **Depth-adaptive scaling**: Scale Gaussian size based on depth for consistent 3D coverage
3. **World-space transformation**: Transform local attributes to world coordinates
4. **SAES compatible**: Output Gaussians in format compatible with SAES evaluation

### 1.2 Core Functionality

```
Input: Pixel coordinate, depth, raw_gaussian features, camera params
Output: 3D Gaussian {mean, covariance, harmonics, opacity}

Steps:
1. Compute 3D position from pixel + depth
2. Build local covariance from scales and rotation
3. Transform covariance to world space
4. Rotate spherical harmonics to world space
```

---

## 2. System Architecture

### 2.1 Component Hierarchy

```
GGUProcessor (main engine)
├── PositionCalculator
│   ├── Ray computation
│   └── Depth-based positioning
├── CovarianceBuilder
│   ├── Scale mapping
│   ├── Quaternion normalization
│   └── Covariance matrix construction
└── SHRotator
    ├── Rotation matrix computation
    └── SH coefficient rotation
```

### 2.2 Data Flow

```
Input: pixel_coord [2], depth, raw_gaussian [C_raw], density, intrinsics [3,3], extrinsics [4,4]

GGUProcessor.generate_gaussian():
  │
  ├─► PositionCalculator.compute_position()
  │   ├─ Compute ray direction from pixel + intrinsics
  │   └─ position = origin + direction * depth
  │
  ├─► Parse raw_gaussian → scales [3], rotation [4], sh [C, D_sh]
  │
  ├─► CovarianceBuilder.build()
  │   ├─ Map scales to valid range [scale_min, scale_max]
  │   ├─ Apply depth-adaptive scaling
  │   ├─ Normalize quaternion
  │   ├─ Build rotation matrix from quaternion
  │   └─ Construct covariance: R @ diag(s²) @ R^T
  │
  ├─► CovarianceBuilder.transform_to_world()
  │   └─ world_cov = R_c2w @ local_cov @ R_c2w^T
  │
  └─► SHRotator.rotate()
      └─ world_sh = rotate_sh(local_sh, R_c2w)

Output: Gaussian {mean, cov, harmonics, opacity}
```

---

## 3. Position Calculator

### 3.1 Ray Computation

```python
class PositionCalculator:
    def get_ray(
        self,
        pixel_coord: Tensor,    # [2] (u, v)
        intrinsics: Tensor,     # [3, 3]
        extrinsics: Tensor,     # [4, 4]
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute camera ray for a pixel.
        
        Returns:
            origin: [3] camera position in world space
            direction: [3] normalized ray direction in world space
        """
        # Pixel to normalized camera coordinates
        uv_homog = torch.tensor([pixel_coord[0], pixel_coord[1], 1.0])
        ray_cam = torch.linalg.solve(intrinsics, uv_homog)
        ray_cam = ray_cam / torch.norm(ray_cam)  # Normalize
        
        # Camera to world rotation
        R_c2w = extrinsics[:3, :3]  # 3x3 rotation
        t_c2w = extrinsics[:3, 3]   # 3x1 translation
        
        # Transform ray to world space
        direction = R_c2w @ ray_cam
        origin = t_c2w
        
        return origin, direction
```

### 3.2 Position from Depth

```python
def compute_position(
    self,
    pixel_coord: Tensor,
    depth: float,
    intrinsics: Tensor,
    extrinsics: Tensor,
) -> Tensor:
    """
    Compute 3D position from pixel and depth.
    
    Formula: position = origin + direction * depth
    
    Returns:
        position: [3] world-space 3D position
    """
    origin, direction = self.get_ray(pixel_coord, intrinsics, extrinsics)
    position = origin + direction * depth
    return position
```

---

## 4. Covariance Builder

### 4.1 Scale Mapping

```python
class CovarianceBuilder:
    def __init__(self, config: GGUConfig):
        self.config = config
        self.scale_min = config.scale_min  # 0.0005
        self.scale_max = config.scale_max  # 0.5
    
    def map_scales(
        self,
        raw_scales: Tensor,    # [3] from network (unbounded)
        depth: float,
    ) -> Tensor:
        """
        Map raw scales to valid range with depth adaptation.
        
        Formula:
            scales = scale_min + (scale_max - scale_min) * sigmoid(raw_scales)
            scales = scales * depth * multiplier
        
        Returns:
            scales: [3] adapted scale values
        """
        # Sigmoid mapping to [scale_min, scale_max]
        scales = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(raw_scales)
        
        # Depth-adaptive scaling (farther objects need larger Gaussians)
        # Multiplier accounts for pixel footprint at depth
        multiplier = 1.0  # Can be adjusted based on intrinsics
        scales = scales * depth * multiplier
        
        return scales
```

### 4.2 Quaternion to Rotation

```python
def quaternion_to_rotation(
    self,
    quaternion: Tensor,    # [4] (w, x, y, z)
    eps: float = 1e-8,
) -> Tensor:
    """
    Convert normalized quaternion to 3x3 rotation matrix.
    
    Returns:
        R: [3, 3] rotation matrix
    """
    # Normalize quaternion
    q = quaternion / (torch.norm(quaternion) + eps)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Rotation matrix from quaternion
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    
    return R
```

### 4.3 Covariance Construction

```python
def build(
    self,
    scales: Tensor,         # [3]
    rotation: Tensor,       # [4] quaternion
) -> Tensor:
    """
    Build 3x3 covariance matrix from scales and rotation.
    
    Formula: Σ = R @ diag(s²) @ R^T
    
    Returns:
        cov: [3, 3] positive semi-definite covariance matrix
    """
    R = self.quaternion_to_rotation(rotation)
    
    # Scale matrix (diagonal)
    S = torch.diag(scales ** 2)
    
    # Covariance: R @ S @ R^T
    cov = R @ S @ R.T
    
    return cov
```

### 4.4 World Transform

```python
def transform_to_world(
    self,
    covariance: Tensor,     # [3, 3] local covariance
    c2w_rotation: Tensor,   # [3, 3] camera-to-world rotation
) -> Tensor:
    """
    Transform covariance to world space.
    
    Formula: Σ_world = R_c2w @ Σ_local @ R_c2w^T
    
    Returns:
        world_cov: [3, 3] world-space covariance
    """
    world_cov = c2w_rotation @ covariance @ c2w_rotation.T
    return world_cov
```

---

## 5. SH Rotator

### 5.1 Spherical Harmonics Rotation

```python
class SHRotator:
    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree
        self.num_coeffs = (sh_degree + 1) ** 2
    
    def rotate(
        self,
        sh_coeffs: Tensor,      # [C, num_coeffs] or [C, D_sh]
        rotation: Tensor,       # [3, 3] rotation matrix
    ) -> Tensor:
        """
        Rotate spherical harmonics coefficients.
        
        For degree-0 (DC): No rotation needed
        For degree-1: Apply rotation directly (linear)
        For higher degrees: Use Wigner D-matrices
        
        Simplified implementation for degree <= 3.
        
        Returns:
            rotated_sh: [C, num_coeffs] rotated coefficients
        """
        C = sh_coeffs.shape[0]
        result = sh_coeffs.clone()
        
        # Degree 0: DC term unchanged
        # result[:, 0] unchanged
        
        # Degree 1: Direct rotation (indices 1-3)
        if self.sh_degree >= 1:
            sh_1 = sh_coeffs[:, 1:4]  # [C, 3]
            rotated_1 = sh_1 @ rotation.T  # [C, 3]
            result[:, 1:4] = rotated_1
        
        # Degree 2: 5 coefficients (indices 4-8)
        if self.sh_degree >= 2:
            # Use precomputed Wigner D-matrix for degree 2
            D2 = self._compute_wigner_d_2(rotation)
            sh_2 = sh_coeffs[:, 4:9]  # [C, 5]
            rotated_2 = sh_2 @ D2.T
            result[:, 4:9] = rotated_2
        
        # Degree 3: 7 coefficients (indices 9-15)
        if self.sh_degree >= 3:
            D3 = self._compute_wigner_d_3(rotation)
            sh_3 = sh_coeffs[:, 9:16]  # [C, 7]
            rotated_3 = sh_3 @ D3.T
            result[:, 9:16] = rotated_3
        
        return result
    
    def _compute_wigner_d_2(self, R: Tensor) -> Tensor:
        """Compute 5x5 Wigner D-matrix for degree 2."""
        # Implementation based on rotation matrix elements
        # See: "Rotation of Real Spherical Harmonics"
        pass  # Omitted for brevity
    
    def _compute_wigner_d_3(self, R: Tensor) -> Tensor:
        """Compute 7x7 Wigner D-matrix for degree 3."""
        pass  # Omitted for brevity
```

---

## 6. GGU Processor

### 6.1 Complete Gaussian Generation

```python
class GGUProcessor:
    def __init__(self, config: GGUConfig):
        self.config = config
        self.position_calc = PositionCalculator()
        self.cov_builder = CovarianceBuilder(config)
        self.sh_rotator = SHRotator(config.sh_degree)
    
    def generate_gaussian(
        self,
        pixel_coord: Tensor,     # [2]
        depth: float,
        raw_gaussian: Tensor,    # [C_raw] = scales(3) + rotation(4) + sh(3*D_sh)
        density: float,
        intrinsics: Tensor,      # [3, 3]
        extrinsics: Tensor,      # [4, 4]
    ) -> 'Gaussian':
        """
        Generate complete Gaussian from network output.
        
        Args:
            pixel_coord: (u, v) pixel location
            depth: Estimated depth
            raw_gaussian: Network output [scales, rotation, sh_coeffs]
            density: Opacity/density value
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics (c2w)
        
        Returns:
            Gaussian: Complete 3D Gaussian primitive
        """
        # Parse raw features
        raw_scales = raw_gaussian[:3]
        raw_rotation = raw_gaussian[3:7]
        raw_sh = raw_gaussian[7:].reshape(3, -1)  # [3, D_sh]
        
        # 1. Compute position
        mean = self.position_calc.compute_position(
            pixel_coord, depth, intrinsics, extrinsics
        )
        
        # 2. Build covariance
        scales = self.cov_builder.map_scales(raw_scales, depth)
        local_cov = self.cov_builder.build(scales, raw_rotation)
        
        # 3. Transform to world space
        R_c2w = extrinsics[:3, :3]
        world_cov = self.cov_builder.transform_to_world(local_cov, R_c2w)
        
        # 4. Rotate spherical harmonics
        world_sh = self.sh_rotator.rotate(raw_sh, R_c2w)
        
        # 5. Create Gaussian (compatible with SAES types)
        from saes.types import Gaussian
        return Gaussian(
            mean=mean,
            cov=world_cov,
            opacity=float(torch.sigmoid(torch.tensor(density))),
            harmonics=world_sh,
        )
```

---

## 7. Hardware Mapping

### 7.1 Position Calculator Unit

```verilog
module position_calculator (
    input  wire [15:0] pixel_coord [0:1],
    input  wire [15:0] depth,
    input  wire [15:0] intrinsics [0:8],
    input  wire [15:0] extrinsics [0:15],
    output reg  [15:0] position [0:2]
);

// Stage 1: Compute ray direction (matrix solve/multiply)
wire [31:0] ray_cam [0:2];
// K^{-1} @ [u, v, 1]^T

// Stage 2: Transform to world
wire [31:0] ray_world [0:2];
// R_c2w @ ray_cam

// Stage 3: Scale by depth and add origin
always @(*) begin
    for (int i = 0; i < 3; i = i + 1) begin
        position[i] = extrinsics[i*4+3] + ray_world[i] * depth;
    end
end

endmodule
```

**Resources**: ~200 LUTs, 6 DSPs  
**Latency**: 3 cycles

### 7.2 Covariance Builder Unit

```verilog
module covariance_builder (
    input  wire [15:0] scales [0:2],
    input  wire [15:0] rotation [0:3],     // Quaternion
    input  wire [15:0] c2w_rotation [0:8], // 3x3
    output reg  [15:0] covariance [0:8]    // 3x3
);

// Stage 1: Quaternion to rotation matrix
wire [15:0] R_local [0:8];
quaternion_to_rotation u_q2r (
    .q(rotation),
    .R(R_local)
);

// Stage 2: Build covariance = R @ diag(s²) @ R^T
wire [15:0] cov_local [0:8];
// 9 multiply-adds for symmetric matrix

// Stage 3: Transform to world = R_c2w @ cov_local @ R_c2w^T
// 27 multiply-adds

endmodule
```

**Resources**: ~400 LUTs, 12 DSPs  
**Latency**: 5 cycles

### 7.3 SH Rotator Unit

```verilog
module sh_rotator (
    input  wire [15:0] sh_in [0:2][0:15],   // [C, D_sh]
    input  wire [15:0] rotation [0:8],       // 3x3
    output reg  [15:0] sh_out [0:2][0:15]
);

// Degree 0: Pass through
assign sh_out[*][0] = sh_in[*][0];

// Degree 1: Direct rotation (3x3 @ 3 for each channel)
// 9 MACs per channel × 3 channels = 27 MACs

// Degree 2: 5x5 Wigner D-matrix
// 25 MACs per channel × 3 channels = 75 MACs

// Degree 3: 7x7 Wigner D-matrix
// 49 MACs per channel × 3 channels = 147 MACs

endmodule
```

**Resources**: ~300 LUTs, 18 DSPs  
**Latency**: 8 cycles

---

## 8. Resource Summary

| Component | LUTs | DSPs | Cycles |
|-----------|------|------|--------|
| Position Calculator | 200 | 6 | 3 |
| Covariance Builder | 400 | 12 | 5 |
| SH Rotator | 300 | 18 | 8 |
| **GGU Total** | **900** | **36** | **16** |

---

## 9. Integration with SAES

The GGU outputs Gaussians in the same format as `saes.types.Gaussian`:

```python
@dataclass
class Gaussian:
    mean: Tensor        # [3] position
    cov: Tensor         # [3, 3] covariance
    opacity: float      # [0, 1] transparency
    harmonics: Tensor   # [C, D_sh] SH coefficients
```

This enables direct use with SAES similarity evaluation:

```python
# GGU generates Gaussians
ggu = GGUProcessor(config)
gaussians = [ggu.generate_gaussian(...) for pixel in tile]

# SAES evaluates similarity
from saes import GaussianSimilarityEvaluator
evaluator = GaussianSimilarityEvaluator()
metrics = evaluator.evaluate(gaussians)
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03  
**Related Issue**: GitHub Issue #2
