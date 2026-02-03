# SH Rotator

Spherical harmonics rotation for view-dependent color.

## External Interface

### SHRotator

```python
class SHRotator:
    def __init__(self, config: GGUConfig)
    
    def rotate(
        self,
        sh_coeffs: torch.Tensor,
        rotation: torch.Tensor
    ) -> torch.Tensor
    
    def rotate_degree_1(
        self,
        sh_coeffs: torch.Tensor,
        rotation: torch.Tensor
    ) -> torch.Tensor
```

**Methods:**

#### rotate(sh_coeffs, rotation) -> Tensor
Rotate SH coefficients by given rotation.

- **Input**:
  - `sh_coeffs` - [C] coefficients (1, 4, or 9 based on degree)
  - `rotation` - [3, 3] rotation matrix
- **Output**: [C] rotated coefficients

**Degree Handling:**
- Degree 0 (C=1): No rotation needed (constant)
- Degree 1 (C=4): `[c0, R @ c1:4]`
- Degree 2 (C=9): Uses Wigner D-matrices

#### rotate_degree_1(sh_coeffs, rotation) -> Tensor
Optimized rotation for degree-1 SH.

- **Input**: `sh_coeffs` - [4] (1 DC + 3 linear)
- **Output**: [4] rotated coefficients
- **Formula**: First coefficient unchanged, last 3 rotated by R

## Internal Helpers

### _wigner_d_matrix(rotation, degree) -> Tensor
Compute Wigner D-matrix for SH rotation.

- Degree 1: 3×3 matrix (same as rotation)
- Degree 2: 5×5 matrix (computed from rotation elements)

### _apply_wigner(coeffs, D_matrix) -> Tensor
Apply Wigner matrix to coefficient block.

## Hardware Mapping

- Degree 0: Pass-through (0 cycles)
- Degree 1: 9 multiplies, 6 adds (~200 LUTs, 4 DSPs)
- Degree 2: 25 multiplies, 20 adds (~600 LUTs, 16 DSPs)
- Total (degree 2): ~600 LUTs, 16 DSPs, 3 cycles
