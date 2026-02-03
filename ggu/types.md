# GGU Types

Data structures for Gaussian Generation Unit.

## External Interface

### GGUConfig

Configuration dataclass for GGU processor.

```python
@dataclass
class GGUConfig:
    sh_degree: int = 2
    scale_min: float = 0.0005
    scale_max: float = 0.5
    opacity_activation: str = 'sigmoid'
```

**Fields:**
- `sh_degree`: Spherical harmonics degree (0, 1, or 2)
  - Degree 0: 1 coefficient (constant color)
  - Degree 1: 4 coefficients (linear variation)
  - Degree 2: 9 coefficients (quadratic variation)
- `scale_min`: Minimum Gaussian scale after mapping
- `scale_max`: Maximum Gaussian scale after mapping
- `opacity_activation`: Activation for opacity ('sigmoid' or 'softplus')

### GaussianOutput

Output Gaussian parameters.

```python
@dataclass
class GaussianOutput:
    position: torch.Tensor      # [3] world position
    covariance: torch.Tensor    # [3, 3] world covariance
    color: torch.Tensor         # [3] or [C] SH coefficients
    opacity: float
    
    def to_dict(self) -> Dict
    def to_saes_gaussian(self) -> 'Gaussian'
```

**Methods:**

#### to_dict() -> Dict
Export as dictionary for serialization.

#### to_saes_gaussian() -> Gaussian
Convert to SAES-compatible Gaussian type.

## Hardware Mapping

- GGUConfig: Compile-time LUT parameters
- GaussianOutput: Pipeline output registers
  - position: 3 × 16-bit fixed-point
  - covariance: 6 × 16-bit (symmetric, upper triangle)
  - color: 3 × 8-bit or 9 × 8-bit (SH)
  - opacity: 8-bit
