# GGU - Gaussian Generation Unit

Hardware simulator for 3D Gaussian generation from depth and network outputs.

## Overview

GGU converts per-pixel depth estimates and network predictions into 3D Gaussians:
1. Calculate 3D world position from pixel + depth
2. Build covariance matrix from scales and rotation
3. Rotate spherical harmonics for view-dependent color

## Key Files

| File | Description |
|------|-------------|
| `types.py` | Data structures: `GGUConfig`, `GaussianOutput` |
| `position_calculator.py` | 3D position from pixel coordinates and depth |
| `covariance_builder.py` | Covariance matrix from scales and quaternion |
| `sh_rotator.py` | Spherical harmonics rotation |
| `ggu_processor.py` | Main processor combining all operations |

## Architecture

```
Pixel + Depth + Raw Gaussian Features
              │
    ┌─────────┼─────────┐
    ↓         ↓         ↓
Position   Covariance   SH
Calculator  Builder    Rotator
    │         │         │
    └─────────┴─────────┘
              │
              ↓
       GaussianOutput
   (position, covariance,
    color, opacity)
```

## Hardware Mapping

- **Position Calculator**: ~200 LUTs, 4 DSPs
- **Covariance Builder**: ~400 LUTs, 12 DSPs
- **SH Rotator**: ~600 LUTs, 16 DSPs (for degree 2)
- **Total**: ~1,200 LUTs, 32 DSPs

## Usage

```python
from ggu import GGUProcessor, GGUConfig

config = GGUConfig(sh_degree=2)
processor = GGUProcessor(config)

gaussian = processor.generate_gaussian(
    pixel_coord, depth, raw_features, density,
    intrinsics, extrinsics
)
print(f"Position: {gaussian.position}")
```

## SAES Compatibility

GaussianOutput is compatible with SAES `Gaussian` type for similarity evaluation.

See `docs/ggu-architecture.md` for detailed documentation.
