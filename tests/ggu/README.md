# GGU Tests

Unit tests for Gaussian Generation Unit components.

## Test Files

| File | Tests |
|------|-------|
| `test_ggu_processor.py` | Position, covariance, SH rotation, full pipeline |

## Fixtures (`conftest.py`)

- `MockGGUConfig`: Configuration without validation
- `intrinsics`: 640×480 camera intrinsics
- `extrinsics_identity`: Identity camera pose
- `extrinsics_rotated`: 90° rotated camera
- `raw_gaussian_features`: Sample network outputs

## Running

```bash
# All GGU tests
pytest tests/ggu/ -v

# Specific test class
pytest tests/ggu/test_ggu_processor.py::TestCovarianceBuilder -v
```

## Coverage

Tests cover:
- Position at image center
- Depth scaling of position
- Scale mapping via sigmoid
- Quaternion to rotation conversion
- Covariance matrix symmetry and positive definiteness
- 90° rotation correctness
- SH rotation for view-dependent color
- Full pipeline integration
