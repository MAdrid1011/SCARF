# DSU Tests

Unit tests for Depth Search Unit components.

## Test Files

| File | Tests |
|------|-------|
| `test_dsu_processor.py` | Projection, sampling, cost volume, softmax, FSDR callbacks |

## Fixtures (`conftest.py`)

- `MockDSUConfig`: Configuration without validation
- `reference_intrinsics`: 640×480 camera intrinsics
- `reference_extrinsics`: Identity camera pose
- `target_extrinsics`: Translated camera pose
- `mock_feature_map`: Sample feature tensor
- `mock_cost_volume`: Sample cost values
- `mock_probability_distribution`: Sample probabilities

## Running

```bash
# All DSU tests
pytest tests/dsu/ -v

# With verbose output
pytest tests/dsu/test_dsu_processor.py -v -s
```

## Coverage

Tests cover:
- Same-camera projection identity
- Cross-camera projection correctness
- Bilinear sampling interpolation
- Cost volume computation
- Softmax normalization
- Expected depth calculation
- FSDR callback creation
