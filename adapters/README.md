# Adapters - Model-Specific Adapters

Adapters isolating model-specific logic from core SCARF components.

## Overview

Different 3DGS models (Transplat, MVSplat, DepthSplat) have different:
- Cost/correlation volume semantics
- Depth candidate generation strategies
- Feature dimensions

Adapters provide a unified interface while encapsulating these differences.

## Key Files

| File | Description |
|------|-------------|
| `base_adapter.py` | Abstract `BaseAdapter` interface |
| `transplat_adapter.py` | Adapter for Transplat (cost volume, inverse depth) |
| `mvsplat_adapter.py` | Adapter for MVSplat (correlation, linear depth) |
| `depthsplat_adapter.py` | Adapter for DepthSplat (DINOv2 features) |

## Supported Models

| Model | Cost Type | Depth Spacing | Feature Dim |
|-------|-----------|---------------|-------------|
| Transplat | Cost (negate before softmax) | Inverse | 128 |
| MVSplat | Correlation (direct softmax) | Linear | 64 |
| DepthSplat | Correlation (direct softmax) | Linear | 384 |

## Usage

```python
from adapters import create_adapter

# Factory function
adapter = create_adapter('transplat')

# Extract depth distribution
probs = adapter.extract_depth_distribution(cost_volume, depth_candidates)

# Generate depth candidates
candidates = adapter.get_depth_candidates(near=0.5, far=10.0, num=32)

# Get model-specific config overrides
fsdr_overrides = adapter.get_fsdr_config_overrides()
```

## Adding New Models

1. Create `new_model_adapter.py` extending `BaseAdapter`
2. Implement required methods:
   - `extract_depth_distribution()`
   - `get_depth_candidates()`
   - `project_to_target()`
3. Register in `__init__.py` factory

See `docs/multi-model-integration.md` for detailed guide.
