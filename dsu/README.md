# DSU - Depth Search Unit

Hardware simulator for depth estimation via cost volume computation.

## Overview

DSU performs multi-view stereo depth estimation by:
1. Projecting reference pixels to target views at candidate depths
2. Sampling target features via bilinear interpolation
3. Computing matching costs (correlation/distance)
4. Aggregating via softmax to produce depth distribution

## Key Files

| File | Description |
|------|-------------|
| `types.py` | Data structures: `DSUConfig`, `DSUResult` |
| `depth_sampler.py` | 3D-2D projection and bilinear sampling |
| `cost_volume.py` | Feature matching cost computation |
| `softmax_aggregator.py` | Probability distribution and depth estimation |
| `dsu_processor.py` | Main processor with FSDR callback support |

## Architecture

```
Reference Pixel + Depth Candidates
              │
              ↓
        Depth Sampler
    (project to target view)
              │
              ↓
        Cost Volume
    (feature correlation)
              │
              ↓
    Softmax Aggregator
              │
              ↓
    Expected Depth + Stats
```

## Hardware Mapping

- **Depth Sampler**: ~800 LUTs, 16 DSPs (projection + bilinear)
- **Cost Volume**: ~400 LUTs, 8 DSPs (dot product)
- **Softmax**: ~600 LUTs, 4 DSPs (exp LUT + division)
- **Total**: ~1,800 LUTs, 28 DSPs

## Usage

```python
from dsu import DSUProcessor, DSUConfig

config = DSUConfig(feature_dim=128, num_candidates=32)
processor = DSUProcessor(config)

result = processor.search_depth(
    ref_feature, target_features, pixel_coord,
    depth_candidates, ref_K, ref_c2w, tgt_K, tgt_c2w
)
print(f"Depth: {result.depth}")
```

## FSDR Integration

DSU provides callback factories for FSDR:
```python
cost_fn = processor.create_fsdr_cost_fn(...)
prob_fn = processor.create_fsdr_prob_fn(...)
```

See `docs/dsu-architecture.md` for detailed documentation.
