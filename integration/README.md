# Integration - Complete SCARF Accelerator

Unified accelerator integrating all SCARF components.

## Overview

The integration module combines FSDR, DSU, GGU, and model adapters into a complete accelerator pipeline for 3DGS inference optimization.

## Key Files

| File | Description |
|------|-------------|
| `accelerator.py` | Main `Accelerator` class integrating all components |
| `pipeline.py` | Batch processing and progress tracking |

## Architecture

```
                    Accelerator
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ↓                   ↓                   ↓
 Adapter            FSDR+DSU              GGU
(model-specific)  (depth estimation)  (Gaussian gen)
    │                   │                   │
    └───────────────────┴───────────────────┘
                        │
                        ↓
                  Gaussians + Profiling
```

## Usage

```python
from integration import Accelerator

# Create accelerator for specific model
accelerator = Accelerator(
    model_type='transplat',
    enable_fsdr=True,
    enable_profiling=True,
)

# Setup FSDR with depth candidates
accelerator.setup_fsdr(depth_candidates)

# Process pixels
gaussian, profiling = accelerator.process_pixel(
    ref_feature, target_features, pixel_coord,
    depth_candidates, raw_gaussian, density,
    ref_K, ref_c2w, tgt_K, tgt_c2w,
)

# Get profiling summary
print(accelerator.get_fsdr_profiling())
```

## Pipeline Usage

```python
from integration import Pipeline, PipelineConfig

config = PipelineConfig(batch_size=64, enable_fsdr=True)
pipeline = Pipeline(config)

pipeline.start_batch()
# ... process pixels ...
pipeline.end_batch(num_pixels=64)

print(pipeline.get_stats())
```

See component READMEs for detailed documentation.
