# FSDR - Feature-Similarity Depth Reuse

Hardware simulator for Feature-Similarity Depth Reuse optimization in 3DGS depth estimation.

## Overview

FSDR reduces redundant memory access during depth search by exploiting semantic similarity in 2D feature space. When neighboring pixels have similar features, their optimal depths are likely similar, allowing cache-based reuse.

## Key Files

| File | Description |
|------|-------------|
| `types.py` | Data structures: `FSDRConfig`, `CacheEntry`, `FSDRResult` |
| `lsh_hasher.py` | Locality-Sensitive Hashing for feature signatures |
| `cache_table.py` | Semantic cache with LRU replacement |
| `depth_corrector.py` | Three-level correction strategy |
| `light_verifier.py` | Local depth search for uncertain hits |
| `fsdr_processor.py` | Main processor orchestrating the workflow |
| `profiler.py` | Performance metrics collection |

## Architecture

```
Feature → LSH Hasher → Cache Lookup
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
          Cache Hit                 Cache Miss
              │                         │
              ↓                         ↓
       DepthCorrector              Full Search
       (3-level strategy)               │
              │                         ↓
              └─────────→ Depth ←───────┘
                           │
                           ↓
                     Cache Update
```

## Hardware Mapping

- **LSH Hasher**: 16 comparators, ~200 LUTs
- **Cache Table**: 256 entries × 70 bits = 2.2KB SRAM
- **Depth Corrector**: ~150 LUTs, 4 DSPs
- **Total**: ~1,500 LUTs, 8 DSPs, 2.2KB SRAM

## Usage

```python
from fsdr import FSDRProcessor, FSDRConfig

config = FSDRConfig(feature_dim=128)
processor = FSDRProcessor(config, depth_candidates)

result = processor.process_pixel(feature, pixel_coord, cost_fn, prob_fn)
print(f"Depth: {result.depth}, Source: {result.source}")
```

## Integration

See `docs/fsdr-architecture.md` for detailed architecture and `docs/multi-model-integration.md` for model-specific usage.
