# SCARF Benchmark Guide

Comprehensive guide for running SCARF inference benchmarks on RE10K dataset.

## Overview

The SCARF benchmark system measures:
1. **Performance** - Hardware cycle counts for each component
2. **Quality** - PSNR/SSIM metrics compared to baseline
3. **Efficiency** - Memory access reduction from FSDR optimization

## Quick Start

```bash
# Run benchmark with default settings
python scripts/run_re10k_benchmark.py \
    --model-path checkpoints/re10k.ckpt \
    --dataset-path datasets/re10k/ \
    --output-dir outputs/benchmark/

# Run with specific number of scenes
python scripts/run_re10k_benchmark.py \
    --model-path checkpoints/re10k.ckpt \
    --num-scenes 50 \
    --output-dir outputs/benchmark/
```

## Hardware Cycle Model

SCARF estimates hardware cycles based on the architecture design. Each operation has a fixed cycle count determined by:
- Number of arithmetic operations
- Memory access patterns
- Pipeline depth

### Cycle Estimates by Component

#### FSDR (Feature-Similarity Depth Reuse)

| Operation | Cycles | Hardware Resources | Notes |
|-----------|--------|-------------------|-------|
| LSH hash | 3 | 16 comparators | 16-bit signature generation |
| Cache lookup | 5 | 256 Hamming comparators | Parallel semantic search |
| Direct reuse | 1 | Pass-through | No computation |
| Interpolation | 4 | 2 DSPs | Weighted blend |
| Light verify | 3-7 | Reuses DSU | Local depth search |

**Memory accesses:**
- Cache hit: 1 read (70 bits)
- Cache miss: 32 reads (feature sampling) + 1 write (cache update)

#### DSU (Depth Search Unit)

| Operation | Cycles | Hardware Resources | Notes |
|-----------|--------|-------------------|-------|
| Project | 8 | 8 DSPs | 3D-2D matrix multiply |
| Bilinear sample | 4 | 8 DSPs | 4-point interpolation |
| Cost compute | 2 | 4 DSPs | Dot product |
| Softmax | 6 | LUT + divider | Exp approximation |

**Full search:** 32 depths × (8 + 4 + 2) + 6 = 454 cycles

#### GGU (Gaussian Generation Unit)

| Operation | Cycles | Hardware Resources | Notes |
|-----------|--------|-------------------|-------|
| Position | 2 | 4 DSPs | Unproject + depth scale |
| Covariance | 5 | 12 DSPs | Quaternion to matrix |
| SH rotation | 3 | 4 DSPs | Degree-1 rotation |

**Per Gaussian:** 10 cycles total

#### SAES (Scene-Adaptive Early Stopping)

| Operation | Cycles | Hardware Resources | Notes |
|-----------|--------|-------------------|-------|
| Probe sample | 4 | 5 lookups | Corner + center pixels |
| Similarity | 6 | 8 DSPs | 4 metric computation |
| Decision | 1 | Comparators | Threshold check |
| Merge | 4 | 4 DSPs | Weighted average |

## Performance Expectations

### Baseline (No SCARF)

For each pixel:
- DSU full search: 454 cycles
- GGU generation: 10 cycles
- **Total: 464 cycles/pixel**

### With SCARF Optimization

Assuming 68% FSDR cache hit rate:
- FSDR overhead: 8 cycles (hash + lookup)
- Cache hit path: 1-4 cycles (reuse/interpolation)
- Cache miss path: 454 cycles (full search)
- GGU generation: 10 cycles

**Average: ~180 cycles/pixel (2.5× speedup)**

### Memory Access Reduction

- Baseline: 32 feature reads per pixel
- With FSDR (68% hit): ~10 reads per pixel average
- **Reduction: ~68%**

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 × log10(MAX² / MSE)
```

- Measures pixel-level reconstruction accuracy
- Higher is better (typical range: 20-40 dB)
- Acceptable degradation: < 0.5 dB

### SSIM (Structural Similarity Index)

```
SSIM = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
```

- Measures structural similarity
- Range: 0-1 (1 = identical)
- Acceptable degradation: < 0.01

## Configuration Options

### BenchmarkRunner Configuration

```python
config = {
    # SCARF settings
    'enable_fsdr': True,
    'enable_saes': False,  # Optional
    'fsdr_config': {
        'hamming_threshold': 4,
        'high_confidence_threshold': 0.8,
    },
    
    # Benchmark settings
    'num_scenes': 100,
    'warmup_scenes': 5,
    'collect_per_scene': True,
}
```

### Quality Thresholds

```python
quality_config = {
    'psnr_drop_threshold': 0.5,  # dB
    'ssim_drop_threshold': 0.01,
    'fail_on_threshold_violation': False,
}
```

## Output Format

### JSON Report

```json
{
    "metadata": {
        "timestamp": "2026-02-03T12:00:00",
        "model": "re10k.ckpt",
        "num_scenes": 100
    },
    "baseline": {
        "psnr_mean": 25.5,
        "ssim_mean": 0.92,
        "cycles_per_pixel": 464
    },
    "scarf": {
        "psnr_mean": 25.3,
        "ssim_mean": 0.915,
        "cycles_per_pixel": 180,
        "cycle_breakdown": {
            "fsdr": {"hash": 3, "lookup": 5, ...},
            "dsu": {"project": 8, ...},
            "ggu": {"position": 2, ...}
        },
        "fsdr_stats": {
            "hit_rate": 0.68,
            "memory_reduction": 0.68
        }
    },
    "comparison": {
        "psnr_delta": -0.2,
        "ssim_delta": -0.005,
        "speedup": 2.58,
        "quality_acceptable": true
    }
}
```

### Markdown Report

The benchmark also generates a human-readable markdown summary with:
- Configuration overview
- Performance comparison table
- Quality metrics comparison
- Per-scene breakdown (optional)

## Interpreting Results

### Good Results

- Speedup > 2× with minimal quality loss
- FSDR hit rate > 60%
- PSNR drop < 0.5 dB
- SSIM drop < 0.01

### Warning Signs

- Low FSDR hit rate (< 50%) - May need threshold tuning
- High quality degradation - Check FSDR/SAES thresholds
- No speedup - Check if SCARF is enabled

### Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Low hit rate | Threshold too tight | Increase `hamming_threshold` |
| Quality loss | Aggressive optimization | Decrease early-stop thresholds |
| No speedup | SCARF not enabled | Check `enable_fsdr=True` |

## Integration with Transplat

The benchmark integrates with transplat's existing infrastructure:

```python
# In encoder_trans.py
from SCARF.integration import Accelerator

class EncoderTrans:
    def __init__(self, ...):
        self.scarf_accelerator = Accelerator(
            model_type='transplat',
            enable_fsdr=True,
        )
    
    def forward(self, ..., enable_scarf=False):
        if enable_scarf:
            # Use SCARF-accelerated depth prediction
            result = self.scarf_accelerator.process_pixel(...)
```

## Dependencies

- `torch >= 1.9.0`
- `numpy >= 1.20.0`
- `scikit-image >= 0.18.0` (for SSIM)
- transplat model checkpoint
- RE10K dataset

## Related Documentation

- `docs/fsdr-architecture.md` - FSDR design details
- `docs/dsu-architecture.md` - DSU implementation
- `docs/ggu-architecture.md` - GGU implementation
- `docs/hardware-resource-summary.md` - Resource estimates
