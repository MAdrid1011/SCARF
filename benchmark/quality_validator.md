# QualityValidator

PSNR/SSIM quality metric validation for SCARF benchmarks.

## Overview

Computes image quality metrics (PSNR and SSIM) between rendered and ground truth images, validates against acceptable degradation thresholds, and supports baseline comparison.

## External Interface

### QualityMetrics

```python
@dataclass
class QualityMetrics:
    psnr_mean: float      # Mean PSNR across samples (dB)
    psnr_std: float       # Standard deviation of PSNR
    ssim_mean: float      # Mean SSIM across samples (0-1)
    ssim_std: float       # Standard deviation of SSIM
    num_samples: int      # Number of samples
    
    def to_dict(self) -> Dict
```

### QualityValidator

```python
class QualityValidator:
    def __init__(
        self,
        baseline_psnr: Optional[float] = None,
        baseline_ssim: Optional[float] = None,
    )
```

**Parameters:**
- `baseline_psnr`: Baseline PSNR for comparison (optional)
- `baseline_ssim`: Baseline SSIM for comparison (optional)

**Methods:**

| Method | Description |
|--------|-------------|
| `record_sample(rendered, ground_truth)` | Record quality metrics for a sample |
| `get_metrics()` | Get aggregated quality metrics |
| `check_quality_threshold(psnr_drop_threshold, ssim_drop_threshold)` | Check if degradation is acceptable |
| `get_comparison()` | Get comparison with baseline metrics |
| `reset()` | Clear all recorded samples |

## Usage Example

```python
from benchmark import QualityValidator

# Create validator with baseline for comparison
validator = QualityValidator(baseline_psnr=25.5, baseline_ssim=0.92)

# Record samples
for rendered, gt in samples:
    validator.record_sample(rendered, gt)

# Get metrics
metrics = validator.get_metrics()
print(f"PSNR: {metrics.psnr_mean:.2f} ± {metrics.psnr_std:.2f} dB")
print(f"SSIM: {metrics.ssim_mean:.4f} ± {metrics.ssim_std:.4f}")

# Check against threshold
if validator.check_quality_threshold(psnr_drop_threshold=0.5, ssim_drop_threshold=0.01):
    print("Quality acceptable!")
else:
    print("Quality degradation exceeds threshold!")

# Get detailed comparison
comparison = validator.get_comparison()
print(f"PSNR delta: {comparison['delta']['psnr']:+.2f} dB")
```

## Internal Helpers

| Function | Description |
|----------|-------------|
| `_compute_psnr(rendered, ground_truth)` | Compute PSNR between two images |
| `_compute_ssim(rendered, ground_truth)` | Compute simplified global SSIM |

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 × log10(MAX² / MSE)
```

- Higher is better (typical range: 20-40 dB)
- Identical images: infinite PSNR
- Acceptable degradation threshold: < 0.5 dB

### SSIM (Structural Similarity Index)

```
SSIM = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
```

- Range: 0-1 (1 = identical)
- Acceptable degradation threshold: < 0.01

**Note**: The current SSIM implementation uses a simplified global computation. For production use, consider using `skimage.metrics.structural_similarity` for the standard windowed version.

## Input Format

Images should be torch tensors:
- Shape: `[C, H, W]` or `[H, W, C]`
- Value range: `[0, 1]`
- Supports both CPU and CUDA tensors (automatically moved to CPU)

## Related Files

- `benchmark_runner.py` - Uses QualityValidator for both baseline and SCARF runs
- `report_generator.py` - Formats quality metrics in reports
