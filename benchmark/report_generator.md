# ReportGenerator

Benchmark report generation for SCARF accelerator.

## Overview

Generates structured benchmark reports comparing baseline and SCARF-accelerated inference, supporting both JSON and Markdown output formats.

## External Interface

### BenchmarkReport

```python
@dataclass
class BenchmarkReport:
    name: str                        # 'baseline' or 'scarf'
    quality: QualityMetrics          # Quality metrics
    cycles: Optional[Dict]           # Cycle breakdown (SCARF only)
    fsdr_stats: Optional[Dict]       # FSDR statistics (SCARF only)
    num_scenes: int                  # Number of scenes processed
    
    def to_dict(self) -> Dict
```

### ComparisonReport

```python
@dataclass
class ComparisonReport:
    baseline: BenchmarkReport        # Baseline report
    scarf: BenchmarkReport           # SCARF report
    speedup: float                   # Cycle speedup factor
    psnr_delta: float                # PSNR difference (dB)
    ssim_delta: float                # SSIM difference
    quality_acceptable: bool         # Whether quality meets threshold
    metadata: Dict                   # Timestamp, model name, etc.
    
    def to_dict(self) -> Dict
```

### ReportGenerator

```python
class ReportGenerator:
    def __init__(self, model_name: str = 'transplat')
```

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_baseline_report(quality, num_scenes)` | Generate baseline benchmark report |
| `generate_scarf_report(quality, cycles, fsdr_stats, num_scenes)` | Generate SCARF benchmark report |
| `generate_comparison(baseline, scarf, baseline_cycles_per_pixel, psnr_threshold, ssim_threshold)` | Generate comparison report |
| `to_json(report)` | Convert report to JSON string |
| `save_json(report, filepath)` | Save report to JSON file |
| `to_markdown(report)` | Convert report to Markdown string |

## Usage Example

```python
from benchmark import ReportGenerator, CycleCounter
from benchmark.quality_validator import QualityMetrics

generator = ReportGenerator(model_name='transplat')

# Generate baseline report
baseline = generator.generate_baseline_report(
    quality=QualityMetrics(psnr_mean=25.5, ssim_mean=0.92),
    num_scenes=100,
)

# Generate SCARF report
scarf = generator.generate_scarf_report(
    quality=QualityMetrics(psnr_mean=25.3, ssim_mean=0.915),
    cycles=cycle_counter,
    fsdr_stats={'hit_rate': 0.68, 'memory_reduction': 0.66},
    num_scenes=100,
)

# Generate comparison
comparison = generator.generate_comparison(
    baseline=baseline,
    scarf=scarf,
    baseline_cycles_per_pixel=454,
    psnr_threshold=0.5,
    ssim_threshold=0.01,
)

# Save reports
generator.save_json(comparison, 'benchmark_report.json')
print(generator.to_markdown(comparison))
```

## Output Formats

### JSON Report Structure

```json
{
    "metadata": {
        "timestamp": "2026-02-03T12:00:00",
        "model": "transplat",
        "num_scenes": 100
    },
    "baseline": {
        "name": "baseline",
        "quality": {"psnr_mean": 25.5, "ssim_mean": 0.92, ...},
        "num_scenes": 100
    },
    "scarf": {
        "name": "scarf",
        "quality": {"psnr_mean": 25.3, "ssim_mean": 0.915, ...},
        "cycles": {"total_cycles": 180000, "by_component": {...}},
        "fsdr_stats": {"hit_rate": 0.68, "memory_reduction": 0.66},
        "num_scenes": 100
    },
    "comparison": {
        "speedup": 2.52,
        "psnr_delta": -0.2,
        "ssim_delta": -0.005,
        "quality_acceptable": true
    }
}
```

### Markdown Report

The Markdown report includes:
- Header with model name, timestamp, scene count
- Performance comparison table (PSNR, SSIM)
- Speedup and quality acceptance status
- FSDR statistics (hit rate, memory reduction)
- Cycle breakdown by component

## Internal Helpers

| Function | Description |
|----------|-------------|
| `convert_numpy(obj)` | Convert numpy types to Python native for JSON serialization |
| `recursive_convert(d)` | Recursively convert all numpy types in nested structures |

## Related Files

- `benchmark_runner.py` - Uses ReportGenerator for saving results
- `cycle_counter.py` - Provides cycle data for reports
- `quality_validator.py` - Provides quality metrics for reports
