# BenchmarkRunner

End-to-end benchmark orchestration for SCARF accelerator.

## Overview

The `BenchmarkRunner` class orchestrates baseline and SCARF-accelerated benchmark runs, collecting quality metrics (PSNR/SSIM) and hardware cycle counts for performance comparison.

## External Interface

### BenchmarkRunner

```python
class BenchmarkRunner:
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        scarf_config: Optional[Dict] = None,
        output_dir: str = 'outputs/benchmark/',
    )
```

**Parameters:**
- `model_path`: Path to model checkpoint (None for simulation mode)
- `dataset_path`: Path to RE10K dataset
- `scarf_config`: SCARF configuration dict with keys:
  - `enable_fsdr`: bool - Enable FSDR optimization
  - `enable_saes`: bool - Enable SAES optimization
- `output_dir`: Output directory for benchmark reports

**Methods:**

| Method | Description |
|--------|-------------|
| `run_baseline(num_scenes, collect_per_scene, progress_callback)` | Run baseline benchmark without SCARF |
| `run_scarf(num_scenes, collect_per_scene, progress_callback)` | Run SCARF-accelerated benchmark |
| `compare_results(baseline, scarf, psnr_threshold, ssim_threshold)` | Compare baseline and SCARF results |
| `save_report(report, filename)` | Save comparison report to JSON and Markdown |

### BaselineResult

```python
@dataclass
class BaselineResult:
    quality: QualityMetrics      # Aggregated quality metrics
    num_scenes: int              # Number of scenes processed
    total_time_s: float          # Total execution time
    per_scene_metrics: Optional[Dict[str, QualityMetrics]]  # Per-scene breakdown
```

### SCARFResult

```python
@dataclass
class SCARFResult:
    quality: QualityMetrics      # Aggregated quality metrics
    cycles: CycleCounter         # Hardware cycle counts
    fsdr_stats: Dict             # FSDR-specific statistics
    num_scenes: int              # Number of scenes processed
    total_time_s: float          # Total execution time
    per_scene_metrics: Optional[Dict[str, QualityMetrics]]  # Per-scene breakdown
```

## Internal Helpers

| Function | Description |
|----------|-------------|
| `_generate_mock_images()` | Generate mock rendered/GT images for simulation |
| `_simulate_scarf_cycles()` | Simulate SCARF cycle counts for a scene |
| `_run_baseline_inference(scene_idx)` | Run actual baseline inference (stub) |
| `_run_scarf_inference(scene_idx)` | Run actual SCARF inference (stub) |

## Usage Example

```python
from benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    model_path='checkpoints/re10k.ckpt',
    dataset_path='datasets/re10k/',
    output_dir='outputs/benchmark/',
)

# Run benchmarks
baseline = runner.run_baseline(num_scenes=100)
scarf = runner.run_scarf(num_scenes=100)

# Compare and save
report = runner.compare_results(baseline, scarf)
runner.save_report(report, 'benchmark_report.json')
```

## Simulation Mode

When `model_path` is None or doesn't exist, the runner operates in simulation mode:
- Generates random images with small noise for quality metrics
- Simulates ~68% FSDR cache hit rate
- Uses predefined cycle constants for timing estimates

## Related Files

- `cycle_counter.py` - Hardware cycle counting
- `quality_validator.py` - PSNR/SSIM computation
- `report_generator.py` - Report formatting
