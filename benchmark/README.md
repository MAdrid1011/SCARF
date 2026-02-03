# Benchmark Module

Performance and quality benchmarking for SCARF accelerator.

## Overview

This module provides infrastructure for:
- Cycle-level performance measurement
- Quality metrics (PSNR/SSIM) validation
- Comparison between baseline and SCARF-accelerated inference

## Key Files

| File | Description |
|------|-------------|
| `cycle_counter.py` | Hardware cycle counting infrastructure |
| `quality_validator.py` | PSNR/SSIM computation and validation |
| `report_generator.py` | Benchmark report generation |
| `benchmark_runner.py` | End-to-end benchmark orchestration |

## Usage

```python
from benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    model_path='checkpoints/re10k.ckpt',
    dataset_path='datasets/re10k/',
    output_dir='outputs/benchmark/',
)

# Run comparison
baseline = runner.run_baseline(num_scenes=100)
scarf = runner.run_scarf(num_scenes=100)
report = runner.compare_results(baseline, scarf)

# Save report
runner.save_report(report, 'benchmark_report.json')
```

## Cycle Model

Based on SCARF hardware architecture:
- FSDR: 3-8 cycles (depending on path)
- DSU: 20 cycles per depth candidate
- GGU: 10 cycles per Gaussian
- SAES: 15 cycles per tile

See `docs/benchmark-guide.md` for detailed cycle breakdown.
