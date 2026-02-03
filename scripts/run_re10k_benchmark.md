# run_re10k_benchmark.py

SCARF RE10K benchmark command-line interface.

## Overview

Runs SCARF inference benchmarks on the RE10K dataset, comparing baseline and SCARF-accelerated performance with quality metrics.

## Usage

```bash
python scripts/run_re10k_benchmark.py [options]
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | None | Path to model checkpoint |
| `--dataset-path` | str | None | Path to RE10K dataset |
| `--output-dir` | str | `outputs/benchmark/` | Output directory for reports |
| `--num-scenes` | int | 100 | Number of scenes to benchmark |
| `--warmup-scenes` | int | 5 | Number of warmup scenes |
| `--per-scene` | flag | False | Collect per-scene metrics |
| `--simulation` | flag | False | Force simulation mode |
| `--psnr-threshold` | float | 0.5 | Maximum acceptable PSNR drop (dB) |
| `--ssim-threshold` | float | 0.01 | Maximum acceptable SSIM drop |

## Examples

### Simulation Mode (No Model)

```bash
python scripts/run_re10k_benchmark.py \
    --simulation \
    --num-scenes 50 \
    --output-dir outputs/benchmark/
```

### Real Inference Mode

```bash
python scripts/run_re10k_benchmark.py \
    --model-path checkpoints/re10k.ckpt \
    --dataset-path datasets/re10k/ \
    --num-scenes 100 \
    --output-dir outputs/benchmark/
```

### With Per-Scene Metrics

```bash
python scripts/run_re10k_benchmark.py \
    --model-path checkpoints/re10k.ckpt \
    --dataset-path datasets/re10k/ \
    --per-scene \
    --output-dir outputs/benchmark/
```

## Output

The script generates:

1. **Console Output**:
   - Progress bars for baseline and SCARF runs
   - Summary metrics (PSNR, SSIM, speedup)
   - Pass/fail status based on quality thresholds

2. **Files** (in `--output-dir`):
   - `benchmark_report.json` - Full report in JSON format
   - `benchmark_report.md` - Human-readable Markdown summary

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Benchmark passed (quality acceptable) |
| 1 | Benchmark failed (quality threshold exceeded) |

## Example Output

```
============================================================
SCARF RE10K Benchmark
============================================================

Mode: SIMULATION (no real model/dataset)
Scenes: 100
Output: outputs/benchmark/

Running baseline benchmark...
Baseline: [========================================] 100/100 (100.0%)
  PSNR: 33.98 dB
  SSIM: 0.9998
  Time: 0.15s

Running SCARF benchmark...
SCARF: [========================================] 100/100 (100.0%)
  PSNR: 33.98 dB
  SSIM: 0.9998
  Time: 0.18s
  FSDR Hit Rate: 68.0%
  Memory Reduction: 65.8%

Generating comparison report...

============================================================
RESULTS
============================================================

Speedup: 2.75×
PSNR Delta: +0.000 dB
SSIM Delta: +0.0000
Quality Acceptable: YES

Report saved to: outputs/benchmark/benchmark_report.json
Markdown saved to: outputs/benchmark/benchmark_report.md

✓ Benchmark passed!
```

## Related Files

- `benchmark/benchmark_runner.py` - Core benchmark logic
- `benchmark/report_generator.py` - Report formatting
- `docs/benchmark-guide.md` - Detailed benchmark documentation
