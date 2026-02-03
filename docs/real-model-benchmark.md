# Real Model Benchmark Guide

Running SCARF benchmarks with real Transplat model and RE10K dataset.

## Overview

This guide covers running SCARF benchmarks using:
- Real Transplat model checkpoint
- RE10K test dataset
- Actual rendered images with quality metrics

## Prerequisites

### Model Checkpoint

Download or link the Transplat RE10K checkpoint:

```bash
# Option 1: Symlink existing checkpoint
ln -s /path/to/checkpoints/re10k.ckpt ./data/checkpoints/re10k.ckpt

# Option 2: Download from wandb (if available)
# wandb artifact get --root ./data/checkpoints/ <artifact-path>
```

**Expected location**: `data/checkpoints/re10k.ckpt` (~500MB)

### RE10K Dataset

Link or download the RE10K test dataset:

```bash
# Symlink existing dataset
ln -s /path/to/datasets/re10k ./data/re10k

# Directory structure expected:
# data/re10k/
#   test/
#     *.torch  (scene chunks)
#   index.json
```

**Expected location**: `data/re10k/` (~10GB for test set)

### Hardware Requirements

- **GPU**: NVIDIA GPU with >= 8GB VRAM
- **CPU**: Supported but significantly slower
- **Disk**: ~20GB for output images

## Installation

Install additional dependencies:

```bash
pip install hydra-core omegaconf pytorch-lightning einops jaxtyping
```

## Usage

### Basic Real Inference

```bash
python scripts/run_re10k_benchmark.py \
    --real \
    --model-path data/checkpoints/re10k.ckpt \
    --dataset-path data/re10k/ \
    --output-dir outputs/real_benchmark/ \
    --num-scenes 100
```

### With SCARF Acceleration

```bash
python scripts/run_re10k_benchmark.py \
    --real \
    --model-path data/checkpoints/re10k.ckpt \
    --dataset-path data/re10k/ \
    --enable-scarf \
    --output-dir outputs/scarf_benchmark/
```

### Save Rendered Images

```bash
python scripts/run_re10k_benchmark.py \
    --real \
    --model-path data/checkpoints/re10k.ckpt \
    --dataset-path data/re10k/ \
    --save-images \
    --output-dir outputs/rendered/
```

### Subset Testing (Quick Validation)

```bash
# Run on first 10 scenes only
python scripts/run_re10k_benchmark.py \
    --real \
    --model-path data/checkpoints/re10k.ckpt \
    --dataset-path data/re10k/ \
    --num-scenes 10
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--real` | flag | False | Enable real inference mode |
| `--model-path` | str | None | Path to model checkpoint |
| `--dataset-path` | str | None | Path to RE10K dataset |
| `--output-dir` | str | `outputs/benchmark/` | Output directory |
| `--num-scenes` | int | 100 | Number of scenes to process |
| `--save-images` | flag | False | Save rendered images |
| `--enable-scarf` | flag | False | Enable SCARF acceleration |
| `--device` | str | `cuda` | Device (`cuda` or `cpu`) |

## Output Structure

```
outputs/real_benchmark/
├── benchmark_report.json     # Full metrics report
├── benchmark_report.md       # Human-readable summary
├── rendered/                 # Rendered images (if --save-images)
│   ├── scene_0001/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── scene_0002/
│   └── ...
└── logs/
    └── inference.log
```

## Metrics

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| PSNR | Peak Signal-to-Noise Ratio | > 24 dB |
| SSIM | Structural Similarity | > 0.85 |
| LPIPS | Learned Perceptual Similarity | < 0.15 |

### Performance Metrics

| Metric | Description |
|--------|-------------|
| Cycles/pixel | Hardware cycle estimate |
| FSDR hit rate | Cache hit percentage |
| Memory reduction | Memory access reduction |
| Speedup | Baseline vs SCARF speedup |

## Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size (if configurable)
2. Use `--device cpu` for CPU inference
3. Process fewer scenes with `--num-scenes 10`

### Checkpoint Not Found

```
FileNotFoundError: Checkpoint not found at data/checkpoints/re10k.ckpt
```

**Solution:** Ensure checkpoint is properly symlinked or downloaded.

### Dataset Loading Error

```
IndexError: No chunks found in data/re10k/test/
```

**Solution:** Verify dataset structure matches expected format.

### Import Errors

```
ModuleNotFoundError: No module named 'hydra'
```

**Solution:** Install missing dependencies:
```bash
pip install hydra-core omegaconf pytorch-lightning
```

## Integration with Transplat

The real benchmark integrates with transplat's inference pipeline:

```python
from benchmark import TransplatRunner

runner = TransplatRunner(
    checkpoint_path='data/checkpoints/re10k.ckpt',
    dataset_root='data/re10k/',
    device='cuda',
)

# Run single scene inference
result = runner.run_inference(batch, enable_scarf=False)

# Access results
print(f"PSNR: {result.psnr:.2f} dB")
print(f"SSIM: {result.ssim:.4f}")

# Save rendered image
save_image(result.rendered, 'output.png')
```

## SCARF Integration

Enable SCARF hooks for accelerated inference:

```python
from benchmark import TransplatRunner, SCARFHooks
from fsdr import FSDRProcessor
from dsu import DSUProcessor
from ggu import GGUProcessor

# Create SCARF processors
hooks = SCARFHooks(
    fsdr_processor=FSDRProcessor(...),
    dsu_processor=DSUProcessor(...),
    ggu_processor=GGUProcessor(...),
)

runner = TransplatRunner(
    checkpoint_path='data/checkpoints/re10k.ckpt',
    dataset_root='data/re10k/',
    scarf_hooks=hooks,
)

# Run with SCARF acceleration
result = runner.run_inference(batch, enable_scarf=True)
print(f"FSDR hit rate: {result.fsdr_stats['hit_rate']:.1%}")
```

## Related Documentation

- `docs/benchmark-guide.md` - Simulation mode benchmarking
- `docs/fsdr-architecture.md` - FSDR design details
- `docs/multi-model-integration.md` - Future multi-model support
