# Multi-Model Demo Guide

This guide explains how to run SCARF demos with different 3D Gaussian Splatting models.

## Supported Models

| Model | Description | Default Dataset |
|-------|-------------|-----------------|
| **Transplat** | Transformer-based with depth priors | RE10K |
| **MVSplat** | Multi-view stereo with cost volume | RE10K |
| **DepthSplat** | DINOv2 features with 3-view support | RE10K, DL3DV |

## Quick Start

### Running with Different Models

```bash
# Transplat (default)
python scripts/demo.py --model transplat

# MVSplat
python scripts/demo.py --model mvsplat

# DepthSplat
python scripts/demo.py --model depthsplat
```

## Checkpoint Requirements

### Transplat

```
SCARF/transplat/checkpoints/
├── re10k.ckpt          # Main checkpoint
└── depth_anything_v2_vitb.pth  # Depth prior model
```

**Download:**
- RE10K checkpoint: Follow instructions in `transplat/README.md`
- Depth Anything: Automatically downloaded on first run

### MVSplat

```
SCARF/mvsplat/checkpoints/
└── re10k.ckpt          # Main checkpoint
```

**Download:**
```bash
# From MVSplat repository
wget https://huggingface.co/donydchen/mvsplat/resolve/main/re10k.ckpt \
     -O SCARF/mvsplat/checkpoints/re10k.ckpt
```

### DepthSplat

```
SCARF/depthsplat/checkpoints/
└── re10k.ckpt          # Main checkpoint (or dl3dv.ckpt)
```

**Download:**
```bash
# From DepthSplat repository
wget https://huggingface.co/cvlab-epfl/depthsplat/resolve/main/re10k.ckpt \
     -O SCARF/depthsplat/checkpoints/re10k.ckpt
```

## Dataset Requirements

### RE10K Dataset

All three models support RE10K. Set up the dataset:

```bash
# Create symlinks (recommended)
ln -s /path/to/re10k SCARF/transplat/datasets/re10k
ln -s /path/to/re10k SCARF/mvsplat/datasets/re10k
ln -s /path/to/re10k SCARF/depthsplat/datasets/re10k
```

### DL3DV Dataset (DepthSplat only)

```bash
ln -s /path/to/dl3dv SCARF/depthsplat/datasets/dl3dv
```

## Expected Performance

### Quality Metrics (PSNR)

| Model | Baseline | With SCARF | Quality Loss |
|-------|----------|------------|--------------|
| Transplat | ~29 dB | ~27 dB | < 3 dB |
| MVSplat | ~27 dB | ~25 dB | < 3 dB |
| DepthSplat | ~28 dB | ~26 dB | < 3 dB |

### Performance Improvements

| Metric | Transplat | MVSplat | DepthSplat |
|--------|-----------|---------|------------|
| Gaussian Reduction | ~25% | ~20-30% | ~20-30% |
| Cycle Reduction | ~40% | ~35-45% | ~35-45% |
| Speedup | ~1.7x | ~1.5-2x | ~1.5-2x |

*Note: Results vary based on scene complexity and SAES threshold tuning.*

## Model-Specific Configurations

### SAES Thresholds

Each model has optimized SAES thresholds via adapters:

| Model | Early-Stop Threshold | Cov Enlarge Factor |
|-------|---------------------|-------------------|
| Transplat | 0.85 | 6.0 |
| MVSplat | 0.85 | 6.0 |
| DepthSplat | 0.92 | 6.0 |

### FSDR Configurations

| Model | Hamming Threshold | High Confidence |
|-------|------------------|-----------------|
| Transplat | 4 | 0.80 |
| MVSplat | 4 | 0.80 |
| DepthSplat | 3 | 0.85 |

## Troubleshooting

### Common Issues

#### 1. Checkpoint Not Found

```
Error: Checkpoint not found at SCARF/mvsplat/checkpoints/re10k.ckpt
```

**Solution:** Download the checkpoint following instructions above.

#### 2. Dataset Not Found

```
Error: Dataset path does not exist
```

**Solution:** Create symlinks to your dataset directories.

#### 3. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce image resolution or use CPU:
```bash
python scripts/demo.py --model mvsplat --device cpu
```

#### 4. Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solution:** Ensure submodules are initialized:
```bash
cd SCARF
git submodule update --init --recursive
```

### Getting Help

1. Check model-specific README files in submodule directories
2. Review adapter configurations in `SCARF/adapters/`
3. Open an issue on GitHub with error logs

## Architecture Overview

```
SCARF Demo Pipeline:

[Input Images]
      │
      ▼
[Model Backbone]  ← Transplat/MVSplat/DepthSplat encoder
      │
      ▼
[SCARF SAES]      ← Progressive early-stopping
      │
  ┌───┴───┐
  ▼       ▼
Early   Continue
Stop    Tiles
  │       │
  │       ▼
  │   [FSDR]     ← Depth reuse optimization
  │       │
  └───┬───┘
      ▼
[SCARF GGU]      ← Gaussian generation
      │
      ▼
[Model Decoder]  ← Original renderer
      │
      ▼
[Output Image + Metrics]
```

## See Also

- [SAES Architecture](saes-architecture.md)
- [FSDR Architecture](fsdr-architecture.md)
- [Benchmark Guide](benchmark-guide.md)
