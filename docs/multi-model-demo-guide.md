# Multi-Model Demo Guide

This guide explains how to run SCARF with different 3D Gaussian Splatting
models. Use `ARTIFACT_EVALUATION.md` for environment setup and the quick
workflow.

## Supported Models

| Model | Description | Default Dataset |
|-------|-------------|-----------------|
| **TranSplat** | Transformer-based with depth priors | RE10K |
| **MVSplat** | Multi-view stereo with cost volume | RE10K |
| **DepthSplat** | DINOv2 features with 3-view support | RE10K, DL3DV |

## Quick Start

### Running with Different Models

```bash
# TranSplat (default)
python scripts/demo.py --model transplat

# MVSplat
python scripts/demo.py --model mvsplat

# DepthSplat
python scripts/demo.py --model depthsplat
```

## Checkpoint Requirements

### TranSplat

```
SCARF/transplat/checkpoints/
├── re10k.ckpt          # Main checkpoint
└── depth_anything_v2_vits.pth  # Depth prior model
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
# ViT-S Re10K checkpoint used by the default 384-wide configuration
wget https://huggingface.co/haofeixu/depthsplat/resolve/ed5116c11f7932fff3714989e2b65716730f6f87/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
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

## Execution Flow

The demo runs the selected model encoder, builds depth-conditioned Gaussian
descriptors, and applies the enabled FSDR and SAES mechanisms. The same command
shape is used for all three adapters; model-specific configuration is loaded by
the selected adapter.

## Configuration

The runner loads one mechanism configuration for a command. Keep the same
configuration when comparing model runs, and use the `--no-fsdr` and
`--no-saes` switches when an ablation is needed.

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

## Execution Order

```
[Input images]
      |
      v
[Model forward with dense S2/S3]
      |
      v
[Full Gaussian descriptors]
      |
      v
[SAES route and materialization]
      |
      v
[Gaussian output]
```

## See Also

- [FSDR + SAES Mechanisms](fsdr-saes-mechanisms.md)
- [Pipeline Architecture](pipeline-architecture.md)
