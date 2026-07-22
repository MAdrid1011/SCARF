# Multi-Model Demo Guide

This guide explains how to run SCARF with different 3D Gaussian Splatting
models. The artifact workflows in `ARTIFACT_EVALUATION.md` provide the
corresponding structured evaluation records.

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

## Execution Model

The demo performs model S2/S3 evaluation, materializes Gaussian descriptors,
then applies SAES tile routing and materialization. It emits image metrics,
route counts, retained-descriptor counts, and the architectural event ledger.
The AE workflows bind those records to the full data-selection and hardware
measurement contracts.

## Configuration Boundary

The demo supplies one global mechanism configuration. The calibration contract
uses 24 DL3DV calibration scenes and eight disjoint DL3DV holdout scenes, and
binds the selected configuration to the resulting execution records. The
configuration is shared across models, datasets, scenes, and samples.

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
[Output image and metrics]
```

FSDR accounting is reported with the same execution record.

## See Also

- [FSDR + SAES Mechanisms](fsdr-saes-mechanisms.md)
- [Pipeline Architecture](pipeline-architecture.md)
