# Multi-Model Demo Guide

This guide explains how to run functional SCARF demos with different 3D Gaussian
Splatting models. The commands support integration and diagnostic inspection.
They are not a performance benchmark or paper-evidence path.

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

## Demo Output Boundary

The current demo completes dense model S2/S3 work and materializes full Gaussian
descriptors before it applies SAES. SAES then classifies tiles and constructs
route and materialization diagnostics from cloned dense descriptors. It does not
verify sparse S2/S3 execution or measure an SAES speedup.

The output may include image metrics, route counts, retained descriptor counts,
and analytic accounting. These are diagnostic values. Gaussian reduction,
derived S2-evaluation counts, analytic cycles, and abstract stage-event
schedules do not establish physical sparse work, RTL timing, or performance.

## Configuration Boundary

The demo supplies checked-in defaults automatically. They are not optimized
per-model SAES thresholds and they cannot be tuned per model, dataset, scene, or
sample for a claim. The current global mechanism configuration is preregistered
and has no selected tuple.

The author-side calibration contract reserves 24 DL3DV training scenes and eight
disjoint DL3DV holdout scenes. It must select one global tuple on the training
split and validate that frozen tuple on the holdout split before any quality,
work-reduction, timing, or speed claim. Re10K and ACID calibration flows remain
Functional regression only and cannot replace that DL3DV contract.

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

## Current Execution Order

```
Current demo execution:

[Input images]
      |
      v
[Model forward with dense S2/S3]
      |
      v
[Full Gaussian descriptors]
      |
      v
[SAES route and materialization diagnostic]
      |
      v
[Output image and diagnostic metrics]
```

FSDR accounting is reported separately and does not make SAES a pre-S2/S3
execution path.

## See Also

- [FSDR + SAES Mechanisms](fsdr-saes-mechanisms.md)
- [Pipeline Architecture](pipeline-architecture.md)
