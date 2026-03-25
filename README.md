# SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator with Semantic Reuse and Fused Dataflow

SCARF is a hardware-realizable accelerator for depth-guided generalizable 3D Gaussian Splatting (G-3DGS) encoders, co-designed with a scene-adaptive dataflow that exploits semantic similarity in 2D feature space (FSDR) and geometric continuity in 3D space (SAES).

## Key Features

### End-to-End Hardware Simulation
- Full pipeline runs through ASIC hardware unit simulators by default
- Feature Extractor → Depth Predictor → GGU, each stage's output feeds the next
- Zero `SW_FALLBACK` — every `F.*` / `torch.*` operation is routed through a hardware unit

### Hardware Units (encoder/)

| Unit | Operations |
|------|-----------|
| **ConvEngine** | Conv2d (k=1,3,5,7,9,14), transposed conv, grouped conv |
| **GEMMUnit** | Matrix multiplication, batched matmul |
| **ActivationUnit** | ReLU, GELU, SiLU, Sigmoid, Softplus |
| **NormalizationUnit** | LayerNorm, BatchNorm, InstanceNorm, GroupNorm (configurable eps) |
| **BilinearUnit** | Bilinear / nearest / bicubic interpolation, grid_sample |
| **SoftmaxUnit** | Softmax with optional temperature (LUT-based exp) |
| **PoolingUnit** | Average pooling, max pooling, adaptive average pooling |
| **PadUnit** | Constant / reflect / replicate / circular padding |
| **DeformableAttentionUnit** | Multi-scale deformable attention |

### SAES (Scene-Adaptive Early Sparsification)
- Analyzes feature similarity within tiles
- Skips depth search for homogeneous regions
- Reduces Gaussian count by ~30%

### FSDR (Feature Similarity Depth Reuse)
- Caches depth results using LSH-based feature signatures
- Reuses cached depths for similar pixels
- Reduces memory access by ~87%

## Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| **TranSplat** | ✅ Full | Transformer-based with depth priors |
| **MVSplat** | ✅ Full | Multi-view stereo with cost volume |
| **DepthSplat** | ✅ Full | DINOv2 features with 3-view support |

See [Multi-Model Demo Guide](docs/multi-model-demo-guide.md) for detailed instructions.

## Repository Structure

```
SCARF/
├── encoder/               # Hardware unit simulators (ASIC building blocks)
│   ├── conv_engine.py         # Convolution engine (systolic array)
│   ├── gemm_unit.py           # Matrix multiplication unit
│   ├── activation_unit.py     # ReLU/GELU/SiLU/Sigmoid/Softplus
│   ├── normalization_unit.py  # LN/BN/IN/GN
│   ├── bilinear_unit.py       # Interpolation & grid_sample
│   ├── softmax_unit.py        # Softmax with temperature
│   ├── pooling_unit.py        # Avg/max pooling
│   ├── pad_unit.py            # Padding modes
│   └── deformable_attention_unit.py  # MS deformable attention
├── feature_extractor/     # HW-simulated feature extraction
│   ├── cnn_simulator.py       # CNN backbone simulator
│   ├── transformer_simulator.py # Transformer layers simulator
│   ├── vit_simulator.py       # ViT (DINOv2) simulator
│   ├── transplat_extractor.py
│   ├── mvsplat_extractor.py
│   └── depthsplat_extractor.py
├── depth_predictor/       # HW-simulated depth prediction
│   ├── hw_depth_predictor.py  # Central depth predictor (dispatches per model)
│   ├── cost_volume_sim.py     # Cost volume construction
│   ├── depth_head_sim.py      # Softmax depth regression
│   ├── unet_sim.py            # U-Net refinement
│   ├── transplat_predictor.py
│   ├── mvsplat_predictor.py
│   └── depthsplat_predictor.py
├── ggu/                   # Gaussian Generation Unit simulator
├── fsdr/                  # Feature Similarity Depth Reuse
├── saes/                  # Scene-Adaptive Early Sparsification
├── adapters/              # Model-specific adapters
├── integration/           # Model loader & bundle utilities
├── scripts/
│   └── demo.py            # Complete end-to-end demo
├── docs/                  # Documentation
├── transplat/             # TranSplat submodule
├── mvsplat/               # MVSplat submodule
└── depthsplat/            # DepthSplat submodule
```

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/MAdrid1011/SCARF.git
cd SCARF

# If already cloned without --recursive:
git submodule update --init --recursive
```

### 2. Environment Setup

```bash
# Create conda environment (recommended)
conda create -n transplat python=3.10
conda activate transplat

# Install SCARF core dependencies
pip install -r requirements.txt

# Install model-specific dependencies (pick the model(s) you need)
pip install -r transplat/requirements.txt
pip install -r mvsplat/requirements.txt
pip install -r depthsplat/requirements.txt
```

### 3. Setup Data

Copy (or symlink) checkpoints and datasets into each submodule directory:

```bash
# --- TranSplat ---
cp -r /path/to/checkpoints/re10k.ckpt          transplat/checkpoints/
cp -r /path/to/checkpoints/depth_anything_v2_vits.pth transplat/checkpoints/
cp -r /path/to/datasets/re10k                   transplat/datasets/

# --- MVSplat ---
cp -r /path/to/checkpoints/re10k.ckpt          mvsplat/checkpoints/
cp -r /path/to/datasets/re10k                   mvsplat/datasets/

# --- DepthSplat ---
cp -r /path/to/checkpoints/depthsplat_re10k.ckpt depthsplat/checkpoints/
cp -r /path/to/datasets/re10k                   depthsplat/datasets/
```

> **Tip:** If you want to save disk space, use symlinks instead of `cp -r`:
> ```bash
> ln -s /path/to/checkpoints transplat/checkpoints
> ln -s /path/to/datasets    transplat/datasets
> ```

Required files per model:

| Model | Checkpoints | Dataset |
|-------|------------|---------|
| TranSplat | `re10k.ckpt`, `depth_anything_v2_vits.pth` | `re10k/` |
| MVSplat | `re10k.ckpt` | `re10k/` |
| DepthSplat | `depthsplat_re10k.ckpt` | `re10k/` |

### 4. Run Demo

```bash
# Run with TranSplat (default) — full HW simulation pipeline
python scripts/demo.py --model transplat

# Run with MVSplat
python scripts/demo.py --model mvsplat

# Run with DepthSplat
python scripts/demo.py --model depthsplat

# Disable FSDR and/or SAES optimizations (pure HW sim only)
python scripts/demo.py --model transplat --no-fsdr --no-saes
```

## Architecture

The end-to-end hardware simulation pipeline:

```
Input Images + Camera Params
       ↓
┌──────────────────────────────────┐
│  HW Feature Extractor            │  CNN / Transformer / ViT (DINOv2)
│  (all ops → ConvEngine, GEMM,    │  routed through hardware units
│   Activation, Norm, Softmax…)    │
└──────────────┬───────────────────┘
               ↓ features
┌──────────────────────────────────┐
│  HW Depth Predictor              │  Cost volume, U-Net, depth head
│  (Conv, GEMM, Pooling, Pad,     │  all routed through hardware units
│   BilinearUnit, SoftmaxUnit…)   │
└──────────────┬───────────────────┘
               ↓ depths
┌──────────────────────────────────┐
│  HW GGU (Gaussian Generation)    │  Opacity, covariance, color
│  (Activation, Sigmoid, Conv…)    │  all routed through hardware units
└──────────────┬───────────────────┘
               ↓ Gaussians + Cycle Counts
        [Optional SAES / FSDR]
               ↓
        Model Decoder → Rendered Image
```

When SAES and FSDR are enabled (default), they sit between Depth Predictor and GGU:

```
┌──────────────────────────────────┐
│  SCARF SAES                      │  Tile-level sparsification decisions
│  → Skip DSU for homogeneous tiles│
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  SCARF FSDR                      │  LSH-based depth cache
│  → Reuse cached depth results    │
└──────────────┬───────────────────┘
```

## Documentation

- [Multi-Model Demo Guide](docs/multi-model-demo-guide.md)
- [FSDR + SAES Mechanisms](docs/fsdr-saes-mechanisms.md)
- [GGU Architecture](docs/ggu-architecture.md)
- [Pipeline Architecture](docs/pipeline-architecture.md)
- [Architecture (中文)](docs/architecture-cn.md)

Module-level documentation:
- [feature_extractor/README.md](feature_extractor/README.md) — Feature extraction
- [depth_predictor/README.md](depth_predictor/README.md) — Depth prediction

## License

MIT License

## Related Projects

- [TranSplat](https://github.com/xingyoujun/transplat) - Original TranSplat implementation (AAAI 2025)

## Citation

If you use SCARF in your research, please cite:

```bibtex
@misc{scarf2025,
  title={SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator with Semantic Reuse and Fused Dataflow},
  author={...},
  year={2025}
}
```
