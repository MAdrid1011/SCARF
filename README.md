# SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator with Semantic Reuse and Fused Dataflow

SCARF is a hardware-realizable accelerator for depth-guided generalizable 3D Gaussian Splatting (G-3DGS) encoders, co-designed with a scene-adaptive dataflow that exploits semantic similarity in 2D feature space (FSDR) and geometric continuity in 3D space (SAES).

## MICRO 2026 Artifact Evaluation

Zenodo DOI: [10.5281/zenodo.21482385](https://doi.org/10.5281/zenodo.21482385)

The reviewer-facing setup, command mapping, and hardware scope are documented
in the [artifact evaluation guide](ARTIFACT_EVALUATION.md). The fastest CUDA
path is the packaged Functional quick workflow:

```bash
docker build -t scarf-ae:1.0.0 .
mkdir -p outputs/docker-quick
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs/docker-quick:/results" scarf-ae:1.0.0
```

The container downloads the hash-pinned MVSplat checkpoint on first use and
writes a structured result below `outputs/docker-quick/quick/`.

## Key Features

### End-to-End Hardware Simulation
- Full pipeline runs through ASIC hardware unit simulators by default
- Feature Extractor → Depth Predictor → GGU, with each stage feeding the next
- Result provenance records the simulated stages and rejects missing cycle data
  in artifact-evaluation mode

### Hardware Units (encoder/)

| Unit | Operations |
|------|-----------|
| **ConvEngine** | Conv2d (k=1,3,5,7,9,14), transposed conv, grouped conv |
| **GEMMUnit** | Matrix multiplication, batched matmul |
| **ActivationUnit** | ReLU, GELU, SiLU, Sigmoid, Softplus |
| **NormalizationUnit** | LayerNorm, BatchNorm, InstanceNorm, GroupNorm (configurable eps) |
| **BilinearUnit** | Bilinear / nearest / bicubic interpolation, `grid_sample` |
| **SoftmaxUnit** | Softmax with optional temperature (LUT-based exp) |
| **PoolingUnit** | Average pooling, max pooling, adaptive average pooling |
| **PadUnit** | Constant / reflect / replicate / circular padding |
| **DeformableAttentionUnit** | Multi-scale deformable attention |

### SAES (Scene-Adaptive Early Sparsification)
- Uses probe feature variance followed by probe-depth standard deviation to
  select L0, L1, or Full at tile granularity
- Keeps the pretrained adaptor as the source of every retained probe descriptor
- Binds retained descriptors, route decisions, and Gaussian materialization to
  the execution record

### FSDR (Feature Similarity Depth Reuse)
- Caches depth results using LSH-based feature signatures
- Narrows only a valid Hamming-matched local candidate set and otherwise falls
  back to full search
- Records cache eligibility and depth-reuse events in the same result schema

## Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| **TranSplat** | Functional adapter | Transformer-based with depth priors |
| **MVSplat** | Functional adapter | Multi-view stereo with cost volume |
| **DepthSplat** | Functional adapter | DINOv2 features with 3-view support |

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
# TranSplat and MVSplat (PyTorch 2.1 profile)
bash install.sh --profile classic --venv .venv/classic

# DepthSplat uses a separate PyTorch 2.4 environment
bash install.sh --profile depthsplat --venv .venv/depthsplat
```

Do not install all three upstream requirement files into one environment.
Their PyTorch and CUDA constraints differ. The AE runner discovers these two
default virtual environments automatically. If they live elsewhere, export
`SCARF_PYTHON_CLASSIC` and `SCARF_PYTHON_DEPTHSPLAT` with their interpreter
paths. If automatic CUDA discovery cannot find the compiler matching the
profile, set `SCARF_NVCC` to its `nvcc` executable; the checker still requires
the locked CUDA release.

For a small Functional smoke test that does not download Re10K, build the
deterministic synthetic fixture and run quick mode:

```bash
bash data/download_checkpoints.sh --profile quick
.venv/classic/bin/python data/build_quick_dataset.py --output datasets/quick-re10k
bash scripts/run_ae.sh quick
```

This fixture cannot be used for a paper-result claim.

### 3. Setup Data

Copy (or symlink) checkpoints and datasets into each submodule directory:

```bash
# --- TranSplat ---
cp -r /path/to/checkpoints/re10k.ckpt          transplat/checkpoints/
cp -r /path/to/checkpoints/depth_anything_v2_vitb.pth transplat/checkpoints/
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
| TranSplat | `re10k.ckpt`, `depth_anything_v2_vitb.pth` | `re10k/` |
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
- [Architecture Overview](docs/architecture.md)

Module-level documentation:
- [Feature extraction](feature_extractor/README.md)
- [Depth prediction](depth_predictor/README.md)

## License

MIT License

## Related Projects

- [TranSplat](https://github.com/xingyoujun/transplat), the original AAAI 2025 implementation

## Citation

If you use SCARF in your research, please cite:

```bibtex
@misc{scarf2025,
  title={SCARF: A Scene-Adaptive Depth-Guided G-3DGS Encoder Accelerator with Semantic Reuse and Fused Dataflow},
  author={...},
  year={2025}
}
```
