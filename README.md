# SCARF: Hardware Simulator for 3D Gaussian Splatting Encoders

SCARF (Scene-Adaptive Cost-volume Accelerator with Reuse Framework) is a hardware-realizable accelerator for 3D Gaussian Splatting (3DGS) encoders.

## Key Features

### SAES (Scene-Adaptive Early-Stopping)
- Analyzes feature similarity within tiles
- Skips depth search for homogeneous regions
- Reduces Gaussian count by ~30%

### FSDR (Feature-Similarity Depth Reuse)
- Caches depth results using LSH-based feature signatures
- Reuses cached depths for similar pixels
- Reduces memory access by ~87%

## Supported Models

SCARF supports multiple 3D Gaussian Splatting models:

| Model | Status | Description |
|-------|--------|-------------|
| **Transplat** | ✅ Full | Transformer-based with depth priors |
| **MVSplat** | ✅ Full | Multi-view stereo with cost volume |
| **DepthSplat** | ✅ Full | DINOv2 features with 3-view support |

See [Multi-Model Demo Guide](docs/multi-model-demo-guide.md) for detailed instructions.

## Repository Structure

```
SCARF/
├── transplat/             # Transplat submodule (3DGS encoder)
├── mvsplat/               # MVSplat submodule
├── depthsplat/            # DepthSplat submodule
├── adapters/              # Model-specific adapters
├── encoder/               # Encoder compute unit simulators
│   ├── conv_engine.py     # Convolution engine (systolic array)
│   ├── gemm_unit.py       # Matrix multiplication unit
│   ├── activation_unit.py # ReLU/GELU/SiLU/Sigmoid
│   ├── normalization_unit.py # LN/BN/IN/GN
│   └── bilinear_unit.py   # Bilinear interpolation
├── dsu/                   # Depth Search Unit simulator
├── ggu/                   # Gaussian Generation Unit simulator
├── fsdr/                  # Feature-Similarity Depth Reuse
├── saes/                  # Scene-Adaptive Early-Stopping
├── scripts/
│   └── demo.py            # Complete demo with real model
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Quick Start

### 1. Clone with Submodule

```bash
git clone --recursive https://github.com/MAdrid1011/SCARF.git
cd SCARF

# If already cloned without --recursive:
git submodule update --init --recursive
```

### 2. Setup Data

Create symlinks to your data (or copy):

```bash
cd transplat
ln -s /path/to/your/checkpoints checkpoints
ln -s /path/to/your/datasets datasets
```

Required files:
- `checkpoints/re10k.ckpt` - Transplat model weights
- `checkpoints/depth_anything_v2_vitb.pth` - Depth-Anything weights
- `datasets/re10k/` - RE10K dataset

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r transplat/requirements.txt
```

### 4. Run Demo

```bash
# Run with Transplat (default)
python scripts/demo.py --model transplat

# Run with MVSplat
python scripts/demo.py --model mvsplat

# Run with DepthSplat
python scripts/demo.py --model depthsplat
```

## Demo Output

```
======================================================================
RESULTS - SCARF Demo
======================================================================

### Quality Metrics
  BASELINE: PSNR=29.19 dB, SSIM=0.9326
  SCARF:    PSNR=26.25 dB, SSIM=0.8309
  Quality loss: -2.94 dB

### Gaussian Count
  BASELINE: 131,072
  SCARF:    91,736
  Reduction: 30.0%

### FSDR Statistics
  Cache hit rate: 99.8%
  Memory reduction: 87.0%

### Hardware Cycle Counts
  Speedup: 1.24x
```

## Architecture

```
[Transplat Backbone] → features
       ↓
[SCARF SAES] → tile decisions (early-stop vs continue)
       ↓
┌──────┴──────┐
↓             ↓
Early-Stop   Continue
↓             ↓
Skip DSU   [SCARF DSU + FSDR] → depths
↓             ↓
[SCARF GGU] [SCARF GGU]
↓             ↓
└──────┬──────┘
       ↓
Gaussians + Cycle Counts
       ↓
[Transplat Decoder] → Rendered Image
```

## Documentation

- [SAES Architecture](docs/saes-architecture.md)
- [FSDR Architecture](docs/fsdr-architecture.md)
- [DSU Architecture](docs/dsu-architecture.md)
- [GGU Architecture](docs/ggu-architecture.md)
- [Hardware Dataflow Mapping](docs/hardware-dataflow-mapping.md)

## License

MIT License

## Related Projects

- [Transplat](https://github.com/xingyoujun/transplat) - Original TranSplat implementation (AAAI 2025)

## Citation

If you use SCARF in your research, please cite:

```bibtex
@misc{scarf2025,
  title={SCARF: Hardware Simulator for 3D Gaussian Splatting Encoders},
  author={...},
  year={2025}
}
```
