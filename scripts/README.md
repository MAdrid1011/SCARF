# SCARF Scripts

Demo scripts for SCARF hardware simulator.

## Quick Start

```bash
# Run from SCARF root directory
cd SCARF
python scripts/demo.py
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `demo.py` | Complete SCARF demo with SAES + FSDR on RE10K dataset |

## Demo Script

The `demo.py` script demonstrates SCARF's two key optimizations:

### SAES (Scene-Adaptive Early-Stopping)
- Analyzes feature similarity within tiles
- Skips depth search for homogeneous regions
- Keeps only representative Gaussians with enlarged covariances

### FSDR (Feature-Similarity Depth Reuse)
- Caches depth results using LSH-based feature signatures
- Reuses cached depths for similar pixels
- Truly modifies Gaussian 3D positions based on cache hits

## Output

The demo outputs:
- **Quality metrics**: PSNR, SSIM vs baseline
- **Performance metrics**: Hardware cycle counts (DSU, GGU)
- **SAES stats**: Early-stop ratio, Gaussian reduction
- **FSDR stats**: Cache hit rate, depth reuse rate
- **Rendered images**: Saved to `outputs/demo/`

## Requirements

Before running, ensure the transplat submodule has:
- `checkpoints/re10k.ckpt` - Transplat model weights
- `checkpoints/depth_anything_v2_vitb.pth` - Depth-Anything weights
- `datasets/re10k/` - RE10K dataset

## Example Output

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

### Hardware Cycle Counts
  Speedup: 1.24x
```
