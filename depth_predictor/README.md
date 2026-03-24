# Depth Predictor Hardware Simulator

Hardware simulator for depth prediction using SCARF encoder compute units.

## Overview

This module simulates the depth prediction stage entirely on hardware units, replacing all
PyTorch functional calls (`F.*`, `torch.*`) with cycle-counted hardware equivalents:
- **ConvEngine**: Convolution (incl. transposed, arbitrary kernel sizes up to 14×14)
- **GEMMUnit**: Matrix multiplication (attention QKV, FFN, cost volume)
- **ActivationUnit**: ReLU, GELU, SiLU, Sigmoid, Softplus (LUT-based)
- **NormalizationUnit**: LayerNorm, BatchNorm, InstanceNorm, GroupNorm
- **BilinearUnit**: Interpolation (bilinear, nearest, bicubic) + grid_sample
- **SoftmaxUnit**: Depth regression softmax
- **PoolingUnit**: Average / max pooling
- **PadUnit**: Constant, replicate, reflect, circular padding
- **DeformableAttentionUnit**: Multi-scale deformable attention

## Key Files

| File | Description |
|------|-------------|
| `types.py` | `DepthPredictorConfig`, `DepthPredictorOutput`, `CycleBreakdown` |
| `base_predictor.py` | `BaseDepthPredictorSim` ABC + `PassThroughDepthPredictorSim` |
| `hw_depth_predictor.py` | Core HW simulator: `HWDepthPredictor`, `HWUNetUnit` |
| `transplat_predictor.py` | `TransplatDepthPredictorSim` — TranSplat depth pipeline |
| `mvsplat_predictor.py` | `MVSplatDepthPredictorSim` — MVSplat depth pipeline |
| `depthsplat_predictor.py` | `DepthSplatDepthPredictorSim` — DepthSplat depth pipeline |
| `cost_volume_sim.py` | `CostVolumeSimulator` — plane-sweep cost volume |
| `depth_head_sim.py` | `DepthHeadSimulator`, `SimplifiedDepthHeadSim` — softmax regression |
| `unet_sim.py` | `UNetSimulator`, `SimplifiedUNetSim` — U-Net refinement |

## Class Hierarchy

```
BaseDepthPredictorSim (ABC)
├── TransplatDepthPredictorSim   → delegates to HWDepthPredictor
├── MVSplatDepthPredictorSim     → delegates to HWDepthPredictor
├── DepthSplatDepthPredictorSim  → delegates to HWDepthPredictor
└── PassThroughDepthPredictorSim → passes original model output through
```

`HWDepthPredictor` is the central engine that implements all three model-specific
forward paths (`_forward_transplat_hw`, `_forward_stereo_batched_hw`,
`_forward_depthsplat_hw`) using only hardware units.

## Data Flow

```
Features (from Feature Extractor)
         │
         ▼
┌─────────────────────────────┐
│   Model-Specific Predictor  │  (TranSplat / MVSplat / DepthSplat)
│   ┌─────────────────────┐   │
│   │  HWDepthPredictor   │   │
│   │  ┌───────────────┐  │   │
│   │  │  Cost Volume   │  │   │
│   │  │  (ConvEngine   │  │   │
│   │  │   + GEMMUnit)  │  │   │
│   │  └───────┬───────┘  │   │
│   │          ▼           │   │
│   │  ┌───────────────┐  │   │
│   │  │  U-Net Refine  │  │   │
│   │  │  (ConvEngine   │  │   │
│   │  │   + NormUnit)  │  │   │
│   │  └───────┬───────┘  │   │
│   │          ▼           │   │
│   │  ┌───────────────┐  │   │
│   │  │  Depth Head    │  │   │
│   │  │  (SoftmaxUnit) │  │   │
│   │  └───────────────┘  │   │
│   └─────────────────────┘   │
└─────────────┬───────────────┘
              ▼
   Depths + Raw Gaussians → GGU
```

## Usage

```python
from depth_predictor import (
    TransplatDepthPredictorSim,
    DepthPredictorConfig,
)

config = DepthPredictorConfig(num_depth_candidates=32)
sim = TransplatDepthPredictorSim(config, device=torch.device('cuda'))
sim.load_from_model(model.encoder.depth_predictor)

output = sim.forward(features, images, extrinsics, intrinsics, near, far)
depths = output.depths            # [B, V, H*W, 1, 1]
raw_gaussians = output.raw_gaussians  # [B, V, H*W, 1, C]
cycles = output.cycle_breakdown   # Per-stage cycle counts
```
