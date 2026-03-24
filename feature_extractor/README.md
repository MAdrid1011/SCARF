# Feature Extractor Hardware Simulator

Hardware simulator for 3DGS encoder feature extraction (CNN backbone + Transformer).

## Overview

This module simulates the feature extraction stage using existing SCARF encoder compute units:
- **ConvEngine**: Convolution operations
- **GEMMUnit**: Matrix multiplications (attention, FFN)
- **ActivationUnit**: ReLU, GELU activations
- **NormalizationUnit**: InstanceNorm, LayerNorm

## Key Features

- **Zero quality loss**: Bit-accurate output matching original PyTorch backbone
- **Cycle counting**: Track hardware cycles for all operations
- **Multi-model support**: TranSplat, MVSplat, DepthSplat
- **Feature replacement**: Simulator outputs used for actual depth prediction

## Model-Specific Extractors

| Extractor | Model | Components |
|-----------|-------|------------|
| `TransplatFeatureExtractor` | TranSplat | CNN + Transformer + DepthAnythingV2 |
| `MVSplatFeatureExtractor` | MVSplat | CNN + Transformer |
| `DepthSplatFeatureExtractor` | DepthSplat | CNN + DINOv2 + Transformer |

## Usage

### Default Mode (Feature Replacement)

```bash
# SCARF computes features and uses them for depth prediction
python demo.py --model transplat

# Same for all models
python demo.py --model mvsplat
python demo.py --model depthsplat
```

### Fallback Mode (Original Backbone)

```bash
# Use original PyTorch backbone (no cycle info)
python demo.py --model transplat --no-feature-sim
```

## API

```python
from feature_extractor import (
    TransplatFeatureExtractor,
    MVSplatFeatureExtractor,
    DepthSplatFeatureExtractor,
)

# Load from model
extractor = TransplatFeatureExtractor.from_encoder(model.encoder)

# Extract features (bit-accurate + cycle counting)
output = extractor.forward(images, extrinsics)

# Access results
features = output.features       # [B, V, C, H/8, W/8]
cycles = output.total_cycles     # Hardware cycle count
```

## Bit-Accuracy Guarantee

All extractors guarantee PSNR > 100 dB vs original PyTorch:

```python
# Verify bit-accuracy
original = model.encoder.backbone(images)
scarf = extractor.forward(images).features

psnr = compute_psnr(original, scarf)
assert psnr > 100.0  # Bit-identical
```

## Architecture

See [Feature Extractor Architecture](../docs/feature-extractor-architecture.md)
