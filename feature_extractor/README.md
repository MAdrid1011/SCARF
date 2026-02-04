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
- **Multi-model support**: Transplat, MVSplat, DepthSplat
- **Fallback mode**: Original backbone available via flag

## Integration with demo.py

```bash
# Use hardware simulator (cycle-accurate)
python demo.py --model transplat --use-feature-sim

# Use original backbone (faster, default)
python demo.py --model transplat
```

## Architecture

See [Feature Extractor Architecture](../docs/feature-extractor-architecture.md)
