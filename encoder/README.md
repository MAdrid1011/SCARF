# SCARF Encoder Compute Units

Hardware simulators for 3DGS encoder computation units.

## Overview

This module provides Python hardware simulators for the fundamental compute units needed to implement a 3DGS encoder in hardware (FPGA/ASIC). These units are designed to be:

1. **Cycle-accurate**: Accurate cycle counting based on hardware dataflow
2. **Resource-aware**: Track LUTs, DSPs, and memory usage
3. **Multi-model compatible**: Work with Transplat, MVSplat, and DepthSplat

## Units

| Unit | Description | LUTs | DSPs | SRAM |
|------|-------------|------|------|------|
| `ConvEngine` | Systolic array convolution | 50K | 256 | 64KB |
| `GEMMUnit` | Matrix multiplication | 20K | 128 | 32KB |
| `ActivationUnit` | ReLU/GELU/SiLU/Sigmoid | 2K | 8 | 1KB |
| `NormalizationUnit` | LN/BN/IN/GN | 3K | 16 | 2KB |
| `BilinearUnit` | 2D interpolation | 800 | 8 | - |

## Usage

```python
from encoder import (
    ConvEngine, GEMMUnit, ActivationUnit, 
    NormalizationUnit, BilinearUnit,
    EncoderConfig, ActivationType, NormType
)

# Create units with default config
config = EncoderConfig.transplat_preset()

conv = ConvEngine(config.conv)
gemm = GEMMUnit(config.gemm)
activation = ActivationUnit(ActivationType.GELU, config)
norm = NormalizationUnit(NormType.LAYER, dim=128, config=config)
bilinear = BilinearUnit(config.bilinear)

# Forward pass with cycle tracking
import torch

# Convolution
x = torch.randn(1, 64, 32, 32)
weight = torch.randn(128, 64, 3, 3)
out, cycles = conv.forward(x, weight)
print(f"Conv cycles: {cycles.total_cycles}")

# GEMM
a = torch.randn(1024, 128)
b = torch.randn(128, 512)
out, cycles = gemm.matmul(a, b)
print(f"GEMM cycles: {cycles.total_cycles}")

# Activation
x = torch.randn(1024, 128)
out, cycles = activation.forward(x)
print(f"Activation cycles: {cycles.total_cycles}")

# Normalization
x = torch.randn(1, 1024, 128)
out, cycles = norm.forward(x)
print(f"Norm cycles: {cycles.total_cycles}")

# Bilinear upsampling
x = torch.randn(1, 128, 32, 32)
out, cycles = bilinear.interpolate(x, scale_factor=2)
print(f"Bilinear cycles: {cycles.total_cycles}")
```

## Architecture

See [encoder-units-architecture.md](../docs/encoder-units-architecture.md) for detailed hardware architecture documentation.

## Integration with SCARF Pipeline

These units can be integrated with the existing SCARF pipeline (DSU, GGU, SAES, FSDR) to provide complete encoder-to-Gaussian hardware simulation.

```
Input Images → ConvEngine → GEMMUnit → NormUnit → ActivationUnit
     ↓
BilinearUnit → Feature Maps → DSU → GGU → Gaussians
```
