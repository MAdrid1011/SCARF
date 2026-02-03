# Encoder Compute Units Architecture

## 1. Overview

This document describes the hardware architecture for encoder compute units in SCARF. These units are the fundamental building blocks for 3DGS encoder computation (CNN backbone, Transformer, UNet).

### 1.1 Supported Models Analysis

Based on analysis of Transplat, MVSplat, and DepthSplat encoders:

| Operation | Transplat | MVSplat | DepthSplat | Common Config |
|-----------|-----------|---------|------------|---------------|
| **Conv2d** | 1x1, 3x3, 7x7 | 1x1, 3x3, 7x7 | 1x1, 3x3, 7x7 | k={1,3,7}, s={1,2} |
| **Linear/GEMM** | d=128, ffn=4x | d=128, ffn=4x | d=128, ffn=4x | d_model=128 |
| **Activation** | ReLU, GELU | ReLU, GELU, SiLU | ReLU, GELU, SiLU | All types |
| **Normalization** | LN, IN, GN | LN, IN, GN, BN | LN, IN, GN, BN | All types |
| **Interpolation** | bilinear, 2x/4x | bilinear, 2x/4x | bilinear, 2x/4x | align_corners=True |

---

## 2. Hardware Resource Summary

| Unit | LUTs | DSPs | SRAM | Latency | Priority |
|------|------|------|------|---------|----------|
| **Convolution Engine** | 50K | 256 | 64KB | var | ⭐⭐⭐ |
| **GEMM Unit** | 20K | 128 | 32KB | var | ⭐⭐⭐ |
| **Activation Unit** | 2K | 8 | 1KB | 1-4 cy | ⭐⭐ |
| **Normalization Unit** | 3K | 16 | 2KB | ~5N cy | ⭐⭐ |
| **Bilinear Unit** | 800 | 8 | - | 4 cy | ⭐ |
| **Total** | **75.8K** | **416** | **~99KB** | - | - |

---

## 3. Convolution Engine

### 3.1 Supported Configurations

| Parameter | Range | Default |
|-----------|-------|---------|
| Kernel Size | 1, 3, 7 | 3 |
| Stride | 1, 2 | 1 |
| Dilation | 1-4 | 1 |
| Input Channels | 3-1024 | 128 |
| Output Channels | 64-1024 | 128 |

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Convolution Engine                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Feature Map              Weight Buffer               │
│   [H, W, Cin]                    [K, K, Cin, Cout]          │
│        │                              │                      │
│        ▼                              ▼                      │
│   ┌─────────────────────────────────────────────────┐       │
│   │         Systolic Array (16x16 PEs)              │       │
│   │                                                  │       │
│   │   PE[0,0] ─ PE[0,1] ─ PE[0,2] ─ ... ─ PE[0,15] │       │
│   │      │        │         │              │        │       │
│   │   PE[1,0] ─ PE[1,1] ─ PE[1,2] ─ ... ─ PE[1,15] │       │
│   │      │        │         │              │        │       │
│   │     ...      ...       ...            ...       │       │
│   │      │        │         │              │        │       │
│   │   PE[15,0]─ PE[15,1]─ PE[15,2]─ ... ─ PE[15,15]│       │
│   └──────────────────────────┬──────────────────────┘       │
│                              │                               │
│                              ▼                               │
│                    Accumulator + Bias                        │
│                              │                               │
│                              ▼                               │
│                    Output Feature Map                        │
│                    [H', W', Cout]                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Cycle Count Model

```python
def conv_cycles(H, W, Cin, Cout, K, stride, PE_array=16):
    """
    Systolic array convolution cycle count.
    
    Output dimensions:
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    
    Total MACs: H_out * W_out * Cin * Cout * K * K
    Throughput: PE_array^2 MACs per cycle
    """
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    total_macs = H_out * W_out * Cin * Cout * K * K
    cycles = total_macs // (PE_array * PE_array)
    return max(cycles, 1)
```

### 3.4 Resource Breakdown

| Component | LUTs | DSPs | Description |
|-----------|------|------|-------------|
| Systolic Array (16x16) | 40K | 256 | 256 MAC units |
| Weight Buffer | 5K | 0 | 64KB SRAM for weights |
| Input Buffer | 3K | 0 | Line buffer for sliding window |
| Control Logic | 2K | 0 | FSM, address generation |
| **Total** | **50K** | **256** | |

---

## 4. GEMM Unit (Matrix Multiply)

### 4.1 Supported Configurations

| Parameter | Range | Default |
|-----------|-------|---------|
| M (batch×sequence) | 1-65536 | 4096 |
| N (output dim) | 64-4096 | 128 |
| K (inner dim) | 64-4096 | 128 |

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       GEMM Unit                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Matrix A [M, K]              Matrix B [K, N]              │
│        │                              │                      │
│        ▼                              ▼                      │
│   ┌─────────────────────────────────────────────────┐       │
│   │      Output-Stationary Systolic Array (8x16)    │       │
│   │                                                  │       │
│   │   - K dimension streams through                  │       │
│   │   - Partial sums accumulate in place            │       │
│   │   - Tile size: 8×16 output elements             │       │
│   └──────────────────────────┬──────────────────────┘       │
│                              │                               │
│                              ▼                               │
│                    Output Matrix [M, N]                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Cycle Count Model

```python
def gemm_cycles(M, N, K, tile_m=8, tile_n=16):
    """
    Tiled GEMM cycle count.
    
    Process M×N output in tile_m×tile_n tiles.
    Each tile takes K cycles to accumulate.
    """
    num_tiles_m = (M + tile_m - 1) // tile_m
    num_tiles_n = (N + tile_n - 1) // tile_n
    cycles_per_tile = K + tile_m + tile_n  # pipeline fill + drain
    total_cycles = num_tiles_m * num_tiles_n * cycles_per_tile
    return total_cycles
```

### 4.4 Resource Breakdown

| Component | LUTs | DSPs | Description |
|-----------|------|------|-------------|
| Systolic Array (8x16) | 15K | 128 | 128 MAC units |
| A Buffer | 2K | 0 | Tile buffer for matrix A |
| B Buffer | 2K | 0 | Tile buffer for matrix B |
| Control Logic | 1K | 0 | FSM, tiling logic |
| **Total** | **20K** | **128** | |

---

## 5. Activation Unit

### 5.1 Supported Functions

| Function | Formula | Implementation | Cycles |
|----------|---------|----------------|--------|
| **ReLU** | max(0, x) | Comparator | 1 |
| **GELU** | x·Φ(x) | 256-entry LUT | 1 |
| **SiLU** | x·σ(x) | 256-entry LUT | 1 |
| **Sigmoid** | 1/(1+e^-x) | 256-entry LUT | 1 |

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Activation Unit                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input x                                                    │
│      │                                                       │
│      ├──────┬──────┬──────┬──────┐                          │
│      │      │      │      │      │                          │
│      ▼      ▼      ▼      ▼      ▼                          │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                          │
│   │ReLU │ │GELU │ │SiLU │ │Sigm │                          │
│   │Comp.│ │ LUT │ │ LUT │ │ LUT │                          │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                          │
│      │      │      │      │                                 │
│      └──────┴──────┴──────┴──────┐                          │
│                                   │                          │
│                              ┌────▼────┐                     │
│                              │   MUX   │◀── activation_type  │
│                              └────┬────┘                     │
│                                   │                          │
│                                   ▼                          │
│                               Output                         │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 LUT Generation

```python
import numpy as np
from scipy.special import erf

def generate_gelu_lut(num_entries=256, input_range=(-4, 4)):
    """Generate GELU lookup table."""
    x = np.linspace(input_range[0], input_range[1], num_entries)
    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu = x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    return x, gelu

def generate_silu_lut(num_entries=256, input_range=(-4, 4)):
    """Generate SiLU (Swish) lookup table."""
    x = np.linspace(input_range[0], input_range[1], num_entries)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu = x / (1.0 + np.exp(-x))
    return x, silu
```

### 5.4 Resource Breakdown

| Component | LUTs | DSPs | SRAM | Description |
|-----------|------|------|------|-------------|
| ReLU Comparator | 100 | 0 | 0 | max(0, x) |
| GELU LUT | 500 | 0 | 256B | 256-entry table |
| SiLU LUT | 500 | 0 | 256B | 256-entry table |
| Sigmoid LUT | 500 | 0 | 256B | 256-entry table |
| Interpolator | 300 | 8 | 0 | Linear interpolation |
| MUX + Control | 100 | 0 | 0 | Function select |
| **Total** | **2K** | **8** | **~1KB** | |

---

## 6. Normalization Unit

### 6.1 Supported Types

| Type | Formula | Dimension | Use Case |
|------|---------|-----------|----------|
| **LayerNorm** | (x-μ)/σ·γ+β | Last dim | Transformer |
| **BatchNorm** | (x-μ_run)/σ_run·γ+β | Channel | CNN |
| **InstanceNorm** | (x-μ)/σ·γ+β | Per sample | CNN backbone |
| **GroupNorm** | (x-μ_g)/σ_g·γ+β | Groups | UNet, depth |

### 6.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Normalization Unit                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input x [N elements]                                       │
│        │                                                     │
│        ├─────────────────────────────────┐                  │
│        │                                  │                  │
│        ▼                                  ▼                  │
│   ┌─────────────┐                  ┌─────────────┐          │
│   │ Mean Calc.  │                  │ Var Calc.   │          │
│   │ Σx/N        │                  │ Σ(x-μ)²/N   │          │
│   └──────┬──────┘                  └──────┬──────┘          │
│          │μ                               │σ²                │
│          └───────────┬───────────────────┘                  │
│                      │                                       │
│                      ▼                                       │
│             ┌─────────────────┐                             │
│             │ Normalize       │                             │
│             │ y = (x-μ)/√(σ²+ε)│                            │
│             └────────┬────────┘                             │
│                      │                                       │
│                      ▼                                       │
│             ┌─────────────────┐                             │
│             │ Scale & Shift   │                             │
│             │ z = y·γ + β     │                             │
│             └────────┬────────┘                             │
│                      │                                       │
│                      ▼                                       │
│                   Output                                     │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Cycle Count Model

```python
def norm_cycles(N, norm_type='layer'):
    """
    Normalization cycle count.
    
    LayerNorm/InstanceNorm: 
      - Mean: N cycles (accumulate) + 1 (divide)
      - Var: N cycles (accumulate) + 1 (divide)
      - Normalize: N cycles
      - Total: ~3N + 2
    
    GroupNorm (G groups):
      - Same as LayerNorm but per group: ~3*(N/G) + 2 per group
      - Total: G * (~3*(N/G) + 2) = ~3N + 2G
    """
    if norm_type in ['layer', 'instance']:
        return 3 * N + 2
    elif norm_type == 'group':
        # Assume 8 groups (common in models)
        G = 8
        return 3 * N + 2 * G
    elif norm_type == 'batch':
        # Uses running stats, just scale/shift
        return N
```

### 6.4 Resource Breakdown

| Component | LUTs | DSPs | Description |
|-----------|------|------|-------------|
| Accumulator | 500 | 4 | Mean/var accumulation |
| Divider | 800 | 0 | Fixed-point division |
| Sqrt Unit | 600 | 4 | √(var+ε) |
| Multiplier | 500 | 8 | Scale/shift |
| Mode Control | 200 | 0 | LN/BN/IN/GN select |
| Running Stats | 400 | 0 | BatchNorm stats buffer |
| **Total** | **3K** | **16** | |

---

## 7. Bilinear Interpolation Unit

### 7.1 Supported Modes

| Mode | Description | align_corners |
|------|-------------|---------------|
| **scale_factor** | 2x, 4x upsampling | True/False |
| **target_size** | Specific output size | True/False |

### 7.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Bilinear Interpolation Unit                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input: [H_in, W_in, C]     Target: [H_out, W_out]         │
│        │                                                     │
│        ▼                                                     │
│   ┌──────────────────────────────────────┐                  │
│   │       Coordinate Calculator          │                  │
│   │                                       │                  │
│   │  For each output pixel (y_out, x_out):│                  │
│   │    y_in = y_out * (H_in-1)/(H_out-1) │  (align_corners) │
│   │    x_in = x_out * (W_in-1)/(W_out-1) │                  │
│   │                                       │                  │
│   │  y0 = floor(y_in), y1 = y0 + 1       │                  │
│   │  x0 = floor(x_in), x1 = x0 + 1       │                  │
│   │  dy = y_in - y0, dx = x_in - x0      │                  │
│   └────────────────────┬─────────────────┘                  │
│                        │                                     │
│                        ▼                                     │
│   ┌──────────────────────────────────────┐                  │
│   │         4-Point Sampler              │                  │
│   │                                       │                  │
│   │    p00 = Input[y0, x0]               │                  │
│   │    p01 = Input[y0, x1]               │                  │
│   │    p10 = Input[y1, x0]               │                  │
│   │    p11 = Input[y1, x1]               │                  │
│   └────────────────────┬─────────────────┘                  │
│                        │                                     │
│                        ▼                                     │
│   ┌──────────────────────────────────────┐                  │
│   │      Bilinear Interpolator           │                  │
│   │                                       │                  │
│   │  out = p00*(1-dx)*(1-dy)             │                  │
│   │      + p01*dx*(1-dy)                 │                  │
│   │      + p10*(1-dx)*dy                 │                  │
│   │      + p11*dx*dy                     │                  │
│   └────────────────────┬─────────────────┘                  │
│                        │                                     │
│                        ▼                                     │
│                  Output pixel                                │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Cycle Count Model

```python
def bilinear_cycles(H_out, W_out, C):
    """
    Bilinear interpolation cycle count.
    
    Per output pixel:
      - Coordinate calculation: 1 cycle
      - 4-point sampling: 1 cycle (parallel)
      - Interpolation: 2 cycles (4 MACs)
    Total: 4 cycles per output pixel
    
    With C channels processed in parallel (up to 8):
    """
    parallel_channels = min(C, 8)
    pixels = H_out * W_out
    cycles_per_pixel = 4
    channel_batches = (C + parallel_channels - 1) // parallel_channels
    return pixels * cycles_per_pixel * channel_batches
```

### 7.4 Resource Breakdown

| Component | LUTs | DSPs | Description |
|-----------|------|------|-------------|
| Coord Calculator | 200 | 0 | Fixed-point arithmetic |
| 4-Port Sampler | 200 | 0 | Address generation |
| Interpolator | 300 | 8 | 4 parallel MACs |
| Control | 100 | 0 | FSM |
| **Total** | **800** | **8** | |

---

## 8. Integration with SCARF Pipeline

### 8.1 Dataflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SCARF Encoder Pipeline                        │
└─────────────────────────────────────────────────────────────────────┘

   Input Images [B, 2, 3, H, W]
         │
         ▼
   ┌─────────────┐
   │ Conv Engine │  ← CNN backbone (ResNet-like)
   │   + ReLU    │    - 7x7 conv, stride 2
   │   + Norm    │    - Residual blocks with 3x3 conv
   └──────┬──────┘
         │
         ▼
   ┌─────────────┐
   │ GEMM Unit   │  ← Transformer layers
   │   + LN      │    - Self/cross attention
   │   + GELU    │    - FFN (d_model → 4*d_model → d_model)
   └──────┬──────┘
         │
         ▼
   ┌─────────────┐
   │ Conv Engine │  ← Depth/Gaussian head
   │   + GN      │    - 3x3 convolutions
   │   + GELU    │    - GroupNorm
   └──────┬──────┘
         │
         ▼
   ┌─────────────┐
   │ Bilinear    │  ← Upsampling
   │   Unit      │    - 2x/4x scale
   └──────┬──────┘
         │
         ▼
   Feature Maps → DSU → GGU → Gaussians
```

### 8.2 Cycle Budget Estimation

For a typical 256×256 input with 2 views:

| Stage | Operations | Cycles | % Total |
|-------|------------|--------|---------|
| CNN Backbone | Conv 7x7 + 3 ResBlocks | ~50K | 25% |
| Transformer (6 layers) | 6 × (Attn + FFN) | ~80K | 40% |
| Depth Head | Conv 3x3 × 3 | ~20K | 10% |
| Upsample | Bilinear 4x | ~10K | 5% |
| DSU + GGU | (existing SCARF) | ~40K | 20% |
| **Total** | | **~200K** | 100% |

---

## 9. Configuration Interface

### 9.1 Runtime Configuration

```python
@dataclass
class EncoderConfig:
    """Unified configuration for encoder compute units."""
    
    # Convolution Engine
    conv_pe_array_size: int = 16  # 16x16 systolic array
    
    # GEMM Unit
    gemm_tile_m: int = 8
    gemm_tile_n: int = 16
    
    # Activation
    activation_lut_size: int = 256
    activation_input_range: Tuple[float, float] = (-4.0, 4.0)
    
    # Normalization
    norm_epsilon: float = 1e-5
    norm_groups: int = 8  # for GroupNorm
    
    # Bilinear
    bilinear_parallel_channels: int = 8
    bilinear_align_corners: bool = True
```

### 9.2 Model-Specific Presets

```python
TRANSPLAT_PRESET = EncoderConfig(
    conv_pe_array_size=16,
    gemm_tile_m=8,
    gemm_tile_n=16,
    norm_groups=8,
)

MVSPLAT_PRESET = EncoderConfig(
    conv_pe_array_size=16,
    gemm_tile_m=8,
    gemm_tile_n=16,
    norm_groups=8,
)

DEPTHSPLAT_PRESET = EncoderConfig(
    conv_pe_array_size=16,
    gemm_tile_m=8,
    gemm_tile_n=16,
    norm_groups=4,  # DepthSplat uses GroupNorm(4)
)
```

---

## 10. Summary

### 10.1 Hardware Feasibility

| Criterion | Status | Notes |
|-----------|--------|-------|
| FPGA Fit (XC7A200T) | ✅ | 56% LUT, 56% DSP |
| ASIC Area (28nm) | ✅ | ~1.2 mm² |
| Power (28nm @ 200MHz) | ✅ | ~150 mW |
| Memory Budget | ✅ | ~100KB on-chip |

### 10.2 Key Design Decisions

1. **Unified Systolic Arrays**: Conv and GEMM share similar dataflow
2. **LUT-based Activations**: Minimizes compute for non-linear functions
3. **Flexible Normalization**: Single unit supports all norm types
4. **Parallel Bilinear**: 8-channel parallel processing

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-03  
**Related Issue**: GitHub Issue #8
