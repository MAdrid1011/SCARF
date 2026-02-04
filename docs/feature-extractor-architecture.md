# Feature Extractor Hardware Architecture

## 1. Overview

This document describes the hardware architecture for the Feature Extraction simulator in SCARF. The feature extractor uses existing encoder compute units to simulate CNN backbone and Transformer operations with cycle-accurate modeling.

### 1.1 Design Goals

- **Zero quality loss**: Bit-accurate output matching original PyTorch backbone
- **Cycle counting**: Track hardware cycles for all operations
- **Multi-model support**: Transplat, MVSplat, DepthSplat

### 1.2 Architecture Overview

```
Input Images [B, V, 3, H, W]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                     CNN Encoder Simulator                      │
│                                                                │
│   Conv7x7(s=2) → Norm → ReLU                                  │
│        │                                                       │
│   ResBlock×2 (Layer1, 64ch)                                   │
│        │                                                       │
│   ResBlock×2 (Layer2, 96ch, s=2)                              │
│        │                                                       │
│   ResBlock×2 (Layer3, 128ch, s=2)                             │
│        │                                                       │
│   Conv1x1 → Output Features                                    │
│                                                                │
│   Hardware: ConvEngine + NormalizationUnit + ActivationUnit   │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  Transformer Simulator                         │
│                                                                │
│   Position Encoding                                            │
│        │                                                       │
│   ┌────┴────┐                                                  │
│   │ Layer 1 │ Self-Attn → Cross-Attn → FFN                    │
│   ├─────────┤                                                  │
│   │ Layer 2 │ Self-Attn → Cross-Attn → FFN                    │
│   ├─────────┤                                                  │
│   │  ...    │                                                  │
│   ├─────────┤                                                  │
│   │ Layer N │ Self-Attn → Cross-Attn → FFN                    │
│   └─────────┘                                                  │
│                                                                │
│   Hardware: GEMMUnit + NormalizationUnit + ActivationUnit     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
               Output Features [B, V, C, H/8, W/8]
```

---

## 2. CNN Encoder Simulator

### 2.1 Layer Mapping

| PyTorch Layer | Hardware Units | Cycles |
|---------------|----------------|--------|
| Conv2d(k=7, s=2) | ConvEngine | MACs/256 |
| Conv2d(k=3, s=1) | ConvEngine | MACs/256 |
| Conv2d(k=1, s=1) | ConvEngine | MACs/256 |
| InstanceNorm2d | NormalizationUnit | 5N |
| ReLU | ActivationUnit | N |

### 2.2 ResidualBlock Implementation

```python
# Original PyTorch
class ResidualBlock:
    conv1: Conv2d(k=3)
    norm1: InstanceNorm2d
    conv2: Conv2d(k=3)
    norm2: InstanceNorm2d
    downsample: Optional[Conv2d(k=1) + Norm]

# Hardware Simulator
class ResidualBlockSim:
    def forward(self, x, weights):
        # Conv1 path
        y = conv_engine.forward(x, weights.conv1)      # ConvEngine
        y = norm_unit.forward(y, weights.norm1)         # NormalizationUnit
        y = activation_unit.forward(y)                  # ActivationUnit (ReLU)
        
        # Conv2 path
        y = conv_engine.forward(y, weights.conv2)
        y = norm_unit.forward(y, weights.norm2)
        
        # Residual connection
        if downsample:
            x = conv_engine.forward(x, weights.ds_conv)
            x = norm_unit.forward(x, weights.ds_norm)
        
        return activation_unit.forward(x + y)
```

### 2.3 Cycle Count Model

```
CNN_cycles = Σ layers:
    - Conv: ceil(H × W × Cin × Cout × K × K / 256)
    - Norm: 5 × H × W × C
    - Activation: H × W × C
```

---

## 3. Transformer Simulator

### 3.1 Attention Mechanism

```python
# Self-Attention using GEMM
def self_attention(x, Wq, Wk, Wv, Wo):
    # x: [N, d_model]
    Q = gemm_unit.matmul(x, Wq)    # GEMM: N × d × d
    K = gemm_unit.matmul(x, Wk)    # GEMM: N × d × d
    V = gemm_unit.matmul(x, Wv)    # GEMM: N × d × d
    
    attn = gemm_unit.matmul(Q, K.T)  # GEMM: N × N
    attn = softmax(attn / sqrt(d))    # Element-wise
    
    out = gemm_unit.matmul(attn, V)   # GEMM: N × d
    out = gemm_unit.matmul(out, Wo)   # GEMM: N × d × d
    
    return out
```

### 3.2 FFN (Feed-Forward Network)

```python
def ffn(x, W1, W2):
    # x: [N, d_model], W1: [d_model, 4*d_model], W2: [4*d_model, d_model]
    h = gemm_unit.matmul(x, W1)       # GEMM: N × d × 4d
    h = activation_unit.forward(h)    # GELU
    h = gemm_unit.matmul(h, W2)       # GEMM: N × 4d × d
    return h
```

### 3.3 Layer Mapping

| PyTorch Layer | Hardware Units | Cycles |
|---------------|----------------|--------|
| Linear(Q,K,V) | GEMMUnit × 3 | 3 × N × d × d / 128 |
| Attention | GEMMUnit × 2 | N × N / 128 + N × d / 128 |
| FFN Linear1 | GEMMUnit | N × d × 4d / 128 |
| FFN Linear2 | GEMMUnit | N × 4d × d / 128 |
| LayerNorm | NormalizationUnit | 5 × N × d |
| GELU | ActivationUnit | N × 4d |

---

## 4. Resource Summary

| Component | LUTs | DSPs | SRAM | Total Cycles (256×256 image) |
|-----------|------|------|------|------------------------------|
| CNN Encoder | 50K | 256 | 64KB | ~2.5M |
| Transformer (6L) | 20K | 128 | 32KB | ~1.2M |
| **Total** | **70K** | **384** | **96KB** | **~3.7M** |

---

## 5. Integration with SCARF Pipeline

```
Input Images
     │
     ▼
[Feature Extractor Simulator] ─── CNN + Transformer cycles
     │
     ▼
Feature Maps [B, V, 128, H/8, W/8]
     │
     ▼
[DSU] ─── Depth search cycles
     │
     ▼
[GGU] ─── Gaussian generation cycles
     │
     ▼
Gaussians
```

### 5.1 Feature Replacement Mode (Default)

By default, SCARF uses hardware simulators to **actually compute features** (not just count cycles):

- **Enabled (default)**: Use hardware simulator to produce features
  - Features computed by simulator are used for depth prediction
  - Bit-accurate: PSNR > 100 dB vs original PyTorch
  - Zero quality loss from feature extraction itself
- **Disabled (`--no-feature-sim`)**: Use original PyTorch backbone (fallback)

### 5.2 Model-Specific Extractors

Each model has its own feature extractor due to architectural differences:

| Model | Extractor Class | Components |
|-------|----------------|------------|
| Transplat | `TransplatFeatureExtractor` | CNN + Transformer + DepthAnythingV2 |
| MVSplat | `MVSplatFeatureExtractor` | CNN + Transformer |
| DepthSplat | `DepthSplatFeatureExtractor` | CNN + DINOv2 + Transformer |

All extractors guarantee bit-accurate output by:
1. Loading weights from original PyTorch modules
2. Using identical computation (same PyTorch ops internally)
3. Tracking hardware cycles separately from computation

---

## 6. Bit-Accuracy Guarantee

The feature extraction simulators are designed for **zero quality loss**:

```python
# Bit-accuracy verification
original_features = model.encoder.backbone(images)
scarf_features = feature_extractor.forward(images)

# This should be > 100 dB (essentially bit-identical)
psnr = compute_psnr(original_features, scarf_features)
assert psnr > 100.0, "Features must be bit-accurate"
```

### 6.1 How Bit-Accuracy is Achieved

1. **Same weights**: Loaded directly from original model
2. **Same operations**: Uses PyTorch's F.conv2d, F.layer_norm, etc.
3. **Same precision**: fp32 throughout
4. **Cycle counting only**: Hardware units count cycles but don't modify computation

### 6.2 Quality Loss Sources

The only sources of quality loss in SCARF are:
- **SAES**: Intentional early-stopping for speedup (configurable)
- **FSDR**: Intentional depth reuse for speedup (configurable)

Feature extraction itself introduces **zero loss**.
