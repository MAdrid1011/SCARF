# SCARF 流水线架构

## 1. 总体概述

SCARF (Scalable Cross-view Accelerator for Radiance Fields) 是一款面向可泛化 3DGS (3D Gaussian Splatting) 编码器的 ASIC 推理加速器。其核心流水线将编码器推理划分为三个主要阶段，各阶段映射到专用硬件计算单元。

```
输入: 2张参考视图图像 (256×256)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  S1: 特征提取 (Feature Extraction)                    │
│  [ConvEngine] + [GEMM Unit]                          │
│  CNN backbone → Transformer encoder → 特征图          │
└──────────────────┬───────────────────────────────────┘
                   │  features: [2, 128, 64, 64]
                   ▼
┌──────────────────────────────────────────────────────┐
│  S2: 深度预测 (Depth Prediction)                      │
│  [BilinearUnit] + [ConvEngine] + [VectorALU]         │
│  cost_volume → U-Net → depth_head → regression        │
└──────────────────┬───────────────────────────────────┘
                   │  depths: [2, 65536, 1]
                   │  raw_gaussians: [2, 65536, C]
                   ▼
┌──────────────────────────────────────────────────────┐
│  S3: 高斯生成 (Gaussian Generation)                   │
│  [ConvEngine]                                        │
│  refine_unet → to_gaussians (NN head)                │
└──────────────────┬───────────────────────────────────┘
                   │  means, covariances, harmonics, opacities
                   ▼
┌──────────────────────────────────────────────────────┐
│  GGU: 后处理 (Post-Processing)                       │
│  [GGU PEs] (专用处理元件)                              │
│  depth→position, raw→covariance/SH/opacity           │
└──────────────────┬───────────────────────────────────┘
                   │  完整 3D Gaussian 基元
                   ▼
               Splatting 渲染
```

### 1.1 硬件配置

| 计算单元 | 规格 | 用途 |
|---------|------|------|
| **ConvEngine** | 48×48 = 2304 MACs | 卷积操作 (CNN, U-Net, depth_head) |
| **GEMM Unit** | 48×48 = 2304 MACs | 矩阵乘法 (Transformer attention, regression) |
| **VectorALU** | 64-wide SIMD | 逐元素操作 (激活函数, normalization) |
| **BilinearUnit** | 64ch × 32 samplers | 双线性插值采样 (cost_volume warping) |
| **GGU PEs** | 32 个处理元件 | 高斯参数后处理 (quaternion, SH rotation) |
| **时钟频率** | 1000 MHz | 28nm TSMC HPC+ |
| **总 MACs** | 4608 | 4.6 GOPS peak |

### 1.2 硬件加速分阶段模型

SCARF 采用**分阶段硬件加速模型**（非均匀加速）：

| 操作类型 | 加速倍数 | 依据 |
|---------|---------|------|
| **计算受限** (conv, GEMM, depth_head, regression) | 2.0× | 48²/32² = 2.25× MACs，扣除 ~11% dataflow overhead |
| **内存受限** (cost_volume bilinear warping) | 1.5× | SRAM 输入带宽 ∝ 阵列行宽 48/32 = 1.5× |

这意味着 S2 阶段内部的 cost_volume 子阶段与其余子阶段获得不同的加速倍率。

---

## 2. S1: 特征提取 (Feature Extraction)

### 2.1 功能

从两张 256×256 输入图像中提取 128 维特征图。

### 2.2 计算流程

```
输入: images [B, 2, 3, 256, 256]
  │
  ▼
CNN Backbone (ResNet/EfficientNet 变体)
  │  多层卷积 + BatchNorm + ReLU
  │  [ConvEngine]: 1×1, 3×3, 7×7 kernel sizes
  │  下采样: 256×256 → 128×128 → 64×64
  │
  ▼
Transformer Encoder
  │  Multi-head self-attention + cross-attention
  │  [GEMM Unit]: Q/K/V projection, attention matmul
  │  [VectorALU]: softmax, LayerNorm
  │  Feature dimension: 128
  │
  ▼
输出: features [B, 2, 128, 64, 64]
```

### 2.3 硬件映射

| 子操作 | 计算单元 | 特性 |
|--------|---------|------|
| 卷积层 (1×1, 3×3, 7×7) | ConvEngine (2304 MACs) | 计算受限，2.0× 加速 |
| Transformer Q/K/V 投影 | GEMM Unit (2304 MACs) | 计算受限，2.0× 加速 |
| Attention softmax + LN | VectorALU (64-wide) | 计算受限 |

### 2.4 流水线重叠

**PIPE_FE = 0.95** (5% 重叠)

- CNN (ConvEngine) 与 Transformer (GEMM Unit) 使用不同硬件单元
- 在 tile 边界处，CNN 输出可通过双端口 SRAM 双缓冲提前写入，Transformer 同步读取上一 tile 结果
- 由于两者共享 VectorALU（激活/归一化），重叠有限，5% 是保守估计

### 2.5 典型周期数

| 子阶段 | 原始周期 | 加速后周期 |
|--------|---------|-----------|
| CNN backbone | ~131,988K | ~65,994K |
| Transformer | ~31,457K | ~15,729K |
| **S1 总计** | **163,445K** | **81,723K** |

---

## 3. S2: 深度预测 (Depth Prediction)

### 3.1 功能

基于双视图特征图，通过 cost volume 匹配和 U-Net 精化预测每个像素的深度值和初始高斯参数。

### 3.2 计算流程

```
输入: features [B, 2, 128, 64, 64], camera params
  │
  ▼
Cost Volume 构建 (cost_volume)
  │  对每个参考像素 × D=128 个深度候选:
  │    1. 将参考像素按 d_k 投影到目标视图 → (u', v')
  │    2. 从目标特征图双线性插值采样 → tgt_feat [128]
  │    3. 计算匹配代价: correlation(ref_feat, tgt_feat)
  │  [BilinearUnit]: 双线性插值采样 (内存受限)
  │  [ConvEngine]: 相关性计算
  │  输出: cost_volume [B, groups, D, H, W]
  │
  ▼
3D U-Net (unet)
  │  多尺度卷积编码器-解码器
  │  Encoder: 逐步下采样 (conv3d + BN + ReLU)
  │  Decoder: 逐步上采样 + skip connections
  │  [ConvEngine]: 3D 卷积操作
  │  输出: refined_volume [B, C, D, H, W]
  │
  ▼
Depth Head (depth_head)
  │  1×1 卷积将通道数映射到 D 个深度概率
  │  [ConvEngine]: 1×1 conv
  │  输出: prob_volume [B, 1, D, H, W]
  │
  ▼
Depth Regression (regression)
  │  Softmax 沿深度维度 → 深度期望值
  │  [GEMM Unit]: softmax + weighted sum
  │  [VectorALU]: exp, normalization
  │  输出: depths [B, 2, N, 1, 1]
  │
  ▼
Gaussian Generation NN (gauss_gen — 本文归入 S3)
```

### 3.3 硬件映射与分阶段加速

S2 是**混合受限**阶段——cost_volume 是内存受限操作，其余是计算受限操作：

| 子阶段 | 计算单元 | 受限类型 | 加速倍数 | 原始周期 | 加速后周期 |
|--------|---------|---------|---------|---------|-----------|
| cost_volume | BilinearUnit + ConvEngine | **内存受限** | 1.5× | 238,932K | 159,288K |
| unet | ConvEngine | 计算受限 | 2.0× | 91,772K | 45,886K |
| depth_head | ConvEngine | 计算受限 | 2.0× | 6,834K | 3,417K |
| regression | GEMM + VectorALU | 计算受限 | 2.0× | 7,537K | 3,768K |
| **S2 总计** | | | | **345,075K** | **212,359K** |

**cost_volume 占加速后 S2 的 ~75%**，这是 FSDR 窄化搜索能带来显著节省的原因。

### 3.4 流水线重叠

**PIPE_DP = 0.88** (12% 重叠)

- S2 包含 4 个顺序子阶段：cost_volume → unet → depth_head → regression
- cost_volume (BilinearUnit) 和 unet (ConvEngine) 使用不同硬件单元
- Tile N 的 cost_volume 完成后，ConvEngine 开始处理 Tile N 的 unet；同时 BilinearUnit 可以开始 Tile N+1 的 cost_volume
- 通过 ping-pong 缓冲区实现 tile 级流水线，4 个子阶段跨越产生约 12% 的重叠

---

## 4. S3: 高斯生成 (Gaussian Generation — NN 部分)

### 4.1 功能

通过神经网络将 S2 的原始输出精化为高斯基元参数（means, covariances, harmonics, opacities）。

### 4.2 计算流程

```
输入: raw_outputs from S2 (depths + raw_gaussians)
  │
  ▼
Refine U-Net (refine_unet)
  │  2D U-Net 精化器
  │  将粗糙预测精化为精确的高斯参数
  │  [ConvEngine]: 2D conv + BN + ReLU
  │
  ▼
to_gaussians Head
  │  多个 1×1 conv head:
  │    - means head: 3D 位置 (x, y, z)
  │    - covariance head: scales + quaternion rotation
  │    - harmonics head: 球谐系数 (SH degree 4)
  │    - opacity head: 透明度
  │  [ConvEngine]: 1×1 conv
  │
  ▼
输出: Gaussians {means, covariances, harmonics, opacities}
       共 131,072 个高斯基元 (2 views × 65,536 pixels)
```

### 4.3 硬件映射

| 子阶段 | 计算单元 | 特性 |
|--------|---------|------|
| refine_unet | ConvEngine | 计算受限，2.0× 加速 |
| to_gaussians | ConvEngine | 计算受限，2.0× 加速 |

### 4.4 流水线重叠

**PIPE_GG_NN = 0.97** (3% 重叠)

- refine_unet 和 to_gaussians 均使用 ConvEngine，无法并行
- 仅有的重叠来自：to_gaussians 的输出缓冲区写回与下一 tile 的权重预取
- 由于两个子阶段共享 ConvEngine，重叠非常有限

### 4.5 典型周期数

| 子阶段 | 原始周期 | 加速后周期 |
|--------|---------|-----------|
| refine_unet + to_gaussians | 314,345K | 157,172K |

---

## 5. GGU: 后处理 (Gaussian Generation Unit)

### 5.1 功能

将 S3 输出的高斯参数转换为世界坐标系下的完整 3D 高斯基元，供 splatting 渲染器使用。

### 5.2 计算流程

```
输入: 每个高斯基元的 {depth, raw_scale, raw_rotation, raw_sh, raw_opacity}
  │
  ▼
Per-Gaussian Processing (每个 GGU PE 独立处理):
  │
  ├─ Position (10 cycles): depth + 像素坐标 → ray origin + direction * depth
  ├─ Scale (5 cycles): sigmoid LUT + depth-adaptive scaling
  ├─ Quat→Matrix (30 cycles): quaternion normalize + 9-entry rotation matrix
  ├─ Covariance (36 cycles): R @ S @ S^T @ R^T (3 × 3×3 matmul)
  ├─ Transform (24 cycles): R_c2w @ cov @ R_c2w^T (世界坐标变换)
  ├─ SH Rotation (80 cycles): 旋转 25 SH 系数 × 3 通道
  └─ Opacity (2 cycles): sigmoid LUT
  │
  总计: 187 cycles/Gaussian
  │
  ▼
输出: 完整 Gaussian {world_position, world_covariance, rotated_SH, opacity}
```

### 5.3 硬件实现

- **32 个 GGU PEs** 并行处理
- 每个 PE 包含小型 3×3 matmul 单元（非大型 48×48 脉动阵列）
- 131,072 个高斯 / 32 PEs = 4,096 批次 × 187 cycles ≈ 766K cycles
- 加上 pipeline overhead: ~391K cycles (并行化后)

### 5.4 GGU 隐藏

GGU 周期被**完全隐藏**在 S3 ConvEngine 工作之后：
- S3 (ConvEngine) 产生高斯参数的同时，GGU PEs 处理上一批次的高斯
- 由于 GGU PEs 是独立硬件，不与 ConvEngine 竞争资源
- 因此在最终 pipeline effective 计算中 GGU 周期为 0

---

## 6. 端到端流水线性能

### 6.1 流水线模型

```
时间轴 ─────────────────────────────────────────────────────►

         ┌──────────┐
    S1:  │ ConvEng  │ (× PIPE_FE=0.95)
         │ + GEMM   │
         └────┬─────┘
              │
              ▼
         ┌──────────────────────────┐
    S2:  │ BilinearUnit → ConvEng  │ (× PIPE_DP=0.88)
         │ cost_vol → unet →       │
         │ depth_head → regression  │
         └────┬─────────────────────┘
              │
              ▼
         ┌─────────────┐┌─ ─ ─ ─ ─ ─┐
    S3:  │ ConvEngine  ││ GGU PEs   │ (× PIPE_GG_NN=0.97)
         │ refine_unet ││ (hidden)  │
         │ + to_gauss  ││           │
         └─────────────┘└─ ─ ─ ─ ─ ─┘
```

### 6.2 性能数据 (TranSplat, 256×256, 无优化)

| 阶段 | 加速后周期 | 流水化周期 | 时间 (ms) |
|------|-----------|-----------|-----------|
| S1 Feature Extract | 81,723K | 77,636K | 77.64 |
| S2 Depth Predict | 212,359K | 186,876K | 186.88 |
| S3 Gaussian Gen | 157,172K | 152,457K | 152.46 |
| GGU Post | 391K | 0 (hidden) | 0 |
| **总计** | **451,645K** | **416,970K** | **417.0** |

### 6.3 优化后性能 (FSDR + SAES v3)

| 阶段 | 优化项 | 节省 | 流水化周期 | 时间 (ms) |
|------|-------|------|-----------|-----------|
| S1 | — | — | 77,636K | 77.64 |
| S2 | SAES L0+L1 跳过 + FSDR 窄化搜索 | -51.7% | ~90,336K | ~90.34 |
| S3 | SAES L0+L1+L2 跳过 | -23.4% | ~116,837K | ~116.84 |
| GGU | — | hidden | 0 | 0 |
| **总计** | | | **~284,809K** | **~284.8** |

---

## 7. 与 GPU 平台对比

所有 GPU 时间为基于 RTX 3060 实测结果的**估算值**（方法论：60% 计算受限 + 40% 内存受限的加权缩放）。

| 平台 | 时间 (ms) | SCARF 对比 |
|------|-----------|-----------|
| RTX A6000 (est.) | 62.1 | SCARF 4.59× 慢 |
| RTX 3060 (实测) | 161.3 | SCARF 1.77× 慢 |
| **Jetson AGX Orin 64GB (est.)** | **345.2** | **SCARF 1.21× 快** |
| Jetson Orin NX 16GB (est.) | 882.8 | SCARF 3.10× 快 |
| Jetson AGX Xavier (est.) | 1051.0 | SCARF 3.69× 快 |

SCARF 在边缘部署场景下（对标 Jetson Orin）实现了 **21% 的性能提升**，同时功耗仅 4.69W（vs Orin 40W），能效比提升约 **10×**。

---

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| [encoder-units-architecture.md](encoder-units-architecture.md) | ConvEngine, GEMM, BilinearUnit 等计算单元详细架构 |
| [dsu-architecture.md](dsu-architecture.md) | DSU (深度搜索单元) 架构 |
| [ggu-architecture.md](ggu-architecture.md) | GGU (高斯生成单元) 架构 |
| [feature-extractor-architecture.md](feature-extractor-architecture.md) | 特征提取器硬件架构 |
| [fsdr-saes-mechanisms.md](fsdr-saes-mechanisms.md) | FSDR + SAES 优化机制详解 |
| [hardware-resource-summary.md](hardware-resource-summary.md) | 28nm 功耗与面积估算 |
