# SCARF: 面向可泛化 3D 高斯泼溅的可扩展交叉视图加速器

## 摘要

可泛化 3D 高斯泼溅 (Generalizable 3D Gaussian Splatting) 是一种新兴的三维重建技术，无需逐场景优化即可从少量输入视图直接预测 3D 高斯基元，实现前馈式新视图合成。然而，其编码器推理涉及大量非规则访存、跨视图几何计算与密集神经网络子任务的异构级联，使得 GPU 部署在边缘场景下面临功耗与时延瓶颈。本文介绍 **SCARF** (**S**calable **C**ross-view **A**ccelerator for **R**adiance **F**ields)，一款基于 28nm TSMC HPC+ 工艺的 ASIC 推理加速器，通过**计算单元时分复用**、**融合 tile 执行**与**自适应早退机制** (SAES) 实现高效编码器推理，在 4.69W 功耗下达到与 Jetson AGX Orin (40W) 相当的吞吐量，能效比提升 10× 以上。

---

## 1. 设计动机与挑战

### 1.1 可泛化 3DGS 编码器的计算特征

可泛化 3DGS 模型（如 Transplat、MVSplat、DepthSplat）的编码器推理包含三种截然不同的计算模式：

| 计算阶段 | 主要操作 | 计算特征 | 主要瓶颈 |
|---------|---------|---------|---------|
| S1 特征提取 | CNN + Transformer | 规则密集计算 | 计算受限 |
| S2 深度预测 | Cost Volume + U-Net | 非规则随机访存 + 密集卷积 | 混合受限 |
| S3 高斯生成 | Refine U-Net + 参数头 | 密集卷积 | 计算受限 |
| GGU 后处理 | 矩阵运算 + 查找表 | 逐元素独立 | 计算受限 |

**关键挑战**：S2 阶段的 Cost Volume 构建涉及对目标视图特征图的**非规则双线性插值采样**（每个像素 × 128 个深度候选 × 4-tap 插值），产生大量随机 SRAM 访问，成为内存带宽瓶颈。同时，三种模型虽然共享相似的流水线结构，但在卷积参数（kernel size、通道数）、深度候选数（32/64/128）和网络子结构上存在差异。

### 1.2 设计目标

1. **统一数据路径**：三种模型复用同一套计算单元，仅通过配置寄存器切换参数
2. **计算单元时分复用**：ConvEngine、GEMM Unit 等大型计算阵列在不同流水线阶段间复用，避免面积浪费
3. **融合 tile 执行**：受片上 SPM 容量限制，S2+S3 在 tile 粒度融合执行，消除中间结果的片外回写
4. **自适应早退 (SAES)**：基于 tile 级特征/深度/高斯均匀性检测，跳过冗余计算
5. **目标性能**：4.69W 功耗下，性能不低于 Jetson AGX Orin (40W)

---

## 2. 系统总览

### 2.1 顶层架构

```
                        ┌─────────────────────────────────────────────────────┐
                        │                    ScarfTop                         │
                        │                                                     │
  AXI4 DRAM IF ◄───────┤  ┌──────────────────────────────────────────────┐   │
                        │  │           PipelineController                  │   │
                        │  │  ┌─────────┐ ┌──────────┐ ┌──────────┐     │   │
                        │  │  │S1 Ctrl  │ │S2S3 Ctrl │ │SAES Ctrl │     │   │
                        │  │  └────┬────┘ └────┬─────┘ └────┬─────┘     │   │
                        │  │       │           │            │            │   │
                        │  │  ┌────▼───────────▼────────────▼─────┐     │   │
                        │  │  │         资源仲裁器 (Arbiter)        │     │   │
                        │  │  └───────────────┬───────────────────┘     │   │
                        │  └──────────────────┼────────────────────────┘   │
                        │                     │                             │
                        │  ┌──────────────────▼───────────────────────┐   │
                        │  │          共享计算单元群 (Compute Units)     │   │
                        │  │                                           │   │
                        │  │  ┌────────────┐  ┌──────────┐            │   │
                        │  │  │ ConvEngine  │  │ GEMM Unit│            │   │
                        │  │  │ 48×48 SA   │  │ 48×48 OS │            │   │
                        │  │  │ 2304 MACs  │  │ 2304 MACs│            │   │
                        │  │  └────────────┘  └──────────┘            │   │
                        │  │  ┌────────────┐  ┌──────────┐            │   │
                        │  │  │BilinearUnit│  │VectorALU │            │   │
                        │  │  │ 32 samplers│  │ 64-wide  │            │   │
                        │  │  └────────────┘  └──────────┘            │   │
                        │  │  ┌──────┐ ┌──────┐ ┌────────┐           │   │
                        │  │  │ActLUT│ │ Norm │ │Softmax │           │   │
                        │  │  └──────┘ └──────┘ └────────┘           │   │
                        │  └──────────────────────────────────────────┘   │
                        │                                                     │
                        │  ┌────────────────┐  ┌──────────────────────┐   │
                        │  │  GGU Array      │  │  Memory Subsystem    │   │
                        │  │  32 PEs         │  │  WeightBuf  128 KB  │   │
                        │  │  PositionCalc   │  │  FeatureBuf 256 KB  │   │
                        │  │  CovBuilder     │  │  TileSPM     64 KB  │   │
                        │  │  SHRotator      │  │  GEMMBuf     64 KB  │   │
                        │  └────────────────┘  └──────────────────────┘   │
                        │                                                     │
                        │  ┌──────────────────────────────────────────────┐   │
                        │  │              ConfigRegs (MMIO)                │   │
                        │  │  numDepthCandidates | featureDim | imageSize │   │
                        │  │  saesThresholds     | fsgrParams | tileSize  │   │
                        │  └──────────────────────────────────────────────┘   │
                        └─────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

**单实例时分复用 (Single-Instance Time-Multiplexing)**：每种计算单元仅实例化一份（ConvEngine ×1, GEMM ×1, BilinearUnit ×1 等），通过顶层 FSM 控制器在不同流水线阶段之间切换。这一设计避免了为三个模型或三个流水线阶段分别实例化计算单元所带来的面积膨胀，同时也简化了物理设计的布局布线。

**配置驱动的模型无关性**：Transplat、MVSplat、DepthSplat 三种模型的差异（深度候选数 32/64/128、CNN 层结构、Transformer 层数、GroupNorm 分组数等）完全通过 ConfigRegs 中的寄存器字段参数化。数据路径中**不存在任何 if-else 模型分支**，所有操作通过相同的硬件路径执行。

---

## 3. 计算单元微架构

### 3.1 ConvEngine — 2D 卷积引擎

ConvEngine 是面积占比最大的计算单元，采用 **48×48 权重驻留 (Weight-Stationary) 脉动阵列**架构。

**微架构**：

```
               Weight Buffer (128 KB SRAM)
                    │ (预加载权重 tile)
                    ▼
    ┌───────────────────────────────────┐
    │       48×48 Systolic Array        │
    │                                   │
    │  输入特征 ──►  PE[0,0] ──► ... ──► PE[0,47]  ──► 部分和
    │               │                    │
    │              ...                  ...
    │               │                    │
    │            PE[47,0] ──► ... ──► PE[47,47] ──► 部分和
    │                                   │
    └───────────────┬───────────────────┘
                    │ (累加树)
                    ▼
              输出特征 Buffer
```

**数据流**：每个 PE 执行一次乘加 (MAC)。权重从 Weight Buffer 预加载后保持不动 (weight-stationary)，输入特征行从左侧流入，部分和从上方向下累积。一个完整的输出 tile 需要 `ceil(C_in / 48) × ceil(C_out / 48)` 次脉动阵列调用。

**Im2col 控制器**：对于 kernel_size > 1 的卷积，由专用 Im2col FSM 将输入特征图按 kernel 窗口展开为列向量，送入脉动阵列。支持的 kernel size：{1, 3, 5, 7, 9, 14}。

**关键参数**：
- 峰值吞吐：2304 MACs/cycle = 2.304 GMAC/s @ 1 GHz
- 权重缓冲：128 KB (可存储 ~32 个 3×3×128 kernel)
- 输入缓冲：16 行 × 全宽 (支持 stride=2 的卷积无需回读)
- 面积估算：~12 mm² (28nm)，含 SRAM

**ASIC 特有优化**：
1. **Winograd F(2,3)**：对 3×3 卷积，乘法次数从 9 降至 4 (2.25×)，变换矩阵硬连线为组合加法器树
2. **Conv-BN-ReLU 融合**：BN 的 scale/shift 在脉动阵列输出级流水化执行，ReLU 紧随其后，消除中间 SRAM 读写
3. **权重驻留复用**：权重一次加载后，对所有空间位置的输入特征复用，摊薄权重加载开销

### 3.2 GEMM Unit — 通用矩阵乘法单元

GEMM Unit 采用 **48×48 输出驻留 (Output-Stationary) 数据流**，针对 Transformer attention 中的大规模矩阵乘法优化。

**微架构**：

```
    A Matrix Buffer (64 KB)     B Matrix Buffer (64 KB)
         │                           │
         ▼                           ▼
    ┌────────────────────────────────────┐
    │    48×48 Output-Stationary Array   │
    │                                    │
    │    输出 tile C[i,j] 累积在 PE 中    │
    │    A 的行与 B 的列交替流入          │
    │                                    │
    └────────────────┬───────────────────┘
                     │
                     ▼
              输出 Buffer → 写回 SRAM
```

**Tiled 执行**：对于维度超过 48 的矩阵乘法 (M×K × K×N)，按 `[M/48, K, N/48]` 分块执行。每个 tile 的 K 维度循环在阵列内完成累积，避免中间结果写回。

**关键参数**：
- 峰值吞吐：2304 MACs/cycle
- Buffer：64 KB (A 侧 32 KB + B 侧 32 KB)
- 数据格式：FP16 (乘法) + FP32 (累积)
- 面积估算：~12 mm² (28nm)

### 3.3 BilinearUnit — 双线性插值采样器

BilinearUnit 是 Cost Volume 构建的核心，处理特征图在深度候选对应坐标上的**非规则采样**。

**微架构**：

```
    坐标计算 (Fixed-Point 8-bit frac)
         │
         ▼
    ┌──────────────────────────────────┐
    │   32 路并行采样器 Pipeline         │
    │                                  │
    │   Sampler 0:  addr_gen → 4-tap   │──► 采样结果 ch[0:31]
    │   Sampler 1:  addr_gen → 4-tap   │──► 采样结果 ch[32:63]
    │        ...                       │
    │   Sampler 31: addr_gen → 4-tap   │──►
    │                                  │
    │   每个 Sampler: 4 次 SRAM 读      │
    │   + 4 次乘法 + 3 次加法           │
    └──────────────────────────────────┘
```

**流水线设计**：每个采样器分 3 级流水：
1. **地址生成**：将浮点坐标 (u', v') 转换为 4 个邻域地址
2. **SRAM 读取**：读取 4 个邻域像素值 (4-tap)
3. **插值计算**：加权混合得到输出值

**关键参数**：
- 并行度：32 通道同时采样
- 吞吐：1 pixel/cycle (128 通道需 4 cycles)
- 坐标精度：8-bit 小数部分 (定点)
- 面积估算：~0.64 mm² (28nm)

### 3.4 VectorALU — SIMD 向量处理单元

64-wide SIMD 单元，处理逐元素操作（激活函数、归一化的 scale/shift、softmax 的 exp/sum）。

**支持的操作**：
- 算术：add, mul, fma, max, min
- 归约：sum, max (树形归约, log₂(64) = 6 级)
- 特殊函数：exp (LUT + 线性插值), rsqrt (Newton 迭代 2 次)

### 3.5 ActivationUnit — 查找表激活

256 条目 LUT，覆盖 [-4.0, 4.0] 范围，线性插值。支持 ReLU (组合逻辑, 0 周期)、GELU、SiLU、Sigmoid。

### 3.6 NormalizationUnit — 归一化单元

支持 LayerNorm、BatchNorm、InstanceNorm、GroupNorm。均值/方差通过 VectorALU 的归约操作计算，scale/shift 通过 VectorALU 的逐元素 fma 操作应用。

### 3.7 SoftmaxUnit — Softmax 回归单元

两阶段流水：(1) 沿指定维度 exp + 求和归一化得到概率分布；(2) 与深度候选值加权求和得到期望深度值。

---

## 4. 流水线控制器

### 4.1 主 FSM 状态机

PipelineController 是 SCARF 的全局调度中枢，采用层次化 FSM 架构：

```
                        PipelineController (顶层 FSM)
                        ┌─────────────────────────┐
                        │                         │
                ┌───────▼──────┐          ┌───────▼──────┐
                │ S1Controller │          │S2S3Controller│
                │ (特征提取)    │          │(融合 tile)    │
                └──────────────┘          └──────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │SAESController │
                                          │(早退判断)      │
                                          └──────────────┘
```

**顶层 FSM 状态转换**：

```
  IDLE ──► LOAD_CONFIG ──► S1_CNN ──► S1_TRANSFORMER
                                            │
                                            ▼
                                    S2S3_TILE_LOOP ◄──┐
                                            │         │
                                            ▼         │
                                    SAES_CLASSIFY     │
                                     │        │      │
                                 [skip]   [full]     │
                                     │        │      │
                                     │    S2_COSTVOL  │
                                     │        │      │
                                     │    S2_UNET     │
                                     │        │      │
                                     │    S2_DHEAD    │
                                     │        │      │
                                     │    S2_REGRESS  │
                                     │        │      │
                                     │    S3_REFINE   │
                                     │        │      │
                                     │    S3_GHEAD    │
                                     │        │      │
                                     ▼        ▼      │
                                    GGU_PROCESS ──────┘
                                            │  (下一 tile)
                                            ▼
                                          DONE
```

### 4.2 资源仲裁

由于 ConvEngine 和 GEMM Unit 各只有一份实例，资源仲裁器在每个 FSM 状态转换时将计算单元分配给当前阶段：

| FSM 状态 | ConvEngine | GEMM Unit | BilinearUnit | VectorALU |
|----------|-----------|-----------|-------------|-----------|
| S1_CNN | ✓ 分配 | — | — | ✓ (BN/ReLU) |
| S1_TRANSFORMER | — | ✓ QKV | — | ✓ (Softmax/LN) |
| S2_COSTVOL | ✓ 相关性 | — | ✓ warping | — |
| S2_UNET | ✓ conv | — | — | ✓ (BN/ReLU) |
| S2_DHEAD | ✓ 1×1 conv | — | — | — |
| S2_REGRESS | — | ✓ 加权和 | — | ✓ softmax |
| S3_REFINE | ✓ conv | — | — | ✓ (BN/ReLU) |
| S3_GHEAD | ✓ 1×1 conv | — | — | — |

**无竞争保证**：FSM 的状态是互斥的，同一时刻只有一个阶段在执行，因此不会出现多阶段竞争同一计算单元的情况。资源切换开销为 0 周期（仅需改变多路选择器配置）。

### 4.3 S2+S3 融合 tile 执行

受片上 SPM 容量限制（64 KB），无法同时存放整幅图像的深度图和高斯参数。因此 S2 和 S3 采用 **tile 粒度融合执行**：

```
对每个 tile (4×4 = 16 pixels):
  1. 从 FeatureBuffer 加载 tile 特征
  2. 执行 S2 (CostVol → UNet → DepthHead → Regression) → 深度存入 TileSPM
  3. 立即执行 S3 (Refine → GaussHead) → 高斯参数存入 TileSPM
  4. GGU 处理当前 tile 的高斯（与下一 tile 的 S2 重叠）
  5. 输出完整高斯基元到 DRAM
```

**优势**：
- 消除 S2→S3 间的 DRAM 回写（深度始终在 TileSPM 中）
- CostVol (BilinearUnit) 与 UNet (ConvEngine) 使用不同硬件，天然流水重叠
- 零 S2→S3 过渡开销

---

## 5. SAES 硬件集成

### 5.1 概述

SAES (Scene-Adaptive Early-Stopping) 是 SCARF 的核心优化技术，利用 tile 级特征/深度/高斯的空间均匀性，跳过冗余的深度预测与高斯生成计算。

### 5.2 三级分类 FSM

SAESController 在每个 tile 进入 S2S3_TILE_LOOP 前执行分类：

| 级别 | 检查内容 | 硬件实现 | 命中时跳过 |
|------|---------|---------|-----------|
| L0 (特征均匀) | 4 个角点 probe 的特征方差 | VectorALU 归约 | 整个 S2+S3 |
| L1 (深度均匀) | 4 个角点 probe 的深度标准差 | 比较器 | 整个 S2+S3 |
| L2 (高斯交叉验证) | 留一法预测误差 (cosine distance) | 4× 加法 + 点积 | S3 部分 |

### 5.3 轻量级 Probe 路径

对于 L0/L1 命中的 tile，仅 4 个角点 probe 像素走完整 S2+S3 路径，其余 12 个像素通过双线性插值生成。Probe 路径进一步简化：跳过 UNet 精化，仅执行 CostVol → SoftArgmax → 简化高斯回归，开销约为完整 tile 的 10%。

### 5.4 流水线吸收

在融合 S2+S3 tile 流水线中，probe-only tile 的执行时间极短 (~10% full tile)。当一个 probe tile 紧接一个 full tile 时，其计算可被前一 tile 的流水线排空阶段完全吸收，实现**零额外周期开销**。

---

## 6. FSGR 硬件集成

### 6.1 概述

FSGR (Feature-Similarity Gaussian Reuse) 利用相邻像素间的特征相似性，对 Cost Volume 计算进行窄化搜索优化。

### 6.2 硬件实现

FSGR 硬件由三个专用模块组成：

**LSHHashUnit** (对应 `fsgr/lsh_hasher.py`)：
- K=16 个随机超平面投影（存储在片上 ROM 中，16 × 128 × 16bit = 4 KB）
- 并行点积计算，利用 VectorALU 的 64-wide SIMD（128 维特征需 2 cycle）
- 符号位提取 + 打包为 16-bit 签名（组合逻辑，0 cycle）
- 总延迟：3 cycles/签名

**FSGRCache** (对应 `fsgr/cache_table.py`)：
- 512 条目的语义索引缓存（每条目：valid[1] + signature[16] + depth[16] + pixel[20] + LRU[8] = 61 bits）
- **并行 Hamming 距离计算**：512 个 XOR + popcount 单元同时比较（组合逻辑）
- **最小值选择树**：log₂(512) = 9 级比较器树，1 cycle 选出最佳匹配
- **LRU 替换策略**：树形归约找到最老条目，避免组合环路
- 查找延迟：1 cycle，插入延迟：1 cycle
- 面积：~300 LUTs + 4 KB SRAM

**FSGRController** (对应 `fsgr/narrowed_search_simulator.py`)：
- 9 状态 FSM：IDLE → HASH → LOOKUP → DECIDE → NARROW/FULL → INSERT → NEXT_PIXEL → DONE
- 逐像素处理：对 tile 中每个像素，先查缓存再决定搜索范围
- 命中时窄化 Cost Volume 从 D 个候选缩减到 D/4（如 128 → 32）
- 未命中时执行完整搜索，并将结果插入缓存

### 6.3 流水线集成

FSGR 在 PipelineController 的 `S2_FSGRLookup` 状态中执行，位于 SAES 分类之后、CostVol 之前：

```
S2S3_SAESClassify
  │
  ├─ [L0/L1/L2] → ProbeOnly (跳过 S2+S3)
  │
  └─ [Full] → S2_FSGRLookup (FSGR 查缓存)
                │
                ├─ [命中] → CostVol (D/4 候选, 窄化搜索)
                │
                └─ [未命中] → CostVol (D 候选, 完整搜索) → 插入缓存
```

### 6.4 与 SAES 的协同

FSGR 仅作用于 SAES 分类为"Full"的 tile（即未命中任何早退级别的 tile），进一步减少这些 tile 的 Cost Volume 计算量。两者组合可实现最高 67% 的 S2+S3 融合块节省。

---

## 7. GGU 阵列

### 7.1 架构

32 个 GGU PE 并行处理高斯基元的后处理变换。每个 PE 包含：

```
GGU PE:
  ├─ PositionCalc:  3D ray 计算 + depth scaling  (10 cycles)
  ├─ CovBuilder:    quaternion → rotation matrix  (30 cycles)
  │                 R × diag(s²) × R^T           (36 cycles)
  │                 世界坐标变换 R_c2w × cov      (24 cycles)
  ├─ SHRotator:     SH 系数旋转 (degree 2-4)      (80 cycles)
  └─ OpacityMap:    sigmoid LUT                   (2 cycles)
  ────────────────────────────────────────────────
  总计: 187 cycles/Gaussian, 32 PEs 并行 → 4096 Gaussians/batch
```

### 7.2 GGU 隐藏

GGU 处理与 S3 ConvEngine 工作完全并行——当 ConvEngine 生成当前 tile 的高斯参数时，GGU PEs 处理上一 tile 的高斯。由于两者使用独立硬件，GGU 周期在端到端流水线中**完全隐藏**。

---

## 8. 存储子系统

### 8.1 SRAM 分区

| 缓冲区 | 容量 | 端口 | 用途 |
|--------|------|------|------|
| WeightBuffer | 128 KB | 单读 | ConvEngine 权重预加载 |
| FeatureBuffer | 256 KB | 双读写 | S1 输出特征图 / S2 输入 |
| TileSPM | 64 KB | 双读写 | S2→S3 tile 中间数据 (深度、raw Gaussian) |
| GEMMBuffer | 64 KB | 双读写 | GEMM A/B 矩阵暂存 |
| **总计** | **512 KB** | | |

### 8.2 数据复用策略

- **权重预取**：当 ConvEngine 执行当前层时，下一层权重已从 DRAM 预取到 WeightBuffer 的 shadow bank
- **特征双缓冲**：FeatureBuffer 分为 ping/pong 两个 bank，S1 写入 ping 的同时 S2 读取 pong
- **Tile SPM 局部性**：融合 tile 执行确保 S2 的深度输出直接留在 SPM 中供 S3 使用，无需 DRAM 回写

---

## 9. 模型无关设计

### 9.1 配置寄存器

所有模型差异通过 ConfigRegs (MMIO 映射) 参数化：

| 寄存器字段 | 宽度 | Transplat | MVSplat | DepthSplat |
|-----------|------|-----------|---------|------------|
| numDepthCandidates | 8-bit | 128 | 32 | 128 |
| featureDim | 8-bit | 128 | 128 | 128 |
| imageH / imageW | 10-bit | 256 | 256 | 256 |
| cnnLayers | 8-bit | 6 | 6 | 6 |
| transformerLayers | 4-bit | 6 | 6 | 6 |
| normGroups | 4-bit | 8 | 8 | 4 |
| hasDINOv2 | 1-bit | 0 | 0 | 1 |
| saesFeatureVarThresh | 16-bit | 0.090 (FP16) | 0.090 | 0.090 |
| saesCrossCheckThresh | 16-bit | 0.030 (FP16) | 0.030 | 0.030 |
| fsgrCacheSize | 10-bit | 512 | 512 | 512 |

### 9.2 控制流差异

三种模型的控制流差异体现在 FSM 的循环次数和跳转条件上，而非数据路径：

- **S1 阶段**：DepthSplat 额外执行 DINOv2 ViT 分支（由 `hasDINOv2` 寄存器控制，FSM 增加 ViT 循环）
- **S2 CostVol**：MVSplat 仅 32 个深度候选（由 `numDepthCandidates` 控制循环次数）
- **S3 GaussHead**：不同模型的 SH degree 不同，通过 `shDegree` 寄存器控制输出通道数

**关键保证**：数据路径中不存在 `if (model == "transplat")` 类型的条件分支。所有条件均基于**数值参数**（循环计数、通道数、阈值），而非模型身份。

---

## 10. 面积、功耗与性能分析

### 10.1 面积分解 (28nm TSMC HPC+)

| 组件 | 面积 (mm²) | 占比 |
|------|-----------|------|
| ConvEngine (48×48 SA + 128KB SRAM) | 12.0 | 36.1% |
| GEMM Unit (48×48 OS + 64KB SRAM) | 12.0 | 36.1% |
| BilinearUnit (32 samplers) | 0.64 | 1.9% |
| GGU PEs (32 PEs) | 1.28 | 3.9% |
| SRAM (512 KB total) | 0.81 | 2.4% |
| I/O + Pads | 5.50 | 16.6% |
| Control + Other | 1.01 | 3.0% |
| **总计** | **33.2** | **100%** |

**芯片尺寸**：5.8 × 5.8 mm

### 10.2 功耗分解

| 组件 | 动态功耗 (mW) | 占比 |
|------|-------------|------|
| ConvEngine | 2028 | 47.5% |
| GEMM Unit | 691 | 16.2% |
| SRAM (全部) | 1089 | 25.5% |
| BilinearUnit | 109 | 2.6% |
| 其他 (VectorALU, GGU, I/O, 控制) | 351 | 8.2% |
| **动态合计** | **4268** | |
| 静态漏电 | 420 | |
| **总功耗** | **4688 mW (4.69W)** | |

### 10.3 性能指标

| 指标 | 数值 |
|------|------|
| 峰值吞吐 | 9.22 TOPS |
| 平均吞吐 | 4.84 TOPS |
| 能效 | 1.03 TOPS/W |
| 面积效率 | 0.15 TOPS/mm² |

### 10.4 与 GPU 平台对比

| 平台 | 推理时间 | 功耗 | 能耗/帧 | vs SCARF 能效 |
|------|---------|------|---------|-------------|
| **SCARF ASIC** | **155 ms** | **4.69 W** | **726 mJ** | **1.0×** |
| RTX A6000 | 62 ms | 200 W | 12,379 mJ | 17× 更多能耗 |
| Jetson AGX Orin | 344 ms | 40 W | 13,760 mJ | 19× 更多能耗 |
| Jetson Orin NX | 880 ms | 15 W | 13,195 mJ | 18× 更多能耗 |

---

## 11. Chisel RTL 实现对应关系

| Python 仿真器 | Chisel 模块 | 说明 |
|-------------|-----------|------|
| `encoder/conv_engine.py` | `scarf.compute.ConvEngine` | 48×48 脉动阵列 + Im2col FSM |
| `encoder/gemm_unit.py` | `scarf.compute.GEMMUnit` | 48×48 OS dataflow |
| `encoder/bilinear_unit.py` | `scarf.compute.BilinearUnit` | 32-sampler pipeline |
| `encoder/activation_unit.py` | `scarf.compute.ActivationUnit` | 256-entry LUT |
| `encoder/normalization_unit.py` | `scarf.compute.NormUnit` | Mean/Var reduction |
| `encoder/softmax_unit.py` | `scarf.compute.SoftmaxUnit` | Exp + normalize |
| `encoder/pooling_unit.py` | `scarf.compute.PoolingUnit` | Max/Avg pool |
| `encoder/pad_unit.py` | `scarf.compute.PadUnit` | Zero/Replicate |
| `ggu/ggu_processor.py` | `scarf.ggu.GGUArray` | 32 PE 阵列 |
| `ggu/position_calculator.py` | `scarf.ggu.PositionCalc` | Depth→3D |
| `ggu/covariance_builder.py` | `scarf.ggu.CovBuilder` | Quat→Cov |
| `ggu/sh_rotator.py` | `scarf.ggu.SHRotator` | SH rotation |
| `depth_predictor/hw_depth_predictor.py` | `scarf.control.PipelineController` | 顶层 FSM (18 状态) |
| `fsgr/lsh_hasher.py` | `scarf.compute.LSHHashUnit` | LSH 哈希签名生成 |
| `fsgr/cache_table.py` | `scarf.memory.FSGRCache` | 512 条目语义缓存 (CAM) |
| `fsgr/narrowed_search_simulator.py` | `scarf.control.FSGRController` | 窄化搜索 FSM |
| `encoder/deformable_attention_unit.py` | BilinearUnit + VectorALU (组合) | 可变形注意力 = 采样 + 加权求和 |
| — | `scarf.control.ConfigRegs` | MMIO 寄存器 (新增) |
| — | `scarf.ScarfTop` | 顶层集成 (新增) |

---

## 参考文献

1. Charatan et al., "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction," CVPR 2024.
2. Chen et al., "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images," ECCV 2024.
3. Xu et al., "DepthSplat: Connecting Gaussian Splatting and Depth," arXiv 2024.
4. Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks," CVPR 2016.
5. Ham et al., "A³: Accelerating Attention Mechanisms in Neural Networks," HPCA 2020.
6. Horowitz, "1.1 Computing's Energy Problem," ISSCC 2014.
7. Bolya et al., "Token Merging: Your ViT But Faster," ICLR 2023.
