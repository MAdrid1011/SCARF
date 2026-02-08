# FSGR 与 SAES 优化机制详解

## 1. 概述

SCARF 加速器采用两项关键优化技术，分别从不同维度减少推理计算量：

| 技术 | 利用的冗余 | 优化阶段 | 核心思想 |
|------|-----------|---------|---------|
| **FSGR** (Feature-Similarity Gaussian Reuse) | 2D 特征空间的语义相似性 | S2 (深度预测) | 缓存特征→深度映射，窄化搜索空间 |
| **SAES** (Scene-Adaptive Early-Stopping) | 3D 高斯基元的空间连续性 | S2 + S3 | 多级 tile 分类，跳过冗余计算 |

两者协同工作的方式：
1. SAES 首先在 tile 级别判断哪些 tile 可以跳过（L0/L1 跳过 S2+S3，L2 跳过 S3）
2. FSGR 对 SAES 未跳过的剩余像素进行逐像素处理，窄化 S2 搜索空间
3. 两者的节省**不重叠**（FSGR 只处理 non-SAES 像素），可以直接相加

```
所有像素 (65,536)
  │
  ├── SAES L0 tiles (feature-uniform): ~7% → 跳过 S2+S3
  ├── SAES L1 tiles (depth-uniform):   ~13% → 跳过 S2+S3
  ├── SAES L2 tiles (gaussian-similar): ~3% → 跳过 S3
  └── 剩余像素 (~77%)
       │
       ├── FSGR guided (~73%): 窄化 S2 搜索 (32 候选代替 128)
       └── FSGR not guided (~27%): 完整 S2 搜索
```

---

## 2. FSGR: Feature-Similarity Gaussian Reuse

### 2.1 设计动机

在 3DGS 编码器的 S2 阶段，深度预测的核心是**cost volume 构建**——对每个像素评估 D=128 个深度候选值。这是 S2 中最耗时的操作（占加速后 S2 的 ~75%），且主要受内存带宽限制（需要为每个候选从目标特征图做双线性采样）。

**关键观察**：在 2D 特征空间中相似的像素，其最优深度通常也非常接近。如果能缓存已计算像素的深度，后续相似像素可以只搜索缓存深度附近的小范围，大幅减少 cost volume 计算。

### 2.2 窄化深度搜索 (Narrowed Depth Search)

FSGR 的核心操作：

```
传统搜索: 评估 128 个深度候选 → 选最优
FSGR 搜索: 从缓存获取 depth_cached → 只评估 32 个候选 (centered on depth_cached)
```

**对比其他方案**：

| 方案 | S2 节省 | S3 节省 | 质量损失 | 问题 |
|------|---------|---------|---------|------|
| 完全跳过 S2 (reuse depth) | 100% | 0% | ~1.4% | 深度不准导致位置偏移 |
| 完全跳过 S2+S3 (reuse gaussian) | 100% | 100% | ~2% | 高斯参数不匹配 |
| **窄化搜索 (FSGR)** | **~57%** | **0%** | **≈0%** | **几乎无损** |

窄化搜索的优势：32 个候选仍然执行完整的 cost volume + regression，只是搜索范围更小。只要真实最优深度在 ±25% 窗口内（这对相似特征的像素几乎总是成立的），就能找到完全相同的最优深度。

### 2.3 ASIC 硬件实现

#### 2.3.1 LSH 特征哈希

将 128 维特征向量压缩为 16-bit 签名，用于快速相似性查找：

```
feature [128-dim float]
  │
  ▼  随机投影矩阵 P ∈ {-1, +1}^{16×128}  (ROM 存储)
sign(P @ feature) → 16-bit signature
```

- 硬件实现：16 个点积 + 符号提取
- 存储：128 × 16 × 1-bit = 256 字节 ROM
- 延迟：1 周期（全并行）

#### 2.3.2 缓存表 (Cache Table)

512 条目的 CAM (Content-Addressable Memory)：

```
┌─────────────────────────────────────────────────┐
│ Cache Entry (每条 70+ bits):                     │
│   signature[16]  position[16]  depth[16]         │
│   best_idx[5]    peak_prob[8]  confidence[8]     │
│   valid[1]                                       │
└─────────────────────────────────────────────────┘
```

查找过程：
1. 计算输入特征的 LSH 签名
2. 与缓存中所有条目的签名并行比较汉明距离
3. 选择汉明距离最小的匹配条目

#### 2.3.3 引导判决逻辑

匹配到缓存条目后，FSGR 根据以下 **ASIC 可实现的标准** 决定是否引导：

```python
can_guide = (
    hamming_distance ≤ 3         # 特征相似度足够
    AND peak_confidence ≥ 0.80   # 原始深度预测有足够置信度
    AND depth_consistency_check   # 深度一致性检查通过
)
```

#### 2.3.4 深度一致性检查 (Depth Consistency Check)

**防止在深度不连续处窄化搜索**的关键安全机制：

```
当前像素位置: (y, x)
查看最近 8 个已计算的邻域像素深度
如果缓存深度与邻域深度的相对差异 > 5%:
  → 深度不一致，放弃引导 → 完整 128 候选搜索
否则:
  → 深度一致，允许引导 → 32 候选窄化搜索
```

这确保在物体边缘（深度跳变处）不会使用窄化搜索，从而保证质量。

硬件实现：
- 8-entry 寄存器文件存储最近像素深度
- 比较器阵列计算相对差异
- 单周期判决

### 2.4 节省模型

FSGR 的节省**仅来自 cost_volume**（保守、可防御的模型）：

```
每个 guided 像素的 S2 节省:
  = cv_fraction_scaled × 0.75
  = 75.0% × 75%
  = 56.3%

原因:
  - cost_volume 占加速后 S2 的 75.0% (内存受限，1.5× 加速)
  - 窄化搜索将 128 候选减为 32 候选 → 75% cost_volume 减少
  - U-Net, depth_head, regression 处理完整空间分辨率，不因单像素候选减少而节省
  - 不额外计算 "带宽奖励"——cost volume 减少已包含更少的内存读取
```

S3 不受影响（FSGR 不修改 S3 计算）。

### 2.5 质量影响

| 场景 | 占比 | 质量影响 |
|------|------|---------|
| Guided + depth in window | ~99.99% | **零损失**（搜索仍包含最优深度） |
| Guided + depth out of window | <0.01% | 轻微 means 偏移（极罕见） |
| Depth inconsistent → fallback | ~17% | **零损失**（使用完整搜索） |
| Not guided → full search | ~10% | **零损失** |

FSGR 对渲染质量的影响在实验中测量为 **0.0000% PSNR 损失**。

---

## 3. SAES: Scene-Adaptive Early-Stopping

### 3.1 设计动机

在 3D 场景中，相邻像素的高斯基元参数往往高度相似（来自同一平面/表面的连续区域）。SAES 利用这种**3D 空间连续性**，在 tile 级别判断哪些区域可以通过少量探针像素推断其余像素的高斯参数。

### 3.2 多级 Tile 分类 (v3 Multi-Level)

SAES 对每个 4×4 tile（16 像素）执行三级分类：

```
4×4 Tile 布局:
┌───┬───┬───┬───┐
│   │   │   │   │  (0,0) (0,1) (0,2) (0,3)
├───┼───┼───┼───┤
│   │ P │ P │   │  (1,0) [1,1] [1,2] (1,3)
├───┼───┼───┼───┤     ↑ Probe 像素
│   │ P │ P │   │  (2,0) [2,1] [2,2] (2,3)
├───┼───┼───┼───┤
│   │   │   │   │  (3,0) (3,1) (3,2) (3,3)
└───┴───┴───┴───┘

P = Probe 像素 (center quad, 4 个)
其余 12 个像素可能被插值代替
```

#### Level 0: 特征预过滤 (Feature Pre-Filter)

```
条件: S1 特征方差 < threshold (0.012)

判断方式:
  1. 取 tile 内 16 个像素的 128 维 S1 特征
  2. 归一化后计算每个通道的标准差
  3. 取通道均值作为方差分数
  4. 若方差 < 阈值 → 特征均匀 tile

处理:
  - 4 个 probe 像素正常通过 S2+S3
  - 12 个非 probe 像素的外观参数 (covariance, harmonics, opacity)
    从 4 个 probe 双线性插值得到
  - 位置 (means) 保持原始值不变

节省: 75% S2 + 75% S3 (跳过 12/16 像素的计算)
```

#### Level 1: 深度均匀性 (Depth-Based)

```
条件: 4 个 probe 像素的深度相对标准差 < threshold (0.005)
      且 probe 高斯相似度 ≥ 0.95

判断方式:
  1. L0 未通过的 tile 进入 L1 检查
  2. 4 个 probe 像素正常通过 S2 获得深度
  3. 计算 probe 深度的相对标准差 (std/mean)
  4. 若均匀 → 还需检查 probe 高斯外观相似性

处理:
  - 4 个 probe 像素正常通过 S2+S3
  - 12 个非 probe 像素从 4 个 probe 插值外观
  - 位置保持原始值

节省: 75% S2 + 75% S3
```

#### Level 2: 高斯相似性 (Gaussian Similarity)

```
条件: 4 个 probe 高斯的综合相似度 > threshold (0.995)

判断方式:
  1. L0、L1 未通过的 tile 进入 L2
  2. 所有 16 个像素正常通过 S2 获得深度
  3. 4 个 probe 像素通过 S3 获得完整高斯参数
  4. 计算 probe 间的综合相似度:
     sim = 0.30 × cov_sim + 0.30 × sh_sim
         + 0.15 × opacity_sim + 0.25 × position_sim

处理:
  - S2 已完成 (所有 16 像素)
  - 仅 S3 被跳过: 12 个非 probe 从 4 个 probe 插值外观

节省: 0% S2 + 75% S3 (仅跳过高斯生成)
```

### 3.3 双线性插值机制

对于 L0/L1/L2 tile 中的 12 个非 probe 像素，使用双线性插值从 4 个 probe 推导外观参数：

```
4 个 Probe 位置 (center quad):
  P00 = (1,1)  P01 = (1,2)
  P10 = (2,1)  P11 = (2,2)

对于非 probe 像素 (y, x):
  ty = clamp((y - 1.0), 0, 1)
  tx = clamp((x - 1.0), 0, 1)

  w00 = (1-ty)(1-tx),  w01 = (1-ty)(tx)
  w10 = (ty)(1-tx),    w11 = (ty)(tx)

  covariance[y,x] = (w00·cov_P00 + w01·cov_P01 + w10·cov_P10 + w11·cov_P11) × 1.02
  harmonics[y,x]  = w00·sh_P00 + w01·sh_P01 + w10·sh_P10 + w11·sh_P11
  opacity[y,x]    = w00·op_P00 + w01·op_P01 + w10·op_P10 + w11·op_P11
```

**关键设计选择**：
- **仅插值外观参数** (covariance, harmonics, opacity)，**位置 (means) 保持原始值**
- 这避免了因位置插值导致的大质量损失（3D 位置对渲染质量极敏感）
- Covariance 乘以 1.02 安全系数，轻微放大以避免接缝

### 3.4 ASIC 可实现的 Probe Cross-Check

在应用插值前，SAES 执行**留一交叉验证 (Leave-One-Out Cross-Check)** 以验证插值质量：

```
对于 4 个 probe，依次:
  1. 用其余 3 个 probe 的平均值预测第 i 个 probe
  2. 计算预测误差: err_i = |predicted_i - actual_i|

max_err = max(err_0, err_1, err_2, err_3)

if max_err > cross_check_threshold (0.015):
  → 该 tile 降级为 Full (不插值)
else:
  → 通过验证，执行插值
```

这个检查是 **ASIC 可实现的**：
- 仅使用已计算的 probe 数据
- 不需要 oracle/ground-truth
- 硬件实现：4 个加法器 + 比较器，2-3 周期

### 3.5 节省汇总

| 级别 | 典型占比 | S2 节省 | S3 节省 | 质量影响 |
|------|---------|---------|---------|---------|
| L0 (特征均匀) | ~7-10% tiles | 75% | 75% | 极小 (均匀区域) |
| L1 (深度均匀) | ~13-18% tiles | 75% | 75% | 小 (平面区域) |
| L2 (高斯相似) | ~3-5% tiles | 0% | 75% | 小 (相似外观区域) |
| Full | ~67-77% tiles | 0% | 0% | — |
| **加权总计** | | **~20%** | **~23%** | **~0.75%** |

### 3.6 质量影响分析

SAES 的质量损失主要来自插值近似：

- L0 tiles：特征方差极低的区域（如天空、墙壁），插值几乎无误差
- L1 tiles：深度均匀区域（如平面地面），外观连续，插值误差小
- L2 tiles：仅外观插值，深度已完整计算，误差最小

实测质量损失：**~0.75% relative PSNR** (约 -0.21 dB)

---

## 4. 协同工作机制

### 4.1 处理顺序

```
  S1 特征提取完成
  │
  ▼
  SAES: classify_tiles_by_features()  ← 使用 S1 特征
  │  识别 L0 tiles
  │
  ▼
  S2 深度预测 (所有 non-L0 像素 + L0 probe 像素)
  │
  ▼
  SAES: check_depth_uniformity()  ← 使用 S2 probe 深度
  │  识别 L1 tiles
  │
  ▼
  S3 高斯生成 (所有 non-L0-L1 像素 + L0/L1 probe 像素)
  │
  ▼
  SAES: compute_tile_similarity()  ← 使用 S3 probe 高斯
  │  识别 L2 tiles
  │
  ▼
  FSGR: process_pixel()  ← 对 non-SAES 的 S2 像素逐个处理
  │  缓存 + 窄化搜索
  │
  ▼
  SAES: interpolate_tile()  ← 对 L0/L1/L2 tiles 执行插值
  │
  ▼
  所有高斯基元就绪 → splatting 渲染
```

### 4.2 节省叠加

FSGR 仅作用于 SAES 未覆盖的剩余像素。设：
- `saes_total` = L0 + L1 比例（这些像素 S2 被完全跳过）
- `fsgr_ratio` = FSGR 在非 SAES 像素中的引导比例
- `fsgr_per_pixel` = 每个 guided 像素的 S2 节省率

则：

```
combined_S2_saving = saes_s2_saving + fsgr_ratio × (1 - saes_total) × fsgr_per_pixel

例如:
  saes_s2_saving = 20.0%
  fsgr_ratio = 73.4%
  fsgr_per_pixel = 56.3%
  remaining = 1 - 20.0% = 80.0%

  fsgr_s2_saving = 73.4% × 80.0% × 56.3% = 33.1%
  combined_S2 = 20.0% + 33.1% = 53.1%
```

这两项技术的节省是**独立且可加的**，不存在双重计算。

### 4.3 典型性能提升

| 配置 | S2 节省 | S3 节省 | 总时间 (ms) | vs 无优化 |
|------|---------|---------|-----------|----------|
| Base ASIC (无优化) | — | — | 417.0 | 1.00× |
| +FSGR only | 41.3% | — | 339.8 | 1.23× |
| +SAES only | 20.0% | 23.4% | 344.0 | 1.21× |
| **+FSGR+SAES** | **51.7%** | **23.4%** | **284.8** | **1.46×** |

---

## 5. 实现文件

| 模块 | 文件路径 | 主要类/函数 |
|------|---------|------------|
| FSGR 模拟器 | `fsgr/narrowed_search_simulator.py` | `FSGRSimulator` |
| FSGR 类型定义 | `fsgr/types.py` | `FSGRConfig`, `CacheEntry` |
| FSGR LSH 哈希 | `fsgr/lsh_hasher.py` | `LSHHasher` |
| FSGR 缓存表 | `fsgr/cache_table.py` | `CacheTable` |
| SAES 模拟器 | `saes/progressive_saes.py` | `ProgressiveSAES`, `apply_progressive_saes` |
| 消融实验 + 性能模型 | `scripts/demo.py` | `SavingsTracker.compute_ablation` |

---

## 6. 与论文的关系

本文档中描述的所有机制均在 `scripts/demo.py` 中通过**真实推理模拟**验证：

1. **真实消融实验**：分别运行 4 种配置（base, +FSGR, +SAES, +FSGR+SAES），每种配置独立渲染输出图像
2. **质量指标真实测量**：PSNR/SSIM 基于实际渲染图像与 ground truth 对比
3. **ASIC 可实现的判决**：FSGR 的引导判决和 SAES 的 tile 分类均基于 ASIC 可用的信息（特征哈希、深度值、高斯参数），不使用 oracle 数据
4. **保守的节省模型**：
   - FSGR 仅计入 cost_volume 节省（不含 U-Net 假设）
   - 不额外计算带宽奖励（cost volume 减少已包含更少的内存读取）
   - 硬件加速分阶段建模（计算受限 2.0× vs 内存受限 1.5×）
5. **probe cross-check 验证**：SAES 的质量保障不依赖后验验证，而是使用 ASIC 可实现的留一交叉检查

---

## 7. 相关文档

| 文档 | 内容 |
|------|------|
| [pipeline-architecture.md](pipeline-architecture.md) | SCARF 流水线各阶段详细描述 |
| [dsu-architecture.md](dsu-architecture.md) | 深度搜索单元硬件架构 |
| [ggu-architecture.md](ggu-architecture.md) | 高斯生成单元硬件架构 |
| [encoder-units-architecture.md](encoder-units-architecture.md) | 计算单元 (ConvEngine, GEMM 等) 架构 |
| [hardware-resource-summary.md](hardware-resource-summary.md) | 28nm ASIC 功耗与面积估算 |
