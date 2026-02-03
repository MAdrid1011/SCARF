# Implementation Plan: FSDR and 3DGS ASIC Hardware Simulator

## Goal

实现完整的可泛化 3DGS 推理加速器硬件模拟器，包括：
1. **FSDR (Feature-Similarity Depth Reuse)** - 特征相似深度复用单元
2. **DSU (Depth Search Unit)** - 深度搜索单元
3. **GGU (Gaussian Generation Unit)** - 高斯生成单元
4. **统一适配层** - 支持 Transplat、MVSPlat、DepthSplat

**Success Criteria:**
- FSDR 实现完整的三级修正策略（直接复用/插值修正/轻量验证）
- 所有组件可独立测试，通过 100% 单元测试
- 硬件资源估算与 Design1.md 规格一致（FSDR < 2KB SRAM, < 5 cycles）
- 支持三种模型的统一接口，无模型特定代码在核心模块中
- 集成测试验证端到端性能（目标：深度搜索阶段访存节省 ~68%）

**Out of Scope:**
- 实际 RTL 实现（只做 Python 模拟器）
- Rendering pipeline 加速（已有成熟实现）
- Feature backbone 加速（通用 CNN/Transformer 加速器，复杂度高）

---

## Codebase Analysis

### Existing Files (SAES already implemented)

```
SCARF/
├── saes/                    # SAES 已实现 (1497 LOC)
│   ├── types.py             # 209 LOC - Gaussian, TileConfig 等
│   ├── similarity_evaluator.py  # 259 LOC
│   ├── decision_controller.py   # 138 LOC
│   ├── gaussian_merger.py       # 234 LOC
│   ├── profiler.py              # 304 LOC
│   └── tile_processor.py        # 308 LOC
├── tests/saes/              # SAES 测试 (1820 LOC)
└── docs/                    # 文档 (3308 LOC)
```

### Files to Create

```
SCARF/
├── fsdr/                    # FSDR 核心模块
│   ├── __init__.py          # 模块导出 (Est: 30 LOC)
│   ├── types.py             # FSDR 数据结构 (Est: 180 LOC)
│   ├── lsh_hasher.py        # LSH 签名生成 (Est: 150 LOC)
│   ├── cache_table.py       # 缓存表管理 (Est: 220 LOC)
│   ├── depth_corrector.py   # 深度修正策略 (Est: 200 LOC)
│   ├── light_verifier.py    # 轻量验证模块 (Est: 150 LOC)
│   ├── fsdr_processor.py    # FSDR 主处理器 (Est: 280 LOC)
│   ├── profiler.py          # FSDR 性能分析 (Est: 180 LOC)
│   └── README.md            # 模块文档 (Est: 150 LOC)
│
├── dsu/                     # 深度搜索单元
│   ├── __init__.py          # (Est: 25 LOC)
│   ├── types.py             # DSU 数据结构 (Est: 120 LOC)
│   ├── cost_volume.py       # Cost volume 计算 (Est: 200 LOC)
│   ├── depth_sampler.py     # 深度采样器 (Est: 180 LOC)
│   ├── softmax_aggregator.py # Softmax 聚合 (Est: 120 LOC)
│   ├── dsu_processor.py     # DSU 主处理器 (Est: 220 LOC)
│   └── README.md            # (Est: 100 LOC)
│
├── ggu/                     # 高斯生成单元
│   ├── __init__.py          # (Est: 25 LOC)
│   ├── types.py             # GGU 数据结构 (Est: 100 LOC)
│   ├── covariance_builder.py # 协方差构建 (Est: 150 LOC)
│   ├── position_calculator.py # 3D 位置计算 (Est: 120 LOC)
│   ├── sh_rotator.py        # 球谐旋转 (Est: 140 LOC)
│   ├── ggu_processor.py     # GGU 主处理器 (Est: 180 LOC)
│   └── README.md            # (Est: 80 LOC)
│
├── adapters/                # 模型适配层
│   ├── __init__.py          # (Est: 20 LOC)
│   ├── base_adapter.py      # 抽象基类 (Est: 100 LOC)
│   ├── transplat_adapter.py # Transplat 适配 (Est: 180 LOC)
│   ├── mvsplat_adapter.py   # MVSPlat 适配 (Est: 150 LOC)
│   ├── depthsplat_adapter.py # DepthSplat 适配 (Est: 150 LOC)
│   └── README.md            # (Est: 80 LOC)
│
├── integration/             # 集成模块
│   ├── __init__.py          # (Est: 20 LOC)
│   ├── accelerator.py       # 完整加速器 (Est: 300 LOC)
│   ├── pipeline.py          # 流水线控制 (Est: 200 LOC)
│   └── README.md            # (Est: 80 LOC)
│
├── tests/
│   ├── fsdr/                # FSDR 测试
│   │   ├── __init__.py      # (Est: 1 LOC)
│   │   ├── conftest.py      # 测试夹具 (Est: 180 LOC)
│   │   ├── test_lsh_hasher.py       # (Est: 200 LOC)
│   │   ├── test_cache_table.py      # (Est: 250 LOC)
│   │   ├── test_depth_corrector.py  # (Est: 220 LOC)
│   │   ├── test_light_verifier.py   # (Est: 180 LOC)
│   │   ├── test_fsdr_processor.py   # (Est: 280 LOC)
│   │   └── test_integration.py      # (Est: 300 LOC)
│   │
│   ├── dsu/                 # DSU 测试
│   │   ├── __init__.py      # (Est: 1 LOC)
│   │   ├── conftest.py      # (Est: 120 LOC)
│   │   ├── test_cost_volume.py      # (Est: 180 LOC)
│   │   ├── test_depth_sampler.py    # (Est: 150 LOC)
│   │   └── test_dsu_processor.py    # (Est: 200 LOC)
│   │
│   ├── ggu/                 # GGU 测试
│   │   ├── __init__.py      # (Est: 1 LOC)
│   │   ├── conftest.py      # (Est: 100 LOC)
│   │   ├── test_covariance_builder.py  # (Est: 150 LOC)
│   │   └── test_ggu_processor.py       # (Est: 180 LOC)
│   │
│   ├── adapters/            # 适配器测试
│   │   ├── __init__.py      # (Est: 1 LOC)
│   │   └── test_adapters.py # (Est: 250 LOC)
│   │
│   └── integration/         # 集成测试
│       ├── __init__.py      # (Est: 1 LOC)
│       └── test_full_pipeline.py  # (Est: 350 LOC)
│
└── docs/
    ├── fsdr-architecture.md       # FSDR 架构文档 (Est: 500 LOC)
    ├── dsu-architecture.md        # DSU 架构文档 (Est: 300 LOC)
    ├── ggu-architecture.md        # GGU 架构文档 (Est: 250 LOC)
    ├── hardware-resource-summary.md # 硬件资源汇总 (Est: 200 LOC)
    └── multi-model-integration.md # 多模型集成指南 (Est: 300 LOC)
```

### Files to Modify

- `SCARF/README.md` - 添加 FSDR/DSU/GGU 模块说明
- `SCARF/requirements.txt` - 添加新依赖（如有）
- `SCARF/saes/types.py` - 可能需要共享一些基础类型

---

## Interface Design

### 1. FSDR Interfaces

#### CacheEntry (types.py)
```python
@dataclass
class CacheEntry:
    """FSDR 缓存条目 - 70 bits total"""
    signature: int           # 16 bits - LSH 签名
    position: Tuple[int,int] # 16 bits - 像素坐标 (u, v)
    best_depth: float        # 16 bits - FP16 最优深度
    best_idx: int            # 5 bits - 最优深度索引
    peak_prob: float         # 8 bits - 最优概率 (量化 0-1)
    second_offset: int       # 5 bits - 次优索引偏移 (signed)
    spread: float            # 8 bits - 分布宽度 (量化)
    valid: bool              # 1 bit - 有效标志

@dataclass
class FSDRConfig:
    """FSDR 配置"""
    cache_size: int = 128                  # 缓存条目数
    lsh_dim: int = 16                      # LSH 签名维度
    feature_dim: int = 128                 # 特征向量维度
    hamming_threshold: int = 4             # 汉明距离阈值
    high_confidence_threshold: float = 0.8 # 高置信度阈值
    medium_confidence_threshold: float = 0.5 # 中等置信度阈值
    hamming_direct_reuse: int = 2          # 直接复用汉明阈值
    hamming_interpolate: int = 3           # 插值汉明阈值

@dataclass 
class FSDRResult:
    """FSDR 处理结果"""
    depth: float
    source: str  # 'direct_reuse', 'interpolation', 'light_verify', 'full_search'
    cache_hit: bool
    hamming_distance: Optional[int]
    timing_ns: Dict[str, int]
```

#### LSHHasher
```python
class LSHHasher:
    def __init__(self, feature_dim: int, lsh_dim: int, seed: int = 42)
    def hash(self, feature: Tensor) -> int  # Returns lsh_dim-bit signature
    def batch_hash(self, features: Tensor) -> Tensor  # [N] signatures
```

#### CacheTable
```python
class CacheTable:
    def __init__(self, config: FSDRConfig)
    def lookup(self, signature: int) -> Tuple[Optional[CacheEntry], int]  # entry, hamming_dist
    def insert(self, entry: CacheEntry) -> None
    def update(self, idx: int, depth: float, boost_confidence: bool) -> None
    def get_statistics(self) -> Dict
```

#### DepthCorrector
```python
class DepthCorrector:
    def __init__(self, config: FSDRConfig, depth_candidates: Tensor)
    def direct_reuse(self, entry: CacheEntry) -> float
    def interpolate(self, entry: CacheEntry, hamming_dist: int) -> float
    def decide_strategy(self, entry: CacheEntry, hamming_dist: int) -> str
```

#### LightVerifier
```python
class LightVerifier:
    def __init__(self, config: FSDRConfig)
    def verify(
        self, 
        entry: CacheEntry,
        feature: Tensor,
        depth_candidates: Tensor,
        cost_fn: Callable,
    ) -> Tuple[float, int]  # depth, num_searches
```

#### FSDRProcessor (主接口)
```python
class FSDRProcessor:
    def __init__(self, config: FSDRConfig, depth_candidates: Tensor)
    
    def process_pixel(
        self,
        feature: Tensor,           # [C] 特征向量
        position: Tuple[int, int], # (u, v)
        cost_fn: Callable,         # 深度搜索代价函数
        prob_fn: Callable,         # 概率分布提取函数
    ) -> FSDRResult
    
    def get_profiling(self) -> FSDRProfilingResult
    def reset(self) -> None
```

### 2. DSU Interfaces

#### DSUConfig
```python
@dataclass
class DSUConfig:
    num_depth_candidates: int = 32      # D
    feature_dim: int = 128              # C
    num_parallel_depths: int = 4        # 并行深度数
```

#### CostVolume
```python
class CostVolume:
    def __init__(self, config: DSUConfig)
    
    def compute_costs(
        self,
        ref_feature: Tensor,       # [C]
        target_features: Tensor,   # [D, C] 从目标图采样的特征
    ) -> Tensor  # [D] costs
    
    def compute_correlation(
        self,
        ref_features: Tensor,      # [B, C, H, W]
        target_features: Tensor,   # [B, D, C, H, W]
    ) -> Tensor  # [B, D, H, W]
```

#### DepthSampler
```python
class DepthSampler:
    def __init__(self, config: DSUConfig)
    
    def sample_target_features(
        self,
        target_feature_map: Tensor,  # [C, H_t, W_t]
        ref_coords: Tensor,          # [N, 2] 参考坐标
        depth_candidates: Tensor,    # [D]
        projection_matrix: Tensor,   # [3, 4]
    ) -> Tensor  # [N, D, C]
```

#### SoftmaxAggregator
```python
class SoftmaxAggregator:
    def __init__(self, config: DSUConfig)
    
    def aggregate(
        self,
        costs: Tensor,            # [D] or [B, D, H, W]
        depth_candidates: Tensor, # [D]
        return_distribution: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]  # depth or (depth, probs)
    
    def extract_statistics(
        self,
        probs: Tensor,            # [D]
        depth_candidates: Tensor, # [D]
    ) -> Tuple[int, float, int, float]  # best_idx, peak_prob, second_idx, spread
```

#### DSUProcessor (主接口)
```python
class DSUProcessor:
    def __init__(self, config: DSUConfig)
    
    def search_depth(
        self,
        ref_feature: Tensor,         # [C]
        target_feature_map: Tensor,  # [C, H, W]
        ref_coord: Tensor,           # [2]
        depth_candidates: Tensor,    # [D]
        projection_matrix: Tensor,   # [3, 4]
    ) -> Tuple[float, Tensor]  # depth, probs
    
    def search_depth_range(
        self,
        ref_feature: Tensor,
        target_feature_map: Tensor,
        ref_coord: Tensor,
        depth_range: Tuple[int, int],  # (start_idx, end_idx)
        depth_candidates: Tensor,
        projection_matrix: Tensor,
    ) -> float  # Optimized for FSDR light_verify
```

### 3. GGU Interfaces

#### GGUConfig
```python
@dataclass
class GGUConfig:
    sh_degree: int = 3              # 球谐次数
    scale_min: float = 0.0005
    scale_max: float = 0.5
```

#### CovarianceBuilder
```python
class CovarianceBuilder:
    def __init__(self, config: GGUConfig)
    
    def build(
        self,
        scales: Tensor,       # [3] 
        rotations: Tensor,    # [4] quaternion
    ) -> Tensor  # [3, 3] covariance
    
    def transform_to_world(
        self,
        covariance: Tensor,   # [3, 3]
        c2w_rotation: Tensor, # [3, 3]
    ) -> Tensor  # [3, 3]
```

#### PositionCalculator
```python
class PositionCalculator:
    def __init__(self)
    
    def compute_position(
        self,
        pixel_coord: Tensor,    # [2]
        depth: float,
        intrinsics: Tensor,     # [3, 3]
        extrinsics: Tensor,     # [4, 4]
    ) -> Tensor  # [3] world position
    
    def get_ray(
        self,
        pixel_coord: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
    ) -> Tuple[Tensor, Tensor]  # origin, direction
```

#### SHRotator
```python
class SHRotator:
    def __init__(self, sh_degree: int)
    
    def rotate(
        self,
        sh_coeffs: Tensor,     # [C, (degree+1)^2]
        rotation: Tensor,      # [3, 3]
    ) -> Tensor  # [C, (degree+1)^2]
```

#### GGUProcessor (主接口)
```python
class GGUProcessor:
    def __init__(self, config: GGUConfig)
    
    def generate_gaussian(
        self,
        pixel_coord: Tensor,     # [2]
        depth: float,
        raw_gaussian: Tensor,    # [C_raw] 包含 scales, rotations, sh
        density: float,
        intrinsics: Tensor,      # [3, 3]
        extrinsics: Tensor,      # [4, 4]
    ) -> Gaussian  # 使用 saes.types.Gaussian
```

### 4. Adapter Interfaces

#### BaseAdapter (Abstract)
```python
class BaseAdapter(ABC):
    @abstractmethod
    def extract_depth_distribution(
        self,
        cost_volume: Tensor,
        depth_candidates: Tensor,
    ) -> Tensor  # [B, D, H, W] probabilities
    
    @abstractmethod
    def get_depth_candidates(
        self,
        near: float,
        far: float,
        num_candidates: int,
    ) -> Tensor  # [D]
    
    @abstractmethod
    def project_to_target(
        self,
        ref_coords: Tensor,       # [N, 2]
        depth: Tensor,            # [N]
        ref_intrinsics: Tensor,
        ref_extrinsics: Tensor,
        tgt_intrinsics: Tensor,
        tgt_extrinsics: Tensor,
    ) -> Tensor  # [N, 2] target coordinates
```

#### TransplatAdapter
```python
class TransplatAdapter(BaseAdapter):
    """Transplat 特定的深度候选和投影逻辑"""
    # Cost volume: 需要取负号再 softmax
    # 深度候选: 逆深度线性采样
```

#### MVSplatAdapter
```python
class MVSplatAdapter(BaseAdapter):
    """MVSPlat 特定的深度候选和投影逻辑"""
    # Correlation volume: 直接 softmax
    # 深度候选: 深度线性采样
```

#### DepthSplatAdapter
```python
class DepthSplatAdapter(BaseAdapter):
    """DepthSplat 特定逻辑 (继承 MVSplat)"""
    # 类似 MVSplat + DINOv2 特征
```

### 5. Integration Interfaces

#### Accelerator
```python
class Accelerator:
    """完整 3DGS 编码器加速器"""
    def __init__(
        self,
        fsdr_config: FSDRConfig,
        saes_config: TileConfig,  # 从 saes.types 导入
        dsu_config: DSUConfig,
        ggu_config: GGUConfig,
        model_type: str,  # 'transplat', 'mvsplat', 'depthsplat'
    )
    
    def process_scene(
        self,
        features: Tensor,          # [B, V, C, H, W]
        intrinsics: Tensor,        # [B, V, 3, 3]
        extrinsics: Tensor,        # [B, V, 4, 4]
        near: Tensor,              # [B, V]
        far: Tensor,               # [B, V]
    ) -> Tuple[List[Gaussian], AcceleratorProfilingResult]
    
    def get_hardware_summary(self) -> Dict
```

---

## Test Strategy

### FSDR Tests (Est: 1630 LOC)

#### test_lsh_hasher.py (200 LOC)
- Test: 相同特征产生相同签名
- Test: 相似特征汉明距离小
- Test: 不同特征汉明距离大
- Test: 批量哈希一致性
- Test: 签名位宽验证（16 bits）

#### test_cache_table.py (250 LOC)
- Test: 空表查找返回 None
- Test: 插入后可查找
- Test: LRU 替换策略
- Test: 置信度加权替换
- Test: 汉明距离计算正确
- Test: 128 条目容量限制

#### test_depth_corrector.py (220 LOC)
- Test: 直接复用策略（高置信度 + 小汉明距离）
- Test: 插值策略（中等置信度或中等汉明距离）
- Test: 轻量验证触发条件（低置信度或大汉明距离）
- Test: 插值系数计算正确
- Test: 边界条件处理

#### test_light_verifier.py (180 LOC)
- Test: 搜索范围基于 spread 正确计算
- Test: 局部搜索结果正确
- Test: 搜索次数在预期范围（3-7 次）
- Test: 边界条件（spread 很小/很大）

#### test_fsdr_processor.py (280 LOC)
- Test: 完整处理流程（未命中 → 命中）
- Test: 三种策略路径覆盖
- Test: 缓存更新逻辑
- Test: 性能计数正确
- Test: 配置参数生效

#### test_integration.py (300 LOC)
- Test: 与 DSU 集成
- Test: 端到端访存节省计算
- Test: 多模型适配器

### DSU Tests (Est: 651 LOC)

#### test_cost_volume.py (180 LOC)
- Test: 单特征代价计算
- Test: 批量相关性计算
- Test: 余弦相似度正确
- Test: GPU/CPU 一致性

#### test_depth_sampler.py (150 LOC)
- Test: 投影坐标正确
- Test: 双线性插值采样
- Test: 边界处理（超出图像范围）

#### test_dsu_processor.py (200 LOC)
- Test: 完整深度搜索
- Test: 局部范围搜索（for FSDR）
- Test: 统计量提取正确

### GGU Tests (Est: 431 LOC)

#### test_covariance_builder.py (150 LOC)
- Test: 协方差矩阵对称正定
- Test: 世界坐标变换正确
- Test: 尺度自适应缩放

#### test_ggu_processor.py (180 LOC)
- Test: 3D 位置计算正确
- Test: 球谐旋转一致性
- Test: 与 saes.types.Gaussian 兼容

### Adapter Tests (Est: 250 LOC)

#### test_adapters.py (250 LOC)
- Test: Transplat 概率分布提取（负代价）
- Test: MVSPlat 概率分布提取（直接相关）
- Test: DepthSplat 继承正确
- Test: 深度候选生成（逆深度 vs 深度）
- Test: 投影计算一致性

### Integration Tests (Est: 350 LOC)

#### test_full_pipeline.py (350 LOC)
- Test: FSDR + SAES 组合
- Test: 端到端性能指标
- Test: 与真实 RE10K 数据（skip if unavailable）
- Test: 硬件资源汇总正确

---

## Implementation Steps

### Phase 1: Documentation (Estimated: 1550 LOC)

**Step 1: FSDR Architecture Documentation** (Est: 500 LOC)
- `docs/fsdr-architecture.md` - FSDR 设计文档
  - Section 1: Overview and Design Goals
  - Section 2: Cache Entry Structure (70-bit breakdown)
  - Section 3: LSH Signature Generation (hardware mapping)
  - Section 4: Three-Level Correction Strategy
  - Section 5: Light Verification Algorithm
  - Section 6: Hardware Resource Estimation
  - Section 7: Integration with DSU
Dependencies: None

**Step 2: DSU Architecture Documentation** (Est: 300 LOC)
- `docs/dsu-architecture.md` - DSU 设计文档
  - Section 1: Cost Volume Computation
  - Section 2: Depth Sampling and Projection
  - Section 3: Softmax Aggregation
  - Section 4: Statistics Extraction
  - Section 5: Hardware Mapping
Dependencies: None (parallel with Step 1)

**Step 3: GGU Architecture Documentation** (Est: 250 LOC)
- `docs/ggu-architecture.md` - GGU 设计文档
  - Section 1: Covariance Construction
  - Section 2: 3D Position Calculation
  - Section 3: Spherical Harmonics Rotation
  - Section 4: Hardware Mapping
Dependencies: None (parallel with Step 1-2)

**Step 4: Hardware Resource Summary** (Est: 200 LOC)
- `docs/hardware-resource-summary.md` - 完整硬件资源汇总
  - All components resource breakdown
  - Total area/power estimation
  - Comparison with baseline
Dependencies: Steps 1-3

**Step 5: Multi-Model Integration Guide** (Est: 300 LOC)
- `docs/multi-model-integration.md` - 多模型集成指南
  - Transplat adapter details
  - MVSPlat adapter details
  - DepthSplat adapter details
  - Unified interface specification
Dependencies: Steps 1-3

### Phase 2: Test Cases (Estimated: 3312 LOC)

**Step 6: FSDR Test Infrastructure** (Est: 180 LOC)
- `tests/fsdr/conftest.py` - FSDR 测试夹具
  - Mock feature vectors
  - Mock depth candidates
  - Mock cost functions
  - Mock probability distributions
Dependencies: Step 1 (FSDR docs)

**Step 7: FSDR Core Tests** (Est: 1130 LOC)
- `tests/fsdr/test_lsh_hasher.py` (200 LOC)
- `tests/fsdr/test_cache_table.py` (250 LOC)
- `tests/fsdr/test_depth_corrector.py` (220 LOC)
- `tests/fsdr/test_light_verifier.py` (180 LOC)
- `tests/fsdr/test_fsdr_processor.py` (280 LOC)
Dependencies: Step 6

**Step 8: DSU Test Infrastructure and Tests** (Est: 651 LOC)
- `tests/dsu/conftest.py` (120 LOC)
- `tests/dsu/test_cost_volume.py` (180 LOC)
- `tests/dsu/test_depth_sampler.py` (150 LOC)
- `tests/dsu/test_dsu_processor.py` (200 LOC)
Dependencies: Step 2 (DSU docs)

**Step 9: GGU Test Infrastructure and Tests** (Est: 431 LOC)
- `tests/ggu/conftest.py` (100 LOC)
- `tests/ggu/test_covariance_builder.py` (150 LOC)
- `tests/ggu/test_ggu_processor.py` (180 LOC)
Dependencies: Step 3 (GGU docs)

**Step 10: Adapter Tests** (Est: 251 LOC)
- `tests/adapters/__init__.py` (1 LOC)
- `tests/adapters/test_adapters.py` (250 LOC)
Dependencies: Step 5 (Multi-model docs)

**Step 11: Integration Tests** (Est: 651 LOC)
- `tests/fsdr/test_integration.py` (300 LOC)
- `tests/integration/__init__.py` (1 LOC)
- `tests/integration/test_full_pipeline.py` (350 LOC)
Dependencies: Steps 6-10

### Phase 3: Implementation (Estimated: 3455 LOC)

**Step 12: FSDR Types** (Est: 180 LOC)
- `fsdr/types.py`
  - CacheEntry dataclass
  - FSDRConfig dataclass
  - FSDRResult dataclass
  - FSDRProfilingResult dataclass
Dependencies: Steps 6-7

**Step 13: LSH Hasher Implementation** (Est: 150 LOC)
- `fsdr/lsh_hasher.py`
  - Random projection matrix initialization
  - Single hash function
  - Batch hash function
  - Hardware-friendly sign extraction
Dependencies: Step 12

**Step 14: Cache Table Implementation** (Est: 220 LOC)
- `fsdr/cache_table.py`
  - LRU + confidence weighted replacement
  - Hamming distance calculation (parallel)
  - Insert/lookup/update operations
  - Statistics collection
Dependencies: Steps 12-13

**Step 15: Depth Corrector Implementation** (Est: 200 LOC)
- `fsdr/depth_corrector.py`
  - Direct reuse strategy
  - Interpolation strategy
  - Strategy decision logic
Dependencies: Step 12

**Step 16: Light Verifier Implementation** (Est: 150 LOC)
- `fsdr/light_verifier.py`
  - Search range calculation based on spread
  - Local depth search
  - Result aggregation
Dependencies: Steps 12, 15

**Step 17: FSDR Processor Implementation** (Est: 280 LOC)
- `fsdr/fsdr_processor.py`
  - Main process_pixel method
  - Cache miss path (full search + cache insert)
  - Cache hit path (three strategies)
  - Statistics update
Dependencies: Steps 13-16

**Step 18: FSDR Profiler Implementation** (Est: 180 LOC)
- `fsdr/profiler.py`
  - Per-pixel timing
  - Path distribution tracking
  - Memory access counting
  - Summary generation
Dependencies: Step 17

**Step 19: FSDR Module Init and README** (Est: 180 LOC)
- `fsdr/__init__.py` (30 LOC)
- `fsdr/README.md` (150 LOC)
Dependencies: Steps 12-18

**Step 20: DSU Types** (Est: 120 LOC)
- `dsu/types.py`
  - DSUConfig dataclass
  - DSUResult dataclass
Dependencies: Steps 8

**Step 21: Cost Volume Implementation** (Est: 200 LOC)
- `dsu/cost_volume.py`
  - Single cost computation (dot product)
  - Batch correlation computation
  - GPU-optimized implementation
Dependencies: Step 20

**Step 22: Depth Sampler Implementation** (Est: 180 LOC)
- `dsu/depth_sampler.py`
  - Projection matrix application
  - Bilinear interpolation sampling
  - Boundary handling
Dependencies: Step 20

**Step 23: Softmax Aggregator Implementation** (Est: 120 LOC)
- `dsu/softmax_aggregator.py`
  - Softmax computation
  - Expected depth calculation
  - Statistics extraction (peak_prob, second_idx, spread)
Dependencies: Step 20

**Step 24: DSU Processor Implementation** (Est: 220 LOC)
- `dsu/dsu_processor.py`
  - Full depth search
  - Range-limited search (for FSDR)
  - Profile collection
Dependencies: Steps 21-23

**Step 25: DSU Module Init and README** (Est: 125 LOC)
- `dsu/__init__.py` (25 LOC)
- `dsu/README.md` (100 LOC)
Dependencies: Steps 20-24

**Step 26: GGU Types** (Est: 100 LOC)
- `ggu/types.py`
  - GGUConfig dataclass
Dependencies: Step 9

**Step 27: Covariance Builder Implementation** (Est: 150 LOC)
- `ggu/covariance_builder.py`
  - Scale mapping
  - Quaternion normalization
  - Covariance matrix construction
  - World transform
Dependencies: Step 26

**Step 28: Position Calculator Implementation** (Est: 120 LOC)
- `ggu/position_calculator.py`
  - Ray computation
  - Depth-based 3D position
Dependencies: Step 26

**Step 29: SH Rotator Implementation** (Est: 140 LOC)
- `ggu/sh_rotator.py`
  - Rotation matrix to SH rotation
  - Coefficient rotation
Dependencies: Step 26

**Step 30: GGU Processor Implementation** (Est: 180 LOC)
- `ggu/ggu_processor.py`
  - Complete Gaussian generation
  - Integration with saes.types.Gaussian
Dependencies: Steps 27-29

**Step 31: GGU Module Init and README** (Est: 105 LOC)
- `ggu/__init__.py` (25 LOC)
- `ggu/README.md` (80 LOC)
Dependencies: Steps 26-30

**Step 32: Base Adapter Implementation** (Est: 100 LOC)
- `adapters/base_adapter.py`
  - Abstract interface
  - Common utilities
Dependencies: Steps 10

**Step 33: Model-Specific Adapters** (Est: 480 LOC)
- `adapters/transplat_adapter.py` (180 LOC)
- `adapters/mvsplat_adapter.py` (150 LOC)
- `adapters/depthsplat_adapter.py` (150 LOC)
Dependencies: Step 32

**Step 34: Adapter Module Init and README** (Est: 100 LOC)
- `adapters/__init__.py` (20 LOC)
- `adapters/README.md` (80 LOC)
Dependencies: Steps 32-33

**Step 35: Integration Pipeline** (Est: 200 LOC)
- `integration/pipeline.py`
  - FSDR + SAES + DSU + GGU orchestration
  - Tile-based processing with FSDR
Dependencies: Steps 19, 25, 31, 34

**Step 36: Complete Accelerator** (Est: 300 LOC)
- `integration/accelerator.py`
  - Scene processing entry point
  - Model selection
  - Hardware summary generation
Dependencies: Step 35

**Step 37: Integration Module Init and README** (Est: 100 LOC)
- `integration/__init__.py` (20 LOC)
- `integration/README.md` (80 LOC)
Dependencies: Steps 35-36

**Step 38: Update Project Files** (Est: 50 LOC)
- `SCARF/README.md` - Add new module documentation
- `SCARF/requirements.txt` - Add dependencies if needed
Dependencies: All previous steps

---

## Total Estimated Complexity

| Phase | LOC | Description |
|-------|-----|-------------|
| **Phase 1: Documentation** | 1,550 | Architecture docs, resource summary, integration guide |
| **Phase 2: Test Cases** | 3,312 | Unit tests, integration tests |
| **Phase 3: Implementation** | 3,455 | Core modules, adapters, integration |
| **Total** | **8,317** | Very Large feature |

### Milestone Strategy

Given the large scope (8,317 LOC), recommend **6 milestone commits**:

**Milestone 1: Documentation Complete** (~Step 5)
- All architecture documentation complete
- Test infrastructure ready
- 0/~100 tests (tests exist but not implemented)
- ~1,550 LOC cumulative

**Milestone 2: FSDR Tests Complete** (~Step 7)
- FSDR test suite complete
- Ready for implementation
- 0/~60 tests (FSDR tests only)
- ~2,860 LOC cumulative

**Milestone 3: All Tests Complete** (~Step 11)
- All unit and integration tests created
- Full test coverage defined
- 0/~100 tests
- ~4,862 LOC cumulative

**Milestone 4: FSDR Implementation** (~Step 19)
- FSDR core fully implemented
- ~30/100 tests passing
- ~6,052 LOC cumulative

**Milestone 5: DSU + GGU Implementation** (~Step 31)
- DSU and GGU fully implemented
- ~60/100 tests passing
- ~7,172 LOC cumulative

**Milestone 6: Full Integration** (~Step 38)
- All components integrated
- All adapters implemented
- 100/100 tests passing
- ~8,317 LOC cumulative (Delivery commit)

---

## Hardware Resource Summary

### FSDR Resources

| Component | LUTs | DSPs | SRAM | Cycles |
|-----------|------|------|------|--------|
| LSH Hasher | 500 | 16 | 0.5KB ROM | 1 |
| Cache Table | 300 | 0 | 1.1KB | 1 |
| Hamming Calculator | 200 | 0 | 0 | 1 |
| Depth Corrector | 150 | 4 | 0 | 1 |
| Light Verifier | 100 | 2 | 0 | varies |
| **FSDR Total** | **1,250** | **22** | **1.6KB** | **<5** |

### DSU Resources (per unit)

| Component | LUTs | DSPs | SRAM | Cycles |
|-----------|------|------|------|--------|
| Feature Sampler | 500 | 8 | 0 | 2 |
| Cost Calculator | 300 | 32 | 0 | 4 |
| Softmax | 400 | 4 | 0 | 8 |
| Statistics | 200 | 2 | 0 | 2 |
| **DSU Total (×4)** | **5,600** | **184** | **0** | **~20** |

### GGU Resources

| Component | LUTs | DSPs | Cycles |
|-----------|------|------|--------|
| Covariance Builder | 400 | 12 | 5 |
| Position Calculator | 200 | 6 | 3 |
| SH Rotator | 300 | 18 | 8 |
| **GGU Total** | **900** | **36** | **16** |

### Overall System

| Module | LUTs | DSPs | SRAM |
|--------|------|------|------|
| FSDR | 1,250 | 22 | 1.6KB |
| DSU ×4 | 5,600 | 184 | 0 |
| GGU | 900 | 36 | 0 |
| SAES | 3,000 | 36 | 0.5KB |
| Control | 1,000 | 0 | 0.5KB |
| **Total** | **11,750** | **278** | **2.6KB** |

**Target**: Mid-range FPGA (Xilinx XC7A100T) or ASIC (~0.3 mm² @ 28nm)

---

## Notes

1. **Design-first TDD**: Documentation (Phase 1) → Tests (Phase 2) → Implementation (Phase 3)
2. **Milestone commits**: Run tests at each milestone, accept partial passage
3. **FSDR priority**: Implement FSDR first as it has highest impact on memory bandwidth
4. **Reuse existing code**: Leverage saes.types.Gaussian for compatibility
5. **Hardware-software co-design**: All Python implementations should map to hardware

---

**Recommended Approach**: Use milestone commits with 6 checkpoints  
**Estimated Total**: 8,317 LOC (Very Large Feature)  
**Development Duration**: Multiple sessions with incremental progress tracking
