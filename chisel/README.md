# SCARF Chisel RTL

SCARF 加速器的 Chisel HDL 实现，严格对应 `encoder/`、`ggu/`、`depth_predictor/` 中的 Python 硬件仿真器。

## 目录结构

```
chisel/
├── build.sbt                        # SBT 构建配置
├── project/plugins.sbt              # Chisel 插件
├── src/
│   ├── main/scala/scarf/
│   │   ├── Config.scala             # 全局配置 & ModelConfig Bundle
│   │   ├── Types.scala              # 共享类型 (DataPacket, ComputeCmd 等)
│   │   ├── ScarfTop.scala           # 顶层模块
│   │   ├── VerilogEmitter.scala     # 分文件 Verilog 生成器
│   │   ├── compute/                 # 共享计算单元 (对应 encoder/)
│   │   │   ├── ConvEngine.scala     # ← encoder/conv_engine.py
│   │   │   ├── GEMMUnit.scala       # ← encoder/gemm_unit.py
│   │   │   ├── BilinearUnit.scala   # ← encoder/bilinear_unit.py
│   │   │   ├── ActivationUnit.scala # ← encoder/activation_unit.py
│   │   │   ├── NormUnit.scala       # ← encoder/normalization_unit.py
│   │   │   ├── VectorALU.scala      # ← encoder/softmax_unit.py (部分)
│   │   │   ├── SoftmaxUnit.scala    # ← encoder/softmax_unit.py
│   │   │   ├── PoolingUnit.scala    # ← encoder/pooling_unit.py
│   │   │   └── PadUnit.scala        # ← encoder/pad_unit.py
│   │   ├── ggu/                     # 高斯生成单元 (对应 ggu/)
│   │   │   ├── GGUArray.scala       # ← ggu/ggu_processor.py
│   │   │   ├── PositionCalc.scala   # ← ggu/position_calculator.py
│   │   │   ├── CovBuilder.scala     # ← ggu/covariance_builder.py
│   │   │   └── SHRotator.scala      # ← ggu/sh_rotator.py
│   │   ├── memory/                  # 存储子系统
│   │   │   ├── WeightBuffer.scala   # 128 KB 权重 SRAM
│   │   │   ├── FeatureBuffer.scala  # 256 KB 特征双端口 SRAM
│   │   │   ├── TileSPM.scala        # 64 KB scratchpad
│   │   │   └── DRAMInterface.scala  # AXI4 接口
│   │   └── control/                 # 控制器 (对应 depth_predictor/hw_depth_predictor.py)
│   │       ├── PipelineController.scala  # 顶层 FSM
│   │       ├── S1Controller.scala        # S1 阶段控制
│   │       ├── S2S3Controller.scala      # 融合 S2+S3 tile 控制
│   │       ├── SAESController.scala      # SAES 早退 FSM
│   │       └── ConfigRegs.scala          # MMIO 配置寄存器
│   └── test/scala/scarf/           # 测试
│       ├── TestUtils.scala          # 公共测试工具
│       ├── ConvEngineTest.scala     # PE, SystolicArray, ConvEngine FSM
│       ├── GEMMUnitTest.scala       # OSPE, OutputStationaryArray, GEMMUnit FSM
│       └── VerilogEmitTest.scala    # 全模块 elaboration 烟雾测试 (26 modules)
├── generated/                       # 生成的 Verilog 文件
└── README.md                        # 本文件
```

## 构建与运行

### 前置依赖

- JDK 11+
- SBT 1.9+
- Verilator (可选，用于 Verilog 编译验证)

### 常用命令

```bash
cd chisel/

# 编译
sbt compile

# 运行全部测试
sbt test

# 生成分文件 Verilog
sbt "runMain scarf.VerilogEmitter"
# 输出到 generated/ 目录，每个模块一个 .v 文件

# 验证 Verilog 可编译 (需要 Verilator)
cd generated/ && verilator --lint-only ScarfTop.v
```

## 设计原则

1. **模型无关**：Transplat / MVSplat / DepthSplat 通过 `ConfigRegs` 参数化，数据路径无模型分支
2. **单实例复用**：每种计算单元仅一份实例，FSM 时分复用
3. **Python 严格对应**：每个 `.scala` 文件对应一个 Python 仿真器，接口行为一致
4. **分文件 Verilog**：`VerilogEmitter` 逐模块生成，便于综合工具分析

## 架构文档

详见 [docs/architecture-cn.md](../docs/architecture-cn.md) — 顶会级中文架构文档。
