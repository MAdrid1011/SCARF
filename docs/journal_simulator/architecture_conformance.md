# 模块一致性表

本页将期刊设计中的五个核心组件映射到 Python 周期模拟器。模拟器对象可以拆分软件职责，但不得改变硬件模块的数据通路、资源共享、状态、背压或机制语义。

## 核心模块映射

| 期刊组件 | 必须保留的内部结构 | 模拟器对象 | 主要输入与输出 | 不变量 |
| --- | --- | --- | --- | --- |
| MVU | $48\times48$ MMCU、BilinearUnit、VectorALU、NormUnit、ActivationUnit、双粒度矩阵加载 | `MMCUModel`、`BilinearModel`、`VectorModel`、`PackLoader` | 区域与 tile 描述符、特征、权重、候选索引 → 矩阵输出、匹配分数和向量统计 | Conv、GEMM 和 Attention 复用同一 MMCU；不得拆成三个可并行阵列 |
| FSDR 缓存 | 32 个 41 位 CAM 条目、16 位 LSH、FP16 深度锚点、8 位 LRU 和逐帧重置 | `FSDRModel` | 查询签名与深度一致性 → 完整或局部候选索引集 | 候选身份、CAM 替换和回退条件必须保留 |
| GGU 阵列 | 32 个 PE；`PositionCalc`、`CovBuilder`、`SH_OPGenerator` 三级流水 | `GGUModel` | 像素、深度、原始 Gaussian 参数和相机 → $\{\mu,\Sigma,\alpha,\mathbf c_{sh}\}$ | 位置与协方差路径使用 FP32；不得用平均每 Gaussian 开销替代流水 |
| 控制子系统 | `Region FIFO`、`Tile FIFO`、`Request FIFO`、`Reference FIFO`、`Tag FIFO`、`PSum RAM`、依赖和流水状态 | `PipelineFSM`、`TaskSelector`、`PackLoader` 中的控制状态 | 指令、描述符和资源反压 → 发射、保持、重放和提交 | RMCF 和 SAES 使用现有控制与 VectorALU；不新增专用加速阵列 |
| SPM 子系统 | 128 KB 权重缓冲区、256 KB 特征缓冲区、64 KB tile 缓冲区 | `SPMModel` | 带真实地址和字节数的访问 → 命中、等待、驱逐和写回 | 容量、bank、端口、占用和驱逐必须显式建模 |

`DRAMBridge` 是 SPM 子系统面向 LPDDR4X-4266 的环境接口，不是第六个核心加速模块。它只把真实地址事务送入开源 DRAM 时序模型并返回完成周期。

## 机制到现有资源的映射

| 机制 | 使用的已有资源 | 合法行为 | 禁止行为 |
| --- | --- | --- | --- |
| RMCF | BilinearUnit、VectorALU、Region FIFO、Tile FIFO、双粒度加载器 | 根据真实匹配概率与深度一致性停止区域工作，生成矩形任务和轻量深度路径 | 新增 RMCF 专用计算阵列、oracle 路径或无证据跳过 |
| FSDR | LSH/CAM、BilinearUnit、SPM 与 DRAM 路径 | 命中时缩小完整候选集中的真实索引集，未命中时回退 | 使用预设命中率、只保留候选数量或改变候选值 |
| SAES | VectorALU、MMCU、GGU、Tile FIFO 和 tile 缓冲区 | 按论文顺序选择 L0、L1 或 Full，并在固定路径内产生保留 Gaussian | 新增第四路径、修改标签、增加 Full tile 或为通过质量门而回退 |

## 支撑软件边界

| 支撑对象 | 职责 | 是否产生硬件周期 | 限制 |
| --- | --- | --- | --- |
| 官方模型适配器 | 加载真实权重、数据和中间张量 | 否 | 不跳过官方基线算子 |
| 任务编译器 | 把真实形状、决策和地址转为描述符 | 否 | 不改变任务身份、依赖或候选集 |
| 事件引擎 | 按下一状态变化推进时间 | 否 | 跳过无事件周期不得改变发射条件 |
| `MetricsSink` | 记录周期、阻塞、利用率和质量 | 否 | 异步移出热路径，不反馈资源决策 |
| 配置与序列化 | 校验参数来源并写出结果 | 否 | 不生成预设周期或加速比 |

如果支撑对象修改了硬件任务图、资源可用时间或关键路径，它就不再是支撑代码。该改动必须删除，除非能在不增加核心技术的前提下证明其是闭环必需结构，并在周期模型中显式记账。

## 参数一致性

| 参数 | 值 | 来源 | 模拟要求 |
| --- | ---: | --- | --- |
| MMCU 行列 | $48\times48$ | 期刊架构 | 边界 tile 使用真实 $m,n,k$ |
| BilinearUnit 通道组 | 32 | 期刊架构 | 保留候选、源视图和组身份 |
| VectorALU 物理 lane | 64 | [Pipeline Architecture](../pipeline-architecture.md#2-hardware-configuration) | 不因 32 维 Bilinear 分组假设双发射 |
| FSDR CAM | 32 个 41 位条目 | 期刊架构 | 并行比较、最佳项读取和真实 LRU |
| GGU | 32 PE、3 级 | 期刊架构 | 流水充填、稳态和背压显式建模 |
| SPM 容量 | 128 KB / 256 KB / 64 KB | 期刊架构 | bank、端口和延迟不从容量猜测 |
| 目标时钟 | 1 GHz | 期刊实验设置 | 周期为主结果，吞吐从完成间隔推导 |
| 外存 | Micron LPDDR4X-4266 | 期刊架构 | 使用开源 DRAM 时序模型和公开数据手册 |

表中未出现的 bank 数、端口数、队列深度、操作延迟和发射间隔不得在实现中猜测。实施者只能使用设计文档、开源综合、器件手册或实测微基准补齐，并写入参数来源图。
