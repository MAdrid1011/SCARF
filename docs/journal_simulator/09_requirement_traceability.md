# 实施约束追踪

本页把期刊版模拟器的实施要求映射到设计页、执行门和必须产物。它不增加新的硬件模块或实验门，只用于确认实施者没有遗漏已有合同。

## 要求映射

| 要求 | 约束页 | 执行门 | 必须产物 |
| --- | --- | --- | --- |
| 无 magic number | [设计来源与约束](01_source_of_truth.md) | 任一生效数值缺少来源或推导时拒绝运行 | 配置来源图与推导式 |
| 禁止完整性哈希 | [设计来源与约束](01_source_of_truth.md)、[实验合同](05_experiment_contract.md) | 流程中不生成或比较 SHA、MD5、内容散列或 digest | 官方来源、版本、文件名、字节数、可加载状态和样本清单 |
| 保留 FSDR LSH | [设计来源与约束](01_source_of_truth.md)、[周期模型](03_cycle_model.md) | LSH 签名、CAM 查询和汉明距离必须按设计存在 | FSDR 查询事件和逐周期资源统计 |
| 避免 GPU--CPU 频繁同步 | [模拟器总体架构](02_simulator_architecture.md) | GPU 热路径不得出现逐 tile、像素、候选或 Gaussian 的主机同步 | profiler 摘要与追踪分块记录 |
| `gpustat` 与长任务预审计 | [实验合同](05_experiment_contract.md)、[实施者工作流](06_implementer_workflow.md) | 预计超过一小时的任务先运行代表性短片段 | `gpu_utilization.jsonl` 和 profiler 摘要 |
| 避免重复证书、smoke 和审计 | [实验合同](05_experiment_contract.md) | 只保留会阻止周期或质量错误的最小验证集 | 单资源合同、单场景复放、八组合闭环和最终评测 |
| 易并行工作使用 GPU 或多核 | [模拟器总体架构](02_simulator_architecture.md)、[实验合同](05_experiment_contract.md) | 大批次 trace 分析不进入单核 Python 循环 | GPU 批处理追踪或多进程运行记录 |
| 只使用真实模型和数据 | [实验合同](05_experiment_contract.md) | 官方权重或数据不可获得时跳过 | `availability.json` 和跳过原因 |
| 模拟器不用平均方法 | [周期模型](03_cycle_model.md) | 周期由任务依赖、资源占用、存储事务和背压生成 | `cycles.json`、逐阶段周期和阻塞统计 |
| 设计模块不增删 | [设计来源与约束](01_source_of_truth.md)、[模块一致性表](architecture_conformance.md)、[模拟器总体架构](02_simulator_architecture.md) | 核心模块逐项映射；支撑模块不改变任务图或加速比 | `architecture_conformance.md` |
| 质量优先 | [SAES 固定路径基元重建](04_saes_primitive_reconstruction.md)、[实验合同](05_experiment_contract.md) | 每个视图计算 PSNR、SSIM 和 LPIPS；不通过者不进入性能汇总 | `quality_per_view.csv` 和最坏视图记录 |
| 商业工具使用开源替代 | [设计来源与约束](01_source_of_truth.md)、[实施者工作流](06_implementer_workflow.md) | 不等待机主提供商业资源 | 开源工具版本、配置和结果 |
| 缺少目标 baseline GPU 时可靠换算 | [实验合同](05_experiment_contract.md) | 仅允许逐阶段或逐算子形状的配对实测关系 | `baseline_measurement.json` 和 `baseline_normalization.json` |
| 每个组合先建静态锚点 | [静态锚点与对照数据](07_static_anchors.md) | 锚点页完成后才允许小时级运行 | `anchors/reports/<model>__<dataset>.md` |
| 输出端到端周期和质量 | [实验合同](05_experiment_contract.md) | 周期、PSNR、SSIM 和 LPIPS 齐全 | 运行目录中的周期与质量文件 |
| 输出全部消融组合 | [周期模型](03_cycle_model.md)、[实验合同](05_experiment_contract.md) | $2^3$ 组合全部使用真实任务图 | `ablation_all_combinations.csv` |
| All 与主运行一致 | [实施者工作流](06_implementer_workflow.md) | 周期和质量必须完全相同 | `cycles(main_full_run) == cycles(mask_111)` |
| 重大进展使用 Git | [实施者工作流](06_implementer_workflow.md) | 每个闭环节点同步更新设计文档和实验日志 | 节点 Git 提交 |

## 强制实施顺序

1. 在 GPU 上跑通一个真实的模型与数据集组合。
2. 从设计文档、官方模型源码和真实追踪建立静态锚点与模拟器设计。
3. 在真实 GPU 后端上分别测量 RMCF、FSDR 和 SAES 的端到端加速上限。
4. 上限低于静态锚点时，只优化三项机制均无法覆盖的软件阶段，并保持逐元素功能语义或官方数值容差内一致。
5. 先完成 NoOpt，再实现单机制和全部组合。模拟加速比低于合法上限时，检查任务打包、资源冲突、存储驻留和流水重叠。
6. 当前模型与数据集组合的周期、质量、八组合和实验日志全部闭环后，才扩展下一个组合。

## 冲突处理

若实施代码、旧实验脚本或历史结果与本目录冲突，按[设计来源与约束](01_source_of_truth.md)的证据优先级处理。历史模拟器行为和论文结果数字不能覆盖当前设计、真实模型追踪或逐资源证据。冲突解决后立即更新相应设计页、锚点页和实验日志，并在重大进展节点创建 Git 提交。
