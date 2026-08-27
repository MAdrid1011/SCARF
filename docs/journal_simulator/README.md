# SCARF 期刊版 Python 模拟器设计

本目录定义一套从期刊论文重新设计的 SCARF Python 模拟器。设计不继承旧仓库的模拟器实现，也不以旧实现的行为作为正确性依据。论文中的阶段定义、模块边界、真实模型输出和真实数据集结果共同构成设计来源。

模拟器只评价吞吐。主要输出为端到端周期数、按阶段周期数、1 GHz 下的推理吞吐率、相对无优化架构的加速比，以及渲染质量 PSNR、SSIM 和 LPIPS。面积、功耗、能效和面积归一化指标不在本项目范围内。

## 文档结构

- [设计来源与约束](01_source_of_truth.md)
- [模拟器总体架构](02_simulator_architecture.md)
- [周期模型与资源合同](03_cycle_model.md)
- [SAES 基元一致性重建](04_saes_primitive_reconstruction.md)
- [真实工作负载与实验合同](05_experiment_contract.md)
- [实施者工作流](06_implementer_workflow.md)
- [静态锚点与对照数据](07_static_anchors.md)
- [高斯简化相关工作审计](08_literature_survey.md)

## 不可违反的设计边界

1. 核心架构只包含论文定义的 MVU、FSDR 缓存、GGU 阵列、控制子系统和 SPM 子系统。
2. 核心优化只包含 RMCF、FSDR 和 SAES。实现不得新增第四项核心优化，也不得删除或重定义已有机制。
3. 支撑代码只负责真实模型适配、任务描述生成、事件调度、结果记录和指标计算。支撑代码不得隐藏模型计算，不得贡献论文声称的加速收益。
4. 周期模拟必须由逐任务依赖、逐资源占用和逐事务存储访问产生。禁止用平均阶段时间、平均命中率或预设加速比直接生成周期结果。
5. 功能模拟必须使用官方预训练模型和真实数据。模型或数据不可获得时，该组合标记为 `SKIPPED_UNAVAILABLE`，不得使用随机张量、代理网络或合成指标补齐。
6. 质量优先于吞吐。任何未通过质量合同的优化组合不得进入性能汇总，也不得通过放宽指标或只报告均值来掩盖最坏视图退化。
7. 所有数值参数都必须携带来源和推导。裸数值配置被视为错误，除非它来自张量形状、数据类型定义或数学恒等式。

## 输出合同

每个模型与数据集组合必须生成以下文件。

```text
runs/<run_id>/
  manifest.json
  availability.json
  cycles.json
  cycles_by_stage.csv
  throughput.json
  quality_per_view.csv
  quality_summary.json
  ablation_all_combinations.csv
  gpu_utilization.jsonl
  trace_manifest.json
```

`cycles.json` 中的全优化周期必须与 `ablation_all_combinations.csv` 的 `RMCF+FSDR+SAES` 行完全相同。任何差异都表示配置或任务图不一致，结果不得发布。

