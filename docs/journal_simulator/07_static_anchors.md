# 静态锚点与对照数据

这一章定义实施前锚点。锚点用于检查真实模型追踪和周期模拟是否出现不可解释的偏差。它们不是模拟器输入，也不是缺失实验的替代结果。

## 数据分类

本目录将数据分为两类。

- `static_estimate` 来自期刊实施前静态分析。它基于模型算子结构、数据集特征和显式先验，并给出敏感性范围。
- `paper_reported` 来自期刊草稿中的结果表和图数据。它用于实现后对照，但必须由新模拟器和真实工作负载重新验证。

任何结果文件都必须保留 `source_kind`。报告中不得把 `static_estimate` 写成测量，也不得把 `paper_reported` 写成新模拟器结果。

## RMCF 静态锚点

[rmcf_static_anchor.csv](anchors/rmcf_static_anchor.csv) 原样保存期刊已有的九组静态锚点，包括早期收敛区域、S1 冗余 MAC、S2 冗余 MAC 和原始 Orin NX 编码器的端到端可避免时间。

这些区间是固定种子敏感性采样的第 10 至第 90 百分位，不是测量置信区间。实现前判断规则如下。

- 真实早期收敛、S1 或 S2 减算低于下界时，检查探针、匹配位置、区域合并、边界保留和回退
- 真实值高于上界时，检查是否错误删除完整路径、遗漏边界任务或泄漏测试标签
- 端到端上限低于静态下界时，先审计机制无法覆盖的软件阶段

论文侧 `challenge1_measurements.csv` 中的端到端列与静态锚点不是同一口径，因此没有复制为静态锚点。新实现必须分别输出原始 Orin NX 时间、工程优化后 Orin NX 时间和硬件 NoOpt 时间，避免再次混合口径。

## RMCF 工作量和利用率对照

- [paper_rmcf_work_reduction_reference.csv](anchors/paper_rmcf_work_reduction_reference.csv)
- [paper_mmcu_utilization_reference.csv](anchors/paper_mmcu_utilization_reference.csv)

这些表保存期刊报告的工作量削减、Full 回退率和 MMCU 活跃利用率。模拟器不接受利用率作为输入。利用率必须由逐周期有效行、有效 PE 和资源占用重新统计。

## FSDR 和 SAES 对照

[fsdr_saes_paper_reference.csv](anchors/fsdr_saes_paper_reference.csv) 保存九组 FSDR 引导率、Top-1 覆盖率、S2 候选节省、特征流量削减，以及 SAES L0、L1、低方差一致率、Gaussian 节省和 S2 评估节省。

FSDR 实现首先核对候选身份和 Top-1 覆盖，再核对引导率。SAES 实现先以论文选择器复现 L0、L1、Full 标签和原始比例，再冻结该标签测试路径内部重建质量。替换算法前后路径比例、Gaussian 数量和任务取消集合必须相同；达到节省比例但质量不通过仍不算成功。

## 消融与平台对照

- [paper_ablation_speedup_reference.csv](anchors/paper_ablation_speedup_reference.csv)
- [paper_platform_speedup_reference.csv](anchors/paper_platform_speedup_reference.csv)
- [paper_stage_breakdown_reference.csv](anchors/paper_stage_breakdown_reference.csv)

这些文件保存论文报告的八组合加速比、Orin NX 数据流与 ASIC 加速比，以及 S1 至 S4 阶段分解。它们只用于实现后异常检测。周期后端不得读取这些文件。

若新模拟器与论文对照不一致，按资源级证据判断。任务计数、字节数、资源周期和真实质量均可解释时，保留新结果并更新论文。不得修改微架构延迟来强制匹配对照柱高。

## 质量对照

[paper_quality_reference.csv](anchors/paper_quality_reference.csv) 保存基线与完整 SCARF 的 PSNR、SSIM 和 LPIPS。新实验使用官方评测分割逐视图重新计算。表中数值不用于阈值搜索的测试分割决策。

## 每组实施前报告

每个模型与数据集组合在实现前按 [anchor report 模板](anchors/anchor_report_template.md) 创建 `anchors/reports/<model>__<dataset>.md`。任何可用组合在该页完成前不得进入小时级运行。模型、官方权重或真实数据不可获得时，该页只记录来源查找结果和跳过状态，不生成代理锚点。

锚点必须由官方模型源码、真实张量形状、数据集的视图数、分辨率、场景结构和已有期刊预估静态推导。已有预估数据直接复制并保留 `source_kind`、原文件和原始口径，不用平均值重建、平滑或修改尾数。新推导中每个延迟、带宽、命中、候选数和质量范围都必须指向来源或公式。

每页至少包含以下内容。

```text
模型和官方提交
权重和训练数据来源
评测 split、分辨率和视图数
S1/S2/S3/S4 静态算子与字节数
NoOpt 的逐资源、逐阶段周期区间及关键路径推导
RMCF 静态区间
FSDR 论文对照
SAES 论文对照
NoOpt、RMCF、FSDR、SAES、RMCF+FSDR、RMCF+SAES、FSDR+SAES 和 All 的预期任务削减、周期区间和论文对照
质量对照
当前 GPU 实测 baseline 和目标 GPU 直测或可靠换算方法
不可获得项和跳过原因
```

锚点还必须记录每项预估的下界、中心值、上界和敏感参数。中心值不得直接输入周期模拟器；它只用来发现实现过差或无法解释地过好。实施后在同一页追加真实值、差异和解释，不覆盖实施前锚点。

## 九组工作负载覆盖门

MVSplat、TranSplat 和 DepthSplat 与 RealEstate10K、ACID 和 DL3DV 形成九个目标单元。实施顺序为先闭环 MVSplat 与 RealEstate10K，再一次扩展一个单元。每个单元的状态只能是 `ANCHOR_REQUIRED`、`READY_TO_RUN`、`MEASURED`、`SKIPPED_NO_OFFICIAL_CHECKPOINT`、`SKIPPED_UNAVAILABLE_DATASET` 或 `SKIPPED_NO_RELIABLE_BASELINE_CONVERSION`。

只有真实模型、官方权重、真实数据集和完整锚点页均可用时，状态才能从 `ANCHOR_REQUIRED` 转为 `READY_TO_RUN`。不可获得单元保留跳过记录，不使用随机张量、代理模型、替代数据集或伪造质量值补齐。
