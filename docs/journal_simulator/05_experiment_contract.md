# 真实工作负载与实验合同

这一章规定模型、数据集、指标、消融和性能测量。任何结果只有在工作负载可追溯、质量通过且周期由任务级模拟产生时才有效。

## 工作负载目标矩阵

目标矩阵包含 MVSplat、TranSplat 和 DepthSplat，以及 RealEstate10K、ACID 和 DL3DV。实施顺序不要求一次填满九个组合。每个单元格只有在官方权重、官方或论文一致的评测配置、真实数据和目标视图均可获得时才启用。

| 模型 | 官方来源 | 首选权重与协议 | 当前设计状态 |
| --- | --- | --- | --- |
| MVSplat | [官方仓库](https://github.com/donydchen/mvsplat) | 官方预训练模型和官方 evaluation index | 首个实施模型 |
| TranSplat | [官方仓库](https://github.com/xingyoujun/transplat) | 官方 Re10K 和 ACID 权重 | 权重支持的组合启用 |
| DepthSplat | [官方仓库](https://github.com/cvg/depthsplat) 和 [Model Zoo](https://github.com/cvg/depthsplat/blob/main/MODEL_ZOO.md) | 官方 Re10K 与 DL3DV 权重 | 在 MVSplat 闭环后启用 |

| 数据集 | 官方来源 | 可用性规则 |
| --- | --- | --- |
| RealEstate10K | [官方数据页](https://google.github.io/realestate10k/download.html) | 使用官方 split。失效视频记录为缺失样本，不替换为其他视频 |
| ACID | [Infinite Nature 官方项目页](https://infinite-nature.github.io/) | 使用官方 ACID 下载和模型仓库定义的 evaluation index |
| DL3DV | [官方项目页](https://dl3dv-10k.github.io/DL3DV-10K/) | 使用模型官方定义的 DL3DV Benchmark split 和分辨率 |

下载器必须记录官方 URL、发布版本、文件名、字节数、许可信息和样本清单，并通过格式解析、官方加载器和最小真实样本读取确认资源可用。下载、缓存和运行阶段不使用 SHA、MD5、内容散列或 digest 校验。需要账户接受公开许可的数据可以使用已接受许可的本机凭据。需要付费、商业授权或无法公开取得的数据直接跳过，不等待机主提供商业资源。

如果某模型没有对应数据集的官方权重，只在模型论文或官方仓库明确规定跨数据集协议时使用其他官方权重。否则该组合标记为 `SKIPPED_NO_OFFICIAL_CHECKPOINT`。

## 首个闭环组合

首个组合固定为 MVSplat 与 RealEstate10K。选择原因是官方代码、权重、评测索引和较轻的代价体路径均公开，适合先验证完整功能输出、任务编译和周期后端。

首个闭环按以下顺序完成。

1. 使用官方配置复现一个真实场景的参考渲染和官方质量指标
2. 在 GPU 上导出 S1 至 S4 的真实形状、候选、地址和 Gaussian 追踪
3. 完成 NoOpt 周期
4. 分别完成 RMCF、FSDR 和 SAES 的功能输出与单机制周期
5. 完成八种组合并验证 All 一致性
6. 扩展到完整 Re10K 评测分割
7. 再扩展其他模型与数据集

在一个组合完成前，不并行启动九个组合的小时级运行。

## 质量指标

渲染使用各官方模型兼容的官方 Gaussian rasterizer。参考输出和每个机制组合使用同一目标相机、图像尺寸、颜色空间、背景处理和裁剪规则。

PSNR、SSIM 和 LPIPS 在 GPU 上批量计算。LPIPS 使用[官方实现](https://github.com/richzhang/PerceptualSimilarity)，网络版本和输入归一化写入清单。输出保留逐视图结果、逐场景结果、均值、中位数、分位数和最坏视图。均值只用于汇总，不能决定质量通过。

期刊论文给出的完整 SCARF 最大变化形成保守质量边界。

- PSNR 最大相对变化 0.072%
- SSIM 最大相对变化 0.24%
- LPIPS 最大绝对变化 0.002

每个单机制和组合都必须满足该完整边界。若后续论文明确给出更严格的单机制边界，使用更严格值。质量检查使用官方验证分割校准参数，测试分割只做一次最终报告。

## 吞吐指标

只报告以下性能指标。

- 单次端到端周期
- 稳态完成间隔周期
- 1 GHz 下的 inference/s
- 相对 NoOpt 的吞吐加速比
- 相对目标 baseline GPU 直接实测或可靠换算吞吐的系统加速比
- S1 至 S4 的周期归属和资源阻塞周期

不报告能效、功耗、面积、面积效率或工艺缩放结果。

目标 baseline GPU 可用时，必须在同一输入、同一官方权重和同一质量配置上直接实测。GPU 时间使用 CUDA Event 和 Nsight Systems 或 PyTorch Profiler 的 CUDA activity。主循环不得用 Python wall clock 包围异步 kernel 后直接读取结果。

### 目标 baseline GPU 不可用时的换算

实现平台没有论文目标 baseline GPU 时，不等待机主提供设备。实施者先在当前 GPU 上实测完整工作负载，再使用可复现的配对实测关系换算到目标 GPU。换算结果用于评审者在缺少目标设备时复现基线，但必须与直接实测明确区分。

换算以阶段或算子形状类别 $k$ 为单位。对当前 GPU 上实测的时间 $t_{\mathrm{local},k}$，使用同一软件路径、精度、batch、形状和功耗模式下的配对校准测量得到

\[
r_k=\frac{t_{\mathrm{target},k}^{\mathrm{cal}}}
{t_{\mathrm{local},k}^{\mathrm{cal}}},\qquad
\hat t_{\mathrm{target}}=\sum_k t_{\mathrm{local},k}r_k.
\]

配对校准可来自实施者可重跑的开源微基准、官方性能追踪或同时覆盖当前 GPU 和目标 GPU 的公开实测产物。每个 $r_k$ 必须保存两端硬件模式、软件版本、输入形状、实测时间和来源。计算密集、带宽受限、不规则采样和 rasterization 不共用一个缩放比。

禁止使用峰值 FLOPS、显存带宽、CUDA 核心数或一个全局平均倍率直接换算端到端基线。当真实算子形状超出配对校准覆盖范围，或目标平台的 CPU、I/O 与 GPU 统一内存开销无法建立实测关系时，对应基线标记为 `SKIPPED_NO_RELIABLE_BASELINE_CONVERSION`，不做外推。

`baseline_normalization.json` 保存当前 GPU 原始测量、分组时间、每组换算关系、适用范围、校准残差和最终求和。结果字段 `baseline_kind` 必须为 `MEASURED_ON_TARGET`、`NORMALIZED_FROM_MEASURED` 或跳过状态之一。

## 八种消融

每个可用工作负载运行 NoOpt、RMCF、FSDR、SAES、RMCF+FSDR、RMCF+SAES、FSDR+SAES 和 All。每行输出真实周期和真实质量。禁止从单机制结果乘法推导组合结果。含 SAES 的运行必须复用同一个论文路径选择器；替换路径内部算法前后的 L0、L1、Full 标签摘要必须一致。

消融表至少包含以下字段。

```text
model,dataset,scene,mechanism_mask,total_cycles,
s1_cycles,s2_cycles,s3_cycles,s4_cycles,
throughput_inf_s,speedup_vs_noopt,
psnr,ssim,lpips,quality_pass,
saes_l0_tiles,saes_l1_tiles,saes_full_tiles,
saes_path_mismatch_count,saes_path_first_mismatch,saes_path_labels_file
```

运行直接逐元素比较替换前后的完整路径标签张量，并把原始标签以列式文件落盘；不生成哈希或 digest。只有 `saes_path_mismatch_count` 为零时才通过路径一致性门。

NoOpt 是相同硬件上的无优化架构。Orin NX 是系统级外部基线。二者不能混用。

## 单机制端到端加速上限

首个工作负载跑通后，先测量每项策略的端到端算法上限，再调试周期模拟器。

对于机制 $m$，使用完整参考运行产生 oracle 决策和真实剩余工作量，在同一 GPU 后端上执行仅包含该机制可合法删除工作后的路径。该路径保留所有不属于 $m$ 的模型算子。上限运行输出质量并满足同一质量合同。

若实测上限低于静态锚点，先检查原生软件中所有机制无法覆盖的阶段。允许的纯工程优化包括 GPU 常驻张量、算子融合、CUDA Graph、异步预取、页锁定缓冲区、批量索引和消除同步。这些优化必须保持逐元素功能输出或在官方数值容差内一致。不得把额外算法减算计入工程优化。

工程优化完成后重新测量参考路径和机制上限。模拟器加速比仍低于合法上限时，再检查任务打包、资源冲突、存储驻留和流水重叠。不能通过修改资源参数追平上限。

## GPU 利用率与长任务审计

所有 GPU 运行启动 `gpustat --json` 旁路记录。监测进程只记录时间、利用率、显存、进程和温度，不在每个推理迭代中查询 GPU。

低利用率不使用固定百分比判断。每个模型与分辨率先执行一个短校准运行，记录在官方 batch 和最大可用显存 batch 下的利用率分布与完成间隔。长任务的利用率分布低于同一配置校准包络时触发审计。包络计算方法、置信水平和参考运行写入配置来源，不能写死在监测脚本中。

预计超过一小时的任务必须先执行一个有代表性的短片段。短片段检查以下内容。

- GPU 利用率是否落入校准包络
- 热循环中是否出现 `.cpu()`、`.item()` 或设备同步
- 数据加载是否使 GPU 等待
- 显存是否稳定
- 输出周期和质量是否持续写入

短片段未通过时不得启动小时级任务。

## GPU 上的追踪分析

候选直方图、区域连通区统计、tile 路径计数、Gaussian 属性差异、地址分桶和质量指标使用 Torch、CuPy 或 RAPIDS 在 GPU 上批量执行。CPU 只负责事件队列、文件清单和最终序列化。

分支密集且具有严格因果顺序的周期事件调度可以保留在 CPU。不得为了形式上使用 GPU 而将每个事件在 CPU 与 GPU 之间往返。只有能够形成足够大列式批次的分析才移至 GPU。

## 最小验证集合

本项目不建立重复的证书、smoke 和审计体系。每个阶段只保留能够阻止错误结果的最小验证。

- 一个官方模型参考场景的功能一致性
- 每个硬件资源的一组发射、背压和尾部任务合同测试
- 一个真实场景的任务计数和地址事务复放
- 一个八组合端到端闭环
- 最终真实分割上的质量和吞吐评测

没有证据表明会影响周期或质量的检查不进入阻塞门。
