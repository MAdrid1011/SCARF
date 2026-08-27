# 模拟器总体架构

这一章描述 Python 模拟器的模块关系和数据通路。模拟器采用功能前端、任务编译器和周期后端三层组织。功能前端运行官方模型并保留真实中间张量。任务编译器将真实算子、RMCF 区域、FSDR 查询和 SAES tile 路径转换为硬件任务描述符。周期后端按照依赖关系和资源可用时间推进整数周期。

<!-- 图：功能前端、任务编译器和周期后端的数据通路 -->
<!-- ![模拟器总体架构](assets/simulator-overview.svg) -->

## 功能前端

功能前端负责产生可渲染结果和真实工作量，不负责估计硬件周期。每个模型适配器从官方仓库加载官方权重和评测配置，以官方数据加载器读取多视图图像、相机参数和目标视图。适配器必须提供 S1 检查位置、S2 候选代价、预测深度、S3 Adaptor 输入输出和 S4 Gaussian 参数。

功能前端在 GPU 上维护中间张量。RMCF 探针统计、FSDR 哈希与候选筛选、SAES 探针统计、逐视图质量指标和大规模追踪归约均使用批量 GPU 运算。热路径中禁止对 tile、像素、候选或 Gaussian 执行 `.item()`、`.cpu()`、隐式 Python 标量转换或显式设备同步。

只在以下边界允许设备到主机的数据传输。

- 一个场景完成后异步写出汇总计数
- 一个追踪分块填满后写入固定页锁定缓冲区
- 一次评测运行完成后写出质量表和清单
- 调试运行显式启用小规模逐任务检查时

## 模型适配器合同

每个适配器实现相同的逻辑接口。

```text
load_official_checkpoint(manifest)
load_official_split(manifest)
run_reference(batch) -> ReferenceOutputs
run_functional(batch, mechanism_mask) -> FunctionalOutputs
emit_operator_descriptors(outputs) -> DeviceDescriptorBatch
render_gaussians(outputs.gaussians, target_cameras) -> Tensor
```

`run_reference` 禁用 RMCF、FSDR 和 SAES，并保持官方模型数学语义。`run_functional` 根据三位机制掩码运行对应组合。适配器不得通过跳过官方基线算子来加快参考执行，也不得为缺失的模型阶段制造代理张量。

## 任务描述符

任务编译器接收 GPU 上的真实张量形状和机制决策，并生成列式描述符。描述符按资源类型分别存储，避免 Python 对象逐条遍历。

每条描述符至少包含以下字段。

- `task_id`、`scene_id`、`view_id` 和 `stage`
- 算子类型、数据类型和精度路径
- 真实的 $M$、$N$、$K$、卷积核、步幅、填充和注意力窗口
- 区域矩形或 tile 坐标、有效像素掩码和边界宽度
- 深度模式、候选索引集合和通道组数量
- 输入、权重、输出和临时数据的字节范围
- 前驱任务集合、输出消费者和允许重叠的资源集合
- 机制来源标签和回退原因

区域任务保留矩形位置、当前层组和边界宽度。tile 任务保留空间位置、深度模式、有效像素状态、SAES 路径和 Gaussian 数量。BilinearUnit 请求还保留参考或 tile 请求身份、深度候选和通道组编号。

## 任务编译

### S1 区域任务

编译器按论文定义将特征图划分为 $16\times16$ 决策块，并在模型允许的层间边界发射九个探针。收敛分数由真实匹配概率和前后预测深度计算。活动块按行形成区间，再将相邻行中起止位置相同的区间合并为矩形。每个矩形的感受野边界由真实算子图反向传播得到，不使用固定 halo。

具有相同权重的矩形任务可以连续进入 Pack Buffer。编译器只描述可合并关系，不预先假设阵列利用率。尾部批次、边界读和注意力上下文均由周期后端产生实际周期。

### S2 探针与 tile 任务

编译器优先排列 SAES 探针和 RMCF 已计算的重合探针。每个深度候选、源视图和 32 维通道组形成 BilinearUnit 请求。FSDR 命中时，候选索引是完整集合中围绕深度锚点的真实子集。未命中或回退时使用完整集合。禁止只记录候选数量而丢失候选身份，因为候选身份影响地址和 DRAM 行局部性。

### S3 路径任务

SAES 路径选择器完全按照论文确定 L0、L1 或 Full，并在任务描述符中冻结该标签。L0 和 L1 首先为论文规定的探针执行 Adaptor，然后在已选路径内部使用固定路径基元重建，只提交原路径规定的保留 Gaussian。Full 为 tile 中全部像素生成 Adaptor 任务，不调用重建算法。重建器不得改变标签或新增 Full tile。具体算法见 [SAES 固定路径基元重建](04_saes_primitive_reconstruction.md)。

### S4 转换任务

S4 描述符按实际保留的 Gaussian 数量生成。每个 Gaussian 顺序经过 GGU 的三级流水。紧凑写回只包含协方差上三角和配置阶数对应的球谐系数。写回地址必须来自真实输出布局。

## 周期后端

周期后端是事件驱动的周期精确模拟器。它维护整数周期、资源预约表、FIFO 状态、SPM 占用、DRAM 请求状态和任务依赖计数。事件驱动只跳过没有状态变化的周期，不改变任一周期的可发射条件。

周期后端包含以下对象。

- `PipelineFSM` 读取指令和任务描述符
- `TaskSelector` 在 Region FIFO 与 Tile FIFO 之间选择就绪任务
- `PackLoader` 维护 Pack Buffer A/B 和 Tag FIFO
- `MMCUModel` 执行 Conv、GEMM 和 Attention 的矩阵 tile
- `BilinearModel` 执行请求、采样、匹配和 PSum 累加
- `VectorModel` 执行归约、概率和机制统计
- `FSDRModel` 维护 32 项 CAM 和逐帧重置
- `GGUModel` 维护 32 个三级流水 PE
- `SPMModel` 维护容量、bank、端口和占用
- `DRAMBridge` 将真实地址事务交给开源 DRAM 时序模型
- `MetricsSink` 记录周期、阻塞原因和逐资源利用率

这些对象是论文模块的软件模型。`MetricsSink`、配置解析和文件写出属于支撑代码，不计入硬件周期。

## 存储系统

SPM 只缓存容量允许的数据。权重、特征和 tile 数据按照描述符中的真实字节范围分配。命中由地址与驻留区间决定，不能使用阶段平均命中率。容量不足时，编译器或运行时产生真实回写和重新读取事务。

DRAM 目标是论文定义的 Micron LPDDR4X-4266。实现优先使用开源 Ramulator 的 LPDDR4 模型，并用器件数据手册补齐 LPDDR4X-4266 的组织和时序配置。每一个时序值都记录器件表格位置。若开源模型缺少必要命令，只允许扩展 DRAM 标准模型，不允许使用固定平均延迟替代。

## 重叠和背压

模拟器必须显式表示以下重叠关系。

- Pack Buffer A 送入 MMCU 时，Pack Buffer B 读取并组装下一批激活
- S1 在未收敛矩形上继续计算时，已收敛区域可以进入 S2
- BilinearUnit 产生代价体时，MMCU 可以处理已经就绪的其他代价体
- S2 的剩余像素任务只有在 SAES 路径判定需要时才发射
- GGU 处理当前 Gaussian 时，MVU 可以处理下一 tile
- DRAM 请求与不依赖该数据的计算可以重叠

FIFO 满、SPM 无空间、DRAM 未返回、权重未就绪、PSum 表项冲突和输出消费者阻塞都会停止对应发射。周期统计必须给出每类阻塞的累计周期和受影响资源。

## 追踪与复现

每次运行保存官方仓库提交、权重校验和、数据集样本清单、配置来源图、软件环境、GPU 型号和模拟器提交。追踪文件采用列式分块，允许 GPU 直接统计候选分布、区域形状、Gaussian 数量和地址跨度。Python 对象追踪只允许用于小规模调试。
