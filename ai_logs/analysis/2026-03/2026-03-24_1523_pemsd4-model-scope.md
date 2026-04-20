### 分析 1
时间：2026-03-24 15:23:00 +08:00

分析类型：调研 / 理解 / 验证
结论状态：已验证

输入内容：
用户通过仓库检索 PEMSD4 数据集配置文件与 GNNTP 模型注册信息，询问该数据集可用模型范围及其分类方式。

分析结果：
- 从数据配置文件确认 PEMSD4 包含三个核心字段：
  - traffic_flow
  - traffic_occupancy
  - traffic_speed
- 数据集本质为多变量数据集，可支持不同预测任务。
- 在 GNNTP 中对应两类任务体系：
  - traffic_flow_prediction（约 40 个模型）
  - traffic_speed_prediction（约 27 个模型）
- 典型模型如下：
  - Flow：AGCRN, ASTGCN, MSTGCN, STSGCN, PDFormer, STWave 等
  - Speed：DCRNN, STGCN, GWNET, MTGNN, STAEformer, MegaCRN 等
- 两类任务对应模型注册独立，不能混用。

关键结论：
- PEMSD4 必须先确定任务类型：
  - 流量预测 → 使用 flow 模型集合
  - 速度预测 → 使用 speed 模型集合
- 模型与任务强绑定，是实验设计的前置条件。

启发，或是下一步可以完成的工作：
- [ ] 明确论文任务类型（flow 或 speed）
- [ ] 根据任务筛选候选模型集合
- [ ] 检查模型与数据维度（output_dim）兼容性

补充说明：
- 风险点：错误混用 flow/speed 模型会导致实验失败或结果无效
- 适用范围：基于 GNNTP 的交通预测任务建模
- 限制条件：需遵循数据配置与模型接口约束
