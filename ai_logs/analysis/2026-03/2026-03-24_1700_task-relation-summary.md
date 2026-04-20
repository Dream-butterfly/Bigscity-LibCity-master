### 分析 6

时间：2026-03-24 17:00:00 +08:00

分析类型：调研 / 理解 / 对比 / 验证
结论状态：已验证

输入内容：用户基于 LibCity 源码，分析 traffic_state_pred、traffic_speed_pred、traffic_flow_pred 三者关系，并要求进行工程化校验与结构化总结。

分析结果：
- 通过对 task_config.json、registry.py 以及模型目录结构（traffic_speed_prediction、traffic_flow_prediction）和数据处理逻辑（traffic_state_dataset.py）的联合分析，可以确认 LibCity 采用“任务统一抽象 + 数据语义驱动”的设计。
- 框架仅注册 traffic_state_pred 作为唯一任务入口，该任务负责训练流程、数据加载和评估逻辑，不区分具体预测对象。
- traffic_speed_pred 与 traffic_flow_pred 并非 task，而是模型实现层的语义分类，分别对应不同交通变量（速度、流量）的建模假设，其差异体现在模型结构设计与适用数据分布上。
- 实际任务语义由以下三者共同决定：数据集字段（如 speed / flow）、配置参数（如 data_col、output_dim）、模型归纳偏置（如 DCRNN 偏向速度建模、STResNet 偏向流量建模）。
- 因此，LibCity 中“预测速度”或“预测流量”并不是通过切换 task 实现，而是通过选择不同模型、数据集及配置参数组合实现。

关键结论：LibCity 中仅存在一个顶层任务 traffic_state_pred；所谓 traffic_speed_pred 和 traffic_flow_pred 实际上是该任务下的模型类别划分，而非独立任务，其差异由数据语义与模型归纳偏置共同决定。

启发，或是下一步可以完成的工作：
- [ ] 梳理当前使用数据集（如 PEMS04 / PEMS08）的字段定义，确认是 speed 还是 flow
- [ ] 建立“数据类型 → 推荐模型”的映射表（用于实验选型）
- [ ] 对比同一数据上 speed 模型 vs flow 模型性能差异（验证归纳偏置影响）
- [ ] 分析 data_col 与 output_dim 对多变量预测的影响机制

补充说明：
- 适用范围：适用于 LibCity 全部 traffic_state_pred 任务体系
- 风险点：部分数据集（如 PeMS）在不同预处理版本中可能既包含 speed 也包含 flow，需明确字段定义后再选模型

