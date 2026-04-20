### 分析 2
时间：2026-03-24 15:23:00 +08:00

分析类型：设计 / 选型 / 对比 / 优化
结论状态：可用

输入内容：
在确认 LibCity 中 PEMSD4 数据集支持的模型范围后，用户希望筛选适合论文对比的模型组合，而不是简单列举全部模型。

分析结果：
- 论文实验设计需要控制模型数量，推荐 5–8 个模型，避免冗余与实验成本过高。
- 模型选择需覆盖技术发展路径：
  - 统计方法 → 经典图神经网络 → 自适应图学习 → Transformer/新模型
- 分任务构建推荐模型集合：

  **Flow 任务推荐：**
  MSTGCN, ASTGCN, AGCRN, STSGCN, PDFormer, STWave

  **Speed 任务推荐：**
  STGCN, DCRNN, GWNET, MTGNN, STAEformer, MegaCRN

- 提炼跨任务统一对比方案（论文主表推荐 6 类模型）：
  1. 传统统计基线：HA（或 VAR）
  2. 经典图卷积模型：STGCN（或 MSTGCN）
  3. 经典递归图模型：DCRNN
  4. 强基线模型：GWNET（或 AGCRN）
  5. 中近年代表模型：MTGNN（或 PDFormer）
  6. 新一代模型：STAEformer（或 STWave / MegaCRN）

关键结论：
- 推荐的论文主表模型组合为：
  **HA + STGCN/MSTGCN + DCRNN + GWNET/AGCRN + MTGNN/PDFormer + STAEformer/STWave/MegaCRN**
- 该组合覆盖完整技术演进路径，兼顾可解释性与性能表现。

启发，或是下一步可以完成的工作：
- [ ] 整理模型对比表（模型名 / 年份 / 类型 / 是否经典 / 是否推荐）
- [ ] 补充各模型在 PEMSD4 上的 benchmark 指标（MAE / RMSE）
- [ ] 设计统一实验设置（输入窗口、预测步长、归一化方式）

补充说明：
- 风险点：不同模型默认输入维度与任务类型不同，可能导致实验不公平
- 适用范围：交通时空预测论文实验设计
- 限制条件：需在统一配置与数据预处理条件下才具备可比性
