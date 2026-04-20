# mixins/INFO.md

## 目录职责

`data/dataset/mixins/` 将 `TrafficStateDataset` 的复杂数据处理流程拆分为可组合模块，降低单文件复杂度并提升复用性。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `resource_mixin.py` | 数据资源命名规范化与文件存在性检查 |
| `graph_mixin.py` | 图结构文件解析与邻接矩阵构建 |
| `temporal_loader_mixin.py` | 时序主数据加载（`dyna/grid/od/gridod/ext`） |
| `external_feature_mixin.py` | 时间特征与外部特征融合 |
| `pipeline_mixin.py` | 滑窗、切分、归一化、DataLoader 组装主流程 |

## 输入/输出

- **输入**：`TrafficStateDataset` 提供的配置、路径、统计状态。
- **输出**：标准化数据处理步骤与中间结果，供主数据集类组合调用。

## 调用关系

1. 由 `traffic_state_dataset.py` 通过多重继承或组合方式使用。
2. `traffic_state_dataset_mixins.py` 作为兼容导入层，对外暴露稳定路径。

## 修改注意事项

1. mixin 方法签名变更要同步检查 `TrafficStateDataset` 及其子类调用点。
2. 混入顺序相关逻辑（依赖字段初始化顺序）改动要谨慎，避免隐式行为变化。
3. 兼容层导出不可随意移除，防止历史代码导入失败。
4. 数据处理步骤变更应关注缓存一致性与评估口径一致性。
