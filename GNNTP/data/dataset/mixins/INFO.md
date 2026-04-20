# mixins/INFO.md

## 目录职责
`data/dataset/mixins/` 存放 TrafficState 数据集的可复用 mixin，实现按职责拆分，降低单文件耦合。

## 关键文件
- `resource_mixin.py`：数据资源 basename 归一化与存在性检查。
- `graph_mixin.py`：`.geo/.rel` 解析与邻接矩阵构建。
- `temporal_loader_mixin.py`：`.dyna/.grid/.od/.gridod/.ext` 时序数据加载。
- `external_feature_mixin.py`：时间与外部特征融合逻辑。
- `pipeline_mixin.py`：滑窗、切分、归一化、dataloader 组装主流程。

## 输入/输出
- 输入：`TrafficStateDataset` 提供的配置、路径、状态字段。
- 输出：供 `TrafficStateDataset` 组合继承后的数据处理能力。

## 调用关系
- 由 `traffic_state_dataset.py` 组合继承。
- `traffic_state_dataset_mixins.py` 作为兼容重导出入口。

## 修改注意事项
1. 变更 mixin 方法签名时需同步检查 `TrafficStateDataset` 与子类调用点。
2. 兼容层 `traffic_state_dataset_mixins.py` 应保持可导入，避免历史引用断裂。
