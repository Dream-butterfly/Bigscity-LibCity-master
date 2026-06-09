# GNNTP/data/dataset/mixins/INFO.md

将 `TrafficStateDataset` 的复杂数据处理流程拆分为 5 个可组合 mixin 模块，降低单文件复杂度并提升复用性。

## 关键文件

| 文件 | 作用 |
|------|------|
| `resource_mixin.py` | 数据资源命名规范化与文件存在性检查 |
| `graph_mixin.py` | 图结构文件（`.rel`）解析与邻接矩阵构建 |
| `temporal_loader_mixin.py` | 时序主数据加载（`.dyna/.grid/.od/.gridod/.ext`） |
| `external_feature_mixin.py` | 时间特征（time_in_day/day_in_week）与外部特征加载 |
| `pipeline_mixin.py` | 滑窗切分 → 归一化 → DataLoader 组装主流程 |

## mixin 组合链（执行顺序）

```
1. resource_mixin     ← 路径规范化 + 文件存在性检查
2. graph_mixin        ← 加载 .rel → 构建邻接矩阵（adj_mx）
3. temporal_loader_mixin  ← 加载时序数据
4. external_feature_mixin ← 融合时间/外部特征
5. pipeline_mixin     ← 滑窗生成样本 → train/valid/test 切分 → 归一化 → DataLoader
```

## 调用关系

1. `traffic_state_dataset.py` 通过多重继承组合使用这 5 个 mixin
2. `traffic_state_dataset_mixins.py` 作为兼容导入层，对外暴露稳定路径

## 修改注意事项

1. mixin 方法签名变更要同步检查 `TrafficStateDataset` 及其子类调用点
2. 混入顺序（MRO）相关逻辑改动要谨慎，避免隐式行为变化
3. 兼容层导出不可随意移除（`traffic_state_dataset_mixins.py`）
4. 数据处理步骤变更应关注缓存一致性与评估口径一致性
