# GNNTP/data/dataset/INFO.md

任务级数据实现层：定义具体数据文件解析、特征构建、切分与缓存策略。

## 关键文件

| 路径 | 作用 |
|------|------|
| `abstract_dataset.py` | 数据集抽象基类（878B） |
| `traffic_state_dataset.py` | 交通状态任务主数据集实现（7.3K） |
| `traffic_state_point_dataset.py` | 点位型交通状态数据集实现（2.4K，最常用） |
| `TrafficStatePointDataset.json` | 点位数据集默认配置 |
| `traffic_state_dataset_mixins.py` | mixin 兼容导入层（620B） |
| `mixins/` | 5 个 mixin 模块：资源/图/时序/外部特征/流水线（数据处理步骤拆分） |
| `traffic_flow_prediction/` | 交通流任务专用数据集（PDFormerDataset） |

## 数据集类层次

```
AbstractDataset (abstract_dataset.py)
  └── TrafficStateDataset (traffic_state_dataset.py)
        ├── TrafficStatePointDataset (traffic_state_point_dataset.py) ← 大多数模型使用
        └── PDFormerDataset (traffic_flow_prediction/pdformer_dataset.py)
```

## 与数据工件的关系

数据集类在 **传统路径** 中直接使用：

```
run_data_prep.py → get_dataset(config) → dataset.get_data() → DataLoader
```

在 **工件路径** 中，数据集类在数据处理阶段被调用一次，训练阶段不再需要：

```
run_data_artifact.py → get_dataset(config) → extract_xy_arrays → 写入 npy 文件
run_train_artifact.py → 直接加载 npy，不再实例化数据集类
```

## 输入/输出

- **输入**：`.geo/.rel/.dyna` 等原始数据文件 + 数据参数（窗口、切分比例、归一化等）
- **输出**：train/valid/test DataLoader + data_feature dict（scaler、num_nodes、feat_dim 等）

## 修改注意事项

1. 字段名、shape、时间维定义改动必须和模型输入约定同步
2. 配置键变更需同步 JSON 配置与解析代码
3. 拆分或移动 mixin 时保留兼容导入路径（`traffic_state_dataset_mixins.py`）
4. 缓存构建策略变更需验证旧缓存行为并明确重建方式

