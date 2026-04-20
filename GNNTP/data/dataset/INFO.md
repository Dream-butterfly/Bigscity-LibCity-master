# dataset/INFO.md

## 目录职责

`data/dataset/` 是任务级数据实现层，定义具体数据文件解析、特征构建、切分与缓存策略。

## 关键文件与目录

| 路径 | 作用 |
| --- | --- |
| `abstract_dataset.py` | 数据集抽象基类 |
| `traffic_state_dataset.py` | 交通状态任务主数据集实现 |
| `traffic_state_point_dataset.py` | 点位型交通状态数据集实现 |
| `TrafficStatePointDataset.json` | 点位数据集默认配置 |
| `mixins/` | 数据集处理能力拆分（资源/图/时序/外部特征/流水线） |
| `traffic_state_dataset_mixins.py` | mixin 兼容导入层 |
| `traffic_flow_prediction/` | 交通流任务专用数据集实现 |

## 输入/输出

- **输入**：`.geo/.rel/.dyna/.grid/.od/.gridod/.ext` 等原始数据文件和数据参数。
- **输出**：可训练的 `train/valid/test` 批数据与数据特征字典。

## 调用关系

1. `data.factory` 按配置实例化本目录数据集类。
2. `pipeline.py` 消费该目录产出的数据迭代器与特征信息。
3. 模型目录依赖本目录定义的输入字段和 shape 约定。

## 修改注意事项

1. 字段名、shape、时间维定义改动必须和模型输入约定同步。
2. 配置键变更需同步 JSON 配置与解析代码。
3. 拆分或移动 mixin 时保留兼容导入路径，避免历史引用失效。
4. 缓存构建策略变更需验证旧缓存行为并明确重建方式。

