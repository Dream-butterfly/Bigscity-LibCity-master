# traffic_flow_prediction/INFO.md

## 目录职责

`data/dataset/traffic_flow_prediction/` 提供交通流预测任务的数据集实现，当前主要服务 PDFormer。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `pdformer_dataset.py` | PDFormer 所需输入特征构建与数据处理逻辑 |
| `PDFormerDataset.json` | 数据集默认配置模板 |

## 输入/输出

- **输入**：交通流原始数据文件、任务配置和模型相关参数。
- **输出**：满足 PDFormer 输入约定的训练/验证/测试批数据和数据特征。

## 调用关系

1. 由 `data.factory` 依据任务与模型配置加载。
2. 与 `models/traffic_flow_prediction/PDFormer` 一一对应配套使用。

## 修改注意事项

1. 数据特征字段、shape 和时间窗口改动需同步模型和执行器逻辑。
2. 配置键名必须与 `PDFormerDataset.json` 和代码读取位置保持一致。
3. 任何数据预处理口径变化都要评估历史实验可比性。

