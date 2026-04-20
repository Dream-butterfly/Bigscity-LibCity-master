# traffic_flow_prediction/INFO.md

## 目录职责

`models/traffic_flow_prediction/` 存放交通流预测任务模型实现，当前主实现为 PDFormer。

## 关键子目录

| 目录 | 作用 |
| --- | --- |
| `PDFormer/` | PDFormer 模型 + 专用执行器实现 |

## 输入/输出

- **输入**：交通流任务数据批次、模型参数、训练配置。
- **输出**：交通流预测结果、损失与评估相关产物。

## 调用关系

1. `models.locator` 根据任务和模型名定位本目录实现。
2. 与 `data/dataset/traffic_flow_prediction/` 数据实现配套协同。

## 修改注意事项

1. 新增模型时需同步补齐注册元信息和默认配置文件。
2. 输入字段和 shape 变更需同步数据集与执行器。
3. 任务级目录应保持结构清晰，避免实验模型和稳定模型混放。

