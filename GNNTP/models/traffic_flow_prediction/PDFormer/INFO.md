# PDFormer/INFO.md

## 目录职责

实现交通流预测模型 PDFormer 及其专用执行器。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | PDFormer 模型定义 |
| `executor.py` | PDFormer 专用训练与评估执行器 |
| `config.json` | 默认超参数配置 |
| `manifest.json` | 模型注册信息 |
| `executor.json` | 执行器注册信息 |

## 输入/输出

- **输入**：`PDFormerDataset` 输出的批数据与模型配置。
- **输出**：预测结果、训练损失、评估指标和缓存产物。

## 调用关系

1. `models.locator` 定位模型，`common.registry_executor` 定位执行器。
2. 与 `data/dataset/traffic_flow_prediction/pdformer_dataset.py` 强绑定配套。

## 修改注意事项

1. 模型输入维度或字段改动需同步调整数据集与执行器。
2. 参数优先放在 `config.json`，减少代码硬编码。
3. 执行器改动要保证日志、保存路径和评估输出行为与项目规范一致。

