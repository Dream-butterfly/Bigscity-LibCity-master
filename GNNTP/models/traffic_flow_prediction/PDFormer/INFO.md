# PDFormer/INFO.md

## 目录职责
实现交通流预测模型 PDFormer 及对应执行器扩展。

## 关键文件
- `model.py`：PDFormer 模型结构。
- `executor.py`：PDFormer 专用执行逻辑。
- `config.json` / `manifest.json` / `executor.json`：模型与执行器配置。

## 输入/输出
- 输入：PDFormerDataset 产出的模型输入张量。
- 输出：预测结果、损失及评估缓存。

## 调用关系
- 与 `data/dataset/traffic_flow_prediction/pdformer_dataset.py` 配套使用。

## 修改注意事项
1. 模型输入维度调整要同步 dataset 与 executor。
2. 训练超参数修改优先走配置文件，减少硬编码。

