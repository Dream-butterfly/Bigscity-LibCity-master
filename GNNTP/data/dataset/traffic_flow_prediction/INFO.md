# traffic_flow_prediction/INFO.md

## 目录职责
该目录放置交通流预测任务的专用数据集实现。

## 关键文件
- `pdformer_dataset.py`：PDFormer 对应的数据集处理逻辑。
- `PDFormerDataset.json`：默认配置项定义。

## 输入/输出
- 输入：交通流原子文件与 PDFormer 配置参数。
- 输出：符合 PDFormer 输入要求的数据批次。

## 调用关系
- 由 `data/dataset` 层按模型/任务组合加载。

## 修改注意事项
1. 特征构建逻辑变更需和 `models/traffic_flow_prediction/PDFormer` 对齐。
2. 保持配置键名与代码读取一致。

