# PDFormer — Traffic Flow Prediction（框架版）

Propagation Delay-aware Dynamic Long-range Transformer 交通流预测模型（专用执行器）。

⚠️ 注意：`GNNTP/models/traffic_flow_prediction/PDFormer/` 与 `GNNTP/models/baseline/PDFormer/` 是两个独立的迁移实现，不要混淆。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | PDFormer 模型定义（延迟感知注意力 + 动态长程 Transformer） |
| `executor.py` | PDFormer 专用训练与评估执行器 |
| `config.json` | 默认超参数 |
| `manifest.json` | 注册信息 |
| `executor.json` | 执行器注册信息 |

## 输入/输出

- **输入**：`PDFormerDataset` 输出的批数据 + model config
- **输出**：预测结果 → evaluate metrics

## 调用关系

1. `manifest.json` → `models/locator.py` 定位模型
2. `executor.json` → `common/registry_executor` 定位本目录执行器
3. 与 `data/dataset/traffic_flow_prediction/pdformer_dataset.py` 强绑定配套
4. 运行入口：`run_train_artifact.py --model PDFormer --dataset <dataset>`

## 修改注意事项

1. 模型输入维度或字段改动需同步调整数据集与执行器
2. 参数优先放在 `config.json`，减少代码硬编码
3. 执行器改动要保证日志、保存路径和评估输出行为与项目规范一致

