# STTN — Traffic Speed Prediction

Spatio-Temporal Transformer Network 交通速度预测模型（通用执行器）。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | STTN 模型结构（时空注意力）与前向逻辑 |
| `config.json` | 默认超参数 |
| `manifest.json` | 注册信息 |

## 输入/输出

- **输入**：速度任务时空批数据 + data_feature + config
- **输出**：预测序列 → evaluate metrics

## 调用关系

1. `manifest.json` → `models/locator.py` 定位模型
2. 通用 `TrafficStateExecutor` 调度训练/评估
3. 运行入口：`run_train_artifact.py --model STTN --dataset <dataset> --artifact_id <id>`

## 修改注意事项

1. 模型结构改动需同步检查 `config.json` 默认参数
2. 输出 shape 变更要同步检查执行器与评估器读写逻辑
3. 参数新增保持与配置文件读取键一致

