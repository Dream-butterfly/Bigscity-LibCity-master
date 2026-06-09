# STGformer — Traffic Speed Prediction（框架版）

框架内 STGformer 实现（模型 + 专用执行器），用于标准训练流水线。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | STGformer 模型结构（Spatio-Temporal Transformer）与前向实现 |
| `executor.py` | STGformer 专用训练流程封装 |
| `config.json` | 默认参数 |
| `manifest.json` | 注册信息 |
| `executor.json` | 执行器注册信息 |

## 输入/输出

- **输入**：速度任务时空特征 + 图信息 + config
- **输出**：预测张量 → evaluate metrics

## 调用关系

1. `manifest.json` → `models/locator.py` 定位模型
2. `executor.json` → `common/registry_executor` 加载本目录执行器
3. 运行入口：`run_train_artifact.py --model STGformer --dataset <dataset> --artifact_id <id>`

## 修改注意事项

1. 与 `STGformer-独立实现` 的差异需保持明确，避免路径和逻辑混用
2. 执行器改动要同步校验续训、模型保存、日志落盘行为
3. 新增参数优先写入 `config.json`

