# STGCN — Traffic Speed Prediction

基于 GCN + 时序卷积的经典交通速度预测模型（通用执行器）。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | STGCN 模型结构（时空卷积块）与前向计算 |
| `config.json` | 默认超参数（KS/Kt/通道数/dropout 等） |
| `manifest.json` | 注册信息（task/model/dataset_class/executor/evaluator） |

## 输入/输出

- **输入**：速度任务时空批数据（时序 + 邻接矩阵）+ data_feature + config
- **输出**：预测速度序列 → executor evaluate → metrics

## 调用关系

1. `manifest.json` → `models/locator.py` 定位模型
2. `common/registry_executor` → 使用默认 `TrafficStateExecutor`
3. 运行入口：`run_train_artifact.py --model STGCN --dataset <dataset> --artifact_id <id>`

## 修改注意事项

1. 卷积输入通道、时间窗长度改动需验证与数据 shape 一致
2. 新参数优先写入 `config.json`，避免硬编码
3. 输出张量维度改动要同步检查 evaluator 指标计算逻辑

