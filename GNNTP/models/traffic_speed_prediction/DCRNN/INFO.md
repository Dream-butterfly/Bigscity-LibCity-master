# DCRNN — Traffic Speed Prediction

扩散卷积 + 门控循环单元（GRU）交通速度预测模型，有专用执行器。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | DCRNN 网络结构（扩散卷积层 + 编码器-解码器 GRU）与前向逻辑 |
| `executor.py` | DCRNNExecutor（专用训练流程） |
| `config.json` | 默认超参数（diffusion_steps、rnn_units 等） |
| `manifest.json` | 注册信息 |
| `executor.json` | 执行器注册信息 |

## 输入/输出

- **输入**：速度时序数据 + 图结构特征 + data_feature + config
- **输出**：预测序列 → evaluate metrics

## 调用关系

1. `manifest.json` → `models/locator.py` 定位模型
2. `executor.json` → `common/registry_executor` 定位本目录的 DCRNNExecutor
3. 运行入口：`run_train_artifact.py --model DCRNN --dataset <dataset> --artifact_id <id>`

## 修改注意事项

1. 图扩散相关参数改动需同步检查数据侧邻接矩阵构建逻辑
2. 执行器行为改动要保持与通用日志和输出目录规范一致
3. `config.json` 参数名与代码读取名必须一致

