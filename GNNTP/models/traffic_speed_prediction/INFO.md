# GNNTP/models/traffic_speed_prediction/INFO.md

交通速度预测任务的模型实现汇总。

## 模型列表

| 目录 | 专用 Executor | 说明 |
|------|---------------|------|
| `STGCN/` | 否 | STGCN（通用执行器），标准基线 |
| `DCRNN/` | ✅ `DCRNNExecutor` | DCRNN：扩散卷积 + 门控循环单元 |
| `STTN/` | 否 | STTN（通用执行器），Spatio-Temporal Transformer |
| `STGformer/` | ✅ 有 | 框架内 STGformer（标准流水线） |
| `STGformer-独立实现/` | 独立脚本 | 实验版 STGformer（不走主流水线） |

## 输入/输出（黑盒）

- **输入**：速度任务批数据（时序 + 图结构 + 可选外部特征）+ `data_feature` + `config`
- **输出**：预测速度序列 → executor evaluate → `evaluate_cache/*.csv` + `*_predictions.npz`

## 调用关系

1. 主流程 `run_train_artifact.py` 通过 `--model` 选择本目录实现
2. 模型通过 `manifest.json` 被 `models/locator.py` 发现
3. 有专用执行器的模型通过 `executor.json` 被 `common/registry_executor` 加载
4. `STGformer-独立实现` 不经过主流水线，由独立脚本触发

## 修改注意事项

1. 新增模型需补齐 `model.py` + `config.json` + `manifest.json`
2. 模型输入字段与 shape 改动需同步 `data/dataset` 与执行器
3. 框架版与独立实现并存时，命名和文档需清晰区分，避免误用

