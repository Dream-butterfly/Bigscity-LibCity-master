# STGformer — 独立实验版（Non-pipeline）

STGformer 独立实验实现，用于与框架版（`traffic_speed_prediction/STGformer/`）做对照。不经过主流水线，不走工件机制。

## 关键子目录

| 目录 | 作用 |
|------|------|
| `model/` | 模型定义、训练入口、YAML 配置 |
| `lib/` | 数据准备、图算法、指标、工具函数 |

## 与框架版 STGformer/ 的关系

| 维度 | 框架版 | 独立版 |
|------|--------|--------|
| 路径 | `traffic_speed_prediction/STGformer/` | 本目录 |
| 流水线 | 主流程工件流水线 | 独立脚本 |
| 配置 | `config.json` + ConfigParser | `STGformer.yaml` |
| 执行器 | 专用 executor.py | `train.py` 内置 |
| 数据 | data_artifact | 独立 data_prepare.py |

## 修改注意事项

1. 避免直接复用主流程内部对象，保持独立实验边界清晰
2. 与框架版结果对比时，需统一数据切分与评估指标口径
3. 若计划并入主流程，需补齐 manifest/config/executor 接线文件

