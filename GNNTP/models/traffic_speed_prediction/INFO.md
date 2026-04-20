# traffic_speed_prediction/INFO.md

## 目录职责

`models/traffic_speed_prediction/` 汇总交通速度预测任务的模型实现及其任务级组织。

## 关键子目录

| 目录 | 作用 |
| --- | --- |
| `DCRNN/` | DCRNN 模型与专用执行器 |
| `STGCN/` | STGCN 模型实现 |
| `STTN/` | STTN 模型实现 |
| `STGformer/` | 框架内 STGformer 模型与执行器 |
| `STGformer-独立实现/` | 独立实验版本 STGformer（不走主流程） |

## 输入/输出

- **输入**：速度任务批数据（时序、图结构、可选外部特征）和模型配置。
- **输出**：预测速度序列、训练损失和评估指标。

## 调用关系

1. 主流程通过 `--model` 选择本目录对应实现。
2. 具备专用执行器的模型（如 DCRNN/STGformer）通过 `executor.json` 对接 `common`。
3. `STGformer-独立实现` 主要由独立实验脚本触发，不是主流程默认链路。

## 修改注意事项

1. 新增模型目录需补齐注册元信息与默认配置文件。
2. 模型输入字段与 shape 改动需同步 `data/dataset` 与执行器。
3. 框架版与独立实现并存时，命名和文档需清晰区分，避免误用。

