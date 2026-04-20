# models/INFO.md

## 目录职责

`models/` 是模型实现中心，负责模型抽象、注册定位、任务子目录模型实现和损失相关逻辑。

## 关键文件与目录

| 路径 | 作用 |
| --- | --- |
| `abstract_model.py` | 模型抽象基类 |
| `abstract_traffic_state_model.py` | 交通状态任务模型抽象层 |
| `registry.py` / `locator.py` | 模型注册与按名称定位 |
| `loss.py` | 损失函数与训练相关损失工具 |
| `traffic_speed_prediction/` | 交通速度预测模型实现 |
| `traffic_flow_prediction/` | 交通流预测模型实现 |
| `new/` | 实验性或新增模型实现 |

## 输入/输出

- **输入**：配置对象、`data_feature`、批数据张量。
- **输出**：模型预测张量、训练损失相关中间结果。

## 调用关系

1. `pipeline.py` 通过 `locator/registry` 创建模型实例。
2. `common` 目录中的执行器调用模型前向完成训练与评估。
3. 模型目录与 `data/` 的字段定义、shape 约定强耦合。

## 修改注意事项

1. 新模型必须提供 `model.py`、`manifest.json`、`config.json`（及需要时的 `executor.*`）。
2. `manifest`、注册名、CLI `--model` 名称要一致，避免定位失败。
3. 输入输出 shape 变更须同步验证数据集与执行器逻辑。
4. 实验模型稳定后再沉淀到主任务目录，保持主流程可维护性。

