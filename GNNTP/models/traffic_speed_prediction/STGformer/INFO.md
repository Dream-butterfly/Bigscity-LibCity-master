# STGformer/INFO.md

## 目录职责

实现框架内 STGformer 版本（模型 + 专用执行器），用于标准训练流水线。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | STGformer 模型结构与前向实现 |
| `executor.py` | STGformer 专用训练流程封装 |
| `config.json` | 模型默认参数 |
| `manifest.json` | 模型注册元信息 |
| `executor.json` | 执行器注册元信息 |

## 输入/输出

- **输入**：速度任务时空特征、图信息和训练配置。
- **输出**：预测张量、训练损失、验证与测试指标。

## 调用关系

1. `pipeline -> models.locator` 定位模型。
2. `common.registry_executor` 加载本目录执行器完成训练。
3. 与 `data/dataset` 约定输入字段强关联。

## 修改注意事项

1. 与 `STGformer-独立实现` 的差异需保持明确，避免路径和逻辑混用。
2. 执行器改动要同步校验续训、模型保存、日志落盘行为。
3. 新增参数优先写入 `config.json` 并保持默认值可复现。

