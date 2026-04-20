# test/INFO.md

## 目录职责

`test/` 用于验证训练框架与基线效果，主要覆盖传统统计/机器学习基线和框架接口级冒烟检查。

## 关键文件与范围

| 文件 | 说明 |
| --- | --- |
| `test_ARIMA.py` | ARIMA 基线评估脚本 |
| `test_HA.py` | Historical Average 基线评估 |
| `test_SVR.py` | SVR 基线评估 |
| `test_VAR.py` | VAR 基线评估 |
| `test_model_api.py` | 框架模型 API/流程可用性检查 |
| `baseline_utils.py` | 基线脚本共享工具（数据准备、指标计算、通用逻辑） |
| `readme.md` | 测试执行说明与使用备注 |

## 输入/输出

- **输入**：`resource_data/` 数据、模型或基线参数、可选缓存。
- **输出**：终端指标、测试日志、必要时写入 `outputs/` 的结果文件。

## 调用关系

1. 测试脚本由开发者手动触发，不作为主训练入口链路的一部分。
2. 基线脚本会复用 `GNNTP.data`、`GNNTP.utils` 的部分能力。
3. `test_model_api.py` 主要验证主框架对模型的加载与调用接口是否可用。

## 修改注意事项

1. 新增测试脚本优先复用 `baseline_utils.py`，避免重复实现数据读写和指标逻辑。
2. 保持 `test_*.py` 命名约定，便于批量执行和检索。
3. 涉及随机性的测试需固定 seed，确保结果稳定可复现。
4. 若依赖特定缓存或大文件，必须在脚本或 `readme.md` 写明前置条件。

