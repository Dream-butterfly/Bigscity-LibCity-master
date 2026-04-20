# common/INFO.md

## 目录职责

`common/` 是训练执行与评估层：负责训练循环、验证评估、模型保存恢复、调参流程和执行器/评估器注册。

## 关键文件与职责

| 文件 | 作用 |
| --- | --- |
| `abstract_executor.py` / `abstract_evaluator.py` | 执行器/评估器抽象接口 |
| `traffic_state_executor.py` | 交通状态任务默认训练执行器 |
| `traffic_state_evaluator.py` | 交通状态任务默认评估器 |
| `registry_executor.py` / `registry_evaluator.py` | 名称到实现类的定位与注册 |
| `hyper_tuning.py` | 调参流程封装（搜索、记录、结果保存） |
| `TrafficStateExecutor.json` / `TrafficStateEvaluator.json` | 执行器与评估器配置模板 |
| `evaluator_utils.py` | 评估辅助函数 |

## 输入/输出

- **输入**：配置对象、模型实例、训练/验证/测试数据迭代器。
- **输出**：训练日志、指标结果、模型缓存、评估产物和调参结果。

## 调用关系

1. 上游由 `pipeline.py` 调用并实例化执行器/评估器。
2. 下游调用 `models/` 的模型前向和 `utils/` 的日志/路径工具。
3. 调参入口通过 `hyper_tuning.py` 循环触发训练评估流程。

## 修改注意事项

1. 新增执行器或评估器时，必须同步注册器与配置声明。
2. 训练循环相关变更要同时考虑：续训、早停、日志、保存路径兼容性。
3. 指标计算口径改动应同步更新 Web 展示与历史结果解读说明。
4. 避免在此层硬编码单模型逻辑，模型特有行为优先下沉到模型目录。

