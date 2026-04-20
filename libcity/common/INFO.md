# common/INFO.md

## 目录职责
`common/` 提供训练执行器、评估器与调参等通用能力，是 pipeline 的执行与评估中枢。

## 关键文件
- `traffic_state_executor.py` / `traffic_state_evaluator.py`：交通状态预测任务的默认执行与评估实现。
- `abstract_executor.py` / `abstract_evaluator.py`：抽象基类。
- `registry_executor.py` / `registry_evaluator.py`：注册与按名称查找实现。
- `hyper_tuning.py`：超参数搜索流程封装。

## 输入/输出
- 输入：模型实例、数据迭代器、配置参数。
- 输出：日志、评估指标、模型保存与评估缓存。

## 调用关系
- 上游由 `libcity/pipeline.py` 调用。
- 下游会调用 `models/` 产出的模型及 `utils/` 工具。

## 修改注意事项
1. 新增执行器/评估器要同步注册。
2. 配置键名要与 `config_parser.py` 和 JSON 配置保持一致。

