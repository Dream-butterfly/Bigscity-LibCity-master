# scripts/run/INFO.md

## 目录职责
`scripts/run/` 存放核心运行入口脚本，供命令行和 Web 后端统一调用。

## 关键文件
- `run_model.py`：训练并评估单模型。
- `run_data_prep.py`：仅执行数据准备与缓存构建。
- `run_hyper.py`：超参数搜索。
- `run_resume.py`：继续训练并评估。

## 输入/输出
- 输入：`task/model/dataset` 与配置参数。
- 输出：日志、模型文件、评估结果与缓存。

## 调用关系
- 可直接由命令行执行。
- 由 `web/train_web_flask.py` 通过 `uv run` 子进程调用。

## 修改注意事项
1. 参数定义变更需同步 Web 端调用参数。
2. 入口行为应保持稳定，避免影响自动化流程。

