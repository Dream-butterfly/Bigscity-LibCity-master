# scripts/run/INFO.md

## 目录职责

`scripts/run/` 是主流程运行入口层，负责把 CLI 参数转换为框架调用，并触发标准训练流水线。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `run_model.py` | 单模型训练 + 评估入口 |
| `run_data_prep.py` | 仅构建数据与缓存，不执行训练 |
| `run_hyper.py` | 超参数搜索入口（Optuna 封装） |
| `run_resume.py` | 断点续训并评估入口 |
| `run_data_artifact.py` | 数据工件构建入口（解耦链路） |
| `run_train_artifact.py` | 仅读取数据工件训练 + 评估入口（解耦链路） |
| `run_hyper_artifact.py` | 仅读取数据工件进行超参数搜索（解耦链路） |
| `run_resume_artifact.py` | 基于 run_id + 数据工件续训入口（解耦链路） |
| `hyper_example.txt` | 调参空间示例配置 |

## 输入/输出

- **输入**：`task/model/dataset/config_file`、通用覆盖参数（如 `batch_size`、`max_epoch`、`use_amp`），以及解耦链路下的 `artifact_id/artifact_path/run_id`。
- **输出**：`outputs/<exp_id>/` 实验目录（日志、模型、指标、调参结果）、`cache/dataset_cache/` 传统缓存、`cache/data_artifacts/` 解耦数据工件。

## 调用关系

1. 命令行直接调用本目录脚本。
2. Web 控制台通过子进程调用本目录脚本。
3. 旧链路脚本内部将参数交给 `GNNTP.pipeline` / `GNNTP.common` / `GNNTP.data` 完成执行。
4. 解耦链路脚本通过 `GNNTP.data.artifact_io` 读写数据工件并执行严格校验。

## 修改注意事项

1. 参数变更要同步 Web 参数映射与文档示例命令。
2. 默认值和行为变更需关注历史实验脚本兼容性（尤其 `exp_id`、`seed`、`saved_model`）。
3. 入口脚本应尽量保持“薄层”，业务逻辑优先下沉到 `GNNTP/` 模块中。
4. 新增入口脚本时，需在根目录 `INFO.md` 与 `README.md` 补充用途。
5. `run_resume_artifact.py` 默认要求 `outputs/<run_id>/run_meta.json` 已存在；该文件由 `run_train_artifact.py` 生成并维护。

