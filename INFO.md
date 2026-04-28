# GNN-TP 项目说明（根目录）

本仓库是交通时空预测实验工程，核心目标是：在统一配置与统一入口下，完成**数据准备、模型训练、调参搜索、断点续训、结果评估**，并保证实验过程可复现、可追踪。

> **不在根 INFO 直接定义**：具体模型结构、单数据集字段语义、子目录实现细节。这些请跳转对应子目录 `INFO.md`。

- **GitHub**：`Dream-butterfly/Bigscity-LibCity-master`，分支 `重构-数据集处理独立`
- **代码准则**：`AI代码准则.md`（Karpathy 四原则，所有 AI 修改的硬约束）
- **安全护栏**：gnntp-guard skill 三层权限（🔴核心 / 🟡确认 / 🟢自由），修改前自动检查

## 技术栈与环境

| 组件 | 版本/工具 |
| --- | --- |
| Python | 3.11–3.12 |
| 包管理 | uv |
| PyTorch | 2.11 |
| torch-geometric | 2.7 |
| CUDA | 12.8 |
| 关键依赖 | Optuna, dtaidistance, timm, tslearn, statsmodels |
| Web 后端 | FastAPI + uvicorn |

## 目录结构

| 路径 | 作用 | 详情 |
| --- | --- | --- |
| `GNNTP/` | 核心框架 | [INFO](GNNTP/INFO.md) |
| `scripts/run/` | 主运行入口（训练/调参/续训/数据准备） | [INFO](scripts/run/INFO.md) |
| `scripts/tools/` | 工具脚本 | [INFO](scripts/tools/INFO.md) |
| `scripts/experiments/` | 独立实验脚本 | [INFO](scripts/experiments/INFO.md) |
| `web/` | FastAPI 训练控制台 | [INFO](web/INFO.md) |
| `resource_data/` | 原始数据（METR_LA, PEMSD4/7/8 等） | — |
| `cache/` | 数据缓存与工件（`dataset_cache/`, `data_artifacts/`） | — |
| `outputs/` | 实验产物 | — |
| `test/` | 测试脚本 | [INFO](test/INFO.md) |
| `paper/` | 论文工作区（LaTeX, 图表） | [INFO](paper/INFO.md) |
| `ai_logs/` | AI 变更与分析记录 | [INFO](ai_logs/INFO.md) |
| `AI代码准则.md` | AI 协作行为规范 | — |
| `README.md` | 对外项目说明 | — |

产物子目录约定：
- `outputs/<exp_id>/logs/` — 运行日志（含 `run.log`）
- `outputs/<exp_id>/model_cache/` — 模型权重
- `outputs/<exp_id>/artifacts/` — 调参结果等附加产物

## 快速开始

所有入口脚本共享核心参数：`--task`（默认 `traffic_state_pred`）/ `--model` / `--dataset`（默认 `METR_LA`）/ `--exp_id`（不传则自动生成 `时间戳__task__model__dataset`）/ `--seed`（默认 0）。训练入口额外支持 `--gpu`、`--batch_size`、`--learning_rate`、`--max_epoch`、`--use_amp` 等覆盖参数。

```bash
# ========== 环境 ==========
uv sync

# ========== 传统链路 ==========
uv run python scripts/run/run_data_prep.py          --task traffic_state_pred --model STGCN --dataset METR_LA    # 仅数据缓存
uv run python scripts/run/run_model.py              --task traffic_state_pred --model STGCN --dataset METR_LA    # 训练+评估
uv run python scripts/run/run_hyper.py              --task traffic_state_pred --model STGCN --dataset METR_LA --params_file scripts/run/hyper_example.txt  # 超参搜索
uv run python scripts/run/run_resume.py             --task traffic_state_pred --model STGCN --dataset METR_LA    # 断点续训

# ========== 解耦链路（推荐：数据−训练分离） ==========
uv run python scripts/run/run_data_artifact.py      --task traffic_state_pred --model STGCN --dataset METR_LA                    # 构建数据工件
uv run python scripts/run/run_train_artifact.py     --task traffic_state_pred --model STGCN --dataset METR_LA --artifact_id <id>  # 消费工件训练
uv run python scripts/run/run_hyper_artifact.py     --task traffic_state_pred --model STGCN --dataset METR_LA --artifact_id <id> --params_file ...  # 消费工件调参
uv run python scripts/run/run_resume_artifact.py    --run_id <run_id> --artifact_id <id> --epoch 10 --max_epoch 20               # 消费工件续训

# ========== Web ==========
uv run python run_web.py --host 127.0.0.1 --port 7817
```

| 场景 | 入口 | 产物 |
| --- | --- | --- |
| 数据准备 | `scripts/run/run_data_prep.py` | `cache/dataset_cache/` |
| 训练评估 | `scripts/run/run_model.py` | `outputs/<exp_id>/` |
| 调参搜索 | `scripts/run/run_hyper.py` | `outputs/<exp_id>/artifacts/hyper.result` |
| 断点续训 | `scripts/run/run_resume.py` | `outputs/<exp_id>/` |
| 数据工件（解耦全流程） | `scripts/run/run_data_artifact.py` → `run_train_artifact.py` / `run_hyper_artifact.py` / `run_resume_artifact.py` | `cache/data_artifacts/` + `outputs/` |
| Web 控制台 | `run_web.py` → `web/train_web_fastapi.py` | Web 页面 → `outputs/` |

## 调用链

1. `scripts/run/*.py` 解析参数并组装 `other_args`
2. `GNNTP/config_parser.py` 合并默认配置、任务配置、CLI 覆盖参数
3. `GNNTP/data/` 构建数据集与 DataLoader
4. `GNNTP/models/` 按任务与模型名完成注册查找并实例化
5. `GNNTP/common/` 中执行器驱动训练/评估/续训
6. `GNNTP/utils/utils.py` 统一处理日志、`exp_id`、输出路径

> **两条链路并存**：传统链路 data → train 在同一进程完成；解耦链路先 `run_data_artifact.py` 写入工件，再 `run_train_artifact.py` 独立消费。详见 [GNNTP/INFO.md](GNNTP/INFO.md)。

## 论文

- **目标期刊**：Information Sciences (Elsevier)
- **论文模型**：FuzDiff（条件扩散 Transformer + 模糊图学习 + 物理守恒损失）
- **代码**：`GNNTP/models/new/new_diffusion_fuzzy/`
- **源码**：`paper/src/`，中文版主力 `main_cn.tex`（6 章：引言/相关工作/方法论/实验/结论/摘要）
- **模板**：Elsevier CAS Bundle 2.4（`paper/els-cas-templates/`，使用 `cas-sc.cls` 单栏格式）
- **编译**：`cd paper/src && pdflatex main_cn && bibtex main_cn && pdflatex main_cn && pdflatex main_cn`
- **写作详情**：[paper/INFO.md](paper/INFO.md) | [paper/src/INFO.md](paper/src/INFO.md)

## 修改须知

1. 修改入口脚本参数后，需同步检查 Web 端调用与文档示例。
2. 新增模型/执行器/评估器必须补齐注册逻辑，否则运行时找不到类。
3. 变更输出路径规则时，优先兼容已有 `exp_id` 目录结构，避免历史结果不可读。
4. 涉及缓存清理的脚本要严格限定目录范围（如 `cache/dataset_cache/`），避免误删 `outputs/`。
5. 先阅读对应目录 `INFO.md` 再改代码，减少跨层误改风险。
6. AI 日志规范：所有修改必须按模板追加到 `ai_logs/change/`，详见 `ai_logs/AI_LOGS_MAIN.md`。

## 导航起点

| 角色 | 推荐路径 |
| --- | --- |
| **首次接触项目** | `README.md` → 本文件 → `GNNTP/pipeline.py` → `scripts/run/run_model.py` |
| **改模型代码** | `GNNTP/models/<name>/INFO.md` → `GNNTP/data/INFO.md` → `GNNTP/common/INFO.md` |
| **写/改论文** | `paper/INFO.md` → `paper/src/INFO.md` → `paper/src/sections_cn/methodology.tex` |
| **排查运行问题** | `ai_logs/index.md` → `outputs/<exp_id>/logs/run.log` → 调用链（见上） |
| **跑实验** | 快速开始 → `scripts/run/INFO.md` → `GNNTP/models/<name>/config.json` |

## 子目录 INFO 模板

每个关键目录的 `INFO.md` 统一包含 5 个小节：

1. **目录职责** — 做什么
2. **关键文件** — 入口与核心模块
3. **输入/输出** — 依赖与产物
4. **调用关系** — 被谁调用、调用谁
5. **修改注意事项** — 易错点与兼容性约束
