# GNN-TP

面向交通时空预测的工程化训练仓库。当前版本以 `scripts/run` 为统一 CLI 入口，以 `run_web.py` 提供 Web 训练控制台，并配套 `ai_logs/` 做结构化变更与分析记录。

## 当前能力

- 单模型训练与评估：`scripts/run/run_model.py`
- 仅数据准备与缓存构建：`scripts/run/run_data_prep.py`
- 超参数搜索：`scripts/run/run_hyper.py`
- 断点续训与评估：`scripts/run/run_resume.py`
- Web 控制台：`run_web.py -> web/train_web_fastapi.py`

## 环境要求

- Python `>=3.11,<3.13`
- 推荐使用 `uv`

## 快速开始

```bash
# 1) 安装依赖
uv sync

# 2) 启动 Web 控制台
uv run python run_web.py --host 127.0.0.1 --port 7817

# 下面的命令行示例均可通过 Web 控制台的命令输入框执行，或直接在终端运行：

# 2.1) 仅准备数据/缓存
uv run python scripts/run/run_data_prep.py --task traffic_state_pred --model STGCN --dataset PEMSD4

# 2.2) 训练并评估
uv run python scripts/run/run_model.py --task traffic_state_pred --model STGCN --dataset PEMSD4

# 2.3) 超参数搜索
uv run python scripts/run/run_hyper.py --task traffic_state_pred --model STGCN --dataset PEMSD4 --params_file scripts/run/hyper_example.txt

# 2.4) 续训并评估
uv run python scripts/run/run_resume.py --task traffic_state_pred --model STGCN --dataset PEMSD4

```

## 目录说明（当前结构）

| 路径 | 作用 |
|---|---|
| `GNNTP/` | 核心训练框架与模型实现 |
| `scripts/run/` | 主运行入口脚本（训练/调参/续训/数据准备） |
| `scripts/tools/` | 辅助工具脚本（依赖/配置/数据检查、缓存维护、冒烟测试） |
| `scripts/` | 脚本总目录（含 `run/`、`tools/`、`experiments/`） |
| `web/` | Web 后端、模板、静态资源 |
| `resource_data/` | 数据资源（如 `METR_LA/`、`PEMSD4/`） |
| `cache/` | 运行缓存与中间产物 |
| `outputs/` | 训练输出、评估结果、模型产物 |
| `test/` | 基线与接口测试脚本 |
| `ai_logs/` | AI 日志规范、索引与按月归档记录 |

## 数据与产物约定

- 数据放在 `resource_data/`，目录名需与 `--dataset` 对应。
- 数据缓存默认写入 `cache/dataset_cache/`。
- 训练产物默认写入 `outputs/<exp_id>/`（日志、模型、评估缓存）。

## AI 日志体系

- 主规范：`ai_logs/AI_LOGS_MAIN.md`
- 总索引：`ai_logs/index.md`
- 变更记录：`ai_logs/change/YYYY-MM/*.md`
- 分析记录：`ai_logs/analysis/YYYY-MM/*.md`

## 进一步阅读

- 根目录总说明：`INFO.md`
- 子目录说明：`scripts/INFO.md`、`scripts/run/INFO.md`、`web/INFO.md`、`resource_data/INFO.md`、`cache/INFO.md`、`outputs/INFO.md`、`test/INFO.md`
