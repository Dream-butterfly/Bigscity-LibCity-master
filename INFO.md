# GNN-TP 项目说明（根目录）

本仓库是交通时空预测实验工程，核心目标是：在统一配置与统一入口下，完成**数据准备、模型训练、调参搜索、断点续训、结果评估**，并保证实验过程可复现、可追踪。

## 项目能力边界

- **已覆盖能力**：单模型训练与评估、数据缓存构建、Optuna 调参、续训、Web 控制台触发训练流程。
- **不在根目录 INFO 直接定义**：具体模型结构细节、单数据集字段语义、单目录内部实现细节（请跳转对应子目录 `INFO.md`）。

## 快速开始

```bash
# 1) 安装依赖（推荐）
uv sync

# 2) 启动 Web 控制台
uv run python run_web.py --host 127.0.0.1 --port 7817

# 3) 命令行：仅数据准备/缓存（不训练）
uv run python scripts/run/run_data_prep.py --task traffic_state_pred --model STGCN --dataset METR_LA

# 4) 命令行：训练并评估单模型
uv run python scripts/run/run_model.py --task traffic_state_pred --model STGCN --dataset METR_LA

# 5) 命令行：超参数搜索
uv run python scripts/run/run_hyper.py --task traffic_state_pred --model STGCN --dataset METR_LA --params_file scripts/run/hyper_example.txt

# 6) 命令行：断点续训并评估
uv run python scripts/run/run_resume.py --task traffic_state_pred --model STGCN --dataset METR_LA
```

## 常用入口与职责（按工作流）

| 场景 | 入口 | 核心行为 | 典型产物 |
| --- | --- | --- | --- |
| 数据准备 | `scripts/run/run_data_prep.py` | 构建数据集对象并触发缓存 | `cache/dataset_cache/` |
| 训练评估 | `scripts/run/run_model.py` | 训练模型并在测试集评估 | `outputs/<exp_id>/` |
| 调参搜索 | `scripts/run/run_hyper.py` | 多组参数试验并记录最优结果 | `outputs/<exp_id>/artifacts/hyper.result` |
| 断点续训 | `scripts/run/run_resume.py` | 基于现有训练状态继续训练并评估 | `outputs/<exp_id>/` |
| Web 控制台 | `run_web.py -> web/train_web_fastapi.py` | 提供可视化训练控制与命令触发 | Web 页面与对应实验输出 |

## 关键参数约定（CLI）

所有 `scripts/run/*.py` 入口都至少支持以下核心参数：

- `--task`：任务名（默认 `traffic_state_pred`）
- `--model`：模型名（默认 `STGCN`）
- `--dataset`：数据集名（默认 `METR_LA`）
- `--config_file`：可选配置文件路径
- `--exp_id`：实验 ID；不传时自动生成 `时间戳__task__model__dataset`
- `--seed`：随机种子（默认 `0`）

另外，训练入口支持通用覆盖参数（如 `--gpu`、`--batch_size`、`--learning_rate`、`--max_epoch`、`--use_amp`、`--use_gradient_checkpointing` 等），用于在不改配置文件时做快速实验。

## 目录结构（根目录）

| 路径 | 作用 |
| --- | --- |
| `GNNTP/` | 核心框架：配置解析、数据加载、模型构建、执行器与评估流水线 |
| `scripts/run/` | 主运行入口（训练/调参/续训/数据准备） |
| `scripts/tools/` | 工具脚本（依赖检查、配置检查、缓存维护、冒烟测试等） |
| `scripts/experiments/` | 不走主流水线的独立实验脚本 |
| `web/` | FastAPI 后端、模板、静态资源（训练控制台） |
| `resource_data/` | 数据资源（如 `METR_LA/`、`PEMSD4/`） |
| `cache/` | 缓存与中间产物（尤其是数据缓存） |
| `outputs/` | 每次实验产物（日志、模型、评估、调参结果） |
| `test/` | 测试脚本与测试辅助代码 |
| `ai_logs/` | AI 变更与分析记录体系 |
| `AI代码准则.md` | AI/开发协作规范 |
| `README.md` | 面向使用者的对外快速说明 |

## 运行输出与落盘约定

- 数据缓存默认在：`cache/dataset_cache/`
- 实验输出默认在：`outputs/<exp_id>/`
- 常见子目录：
  - `outputs/<exp_id>/logs/`：运行日志（含 `run.log`）
  - `outputs/<exp_id>/model_cache/`：模型缓存
  - `outputs/<exp_id>/artifacts/`：调参结果等附加产物

## 调用链总览（帮助快速定位问题）

1. `scripts/run/*.py` 解析参数并组装 `other_args`
2. `GNNTP/config_parser.py` 合并默认配置、任务配置、CLI 覆盖参数
3. `GNNTP/data/` 构建数据集与 DataLoader
4. `GNNTP/models/` 按任务与模型名完成注册查找并实例化
5. `GNNTP/common/` 中执行器驱动训练/评估/续训
6. `GNNTP/utils/utils.py` 统一处理日志、`exp_id`、输出路径

## AI 日志体系（ai_logs）

- 主规范：`ai_logs/AI_LOGS_MAIN.md`
- 总索引：`ai_logs/index.md`（最新在上）
- 变更记录：`ai_logs/change/YYYY-MM/*.md`
- 分析记录：`ai_logs/analysis/YYYY-MM/*.md`
- 要求：所有记录都带来源信息（来源类型与来源说明）

## 开发与修改注意事项

1. 修改入口脚本参数后，需同步检查 Web 端调用参数与文档示例。
2. 新增模型/执行器/评估器时，必须同步注册逻辑（否则运行时找不到类）。
3. 变更输出路径规则时，优先兼容已有 `exp_id` 目录结构，避免历史结果不可读。
4. 涉及缓存清理的脚本要严格限定目录范围，避免误删 `outputs/`。
5. 先阅读对应目录 `INFO.md` 再改代码，减少跨层误改风险。

## 建议阅读顺序

1. `README.md`：快速理解项目能力和入口命令
2. `scripts/run/run_model.py`：训练入口参数与调用方式
3. `GNNTP/pipeline.py`：训练/评估主流程
4. `GNNTP/config_parser.py`、`GNNTP/common/`：配置与执行机制
5. `GNNTP/data/`、`GNNTP/models/`：数据与模型实现细节
6. `ai_logs/index.md`：近期变更与分析上下文

## 子目录 INFO 统一模板（后续补充时遵循）

每个关键目录的 `INFO.md` 建议统一包含 5 个小节：

1. 目录职责（做什么）
2. 关键文件（入口与核心模块）
3. 输入/输出（依赖与产物）
4. 调用关系（被谁调用、调用谁）
5. 修改注意事项（易错点与兼容性约束）

