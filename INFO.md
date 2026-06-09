# GNNTP 项目根目录

交通时空预测实验工程，核心设计理念：**数据处理与模型训练解耦**（数据工件机制）。

## 项目架构全景

```
原始数据 (resource_data/<dataset>/)               用户 (Web / CLI)
       │                                                  │
       ▼                                                  ▼
┌─────────────────┐                          ┌──────────────────────┐
│ 数据处理          │                          │ Web 控制台             │
│ run_data_artifact│                          │ train_web_v2.html     │
│      .py         │                          │ train_web_fastapi.py  │
└────────┬─────────┘                          └───────────┬──────────┘
         │ 产物: data_artifact                             │ subprocess
         ▼                                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    GNNTP 核心框架                                     │
│  ConfigParser → DataRuntime → get_model → Executor.train/evaluate   │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│ outputs/<exp_id__task__model__dataset>/    │
│   ├── model_cache/       ← checkpoint      │
│   ├── evaluate_cache/    ← metrics+pred    │
│   └── logs/              ← run.log         │
└────────────────────────────────────────────┘
```

## 快速开始

```bash
# 安装依赖
uv sync

# 启动 Web 控制台（推荐）
uv run run_web.py

# 数据处理（生成数据工件）
uv run scripts/run/run_data_artifact.py --model STGCN --dataset METR_LA

# 训练（绑定数据工件）
uv run scripts/run/run_train_artifact.py --model STGCN --dataset METR_LA --artifact_id <id>

# 续训
uv run scripts/run/run_resume_artifact.py --run_id <run_id> --artifact_id <id> --epoch 10 --max_epoch 20

# 评估 checkpoint
uv run scripts/run/run_eval_checkpoint.py --run_id <run_id> --epoch 10 --artifact_id <id>
```

## 目录速查

| 路径 | 作用 |
| --- | --- |
| `GNNTP/` | 核心框架：ConfigParser → DataRuntime → 模型 → Executor |
| `GNNTP/common/` | 执行器/评估器抽象与实现（含 TrafficStateExecutor + Evaluator） |
| `GNNTP/data/` | 数据处理流水线：数据集、DataLoader、数据工件读写 |
| `GNNTP/data/dataset/` | 具体数据集实现（TrafficStatePointDataset、PDFormerDataset 等） |
| `GNNTP/data/artifact_io.py` | **数据工件机制**：签名校验、npy 读写、scalar 序列化 |
| `GNNTP/models/` | 模型实现：分层（traffic_speed_prediction/traffic_flow_prediction/new/baseline） |
| `GNNTP/models/locator.py` | 模型按名称定位与注册 |
| `GNNTP/utils/` | 工具集（路径/日志/seed/归一化/参数解析） |
| `web/` | FastAPI 后端 + v2 前端（train_web_v2.html + train_web_fastapi.py） |
| `scripts/run/` | 命令行运行入口（数据/训练/续训/评估/调参） |
| `scripts/tools/` | 运维工具（检查依赖、配置、缓存、冒烟测试） |
| `scripts/experiments/` | 独立实验脚本 |
| `resource_data/` | 原始数据资源（METR_LA、PEMSD4 等） |
| `cache/` | 中间缓存（dataset_cache、data_artifacts） |
| `outputs/` | 实验产物（checkpoints、evaluate_cache、logs） |
| `test/` | 基线测试（ARIMA/SVR/HA/VAR）+ 冒烟测试 |
| `paper/` | 论文写作工作区（LaTeX 源码、图表、模板） |
| `ai_logs/` | AI 变更日志体系 |

## 两条流水线

### 主流：数据工件流水线（推荐）

```
run_data_artifact.py → data_artifact (cache/data_artifacts/<id>/)
  → run_train_artifact.py / run_resume_artifact.py / run_eval_checkpoint.py
  → outputs/<exp_id>/
```

数据版本元信息同时写入 `outputs/data_versions/<version_id>/`，Web 端通过版本管理选择数据。

### 旧流水线（兼容）

```
run_data_prep.py → cache/dataset_cache/
  → run_model.py / run_resume.py / run_hyper.py
  → outputs/<exp_id>/
```

## 数据工件机制（核心设计）

数据处理与训练解耦的关键：

- **数据处理**：`run_data_artifact.py` → `build_dataset_runtime()` → `extract_xy_arrays()` → 将 numpy 数组写入 `data_artifacts/<id>/`
- **训练**：`run_train_artifact.py` → `build_artifact_runtime(artifact_id)` → 加载预处理的 npy 文件 → 重新包装为 DataLoader → 训练
- **签名校验**：数据签名（dataset + seed + split + scaler + window 的哈希）确保数据与配置一致
- **锁定键**：训练时 `DATA_LOCKED_CONFIG_KEYS`（dataset/seed/dataset_class/train_rate/eval_rate/input_window/output_window/scaler 等）不可覆盖

## 运行输出落盘约定

- 数据工件：`cache/data_artifacts/<artifact_id>/`
- 数据版本元信息：`outputs/data_versions/<version_id>/`
- 实验输出：`outputs/<timestamp>__<task>__<model>__<dataset>/`
  - `model_cache/<model>_<dataset>_epoch<N>.tar` — checkpoint
  - `evaluate_cache/<metrics>.csv` + `*_predictions.npz` — 评估结果
  - `effective_config.json` — 训练完整配置（供续训恢复）
  - `run_meta.json` — 运行元信息
- 运行历史：`outputs/web_train_history.json`

## Web 控制台六个页面

1. **数据处理**：生成数据工件 + 版本管理 + 数据集预览
2. **训练**：选数据版本 → 配置参数 → 异步训练（实时日志/参数图/Loss 曲线/预测演示）
3. **继续训练**：选历史 run + checkpoint epoch → 续训
4. **模型对比**：多模型 predictions 叠加对比图
5. **模型评估**：从 checkpoint 加载模型纯评估（可独立并行）
6. **运行历史**：所有训练/评估任务记录

## 模型黑盒约定

- **输入**：`data_feature`（scaler, num_nodes, feat_dim, output_dim）+ `config`（超参数）
- **前向**：`model(x)` 其中 x 形状 `[B, input_window, num_nodes, feat_dim]`
- **输出**：y 形状 `[B, output_window, num_nodes, output_dim]`
- **评估产物**：`evaluate_cache/*.csv`（horizon × metrics）+ `*_predictions.npz`（pred/truth）

## 开发注意事项

1. 新增模型需提供 `model.py` + `config.json` + `manifest.json`，必要时加 `executor.py` + `executor.json`
2. 修改输出路径规则要兼容已有 run 目录
3. 涉及删除的脚本必须限定目录范围，禁止默认波及 `outputs/`
4. 修改 Web 参数名时同步更新 CLI 参数名，保持同构
5. 先读对应目录 `INFO.md` 再改代码

## 建议阅读顺序

1. `README.md` → 快速上手
2. 本文件（根 INFO.md）→ 全景理解
3. `web/INFO.md` → Web 控制台
4. `scripts/run/INFO.md` → 命令行入口
5. `GNNTP/data/INFO.md` + `GNNTP/common/INFO.md` → 数据与训练核心
6. `GNNTP/models/INFO.md` + 对应模型子目录 → 模型实现细节

