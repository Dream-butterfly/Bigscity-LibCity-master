### 更改 13

时间：2026-04-22 12:03:20 +08:00
来源类型：提问
来源说明：用户要求按“数据处理与模型训练解耦”流程开始落地，允许分步推进并阶段汇报。

更改类型-动作：新增功能/结构重构/实验流程变更/数据处理变更
更改类型-范围：跨模块
变更状态：已应用

需求/目标：
- 在保留旧命令可用的前提下，新增一套独立脚本链路，实现“数据处理与训练运行时强隔离”。
- 训练/续训/调参阶段仅消费数据工件，不允许再触发数据处理。
- 提供默认严格校验，并支持“强制复用（高风险）”开关。

变更文件：
- `GNNTP/data/artifact_io.py`
- `scripts/run/run_data_artifact.py`
- `scripts/run/run_train_artifact.py`
- `scripts/run/run_resume_artifact.py`
- `scripts/run/run_hyper_artifact.py`
- `README.md`
- `INFO.md`
- `scripts/run/INFO.md`
- `GNNTP/data/INFO.md`
- `ai_logs/change/2026-04/2026-04-22_1203_data-artifact-decoupling-phase1.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 新增 `GNNTP/data/artifact_io.py`，统一实现：
    - 数据工件目录约定（`cache/data_artifacts/`）
    - 工件命名（`da_YYYYMMDD_HHMMSS__<dataset>__<dataset_class>__<sig8>`）
    - 数据签名（`data_signature`）构建与比对
    - scaler 序列化/反序列化
    - 工件读写与 run 绑定元数据（`run_meta.json`）
  - 新增 `run_data_artifact.py`：完整数据处理并生成工件（含数组、特征、签名、元信息）。
  - 新增 `run_train_artifact.py`：仅读取工件训练评估，支持严格校验与 `--force_reuse`。
  - 新增 `run_resume_artifact.py`：基于 `run_id` + 工件续训，默认要求与 run 绑定一致。
  - 新增 `run_hyper_artifact.py`：基于固定工件执行超参数搜索。
  - 更新 `README.md` / `INFO.md` / `scripts/run/INFO.md` / `GNNTP/data/INFO.md`，补充新链路命令与职责说明。
- 变更原因：
  - 现有链路中 `data_version_id` 约束主要在 Web 层，核心 CLI 训练链路仍可能触发数据处理，无法满足“运行时强隔离”。
  - 通过“新脚本独立 + 旧脚本保留”可降低迁移风险并确保兼容。
- 影响范围：
  - 新增一组解耦脚本，不改变旧 `run_model.py/run_data_prep.py/run_hyper.py/run_resume.py` 逻辑。
  - 数据工件与 run 绑定元信息将作为新链路的核心约束机制。

后续迭代建议：
- 下一阶段将 Web 新增对 artifact 脚本的调用入口，并保留旧 API 并行。
- 可进一步将严格校验结果在 Web 侧以卡片标签形式展示（含“高风险强制复用”提示）。

修改注意事项：
- 当前环境缺少 `pwsh`，本轮无法执行 `uv run ...` 命令做自动化验证；后续需在可用 shell 环境补跑 lint/脚本级验证。
