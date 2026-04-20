### 更改 4

时间：2026-04-20 16:25:41 +08:00
来源类型：提问
来源说明：用户要求将默认模型改为 STGCN，清理失效文档引用并补充 resource_data/cache/outputs 的 INFO.md，同时记录本次对话。

更改类型-动作：修复问题 / 结构重构
更改类型-范围：跨模块
变更状态：已应用

需求/目标：
1. 将主运行入口默认模型由 GRU 改为 STGCN。  
2. 删除 `AI_EDIT_LOG.md`、`readme_zh.md` 的文档引用。  
3. 补充 `resource_data/INFO.md`、`cache/INFO.md`、`outputs/INFO.md`。  
4. 其余用户指定项（2/4/5）暂不更改。

变更文件：
- `scripts/run/run_model.py`
- `scripts/run/run_data_prep.py`
- `scripts/run/run_hyper.py`
- `scripts/run/run_resume.py`
- `README.md`
- `INFO.md`
- `resource_data/INFO.md`
- `cache/INFO.md`
- `outputs/INFO.md`

变更摘要：
- 变更内容：
  - 将四个 run 入口脚本中的默认 `--model` 从 `GRU` 统一改为 `STGCN`。
  - 从根文档中移除不存在文件的引用：`README.md` 删除 `AI_EDIT_LOG.md` 条目，`INFO.md` 删除 `readme_zh.md` 与 `AI_EDIT_LOG.md` 条目。
  - 新增 `resource_data/INFO.md`、`cache/INFO.md`、`outputs/INFO.md`，按项目约定补齐目录职责、输入输出、调用关系与修改注意事项。
- 变更原因：
  - 默认模型应与当前 manifest 注册体系保持一致，避免默认参数失配。
  - 清理文档失效引用并补齐目录注释，提升结构可读性与维护一致性。
- 影响范围：
  - 影响 CLI 默认行为（仅未显式传 `--model` 时生效）。
  - 影响根目录文档与目录说明完整性，不改变训练核心逻辑。

后续迭代建议：
- 若后续继续收敛入口契约，可再处理 Web resume `--train` 参数与 `run_resume.py` 的一致性问题。

修改注意事项：
- 本次按用户要求未改动独立实现目录归位与实验脚本冗余问题。
