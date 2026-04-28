### 更改 14

时间：2026-04-28 +08:00
来源类型：提问
来源说明：普罗米修斯计划 Phase 0 — 数据-训练完全解耦

更改类型-动作：新增功能/结构重构/数据处理变更
更改类型-范围：跨模块
变更状态：已应用

需求/目标：
- 数据预处理完全独立于模型，不再需要 --model 参数
- 数据工件按 dataset_class 格式保存，训练按 dataset + dataset_class 自动匹配
- 保留旧链路兼容性

变更文件：
- `GNNTP/config_parser.py` — model 参数改为可选，无模型时仅加载数据配置
- `GNNTP/data/artifact_io.py` — 新增 `list_artifact_metas()`, `find_latest_artifact()`
- `scripts/run/run_data_artifact.py` — 支持 --dataset_class，不传 --model 时纯数据集驱动
- `scripts/run/run_data_prep.py` — 重写为 run_data_artifact 的薄封装
- `scripts/run/run_train_artifact.py` — 新增 --artifact_list 和自动匹配（不传 --artifact_id 时）
- `scripts/experiments/train_new_diffusion_fuzzy.py` — 新建 fuzzy 实验入口

变更摘要：
- 变更内容：
  - `ConfigParser(model=None)` 现在合法：跳过模型 config 加载，仅要求 dataset_class 显式提供
  - `run_data_artifact.py` 新增 `--dataset_class` CLI 参数，不传 `--model` 时走纯数据模式
  - `run_train_artifact.py` 不传 `--artifact_id` 时自动扫描 `cache/data_artifacts/`，按 dataset + dataset_class 匹配最新签名一致的工件
  - `run_data_prep.py` 重写为上述流程的便捷入口
  - `train_new_diffusion_fuzzy.py` 使用上述自动匹配，简化实验命令
- 变更原因：
  - 原 data artifact 系统仍绑定 model config（需传 --model），不符合数据-训练解耦理念
  - 需要 artifact 按 dataset_class 类型自动匹配，而非手动指定 artifact_id
- 影响范围：
  - ConfigParser 旧调用方（所有位置参数传 model）完全兼容
  - 旧链路 run_model.py 不受影响
  - 新增 3 个函数于 artifact_io.py，不影响现有逻辑

后续迭代建议：
- 在 Web 控制台加入 artifact 列表展示
- 考虑自动清理过期 artifact
- run_hyper_artifact.py 和 run_resume_artifact.py 可同样支持自动匹配

修改注意事项：
- 无模型模式下 dataset_class 必须显式提供（否则 ConfigParser 报错）
- 自动匹配签名不通过时会报错提示重新构建工件或使用 --force_reuse
