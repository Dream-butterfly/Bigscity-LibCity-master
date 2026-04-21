### 更改 8

时间：2026-04-21 00:40:00 +08:00
来源类型：提问
来源说明：用户要求对 `new_diffusion_fuzzy` 的繁杂 `model.py` 做功能抽离，降低单文件耦合。

更改类型-动作：结构重构/性能优化
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 将 `new_diffusion_fuzzy/model.py` 中的无状态辅助逻辑抽出到 `new_diffusion_fuzzy/utils/`。
- 保持模型主流程与外部接口行为不变。
- 减少 `model.py` 的代码密度和重复，提升可维护性。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/__init__.py`
- `GNNTP/models/new/new_diffusion_fuzzy/INFO.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 将 `build_normalized_adjacency`、`expand_adjacency_batch`、`apply_temporal_attention`、`apply_node_temporal_cross_attention`、`apply_spatiotemporal_attention` 以及 `SinusoidalTimeEmbedding` 抽到 `new_diffusion_fuzzy/utils/__init__.py`。
  - `model.py` 改为从 `.utils` 导入这些共享工具，保持原有类定义和调用方式不变。
  - 更新 `INFO.md`，说明 `utils/` 已承载通用辅助逻辑。
- 变更原因：
  - 原 `model.py` 中存在多段纯函数和通用工具，职责较散，影响阅读和后续维护。
  - 抽离无状态工具后，主模型文件更聚焦于扩散、图学习与损失逻辑，符合“精准修改”和“简洁优先”。
- 影响范围：
  - 影响 `new_diffusion_fuzzy` 包内的导入组织方式。
  - 不改变模型算法、配置项或训练入口。

后续迭代建议：
- 若后续仍觉得 `model.py` 偏长，可继续第二阶段抽离 `MultiHeadAttention`、`GraphConvolution`、`FeedForwardNetwork` 到更细粒度模块。
- 如需要跨 `new_diffusion` / `new_diffusion_2` 复用，可再做公共基座抽象，但建议单独成阶段推进。

修改注意事项：
- 本次仅抽离无状态辅助逻辑，模型类仍保留在 `model.py`，以降低行为回归风险。
