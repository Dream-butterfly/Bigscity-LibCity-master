### 更改 7

时间：2026-04-21 00:20:00 +08:00
来源类型：提问
来源说明：用户要求对当前 `new_diffusion_fuzzy` 继续修订并补充 AI 日志。

更改类型-动作：修复问题/性能优化/结构重构
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 修复 `new_diffusion_fuzzy` 中影响可用性与质量的关键实现问题。
- 优化热路径中的冗余转换与可读性问题。
- 将本次修订按规范记录到 `ai_logs`。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`
- `ai_logs/change/2026-04/2026-04-21_0020_fuzzy-model-revision.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 将模糊参数校验绑定到 `use_fuzzy_graph` 开关，仅在启用模糊图时检查 `fuzzy_graph_num_sets` 与 `fuzzy_graph_sigma_init`。
  - 优化 `AdaptiveGraphLearner` 模糊关系构建中的 dtype/device 处理，移除每次前向的显式 `.to(device=..., dtype=...)`，改为 `type_as(similarity)`。
  - 修正 `_sample_once` 的采样步策略：使用从 `diffusion_steps-1` 到 `0` 的线性重映射时间步序列，避免 `num_sampling_steps < diffusion_steps` 时直接使用前段步号。
  - 简化守恒损失模糊权重表达：`0.5 * (1-high) + 1.5 * high` 简化为 `0.5 + high`，降低中间张量与阅读复杂度。
- 变更原因：
  - 解决“关闭模糊开关仍可能因模糊参数触发报错”的配置一致性问题。
  - 降低训练热路径中的冗余计算开销。
  - 提升采样步与训练扩散步的一致性，减少分布偏移风险。
- 影响范围：
  - 影响 `new_diffusion_fuzzy` 的初始化校验逻辑、采样流程与模糊损失计算细节。
  - 不影响 `new_diffusion_2` 原模型目录与行为。

后续迭代建议：
- 增加最小单元测试覆盖：
  1) `use_fuzzy_graph=False` 时允许弱约束模糊参数通过；
  2) `num_sampling_steps<diffusion_steps` 时采样时间步映射正确。
- 增加采样策略对比实验（原始倒序 vs 线性重映射）并记录指标变化。

修改注意事项：
- 本次为行为修正与轻量优化，未改变主干网络结构；建议在目标数据集上做一次短训和推理对齐验证。
