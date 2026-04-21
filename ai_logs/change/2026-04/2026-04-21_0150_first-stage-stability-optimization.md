### 更改 10

时间：2026-04-21 01:50:00 +08:00
来源类型：提问
来源说明：用户要求按“最小风险执行顺序”先落地第一次优化，并同步记录日志。

更改类型-动作：性能优化/实验流程变更
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 在不改变 `new_diffusion_fuzzy` 核心结构前提下，优先做第一阶段稳训优化。
- 降低早期训练失效风险（发散/震荡），并保持改动可回滚。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`
- `GNNTP/models/new/new_diffusion_fuzzy/config.json`
- `GNNTP/models/new/new_diffusion_fuzzy/INFO.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 在 `NewDiffusion` 中新增物理损失预热机制：
    - 新增配置读取：`physics_warmup_steps`、`physics_warmup_start_ratio`、`physics_warmup_mode`。
    - 新增 `_get_effective_physics_weight()`，按 `linear/cosine` 方式将物理损失权重从起始比例渐进到目标值。
    - `calculate_loss` 中改为使用 `effective_physics_weight`，并增加预热起止日志。
  - 调整默认稳训配置（`config.json`）：
    - `learning_rate`：`0.001 -> 0.0003`
    - `max_grad_norm`：`5 -> 3`
    - 新增 `physics_warmup_*` 默认参数（3000 steps、起始比例 0.2、线性预热）。
  - 在 `INFO.md` 增补“已加入物理损失预热机制”说明。
- 变更原因：
  - 第一阶段目标是“先稳训练，再谈更强改动”；避免物理项在训练早期压制扩散主目标。
  - 通过更保守学习率与更紧梯度裁剪，降低数值不稳定与损失震荡风险。
- 影响范围：
  - 仅影响 `new_diffusion_fuzzy` 训练损失权重调度与默认训练超参。
  - 不改变模型核心网络结构、输入输出接口和注册入口。

后续迭代建议：
- 继续第二阶段（同一执行顺序）：优先做推理降耗（`num_sampling_steps` / `num_prediction_samples` 网格）。
- 增加最小监控：记录每轮有效物理权重、梯度范数分布与 NaN/Inf 统计。

修改注意事项：
- 预热以“训练步数”而非 epoch 驱动，跨不同 batch size 时需重新评估 `physics_warmup_steps`。
- 当前为低风险默认值，建议在目标数据集上做短训对照后再进一步调大物理约束。
