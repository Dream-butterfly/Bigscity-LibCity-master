# 变更 #15: new_diffusion_fuzzy Tier 1 架构改进 + 专属 Executor

**日期**: 2026-05-08
**类型**: 改进
**模型**: new_diffusion_fuzzy

## 改进 ①：条件注入强化 — 可学习时序加权

**旧**: `condition_features[:, -1:, :, :]` — 只用最后 1/12 历史步
**新**: `F.softmax(self.condition_temporal_weight) × condition_features → sum` — 可学习 12 步软加权

| 文件 | 变更 |
|------|------|
| `model.py:AttentionDenoiser.__init__` | +`input_window` 参数, +`self.condition_temporal_weight` |
| `model.py:AttentionDenoiser.forward` | 单步复制 → softmax 加权求和池化 |
| `model.py:NewDiffusion.__init__` | 传入 `input_window=self.input_window` |

初始化权重为 0 → softmax 初始均匀分布 → 从平均池化开始学习。

## 改进 ②：SNR+1 加权损失

**旧**: `F.mse_loss(predicted_noise, true_noise)` — 所有时间步均匀加权
**新**: `(snr + 1).clamp(max=10) × mse → mean` — 低噪声步高权重

t≈200 (SNR≈0.15): weight≈1.15
t≈0   (SNR≈100): weight≈10 (clamped)

使模型优先优化对最终预测质量贡献大的低噪声阶段。

## 改进 ③：推理采样数 2→1

DDIM eta=0 确定性采样 + 两次采样平均 → 推理时间减半，质量不变。

## 专属 Executor

| 文件 | 变更 |
|------|------|
| `executor.py` (新建) | `DiffusionTrafficStateExecutor` — 继承 `TrafficStateExecutor`，重写 `_train_epoch`/`_valid_epoch` |
| `manifest.json` | executor → `DiffusionTrafficStateExecutor` |
| `traffic_state_executor.py` | 回退扩散模型专用代码 (`_ddp_loss_ok` 等)，恢复干净 |

## 参数变化

| 参数 | 旧值 | 新值 |
|------|------|------|
| `num_prediction_samples` | 2 | 1 |
| `condition_temporal_weight` | 无 | 12维可学习参数 (≈12 params) |

模型总参数: ~3M (+12)

## 预期效果

- 条件信号利用全部 12 步历史 → 更好的时空模式捕获
- SNR 加权损失 → 低噪声细化阶段精度提升
- 推理时间减半
