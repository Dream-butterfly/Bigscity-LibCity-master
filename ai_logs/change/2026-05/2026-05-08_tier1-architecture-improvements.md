# 变更 #15: new_diffusion_fuzzy 架构全面改进

**日期**: 2026-05-08
**类型**: 改进
**模型**: new_diffusion_fuzzy

---

## 已实施改进汇总

### 条件注入 (Tier 1 ① → Tier 2 ⑤ 升级)

| 阶段 | 实现 | 效果 |
|------|------|------|
| 原始 | `condition_features[:, -1:]` — 仅最后 1/12 步, concat+linear | 条件信号极弱 |
| 改进 ① | softmax 时序加权池化 — 全部 12 步 | 利用全部历史 |
| 改进 ⑤ | concat+linear → FiLM (scale+shift modulation) | 条件门控调制 |

### 损失函数 (Tier 1 ②)

**旧**: `F.mse_loss(predicted_noise, true_noise)` — 均匀加权
**新**: `(snr + 1).clamp(max=10) × mse → mean` — SNR+1 加权

### 推理优化 (Tier 1 ③ + 其他)

| 参数 | 旧值 | 新值 | 加速比 |
|------|------|------|--------|
| `num_prediction_samples` | 2 | 1 | 2× |
| `num_sampling_steps` | 50 | 25 | 2× |
| `use_spatiotemporal_attention` | true | false | ~10× |
| **合计** | | | **~20×** |

### 噪声调度

| 参数 | 旧值 | 新值 |
|------|------|------|
| `diffusion_schedule` | linear | cosine |
| `beta_end` | 0.02 | 0.01 |

Cosine 调度在低噪声区域步长更细，β_end 减半使最大噪声时信号从 37% 升到 ~60%。

### 模型容量

| 参数 | 旧值 | 新值 |
|------|------|------|
| `hidden_dim` | 96 | 128 |
| `denoiser_layers` | 3 | 4 |
| `ffn_hidden_dim` | 192 | 256 |
| `head_dim` | 24 | 32 (128/4) |

### 专属 Executor

`DiffusionTrafficStateExecutor` — 继承 TrafficStateExecutor，DDP-safe 训练。

---

## 修改文件清单

| 文件 | 变更 |
|------|------|
| `config.json` | 10 个参数更新 |
| `model.py:AttentionDenoiser.__init__` | +`input_window`, +`condition_temporal_weight`, +`condition_film_proj` |
| `model.py:AttentionDenoiser.forward` | concat+linear → FiLM modulation |
| `model.py:NewDiffusion.__init__` | 传入 `input_window` |
| `model.py:NewDiffusion.calculate_loss` | SNR+1 加权损失 |
| `model.py:NewDiffusion.forward` | training 时返回 loss (DDP-safe) |
| `model.py:NewDiffusion` | +`_ddp_loss_through_forward = True` |
| `executor.py` (新建) | `DiffusionTrafficStateExecutor` |
| `manifest.json` | executor → 新 executor |
| `traffic_state_executor.py` | 回退扩散专用代码 |

---

## 剩余未实施改进

| # | 改进 | 类型 |
|---|------|------|
| ⑥ | Encoder-Denoiser skip connections (U-Net 模式) | 架构重构 |

评估时间预计从 ~17min 降至 ~1-2min。
