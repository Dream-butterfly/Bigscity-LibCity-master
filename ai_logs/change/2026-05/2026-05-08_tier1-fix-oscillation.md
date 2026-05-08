# 变更 #16: new_diffusion_fuzzy 训练震荡修复

**日期**: 2026-05-08
**类型**: 修复
**模型**: new_diffusion_fuzzy

---

## 问题诊断

变更 #15 后训练效果极差（MAE ~38000 vs 旧 MAE ~44）。

### 根因分析

三层问题叠加：

1. **FiLM 零初始化砍掉了直接条件注入路径**
   - 旧: `concat(denoiser, condition_pooled) → Linear(2D,D)` — 条件从第 1 步参与
   - 新: `condition_film_proj[-1]` 全零初始化 → scale=shift=0 → FiLM 是 no-op
   - 条件仅剩 cross-attention 间接传递，信号大幅减弱

2. **cosine schedule + DDIM 误差爆炸**
   - cosine schedule 在 t≈199 时 ᾱ ≈ 4.5×10⁻⁵（信号仅 0.67%），linear 有 37%
   - 去噪器 val_loss=0.40 时，第一步 start prediction 误差被放大 ~200 倍
   - 25 步 DDIM 误差累积 → 最终 MAE 爆炸至 38000

3. **训练停滞/震荡**
   - val_loss 在 E4 后卡在 0.40-0.72 震荡，无持续下降
   - 去噪器从未达到 DDIM 采样所需的最低精度

### 日志证据

| Epoch | val_loss (新) | val_loss (旧) |
|-------|--------------|--------------|
| 0 | 0.711 | 0.258 |
| 4 | 0.497 | 0.096 |
| 9 | 0.506 | 0.083 |

新配置从 E4 后完全停滞，旧配置持续收敛。

---

## 修复内容

| 文件 | 变更 |
|------|------|
| `model.py:453-455` | 移除 `output_projection` 零初始化 (P5.12 回归) |
| `config.json` | `diffusion_schedule`: cosine → linear |
| `config.json` | `beta_end`: 0.01 → 0.02 |
| `config.json` | `use_spatiotemporal_attention`: false → true |
| `config.json` | `num_sampling_steps`: 25 → 50 |

### 修复原理

| 修复 | 原理 |
|------|------|
| 移除零初始化 | 恢复 Xavier init，predicted_noise 从合理值起步 |
| linear schedule | ᾱ₁₉₉≈0.13，DDIM 第一步信号强度 37%（vs cosine 0.67%） |
| spatiotemporal attention | 恢复去噪器容量，稳定训练 |
| 50 采样步 | 减小每步 DDIM 误差 |

### 注意

FiLM 的零初始化 (`condition_film_proj[-1]`) 暂保留——默认 init 会引入随机调制干扰，反而更不稳定。
下一步可考虑改为「线性条件注入 + FiLM」并行结构，或改用 Xavier init 并降低 LR。
