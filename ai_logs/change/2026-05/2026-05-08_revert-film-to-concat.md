# 变更 #17: 回退 FiLM → concat+linear 条件注入

**日期**: 2026-05-08
**类型**: 修复 (回归)
**模型**: new_diffusion_fuzzy

---

## 问题

变更 #15 (FiLM) + #16 (config 修复) 后训练结果持续灾难性偏离 (MAE ~71000 vs 基准 ~19)。

变更 #16 的 config 修复（linear schedule、spatiotemporal attention、50 sampling steps、移除 output_projection 零初始化）完全无效，证明问题不在配置。

## 根因确认

FiLM 的零初始化 (`nn.init.zeros_(condition_film_proj[-1])`) 导致训练初期条件注入完全失效：

- 旧 concat+linear: `Linear(2D,D)(cat(denoiser, condition))` → 条件从 step 1 参与
- FiLM: `denoiser * (1+0) + 0` → 条件完全不参与

唯一剩余条件路径是 cross-attention，但 cross-attn 的 Q 来自去噪器特征——而去噪器特征本身基于无条件噪声输入推导，形成死循环。

后果：训练收敛到弱去噪器（val_loss 最低 0.52 vs 需要的 <0.1），DDIM 采样第一步就产生 OOD 输入，后续步骤预测完全失控，输出值飙至 Z-score 数千倍。

## 修复

| 文件 | 变更 |
|------|------|
| `model.py:AttentionDenoiser.__init__` | `condition_film_proj` → `condition_fusion = nn.Linear(hidden_dim*2, hidden_dim)` |
| `model.py:AttentionDenoiser.forward` | FiLM scale/shift 调制 → `torch.cat + Linear` |

保留: `condition_temporal_weight` (softmax 加权池化)，它本身不是问题。

## 验证标准

训练后 val_loss 应 < 0.1，评估 MAE 应 < 45（Z-score 空间），最终 inverse_transform 后 MAE ~19-22。
