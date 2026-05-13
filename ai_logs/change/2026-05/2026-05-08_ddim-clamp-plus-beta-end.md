# DDIM 采样 clamp + beta_end 降低

**日期**: 2026-05-08
**类型**: 修复
**模型**: new_diffusion_fuzzy

## 问题

两个预测质量问题：
1. **峰值低估**：高点只有实际值的 70-80%，DDIM 多步去噪累积偏置将预测拉向均值
2. **缺失数据极端异常值**：实际=0 的节点预测出现 1000+，DDIM 无约束迭代导致数值发散

## 根因

### 峰值低估
- DDIM 50 步去噪中，每步 denoiser 在噪声中的 `x₀_pred` 偏保守，50 步累积偏置
- `beta_end=0.02` 过于激进，最大噪声步信号保留仅 ~13%
- 模型仅 3M 参数，缺乏对尾部分布（极端值）的建模能力

### 极端异常值
- 缺失数据（填充 0）没有 mask 机制，模型在该位置无学习信号
- DDIM 多步迭代无天然约束，OOD 输入 → denoiser 误判 → 指数放大

## 修改

| 文件 | 变更 |
|------|------|
| `config.json` | `beta_end: 0.02 → 0.01`（信号保留 13%→37%）；新增 `prediction_clamp_min: -3.0`, `prediction_clamp_max: 3.0` |
| `model.py:__init__` | 读取 clamp 配置参数，存为 `self.prediction_clamp_min/max` |
| `model.py:_sample_once` | 每次 DDIM/DDPM step 后 clamp `future_state.clamp(-3, 3)` |

## 原理

```
beta_end=0.02: ᾱ(t=200) ≈ 0.134 → DDIM 从 13% 信号起步
beta_end=0.01: ᾱ(t=200) ≈ 0.368 → DDIM 从 37% 信号起步（3 倍信号）

clamp(-3, 3) 在采样循环中:
  - 防止峰值被去噪偏置拉回均值
  - 缺失节点 OOD 发散在第 1 步就被截断，阻止指数放大
  - 训练数据 99.7% 在 ±3σ 内，clamp 不丢失有效信息
```

## 验证

- 重新训练后对比预测曲线，高点幅度应显著提升
- 缺失节点不应再出现 1000+ 异常值
- 指标预期：MAE 下降，R² 提升
