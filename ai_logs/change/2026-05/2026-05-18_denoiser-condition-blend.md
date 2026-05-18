# denoiser 条件 blend 参数 — 防止无条件去噪捷径

**日期**: 2026-05-18
**类型**: 修复
**模型**: new_diffusion_fuzzy_2
**影响文件**: `GNNTP/models/new/new_diffusion_fuzzy_2/denoiser.py`

## 问题

诊断发现模型学到了"无条件去噪"捷径：
- 编码器输出有信息（node/time/batch-wise std≈0.89）
- 但去噪器 bypass 了条件，只用带噪输入 `Y_t` 预测噪声
- 噪声预测 MSE 0.60 靠无条件去噪就能达到，条件分支收不到有效梯度

## 修改

### `__init__`
添加 `self.condition_blend = nn.Parameter(torch.tensor(2.0))`

### `forward`
condition fusion 后添加 blend 逻辑：
```python
noisy_only = denoiser_input  # 保存无条件路径
fused = self.condition_fusion(torch.cat([denoiser_input, condition_pooled], dim=-1))
alpha = 0.5 + 0.5 * torch.sigmoid(self.condition_blend)  # 范围 [0.5, 1.0]
denoiser_input = (1.0 - alpha) * noisy_only + alpha * fused
```

- 初始化 sigmoid(2.0) ≈ 0.88，bias 强烈朝向条件路径
- alpha ∈ [0.5, 1.0]，确保条件至少贡献 50%
- 训练中 alpha 可自适应调整
