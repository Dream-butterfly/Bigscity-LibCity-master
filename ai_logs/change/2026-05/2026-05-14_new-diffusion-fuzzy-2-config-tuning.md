# new_diffusion_fuzzy_2 配置文件调整：训练稳定性 + LR 调度修复

**日期**: 2026-05-14
**类型**: 配置调整
**模型**: new_diffusion_fuzzy_2

## 变更

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `lr_scheduler` | cosineannealinglr | reducelronplateau | CosineAnnealingLR 单调衰减导致后 30 epoch 无效 |
| `lr_T_max` | 10 | _(删除)_ | ReduceLROnPlateau 不需要 |
| `lr_eta_min` | 1e-5 | _(删除)_ | ReduceLROnPlateau 不需要 |
| `lr_patience` | _(缺失)_ | 5 | val_loss 连续 5 epoch 不降才减半 LR |
| `lr_threshold` | _(缺失)_ | 0.001 | 变化 < 0.001 视为停滞 |
| `learning_rate` | 0.001 | 0.0003 | 更保守的初始 LR，减少震荡 |
| `denoiser_layers` | 6 | 4 | 降低深度减少梯度传播问题 |
| `diffusion_steps` | 200 | 100 | 减半训练步数，每个 batch 覆盖率翻倍 |
| `num_sampling_steps` | 20 | 50 | DDIM 子采样比 2:1（原 10:1），减少累积误差 |
| `ddim_eta` | 0.0 | 0.3 | 加随机修正，补偿噪声预测误差 |

## LR 行为对比

```
旧 CosineAnnealingLR:       新 ReduceLROnPlateau:
epoch  0: lr=0.0010          epoch  0: lr=0.0003
epoch 10: lr=0.00089         epoch 10: lr=0.0003 (未停滞)
epoch 20: lr=0.00063         epoch 20: lr=0.0003 (未停滞)
epoch 30: lr=0.00032         epoch 30: lr=0.00015 (第1次减半)
epoch 40: lr=0.00009         epoch 40: lr=0.00015 (未停滞)
epoch 49: lr=0.00001 ← 废    epoch 49: lr=0.000075 (第2次减半)
```

## DDIM 采样对比

```
旧: 200训练步 → 20采样步 (10:1 子采样), eta=0 确定性
新: 100训练步 → 50采样步 (2:1 子采样),  eta=0.3 随机修正
```

每步 DDIM 去噪时，随机项 `eta * σ_t * z` 可补偿噪声预测误差，防止确定性路径累积偏置。
