# 变更 #14: DDP 多卡训练修复

**日期**: 2026-05-07
**类型**: 修复
**模型**: new_diffusion_fuzzy
**关联**: 变更 #13 (DDP 支持)、DDIM fix

## 问题

torchrun --nproc_per_node=4 多卡训练：单卡正常（MAE~41, R²~0.6），多卡预测纯随机噪声。

### 根因链

| # | 问题 | 文件 | 严重度 |
|---|------|------|--------|
| 1 | **DDP 梯度同步绕过**：`_train_epoch()` 调用 `self._unwrap_model().calculate_loss()` → 走裸模型 → DDP 的 all_reduce hook 从未触发 → 4 卡独立训练权重发散 | `traffic_state_executor.py:L436` | 🔴 |
| 2 | **LR 缩放**：`_init_device()` 默认 `scale_lr=True`，`lr *= world_size` → 0.001→0.004 → 收敛到 ε̂≈0 | `config_parser.py:L151` | 🔴 |
| 3 | **缺少 set_epoch()**：DistributedSampler 每轮同样 shuffle | `traffic_state_executor.py:L365` | 🟡 |

修复 3 已在上一轮完成。本次修复 1+2。

## 修改

| 文件 | 变更 |
|------|------|
| `new_diffusion_fuzzy/config.json` | +`"scale_lr": false` |
| `new_diffusion_fuzzy/model.py` | `forward()` 训练模式返回 `calculate_loss(batch)` |
| `traffic_state_executor.py:_train_epoch` | `loss_func=None` 时走 `self.model(batch)` 触发 DDP hook |
| `traffic_state_executor.py:_valid_epoch` | 同上（一致性） |

## 原理

```
旧: self._unwrap_model().calculate_loss(batch)
    → model.module 裸调用 → DDP hook 未触发
    → loss.backward() 只计算本地梯度
    → 4 卡权重独立更新 → 发散 → 随机噪声

新: self.model(batch)
    → DDP wrapper.forward() → hook 触发
    → forward() 内调用 calculate_loss() → 返回 loss
    → loss.backward() → DDP all_reduce 梯度
    → 4 卡同步更新 → 权重一致
```

## 验证

- `torchrun --nproc_per_node=4` 应显示 `learning_rate: 0.001`（非 0.004）
- train_loss 应从 >1.0 正常下降
- 多卡预测结果应与单卡一致
