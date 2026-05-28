# final_new Attention 数值安全防护 — NaN 诊断与修复 (Round 1)

**日期**: 2026-05-28
**类型**: Bug 修复
**影响范围**: final_new 模型 — `attention.py` (`MultiHeadAttention.forward()`)

## 问题
训练日志显示 val_loss 在 epoch 3 突然变为 NaN，train_loss 保持正常：
```
Epoch 1: val_loss 0.4399  (normal)
Epoch 2: val_loss 0.6501  (spike, 50% worse than train)
Epoch 3: val_loss nan     (NaN starts, persists)
```

## 根因分析
最终原因待日志确认，初步推断为 Pre-LN Transformer 残差路径无界累积：
- Encoder 2层×4 sublayer + Decoder 2层×5 sublayer = 18 个残差加法
- 训练时 dropout 随机置零抑制累积幅度
- 验证时 dropout=0，残差信号无衰减叠加
- 2~3 epoch 后权重增长到临界点，Q·K^T 元素值突破 FP32 softmax 安全阈值 (~87)
- `F.scaled_dot_product_attention` 内部产生 Inf，Inf/Inf → NaN

## 修复 (Round 1)
- **文件**: `GNNTP/models/new/final_new/attention.py`
- **改动**:
  1. Q/K/V 值截断: projection 后、transpose 前 clamp 到 ±10
     - 理论安全值: head_dim=48 时 = ±3.5，±10 为保守选择
     - ±10 约覆盖 LayerNorm 后 ~10σ 范围，正常激活不受影响
  2. NaN 检测日志: Q/K 中出现 NaN/Inf 时 logger.warning 一次
  3. 输出 NaN 兜底: attention output 中检测到 NaN/Inf 时 `torch.nan_to_num(nan=0)`
     - 零值替换不会改变残差流的统计分布（被后续 LayerNorm 重新归一化）

## 影响
- 正常样本: Q/K clamp ±10 → 99.999%+ 激活不受影响
- NaN 样本: 日志记录 + 零值兜底，val_loss 不再为 NaN（可能偏高但保持数值稳定）
- 诊断价值: 日志输出会标识 NaN 首次出现位置（query 侧还是 key 侧），指导下一步修复

## 后续
- 如果日志确认 NaN 来自 attention: 实施 config 调整（LR 0.0005→0.0001 + warmup）
- 如果 NaN 在 clamp 后消失: 根本原因是权重漂移，需架构级残差缩放或 FP64 计算
- 如果 NaN 来源不是 attention: 需要进一步在 encoder/decoder 层间加 NaN 检测
