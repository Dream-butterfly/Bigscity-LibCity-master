# 评估广播错误 + loss 原地修改修复

**日期**: 2026-05-14
**类型**: 修复
**影响文件**: `GNNTP/common/traffic_state_executor.py`, `GNNTP/models/loss.py`

## 问题 1: inverse_transform 广播导致评估指标失真

`traffic_state_executor.py:evaluate()` 中，`output_dim=1` 但 `feature_dim=3`（speed + tod + dow）。
`inverse_transform(output[..., :1])` 内部执行 `data * std[3]`，将 `[B,12,307,1]` 广播成 `[B,12,307,3]`，
channel 1/2 是速度值错误地用 tod/dow 统计量变换，evaluator 对所有通道求平均导致指标被污染。

### 修复
```diff
- y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
- y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
+ y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])[..., :self.output_dim]
+ y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])[..., :self.output_dim]
```

外层的 `[..., :output_dim]` 取回正确的 channel 0，丢弃被广播污染的 channel 1/2。

影响范围：所有 `output_dim < feature_dim` 且使用 StandardScaler 的模型（STGCN/DCRNN/STTN/new_diffusion 等）。

## 问题 2: loss 函数原地修改 labels

`masked_mae_torch`, `masked_mape_torch`, `masked_mse_torch`, `masked_rmse_torch` 中：
`labels[torch.abs(labels) < 1e-4] = 0` 原地修改调用者的 tensor，可能导致 MAPE 爆炸（20000%）。

### 修复
四个函数均在修改前添加 `labels = labels.clone()`，避免污染调用者数据。

## 训练建议（非代码）

对于当前 new_diffusion_fuzzy_2 的差效果，根本原因是严重欠训练：
- 噪声预测 MSE 0.73 仅略好于随机（1.0）
- 10 epochs 远不够，扩散模型需要 50-100+
- DDIM 20 步 / 训练 200 步 = 10:1 子采样过于激进，建议增加 `num_sampling_steps` 到 50-100
