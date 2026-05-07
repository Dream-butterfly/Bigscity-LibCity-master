# new_diffusion_fuzzy 训练无效 & 预测散乱 — 诊断报告

**日期**: 2026-05-07
**状态**: 分析完成，待执行修复
**涉及文件**:
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`（871行）
- `GNNTP/models/new/new_diffusion_fuzzy/config.json`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/attention_ops.py`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/adjacency.py`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/time_embedding.py`
- `GNNTP/common/traffic_state_executor.py`
- `GNNTP/data/dataset/mixins/pipeline_mixin.py`

---

## 现象确认

| 现象 | 状态 |
|---|---|
| 训练 loss 基本不变 | ✅ 确认 |
| 预测呈正态分布，围绕一个值波动 | ✅ 确认 |
| 模型未拟合数据 | ✅ 确认 |
| 去噪器预测噪声 ≈ 0（loss ≈ 1.0） | 高度可能 |

---

## 根因分析

### 🔴 Critical 1：DDPM 采样使用非连续时间步 — 数学错误

**位置**: `model.py` L837-L847 `_get_sampling_schedule` + L555-L569 `ddpm_step`

```python
# L841-L846：从 199 → 0 均匀取 50 个点
torch.linspace(self.diffusion_steps - 1, 0, self.num_sampling_steps, ...)
# 结果：[199, 195, 191, ..., 4, 0]，每步跳约 4 个时间步
```

**问题**：DDPM 反向公式 `p(Y_{t-1}|Y_t)` 的均值推导**假设连续步长**：

```python
# L564：仅用 alpha_t（单步），未用 alpha_bar_{t-4} / alpha_bar_t
model_mean = sqrt_recip_alpha_t * (current_state
    - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
```

**正确公式**（步长 Δ > 1）：
```
Y_s = sqrt(alpha_bar_s / alpha_bar_t) * Y_t
      + sqrt(1 - alpha_bar_s / alpha_bar_t) * epsilon_theta(Y_t, t)
```
而代码中 `1/sqrt(alpha_t)` ≠ `1/sqrt(alpha_bar_{t-4} / alpha_bar_t)`。

**影响**：即使模型训练良好，采样也会严重偏离。对于 50 步采样，累积误差可超过 30%，导致预测近似纯噪声。

---

### 🔴 Critical 2：配置缺少时间特征 → 条件信号极弱

**config.json**：
```json
"add_time_in_day": false,
"add_day_in_week": false,
```

**后果**：
- `feature_dim = 1`（仅速度标量，无时间上下文）
- 编码器 `nn.Linear(1, 96)` 将标量线性放大为 96 个冗余特征
- 交叉注意力（DenoiserBlock.cross_attention）中：
  - Query = 带噪未来（高 t 时 ≈ 纯噪声）
  - Key/Value = 历史编码（仅速度信息，无周期性模式）
  - 模型无法区分凌晨 2 点和早高峰 8 点，去噪任务不可能完成
- 对比：STGCN/STTN 等工作正常是因为直接回归未来值（有历史速度就够），无需区分噪声和信号

**影响**：条件信息不足以指导去噪，模型退化为预测零噪声（loss ≈ 1.0 停滞）。

---

### 🟡 Critical 3：AMP float16 + GradScaler 边界溢出

**config.json**：
```json
"use_amp": true,
"amp_dtype": "float16",
```

**机制**：
- 初始 diffusion loss ≈ 1.0（去噪器预测 ≈ 0 vs N(0,1) 真实噪声）
- GradScaler 初始 scale = 65536（2¹⁶）
- `scaled_loss = 1.0 × 65536 = 65536` — **刚好逼近 float16 max（65504）**
- 溢出 → scale 减半（GradScaler 自适应机制）
- 若干步后 scale 恢复增长 → 再次溢出 → 振荡
- 每次溢出该步的梯度被丢弃（GradScaler 跳过 `step()`）
- 每约 2000 步触发一次溢出周期，有效训练步数大幅减少

**影响**：梯度震荡导致模型无法稳定学习。

---

### 🟡 Critical 4：守恒损失反向干扰扩散目标

**位置**: `model.py` L774-L776 `calculate_loss` + L547-L553 `predict_start_from_noise`

```python
predicted_future = self.diffusion_scheduler.predict_start_from_noise(
    noisy_future, timesteps, predicted_noise
)
```

`predict_start_from_noise` 中：
```python
return (noisy_future - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
```

**问题**：
- 当 t 较大时，`sqrt_alpha_bar` ≈ 0.366
- 除以此小值**放大了噪声预测误差 ~2.7×**
- 早期训练噪声预测不准 → `predicted_future` 被误差放大后的伪影主导
- 守恒损失惩罚的是伪影而非真实守恒违规
- 产生与扩散目标（MSE 噪声）冲突的梯度

**影响**：即使 warmup 权重 0.01，放大后的错误信号仍干扰早期学习。

---

## 修复方案（按优先级排序）

### 方案 1：开启时间特征（最根本）

```json
"add_time_in_day": true,
"add_day_in_week": true,
```

`feature_dim` 从 1 → 3，编码器获得 meaningful 输入，条件信号显著增强。

### 方案 2：关闭 AMP 或改用 bfloat16

```json
"use_amp": false
```
或
```json
"amp_dtype": "bfloat16"
```

bfloat16 动态范围与 float32 相同（指数位宽相同），不会溢出。

### 方案 3：修复采样方式（二选一）

**选项 A**：改用 DDIM（推荐，天然支持非连续步长）
```json
"sampling_method": "ddim",
"ddim_eta": 0.0,
```

**选项 B**：全步采样
```json
"num_sampling_steps": 200
```
计算量增大 4 倍，但数学正确。

### 方案 4：关闭守恒损失（初始调试用）

```json
"physics_loss_weight": 0.0
```

等扩散 loss 稳定下降后再逐步开启。

---

## 建议最小配置变更

```diff
- "add_time_in_day": false,
- "add_day_in_week": false,
+ "add_time_in_day": true,
+ "add_day_in_week": true,

- "use_amp": true,
- "amp_dtype": "float16",
+ "use_amp": true,
+ "amp_dtype": "bfloat16",

- "sampling_method": "ddpm",
+ "sampling_method": "ddim",

- "physics_loss_weight": 0.05,
+ "physics_loss_weight": 0.0,
```

预期效果：训练 loss 从 ~1.0 开始下降，预测不再呈无意义正态分布。
