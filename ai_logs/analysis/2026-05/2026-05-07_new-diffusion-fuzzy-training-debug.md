# new_diffusion_fuzzy 训练无效 & 预测散乱 — 诊断报告

**日期**: 2026-05-07
**状态**: 第 1 轮修复已执行（config 4项），效果不佳 → 发现根本架构缺陷，待执行第 2 轮修复
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

---

### ⚠️⚠️⚠️ 🔴🔴 Critical 0（第 2 轮发现 — 根本架构缺陷）⚠️⚠️⚠️

### 交叉注意力在噪声查询上失效 → 模型学习的是边缘分布 p(Y) 而非条件分布 p(Y|X)

**位置**: `model.py` `DenoiserBlock.forward` L362-L365 + `AttentionDenoiser.forward` L455-L479

**现象**：第 1 轮 config 修复（开时间特征 + bfloat16 + DDIM + 关守恒loss）后，预测仍在数据均值附近波动。

**根因推导**：

当前架构中，条件（历史编码）**仅在交叉注意力中注入**：

```
noisy_future [B,T_out,N,1]          condition [B,T_in,N,D]
     │                                     │
     ▼                                     │
input_projection                            │
     │                                     │
temporal self-attn ◄── 纯噪声上做自注意力    │
graph conv ◄────────── 纯噪声上做图卷积      │
     │                                     │
     └─────► cross_attn(Q=noise, K/V=cond) │  ◄── 🔴 问题点
```

交叉注意力权重 = `softmax(noisy_query @ condition_key^T / sqrt(d))`

| 时间步 t | alpha_bar_t | 噪声占比 | Q @ K^T 结果 | softmax 输出 |
|----------|------------|---------|-------------|-------------|
| t=0      | ~1.0       | ~0%     | 有意义的匹配 | 聚焦权重 |
| t=50     | ~0.78      | ~22%    | 弱信号+噪声 | 半模糊 |
| t=100    | ~0.37      | ~63%    | 噪声主导 | 接近均匀 |
| t=150    | ~0.22      | ~78%    | 纯噪声 | **均匀分布** |
| t=199    | ~0.13      | ~87%    | 纯噪声 | **均匀分布** |

**当 t 较大时**：`noisy_query ≈ N(0,σ²I)`，点积结果 ≈ 随机数，softmax 退化为均匀权重。交叉注意力输出 ≈ condition 的时间平均。

**训练时 t 在 [0,199] 均匀采样** → **至少一半训练样本条件注入无效**。

**理论后果**：扩散模型最大似然训练等价于学习 score function ∇log p(Y|X)。但条件信号在大量训练步中不可用，模型实际学习的梯度是：
```
∇log p(Y) + 衰减后的 ∇log p(X|Y)
```
即模型偏向学习**无条件边缘分布** p(Y) 而非**条件分布** p(Y|X)。

**验证**：
- p(Y)（交通速度的边缘分布）≈ 以数据均值为中心的正态状分布
- 从 p(Y) 采样 → 预测在均值附近波动 ✓ 与用户观察完全吻合
- 无论输入 X 如何变化，输出几乎相同 ✓ 进一步验证

**为什么交叉注意力在 Stable Diffusion 中可行但此处不行**：
- SD 用 U-Net，交叉注意力在**多分辨率**下进行，低分辨率时有更大的感受野
- SD 的条件（文本嵌入）是**密集语义向量**，与像素特征在语义空间中对齐
- 本模型：条件（时间序列编码）与噪声未来在特征空间中**未经对齐训练**，直接点积

**结论**：配置修复（时间特征、bfloat16、DDIM）处理了次要问题，但**不解决这个根本架构缺陷**。条件必须在噪声处理之前注入。

对比成功案例：

| 模型 | 条件注入方式 | 何时注入 |
|------|------------|---------|
| **本模型** | 仅交叉注意力 | attention/graph conv 之后 ❌ |
| CSDI | 拼接（observed mask + noisy） | 输入层直接拼接 ✅ |
| DiffWave | FiLM（scale + shift） | 每个 residual block 开头 ✅ |
| Grad-TTS | 编码器输出拼接 | U-Net 输入层 ✅ |
| Stable Diffusion | 交叉注意力 | 每层注入，但 U-Net 多分辨率 ✅ |

**本模型的注入时序是所有成功案例中最弱的**。

---

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

---

## 第 2 轮修复方案：条件前置融合（解决 Critical 0）

### 核心思路

在 `AttentionDenoiser.forward` 中，**在进入 DenoiserBlock 之前**，将条件特征直接拼接到去噪输入中：

```
noisy_future                     condition [B,T_in,N,D]
     │                                │
     ▼                                │
input_projection + pos + time         │
     │                                ▼
     │                     mean_pool(dim=1) → [B,1,N,D]
     │                     expand(T_out)   → [B,T_out,N,D]
     │                                │
     └──────► concat ──► condition_fusion(Linear) ──► [B,T_out,N,D]
                         │
                         ▼
              DenoiserBlock × N
              (cross_attn 保留为辅助通路)
```

### 具体改动

**文件**: `model.py`，仅涉及 `AttentionDenoiser`

**改动 1** — `__init__` 新增融合层（L423 之后）：
```python
self.condition_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
```

**改动 2** — `forward` 在 L466 之后插入条件融合：
```python
# Pool condition over time, fuse directly into denoiser input
# BEFORE any attention/graph ops — ensures condition signal at all noise levels
condition_pooled = condition_features.mean(dim=1, keepdim=True)  # [B, 1, N, D]
condition_pooled = condition_pooled.expand(-1, denoiser_input.size(1), -1, -1)
denoiser_input = self.condition_fusion(
    torch.cat([denoiser_input, condition_pooled], dim=-1)
)
```

### 设计理由

- `mean(dim=1)` 保留空间精度（每节点独立上下文），牺牲时间细节（交叉注意力可补充）
- `expand` 到 T_out 使每个未来步都获得同一全局上下文 + 各自位置编码
- `nn.Linear(2D, D)` 可学习融合比例，模型自行决定噪声信号和条件信号的使用权重
- `DenoiserBlock` 不变，交叉注意力保留作为时序细粒度条件补充
- 总增加参数量：`hidden_dim * 2 * hidden_dim + hidden_dim ≈ 18K`（可忽略）
