# Phase C 扩散集成 — 知识总汇

> 创建: 2026-05-21 | 基于 new_diffusion_fuzzy v1/v2 的全部经验教训
> 目标: 将 DDIM 条件扩散集成到 new_fuzzy_2，替换确定性 FutureDecoder

---

## 一、架构总览

```
                      FuzDiff (Phase C)
                      
  Train:                                                          
    History X ──► STEncoder ──► H ──┐                             
                                      ├──► FuzzyGuidedDenoiser ──► ε_θ
    Future Y_0 ──► +ε ──► Y_t ───────┘       ↑        ↑          
    Timestep t ──────────────────────────────┘        │          
    FuzzyRelationalGraph ──► R ───────────────────────┘          
                                                                  
    Loss = SNR-weighted MSE(ε_θ, ε) + λ * FuzzyConservation(Y_0_pred)
                                                                  
  Inference:                                                      
    History X ──► STEncoder ──► H                                 
    FuzzyRelationalGraph ──► R                                    
    Y_T ~ N(0,I)                                                  
    for t in [T-1, ..., 0]:                                       
      ε_θ = FuzzyGuidedDenoiser(Y_t, t, H, R)                    
      Y_{t-1} = DDIM_step(Y_t, ε_θ, t)                           
      Y_{t-1} = clamp(Y_{t-1}, -3, 3)                            
    return Y_0                                                    
```

---

## 二、扩散数学速查

### 噪声调度（只用 Linear，不用 Cosine）

```python
T = 50  # 训练步数
beta_start = 1e-4
beta_end   = 0.01    # ← 0.01 不是 0.02。cosine 已证明危险

betas = linspace(beta_start, beta_end, T)
alphas = 1 - betas
alpha_bars = cumprod(alphas)

# 关键数值（T=50, beta_end=0.01）:
#   ᾱ_0   = 0.9999  (0.01% 噪声)
#   ᾱ_49  ≈ 0.78    (22% 噪声，信号保留78%)
#   对比 beta_end=0.02: ᾱ_49 ≈ 0.61 (信号仅61%)
```

### 前向扩散

```
Y_t = √ᾱ_t · Y_0 + √(1-ᾱ_t) · ε    ε ~ N(0, I)
```

### 单步重建（用于守恒损失）

```
Ŷ_0 = (Y_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
```

⚠ 当 ᾱ_t 小时（大 t），除法放大误差。只在 `t < 0.7T` 时计算守恒损失。

### DDIM 反向采样

```python
# t_next 可以不等于 t-1（支持 strided schedule）
alpha_bar_t      = ᾱ[t]
alpha_bar_next   = ᾱ[t_next]  # 如果 t_next < 0: alpha_bar_next = 1.0

# 预测的干净信号
Y_0_pred = (Y_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t

# 指向 next 的方向
direction = √(1-ᾱ_next) · ε_θ

# DDIM 确定性步（eta=0）
Y_next = √ᾱ_next · Y_0_pred + direction
```

### SNR 加权 MSE

```python
snr = ᾱ_t / (1 - ᾱ_t)
weight = clamp(snr + 1.0, max=10.0)
loss = weight * MSE(ε_θ, ε)
```

---

## 三、过去 13 个坑（按严重程度）

| # | 坑 | 症状 | 正确做法 |
|---|-----|------|---------|
| 1 | **FiLM zero-init** | MAE ~71000 | 用 concat+Linear 注入条件 |
| 2 | **Cosine schedule** | DDIM 发散，MAE ~38000 | 用 Linear schedule，beta_end ≤ 0.01 |
| 3 | **无条件捷径** | ε_θ MSE 0.60 靠无去噪就能达到 | condition_blend gate，bias 向条件路径 |
| 4 | **仅靠 cross-attention 注入条件** | 高 t 时 attention 均匀分布 | concat condition 到输入，在 attention 之前 |
| 5 | **DDPM 用非连续时间步** | 数学错误，结果无效 | 推理必须用 DDIM（支持任意步长） |
| 6 | **AMP float16 溢出** | scale=65536 接近 fp16 max | 用 bfloat16 |
| 7 | **守恒损失在高 t** | √(1-ᾱ)/√ᾱ 放大噪声误差 | 仅在 t < 0.7T 时计算 |
| 8 | **缺少时间特征** | feature_dim=1，编码器退化 | add_time_in_day + add_day_in_week = True |
| 9 | **DDIM 无 clamp** | 缺失节点值发散到 1000+ | 每步后 clamp(-3, 3) |
| 10 | **output_projection 零初始化** | 输出全零，无法学习 | 不要改默认 Xavier init |
| 11 | **预训练编码器被 untrain** | encoder lr 太大 | 对 encoder 用更小的 lr (0.1x) 或 freeze 前几个 epoch |
| 12 | **eval 时 inverse_transform broadcast 错误** | 指标被污染 | `y_pred[..., :output_dim]` 后再 inverse_transform |
| 13 | **loss 函数原地修改 labels** | 评估指标计算错误 | `labels = labels.clone()` 先 clone |

---

## 四、条件注入的最终方案

经过 FiLM（失败）→ concat（成功）→ blend_gate（加固）的迭代，最终方案：

```python
# 1. 时间池化条件（加权平均编码器输出各时间步）
temporal_weight = softmax(condition_temporal_weight)  # [T_in]
condition_pooled = (H * temporal_weight).sum(dim=1, keepdim=True)  # [B, 1, N, D]
condition_pooled = condition_pooled.expand(-1, T_out, -1, -1)     # [B, T_out, N, D]

# 2. 与噪声输入 concat
fused = Linear(2D → D)(cat([Y_t_embedded, condition_pooled], dim=-1))

# 3. Blend gate 防止无条件捷径
noisy_only = Y_t_embedded  # 保留无条件路径
alpha = 0.5 + 0.5 * sigmoid(condition_blend)
denoiser_input = (1 - alpha) * noisy_only + alpha * fused
```

**每层额外注入**（防止深层信号稀释）：
```python
# 在每个 DenoiserBlock 中：
scale, shift = time_scale_shift(timestep_embedding).chunk(2, dim=-1)
x = x * (1.0 + scale[:,None,None,:]) + shift[:,None,None,:]
```

---

## 五、文件清单与改动

### 新建

| 文件 | 来源 | 用途 |
|------|------|------|
| `new_fuzzy_2/diffusion.py` | 从 `new_diffusion_fuzzy_2/diffusion.py` 复制 | `DiffusionScheduler`（不改） |
| `new_fuzzy_2/denoiser.py` | 从 `new_diffusion_fuzzy_2/denoiser.py` 改编 | `FuzzyGuidedDenoiser` |
| `new_fuzzy_2/executor.py` | 从 `new_diffusion_fuzzy_2/executor.py` 复制 | `DiffusionTrafficStateExecutor`（不改） |

### 修改

| 文件 | 改动 |
|------|------|
| `new_fuzzy_2/model.py` | 集成 DiffusionScheduler + Denoiser；替换 FutureDecoder；`calculate_loss` 重写；`predict` 重写为 DDIM 采样 |
| `new_fuzzy_2/config.json` | 加 diffusion 参数 |
| `new_fuzzy_2/manifest.json` | executor → `DiffusionTrafficStateExecutor` |
| `new_fuzzy_2/executor.json` | 适配扩散训练 |

### 不改

| 文件 | 原因 |
|------|------|
| `new_fuzzy_2/encoder.py` | 完美复用 |
| `new_fuzzy_2/graph.py` | 完美复用（FuzzyRelationalGraphLearner） |
| `new_fuzzy_2/attention.py` | 完美复用 |
| `new_fuzzy_2/utils/` | 完美复用 |

---

## 六、Denoiser 设计详案

### FuzzyGuidedDenoiser

```python
class FuzzyGuidedDenoiser(nn.Module):
    """
    输入:
      Y_t  [B, T_out, N, C]  噪声未来
      t    [B]               扩散时间步
      H    [B, T_in, N, D]   编码历史条件
      R    [N, N]            模糊关系图

    输出:
      ε_θ  [B, T_out, N, C]  预测噪声

    架构（复用已证明的模式）:
      1. Y_t → Linear(C→D) → Y_t_embedded
      2. Y_t_embedded + temporal_position_embedding
      3. H → temporal_pool → condition_pooled → expand
      4. cat(Y_t_embedded, condition_pooled) → Linear(2D→D)
      5. Blend gate (防止无条件捷径)
      6. N × FuzzyDenoiserBlock:
           - timestep re-injection (scale/shift)
           - temporal self-attention
           - FuzzyGraphConvolution(R)  ← 用模糊图替代普通GCN
           - cross-attention over H
           - [spatiotemporal attention]
           - FFN
      7. U-Net skip connections（前半存，后半加）
      8. LayerNorm → Linear(D→C) → ε_θ
    """
```

### 与 old denoiser 的关键差异

| 维度 | old (new_diffusion_fuzzy_2) | new (FuzzyGuidedDenoiser) |
|------|--------------------------|-------------------------|
| 图卷积 | `GraphConvolution(Â)` | **`FuzzyGraphConvolution(R)`** |
| 图学习 | `AdaptiveGraphLearner`（每层调用） | **预计算的 R**（model 层传入） |
| 空间先验 | softmax/Gaussian kernel | **max-min fuzzy relation** |
| 理论支撑 | 无 | 模糊关系代数 |

---

## 七、Model 改动详案

### `NewFuzzy2.__init__` 新增

```python
# 扩散参数
self.diffusion_steps = config.get("diffusion_steps", 50)
self.num_sampling_steps = config.get("num_sampling_steps", 20)
self.ddim_eta = config.get("ddim_eta", 0.0)
self.prediction_clamp_min = config.get("prediction_clamp_min", -3.0)
self.prediction_clamp_max = config.get("prediction_clamp_max", 3.0)
self.denoiser_layers = config.get("denoiser_layers", 2)

# 扩散调度器
self.diffusion = DiffusionScheduler(
    num_steps=self.diffusion_steps,
    schedule=config.get("diffusion_schedule", "linear"),
    beta_start=config.get("beta_start", 1e-4),
    beta_end=config.get("beta_end", 0.01),
)

# 去噪器（替代 FutureDecoder）
self.denoiser = FuzzyGuidedDenoiser(
    output_dim=self.output_dim,
    hidden_dim=self.hidden_dim,
    num_heads=self.num_heads,
    num_layers=self.denoiser_layers,
    ffn_hidden_dim=self.ffn_hidden_dim,
    graph_k_hop=self.graph_k_hop,
    dropout=self.dropout,
    max_future_steps=self.output_window,
    use_spatiotemporal_attention=self.use_spatiotemporal_attention,
    use_gradient_checkpointing=self.use_gradient_checkpointing,
)

# DDP 兼容
self._ddp_loss_through_forward = True
```

### `NewFuzzy2.calculate_loss` 重写

```python
def calculate_loss(self, batch):
    history = batch["X"]
    future  = batch["y"][..., :self.output_dim]  # Y_0

    # 1. 编码条件
    H = self.encode_condition(history)  # [B, T_in, N, D]
    R = self.fuzzy_graph(history) if self.use_fuzzy_graph else self.adjacency_matrix

    # 2. 扩散训练
    t = torch.randint(0, self.diffusion_steps, (future.size(0),), device=future.device)
    Y_t, epsilon = self.diffusion.add_noise(future, t)
    epsilon_theta = self.denoiser(Y_t, t, H, R)

    # 3. SNR 加权损失
    alpha_bar_t = self.diffusion.alphas_cumprod[t]
    snr = alpha_bar_t / (1.0 - alpha_bar_t).clamp_min(1e-8)
    loss_weight = (snr + 1.0).clamp(max=10.0)
    diffusion_loss = (loss_weight * F.mse_loss(epsilon_theta, epsilon, reduction='none')).mean()

    # 4. 模糊守恒（仅在低噪声步）
    effective_weight = self._get_effective_conservation_weight()
    if effective_weight > 0:
        mask = t < int(0.7 * self.diffusion_steps)  # 仅低噪声
        if mask.any():
            Y_0_pred = self.diffusion.predict_start_from_noise(Y_t[mask], t[mask], epsilon_theta[mask])
            conservation_loss = self._fuzzy_conservation_loss(Y_0_pred, R)
            return diffusion_loss + effective_weight * conservation_loss

    return diffusion_loss
```

### `NewFuzzy2.predict` 重写

```python
def predict(self, batch):
    history = batch["X"]
    B = history.size(0)
    H = self.encode_condition(history)
    R = self.fuzzy_graph(history) if self.use_fuzzy_graph else self.adjacency_matrix

    # DDIM 采样
    Y_t = torch.randn(B, self.output_window, self.num_nodes, self.output_dim,
                      device=history.device, dtype=history.dtype)

    timesteps = torch.linspace(self.diffusion_steps - 1, 0, self.num_sampling_steps,
                               dtype=torch.long, device=history.device)

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = t_curr.expand(B)
        epsilon_theta = self.denoiser(Y_t, t_batch, H, R)
        Y_t = self.diffusion.ddim_step(Y_t, t_curr, t_next, epsilon_theta, eta=self.ddim_eta)
        Y_t = Y_t.clamp(self.prediction_clamp_min, self.prediction_clamp_max)

    return Y_t
```

---

## 八、Config 新增参数

```json
{
  "diffusion_steps": 50,
  "diffusion_schedule": "linear",
  "beta_start": 0.0001,
  "beta_end": 0.01,
  "num_sampling_steps": 20,
  "ddim_eta": 0.0,
  "denoiser_layers": 2,
  "prediction_clamp_min": -3.0,
  "prediction_clamp_max": 3.0,
  "amp_dtype": "bfloat16"
}
```

---

## 九、实施检查清单

- [ ] 复制 `diffusion.py`（不改）
- [ ] 编写 `denoiser.py`（`FuzzyGuidedDenoiser` + `FuzzyDenoiserBlock`）
  - [ ] concat+Linear 条件注入 + blend gate
  - [ ] 每层 timestep re-injection（scale/shift）
  - [ ] `FuzzyGraphConvolution` 替代 `GraphConvolution`
  - [ ] 预计算的 R 替代 per-block AdaptiveGraphLearner 调用
  - [ ] U-Net skip connections
  - [ ] 不零初始化任何 projection
- [ ] 修改 `model.py`
  - [ ] `__init__` 加 diffusion scheduler + denoiser
  - [ ] `calculate_loss` 改为扩散训练
  - [ ] `predict` 改为 DDIM 采样
  - [ ] `_ddp_loss_through_forward = True`
  - [ ] 移除 `FutureDecoder`（或保留做消融对比）
- [ ] 复制 `executor.py`（DiffusionTrafficStateExecutor，不改）
- [ ] 更新 `config.json`、`manifest.json`、`executor.json`
- [ ] 更新 `embedding.py`（如需要 SinusoidalTimeEmbedding）
- [ ] 导入测试：`from GNNTP.models.new.new_fuzzy_2.model import NewFuzzy2`
- [ ] 冒烟测试：小 batch 一次 forward + 一次 predict
- [ ] 小数据快速训练测试（1 epoch，确认 loss 下降）
