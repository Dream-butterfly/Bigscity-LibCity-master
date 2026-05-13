# new_diffusion_fuzzy 模型完整规格说明书

> **目标读者**: 另一个 Agent / 开发者，需要从零重写此模型。
> **独立性**: 本文档不依赖原始代码上下文，所有定义、接口、数据流都显式写明。
> **日期**: 2026-05-08

---

## 1. 模型目标与论文定位

### 1.1 要解决的问题
交通时空预测 (Traffic Spatio-Temporal Prediction)：给定过去 12 个时间步 (input_window=12) 的交通数据，预测未来 12 个时间步 (output_window=12)。

### 1.2 论文动机
这是目标期刊 Information Sciences 的论文模型。其核心 novel contribution 是三个方面的融合：

1. **模糊图学习 (Fuzzy Graph Learning)**: 不使用固定的距离邻接矩阵，而是学习一个自适应图，通过模糊隶属度函数 (Gaussian membership over similarity) 建模节点间"软"连接关系。
2. **模糊守恒损失 (Fuzzy Conservation Loss)**: 基于交通流的图守恒先验——节点状态的时间变化 ≈ 流入量 - 流出量。用模糊隶属度加权高拥堵节点。
3. **条件扩散 (Conditional Diffusion)**: 用扩散模型替代确定性回归，通过条件编码器 + 去噪器进行多步预测。

---

## 2. 数据接口

### 2.1 输入格式
每个 batch 是一个 Python dict:
```python
batch = {
    'X': torch.FloatTensor,  # shape: (B, 12, num_nodes, feature_dim)
    'y': torch.FloatTensor,  # shape: (B, 12, num_nodes, output_dim)
}
```

- `feature_dim`: 包含原始交通特征 + 时间特征 (time_of_day, day_of_week)
  - PEMSD4 数据集: 原始 3 维 (flow, occupy, speed) + tod embedding + dow embedding → feature_dim ≥ 3
- `output_dim`: 输出维度，通常 = 3 (flow, occupy, speed)
- `num_nodes`: 图的节点数 (PEMSD4 = 307)
- 数据已经过 Z-score (StandardScaler) 归一化
- 数据已经过滑窗切片：每条样本是 `(X_12steps, y_12steps)`

### 2.2 `data_feature` 字典
创建模型时传入:
```python
data_feature = {
    'num_nodes': 307,           # 节点数
    'feature_dim': 3,           # X 的最后一维 (不含时间特征时)
    'output_dim': 3,            # y 的最后一维
    'adj_mx': np.ndarray,       # 静态邻接矩阵，shape (N, N)，元素≥0
    'scaler': Scaler对象,       # 用于 inverse_transform
}
```

### 2.3 输出格式
```python
# predict() 返回:
y_pred: torch.FloatTensor  # shape: (B, 12, num_nodes, output_dim)
# 在 Z-score 空间，由 executor 调用 scaler.inverse_transform 还原
```

---

## 3. 模型架构 (NewDiffusion)

模型 = 条件编码器 + 扩散调度器 + 去噪器，三者组合工作。

### 3.1 整体数据流

```
                 ┌─────────────────────┐
  history X      │    STEncoder        │
  (B,12,N,Cin)──▶│  (condition_encoder)│──▶ H (B,12,N,D)
                 └─────────────────────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                │
     [Training]      [Inference]     [Physics Loss]
          │               │                │
    Y₀ = batch['y']  randn(B,12,N,Cout)   │
    t ~ Uniform(0,T)       │               │
    Y_t = √ᾱₜY₀+√(1-ᾱₜ)ε  │               │
          │               │               │
    ε̂ = denoiser(Y_t,t,H,A)  DDIM采样      │
          │               │               │
    loss = SNR_w×(ε̂-ε)²    Y_pred ←───────┤
    [+ conservation]   clamp(-3,3)        │
                      mean(多采样)
```

### 3.2 核心超参数 (config.json)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `input_window` | 12 | 历史步数 |
| `output_window` | 12 | 预测步数 |
| `hidden_dim` | 128 | 所有隐藏层维度 (D) |
| `num_heads` | 4 | MHA 头数 |
| `encoder_layers` | 2 | 条件编码器层数 |
| `denoiser_layers` | 4 | 去噪器层数 |
| `ffn_hidden_dim` | 256 | FFN 中间维 (通常是 2×hidden_dim) |
| `graph_k_hop` | 2 | 图卷积跳数 K |
| `dropout` | 0.1 | |
| `diffusion_steps` | 200 | 扩散总步数 T |
| `diffusion_schedule` | "linear" | 噪声调度: "linear" 或 "cosine" |
| `beta_start` | 1e-4 | 最小噪声 |
| `beta_end` | 0.01 | 最大噪声 (原 0.02 已降低) |
| `num_sampling_steps` | 50 | DDIM 采样步数 (≤ diffusion_steps) |
| `num_prediction_samples` | 1 | 推理时多采样次数 |
| `sampling_method` | "ddim" | "ddim" 或 "ddpm" |
| `ddim_eta` | 0.0 | DDIM 随机性 (0=确定性) |
| `use_spatiotemporal_attention` | true | 去噪器中是否使用 flattend ST-attn |
| `use_temporal_position_embedding` | true | 是否加时序位置嵌入 |
| `use_gradient_checkpointing` | false | 梯度检查点 (省内存) |
| `use_adaptive_graph` | true | 是否启用自适应图学习 |
| `adaptive_graph_embed_dim` | 32 | 自适应图节点嵌入维 |
| `adaptive_graph_topk` | 12 | 动态图 top-k 稀疏化 (None=全连接) |
| `adaptive_graph_blend_init` | 0.5 | 静态/动态图初始混合比 |
| `use_fuzzy_graph` | true | 是否用模糊隶属度替代 softmax |
| `fuzzy_graph_num_sets` | 3 | 模糊集个数 |
| `fuzzy_graph_sigma_init` | 0.7 | 模糊高斯核初始 σ |
| `physics_loss_weight` | 0.0 | 物理守恒损失权重 |
| `physics_warmup_steps` | 3000 | 物理损失 warmup 步数 |
| `physics_warmup_start_ratio` | 0.2 | warmup 起始比例 |
| `physics_channel_idx` | 0 | 守恒约束作用于哪个通道 |
| `use_fuzzy_conservation` | true | 模糊守恒损失开关 |
| `fuzzy_conservation_threshold` | 0.6 | 高拥堵阈值 |
| `fuzzy_conservation_temperature` | 8.0 | 模糊温度 |
| `prediction_clamp_min` | -3.0 | DDIM 采样每步 clamp 下限 |
| `prediction_clamp_max` | 3.0 | DDIM 采样每步 clamp 上限 |

---

## 4. 组件详细规格

### 4.1 MultiHeadAttention (model.py 顶部的独立类)

标准 batch-first 多头注意力。

**初始化**:
```python
MultiHeadAttention(hidden_dim: int, num_heads: int, dropout: float = 0.1)
```

**forward**:
```python
forward(query: Tensor[B, Lq, D], context: Tensor[B, Lkv, D] | None, mask) -> Tensor[B, Lq, D]
```
- 如果 context 是 None，做 self-attention
- 内部 split heads: `resize(B, L, D) → (B, L, num_heads, D//num_heads) → transpose → (B, num_heads, L, D/heads)`
- 使用 `F.scaled_dot_product_attention` (PyTorch 2.x 原生 flash attention)
- mask 处理: bool mask 取反；2D/3D 补齐到 4D

### 4.2 GraphConvolution

K-hop 图卷积: `X' = Σ_{k=0}^{K} Â^k X W_k`

**初始化**: `GraphConvolution(hidden_dim, k_hop)`
- `projections`: ModuleList of K+1 个 Linear(hidden_dim, hidden_dim)

**forward**: `forward(node_features[B,N,D], adjacency_matrix[N,N] or [B,N,N]) → [B,N,D]`
- 对 adjacency 调用 `expand_adjacency_batch` 扩展到 batch 维
- 调用 `build_normalized_adjacency` 做 D^{-1/2}(A+I)D^{-1/2}
- 循环 k=0..K: `adj_power = adj_power @ adj_norm`, `output += W_k(adj_power @ X)`

### 4.3 AdaptiveGraphLearner

学习动态图结构。核心公式: `A_adapt = (1-blend)×A_static + blend×A_dynamic`

**初始化参数**:
- `node_embeddings`: 可学习 `[N, embed_dim]`
- `feature_projection`: Linear(D→embed_dim)
- `time_projection`: Linear(D→embed_dim)
- `blend_logit`: 标量，sigmoid 后得到 blend 权重

**模糊图扩展**: 当 `fuzzy_enabled=True`
- `fuzzy_centers`: 可学习 `[num_sets]`，初始化为 linspace(-1,1,num_sets)
- `fuzzy_log_sigmas`: 可学习 `[num_sets]`，初始化为 log(sigma_init)
- `fuzzy_rule_logits`: 可学习 `[num_sets]`，规则权重

**forward 算法**:
```
1. 若输入是 4D [B,T,N,D]，取时间均值 → [B,N,D]
2. node_repr = feature_proj(X) + node_embeddings + time_proj(t_emb)
3. node_repr = tanh(node_repr)
4. similarity = node_repr @ node_repr.T / sqrt(embed_dim)  # [B,N,N]
5. 若 fuzzy: A_dyn = row_normalize(fuzzy_similarity(similarity))
   否则: A_dyn = softmax(similarity, dim=-1)
6. 若 top_k 设定: 保留每行 top_k
7. A_dyn += I; row_normalize
8. A_adapt = (1-sigmoid(blend_logit))×A_static + sigmoid(blend_logit)×A_dyn
9. 返回 row_normalize(A_adapt)
```

**fuzzy_similarity 的内部逻辑**:
```
x = tanh(similarity).unsqueeze(-1)  # [B,N,N,1]
centers = fuzzy_centers[1,1,1,:]    # [1,1,1,S]
sigmas = exp(fuzzy_log_sigmas)[1,1,1,:]
membership = exp(-0.5*((x-centers)/sigmas)^2)  # [B,N,N,S]
rule_weights = softmax(fuzzy_rule_logits)      # [S]
relation = sum(membership * rule_weights, dim=-1)  # [B,N,N]
return relation.clamp_min(1e-12)
```

### 4.4 FeedForwardNetwork

两层 MLP:
```
Linear(D → ffn_hidden_dim) → GELU → Dropout → Linear(ffn_hidden_dim → D) → Dropout
```

### 4.5 STEncoderBlock

时空编码块，处理 `[B,T,N,D]`:
```
1. Temporal Self-Attention: 每个节点独立沿时间轴做 self-attn
   (B,T,N,D) → permute→reshape→(B×N,T,D) → MHA → reshape→permute→(B,T,N,D)
2. Add & Norm + Dropout
3. Graph Convolution: (B,T,N,D) → reshape→(B×T,N,D) → GraphConv → reshape→(B,T,N,D)
4. Add & Norm + Dropout
5. FFN
6. Add & Norm
```

### 4.6 STEncoder (条件编码器) = condition_encoder

多个 STEncoderBlock 的堆栈。

**初始化参数**: input_dim, hidden_dim, num_heads, num_layers, ffn_hidden_dim, graph_k_hop, dropout,
use_temporal_position_embedding, max_time_steps (=input_window), use_gradient_checkpointing

**forward**: `forward(X[B,T_in,N,C_in], adjacency) → H[B,T_in,N,D]`
- 首先 `input_projection(C_in→D)`
- 加上 temporal_position_embedding[1,T_in,1,D]
- 逐 block 处理 (训练时可选 gradient checkpointing)
- final LayerNorm

### 4.7 DenoiserBlock

去噪器核心块，处理 `[B,T_out,N,D]`:
```
1. Temporal Self-Attention (与 Encoder 相同)
2. Graph Convolution (与 Encoder 相同)
3. Cross-Attention: 对每个节点, query=noisy_future, key/value=condition_features
   格式: (B,T_out,N,D) 和 (B,T_in,N,D) → 每个节点独立: (B×N,T_out,D) vs (B×N,T_in,D)
4. (可选) Spatiotemporal Attention: (B,T,N,D) → flatten→(B,T×N,D) → self-attn → reshape
5. FFN
```
每步后都有 Add & Norm (Pre-LN 风格)。

### 4.8 AttentionDenoiser (noise_predictor)

**初始化参数**: output_dim, hidden_dim, num_heads, num_layers, ffn_hidden_dim, graph_k_hop, dropout,
use_spatiotemporal_attention, use_temporal_position_embedding, max_future_steps (=output_window),
input_window, use_gradient_checkpointing,
adaptive_graph_enabled, adaptive_graph_embed_dim, adaptive_graph_topk, adaptive_graph_blend_init,
fuzzy_graph_enabled, fuzzy_graph_num_sets, fuzzy_graph_sigma_init, num_nodes, static_adjacency

**条件注入机制 (concat+linear)**:
```
1. temporal_weights = softmax(condition_temporal_weight)  # [T_in]
2. condition_pooled = Σ(H[:,t] * weights[t])  # [B,1,N,D]
3. condition_pooled 扩展到 [B,T_out,N,D]
4. denoiser_input = Linear(concat(denoiser_input, condition_pooled), 2D→D)
```

注意：原来试过 FiLM 调制但导致训练初期条件信号不足，改成了更简单的 concat+linear。

**forward**:
```
forward(noisy_future[B,T_out,N,C_out], timesteps[B], condition_features[B,T_in,N,D],
        adjacency_matrix, return_last_adjacency=False)

1. input_projection(C_out→D)
2. 加 future_position_embedding[1,T_out,1,D]
3. 加 time_embedding: SinusoidalTimeEmbedding(t) → MLP(D→D) → add [B,1,1,D]
4. 条件注入 (见上)
5. 逐 block 处理:
   - 若 adaptive_graph_learner 存在: current_adj = learner(denoiser_input, time_emb)
   - block(denoiser_input, condition_features, current_adj)
6. final_norm → output_projection(D→C_out)  # 预测噪声 ε
```

### 4.9 SinusoidalTimeEmbedding (utils/time_embedding.py)

```python
def forward(timesteps[B]):
    half_dim = D // 2
    freqs = exp(-log(10000) / (half_dim-1) * arange(half_dim))
    angles = t[:,None] * freqs[None,:]
    return cat([sin(angles), cos(angles)], dim=-1)  # [B, D]
    # D 为奇数时 padding 1 维
```

### 4.10 DiffusionScheduler

管理扩散过程的 β, α, ᾱ 等缓存张量。

**初始化**: `DiffusionScheduler(diffusion_steps, schedule, beta_start, beta_end)`
- 构建 betas[T]: linear 模式 = linspace(beta_start, beta_end, T); cosine 模式用改进 cosine 调度
- 预计算并 register_buffer: alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance

**方法**:
- `sample_timesteps(B, device)`: 返回 `randint(0, T, (B,))`
- `add_noise(Y0[B,*,N,*], t[B], noise=None)`: Y_t = √ᾱₜ·Y₀ + √(1-ᾱₜ)·ε
- `predict_start_from_noise(Y_t, t, ε_hat)`: Ŷ₀ = (Y_t - √(1-ᾱₜ)·ε_hat) / √ᾱₜ
- `ddpm_step(x_t, t, ε_hat)`: 单步 DDPM 反向采样
- `ddim_step(x_t, t, ε_hat, eta, t_next)`: 单步 DDIM 反向采样 (支持非连续步数)

### 4.11 辅助函数 (utils/)

**adjacency.py**:
- `build_normalized_adjacency(adj[B,N,N] or [N,N], add_self_loop=True)`: 计算 D^{-1/2}(A+I)D^{-1/2}
- `expand_adjacency_batch(adj, target_B)`: 将邻接矩阵扩展到 target batch size

**attention_ops.py**:
- `apply_temporal_attention(seq[B,T,N,D], attn_module)`: 每个节点独立做时序 self-attn
- `apply_node_temporal_cross_attention(query[B,Tq,N,D], context[B,Tkv,N,D], attn_module)`: 每个节点独立做 cross-attn
- `apply_spatiotemporal_attention(query[B,T,N,D], attn_module, context)`: flatten T×N 做 self/cross-attn

---

## 5. NewDiffusion (顶层模型)

继承 `AbstractTrafficStateModel → AbstractModel → nn.Module`。

### 5.1 类属性
```python
_ddp_loss_through_forward = True  # 训练时 forward() 返回 loss，DDP hook 触发梯度同步
```

### 5.2 核心方法

**`forward(batch)`**:
```python
if self.training:
    return self.calculate_loss(batch)  # → scalar loss
return self.predict(batch)  # → [B, Tout, N, Cout]
```

**`encode_condition(history[B,T_in,N,C_in])`**:
```python
adj = self.adjacency_matrix.to(history.device)
return self.condition_encoder(history, adj)  # → [B,T_in,N,D]
```

**`calculate_loss(batch)`**:
```python
1. X = batch['X'], Y0 = batch['y'][..., :output_dim]
2. H = encode_condition(X)
3. t = uniform_sample(0, T-1, B)
4. Y_t, ε = scheduler.add_noise(Y0, t)
5. ε_hat = noise_predictor(Y_t, t, H, adj)  # 或带 return_last_adjacency
6. loss_per_element = MSE(ε_hat, ε)
7. snr = ᾱₜ / (1-ᾱₜ)
8. loss_weight = clamp(snr + 1, max=10)  # SNR+1 weighting (Improved DDPM)
9. diffusion_loss = (loss_weight * loss_per_element).mean()

10. 若 physics_loss_weight > 0:
    Y0_hat = scheduler.predict_start_from_noise(Y_t, t, ε_hat)
    phys_loss = traffic_conservation_loss(Y0_hat, adaptive_adj)
    总损失 = diffusion_loss + effective_weight * phys_loss
    否则返回 diffusion_loss
```

**`_traffic_conservation_loss(future[B,T,N,C], adj)`**:
```python
1. 取 physics_channel_idx 通道 → state[B,T,N]
2. Δ = state[:,1:,:] - state[:,:-1,:]  # [B,T-1,N] 时间差分
3. outflow = state[:,-1:] * adj.sum(dim=-1)[:,None,:]
4. inflow = einsum('bij,btj→bti', adj.T, state[:,:-1])
5. net_flow = inflow - outflow
6. residual = Δ - flow_conservation_coeff * net_flow
7. 若 use_fuzzy_conservation:
   norm_state = |state| / max|state|
   high_congestion = sigmoid(temperature * (norm_state - threshold))
   weight = 0.5 + high_congestion
   return (weight * residual²).mean()
   否则 return residual².mean()
```

**`predict(batch)`**:
```python
return self.sample(batch['X'], num_samples=num_prediction_samples, return_all=False)
```

**`sample(history[B,T_in,N,C], num_samples, return_all)`**:
```python
1. H = encode_condition(history)
2. expand H: repeat_interleave(num_samples, dim=0)  # [B×K, T_in, N, D]
3. sampling_schedule = [T-1, T-2, ..., 0] 等间距取 num_sampling_steps 个点
4. x_T = randn(B×K, T_out, N, C_out)  # 纯噪声初始化
5. for each step i in schedule:
   t = step.expand(B×K)
   ε_hat = noise_predictor(x_t, t, H, adj)
   if ddim: x_{t-1} = ddim_step(x_t, t, ε_hat, eta, t_next)
   else: x_{t-1} = ddpm_step(x_t, t, ε_hat)
   clamp(x_{t-1}, min, max)  # 防止发散
6. 若 return_all: 返回 [K, B, T_out, N, C_out]
   否则: 返回 mean(dim=0) → [B, T_out, N, C_out]
```

---

## 6. 训练和评估流程

### 6.1 执行器 (DiffusionTrafficStateExecutor)

继承 `TrafficStateExecutor`，只重写了两个方法：

**`_train_epoch`** (与基类的区别):
```python
# 基类: loss_func = loss_func or self.model.calculate_loss
#        loss = loss_func(batch)  ← 调用 calculate_loss，不走 DDP hook
# 本类: loss = self.model(batch)  ← 调用 forward() → DDP hook 正常同步梯度
```
原因：扩散模型的 `forward()` 在训练时返回 loss，需要确保 DDP 的梯度同步 hook 被触发。直接调用 `model.calculate_loss(batch)` 会绕过 DDP wrapper。

**`_valid_epoch`** (与基类的区别):
```python
# 基类: loss_func = loss_func or self.model.calculate_loss
# 本类: loss = self._unwrap_model().calculate_loss(batch)  # 取原始模型
# 并且: DDP 下 all_reduce 求全局平均 val_loss
```

### 6.2 训练循环 (基类 TrafficStateExecutor.train)
```
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # DDP shuffle
    losses = _train_epoch(train_loader)
    val_loss = _valid_epoch(val_loader)
    lr_scheduler.step(val_loss)  # 若 ReduceLROnPlateau
    若 val_loss < min_val_loss: 保存 checkpoint
    否则 wait++: early stop 若 wait >= patience
```

### 6.3 推理评估 (TrafficStateExecutor.evaluate)
```
for batch in test_loader:
    output = model.predict(batch)
    y_pred = scaler.inverse_transform(output)
    y_true = scaler.inverse_transform(batch['y'])
保存 npz 文件，计算 MAE/RMSE/MAPE 等指标
```

---

## 7. 依赖关系图

```
new_diffusion_fuzzy/
  model.py ← 核心
  executor.py ← 继承 GNNTP.common.traffic_state_executor:TrafficStateExecutor
  executor.json ← executor 专属默认配置
  config.json ← 模型默认配置
  manifest.json ← 模型注册元信息
  utils/
    adjacency.py ← 独立的邻接矩阵工具
    attention_ops.py ← 张量 reshape + attention 调用
    time_embedding.py ← SinusoidalTimeEmbedding

外部依赖:
  GNNTP.models.abstract_traffic_state_model:AbstractTrafficStateModel
    └── GNNTP.models.abstract_model:AbstractModel ← nn.Module
  GNNTP.common.traffic_state_executor:TrafficStateExecutor
    └── GNNTP.common.abstract_executor:AbstractExecutor
  GNNTP.common.traffic_state_evaluator:TrafficStateEvaluator
  GNNTP.models.loss (损失函数: masked_mae/mse/rmse/mape 等)
  GNNTP.pipeline:run_model (顶层入口)
  GNNTP.config_parser:ConfigParser (配置合并)
  GNNTP.data.runtime:DataRuntime, build_dataset_runtime
```

---

## 8. 已知问题与陷阱

### 8.1 性能问题

| 问题 | 严重性 | 说明 |
|------|--------|------|
| **峰值低估** | 🔴 严重 | DDIM 多步去噪累积偏置 + 欠参数化 (hidden_dim=128) 导致预测高值只有真实的 70-80% |
| **缺失数据异常值** | 🔴 严重 | DDIM 在缺失数据节点上无约束迭代，产出 1000+ 的异常值 (已加 clamp(-3,3) 部分缓解) |
| **保守损失不收敛** | 🟡 中等 | physics_loss_weight=0.0 时实际关闭了该损失；即使打开，warmup 机制可能过于保守 |

### 8.2 架构缺陷

| 问题 | 说明 |
|------|------|
| **FiLM 被回退** | 原本用 FiLM 做条件调制，因零初始化导致训练初期条件信号缺失，回退到 concat+linear |
| **欠参数化** | hidden_dim=128 + denoiser_layers=4 对于 T_out=12 的多步预测不够 |
| **条件注入简单** | 仅通过 softmax 加权平均所有历史步 → 丢失了时序模式 |
| **SNR+1 权重** | clamp(max=10) 是 ad-hoc 的，没有充分调优 |

### 8.3 建议的改进方向 (按优先级)

1. **扩容**: hidden_dim 128→256, denoiser_layers 4→6
2. **条件升级**: 用 FiLM + 更激进的初始化 (非零偏置)
3. **Cosine 噪声调度**: 替换 linear
4. **U-Net 模式 skip connections**: encoder ↔ denoiser
5. **增加 num_sampling_steps**: 50→100-200
6. **多步预测解耦**: 每个预测步独立采样而非一次性 12 步

### 8.4 代码异味

- **model.py 过于庞大** (922行/45KB): 全部组件塞在一个文件里
- **重复的线性层模式**: STEncoderBlock 和 DenoiserBlock 有大量相似的前后处理
- **_extract 方法是静态方法却用了 self**: 不必要的耦合
- **DDPM/DDIM 两个方法参数不一致**: ddim_step 多一个 t_next 参数
- **physics_warmup 日志逻辑** 嵌入在 calculate_loss 中，关注点不分离

---

## 9. 重写建议

### 9.1 必须保留的核心创新

1. **模糊图学习** (AdaptiveGraphLearner + fuzzy membership): 论文的核心 novel contribution
2. **模糊守恒损失** (_traffic_conservation_loss + fuzzy weighting): 论文的另一个核心
3. **条件扩散框架** (STEncoder + AttentionDenoiser + DiffusionScheduler): 整体方法论
4. **SNR+1 噪声加权**: Improved DDPM 的最佳实践

### 9.2 可以简化的部分

1. **MultiHeadAttention**: 直接用 PyTorch 的 `nn.MultiheadAttention(batch_first=True)`
2. **SinusoidalTimeEmbedding**: 考虑用可学习的 MLP embedding 替代
3. **executor.py**: 如果解决了 DDP hook 的问题，可以直接在基类中集成
4. **utils/**: 三个工具文件可以内联到更合适的模块中

### 9.3 建议的文件划分

```
new_diffusion_fuzzy/
  __init__.py
  config.json
  manifest.json
  executor.json (executor 默认配置)
  model.py           # 只放 NewDiffusion 类
  encoder.py         # STEncoder + STEncoderBlock
  denoiser.py        # AttentionDenoiser + DenoiserBlock
  graph.py           # GraphConvolution + AdaptiveGraphLearner
  diffusion.py       # DiffusionScheduler
  attention.py       # MultiHeadAttention + FeedForwardNetwork
  embedding.py       # SinusoidalTimeEmbedding + temporal positioning
  executors/
    diffusion_executor.py  # DiffusionTrafficStateExecutor
  utils/
    adjacency.py      # 邻接矩阵工具
    attention_ops.py  # 张量 reshape 工具
```

### 9.4 输入/输出规格 (用于测试)

```python
# 最小测试用例
B, N, Tin, Tout, Cin, Cout, D = 2, 10, 12, 12, 3, 3, 128
batch = {
    'X': torch.randn(B, Tin, N, Cin),
    'y': torch.randn(B, Tout, N, Cout),
}
adj = torch.eye(N) + torch.rand(N, N) * 0.1
data_feature = {
    'num_nodes': N, 'feature_dim': Cin, 'output_dim': Cout,
    'adj_mx': adj.numpy(), 'scaler': StandardScaler(...)
}

model = NewDiffusion(config, data_feature)
# 训练: model.training=True → model(batch) 返回标量 loss
# 推理: model.training=False → model(batch) 返回 [B,Tout,N,Cout]
```

---

## 10. 运行方式

### 10.1 数据准备
```bash
uv run scripts/run/run_data_artifact.py \
  --task traffic_speed_prediction \
  --model new_diffusion_fuzzy \
  --dataset PEMSD4
```
在 `cache/data_artifacts/` 生成工件目录。

### 10.2 训练
```bash
uv run scripts/run/run_train_artifact.py \
  --model new_diffusion_fuzzy \
  --dataset PEMSD4 \
  --artifact_id <da_xxx>
```

### 10.3 单卡 vs 多卡
- 单卡: 直接 `python` 或 `uv run`
- 多卡(DDP): `torchrun --nproc_per_node=N scripts/run/run_train_artifact.py ...`

### 10.4 诊断
```bash
uv run scripts/tools/test_diffusion_denoiser_recovery.py --artifact_id <id>
```

---

*本文档由 CherryClaw 于 2026-05-08 生成，基于对 `new_diffusion_fuzzy` 分支源码的完整分析。*
