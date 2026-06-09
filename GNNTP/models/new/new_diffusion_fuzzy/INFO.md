# new_diffusion_fuzzy — Conditional Diffusion with Fuzzy Graph

当前核心实验模型：模糊图学习 + 模糊守恒损失 + 条件扩散（DDIM/DDPM 采样），有专用 Executor。

通过工件流水线运行：`run_train_artifact.py --model new_diffusion_fuzzy --dataset <dataset> --artifact_id <id>`

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | 模型结构、训练损失、DDIM/DDPM 采样（621行） |
| `executor.py` | DiffusionTrafficStateExecutor（DDP-safe 训练） |
| `utils/adjacency.py` | 邻接矩阵归一化与批量扩展 |
| `utils/attention_ops.py` | 时空/时序/cross-attention 张量变换 |
| `utils/time_embedding.py` | 正弦时间步嵌入 |
| `config.json` | 默认实验参数 |
| `manifest.json` | 注册信息 |

## 架构概览

```
history (B, Tin, N, Cin)
    │
    ▼
STEncoder (condition_encoder) → H (B, Tin, N, D)
    │
    ├── [Training] calculate_loss()
    │       condition H + future Y₀ → add_noise(Y₀, t) → noisy Y_t
    │       noise_predictor(Y_t, t, H, A) → ε̂
    │       loss = SNR_weighted × MSE(ε̂, ε) [+ conservation]
    │
    └── [Inference] predict() → sample()
            condition H → randn() start
            DDIM 50-step reverse: ε̂ = noise_predictor(x_t, t, H, A) → x_{t-1}
            → mean(N samples) → (B, Tout, N, Cout)
```

## 组件清单

| 组件 | 位置 | 功能 |
|------|------|------|
| `MultiHeadAttention` | model.py:24 | Batch-first MHA，支持 cross-attn |
| `GraphConvolution` | model.py:85 | K-hop 图卷积 X' = Σ Âᵏ X Wₖ |
| `AdaptiveGraphLearner` | model.py:111 | 自适应图 + 模糊关系矩阵 |
| `FeedForwardNetwork` | model.py:217 | 位置 FFN |
| `STEncoderBlock` | model.py:235 | 时序 attn + 图卷积 + FFN |
| `STEncoder` | model.py:265 | 历史序列 → 条件编码 H |
| `DenoiserBlock` | model.py:322 | 时序 self + 图 conv + cross + [spatiotemporal] + FFN |
| `AttentionDenoiser` | model.py:380 | 噪声预测 ε_θ(Y_t, t, H, A)，concat+linear 条件注入 |
| `DiffusionScheduler` | model.py:512 | 噪声调度、前向加噪、DDPM/DDIM 反向步 |
| `NewDiffusion` | model.py:621 | 顶层模型，整合训练/采样/预测 |
| `DiffusionTrafficStateExecutor` | executor.py | DDP 兼容训练器 |

## 默认配置 (config.json)

| 参数 | 值 | 说明 |
|------|----|------|
| hidden_dim | 128 | 隐藏维度 |
| encoder_layers | 2 | 条件编码器层数 |
| denoiser_layers | 4 | 去噪器层数 |
| ffn_hidden_dim | 256 | FFN 隐藏维 |
| graph_k_hop | 2 | 图卷积跳数 |
| diffusion_steps | 200 | 扩散总步数 |
| diffusion_schedule | linear | 噪声调度策略 (linear/cosine) |
| beta_start / beta_end | 1e-4 / 0.02 | 噪声范围 (仅 linear 模式生效) |
| num_sampling_steps | 50 | DDIM 采样步数 |
| num_prediction_samples | 1 | 推理采样次数 |
| sampling_method | ddim | 采样方式 (ddim/ddpm) |
| ddim_eta | 0.0 | DDIM 随机性控制 |
| physics_loss_weight | 0.0 | 物理守恒损失权重 (0=关闭) |
| use_fuzzy_graph | true | 模糊图学习开关 |
| use_fuzzy_conservation | true | 模糊守恒损失开关 |
| use_spatiotemporal_attention | true | 时空注意力开关 |
| max_epoch | 100 | |
| learner | adamw | |
| learning_rate | 0.001 | |
| batch_size | 32 | |

## 输入/输出

- **输入**: dict `{X: [B, 12, N, 3], y: [B, 12, N, 3]}`
  - X: 历史 12 步 (traffic_flow, traffic_occupancy, traffic_speed)
  - y: 未来 12 步真值
- **输出**: 预测 ŷ [B, 12, N, 3] (Z-score 空间，由 executor 调用 scaler.inverse_transform 还原)

## 已知问题与注意事项

1. **FiLM 零初始化**: `condition_film_proj[-1]` 全零初始化导致条件信号在训练初期不可用，需要足够 epoch 才能学到有效调制
2. **DDIM 对去噪器精度敏感**: val_loss < 0.1 才能稳定多步采样，否则误差指数级累积
3. **决策**: 修改 `config.json` 时同步确认模型需要的 config key 在 `__init__` 中有对应 default 值
4. **参数调整**: 优先改 `config.json`，避免硬编码

## 调用关系

- 运行: `uv run scripts/run/run_train_artifact.py --model new_diffusion_fuzzy --dataset PEMSD4`
- 注册: 通过 `manifest.json` 由 `GNNTP/models/locator.py` 索引
- Executor: `DiffusionTrafficStateExecutor` (executor.py) 而非默认 `TrafficStateExecutor`
