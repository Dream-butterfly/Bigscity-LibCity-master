# final_new 目标架构：Fuzzy Region Routing (FRR) 深度重构方案（v3）

> 日期: 2026-05-26
> v2: 加入 Region Transformer + 叙事重构
> v3: 修正 batch mixing bug / 1/d 风险 / Floyd-Warshall / 叙事压缩 / 冻结扩展
> 状态: 方案制定完成，待实施
> 前置: `2026-05-26_new_fuzzy_cellattention_full_analysis.md`

---

## 〇、核心收敛（第三轮）

经过三轮分析，最终的架构定位从"拼接插件"收敛为"空间交互替代方案"：

> **FRR (Fuzzy Region Routing) 替代空间 self-attention，与 GCN 形成 local-global spatial dual。**
> **在 latent region space 中执行 Transformer interaction，复杂度 O(NK+K²)。**

关键判断链：

```
① CellAttention ≠ 另一个 attention 模块
② CellAttention = 空间全局交互的唯一机制
③ 当前架构没有空间 self-attention（时间 attention 是 per-node 的）
④ 因此：FRR 不是"加进来的东西"，它 IS 空间全局交互
⑤ 升级：在 region space 中做 Transformer → Region Transformer
```

---

## 一、理论减法原则

本次重构的指导思想：

```
删掉语义包装，保留结构本质。
删掉冗余机制，保留正交互补。
删掉"又一个插件"叙事，改为"空间交互替代方案"叙事。
```

### 明确删除

| 删除项 | 原因 |
|--------|------|
| ❌ 特征距离空心核 (DoG on cdist) | 与 region affinity 双重调制 |
| ❌ 全时空交叉注意力 (ST-Attn in Decoder) | 与 FRR 功能重叠，削弱独特性论证 |
| ❌ quantum amplitude / Fisher-Riemannian / Hellinger 语言 | 装饰性数学，叙事崩塌 |
| ❌ "GMM 才是正确" / "Bhattacharyya 系数" 贡献级叙事 | Laplacian softmax 有长尾优势；sqrt 只是实现细节 |
| ❌ "并行叠加"叙事 | 改为 "local-global spatial dual" |
| ❌ embedding.py | 从未使用 |
| ❌ AdaptiveGraphLearner | 已 DEPRECATED |
| ❌ Floyd-Warshall O(N³) 最短路径 | 改为迭代 hop distance O(K·N²) |
| ❌ band-pass 中的 `1/d` 因子 | 训练初期可能导致 gate 爆炸 / NaN |

### 明确保留

| 保留项 | 原因 |
|--------|------|
| ✅ FuzzyRelationalGraphLearner (max-min) | 提供 μ 和 R |
| ✅ FuzzyGraphConvolution | 空间-局部：K-hop 拓扑传播 |
| ✅ Query Decoder | 确定性解码 |
| ✅ 可学习混合权重 λ₁/λ₂ | 自动平衡 GCN 与 FRR |
| ✅ Łukasiewicz 守恒损失（→ FIR） | 模糊交互正则 |
| ✅ 稳定性诊断 API | 论文分析 |
| ✅ 逐节点时间 Transformer | 时间维建模 |

### 新增

| 新增项 | 目的 |
|--------|------|
| ➕ 迭代 hop distance (graph_dist) | band-pass 的拓扑基础，O(K·N²)，1s 内完成 |
| ➕ 显式三步路由 | 揭示低秩本质，降低计算 |
| ➕ 纯 log-Gaussian band-pass gate | GCN→FRR 分工的精确表达（无 1/d 因子） |
| ➕ Linear(K_f→K_c) 条件链路 | 统一两套模糊空间 |
| ➕ **Region Transformer (self-attn on [T×K, D])** | **核心升级：时间感知的 region token Transformer** |
| ➕ Temporal-aware region tokens [T, K, D] | 保留时间动态信息（非逐个时间步循环） |
| ➕ 简单一致性 baseline (FIR 消融) | 证明 T-norm 必要性 |
| ➕ 论文叙事压缩为两条主线 | "FRR" + "topology-aware gate" |

---

## 二、核心架构定位：Local-Global Spatial Dual

### 2.1 三层处理框架

```
每个 Block 的三层处理：

  ┌─────────────────────────────────────────────────────┐
  │  Temporal:   逐节点 Time Transformer                │
  │              复杂度 O(N·T²)  |  建模长期时序依赖      │
  ├─────────────────────────────────────────────────────┤
  │  Spatial-Local:  FuzzyGCN (K-hop via R)             │
  │              复杂度 O(K·|E|)  |  高保真邻域传播      │
  ├─────────────────────────────────────────────────────┤
  │  Spatial-Global:  FRR (Fuzzy Region Routing)        │
  │              复杂度 O(NK+K²)  |  低秩远程功能路由    │
  │              ★ 替代空间 self-attention              │
  └─────────────────────────────────────────────────────┘
```

**关键叙事**:

> 我们不在空间维使用 self-attention（O(N²)）。
> 空间交互由 local GCN（邻域保真）+ global FRR（远程低秩）共同完成。
> FRR 将节点投影到 K 个 latent region tokens，
> 在 region space 中执行 Transformer interaction，
> 再回写到节点空间。

### 2.2 Block 子层结构（不变，叙事升级）

```python
# 每个 Encoder/Decoder Block:
x = x + TempAttn(x)       # ① Temporal: 时间维 self-attention
x = x + FuzzyGCN(x, R)    # ② Spatial-Local: K-hop 图传播
x = x + FRR(x, ...)       # ③ Spatial-Global: 低秩区域路由
x = x + CrossAttn(x, H)   # ④ [仅 Decoder] 历史条件注入
x = x + FFN(x)            # ⑤ 逐位置非线性
```

三层空间交互是**有序子层**（非并行分支），顺序保证渐进式特征精炼：先理解时间模式 → 聚合局部邻域 → 全局路由（输入已融合了前两步的信息）。

---

## 三、FRR (Fuzzy Region Routing) 完整数学流程

### 3.1 八步前向（静态模式 [N,D]）

```
输入: x [N,D] 节点特征, μ_fuzzy [N,K_f] 来自 FuzzyGraph, graph_dist [N,N] hop distance

 ① 区域归属（soft assignment via learnable prototypes）
    u_ik = softmax( -||x_i - c_k||² / 2σ²  +  Linear(μ_fuzzy_i)_k )
    u ∈ [0,1]^(N×K_c),  Σ_k u_ik = 1
    σ² = softplus(log_sigma_sq) + 0.01

 ② Routing weights
    B_ik = sqrt(u_ik)
    使用 sqrt 而非 u 本身: 保留概率单纯形上的几何结构，
    避免 L2-norm 对低熵分布的失真。

 ③ 消息编码
    X̃ = W_enc · x                                     [N, D_enc]

 ④ Region token 聚合
    M = B.T @ X̃                                        [K_c, D_enc]
    每个 region token 是 N 个节点的加权聚合。
    rank(M) ≤ K_c — 信息瓶颈，K_c << N。

 ⑤ 拓扑 band-pass gate（node→region）
    d_ik = Σ_j B_jk² · graph_dist[i,j] / Σ_j B_jk²    [N, K_c]
    gate_ik = exp( -(log d_ik - μ)² / 2σ² )           [N, K_c]
    (纯 log-Gaussian，无 1/d 因子，d→0 时自动 gate→0)
    M_k ← M_k · Σ_i(gate_ik · B_ik²) / Σ_i B_ik²

 ⑥ Region Transformer ★ 核心升级
    M = M + MHA_region(M, M, M)                        [K_c, D_enc]
    M = LayerNorm(M + FFN_region(M))                   [K_c, D_enc]
    复杂度: O(K_c²) — K_c=8 时 64 attn weights。
    区域 tokens 之间通过 self-attention 互相通信，
    筛选、重组来自不同节点的全局信息。

 ⑦ 节点读回
    H = B @ M                                          [N, D_enc]

 ⑧ 输出投影 + 残差混合
    output = W_out · H                                 [N, D]
    final = x + sigmoid(λ₂) · output
```

### 3.2 时序模式（[T,N,D] 或 [B,T,N,D]）

```
 输入: x [T,N,D] (batch 均值化后)

 ① 区域归属: 从时间平均特征计算（城市功能区是共享的）
    u = softmax( -||mean_t(x) - c_k||² / 2σ² + Linear(μ_fuzzy) )  [N, K]
    B = sqrt(u).clamp(min=1e-8)

 ③④ 逐时间步聚合 → 时间感知 region tokens
    X̃ = W_enc · x                                       [T, N, D]
    M = einsum('nk,tnd->tkd', B, X̃)                     [T, K, D]

 ⑤ 拓扑门控: 基于静态归属计算（同静态模式）
    gate_weight = (gate * B²).sum(0) / B².sum(0)        [K]
    M = M * gate_weight                                 [T, K, D]

 ⑥ Region Transformer on [T×K, D]
    M = reshape(T×K, D) → RegionTransformer → reshape(T, K, D)
    这样 K 个区域 × T 个时间步联合参与 self-attention。
    区域间交互和跨时间步交互在一个注意力矩阵中完成。

 ⑦ 逐时间步读回 + 时间平均
    H = einsum('tkd,nk->tnd', M, B)                     [T, N, D]
    H_mean = H.mean(dim=0)                              [N, D]
```

### 3.3 Region Transformer 子模块

```python
class RegionTransformer(nn.Module):
    """Self-attention Transformer on region tokens [K, D] or [T×K, D].
    
    在 region space 中执行标准 Transformer interaction。
    静态模式: [K, D]; 时序模式: [T×K, D] — K 区域 × T 时间步联合注意力。
    复杂度: O((T·K)²·D) — T=12, K=8 → 9600 attn weights，≈ O(N²/40)。
    """
    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim * 2, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, region_tokens):
        # region_tokens: [K, D] or [T×K, D]
        x = self.norm1(region_tokens + self.dropout(
            self.self_attn(region_tokens)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

### 3.4 与 Perceiver / Latent Transformer 的区分

| 维度 | Perceiver | FRR (Ours) |
|------|-----------|------------|
| Latent tokens | Learned queries | Prototypes (Gaussian centers) |
| 节点→latent | Cross-attn (Q=latent, K/V=nodes) | Soft assignment via prototype distance |
| 空间先验 | 无 | Hop-distance band-pass gate |
| 局部传播 | 无 | GCN 互补分支 |
| 时间处理 | Cross-attn over flattened [T×N] | Per-timestep region tokens [T, K] → [T×K] Transformer |
| 复杂度 | O(NK) cross-attn / layer | O(NK) routing + O((T·K)²) region Transformer |

### 3.5 三种输入形态支持

```python
def forward(self, x, graph_dist=None, mu_fuzzy=None, return_stability=False):
    """Auto-dispatch to static / temporal forward.
    
    支持: [N,D] | [T,N,D] | [B,T,N,D]
    
    [B,T,N,D] → mean(dim=0) → [T,N,D]: 跨 batch 共享城市功能区划分，
    但保留时间维用于 per-timestep region token 聚合。
    """
    if x.dim() == 4:
        x = x.mean(dim=0)               # [B,T,N,D] → [T,N,D]
    if x.dim() == 3:
        return self._forward_temporal(x, graph_dist, mu_fuzzy, return_stability)
    return self._forward_static(x, graph_dist, mu_fuzzy, return_stability)
```

### 3.5 诊断 API

```python
def get_region_diagnostics(x, graph_dist=None):
    """区域级别诊断。
    Returns:
        region_size: [K] 有效节点数
        region_entropy: scalar 区域分布熵
        region_attention: [K,K] 区域间注意力矩阵
        boundary_nodes: [N] 高熵节点
    """
```

---

## 四、全模型数据流

```
X [B, 12, N, C_in]
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  FuzzyRelationalGraphLearner                         │
│  → μ [N, K_f=4]     (sigmoid, 独立激活)               │
│  → R [N, N]          (max-min 模糊关系)               │
│  → graph_dist [N, N] (预计算 hop distance)              │
└──────────────┬───────────────────────────────────────┘
               │ R, μ, graph_dist
               ▼
┌──────────────────────────────────────────────────────┐
│  STEncoder (条件编码器)                                │
│  InputProj + TemporalPE                              │
│  × 2 blocks:                                         │
│    Temporal Attn → FuzzyGCN → FRR → FFN              │
│  → H [B, 12, N, D]                                    │
└──────────────┬───────────────────────────────────────┘
               │ H, R, μ, graph_dist
               ▼
┌──────────────────────────────────────────────────────┐
│  FutureDecoder                                        │
│  Learnable queries [1, 12, N, D]                     │
│  × 2 blocks:                                         │
│    Temporal Attn → FuzzyGCN → FRR → CrossAttn → FFN  │
│  → Ŷ [B, 12, N, C_out]                                │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
  Loss = L1(Ŷ, Y) + λ · FIR(Ŷ, R)
```

### 与当前架构的关键差异

| 维度 | 当前 | 目标 |
|------|------|------|
| 空间全局交互 | CellAttn (隐式, `[N,N]@[N,D]`) | FRR (显式三步 + Region Transformer) |
| 距离度量 | `cdist(x,x)` 特征距离 | `hop_distance(adj)` 拓扑跳数 |
| 滤波函数 | DoG (双高斯 3 参数) | 纯 log-Gaussian band-pass (2 参数, 无 1/d) |
| Region 角色 | 被动聚合桶 | **活跃 latent tokens (self-attn)** |
| Decoder 子层数 | 6 (含 ST-Attn) | 5 (去除 ST-Attn) |
| 叙事 | "模糊胞型注意力作为补充" | "FRR 替代空间 self-attention" |

---

## 五、模块级规范

### 5.1 cell_attention.py（重写，~400 行）

**核心类:**
- `RegionTransformer`: Self-attn + FFN on [K, D] or [T×K, D]
- `FuzzyCellAttention`: 完整八步 FRR（类名保留不改为稳定 import 链；所有 docstring 使用 "Fuzzy Region Routing (FRR)"）

**关键方法:**
- `_compute_membership(x, mu_fuzzy)`: ① 区域归属（soft assignment via prototypes）
- `_bandpass_gate(graph_dist, B)`: ⑤ 拓扑门控（纯 log-Gaussian, 无 1/d）
- `_forward_static(x, graph_dist, mu_fuzzy)`: 静态模式 [N,D] 全流程
- `_forward_temporal(x, graph_dist, mu_fuzzy)`: 时序模式 [T,N,D] → temporal-aware [T,K,D] Region Tokens
- `forward(x, graph_dist, mu_fuzzy)`: 入口分发（[N,D]|[T,N,D]|[B,T,N,D]）
- `get_stability_metrics(x)`: 稳定性诊断（H 熵 + S 稳定度）
- `get_region_diagnostics(x, graph_dist)`: 区域级诊断（可选）

### 5.2 graph.py（修改）

- 保留: `GraphConvolution`, `FuzzyGraphConvolution`, `FuzzyRelationalGraphLearner`
- `FuzzyRelationalGraphLearner`: 新增 `num_fuzzy_sets` 属性
- 删除: `AdaptiveGraphLearner` (L319-432)

### 5.3 encoder.py（修改）

- `STEncoderBlock.__init__`: 移除 `use_hollow_kernel` 参数
- `STEncoderBlock.forward`: 新增 `graph_dist`, `mu_fuzzy` 参数
- CellAttention 调用: 直接传 4D tensor
- 文档: "Temporal → Local(GCN) → Global(FRR)" 三层叙事

### 5.4 decoder.py（修改）

- `DecoderBlock.__init__`: 移除 `use_spatiotemporal_attention` 及相关模块
- `DecoderBlock.forward`: 移除 ST-Attn 子层
- 子层顺序: TempAttn → FuzzyGCN → FRR → CrossAttn → FFN (5 层)

### 5.5 model.py（修改）

- `__init__`: 预计算 `graph_dist` via `compute_hop_distance()`；移除 5 个废弃开关
- `encode_condition`: 返回 `mu_fuzzy`
- `calculate_loss`: `_fuzzy_conservation_loss` → `_fuzzy_interaction_regularization`
- 新增: `_simple_consistency_loss` (FIR baseline)

### 5.6 utils/adjacency.py（新增函数）

```python
def compute_hop_distance(adjacency, max_hops=20):
    """Iterative hop distance propagation. O(K·N²), <1s for N≤1000.
    
    交通图不需要精确欧氏最短路径，hop distance 完全足够。
    """
    N = adjacency.shape[0]
    dist = torch.full((N, N), float(max_hops), dtype=torch.float32)
    dist.fill_diagonal_(0.0)
    dist[adjacency > 0] = 1.0
    
    A_k = adjacency.float()
    for hop in range(2, max_hops + 1):
        A_k = (A_k @ adjacency.float()).clamp(min=0, max=1)
        mask = (A_k > 0) & (dist == max_hops)
        if not mask.any():
            break
        dist[mask] = float(hop)
    return dist
```

### 5.7 utils/attention_ops.py（删除函数）

- 删除: `apply_spatiotemporal_attention`

---

## 六、config.json 变更

```json
{
  "input_window": 12,
  "output_window": 12,
  "hidden_dim": 64,
  "num_heads": 2,
  "encoder_layers": 2,
  "decoder_layers": 2,
  "ffn_hidden_dim": 128,
  "graph_k_hop": 2,
  "dropout": 0.1,
  "use_temporal_position_embedding": true,
  "use_gradient_checkpointing": false,
  "use_amp": true,
  "scaler": "standard",
  "load_external": false,
  "normal_external": false,
  "add_time_in_day": true,
  "add_day_in_week": true,
  "max_epoch": 100,
  "learner": "adamw",
  "learning_rate": 0.0005,
  "lr_decay": true,
  "lr_scheduler": "reducelronplateau",
  "lr_decay_ratio": 0.5,
  "lr_patience": 5,
  "lr_threshold": 0.001,
  "weight_decay": 0.0001,
  "clip_grad_norm": true,
  "max_grad_norm": 3,
  "use_early_stop": true,
  "patience": 30,
  "batch_size": 64,
  "eval_batch_size": 64,

  "conservation_loss_weight": 0.1,
  "conservation_warmup_epochs": 5,
  "conservation_steps_per_epoch": 80,
  "physics_channel_idx": 0,

  "fuzzy_num_sets": 4,
  "num_cells": 8,
  "cell_blend_init": 0.3,
  "band_center_init": 1.1,
  "band_width_init": 0.7,
  "region_transformer_layers": 1,
  "fir_mode": "lukasiewicz"
}
```

---

## 七、论文叙事主线（定稿版）

### 一句话

> We replace spatial self-attention with **Fuzzy Region Routing (FRR)**: nodes project into K latent region tokens, where a lightweight Transformer performs inter-region interaction, then redistributes — O(NK+K²).

### 贡献列表（两条主线）

1. **Fuzzy Region Routing (FRR)** — 核心贡献:
   - Soft region assignment via learnable prototypes
   - Region token aggregation (nodes→K regions, low-rank bottleneck)
   - Region Transformer (self-attention among region tokens)
   - Node readback (regions→nodes)
   - 与 GCN 形成 local-global spatial dual
   - sqrt routing weights, Gaussian prototype distance 均为**实现细节**，不作为独立贡献

2. **Topology-aware Band-pass Gate** — 辅助贡献:
   - 基于 hop distance 的纯 log-Gaussian gate
   - 抑制 GCN 覆盖的近邻，聚焦中程功能路由
   - 2 个可学习参数 (center + width)

### Figure 1 构思

```
┌──────────────────────────────────────────────────────┐
│              Spatial Interaction Architecture         │
│                                                      │
│   ┌─────────────┐          ┌────────────────────┐    │
│   │   Local     │          │      Global        │    │
│   │   GCN       │          │  Region Routing    │    │
│   │             │          │                    │    │
│   │  K-hop      │          │  Nodes             │    │
│   │  neighbor   │          │    ↓ soft assign   │    │
│   │  propagation│          │  Region Tokens     │    │
│   │             │          │    ↓ self-attn     │    │
│   │  O(K·|E|)  │          │  Region Transf.    │    │
│   │             │          │    ↓ redistribute  │    │
│   └──────┬──────┘          │  Nodes             │    │
│          │                 │                    │    │
│          │                 │  O(NK + K²)        │    │
│          └────────┬────────┘                    │    │
│                   ▼                             │    │
│         H' = λ₁·GCN(H) + λ₂·FRR(H)              │    │
│                                                      │
│  Fig 1. Local-global spatial dual.                   │
└──────────────────────────────────────────────────────┘
```

---

## 八、参数语义对照

| 参数 | 旧 | 新 | 语义 |
|------|-----|-----|------|
| K_f | `fuzzy_num_sets: 3` | `fuzzy_num_sets: 4` | FuzzyGraph 模糊集数量 |
| K_c | `num_cells: 8` | `num_cells: 8` | FRR 区域数量 |
| 归属控制 | `log_temperature` | `log_sigma_sq` | 原型方差（控制区域归属的模糊程度） |
| 拓扑门控中心 | — (DoG) | `band_center_init: 1.1` | log(目标跳数) ≈ log(3) |
| 拓扑门控宽度 | — (DoG) | `band_width_init: 0.7` | band-pass 容忍度（纯 log-Gaussian, 无 1/d） |
| Region Transf. | — (恒等) | `region_transformer_layers: 1` | Region token 上的 Transformer 层数 |
| ST-Attn | `use_spatiotemporal_attention: true` | — | 删除 |
| 空心核 | `use_hollow_kernel: true` | — | band-pass 替代 |
| 模糊图开关 | `use_fuzzy_graph: true` | — | 始终启用 |
| CellAttn开关 | `use_cell_attention: true` | — | 始终启用 |
| 守恒模式 | `use_fuzzy_conservation: true` | `fir_mode: "lukasiewicz"` | 三选一 |

---

## 九、分步实施计划（修订版）

### Phase 0: 创建副本

```
Step 0.1: cp new_fuzzy_cellattention/ → final_new/
Step 0.2: 更新 manifest.json ("model": "final_new")
验证: python -c "from GNNTP.models.new.final_new import NewFuzzyCellAttention"
```

### Phase 1: 重写 cell_attention.py（核心，~400 行）

```
Step 1.1: 新增 RegionTransformer 类 (支持 [K,D] / [T×K,D])
Step 1.2: 重写 FuzzyCellAttention.__init__
  参数: hidden_dim, num_cells, cell_blend_init,
        band_center_init, band_width_init,
        region_transformer_layers, num_heads, dropout
  删除: use_hollow_kernel, membership_temperature,
        log_sigma_excite, log_sigma_inhibit, inhibit_weight
  新增: log_sigma_sq (替代 temperature),
        band_center, band_width_raw (替代 excite/inhibit),
        region_transformer (替代 hollow kernel),
        fuzzy_to_cell (条件链路, 默认 None)

Step 1.3: 实现 _compute_membership(x, mu_fuzzy)
  u = softmax(-L2²/2σ² + Linear(μ_fuzzy))
  docstring: "Soft region assignment via learnable prototypes"
  (不写 GMM / Gaussian posterior)

Step 1.4: 实现 _bandpass_gate(graph_dist, B)
  d_ik = Σ_j B_jk² · graph_dist[i,j] / Σ_j B_jk²
  gate = exp(-(log_d - μ)² / 2σ²)   ← 纯 log-Gaussian, 无 1/d

Step 1.5: 实现 _forward_static(x, graph_dist, mu_fuzzy)
  ① membership → ② sqrt → ③ encode → ④ aggregate
  → ⑤ bandpass → ⑥ region_transformer → ⑦ readback → ⑧ output+blend

Step 1.6: 实现 _forward_temporal(x, graph_dist, mu_fuzzy)
  ★ Region tokens: [T, K, D] (非逐个时间步循环)
  ★ Region Transformer on flattened [T×K, D]
  ★ 时间平均读回

Step 1.7: 重写全部 docstring
  删除: GMM / Bhattacharyya / Hellinger / quantum amplitude
  统一: "Fuzzy Region Routing (FRR)"
  贡献级语言仅用于: "soft region assignment" + "Region Transformer"
```

### Phase 2: graph.py + utils 改造

```
Step 2.1: utils/adjacency.py — 新增 compute_hop_distance()  (iterative O(K·N²))
Step 2.2: graph.py — FuzzyRelationalGraphLearner 新增 @property num_fuzzy_sets
Step 2.3: graph.py — 删除 AdaptiveGraphLearner (L319-432)
Step 2.4: utils/attention_ops.py — 删除 apply_spatiotemporal_attention
Step 2.5: utils/__init__.py — 移除 spatiotemporal_attention 导出
```

### Phase 3: encoder.py + decoder.py 适配

```
Step 3.1: STEncoderBlock
  - __init__: 移除 use_hollow_kernel
  - forward: 新增 graph_dist, mu_fuzzy 参数
  - CellAttention 调用: 直接传 4D tensor + graph_dist + mu_fuzzy

Step 3.2: STEncoder.forward 透传 graph_dist, mu_fuzzy

Step 3.3: DecoderBlock
  - __init__: 移除 use_spatiotemporal_attention 及模块
  - forward: 移除 ST-Attn 子层, 新增 graph_dist, mu_fuzzy

Step 3.4: FutureDecoder 同适配
```

### Phase 4: model.py 集成

```
Step 4.1: __init__ 预计算 graph_dist → register_buffer
Step 4.2: 移除所有开关参数
Step 4.3: encode_condition 返回 mu_fuzzy
Step 4.4: _fuzzy_conservation_loss → _fuzzy_interaction_regularization
Step 4.5: 新增 _simple_consistency_loss
Step 4.6: 删除 embedding.py
```

### Phase 5: 配置与清理

```
Step 5.1: config.json — 更新参数 (移除 5 个开关, 新增 6 个参数)
Step 5.2: manifest.json — "model": "final_new"
Step 5.3: executor.json — 不变
Step 5.4: __init__.py — 更新 docstring
Step 5.5: 清理 __pycache__/ 和 embedding.py
Step 5.6: [可选] 文件重命名
  cell_attention.py → region_routing.py
  class FuzzyCellAttention → class FuzzyRegionRouting
  (需同步更新 encoder.py / decoder.py / model.py / __init__.py 的 import)
  如时间紧则跳过 — 代码名字不影响论文叙事
```

### Phase 6: 验证

```
Step 6.1: 语法检查 (compile all .py)
Step 6.2: 导入检查
Step 6.3: 前向传播形状检查
Step 6.4: 完整训练 1 epoch
```

### Phase 7: 凝固 — 停止扩展，转入实验

```
Phase 6 完成后，功能冻结。不再新增模块、参数或数学机制。

后续工作:
  ✅ FRR 稳定性调参 (band_center/width sweep)
  ✅ 完整消融实验 (8 项，按 §11)
  ✅ 复杂度分析 (FLOPs/参数量 vs baselines: HA, STGCN, DCRNN, STTN, AGCRN, STGformer)
  ✅ 可解释性可视化 (region assignment heatmap, band-pass gate distribution, 
     region attention matrix)
  ✅ 长预测 horizon (24/36 steps)
  ✅ 多数据集 (METR-LA, PEMS-BAY, PEMSD4, PEMSD7)
  ✅ 数据稀疏场景 (mask 20%/40%/60% 输入)

不做的:
  ❌ 层级区域结构 (hierarchical regions)
  ❌ 动态 K (adaptive rank)
  ❌ 多尺度 band-pass
  ❌ diffusion distance 距离度量
  ❌ 额外的正则项
  ❌ 渐进蒸馏/一致性模型
```

---

## 十、风险与回退

| 风险 | 缓解 |
|------|------|
| band-pass gate 退化 (gate→0) | gate 按列归一化保证每区域至少一个节点的 gate=1 |
| sqrt 数值不稳定 (u→0) | `u.clamp(min=1e-8)` 在 sqrt 前 |
| Region Transformer 过度设计 | `region_transformer_layers=0` 回退为恒等映射 (纯 low-rank smoothing) |
| Hop distance 在大图上慢 (N>1000) | Dijkstra from each node 或 pre-computed once, cache to disk |
| Temporal region tokens 显存增加 (T×K×D) | T=12, K=8, D=64 → 6KB, 可忽略 |
| 1/d 因子已移除 | 纯 log-Gaussian 在 d→0 时自动 gate→0, 无 NaN 风险 |

---

## 十一、消融实验设计（论文用）

| # | 消融 | 验证目标 |
|---|------|---------|
| A1 | FRR 完全移除 (仅 GCN) | FRR 的绝对贡献 |
| A2 | Region Transformer 关闭 (`layers=0`) | Region self-attention 的增益 |
| A3 | band-pass → 恒等 gate=1 | 拓扑门控的必要性 |
| A4 | μ_fuzzy 条件链路移除 | FuzzyGraph→FRR 连接的价值 |
| A5 | sqrt → cosine (L2_norm) | Routing weights 形式的影响 |
| A6 | K_c = 2, 4, 8, 12, 16 | 低秩瓶颈的敏感度分析 |
| A7 | FIR → simple consistency | T-norm vs 简单一致性 |
| A8 | FIR → none | 正则化整体贡献 |
| A9 | FuzzyGCN → StandardGCN | 模糊图卷积 vs 标准图卷积 |
