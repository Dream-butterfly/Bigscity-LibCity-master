# final_T2: Interval Type-2 Fuzzy Graph Transformer for Traffic Prediction

## Overview

final_T2 is a spatiotemporal traffic forecasting model that replaces the fixed graph convolution
of conventional approaches with a **learnable Interval Type-2 Fuzzy Relational Graph**. The model
processes historical traffic states `X ∈ R^(B×T_in×N×F)` and predicts future states 
`ŷ ∈ R^(B×T_out×N×1)`.

**Key innovation**: Instead of a fixed adjacency matrix, the model learns *K* fuzzy prototypes in
the node embedding space. Each node's membership to these prototypes forms a soft fuzzy relation
`R ∈ R^(N×N)`. The Type-2 extension introduces **uncertain standard deviation** on each prototype's
Gaussian membership function, producing three relational views (pessimistic, midpoint, optimistic)
that are mixed via a learnable router.

**Parameter efficiency**: ~100K parameters vs ~300K for STGCN at comparable accuracy.

```
Input [B,T_in,N,F]
  │
  ├─► Fuzzy Graph Learner ─► [N,N] graph ─┐
  │                                         │
  ├─► Encoder (2× blocks) ◄────────────────┘
  │     │
  │     ├─ Temporal Self-Attention
  │     ├─ Fuzzy Graph Convolution (R^1, R^2 @ X)
  │     ├─ Fuzzy Cell Attention (residual)
  │     └─ Feed-Forward Network
  │
  └─► Decoder (1× block) ◄─── encoded features
        │
        ├─ Temporal Self-Attention (on learnable queries)
        ├─ Fuzzy Graph Convolution
        ├─ Cross-Attention (to encoder output)
        ├─ Fuzzy Cell Attention (residual)
        └─ Feed-Forward Network
              │
              └─► Output [B,T_out,N,1]
```

---

## 1. Fuzzy Graph Learner (`FuzzyRelationalGraphLearner`)

*File: `graph.py`, line 118*

### 1.1 Learnable Parameters

| Parameter | Shape | Role |
|-----------|-------|------|
| `prototype_center` | [K, D] | *K* fuzzy prototypes in *D*-dimensional hidden space |
| `log_sigma_low` | [K] | Per-set lower Gaussian width (positive via `softplus`) |
| `log_sigma_high` | [K] | Per-set upper Gaussian width (positive via `softplus`) |
| `node_transform` | D→D | 2-layer MLP mapping raw node features to prototype space |
| `blend_logit` | [1] | Fuzzy/static graph mixture weight, `sigmoid(blend_logit)` |
| `relation_mix_logits` | [3] | β router: weight distribution over [R_low, R_mid, R_high] |

Total: ~5K parameters (for K=8, D=48).

### 1.2 Membership Computation

For each node *i* and fuzzy set *k*:

1. Extract node representation: `node_latent = node_transform(raw_projection(history.mean_time))` → [N, D]
2. Compute squared distances: `d²_ik = ||node_latent_i − prototype_k||²` → [N, K]
3. **Genuine Interval Type-2** via independent dual widths:

```
σ_low  = softplus(log_sigma_low_k) + 1e-3      # ← independent, per-set
σ_high = softplus(log_sigma_high_k) + 1e-3     # ← independent, per-set
σ_low  = min(σ_low, σ_high)                     # enforce ordering
σ_high = max(σ_low, σ_high)

μ_lower_raw = exp(−d² / 2σ_high²)              # wider  → more connections, lower confidence
μ_upper_raw = exp(−d² / 2σ_low²)               # narrower → fewer connections, higher confidence

μ_upper = max(μ_lower_raw, μ_upper_raw)        # upper envelope
μ_lower = min(μ_lower_raw, μ_upper_raw)        # lower envelope
μ_mid   = (μ_lower + μ_upper) / 2              # midpoint
```

Each fuzzy set has **independently learned** lower and upper Gaussian widths. Unlike the
previous version which used a symmetric perturbation `σ·(1±r)`, the two widths are free
to learn different scales from data. A set may have σ_low=1.8 (sharp) and σ_high=8.3 (diffuse),
or σ_low=4.0 and σ_high=4.2 (nearly Type-1).

**Degeneration**: If `σ_low → σ_high` for all sets, then `μ_lower → μ_upper` and the
Interval Type-2 collapses to Type-1. This is not a training failure — it is the model
autonomously determining that interval uncertainty provides no incremental value for the
prediction task.

### 1.3 Fuzzy Relation Construction

The fuzzy relation uses **max-min composition** (standard in fuzzy set theory):

```
R_ij = max_k min(μ_i(k), μ_j(k))    ∈ [0, 1]
```

This is a tolerance relation: two nodes are related if they share membership in at least one
fuzzy set. Self-loops are enforced (`R_ii = 1`).

Three relations are constructed:
- `R_low`  from `μ_lower`  → pessimistic (wider Gaussian, more connections, lower confidence)
- `R_mid`  from `μ_mid`    → midpoint (optimal estimate)
- `R_high` from `μ_upper`  → optimistic (narrower Gaussian, fewer but higher-confidence connections)

### 1.4 Graph Assembly

```
β      = softmax(relation_mix_logits)           # [3], sum=1
R_mixed = β₀·R_low + β₁·R_mid + β₂·R_high      # interval-valued propagation

# Optional: blend with static road network
blend   = sigmoid(blend_logit)
R_final = max(R_mixed · blend,  R_static · (1−blend))
```

The final graph `R_final` is used for all Graph Convolution operations.

### 1.5 Power Precomputation

Before encoder/decoder forward, k-hop relational powers are computed once and shared across all blocks:

```
R⁰ = I,  R¹ = R_final,  R² = R_final ∘ R_final   (max-min composition)
```

For large graphs (N > 500), optional top-K sparsification reduces the compose from O(N³) to O(N²·K).

---

## 2. Encoder (`STEncoder` → `STEncoderBlock`)

*Files: `encoder.py`*

Each encoder block has four sub-layers with Pre-LN residuals:

### 2.1 Temporal Self-Attention

Per-node multi-head attention over the time dimension:
```
X_flat = [B×N, T, D]              # flatten batch and node dims
Q, K, V = Linear(X_flat)           # separate projections per head
attn = FlashAttention(Q, K, V)     # F.scaled_dot_product_attention
X ← X + Dropout(attn_out)
X ← LayerNorm(X)
```

Each node attends to its own T=12 historical timesteps.

### 2.2 Fuzzy Graph Convolution

K-hop graph propagation using the shared fuzzy relation:
```
g_in  = [B×T, N, D]                # flatten batch and time
out   = proj₀(g_in)                # R⁰ projection
out  += proj₁(R¹ @ g_in_flat)      # 1-hop propagation
out  += proj₂(R² @ g_in_flat)      # 2-hop propagation
g_out = out.reshape(B, T, N, D)
X ← X + Dropout(g_out)
X ← LayerNorm(X)
```

Power matrices R¹, R² are precomputed and shared across all blocks (only one O(N³) compose per forward, not per block).

### 2.3 Fuzzy Cell Attention (residual)

Global spatial interaction via learnable cell prototypes:
```
node_repr = X.mean(dim=(B,T))      # [N, D] per-node average
u = softmax(−cdist(node_repr, centers))  # [N, K_c] soft prototype assignment
affinity = u_norm @ u_norm.T        # [N, N] cell-based similarity
cell_out = affinity @ transform(node_repr)  # [N, D] global spatial features
blend = sigmoid(cell_blend)         # learnable residual weight
X ← X + blend · cell_out           # residual connection
X ← LayerNorm(X)
```

### 2.4 Feed-Forward Network
```
X ← X + Dropout( Linear₂(GELU(Linear₁(X))) )
X ← LayerNorm(X)
```

---

## 3. Decoder (`FutureDecoder` → `DecoderBlock`)

*Files: `decoder.py`*

The decoder generates future predictions from learnable query embeddings.
The queries are factorized: `horizon_embed[T, 1, D] + node_embed[1, N, D]` → [T, N, D].

Each decoder block has five sub-layers:

1. **Temporal Self-Attention** — same as encoder, on learnable queries
2. **Fuzzy Graph Convolution** — same as encoder
3. **Fuzzy Cell Attention** — same as encoder (residual)
4. **Cross-Attention** — queries attend to encoder output:
   ```
   Q from queries [B×N, T_out, D],  K,V from encoder_out [B×N, T_in, D]
   ```
5. **Feed-Forward Network** — same as encoder

After all blocks: `output = LayerNorm(queries) → Linear₁`

---

## 4. Training

### 4.1 Loss Function
```
L = L1(pred, target)                                    # primary
  + λ_cons · conservation_loss(pred, R)                 # optional
  + λ_explore · (H(β) + relu(1 − σ.std))               # optional exploration
```

### 4.2 Conservation Loss (Fuzzy Interaction Regularization)
Encourages traffic flow consistency along fuzzy relations:
```
flow_pressure = max(congestion + R − 1, 0)              # fuzzy implication
residual = temporal_delta − net_pressure                 # violation
L_cons = mean(residual²)
```

### 4.3 Diagnostics
Per-epoch logging captures: β distribution, σ/r statistics, FOU width, gradient norms,
R_gap (interval validity), graph sparsity, and membership quality.

---

## 5. Key Mathematical Properties

### Degeneration Theorem
```
r_k → 0  ⇒  σ_low → σ_high → σ_k
         ⇒  μ_lower → μ_mid → μ_upper
         ⇒  R_low → R_mid → R_high
         ⇒  Interval Type-2 collapses to Type-1
```

### Parameter Efficiency
The prototype-based parameterization requires only O(K·D) parameters for K fuzzy sets,
regardless of graph size N. The fuzzy relation R ∈ R^(N×N) has zero learnable parameters
(it is computed from memberships). This is unlike GCN approaches that require O(N²) or
O(N·D) adjacency parameters.

### Scale Invariance
σ_k is learned per fuzzy set from data. Different fuzzy sets naturally develop
different widths (σ_std ≈ 1.0 observed empirically), creating a multi-scale
fuzzy partition without manual tuning.

---

## 6. Configuration Reference

*File: `config.json`*

| Key | Default | Description |
|-----|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension D |
| `fuzzy_num_sets` | 3 | Number of fuzzy prototypes K |
| `num_cells` | 8 | Number of cell attention prototypes K_c |
| `encoder_layers` | 2 | Encoder blocks |
| `decoder_layers` | 2 | Decoder blocks |
| `graph_k_hop` | 2 | Graph convolution hops |
| `graph_topk` | 32 | Top-K sparsification (None=disable) |
| `graph_closure_steps` | 0 | Semantic closure depth (0=disable) |
| `cell_blend_init` | 0.3 | Cell attention residual init |
| `blend_logit_init` | — | Fuzzy/static graph mix init (hardcoded 0.5) |
| `beta_init_random` | false | Randomize β init (diagnostic) |
| `use_spatiotemporal_attention` | true | Decoder global spatiotemporal attn |
| `use_torch_compile` | false | Enable torch.compile |
| `t2_explore_weight` | 0.0 | Type-2 exploration loss weight |
| `t2_lr_boost` | 1.0 | Type-2 gradient multiplier |
