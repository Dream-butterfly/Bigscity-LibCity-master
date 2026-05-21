# Phase C 扩散集成 — 实施方案

> 日期: 2026-05-21 | 目标: 在 new_fuzzy_3 上集成 DDIM 扩散，同时保持确定性路径可切换

---

## 决策 1：双路径 vs 独立模型

**结论：双路径，一个模型，config 切换。**

| 方案 | 描述 | 消融友好 | 代码量 |
|------|------|:---:|:---:|
| A | new_fuzzy_3 只做扩散，保留 new_fuzzy_2 做确定性对比 | ✅ 需两个 codebase | 多 |
| B | new_fuzzy_3 内双路径，config 开关控制 | ✅✅ 同 codebase | 少 |

**选 B。** `use_diffusion=true/false` 一个参数控制。

```
new_fuzzy_3/model.py

  use_diffusion = true
    ┌────────────────────────────────────────┐
    │ Encoder → H                             │
    │ FuzzyGraph → R                          │
    │                                         │
    │ Train: Y_0 → +ε → Y_t                   │
    │        Denoiser(Y_t, t, H, R) → ε_θ     │
    │        Loss = SNR-MSE + Conservation    │
    │                                         │
    │ Predict: DDIM(Y_T→...→Y_0)             │
    └────────────────────────────────────────┘

  use_diffusion = false   (等同于 new_fuzzy_2)
    ┌────────────────────────────────────────┐
    │ Encoder → H                             │
    │ FuzzyGraph → R                          │
    │                                         │
    │ FutureDecoder(H, R) → prediction        │
    │ Loss = L1 + Conservation               │
    └────────────────────────────────────────┘
```

---

## 决策 2：消融配置矩阵

`new_fuzzy_3/config.json` 新增：

```json
{
  "use_diffusion": true,
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

消融矩阵（不需要改代码，只改 config / CLI 参数）：

| 实验 | `use_diffusion` | `use_fuzzy_graph` | `use_fuzzy_conservation` | 对应 |
|------|:---:|:---:|:---:|------|
| A | true | true | true | **FuzDiff 完整模型** |
| B | true | true | false | 扩散 + 模糊图，无守恒 |
| C | true | false | true | 扩散 + 标准GCN + 守恒 |
| D | true | false | false | 纯扩散 + 标准GCN |
| E | false | true | true | 确定性 + 模糊图（= new_fuzzy_2） |
| F | false | true | false | 确定性 + 模糊图，无守恒 |
| G | false | false | true | 确定性 + 标准GCN + 守恒 |
| H | false | false | false | 纯确定性 + 标准GCN（baseline） |

**关键消融对比**：
- A vs E：**扩散的增量价值**（同一模糊图下）
- A vs C：**模糊图的增量价值**（同扩散下）
- A vs B：**守恒损失的增量价值**（同模糊图+扩散下）

---

## 决策 3：各模块职责

```
new_fuzzy_3/
├── __init__.py       → export NewFuzzy3
├── model.py          → NewFuzzy3（双路径调度）
├── encoder.py        → 不改（复用）
├── decoder.py        → 保留（use_diffusion=false 路径）
├── graph.py          → 不改（复用 FuzzyRelationalGraphLearner + FuzzyGraphConvolution）
├── attention.py      → 不改（复用）
├── diffusion.py      → 新建：DiffusionScheduler（从 old diffusion 复制）
├── denoiser.py       → 新建：FuzzyGuidedDenoiser（改编 old denoiser，用模糊图）
├── executor.py       → 新建：DiffusionTrafficStateExecutor（从 old 复制）
├── embedding.py      → 更新：加 SinusoidalTimeEmbedding（diffusion 需要）
├── config.json       → 加 diffusion 参数
├── manifest.json     → model="new_fuzzy_3", executor="DiffusionTrafficStateExecutor"
├── executor.json     → 适配扩散训练参数
└── utils/            → 不改（复用）
```

**模型初始化分支**：

```
NewFuzzy3.__init__
  │
  ├─ Encoder（始终创建）
  ├─ FuzzyRelationalGraphLearner（use_fuzzy_graph 控制）
  │
  ├─ if use_diffusion:
  │    ├─ DiffusionScheduler
  │    └─ FuzzyGuidedDenoiser（替代 FutureDecoder）
  │
  └─ else:
       └─ FutureDecoder（确定性路径）
```

---

## 决策 4：Executor 选择

**结论：始终用 `DiffusionTrafficStateExecutor`。**

理由：
- 确定性路径下：executor 的 `_train_epoch` 调用 `model(batch)` → `forward()` → `calculate_loss()`，与 `TrafficStateExecutor` 行为一致
- 扩散路径下：必须用 `model(batch)` 触发 DDP hook（`TrafficStateExecutor` 直接调用 `calculate_loss` 会绕过 DDP）
- 一个 executor，两种路径都工作

`manifest.json`：
```json
"executor": "DiffusionTrafficStateExecutor",
"executor_entry": "GNNTP.models.new.new_fuzzy_3.executor:DiffusionTrafficStateExecutor"
```

---

## 决策 5：扩散训练细节

### 编码器冻结策略

```
epochs 1-5: encoder lr = base_lr * 0.1   （微调）
epochs 6+:  encoder lr = base_lr          （正常训练）
```

原因：old diffusion 的教训——编码器在扩散训练初期收到混乱梯度（去噪器还没学会），不加约束会退化。

### 守恒损失仅在低噪声步计算

```python
# 仅对 t < 0.7 * T 的样本计算（ᾱ_t 足够大，单步重建有意义）
mask = t < int(0.7 * self.diffusion_steps)
```

原因：高噪声步（t 大）时 ᾱ_t 小，单步重建 `(Y_t - √(1-ᾱ)*ε_θ)/√ᾱ` 放大误差，守恒损失惩罚噪声而非模型偏差。

### AMP dtype

```python
# 默认 bfloat16（避免 float16 溢出），可回退到 float16
amp_dtype = config.get("amp_dtype", "bfloat16")
```

---

## 决策 6：Denoiser 与 Decoder 的共享结构

`FuzzyGuidedDenoiser` 和 `FutureDecoder` 结构高度相似，可以共享 `DecoderBlock` 的部分逻辑，但不合并——保持干净的分支。

| 组件 | FutureDecoder | FuzzyGuidedDenoiser |
|------|:---:|:---:|
| 输入 | learnable queries [1,T,N,D] | Y_t [B,T,N,C] |
| 时间位置嵌入 | 无 | temporal_position_emb [1,T,1,D] |
| 条件注入 | Cross-attention only | **Concat+Linear** + Cross-attention |
| 图卷积 | FuzzyGraphConvolution(R) | FuzzyGraphConvolution(R) |
| 时间步注入 | 无 | **Per-block scale/shift** |
| U-Net skip | 无 | **前半存，后半加** |
| 输出投影 | Linear(D→C) | Linear(D→C) → ε_θ |

---

## 决策 7：实施顺序

```
Step 1 ─ 复制 diffusion.py、executor.py（5分钟）
  │   从 new_diffusion_fuzzy_2/ 直接复制，不改代码
  │
Step 2 ─ 更新 embedding.py（5分钟）
  │   加 SinusoidalTimeEmbedding（从 old 复制）
  │
Step 3 ─ 编写 denoiser.py（核心，1-2小时）
  │   改编 AttentionDenoiser → FuzzyGuidedDenoiser
  │   - GraphConvolution → FuzzyGraphConvolution
  │   - AdaptiveGraphLearner 调用 → 接收预计算的 R
  │   - 保留: concat+linear fusion, blend gate, per-block time, U-Net
  │
Step 4 ─ 修改 model.py（1-2小时）
  │   - 类名 NewFuzzy2 → NewFuzzy3
  │   - __init__: 双路径（use_diffusion 分支）
  │   - calculate_loss: 双路径
  │   - predict: 双路径（DDIM 采样 vs 单次前向）
  │   - _ddp_loss_through_forward = True (always)
  │
Step 5 ─ 更新配置（10分钟）
  │   - __init__.py → NewFuzzy3
  │   - manifest.json → new_fuzzy_3, DiffusionTrafficStateExecutor
  │   - config.json → 加 diffusion params
  │   - executor.json → 适配
  │
Step 6 ─ 冒烟测试（15分钟）
  │   - 导入测试
  │   - 确定性路径 forward/predict
  │   - 扩散路径 forward/predict
  │   - use_diffusion=false 时结果应与 new_fuzzy_2 一致
  │
Step 7 ─ 小数据训练验证（30分钟）
      - 1 epoch, batch=4, PEMSD4, 确认 loss 下降
      - 确认无 OOM（N=307, max-min compose 已在 [N,N] 上）
```

---

## 总结

| 决策 | 结论 |
|------|------|
| 独立模型还是双路径？ | **双路径**，config `use_diffusion` 切换 |
| Executor 用哪个？ | **始终 `DiffusionTrafficStateExecutor`** |
| Denoiser vs Decoder 共存？ | **共存**，config 控制哪个生效 |
| 消融怎么做？ | **不改代码**，只改 config/CLI 参数 |
| 旧 FutureDecoder 删不删？ | **保留**（确定性路径需要） |
| 编码器要 frozen 吗？ | **前 5 epoch lr×0.1**，之后正常 |
