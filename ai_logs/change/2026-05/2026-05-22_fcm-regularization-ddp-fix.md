# FCM 正则化 — 修复 new_fuzzy_3 DDP 未用参数崩溃

**日期**: 2026-05-22
**类型**: 修复 + 新增功能
**模型**: new_fuzzy_3
**影响文件**: `GNNTP/models/new/new_fuzzy_3/{graph.py, model.py, config.json}`

## 问题

DDP 4 卡训练在第一轮就崩溃，所有 rank 报相同错误：
```
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. Parameter indices which did not receive grad
for rank N: 1
```

参数 index=1 = `module.fuzzy_graph.fuzzy_prototypes` (shape [3, 64])。

根因：`FuzzyRelationalGraphLearner.fuzzy_prototypes` 被定义为 `nn.Parameter`，但从未参与任何前向计算。`get_prototypes()` 只是访问器，不接入计算图。DDP 的 `find_unused_parameters=False` 要求所有参数都必须贡献于 loss，否则第二次 forward 时 bucket 重建失败。

单卡不受影响（无 DDP 梯度同步检查）。

## 修改

### 1. graph.py — 新增 `fcm_loss()` 方法

标准 Fuzzy C-Means 目标函数：

$$J_{FCM} = \frac{1}{N \cdot K} \sum_{i=1}^{N}\sum_{k=1}^{K} \mu_{ik}^2 \cdot ||\mathbf{h}_i - \mathbf{v}_k||^2$$

- 聚合节点特征到 [N, D]（跨 batch/time mean）
- 复用 `_compute_memberships()` 获取隶属度 μ [N, K]
- 计算投影特征 h_i 与原型 v_k 的欧式距离平方
- mean 归约（与 batch_size 无关）

### 2. model.py — 集成到 loss 计算

**`__init__`**: 读取 `fcm_loss_weight`(0.01)、`fcm_warmup_epochs`(3)、`fcm_steps_per_epoch`(80)

**`calculate_loss()`**: 
```python
total_loss = main_loss  # diffusion SNR-MSE 或 deterministic L1
if fcm weight active:
    total_loss += effective_fcm_weight * self.fuzzy_graph.fcm_loss(history_sequence)
```

**`_get_effective_fcm_weight()`**: 与 `_get_effective_conservation_weight()` 对称的 warmup 线性 ramp-in

### 3. config.json — 新增默认参数

```json
"fcm_loss_weight": 0.01,
"fcm_warmup_epochs": 3,
"fcm_steps_per_epoch": 80
```

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| FCM 位置 | calculate_loss 顶层 | 与 conservation 对称，两种模式均生效 |
| 归约 | mean() | 归一化到 per-(node, set)，跨 batch 稳定 |
| fuzziness | m=2 硬编码 | 标准 FCM，不引入额外超参 |
| warmup | 3 epochs | 让主 loss 先稳定，FCM 再介入 |
| λ_fcm | 0.01 | 远小于 λ_cons(0.1)，避免喧宾夺主 |
| 特征聚合 | mean(dim=(0,1)) | 跨 batch 和时间平均 → 稳定 |
| node_features=None | 返回 0 | 纯 base_membership 模式无特征可用 |

## 潜在风险

- **隶属度坍缩**：FCM 倾向硬聚类 (μ→0/1)，小权重 + warmup 缓解
- **与 conservation 冲突**：λ_fcm << λ_cons，且两者作用在不同对象上（图结构 vs 物理约束）
