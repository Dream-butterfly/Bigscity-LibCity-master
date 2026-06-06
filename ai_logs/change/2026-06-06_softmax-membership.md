# 2026-06-06: μ_low / μ_feat 从 sigmoid 改为 softmax

## 变更文件
- `GNNTP/models/new/final_3_type2/graph.py` — `_compute_memberships()`

## 原因
审计 M7 (epoch26, τ≈1.0) 确认：top1_mean=0.186, H_normalized=0.967。
sigmoid 独立激活 → 8 个 fuzzy set 之间零竞争 → 模型"最省力解"是全停在 0.5 附近。
sharpness loss 和温度退火都逆着参数化方式工作，边际收益极低。

根因不是 weight 不够大，而是 **membership 参数化本身不产生集合间竞争**。

## 改动

### graph.py `_compute_memberships` (3 行替换)

```diff
- mu_low = torch.sigmoid(self.base_membership_lower / tau)
+ mu_low = F.softmax(self.base_membership_lower / tau, dim=-1)

- mu_feat = torch.sigmoid(self.feature_to_membership(node_feat) / tau)
+ mu_feat = F.softmax(self.feature_to_membership(node_feat) / tau, dim=-1)

  # μ_delta 保留 sigmoid — interval width 是独立属性，不应参与竞争
  mu_delta = torch.sigmoid(self.base_membership_delta / tau)
```

### 保留不变
- `mu_delta`: sigmoid — 每维独立的 uncertainty interval (FOU)
- `mu_high = mu_low + mu_delta*(1 - mu_low)`: Type-2 区间结构完整
- 温度退火: τ 1.0→0.3，现在对 softmax 有实质作用
- sharpness/diversity loss: 继续工作，但压力大大降低

## 预期效果

| 指标 | sigmoid (epoch26) | softmax 预期 (τ=1.0) |
|------|:---:|:---:|
| M7 top1_mean | 0.186 | 0.30-0.45 |
| M1 H_normalized | 0.967 | 0.6-0.8 |
| M2 mean_cosine | 0.888 | 0.4-0.7 |
| M5 R_density@0.3 | 1.00 | 0.4-0.7 |

τ 退火到 0.3 后，top1_mean 预计可到 0.6-0.8。

## 注意
- init 不变 (trunc_normal std=1.2): softmax 下仍能产生足够初始多样性
- 现有 checkpoint 不兼容（参数语义变了，需重新训练）
- 温度退火策略可能需要微调：softmax 对 τ 更敏感
