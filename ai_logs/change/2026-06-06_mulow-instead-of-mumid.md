# 2026-06-06: 正则化 + 审计改用 μ_low 替代 μ_mid

## 变更文件
- `GNNTP/models/new/final_3_type2/model.py` — sharpness/diversity loss
- `scripts/tools/audit_final3_type2_final.py` — M1/M2/M4/M6/M7

## 原因
`μ_high = μ_low + μ_delta·(1-μ_low)` 中 `(1-μ_low)` 是天然均衡器：
- 小的 μ_low → 大的 (1-μ_low) → delta 给更多补充
- 软max 拉开的差距被 delta 补回去
- 归一化 mu_mid 后 top1 始终卡在 ~0.21

## 修复
所有正则化和审计指标从 `mu_mid` 改为 `mu_low`（softmax 输出）：
- 损失函数: sharpness/diversity 直接用 softmax simplex，无需归一化
- 审计 M1/M2/M6/M7: 同上
- 审计 M4 argmax: 改用 mu_low 分配节点到集合
- 模糊关系 (max-min) 仍用 mu_mid（Type-2 语义保留）

## 预期效果
- mu_low 是真正的 softmax 竞争输出 → 正则化直接生效
- M7 top1_mean 预计从 0.21 跳到 0.35-0.55（取决于训练程度）
- M1 H_normalized 预计从 0.97 降到 0.5-0.7
