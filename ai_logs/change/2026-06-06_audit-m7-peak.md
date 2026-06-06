# 2026-06-06: 审计脚本新增 M7 Membership Peak 指标

## 变更文件
- `scripts/tools/audit_final3_type2_final.py`

## 原因
M6 (Set Utilization) 显示所有 8 个集合都在使用（utilization_entropy ≈ 1.0），但 M1/M2 显示 membership 仍然接近均匀。这个看似矛盾的现象需要一个新指标来诊断：
- M7 测量每个节点的 top-1 membership 值及其分布
- 回答核心问题："8个集合都在用，但每个节点是否同时属于所有集合？"

## 新增 M7: Membership Peak Analysis
- `top1_mean/std/min/max`: 每节点最大隶属度的统计量
  - ≈0.125 → 完全均匀（K=8 baseline）
  - ≈0.25 → 薄层分布（当前疑似状态）
  - >0.70 → 真正的 fuzzy clustering
- `top1_histogram`: 10-bin 直方图
- `argmax_counts`: 每集合有多少节点以它为 top-1
- `argmax_gini_normalized`: argmax 分布的 Gini 系数
- `per_set`: 每个集合的节点数、占比、平均 top1

## 联合解读
如果 M7 显示 argmax 均匀分布 + top1_mean ≈ 0.25：
→ 确认"所有集合都在用，但没有节点真正 committed"（解释了 M6 和 M1/M2 的表面矛盾）
