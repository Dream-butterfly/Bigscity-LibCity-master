# 2026-06-06: Sharpness loss 改为 Gini impurity

## 变更文件
- `GNNTP/models/new/final_3_type2/model.py`
- `GNNTP/models/new/final_3_type2/config.json`

## 原因
训练日志分析发现 Shannon entropy 版本的 sharpness loss 梯度太弱：
- sharp_loss 从 0.0378 仅降到 0.0367（-2.9%）
- μ_std 卡在 ~0.10，离理论上限 0.33 还很远（~30%）
- Gini impurity 在 p≈0.5 时梯度最大，正好命中当前 μ 分布的 [0.15, 0.35] 区间

## 改动
1. `model.py` `__init__`: 新增 `membership_sharpness_mode` 配置读取（默认 "gini"）
2. `model.py` `_membership_sharpness_loss`: 增加 mode 分支
   - `"gini"`: `1.0 - (p**2).sum(-1).mean()` — Gini impurity
   - `"entropy"`: `-(p * (p+1e-8).log()).sum(-1).mean()` — 原实现（保留）
3. `config.json`: 新增 `"membership_sharpness_mode": "gini"`

## 注意
- Gini 梯度更强，weight=0.02 可能需要下调到 0.005~0.01
- 建议首轮不调 weight，观察 sharp_loss 下降幅度后再决定
- 诊断日志 (mu_std, sharp_loss, div_loss) 不受影响
