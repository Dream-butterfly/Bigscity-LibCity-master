# 2026-06-06: Membership 温度退火 (Temperature Annealing)

## 变更文件
- `GNNTP/models/new/final_3_type2/graph.py`
- `GNNTP/models/new/final_3_type2/model.py`
- `GNNTP/models/new/final_3_type2/config.json`

## 原因
审计 M7 确认 top1_mean=0.188（K=8, baseline=0.125），M1 H_normalized=0.965。
sigmoid 独立激活导致模型无需在 K 个 fuzzy set 之间做选择——"最省力解"就是全部停在 0.5 附近。
sharpness loss 逆着参数化方式工作，调 weight 杯水车薪。

## 方案
sigmoid 加温度退火：`σ(θ/τ)`，τ 从 1.0 线性衰减到 0.3。
- τ=1.0 → 等同当前行为（向后兼容）
- τ→0.3 → sigmoid 输出被推向 0/1 两端
- 保持每维独立激活的模糊逻辑语义
- 不破坏 Type-2 FOU

## 具体改动

### graph.py
- `__init__`: 新增 `self.membership_temperature = 1.0`
- `_compute_memberships`: `torch.sigmoid(θ) → torch.sigmoid(θ / τ)`（3 处：mu_low, mu_delta, mu_feat）
- 新增 `set_temperature(tau)` 方法

### model.py
- `__init__`: 读取 `membership_temperature`(1.0), `membership_temperature_min`(0.3), `membership_temperature_anneal_steps`(8000)
- `forward`: 每步调用 `_anneal_membership_temperature()`
- 新增 `_anneal_membership_temperature()`: 线性衰减 τ
- 诊断日志: 新增 `mem_temp` 字段

### config.json
- 新增 `"membership_temperature": 1.0`
- 新增 `"membership_temperature_min": 0.3`
- 新增 `"membership_temperature_anneal_steps": 8000`

## 注意
- 8000 steps ≈ batch_size=64 时约 15 epoch（每 epoch ~550 steps）
- 如需更快退火，减 `membership_temperature_anneal_steps`
- τ_min=0.3 是保守值，可探索 0.1~0.2
