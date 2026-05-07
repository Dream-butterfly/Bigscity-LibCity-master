# 更改 18：new_diffusion_fuzzy 训练修复 — 配置变更

**日期**: 2026-05-07
**类型**: 修复问题 / 实验流程变更
**触发**: 训练无效 & 预测散乱诊断（分析 11）
**安全级别**: 🟡 确认区（仅 config.json 变更）

## 变更摘要

4 项配置修改，修复 new_diffusion_fuzzy 模型不学习、预测无意义的问题。

## 修改文件

| 文件 | 变更 |
|------|------|
| `GNNTP/models/new/new_diffusion_fuzzy/config.json` | 4 项参数修改 |

## 具体变更

### 1. 开启时间特征（解决条件信号极弱）
```diff
- "add_time_in_day": false,
- "add_day_in_week": false,
+ "add_time_in_day": true,
+ "add_day_in_week": true,
```
- `feature_dim` 从 1 → 3（速度 + 时刻编码 + 星期编码）
- 编码器获得有意义的多维输入，交叉注意力能区分不同时间段

### 2. AMP 改用 bfloat16（解决 GradScaler 溢出）
```diff
- "amp_dtype": "float16",
+ "amp_dtype": "bfloat16",
```
- bfloat16 指数位宽与 float32 相同，不会溢出
- 无需 GradScaler，消除缩放振荡

### 3. 采样改用 DDIM（修复非连续步长数学错误）
```diff
- "sampling_method": "ddpm",
+ "sampling_method": "ddim",
```
- DDIM 天然支持任意步长跳跃，50 步采样数学正确
- 配合 `ddim_eta: 0.0` 做确定性采样

### 4. 关闭守恒损失（消除训练期干扰）
```diff
- "physics_loss_weight": 0.05,
+ "physics_loss_weight": 0.0,
```
- 等扩散 loss 稳定下降后再逐步恢复

## 预期效果

- 训练 loss 从 ~1.0 开始下降（目标 < 0.5）
- 预测不再呈无意义正态分布
- 验证 MAE/RMSE 应接近同类模型水平
