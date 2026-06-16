# scripts/analysis/INFO.md

训练后分析脚本，用于从已完成训练的 checkpoint 中提取论文所需的数据和可视化素材。

## 关键脚本

| 文件 | 作用 | 对应模型 |
|------|------|----------|
| `export_final2_diagnostics.py` | 从 final_T2 checkpoint 导出诊断数据 | `GNNTP/models/new/final_T2/` |

## `export_final2_diagnostics.py`

### 用途

从训练完成的 final_T2 (MVF-STGFormer) 最佳 checkpoint 加载模型，在测试集上跑一次前向传播，导出所有论文制图所需的空间诊断数据。

### 用法

```bash
uv run scripts/analysis/export_final2_diagnostics.py \
    --run_id 20260615_120000__traffic_state_pred__final_T2__METR_LA \
    --epoch 42
```

参数：
- `--run_id`：训练运行目录名（`outputs/<run_id>/`）
- `--epoch`：要加载的 checkpoint epoch
- `--task`（可选）：任务名，默认从 run_meta 读取
- `--model`（可选）：模型名，默认 `final_T2`
- `--dataset`（可选）：数据集名，默认从 run_meta 读取
- `--artifact_id`（可选）：数据工件 ID，默认从 run_meta 读取

### 输出

#### `outputs/<run_id>/evaluate_cache/diagnostics.npz`

空间诊断数据，包含以下数组：

| 数组 | 形状 | 论文用途 |
|---|---|---|
| `fou_mean` / `fou_std` | `[N]` | MDI 空间分布（热力图） |
| `beta_per_node` | `[N, 3]` | 逐节点视角偏好空间分布 |
| `beta_global` | `[3]` | 全局视角混合系数 |
| `R_low` / `R_mid` / `R_high` | `[N, N]` | 三视角原始模糊关系图（核心卖点可视化） |
| `R_final_mean` | `[N, N]` | 聚合后的最终模糊图 |
| `R_final_snapshot` | `[N, N]` | 混合后的关系图快照 |
| `mu_upper` / `mu_lower` / `mu_mid` | `[N, K]` | 隶属度包络 |
| `mu_low_raw` / `mu_mid_raw` / `mu_high_raw` | `[N, K]` | 三视角原始隶属度 |
| `proto_low` / `proto_mid` / `proto_high` | `[K, D]` | 三视角原型中心 |
| `sigma_low` / `sigma_mid` / `sigma_high` | `[K]` | 高斯宽度参数 |
| `tau_low` / `tau_mid` / `tau_high` | `[K]` 或 scalar | 温度参数 |
| `blend_alpha` | scalar | 动静图融合比例 |
| `predictions` / `truth` | `[N, T', N, O]` | 预测值与真值 |
| `R_corr_lm/lh/mh` | scalar | 三视角关系图相关性 |
| `R_gap` | scalar | 归一化关系图差距 |
| `R_diff_lm/hm` | scalar | 关系图 L1 差异 |
| `view_dist_lm/lh/mh` | scalar | 三视角隐向量距离 |
| `adapter_ratio` | scalar | Δ/shared 范数比 |
| `mu_diff_mean/max` | scalar | 包络宽度统计 |
| `d2_mean/std` | scalar | 距离场统计 |
| `eff_width` | scalar | 有效包络宽度 |
| `mu_raw_mean` | `[3]` | 三视角平均隶属度 |

#### `outputs/<run_id>/evaluate_cache/diagnostics_curves.csv`

从 `run.log` 自动解析的收敛曲线数据，分组存储：

| 曲线组 | 内容 |
|---|---|
| `epoch` | train_loss, val_loss, lr per epoch |
| `t2_beta` | β_L, β_M, β_H, H(β) per epoch |
| `fou_stats` | MDI 均值/标准差演变 |
| `t2_sigma` | σ_L/σ_M/σ_H, δ, τ_L/τ_M/τ_H, σ_ratio |
| `r_corr` | R 相关性 ρ(L,M), ρ(L,H), ρ(M,H) |
| `losses` | 10项损失分解（mae, ent, gap, fou, fce, consv, pn, ln, dv, dd） |
| `grads` | ∇β, ∇σ_low, ∇δ, ∇proto |
| `r_gap` | R_gap, eff_width |
| `gcn_view` | Δ/sh, GCN energy, ∇z(LM,LH,MH) |

### 设计要点

- **不需要训练中途保存额外数据**：所有空间矩阵/向量数据可从 checkpoint + 一次测试集前向恢复
- **三视角原始关系图**：脚本直接调用 `_compute_memberships()` + `_build_fuzzy_relation()` 获取未混合的 `R_low/R_mid/R_high`，用于论文核心三视角对比图
- **收敛曲线**：从 `run.log` 文本日志自动解析，无需额外配置
- **其他模型不需要此脚本**：标准 evaluator 输出的 csv + npz 已包含所有评估指标

### 与其他脚本的关系

- 依赖已完成训练的 checkpoint（由 `run_train_artifact.py` 产生）
- 依赖数据工件（由 `run_data_artifact.py` 产生）
- 不修改 checkpoint、不重新训练
- 输出供论文制图脚本（matplotlib/seaborn）直接使用
