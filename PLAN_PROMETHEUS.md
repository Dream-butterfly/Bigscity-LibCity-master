# 普罗米修斯计划 v2：FuzDiff (new_diffusion_fuzzy) 迭代优化

> 修订: 2026-04-28 | 背景更新：
> - `new_diffusion` / `new_diffusion_2` 是 git 前手写版本迭代产物，`new_diffusion_fuzzy` 是当前唯一主线
> - 数据预处理需完全与模型训练解耦，按 dataset 类型链接
> - 不需要单元测试模块

## 架构变更概览

```
当前：run_model.py ──► ConfigParser(task, model, dataset) ──► data + model + train 一体
                    run_data_artifact.py 同样依赖 --model 读取 config

目标：run_data_prep.py ──► 纯 dataset 驱动（不需要 --model）
      run_train.py      ──► 按 dataset + dataset_class 自动匹配 artifact
```

核心改动：**ConfigParser 不再要求 `model` 参数用于数据预处理**。数据预处理只需 `task + dataset + dataset_class + 数据参数`，模型配置仅训练阶段需要。

---

## Phase 0: 数据-训练完全解耦（架构基础）

**收益**：数据预处理独立于模型，artifact 按 dataset 类型自动匹配
**风险**：中（改动 ConfigParser 和入口脚本，需确保现有链路兼容）
**改动范围**：`config_parser.py`, `artifact_io.py`, `scripts/run/`

### Step 0.1 — ConfigParser 支持无模型模式

当前 `ConfigParser.__init__` 强制要求 `model` 参数。改造为：
- `model=None` 时跳过模型 config 加载（`_load_default_config` 中 model/executor/evaluator 分支）
- 仅加载 dataset class config + resource_data/{dataset}/config.json
- 数据相关参数（`input_window`, `output_window`, `scaler` 等）从 dataset class 默认配置 + resource_data config 获取

### Step 0.2 — 重构 run_data_prep.py

新的 `scripts/run/run_data_prep.py`：
```bash
# 数据预处理只需要 task + dataset + dataset_class
uv run python scripts/run/run_data_prep.py \
  --task traffic_state_pred \
  --dataset METR_LA \
  --dataset_class TrafficStatePointDataset

# 可选覆盖参数
uv run python scripts/run/run_data_prep.py \
  --task traffic_state_pred \
  --dataset METR_LA \
  --dataset_class TrafficStatePointDataset \
  --input_window 12 --output_window 12 --scaler standard
```

产物：`cache/data_artifacts/da_<ts>__METR_LA__TrafficStatePointDataset__<sig8>/`

### Step 0.3 — run_train_artifact.py 自动匹配 artifact

当前训练必须传 `--artifact_id`。改造为：
- 不传 `--artifact_id` 时自动扫描 `cache/data_artifacts/`
- 按 `dataset`（名称）+ `dataset_class`（manifest 中定义）匹配
- 选择最新（`created_at` 最大）+ 签名验证通过 + status=ready 的 artifact
- 支持 `--artifact_latest` flag 强制使用最新匹配，`--artifact_list` 列出可用 artifact

```bash
# 自动匹配最新 artifact
uv run python scripts/run/run_train_artifact.py \
  --task traffic_state_pred \
  --model new_diffusion_fuzzy \
  --dataset METR_LA

# 查看可用 artifact
uv run python scripts/run/run_train_artifact.py \
  --task traffic_state_pred \
  --model new_diffusion_fuzzy \
  --dataset METR_LA --artifact_list
```

### Step 0.4 — 创建 fuzzy 实验入口脚本

新建 `scripts/experiments/train_new_diffusion_fuzzy.py`，使用 Step 0.3 的自动匹配机制：
```bash
uv run python scripts/experiments/train_new_diffusion_fuzzy.py --dataset METR_LA
```
内部调用 `run_train_artifact(task, model="new_diffusion_fuzzy", dataset, ...)`

### Step 0.5 — 废弃旧实验脚本（可选）

- 删除或归档 `scripts/experiments/train_new_diffusion.py`
- 删除或归档 `scripts/experiments/train_new_diffusion_2.py`
- 记录到 `ai_logs/change/`：这些是 git 前手写迭代产物，已被新流程替代

### ✅ 验证标准
- `run_data_prep.py --dataset METR_LA` 不传 `--model` 可成功构建 artifact
- `run_train_artifact.py --model new_diffusion_fuzzy --dataset METR_LA` 自动匹配 artifact
- 传统链路 `run_model.py` 不受影响

### 🔖 Git 节点
- `phase-0-data-decouple` 分支，每 Step 独立 commit

---

## Phase 1: 模型架构增强

**收益**：提取共享基础设施，增强注意力与图学习，为后续优化打基础
**风险**：中（改动前向传播，需训练验证）
**改动范围**：`new_diffusion_fuzzy/model.py`, `new_diffusion_fuzzy/utils/`

### Step 1.1 — 提取共享扩散基础设施

在 `GNNTP/models/new/` 下创建 `_diffusion_common/`：
```
_diffusion_common/
├── __init__.py
├── attention.py        # MultiHeadAttention
├── ffn.py              # FeedForwardNetwork
├── diffusion.py         # DiffusionScheduler (beta schedule, add_noise, ddpm/ddim step)
├── embedding.py         # SinusoidalTimeEmbedding
├── graph.py             # build_normalized_adjacency, expand_adjacency_batch
├── attention_ops.py     # apply_temporal/cross/spatiotemporal_attention
└── INFO.md
```

`new_diffusion_fuzzy/model.py` 改为从 `_diffusion_common` import。

### Step 1.2 — 配置校验

`NewDiffusion.__init__` 末尾增加 `_validate_config()`：
- `hidden_dim % num_heads == 0`
- `num_sampling_steps <= diffusion_steps`
- `physics_warmup_steps > 0` 时 `physics_loss_weight > 0`
- `adaptive_graph_topk` 存在时 `<= num_nodes`

### Step 1.3 — 形状标注补充

为所有 forward 方法补全 shape docstring，最低覆盖：
- `MultiHeadAttention.forward` — `[B*N, T, D] → [B*N, T, D]`（含 cross-attn 变体）
- `DiffusionScheduler.add_noise` — `[B, T, N, D_out] + [B] → [B, T, N, D_out]`
- `AdaptiveGraphLearner._fuzzy_similarity_to_adjacency` — `[B, N, D_fn] → [B, N, N]`
- `AttentionDenoiser.forward` — 所有中间张量

### Step 1.4 — 位置编码可学习化

当前 `temporal_position_embedding` 和 `future_position_embedding` 是固定 sinusoidal。
- 新增 flag `use_learnable_position_embedding`（默认 false）
- true 时替换为 `nn.Parameter(torch.randn(max_time_steps, hidden_dim) * 0.02)`
- 保留 sinusoidal 作为 fallback

### ✅ 验证标准
- import 路径正确，模型可正常实例化
- 配置非法值时报清晰错误
- 10 epoch METR-LA 快速训练 loss 不退化

### 🔖 Git 节点
- `phase-1-architecture` 分支

---

## Phase 2: 训练优化

**收益**：收敛速度 + 最终性能 + 训练稳定性
**风险**：中（涉及损失函数和训练动态）
**改动范围**：`model.py`（损失计算部分）, `config.json`

### Step 2.1 — 扩散损失改进

- **2.1a**: 支持 `diffusion_loss_type` 配置：`"mse"` / `"huber"` / `"smooth_l1"`
- **2.1b**: 支持 improved DDPM 的 timestep-dependent 加权（`loss_weight_type: "uniform"` / `"snr"` / `"truncated_snr"`）

### Step 2.2 — 物理预热自适应化

当前固定 3000 步线性/余弦 warmup。增加自适应模式：
- `physics_warmup_mode: "adaptive"`：监控守恒残差均值，当 5 个连续窗口的残差方差 < 阈值时自动结束预热
- 与固定步数模式可切换

### Step 2.3 — 学习率调度改进

- 新增 `lr_scheduler_type` 配置：`"multistep"` / `"cosine_warm_restart"` / `"onecycle"`
- 对应新增配置：`lr_T_0`, `lr_T_mult`, `lr_warmup_epochs`

### Step 2.4 — 梯度累积

- 新增 `gradient_accumulation_steps`（默认 1）
- 仅每 N 步更新一次 optimizer
- 等效增大 batch 不增加显存

### ✅ 验证标准
- 50 epoch METR-LA 全量训练，test MAE 优于当前 baseline
- 物理损失曲线无震荡

### 🔖 Git 节点
- `phase-2-training` 分支

---

## Phase 3: 推理优化

**收益**：推理延迟降低 + 显存减少
**风险**：低-中
**改动范围**：`model.py`（采样逻辑部分）

### Step 3.1 — DDIM 加速扫描

- 在 METR-LA 上扫描 `num_sampling_steps` ∈ {10, 20, 30, 50, 100}
- 输出精度-延迟表，找最佳性价比
- 若 20 步精度损失 < 2%，更新默认值为 20

### Step 3.2 — 条件编码器缓存

`encode_condition(history)` 在 `_sample_once` 每次调用时都传递，但交叉注意力的 K/V 已是编码器输出。
- 采样循环中不重复传递 condition
- 使用 `torch.inference_mode()` 包裹采样

### Step 3.3 — 参数与 FLOPs 统计

- 新增 `scripts/tools/model_stats.py`
- 使用 `torchinfo` 输出：参数量、每层 FLOPS、显存占用预估
- 输出到 `outputs/model_stats/fuzdiff_<dataset>.json`

### ✅ 验证标准
- 推理延迟 < 80ms（METR-LA, 单样本）
- 精度退化 < 1% MAE

### 🔖 Git 节点
- `phase-3-inference` 分支

---

## Phase 4: 消融实验与论文支撑

**收益**：论文核心实证章节
**风险**：低（纯实验执行）
**改动范围**：`scripts/experiments/`, `paper/`

### Step 4.1 — 消融实验脚本

新建 `scripts/experiments/ablation_new_diffusion_fuzzy.py`：
```bash
uv run python scripts/experiments/ablation_new_diffusion_fuzzy.py \
  --dataset METR_LA \
  --ablations fuzzy_graph,fuzzy_conservation,physics_warmup,spatiotemporal_attn
```

自动执行预设的消融配置组合，输出 `outputs/ablation/<dataset>/results.json`

### Step 4.2 — 消融维度定义

| 消融 | 配置变更 | 验证假设 |
|------|---------|---------|
| w/o Fuzzy Graph | `use_fuzzy_graph=False` | 模糊隶属度 > 普通 softmax |
| w/o Fuzzy Conservation | `use_fuzzy_conservation=False` | 拥堵加权 > 均匀守恒 |
| w/o Physics Warmup | `physics_warmup_steps=0` | 预热机制是否必要 |
| w/o Adaptive Graph | `use_adaptive_graph=False` | 动态图 vs 静态图 |
| w/o Spatiotemporal Attn | `use_spatiotemporal_attention=False` | 全展平注意力贡献 |
| w/o Diffusion | 直接回归（编码器 + MLP head） | 扩散框架价值 |

### Step 4.3 — 随机种子扫描

- 5 种子 (42, 123, 456, 789, 1024) × METR-LA + PEMSD4
- 输出 mean ± std 到 `outputs/seed_scan/results.json`

### Step 4.4 — 结果写入论文

- 填充 `sections_cn/experiments.tex` 中所有 `--` 占位符
- 更新 `paper/tables/main_results.tex`

### ✅ 验证标准
- 所有 `--` 占位符已填充
- 至少 3 个消融维度有明确结论

### 🔖 Git 节点
- `phase-4-ablation` 分支

---

## 执行策略

```
Phase 0 (数据解耦) ──► Phase 1 (架构增强) ──► Phase 2 (训练优化)
                                                    │
                          Phase 3 (推理优化) ◄───────┘
                                                    │
                          Phase 4 (消融+论文) ◄──────┘
```

- **Phase 0 是阻塞项**：所有后续训练都依赖新的数据流程
- **Phase 3 可与 Phase 2/4 并行**：推理优化不依赖训练结果
- **每个 Phase 结束时** merge 到 `新模型` 分支

## 预期工作量

| Phase | 文件改动 | GPU 训练 | 说明 |
|-------|:-------:|:-------:|------|
| Phase 0 | ~8 | — | 数据解耦，架构基础 |
| Phase 1 | ~6 | 10 epoch 验证 | 提取共享 + 标注 + 配置校验 |
| Phase 2 | ~3 | 50 epoch 全量 | 损失/预热/LR改进 |
| Phase 3 | ~2 | benchmark | 推理加速 |
| Phase 4 | ~4 | 消融网格 | 实验 + 论文数据 |

---

## 当前状态

- [x] Phase 0: 数据-训练解耦
- [ ] Phase 1: 模型架构增强
- [ ] Phase 2: 训练优化
- [ ] Phase 3: 推理优化
- [ ] Phase 4: 消融实验与论文
