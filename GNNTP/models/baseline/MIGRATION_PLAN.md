# 基线模型迁移计划

> 从 `Bigscity-LibCity-master` → `LibCity/GNNTP`
> 创建时间: 2026-06-05

---

## 1. 架构差异分析

### 1.1 框架对比

| 维度 | Bigscity-LibCity-master | LibCity/GNNTP |
|------|------------------------|---------------|
| 模型基类 | `libcity.model.abstract_traffic_state_model.AbstractTrafficStateModel` | `GNNTP.models.abstract_traffic_state_model.AbstractTrafficStateModel` |
| 模型注册 | 手动 `__init__.py` 导入 | `manifest.json` 自动发现 |
| 模型路径 | `libcity/model/traffic_flow_prediction/*.py` | `GNNTP/models/<task>/<ModelName>/model.py` |
| 数据集路径 | `libcity/data/dataset/dataset_subclass/*.py` | `GNNTP/data/dataset/` |
| 执行器基类 | `libcity.executor.traffic_state_executor.TrafficStateExecutor` | `GNNTP.common.traffic_state_executor.TrafficStateExecutor` |
| loss 导入 | `from libcity.model import loss` | `from GNNTP.models import loss` |
| config 解析 | `ConfigParser` 统一管理 | 相同机制，`config.json` + 全局参数 |

### 1.2 基类接口（两端一致）

两端的 `AbstractTrafficStateModel` 接口完全相同：
- `__init__(self, config, data_feature)` — config 是 dict, data_feature 是 dict
- `predict(self, batch)` → torch.Tensor — 返回预测值
- `calculate_loss(self, batch)` → torch.Tensor — 返回训练 loss

### 1.3 manifest.json 自动发现机制

LibCity 通过 `GNNTP/models/locator.py` 遍历所有子目录，自动读取 `manifest.json` 来注册模型。
**任何包含 `manifest.json` 的目录都会被自动发现，无需手动导入。**

manifest.json 格式：
```json
{
  "task": "traffic_state_pred",
  "model": "ModelName",
  "model_entry": "GNNTP.models.baseline.ModelName.model:ModelName",
  "dataset_class": "TrafficStatePointDataset",
  "dataset_entry": "GNNTP.data.dataset.traffic_state_point_dataset:TrafficStatePointDataset",
  "executor": "TrafficStateExecutor",
  "executor_entry": "GNNTP.common.traffic_state_executor:TrafficStateExecutor",
  "evaluator": "TrafficStateEvaluator",
  "evaluator_entry": "GNNTP.common.traffic_state_evaluator:TrafficStateEvaluator"
}
```

---

## 2. 已迁移参考模型

### 2.1 STGCN（无自定义执行器）

- 路径: `GNNTP/models/traffic_speed_prediction/STGCN/`
- 文件: `model.py`, `config.json`, `manifest.json`
- 类型: 最简迁移，无 executor.py，直接用 `TrafficStateExecutor`

### 2.2 DCRNN（有自定义执行器）

- 路径: `GNNTP/models/traffic_speed_prediction/DCRNN/`
- 文件: `model.py`, `config.json`, `manifest.json`, `executor.py`, `executor.json`
- 类型: 需要自定义 executor（curriculum learning + 自定义 loss 格式）

### 2.3 STGformer（有自定义执行器）

- 路径: `GNNTP/models/traffic_speed_prediction/STGformer/`
- 文件: `model.py`, `config.json`, `manifest.json`, `executor.py`, `executor.json`

---

## 3. 待迁移模型清单

### 3.1 模型概览

| # | 模型 | 源文件 | 自定义数据集 | 自定义执行器 | 复杂度 |
|---|------|--------|:---:|:---:|:---:|
| 1 | HA | 手写 | - | - | 低 |
| 2 | VAR | 手写 | - | - | 中 |
| 3 | Graph WaveNet | `traffic_speed_prediction/GWNET.py` | - | - | 中 |
| 4 | MTGNN | `traffic_speed_prediction/MTGNN.py` | - | ✅ MTGNNExecutor | 中 |
| 5 | ASTGCN | `traffic_flow_prediction/ASTGCN.py` + `ASTGCNCommon.py` | ✅ ASTGCNDataset | - | 中 |
| 6 | GMAN | `traffic_speed_prediction/GMAN.py` | ✅ GMANDataset | - | 中 |
| 7 | PDFormer | `traffic_flow_prediction/PDFormer.py` | ✅ PDFormerDataset | ✅ PDFormerExecutor | 高 |
| 8 | STAEformer | `traffic_speed_prediction/STAEformer.py` | ✅ STAEformerDataset | - | 中 |
| 9 | STID | `traffic_speed_prediction/STID.py` | - | - | 低 |
| 10 | AGCRN | `traffic_flow_prediction/AGCRN.py` | - | - | 中 |

### 3.2 数据集依赖详情

需要在 LibCity 侧实现的预处理逻辑：

| 数据集 | 额外处理 | 处理方式 |
|--------|---------|----------|
| ASTGCNDataset | 时间特征计算 (time_in_day, day_in_week) + 邻接矩阵标准化 | 与 LibCity 现有 `TrafficStatePointDataset` 基本相同，config 中设置 `add_time_in_day`/`add_day_in_week` 即可 |
| GMANDataset | 空间嵌入 (spatial embedding from adjacency) | 需要在 `data_feature` 中预计算 SE 矩阵并传入 |
| STAEformerDataset | 时间特征拼接到输入 (TOD+DOW) | 在 DataLoader 的 collate 中将时间特征拼入输入张量 |
| PDFormerDataset | DTW 矩阵 + 短路径矩阵 + 模式聚类 (pattern_keys) + Laplacian PE | 最复杂，需在数据预处理阶段完成 |

---

## 4. 标准化迁移流程

### 4.1 目录结构模板

```
GNNTP/models/baseline/<ModelName>/
├── model.py          # 模型定义
├── config.json       # 默认超参数
├── manifest.json     # 注册元信息
├── executor.py       # (可选) 自定义执行器
├── executor.json     # (可选) 执行器配置
└── INFO.md           # 模型说明
```

### 4.2 迁移步骤

#### Step 1: 复制源模型代码
从 `Bigscity-LibCity-master/libcity/model/traffic_*_prediction/<ModelName>.py` 复制到 `model.py`

#### Step 2: 修改 import 路径
```python
# 改前 (Bigscity)
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

# 改后 (LibCity)
from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel
from GNNTP.models import loss
```

#### Step 3: 适配 `data_feature` 数据结构
LibCity 的 `data_feature` 与 Bigscity 结构基本一致：
```python
{
    'num_nodes': int,
    'feature_dim': int,
    'output_dim': int,
    'adj_mx': np.ndarray,  # 邻接矩阵
    'scaler': StandardScaler/NormalScaler,
    # ... 模型自定义字段
}
```

#### Step 4: 处理数据集依赖
- **无自定义数据集**: 直接用 `TrafficStatePointDataset`，在 manifest.json 中指定
- **有自定义数据集**: 
  1. 在 `GNNTP/data/dataset/` 下创建 `*_dataset.py`
  2. 继承 `TrafficStatePointDataset` 或 `TrafficStateDataset`
  3. 在 manifest.json 中指定 `dataset_class` 和 `dataset_entry`
  4. 将预处理逻辑放入 `__init__` 或重写的 `_load_*` 方法

#### Step 5: 处理执行器依赖
- **无自定义执行器**: 直接用 `TrafficStateExecutor`
- **有自定义执行器**: 
  1. 创建 `executor.py`，继承 `GNNTP.common.traffic_state_executor.TrafficStateExecutor`
  2. 创建 `executor.json`，指定执行器级配置
  3. 在 manifest.json 中指定 `executor` 和 `executor_entry`

#### Step 6: 创建 manifest.json
#### Step 7: 创建 config.json（从 Bigscity task_config.json 提取对应参数）
#### Step 8: 验证可运行

---

## 5. 各模型迁移要点

### 5.1 HA (Historical Average)

- **类型**: 统计方法，非神经网络
- **实现**: 继承 `AbstractTrafficStateModel`，predict 返回历史同期均值
- **config**: `ha_window`（历史窗口天数，默认 7）

### 5.2 VAR (Vector Autoregression)

- **类型**: 统计时序模型
- **依赖**: `statsmodels.tsa.vector_ar.var_model.VAR`
- **实现**: 对每个节点独立拟合 VAR(p) 模型，p 由 AIC/BIC 选择
- **注意**: 大数据集（PEMS07: 883 节点）可能很慢

### 5.3 Graph WaveNet

- **源**: `GWNET.py` (~450行，单文件)
- **关键模块**: GCN + TCN + adaptive adjacency
- **自有**: 无需自定义 dataset/executor
- **需适配**: `supports` (图支撑矩阵) 从 `data_feature['adj_mx']` 计算

### 5.4 MTGNN

- **源**: `MTGNN.py` (~600行，单文件)
- **关键模块**: MixProp GCN + dilated inception TCN + adaptive graph learner
- **自有**: 需要自定义 `MTGNNExecutor`（节点拆分训练：num_split 参数）
- **需适配**: 
  - executor 中的 `_train_epoch` 实现 node-splitting
  - `step_size2` 控制节点重排频率
  - `num_split` 控制子图数量

### 5.5 ASTGCN

- **源**: `ASTGCN.py` + `ASTGCNCommon.py`
- **关键模块**: Spatial-Temporal Attention + ChebConv GCN
- **自有**: 需自定义 `ASTGCNDataset`（时间特征提取 TOD/DOW）
- **策略**: LibCity 的 `TrafficStatePointDataset` 本身就支持 `add_time_in_day`/`add_day_in_week`，可能不需要额外 dataset

### 5.6 GMAN

- **源**: `GMAN.py` (~900行，单文件大模型)
- **关键模块**: Spatial-Temporal Encoder-Decoder Attention + Gated Fusion
- **自有**: 需自定义 `GMANDataset`（空间嵌入 SE 计算）
- **需适配**: SE 矩阵在 `data_feature` 中预计算

### 5.7 PDFormer（重新迁移）

- **当前状态**: 已有迁移到 `GNNTP/models/traffic_flow_prediction/PDFormer/`，但效果不好
- **问题分析**: 
  1. 数据集预处理差异（DTW 矩阵、pattern_keys 计算方式）
  2. Laplacian PE 在 executor 中计算（不在 data_feature 中），导致不同运行可能不一致
  3. 随机翻转 (random_flip) 的符号一致性
  4. 可能缺少 `sd_mx`/`sh_mx` 等数据
- **迁移策略**: 
  1. 从 Bigscity 源码重新开始
  2. 将 `sd_mx`, `sh_mx`, `dtw_matrix`, `pattern_keys` 预计算并放入 `data_feature`
  3. 将 Laplacian PE 预计算并放入 `data_feature['lap_mx']`
  4. 创建 `PDFormerDataset` 子类
  5. 或：保持现有数据集，在 executor 中完成预处理（参考 DCRNNExecutor 模式）

### 5.8 STAEformer

- **源**: `STAEformer.py` (~400行)
- **关键模块**: Spatial-Temporal Adaptive Embedding + Transformer
- **自有**: 需自定义 `STAEformerDataset`（TOD/DOW 拼接到输入张量作为额外特征通道）
- **策略**: 可在 `data_feature` 中通过 `load_external=True` 实现

### 5.9 STID

- **源**: `STID.py` (~180行，最简模型)
- **关键模块**: MLP + Temporal/Spacial Identity Embeddings
- **自有**: 无需自定义 dataset/executor
- **迁移难度**: 最低

### 5.10 AGCRN

- **源**: `AGCRN.py` (~250行)
- **关键模块**: Adaptive Graph Convolutional Recurrent Network (AVWGCN + AGCRNCell)
- **自有**: 无需自定义 dataset/executor
- **需适配**: `node_embeddings` 参数 (自适应图学习的节点嵌入)

---

## 6. 迁移顺序建议

按从易到难排序，建立信心后攻克难点：

```
Phase 1 (低难度，验证流程):
  1. STID      — 最简，180行，无自定义组件
  2. AGCRN     — 中等，无自定义组件
  3. Graph WaveNet — 中等，无自定义组件

Phase 2 (中难度，需要自定义组件):
  4. ASTGCN    — 需要自定义 dataset
  5. STAEformer — 需要自定义 dataset
  6. GMAN      — 需要自定义 dataset

Phase 3 (中难度，需要自定义执行器):
  7. MTGNN     — 需要自定义 executor

Phase 4 (高难度):
  8. PDFormer  — 重新迁移，最复杂

Phase 5 (手写):
  9. HA        — 统计方法
  10. VAR      — 统计方法，计算量大
```

---

## 7. 验证标准

每个模型迁移完成后需验证：

1. ✅ `manifest.json` 格式正确，能被 locator 发现
2. ✅ `import` 路径全部指向 GNNTP
3. ✅ 模型能正常初始化（`Model(config, data_feature)` 不报错）
4. ✅ 前向传播shape正确
5. ✅ 训练 1 epoch 不报错（loss 正常下降）
6. ✅ 在单个数据集（如 METR_LA）上指标与原仓库一致（误差 < 1e-4）

---

## 8. 目标目录结构

```
GNNTP/models/baseline/
├── MIGRATION_PLAN.md
├── HA/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
├── VAR/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
├── GraphWaveNet/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
├── MTGNN/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   ├── executor.py
│   ├── executor.json
│   └── INFO.md
├── ASTGCN/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
├── GMAN/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
├── PDFormer/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   ├── executor.py
│   ├── executor.json
│   └── INFO.md
├── STAEformer/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
├── STID/
│   ├── model.py
│   ├── config.json
│   ├── manifest.json
│   └── INFO.md
└── AGCRN/
    ├── model.py
    ├── config.json
    ├── manifest.json
    └── INFO.md
```

---

## 9. 数据集适配说明

需要在 `GNNTP/data/dataset/` 下新增的自定义数据集类：

| 新数据集类 | 继承自 | 用途 |
|-----------|--------|------|
| `GMANDataset` | `TrafficStatePointDataset` | 预计算空间嵌入 SE |
| `STAEformerDataset` | `TrafficStatePointDataset` | 时间特征 (TOD/DOW) 拼入输入 |
| `ASTGCNDataset` | `TrafficStatePointDataset` | 时间周期特征嵌入 |
| `PDFormerDataset` | `TrafficStatePointDataset` | DTW + 模式聚类 + Laplacian PE |

对于不需要自定义 dataset 的模型，manifest.json 中统一使用：
```json
"dataset_class": "TrafficStatePointDataset",
"dataset_entry": "GNNTP.data.dataset.traffic_state_point_dataset:TrafficStatePointDataset"
```

## 10. 已知风险

1. **PDFormer 当前迁移效果差**: 可能是 DTW 矩阵计算差异、pattern_keys 聚类不一致、Laplacian PE 随机翻转问题
2. **MTGNN node-splitting**: executor 中的节点拆分训练逻辑在 DDP 模式下需特殊处理
3. **GMAN 空间嵌入**: SE 计算依赖 `adj_mx` 的稀疏格式，与 LibCity 的 `adj_mx` 格式可能不同
4. **ASTGCN Common 模块**: 源文件拆分为两个 .py，合并到一个 model.py 中
5. **VAR 性能**: PEMS07 (883节点) 上 VAR 拟合可能非常慢，需考虑并行或采样
