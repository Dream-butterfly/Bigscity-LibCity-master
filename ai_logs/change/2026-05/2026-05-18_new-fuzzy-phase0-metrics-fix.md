# Phase 0 — new_fuzzy 评估指标修复 + 代码清理

**日期**: 2026-05-18
**类型**: 修复 + 清理
**模型**: new_fuzzy
**影响文件**:
- `GNNTP/data/dataset/traffic_state_dataset.py` (🔴, L 批准)
- `GNNTP/data/dataset/TrafficStatePointDataset.json` (🟡)
- `GNNTP/data/dataset/traffic_flow_prediction/PDFormerDataset.json` (🟡)
- `GNNTP/models/loss.py` (🟡)
- `GNNTP/models/new/new_fuzzy/model.py` (🟢)
- `GNNTP/models/new/new_fuzzy/config.json` (🟢)
- `GNNTP/models/new/new_fuzzy/embedding.py` (🟢)
- `GNNTP/models/new/new_fuzzy/decoder.py` (🟢)
- `GNNTP/models/new/new_fuzzy/utils/__init__.py` (🟢)
- `scripts/tools/verify_new_fuzzy_metrics.py` (🟢, 新建)

## 问题

0. **Scaler 默认值为 "none"**：基类 `TrafficStateDataset` 和 dataset JSON 配置
   均将 scaler 默认为 "none"（不归一化）。导致多特征数据（flow/occupancy/speed）
   量级差异巨大，L1 loss 被 flow 完全主导，occupancy R²=-114，speed R²=0.46。
   诊断脚本证实这也是之前 R²=0.98 为统计假象的根本原因。
1. **MAPE 爆炸**：Z-score 标准化后近零值导致 MAPE 达 6000-10000%，
   原 `masked_mape_torch` unmasked 路径直接对 `(label + 1e-5)` 取除法
2. **steps_per_epoch 硬编码**：`model.py:170` 写死 160，DDP 下每个 rank
   的 step 计数减半，warmup 时长错误
3. **死代码 + 误导注释**：`embedding.py` 的 SinusoidalTimeEmbedding 未使用，
   `decoder.py` 和 `utils/__init__.py` 残留扩散模型注释

## 修改

### 0. Scaler 默认值统一为 "standard"

- `traffic_state_dataset.py:42`: `config.get("scaler", "none")` → `"standard"`
- `TrafficStatePointDataset.json:8`: `"scaler": "none"` → `"standard"`
- `PDFormerDataset.json:8`: `"scaler": "none"` → `"standard"`

影响：所有新创建的数据工件默认使用 Z-score 归一化（除非显式覆盖）。

### 1. `loss.py` — masked_mape_torch unmasked 路径加近零过滤

```python
# 旧：直接除法，近零 label 产生极大百分比误差
loss = torch.abs((preds - labels) / (labels + eps))
return torch.mean(loss)

# 新：过滤 |label| < 1e-3 的值
valid = torch.abs(labels) > 1e-3
if valid.sum() == 0:
    return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
loss = torch.abs((preds - labels) / (labels + eps))
loss = loss * valid.float()
return loss.sum() / valid.sum()
```

masked_MAPE 路径（null_val != nan 时）不变。

### 2. `model.py` — steps_per_epoch 改为可配置参数

- 新增 `conservation_steps_per_epoch` config 参数（默认 80）
- `_get_effective_conservation_weight()` 使用该参数替代硬编码 160

### 3. `config.json` — 添加 conservation_steps_per_epoch

默认值 80（PEMSD4, batch_size=64, 2-GPU DDP: 每 rank 80 batches/epoch）

### 4. 注释清理

- `embedding.py`: 标记 SinusoidalTimeEmbedding 为未使用（保留文件供参考）
- `decoder.py`: 移除 "Based on the denoiser architecture" 误导注释
- `utils/__init__.py`: docstring 从 `new_diffusion_fuzzy_2` 改为 `new_fuzzy`

### 5. 新建 R²/MAPE 验证脚本

`scripts/tools/verify_new_fuzzy_metrics.py` — 诊断工具：
- 标准化 vs 原始尺度 R² 对比
- 逐特征 R² 分解 (flow/occupancy/speed)
- HA baseline R² 对比
- 修复后 MAPE 验证

## 验证

MAPE 修复后应 < 50%（而非 6000%+）。
R² 验证脚本需在训练环境运行：`uv run scripts/tools/verify_new_fuzzy_metrics.py --run_id <id>`
