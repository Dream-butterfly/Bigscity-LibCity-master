# 变更 #16: 数据管线时间特征缺失修复 + output_projection 零初始化

**日期**: 2026-05-08
**类型**: 修复
**影响范围**: 所有模型 + new_diffusion_fuzzy

---

## 问题

### 1. pipeline_mixin.py: 时间特征条件调用 bug
`_generate_data()` 中 `if ext_data is not None: df = self._add_external_information(df, ext_data)` 导致当 `load_external=false` 时 `add_time_in_day`/`add_day_in_week` 完全无效。`_add_external_information_3d` 内部已正确处理 ext_data=None，但调用方拦截了。

**影响**: 所有 `add_time_in_day=true, add_day_in_week=true, load_external=false` 的模型训练都缺少时间特征。

### 2. model.py: output_projection 未零初始化
扩散模型去噪器的最终输出层未零初始化，导致初始噪声预测有任意量级，训练初期极不稳定。

### 3. 数据工件配置不匹配导致 MAE~40,000
旧工件 da_20260506_220144 用旧 config (scaler=none, 无时间特征) 创建，新 config 改为 scaler=standard + 时间特征。工件签名不匹配 + scaler 不匹配。

---

## 修改

### 1. pipeline_mixin.py (L108-110)
```diff
-            if ext_data is not None:
-                df = self._add_external_information(df, ext_data)
+            df = self._add_external_information(df, ext_data)
```

### 2. model.py: AttentionDenoiser.__init__ (L453+)
```diff
 self.output_projection = nn.Linear(hidden_dim, output_dim)
+nn.init.zeros_(self.output_projection.weight)
+nn.init.zeros_(self.output_projection.bias)
```

### 3. 新数据工件
- `da_20260508_131627__PEMSD4__TrafficStatePointDataset__44f9874b`
- StandardScaler: mean=63.44, std=8.25
- feature_dim=9 (1 speed + 1 tod + 7 dow one-hot), output_dim=1
- train/eval/test: 11878/1697/3394

---

## 修改文件清单
| 文件 | 变更 |
|------|------|
| `GNNTP/data/dataset/mixins/pipeline_mixin.py` | 移除 ext_data 条件限制 |
| `GNNTP/models/new/new_diffusion_fuzzy/model.py` | output_projection 零初始化 |
