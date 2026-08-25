# 2026-08-25: Evaluator 补充论文口径的全步 pooled masked 指标

## 变更文件
- `GNNTP/common/traffic_state_evaluator.py`

## 原因
对照 5 篇论文（STGformer、STGCN、PDFormer、STAEformer、STG4Traffic）及其官方源码后确认：
论文主流（STGformer/STAEformer/STG4Traffic 的流量数据、PDFormer 的 average 模式）报告的是
**全步 pooled** 指标——把全部 (样本×时间步×节点) 联合计算并 mask 剔除 `y_true==0`，
即所谓 All-Steps 平均；而我们默认 `evaluator_mode=single` 只有逐 horizon 单步指标，
web 的 `avg` 是对 12 个单步指标取算术平均，与 pooled 不严格等价（RMSE/MAPE 非线性）。

## 方案
不改变现有 `single`/`average` 逐 horizon 结果的任何语义，在 evaluator 内**补充输出三项**
全步 pooled masked 指标：`masked_MAE_avg` / `masked_MAPE_avg` / `masked_RMSE_avg`。

- 在 `collect()` 累积全量 `y_true`/`y_pred` 张量
- 在 `evaluate()` 末尾对全量张量调用 `loss.masked_*_torch(..., 0, mask_val=self.mask_val)`
  （`null_val=0` 即 mask 剔除 y_true==0，与 STGformer/STAEformer 的 `mask = (y_true != 0)` 一致）
- 在 `clear()` 重置累积列表

## 具体改动
### 1) collect(): 累积原始张量（shape 校验之后）
```python
# 累积全量真值/预测，用于计算论文口径（STGformer/STAEformer All-Steps）的 pooled 指标
if not hasattr(self, '_y_trues'):
    self._y_trues = []
    self._y_preds = []
self._y_trues.append(y_true)
self._y_preds.append(y_pred)
```

### 2) evaluate(): 追加 pooled 三项（逐 horizon 循环之后、return 之前）
```python
# 论文主流口径：全步 pooled masked 指标（mask 剔除 y_true==0）。
# 与 STGformer/STAEformer/STG4Traffic 的 All-Steps 平均一致：对全部
# (样本×时间步×节点) 联合计算，而非对逐 horizon 指标取算术平均。
if getattr(self, '_y_trues', None):
    y_true_all = torch.cat(self._y_trues, dim=0)
    y_pred_all = torch.cat(self._y_preds, dim=0)
    self.result['masked_MAE_avg'] = loss.masked_mae_torch(y_pred_all, y_true_all, 0,
                                                          mask_val=self.mask_val).item()
    self.result['masked_MAPE_avg'] = loss.masked_mape_torch(y_pred_all, y_true_all, 0,
                                                            mask_val=self.mask_val).item()
    self.result['masked_RMSE_avg'] = loss.masked_rmse_torch(y_pred_all, y_true_all, 0,
                                                            mask_val=self.mask_val).item()
```

### 3) clear(): 重置累积列表
```python
self._y_trues = []
self._y_preds = []
```

## 预期效果
- 现有 `MAE@1..@12`、`masked_MAE@1..@12` 等逐 horizon 结果完全不变（兼容历史）
- 每次 `evaluate()` 的 JSON 结果额外多出 `masked_MAE_avg`/`masked_MAPE_avg`/`masked_RMSE_avg`
  三项，即论文主流口径的全步平均，可直接用于论文表格
- 不改 CSV 逐 horizon 表格（避免影响 web 前端展示）
- MAPE 分母为 `|label|+1e-5`，与 STGCN 同款 epsilon；STGformer/STAEformer 无 epsilon，
  数值差异可忽略（1e-5 ≪ 流量量级）
