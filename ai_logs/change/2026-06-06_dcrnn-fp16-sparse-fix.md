# DCRNN FP16 稀疏矩阵乘法修复

**时间**: 2026-06-06 19:19 (v1) / 19:30 (v2)
**类型**: Bug修复
**文件**: `GNNTP/models/traffic_speed_prediction/DCRNN/model.py`
**位置**: `GCONV.forward` 方法 (L110-126)

## 问题

评估时报错：
```
NotImplementedError: "addmm_sparse_cuda" not implemented for 'Half'
```

## 根因

**两层原因叠加**:

1. `support` 稀疏邻接矩阵通过 `register_buffer` 注册。`model.half()` 时 buffer 也被转为 Half。
2. `traffic_state_executor.py:330` — **评估时也启用了 `torch.autocast`**，autocast 会自动把 `torch.sparse.mm` 的 float32 操作数重新 cast 成 Half。

第一次修复（v1）只做了 `.float()` 转换，但 autocast 在操作执行时又会把张量转回 Half，所以无效。

## 修复 (v2)

在 Chebyshev 扩散循环外用 `torch.autocast(enabled=False)` 包裹，同时显式做 FP32 转换：

```python
with torch.autocast(device_type=self._device.type, enabled=False):
    x0_fp32 = x0.float()
    for support in supports:
        support_fp32 = support.float()
        x1 = torch.sparse.mm(support_fp32, x0_fp32)
        # ... Chebyshev 迭代中也使用 support_fp32 / x0_fp32
```

这样同时处理了两种场景：
- **AMP autocast**: disabled 块内 sparse.mm 不会被 cast 成 Half
- **model.half()**: `.float()` 把 Half buffer 转成 FP32

autocast 块外的 `torch.matmul` 仍能正常享受 AMP 加速。

## 影响范围

仅影响 DCRNN 模型在 FP16/AMP 模式下的评估/推理。
