# DCRNN AMP 混合精度 FP16 稀疏矩阵乘法不兼容修复

**日期**: 2026-05-26
**类型**: Bug 修复
**影响范围**: DCRNN 模型 — `GCONV.forward()`

## 问题
AMP 混合精度训练/评估时，`torch.sparse.mm` 在 CUDA 上不支持 Half (FP16) 精度：
```
NotImplementedError: "addmm_sparse_cuda" not implemented for 'Half'
```

## 根因
`GCONV.forward()` 中的 Chebyshev 多项式图卷积（`torch.sparse.mm(support, x0)`），当 AMP autocast 将输入转为 FP16 时，CUDA 后端没有实现 Half 精度的稀疏矩阵乘法。

## 修复
- **文件**: `GNNTP/models/traffic_speed_prediction/DCRNN/model.py`
- **位置**: `GCONV.forward()` (L99-130)
- **改动**: 
  - 稀疏矩阵乘法前将输入 `x0`, `x` 临时转为 FP32
  - 乘法完成后、`matmul(self.weight)` 之前转回原始 dtype
  - FP32 输入时跳过转换（零开销）

## 影响
- 不影响 FP32 训练/评估路径
- 不影响模型精度（稀疏运算内部始终以 FP32 执行）
- 允许 DCRNN 正常使用 AMP FP16 加速
