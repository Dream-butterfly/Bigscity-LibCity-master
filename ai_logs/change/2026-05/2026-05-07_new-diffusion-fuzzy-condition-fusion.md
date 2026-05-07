# 更改 19：new_diffusion_fuzzy 条件前置融合 — 架构修复

**日期**: 2026-05-07
**状态**: 已修订（v2：time-mean → last-step）
**触发**: Critical 0 诊断 — 交叉注意力在噪声查询上失效导致模型学习边缘分布
**安全级别**: 🟡 确认区（仅 AttentionDenoiser 改动）

## 变更摘要

修复扩散模型条件注入时序缺陷：条件特征从"attention/graph conv 之后通过交叉注意力注入"改为"进入 blocks 之前直接拼接融合"。

## 修改文件

| 文件 | 变更 |
|------|------|
| `GNNTP/models/new/new_diffusion_fuzzy/model.py` | `AttentionDenoiser` 新增 1 层 + forward 新增 4 行 |

## 具体变更

### 1. `AttentionDenoiser.__init__` — 新增融合层
```python
self.condition_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
```
- 参数量：~18K（可忽略）

### 2. `AttentionDenoiser.forward` — 条件前置融合
```python
condition_pooled = condition_features.mean(dim=1, keepdim=True)
condition_pooled = condition_pooled.expand(-1, denoiser_input.size(1), -1, -1)
denoiser_input = self.condition_fusion(torch.cat([denoiser_input, condition_pooled], dim=-1))
```

### 数据流变化

```
之前：
noisy_future → proj → [temporal_attn, graph_conv(纯噪声)] → cross_attn(条件) → ...

之后：
noisy_future → proj → concat(条件pooled) → Linear融合 → [temporal_attn, graph_conv] → cross_attn(辅助) → ...
```

条件在第一次 temporal attention 和 graph conv 之前就已注入，确保所有噪声水平下都能访问条件信号。

## 设计要点

- `mean(dim=1)` 时间池化：保留每节点独立的空间上下文
- `expand(T_out)` ：每个未来步共享全局条件 + 各自位置编码
- `nn.Linear(192, 96)` ：可学习融合比例，无需手动调参
- 交叉注意力保留：提供细粒度时序条件补充
- DenoiserBlock 无改动

## 预期效果

- 模型从学习 p(Y) 转为学习 p(Y|X)
- 训练 loss 开始稳定下降
- 预测响应输入变化，不再固定在均值附近
