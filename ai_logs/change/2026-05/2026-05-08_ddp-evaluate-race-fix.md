# 变更 #16: DDP 评估写入竞态修复 + val_loss 全局平均

**日期**: 2026-05-08
**类型**: 修复
**模型**: 全局（所有使用 DDP 训练的模型）
**关联**: 变更 #14 (DDP 多卡训练修复)

## 问题

4 卡 DDP 训练到 `evaluate()` 阶段，rank 1 崩溃（exit code 1），导致 torchrun 杀掉所有进程。

错误日志截取：
```
Start evaluating ...
W0508 torch/distributed/elastic/multiprocessing/api.py:1012] Sending process SIGTERM
E0508 failed (exitcode: 1) local_rank: 1
ChildFailedError: rank 1 exitcode: 1
```

### 根因

1. **`evaluate()` 写入竞态**（🔴致命）：基类 `traffic_state_executor.py:evaluate()` 在 DDP 下，所有 rank 同时写同一个 npz 文件和 evaluator 结果文件 —— 没有任何 `dist.gather` 汇总，也没有 rank 保护。

2. **`new_diffusion_fuzzy` val_loss 非全局**（🟡次要）：`_valid_epoch` 缺少 `dist.all_reduce`，val_loss 只是单卡平均值（STGformer/PDFormer/DCRNN 都有，唯独 diffusion 遗漏）。

## 修改

| 文件 | 变更 |
|------|------|
| `GNNTP/common/traffic_state_executor.py:evaluate` | + DDP 下 `all_gather` 汇总所有 rank 的预测结果 → 仅 rank 0 写文件；非 rank 0 返回 `{}` |
| `GNNTP/models/new/new_diffusion_fuzzy/executor.py:_valid_epoch` | + `dist.all_reduce(AVG)` 做全局 val_loss 平均 |

## 原理

```
evaluate() 旧行为（DDP 下）:
  rank 0: 推理 25% 数据 → np.savez (写 conflicts)
  rank 1: 推理 25% 数据 → np.savez (写 conflicts) → CRASH
  rank 2: 推理 25% 数据 → np.savez (写 conflicts)
  rank 3: 推理 25% 数据 → np.savez (写 conflicts)
  → 4 进程同时写同一个文件 → 竞态崩溃

evaluate() 新行为（DDP 下）:
  rank 0: 推理 25% 数据 ─┐
  rank 1: 推理 25% 数据 ─┤ all_gather → rank 0 写文件
  rank 2: 推理 25% 数据 ─┤               rank 1/2/3 return {}
  rank 3: 推理 25% 数据 ─┘
```

## 验证

- 4 卡 DDP 训练应能完整走过 `evaluate()` 不崩溃
- 评估结果 npz 文件应包含完整测试集（非 1/4）
- val_loss 应显示全局平均值（每张卡相同数字）
