# 更改 14：多卡训练（DDP）支持

> 时间: 2026-05-06 | 状态: 已实现 (Phase 1-4) | 基于计划: ai_logs/analysis/2026-05/multi_gpu_refactoring_plan.md

## 概述

从单卡（`cuda:{gpu_id}`）改造为 `torchrun` + `DistributedDataParallel` 多卡训练，兼容 2/4/8 卡，向后兼容单卡。

## 改动清单

### Phase 1: 基础设施

| 文件 | 改动 | 行数 |
|------|------|------|
| `GNNTP/utils/argument_list.py` | `general_arguments` 和 `hyper_arguments` 新增 `local_rank`, `world_size`, `dist_backend`, `scale_lr` | +24 |
| `GNNTP/config_parser.py` | `_init_device()` 检测 `LOCAL_RANK` 环境变量进入 DDP 模式；自动线性缩放学习率 | ~30 |
| `GNNTP/data/dataloader.py` | `generate_dataloader` 和 `generate_dataloader_pad` 新增 `train_sampler/eval_sampler/test_sampler` 参数；`shuffle` 与 `sampler` 互斥处理；`pin_memory=True` | ~25 |
| `GNNTP/data/runtime.py` | 新增 `_make_samplers()` 创建 `DistributedSampler`；`build_artifact_runtime` 传递 sampler 给 DataLoader；DDP 禁用 `pad_with_last_sample` | ~30 |

### Phase 2: 训练循环

| 文件 | 改动 | 行数 |
|------|------|------|
| `GNNTP/common/traffic_state_executor.py` | `is_distributed` 标志；`_unwrap_model()`/`_is_rank0()`/`_barrier()` helpers；`_autocast_context` 和 `GradScaler` 修复硬编码 `'cuda'`；`save_model`/`load_model`/`save_model_with_epoch`/`load_model_with_epoch` 使用 `_unwrap_model()`；`train()` 增加 rank 0 守卫 + `sampler.set_epoch`；`_train_epoch`/`_valid_epoch` 增加 loss `all_reduce` | ~50 |
| `GNNTP/pipeline.py` | 新增 `_maybe_wrap_ddp()` 进行 `init_process_group` + `DDP` 包装；`run_model()` 和 `objective_function()` 增加 rank 0 守卫和 `dist.barrier` | ~35 |
| `scripts/run/run_train_artifact.py` | 导入 `_maybe_wrap_ddp`；模型 DDP 包装；rank 0 守卫保存/评估；`dist.barrier/destroy` | ~15 |

### Phase 3: 模型修复（普通 tensor → register_buffer）

| 文件 | 改动 |
|------|------|
| `GNNTP/models/traffic_speed_prediction/STGCN/model.py` | `self.Lk` 从 `torch.FloatTensor(...).to(device)` 改为 `self.register_buffer('Lk', ...)` |
| `GNNTP/models/traffic_speed_prediction/STTN/model.py` | `self.adj_mx` 从 `torch.FloatTensor(...)` 改为 `self.register_buffer('adj_mx', ...)` |
| `GNNTP/models/traffic_flow_prediction/PDFormer/model.py` | `self.far_mask`, `self.geo_mask`, `self.sem_mask`, `self.pattern_keys` 改为 `register_buffer`；`.bool()` 操作在注册前完成 |

### Phase 4: 配置与子 executor

| 文件 | 改动 |
|------|------|
| `GNNTP/common/TrafficStateExecutor.json` | 新增 `"dist_backend": "nccl"`, `"scale_lr": true` |
| `GNNTP/models/*/executor.json` (3个) | 同上 |
| `STGformer/executor.py` | `best_state_dict` 使用 `_unwrap_model()`；rank guard 日志 |
| `PDFormer/executor.py` | `train()`/`_train_epoch()`/`_valid_epoch()` 增加 rank guard + all_reduce；`load_model_with_initial_ckpt` 修复 `map_location` 为 `self.device`，使用 `_unwrap_model()` |
| `DCRNN/executor.py` | `train()`/`_train_epoch()`/`_valid_epoch()` 增加 rank guard + all_reduce |

## 启动方式

```bash
# 单卡（向后兼容）
python run_model.py --gpu_id 0

# 多卡（新功能）
torchrun --nproc_per_node=2 run_model.py --model STGformer --dataset METR_LA
torchrun --nproc_per_node=4 run_train_artifact.py --model STGformer --dataset METR_LA
```

## 关键设计决策

1. **DDP 而非 DataParallel**：DDP 多进程无 GIL 瓶颈，梯度 all-reduce 与 backward 重叠
2. **`torchrun` 启动**：自动设置 `LOCAL_RANK`/`WORLD_SIZE`/`RANK` 环境变量
3. **向后兼容**：`LOCAL_RANK` 不存在时退化为原有单卡逻辑
4. **`find_unused_parameters=False`**：性能最优，若模型有条件分支需改为 `True`
5. **LR 线性缩放**：`lr = lr_base × world_size`，可在 executor.json 中通过 `scale_lr: false` 关闭

## 待完成（后续 Phase）

- Phase 5: Web 控制台适配（`web/train_web_fastapi.py` 命令行改为 `torchrun`）
- Phase 5: 独立实验脚本适配（`train_new_diffusion.py`, `train_new_diffusion_2.py`）
- Phase 6: 测试验证

---

*Co-Authored-By: CherryClaw*
