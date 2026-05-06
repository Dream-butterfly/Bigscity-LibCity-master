# 更改 14：多卡训练（DDP）支持

> 时间: 2026-05-06 | 状态: 已实现 (Phase 1-5) | 基于计划: ai_logs/analysis/2026-05/multi_gpu_refactoring_plan.md

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

### Phase 5: Web 控制台适配 ✅
**文件: 3**

| 文件 | 改动 |
|------|------|
| `web/train_web_fastapi.py` | CLI_OPTION_KEYS/TYPE 新增 `num_gpus`(int) `gpu_ids`(str)；`_run_training_background()` 检测 `num_gpus>1` 时自动切换为 `uv run torchrun --nproc_per_node=N --tee 0` 启动器；跳过 `gpu`/`gpu_id` 参数，注入 `--dist_backend nccl --scale_lr true`；`CUDA_VISIBLE_DEVICES` 通过 `Popen(env=...)` 传递 |
| `web/templates/train_web_fastapi.html` | 运行环境组重设计：`num_gpus` 下拉(1/2/4/8) + `gpu_single_group`(gpu开关+gpu_id) + `gpu_multi_group`(gpu_ids) |
| `web/static/main.js` | `CLI_FIELDS` 新增 `num_gpus`/`gpu_ids`；`onNumGpusChanged()` 根据 GPU 数量切换单/多卡控件的显隐并清理隐藏字段值；`applyDefaultToCliFields` 支持 num_gpus/gpu_ids |

**关键设计：**
- `torchrun --tee 0`：仅 rank 0 进程的 stdout 输出到控制台，其他 rank 写入文件 → Web 端日志自动只显示主进程输出
- 单卡/多卡控件互斥：切换 `num_gpus` 时自动清空隐藏字段，避免参数泄漏
- 数据预处理 (`_run_data_prep_background`) 不受影响：其前端使用独立的 `collectDataCliOptions()`，不含 GPU 字段

### 审计发现与修复（2026-05-06，当天修复）✅
系统性 DDP 正确性审计发现 6 个问题，全部修复：

| # | 严重度 | 文件 | 问题 | 修复 |
|---|--------|------|------|------|
| 1 | 致命 | `STTN/model.py:169-171` | `forward()` 中用普通 tensor 重新赋值 `self.adj_mx`，破坏 registered buffer | 改用局部变量 `adj_mx_norm` |
| 2 | 高 | `STGformer/model.py:181` | `GraphPropagate.gso` 为普通 Python 属性，`model.to(device)` 不会移动 | 改为 `register_buffer('gso', gso)` |
| 3 | 高 | `STGformer/executor.py` | 缺少 `sampler.set_epoch()` + `all_reduce` + rank guard | 补全三项，与基类一致 |
| 4 | 高 | `traffic_state_executor.py:398` | `hyper_tune` 块所有 rank 同时执行 → checkpoint 损坏 | 加 `and rank0` 条件 |
| 5 | 高 | `pipeline.py:80-94` | `objective_function()` 无 rank guard 无 cleanup | 加 rank guard + barrier + destroy |
| 6 | 高 | `run_train_artifact.py:127-149` | 所有 rank 调用 `evaluate()` + `write_run_meta()` → 文件写入竞态 | 加 rank guard |

## 待完成（后续 Phase）

- Phase 6: 测试验证（单卡回归、2/4 卡、checkpoint 兼容）

---

*Co-Authored-By: CherryClaw*
