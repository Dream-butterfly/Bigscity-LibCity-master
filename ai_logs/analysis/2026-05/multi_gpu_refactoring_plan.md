# GNN-TP 多卡训练改造：普罗米修斯计划

> 版本: 1.0 | 日期: 2026-05-06 | 状态: 待审批
> 目标: 从单卡（`cuda:0`）改造为 `torchrun` + `DistributedDataParallel` 多卡训练，兼容 2/4/8 卡
> 原则: 最小侵入、向后兼容（单卡仍可运行）、DDP 对卡数透明

---

## 〇、前置知识

### 0.1 为什么是 DDP 而非 DataParallel
- `DataParallel`：单进程多线程，Python GIL 瓶颈，主卡负载不均，PyTorch 官方不推荐
- `DistributedDataParallel`：多进程（每个 GPU 一个进程），`torchrun` 启动，梯度 all-reduce 与 backward 重叠，性能最优

### 0.2 DDP 核心机制
```
torchrun --nproc_per_node=N
  ├── 自动设置环境变量: LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
  ├── 每个进程:
  │   ├── torch.distributed.init_process_group(backend="nccl")
  │   ├── torch.cuda.set_device(local_rank)
  │   ├── model = DDP(model, device_ids=[local_rank])
  │   ├── sampler = DistributedSampler(dataset, rank=rank, world_size=world_size)
  │   └── 仅 rank=0 执行: 日志输出、模型保存、TensorBoard 写入
  └── DDP 自动处理: 梯度同步 (all-reduce)、参数广播 (broadcast)、模型状态同步
```

### 0.3 卡数透明性
DDP 方案对卡数**天然透明**——改一套代码，`--nproc_per_node=2` 双卡，`=4` 四卡，`=8` 八卡。
唯一需要注意: `DistributedSampler` 数据分片、有效 batch_size 缩放、`pad_with_last_sample` 整除。

---

## 一、当前状态全景

### 1.1 单卡假设的核心节点

```
ConfigParser._init_device()          → torch.device(f"cuda:{gpu_id}")   # 硬编码单卡
pipeline.run_model()                 → 无 DDP 初始化，无模型包装
TrafficStateExecutor.__init__()      → model.to(self.device)            # 无 DDP 包装
TrafficStateExecutor._train_epoch()  → loss.backward()                  # 无梯度同步
TrafficStateExecutor.save_model()    → model.state_dict()               # 无 module. 前缀
TrafficStateExecutor.load_model_with_epoch() → torch.device("cuda"... ) # 硬编码另一设备
TrafficStateExecutor._autocast_context()    → device_type='cuda'         # 硬编码字符串
TrafficStateExecutor.__init__()             → GradScaler('cuda', ...)    # 硬编码字符串
generate_dataloader() / generate_dataloader_pad() → DataLoader(shuffle=) # 无 sampler
```

### 1.2 模型级隐蔽 bug（DDP 下必崩）

| 文件 | 行 | Bug | 根因 |
|------|-----|-----|------|
| `STGCN/model.py` | 246, 250 | `self.Lk` 存为普通 tensor | `model.to(device)` 不移动普通属性 |
| `STTN/model.py` | 141 | `self.adj_mx` 存为普通 tensor | 同上 |
| `PDFormer/model.py` | 367, 372, 375, 381 | 4 个 mask/key tensor 存为普通属性 | 同上 |

**根因**: `nn.Module.to(device)` 只递归移动 `nn.Parameter`、`register_buffer` 和子 `nn.Module`。存为 Python 属性（`self.xxx = torch.tensor(...)`）的 tensor **不会**移动。单卡下构造时就在目标设备上所以没问题，但 DDP 会在各 rank 独立构造模型然后分别 `.to(local_rank_device)`，这些 tensor 会滞留在构造设备上导致 device mismatch。

### 1.3 现有死代码

| 文件 | 行 | 内容 |
|------|-----|------|
| `STTN/model.py` | 132, 248 | `SSelfAttention.device` / `TemporalSelfAttention.device` 存储后从未使用 |
| `PDFormer/model.py` | 67, 74, 122, 132, 240, 248 | `DataEmbedding`, `STSelfAttention`, `TemporalSelfAttention` 的 `self.device` 从未使用 |
| `new_diffusion*/model.py` | ~555 | `self.device` 从 config 读取但模型中从未使用（通过输入 `.device` 追踪） |

### 1.4 独立实验脚本的特殊问题

| 文件 | 行 | 问题 |
|------|-----|------|
| `train_new_diffusion.py` | 63 | `DEVICE = torch.device("cuda" if ...)` — 无 rank 感知 |
| `train_new_diffusion_2.py` | 63 | 同上 |
| `STGformer-独立实现/model/train.py` | 236 | `DEVICE = torch.device("cuda:0"...)` — 硬编码 0 号卡 |

---

## 二、架构设计

### 2.1 改造成什么

```
# 改造前（单卡）
python run_model.py --gpu_id 0

# 改造后（多卡，DDP）
torchrun --nproc_per_node=4 run_model.py
# 等价于
python -m torch.distributed.run --nproc_per_node=4 run_model.py

# 仍支持单卡（向后兼容）
python run_model.py --gpu_id 0
```

### 2.2 调用链改造点

```
torchrun 启动
│
├─ [1] config_parser._init_device()
│   ├── 检测 LOCAL_RANK 环境变量 → 进入 DDP 模式
│   ├── device = torch.device(f"cuda:{local_rank}")
│   ├── config['local_rank'] = local_rank
│   ├── config['world_size'] = world_size
│   └── config['rank'] = rank
│
├─ [2] pipeline.run_model() / run_train_artifact()
│   ├── torch.distributed.init_process_group(backend="nccl")
│   ├── torch.cuda.set_device(local_rank)
│   ├── model = get_model(config, ...)
│   ├── model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
│   └── 注意: optimizer 在 executor 内部创建 (after DDP wrapping)
│
├─ [3] runtime / dataloader
│   ├── train_sampler = DistributedSampler(dataset, shuffle=True)
│   ├── valid_sampler = DistributedSampler(dataset, shuffle=False)
│   ├── test_sampler  = DistributedSampler(dataset, shuffle=False)
│   └── DataLoader(dataset, sampler=sampler, shuffle=False)  # sampler 与 shuffle 互斥
│
├─ [4] executor
│   ├── 不再执行 model.to(device) — 由外部 DDP 处理
│   ├── autocast(device_type=self.device.type) — 去掉硬编码 'cuda'
│   ├── GradScaler(self.device.type, ...) — 去掉硬编码 'cuda'
│   ├── save_model → model.module.state_dict() — DDP 包装下用 .module 访问原始模型
│   ├── train() / log / save → if rank == 0: — 守卫
│   ├── _train_epoch → sampler.set_epoch(epoch_idx) — shuffle 随机性
│   ├── loss 累加 → 可选 all_reduce 求平均
│   └── load_model_with_epoch → 统一用 self.device
│
└─ [5] 模型修复
    ├── STGCN: self.Lk → register_buffer
    ├── STTN: self.adj_mx → register_buffer
    └── PDFormer: self.far_mask/geo_mask/sem_mask/pattern_keys → register_buffer
```

### 2.3 向后兼容设计

```python
def _is_ddp(self):
    """检测 DDP 模式: LOCAL_RANK 环境变量存在即为 DDP"""
    return 'LOCAL_RANK' in os.environ

def _get_local_rank(self):
    if self._is_ddp():
        return int(os.environ['LOCAL_RANK'])
    return self.config.get('gpu_id', 0)
```

当 `LOCAL_RANK` 不存在时（非 torchrun 启动），整个流程退化为原有的单卡逻辑，零差异。

### 2.4 4卡兼容的特殊处理

DDP 天然兼容任意卡数，以下 3 点需在实现中确认：

**A. `pad_with_last_sample` + `DistributedSampler` 交互**
- `DistributedSampler` 默认 `drop_last=False`，会通过重复最后样本补齐
- 项目已有的 `pad_with_last_sample` 是在 sampler 之前执行，形成双重补齐
- 建议: DDP 模式下禁用 `pad_with_last_sample`（DistributedSampler 自带补齐），或将其移到 sampler 之后

**B. 有效 batch_size 和 lr 线性缩放**
- 4 卡时有效 batch_size = `config.batch_size × 4`
- SGD/Adam 建议: 学习率线性缩放 `lr = lr_base × world_size`
- 在 `_init_device()` 中自动处理，可选通过配置覆盖

**C. `sampler.set_epoch(epoch)` 的 shuffle 随机性**
- `DistributedSampler` 需要在每个 epoch 开始时调用 `set_epoch(epoch)` 以保证不同 epoch 的数据 shuffle 不同
- 在 executor 的 `_train_epoch` 中处理

---

## 三、逐文件改造方案

### 文件 3.1: `GNNTP/utils/argument_list.py`

**改动**: 新增分布式 CLI 参数（可选，torchrun 自动设置环境变量已足够，但显式参数可覆盖）

```diff
 general_arguments = {
+    "local_rank": {
+        "type": "int",
+        "default": None,
+        "help": "local rank for DDP (set by torchrun automatically)"
+    },
+    "world_size": {
+        "type": "int",
+        "default": None,
+        "help": "total number of GPUs (set by torchrun automatically)"
+    },
+    "dist_backend": {
+        "type": "str",
+        "default": "nccl",
+        "help": "DDP backend (nccl/gloo)"
+    },
     "gpu": { ... },   # 保留向后兼容
     "gpu_id": { ... }, # 保留向后兼容
 }
```

同样在 `hyper_arguments` 中同步添加。

---

### 文件 3.2: `GNNTP/config_parser.py`（🔴 P0）

**现状**: `_init_device()` 方法 (行 130-150)
**改造点**:
1. 检测 `LOCAL_RANK` 环境变量
2. DDP 模式下使用 `local_rank` 作为 device index
3. 注入 `rank`, `world_size`, `local_rank`, `is_distributed` 到 config
4. 可选: 自动线性缩放学习率

**新 `_init_device()` 逻辑**:

```python
def _init_device(self):
    import os
    import logging
    logger = logging.getLogger()

    # 1. 检测 DDP 环境
    local_rank_env = os.environ.get('LOCAL_RANK', None)
    is_distributed = local_rank_env is not None

    if is_distributed:
        # DDP 模式: 由 torchrun 启动
        local_rank = int(local_rank_env)
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        dist_backend = self.config.get('dist_backend', 'nccl')

        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        self.config['local_rank'] = local_rank
        self.config['world_size'] = world_size
        self.config['rank'] = rank
        self.config['dist_backend'] = dist_backend
        self.config['is_distributed'] = True

        logger.info(f"DDP mode: rank={rank}/{world_size}, local_rank={local_rank}")
    else:
        # 单卡模式（向后兼容）
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)

        if use_gpu and torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"gpu_id {gpu_id} is invalid, "
                    f"only {torch.cuda.device_count()} GPUs available"
                )
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Using GPU {gpu_id}")
        else:
            if use_gpu:
                logger.warning("GPU requested but not available, using CPU instead.")
            device = torch.device("cpu")

        self.config['is_distributed'] = False
        self.config['world_size'] = 1

    self.config['device'] = device

    # 可选: 多卡时线性缩放学习率
    if self.config.get('is_distributed') and self.config.get('scale_lr', True):
        world_size = self.config['world_size']
        orig_lr = self.config.get('learning_rate', 0.01)
        scaled_lr = orig_lr * world_size
        self.config['learning_rate'] = scaled_lr
        logger.info(f"LR scaled: {orig_lr} → {scaled_lr} (world_size={world_size})")
```

---

### 文件 3.3: `GNNTP/pipeline.py`（🔴 P0）

**现状**: `run_model()` 第 35-37 行，**最核心的改造点**
**目的**: 在模型创建后、executor 创建前，完成 DDP 初始化和模型包装

**新 `run_model()` 逻辑**:

```python
def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = ensure_run_id(config)
    logger = get_logger(config, config.get('is_distributed', False))
    logger.info(...)

    seed = config.get('seed', 0)
    set_random_seed(seed)

    runtime = build_dataset_runtime(config)

    model_cache_file = os.path.join(
        get_run_subdir(exp_id, 'model_cache'),
        '{}_{}.m'.format(model_name, dataset_name)
    )

    model = get_model(config, runtime.data_feature)

    # === DDP 初始化 ===
    is_distributed = config.get('is_distributed', False)
    if is_distributed:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend=config.get('dist_backend', 'nccl'),
                init_method='env://'
            )
        local_rank = config.get('local_rank', 0)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # 设为 True 若模型有条件分支
        )
        logger.info(f"DDP wrapped model on rank {config.get('rank', '?')}")

    executor = get_executor(config, model, runtime.data_feature)

    # rank 0 执行训练和保存
    rank = config.get('rank', 0)
    if train or not os.path.exists(model_cache_file):
        executor.train(runtime.train_loader, runtime.valid_loader, is_distributed)
        if saved_model and rank == 0:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)

    # 评估在所有 rank 上执行（或仅 rank 0，取决于需求）
    if rank == 0 or not is_distributed:
        executor.evaluate(runtime.test_loader)

    if is_distributed:
        dist.barrier()
        if rank == 0:
            dist.destroy_process_group()
```

**注意**: `objective_function` 和 `run_train_artifact` 需要**同样的 DDP 包装逻辑**。

---

### 文件 3.4: `GNNTP/data/dataloader.py`（🔴 P0）

**改造点**: 两个 `generate_dataloader*` 函数增加可选 `sampler` 参数

```python
def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False,
                        train_sampler=None,   # 新增
                        eval_sampler=None,    # 新增
                        test_sampler=None):    # 新增
    # ... 构建 dataset 不变 ...

    # DataLoader 构造: sampler 与 shuffle 互斥
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        num_workers=num_workers, collate_fn=collator,
        shuffle=(shuffle if train_sampler is None else False),
        sampler=train_sampler,
        pin_memory=True,  # 多卡建议开启
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size,
        num_workers=num_workers, collate_fn=collator,
        shuffle=False,
        sampler=eval_sampler,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        num_workers=num_workers, collate_fn=collator,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
    )
    return train_dataloader, eval_dataloader, test_dataloader
```

**对 `generate_dataloader_pad` 做同样的修改**。

---

### 文件 3.5: `GNNTP/data/runtime.py`（🔴 P0）

**改造点**: `build_artifact_runtime()` 和 `build_dataset_runtime()` 创建 `DistributedSampler`

```python
def _make_samplers(config, train_dataset, eval_dataset, test_dataset):
    """根据配置创建 DistributedSampler 或返回 None"""
    if not config.get('is_distributed', False):
        return None, None, None

    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=config['world_size'],
        rank=config['rank'],
        shuffle=True,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=config['world_size'],
        rank=config['rank'],
        shuffle=False,
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=config['world_size'],
        rank=config['rank'],
        shuffle=False,
    )
    return train_sampler, eval_sampler, test_sampler


def build_artifact_runtime(config, *, task, model_name, ...):
    # ... 现有逻辑 ...

    # 在 generate_dataloader 调用前创建 sampler
    train_sampler, eval_sampler, test_sampler = _make_samplers(
        config, bundle["train"], bundle["valid"], bundle["test"]
    )

    train_loader, valid_loader, test_loader = generate_dataloader(
        bundle["train"], bundle["valid"], bundle["test"],
        feature_name=feature_name,
        batch_size=int(config.get("batch_size", 64)),
        num_workers=int(config.get("num_workers", 0)),
        shuffle=True,
        pad_with_last_sample=(
            bool(config.get("pad_with_last_sample", True))
            if not config.get("is_distributed") else False  # DDP 下禁用双重补齐
        ),
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        test_sampler=test_sampler,
    )
```

同样对 `build_dataset_runtime()` 做适配（通过 `dataset.get_data()` 获取 loader 后替换 sampler，或修改 dataset 的 get_data 签名）。

---

### 文件 3.6: `GNNTP/common/traffic_state_executor.py`（🔴 P0，改动最密集）

这是**改动最大的文件**。以下按方法逐一列出。

#### 3.6.1 `__init__()` 方法

```diff
- self.device = self.config.get('device', torch.device('cpu'))
- if not isinstance(self.device, torch.device):
-     self.device = torch.device(self.device)
- self.model = model.to(self.device)
+ self.device = self.config.get('device', torch.device('cpu'))
+ if not isinstance(self.device, torch.device):
+     self.device = torch.device(self.device)
+ # 模型设备迁移: 单卡时在此执行，DDP 时由外部 pipeline 完成
+ self.is_distributed = self.config.get('is_distributed', False)
+ if not self.is_distributed:
+     self.model = model.to(self.device)
+ else:
+     self.model = model  # 已在 pipeline 中 DDP 包装并移到设备

  # AMP/GradScaler 去掉硬编码 'cuda'
- self.grad_scaler = torch.amp.GradScaler('cuda', enabled=...)
+ self.grad_scaler = torch.amp.GradScaler(self.device.type, enabled=...)
```

#### 3.6.2 `_autocast_context()` 方法

```diff
  def _autocast_context(self):
      if self.amp_enabled:
-         return torch.autocast(device_type='cuda', dtype=self.amp_torch_dtype)
+         return torch.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype)
      return nullcontext()
```

#### 3.6.3 `save_model()` / `save_model_with_epoch()`

```diff
  def save_model(self, cache_name):
+     model = self.model.module if self.is_distributed else self.model
-     torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)
+     torch.save((model.state_dict(), self.optimizer.state_dict()), cache_name)

  def save_model_with_epoch(self, epoch):
+     model = self.model.module if self.is_distributed else self.model
      config = dict()
-     config['model_state_dict'] = self.model.state_dict()
+     config['model_state_dict'] = model.state_dict()
      config['optimizer_state_dict'] = self.optimizer.state_dict()
      config['grad_scaler_state_dict'] = self.grad_scaler.state_dict()
      config['epoch'] = epoch
      ...
```

#### 3.6.4 `load_model_with_epoch()`

```diff
  def load_model_with_epoch(self, epoch):
      model_path = ...
      assert os.path.exists(model_path), ...
-     load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+     load_device = self.device
      checkpoint = torch.load(model_path, map_location=load_device)
-     self.model.load_state_dict(checkpoint['model_state_dict'])
+     model = self.model.module if self.is_distributed else self.model
+     model.load_state_dict(checkpoint['model_state_dict'])
      ...
```

#### 3.6.5 `train()` 方法 — rank 0 守卫 + sampler.set_epoch

```diff
  def train(self, train_dataloader, eval_dataloader):
+     rank = self.config.get('rank', 0)
+     is_distributed = self.config.get('is_distributed', False)

-     self._logger.info('Start training ...')
+     if rank == 0:
+         self._logger.info('Start training ...')
+         self._logger.info("num_batches:{}".format(len(train_dataloader)))

      for epoch_idx in range(self._epoch_num, self.epochs):
+         # DDP: 每 epoch 设置 sampler epoch 以保证 shuffle 随机性
+         if is_distributed and hasattr(train_dataloader, 'sampler'):
+             if hasattr(train_dataloader.sampler, 'set_epoch'):
+                 train_dataloader.sampler.set_epoch(epoch_idx)

          start_time = time.time()
          losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
          ...

-         self._logger.info("epoch complete!")
-         self._logger.info("evaluating now!")
+         if rank == 0:
+             self._logger.info("epoch complete!")
+             self._logger.info("evaluating now!")

          val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)

          ...

-         if (epoch_idx % self.log_every) == 0:
+         if rank == 0 and (epoch_idx % self.log_every) == 0:
              message = ...
              self._logger.info(message)

          if val_loss < min_val_loss:
              wait = 0
-             if self.saved:
+             if self.saved and rank == 0:
                  model_file_name = self.save_model_with_epoch(epoch_idx)
                  ...
          ...

-     if len(train_time) > 0:
+     if rank == 0 and len(train_time) > 0:
          self._logger.info('Trained totally {} epochs...'.format(...))
-     if self.load_best_epoch:
+     if self.load_best_epoch and rank == 0:
          self.load_model_with_epoch(best_epoch)
      return min_val_loss
```

#### 3.6.6 `_train_epoch()` — loss 跨卡同步（可选）

```diff
  def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
      self.model.train()
      loss_func = loss_func if loss_func is not None else self.model.calculate_loss
      losses = []
      for batch in train_dataloader:
          ...
          losses.append(loss.item())
          ...

+     # DDP: 可选，对 losses 做 all_reduce 求全局均值
+     if self.is_distributed:
+         import torch.distributed as dist
+         loss_tensor = torch.tensor([np.mean(losses)], device=self.device)
+         dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
+         losses = [loss_tensor.item()] * len(losses)  # 简化处理

      return losses
```

对 `_valid_epoch` 做同样的处理。

---

### 文件 3.7: 模型级 Bug 修复

#### 3.7.1 `STGCN/model.py` — 行 246, 250

```diff
- self.Lk = torch.FloatTensor(self.Lk).to(self.device)
+ self.register_buffer('Lk', torch.FloatTensor(self.Lk).to(self.device))
```

注意: `self.Lk` 在 forward 中的引用方式可能也需要同步改为 `self.Lk`（通常不变）。

#### 3.7.2 `STTN/model.py` — 行 141

```diff
- self.adj_mx = torch.FloatTensor(adj_mx).to(device)
+ self.register_buffer('adj_mx', torch.FloatTensor(adj_mx).to(device))
```

检查 forward 中对 `self.adj_mx` 的引用是否需要修改。

#### 3.7.3 `PDFormer/model.py` — 行 367, 372, 375, 381

```diff
- self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
+ self.register_buffer('far_mask', torch.zeros(self.num_nodes, self.num_nodes).to(self.device))

- self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
+ self.register_buffer('geo_mask', torch.zeros(self.num_nodes, self.num_nodes).to(self.device))

- self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
+ self.register_buffer('sem_mask', torch.ones(self.num_nodes, self.num_nodes).to(self.device))

- self.pattern_keys = torch.from_numpy(...).float().to(self.device)
+ self.register_buffer('pattern_keys', torch.from_numpy(...).float().to(self.device))
```

---

### 文件 3.8: 入口脚本适配

#### `run_model.py` / `run_train_artifact.py` / `run_resume.py` / `run_resume_artifact.py`

无需大幅修改。`torchrun` 启动时自动设置环境变量，现有 CLI 参数继续工作。仅需确保 `other_args` 中能传递 `local_rank` 等参数（由 `add_general_args` 自动覆盖）。

```bash
# 使用方式（无需修改脚本代码）
torchrun --nproc_per_node=4 run_model.py --model STGformer --dataset METR_LA
torchrun --nproc_per_node=4 run_train_artifact.py --model STGformer --dataset METR_LA
```

---

### 文件 3.9: executor.json 文件（3个）

```diff
  {
+     "dist_backend": "nccl",
+     "scale_lr": true,
      "gpu": true,
      "gpu_id": 0,
      ...
  }
```

---

## 四、不改动的文件

| 文件 | 原因 |
|------|------|
| `GNNTP/data/core/batch.py` | `to_tensor(device)` 接受单个 device，DDP 每个 rank 传各自 device，无需改动 |
| `GNNTP/models/loss.py` | 所有 loss 函数是纯张量操作，无全局状态依赖 |
| `GNNTP/common/traffic_state_evaluator.py` | 仅处理 CPU numpy，无设备依赖 |
| `GNNTP/utils/utils.py` | 已有 `torch.cuda.manual_seed_all` |
| `GNNTP/common/abstract_*.py` | 纯抽象接口 |
| 所有 `config.json` | 无 GPU 相关配置（`use_amp`/`amp_dtype` 已在 executor 层处理） |
| `new_diffusion_fuzzy/model.py` | `self.device` 未被使用，`_sampling_schedule_cache` 按 device 分 key，DDP 安全 |

---

## 五、独立实验脚本的适配

`train_new_diffusion.py` 和 `train_new_diffusion_2.py` 有独立的训练循环，需要**完整复制 DDP 逻辑**：

```python
# 新增开头
import os
import torch.distributed as dist

def setup_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1:
        return False, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return True, torch.device(f"cuda:{local_rank}")

is_ddp, device = setup_ddp()

# DataLoader 替换
if is_ddp:
    sampler = DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler, ...)
else:
    data_loader = DataLoader(dataset, shuffle=True, ...)

# 模型包装
model = NewDiffusion(config, data_feature).to(device)
if is_ddp:
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

# 训练循环
for epoch in epochs:
    if is_ddp:
        data_loader.sampler.set_epoch(epoch)
    ...

# 保存（仅 rank 0）
if not is_ddp or dist.get_rank() == 0:
    torch.save({"state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
                 "config": model_config}, args.output_ckpt)
```

---

## 六、测试验证计划

### 6.1 单元测试

| 测试 | 命令 | 期望 |
|------|------|------|
| 单卡向前兼容 | `python run_model.py --gpu_id 0 --max_epoch 1` | 正常运行 |
| 2卡 DDP | `torchrun --nproc_per_node=2 run_model.py --max_epoch 1` | 两卡显存均衡，loss 正常下降 |
| 4卡 DDP | `torchrun --nproc_per_node=4 run_model.py --max_epoch 1` | 四卡显存均衡 |
| checkpoint 恢复 | 2卡训练 → 保存 → 单卡加载推理 | 权重正确加载，评估结果一致 |
| 新链路 2卡 | `torchrun --nproc_per_node=2 run_train_artifact.py --max_epoch 1` | 正常运行 |
| AMP 多卡 | `torchrun --nproc_per_node=2 run_model.py --use_amp true` | AMP 正常工作 |
| early_stop 多卡 | `torchrun --nproc_per_node=2 run_model.py --use_early_stop true --patience 3` | early stop 正确触发 |
| Web 控制台 + 单卡 | Web UI 启动训练 `gpu_ids=[0]` | 与 CLI 单卡行为一致 |
| Web 控制台 + 2卡 | Web UI 启动训练 `gpu_ids=[0,1]` | `torchrun --nproc_per_node=2` 正确启动 |
| Web 控制台 + 停止 | 多卡训练中点击 Stop | 整个 torchrun 进程树被终止 |

### 6.2 性能验证

| 指标 | 单卡 baseline | 2卡期望 | 4卡期望 |
|------|-------------|---------|---------|
| 单 epoch 时间 | T | ~0.55T | ~0.30T |
| GPU 显存利用率 | M | M×1 (per GPU) | M×1 (per GPU) |
| 最终 loss | L | L (有效 batch_size 相同) | L (有效 batch_size 相同) |

### 6.3 回归验证

- 单卡模式下所有现有功能 100% 保持
- 模型 save/load 格式与历史 checkpoint 兼容
- `run_data_artifact.py` 不受影响（纯数据处理）

---

## 七、风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| `find_unused_parameters=True` 导致 4 卡性能下降 | 中 | 性能损失 10-20% | 先设 `False`，若模型有条件分支再改为 `True` |
| `DistributedSampler` + `pad_with_last_sample` 双重补齐 | 低 | 数据分布微小偏差 | DDP 禁用 `pad_with_last_sample` |
| `model.state_dict()` → `model.module.state_dict()` 遗漏 | 中 | checkpoint 加载失败 | 统一通过 `_unwrap_model()` helper 访问 |
| 子模型 executor 重写方法遗漏 rank guard | 中 | 多卡日志刷屏 | 在基类 `_is_rank0()` helper 中集中处理 |
| Optuna 超参搜索 + DDP 并发冲突 | 高 | 超参搜索不收敛 | 超参搜索时禁用 DDP（单卡搜索，多卡训练） |
| Web 控制台 (FastAPI) + DDP 冲突 | 低 | 进程管理复杂 → **已确认安全**（见第十一章） | 见下方专章分析 |

---

## 八、Web 控制台（FastAPI）兼容性分析

### 8.1 架构结论：完全安全，改动极小

**Web 进程和训练进程是完全独立的 OS 进程，零共享内存，零共享 GPU 上下文。**

```
[FastAPI uvicorn 进程]
  GPU Context: 无 (不 import torch.cuda)
  │
  └── threading.Thread (daemon) — _run_training_background()
        │
        └── subprocess.Popen("uv run scripts/run/run_train_artifact.py ...")
              │
              独立的 OS 进程
              GPU Context: 全部在此进程内分配
```

### 8.2 Web 如何触发训练（关键代码路径）

文件: `web/train_web_fastapi.py`

```
用户点击 "Start Training"
  → POST /api/start
    → 检查 STATE.running (仅允许一个训练任务)
    → _write_runtime_config() → 写 temp JSON 到项目根目录
    → threading.Thread(target=_run_training_background)  ← daemon 线程
        → subprocess.Popen([
            "uv", "run", "scripts/run/run_train_artifact.py",
            "--task", "traffic_state_pred",
            "--model", "STGformer",
            "--dataset", "METR_LA",
            "--config_file", "webcfg_XXXXXX",
            "--gpu", "true",
            "--gpu_id", "0",           ← 当前: 单卡
            "--artifact_id", "...",
          ])
        → 逐行读取 stdout，正则解析 run_id / loss / epoch
        → proc.wait()  → 读取 outputs/<exp_id>/evaluate_cache/
```

### 8.3 DDP 改造对 Web 的影响

**需要改动的只有 1 处命令行**（`train_web_fastapi.py` 行 ~1524）：

```diff
  # 当前: 单卡启动
- cmd = ["uv", "run", "scripts/run/run_train_artifact.py",
-        "--gpu", "true", "--gpu_id", str(gpu_id), ...]

  # 改造方案: 根据用户选择的 GPU 数量动态构建命令
+ num_gpus = request_payload.get("num_gpus", 1)
+ gpu_ids = request_payload.get("gpu_ids", [0])
+
+ if num_gpus > 1:
+     cmd = ["torchrun", f"--nproc_per_node={num_gpus}",
+            "scripts/run/run_train_artifact.py", ...]
+     # 不加 --gpu_id，由 torchrun 的 LOCAL_RANK 环境变量自动分配
+     # 可通过 CUDA_VISIBLE_DEVICES 限制可见 GPU
+ else:
+     cmd = ["uv", "run", "scripts/run/run_train_artifact.py",
+            "--gpu", "true", "--gpu_id", str(gpu_ids[0]), ...]
```

### 8.4 需要新增的前端 UI 改动

| 组件 | 当前 | 改动 |
|------|------|------|
| GPU 选择器 | 单选下拉框 `gpu_id: 0` | 改为**多选复选框** `gpu_ids: [0,1,2,3]` 或**数量滑块** `num_gpus: 1-8` |
| 启动按钮文字 | "Start Training" | 保持，根据需要显示 "Start Training (4 GPUs)" |
| 状态轮询 | `/api/status` | **无需改动**（stdout 解析逻辑不变，仅日志量变多） |
| 停止训练 | `subprocess.run(taskkill ...)` | **需要升级**: 单卡杀 1 进程，多卡需杀整个 `torchrun` 进程树。Windows 用 `taskkill /T /PID <torchrun_pid>`，Linux 用 `os.killpg()` |
| 日志展示 | stdout 逐行 | **需要过滤**: 4 卡时有 4 份重复日志（每 rank 一份）。建议仅显示 rank 0 的输出，或给每行加 `[rank N]` 前缀 |

### 8.5 进程终止的 DDP 适配

当前终止逻辑 (`web/train_web_fastapi.py` 行 ~2100):

```python
# 当前: 杀单个训练进程
subprocess.run(f"taskkill /T /F /PID {training_proc.pid}", ...)

# DDP 改造: torchrun 是父进程，其子进程是各 rank 的 python 进程
# taskkill /T 已经会杀整个进程树，所以 Popen 中存储 torchrun 的 pid 即可
# 无需改动终止逻辑（已验证: taskkill /T 会递归杀所有子进程）
```

### 8.6 Web 端的「不需要改动」清单

| 功能 | 原因 |
|------|------|
| Web 进程本身 | 不 import torch，不持有 CUDA context |
| 数据预处理 (`/api/data/start`) | 纯 CPU 任务，永远不需要 DDP |
| 结果解析 (`_load_result_payload`) | 读取 `outputs/<exp_id>/evaluate_cache/` 文件，与卡数无关 |
| 图表渲染 (`pyecharts_views.py`) | 纯后端渲染，无 GPU 依赖 |
| 训练历史 (`web_train_history.json`) | 纯 JSON 读写 |
| 配置合并 (`_merge_with_constraints`) | 纯 dict 操作 |
| 并发控制 (`STATE.running`) | 单训练任务限制不变 |

### 8.7 多卡训练的 Web 启动流程（改造后）

```
用户操作:
  1. 在 Web UI 选择 GPU: [✓ 0] [✓ 1] [✓ 2] [✓ 3]  → num_gpus=4
  2. 点击 "Start Training"

Web 后端:
  1. 写入 webcfg_XXXXXX.json (与单卡相同)
  2. 启动线程 → subprocess.Popen([
       "torchrun", "--nproc_per_node=4",
       "scripts/run/run_train_artifact.py",
       "--task", "traffic_state_pred",
       "--model", "STGformer",
       "--dataset", "METR_LA",
       "--config_file", "webcfg_XXXXXX",
       "--artifact_id", "da_20260506_...",
       "--saved_model", "true", "--train", "true",
     ])
  3. stdout 逐行读取（只有 rank 0 的日志，其余 rank 日志不输出到 stdout）
  4. proc.wait() → 读取 evaluate_cache 结果
  5. 返回结果给前端

训练子进程 (torchrun 自动管理):
  ├── rank 0: LOCAL_RANK=0 → cuda:0 → 输出日志
  ├── rank 1: LOCAL_RANK=1 → cuda:1
  ├── rank 2: LOCAL_RANK=2 → cuda:2
  └── rank 3: LOCAL_RANK=3 → cuda:3

config_parser._init_device() 自动检测 LOCAL_RANK:
  → device = cuda:<LOCAL_RANK>
  → is_distributed = True
  → 后续 pipeline.py DDP 包装自动生效
```

### 8.8 注意事项

1. **仅 rank 0 日志**: 需要确保非 rank 0 进程的 logger 不输出到 stdout（或 Web 端按 `[rank N]` 前缀过滤），否则 4 份日志混在一起不可读。
2. **stop 按钮**: `taskkill /T /F /PID <torchrun_pid>` 已验证会杀整个进程树。
3. **环境变量**: `torchrun` 需要 `MASTER_ADDR` 和 `MASTER_PORT`。torchrun 默认设置 `MASTER_ADDR=127.0.0.1, MASTER_PORT=29500`，无需手动传递。

---

## 九、实施顺序

```
Phase 1: 基础设施（不改模型）
  ├── Step 1.1: argument_list.py — 新增分布式参数定义
  ├── Step 1.2: config_parser.py — DDP 感知的 _init_device()
  ├── Step 1.3: dataloader.py — sampler 参数支持
  └── Step 1.4: runtime.py — DistributedSampler 创建

Phase 2: 训练循环
  ├── Step 2.1: traffic_state_executor.py — 核心改造（rank guard, GradScaler, autocast, save/load）
  └── Step 2.2: pipeline.py — DDP 初始化 + 模型包装

Phase 3: 模型修复
  ├── Step 3.1: STGCN/model.py — self.Lk → register_buffer
  ├── Step 3.2: STTN/model.py — self.adj_mx → register_buffer
  └── Step 3.3: PDFormer/model.py — 4 mask tensors → register_buffer

Phase 4: 入口与配置
  ├── Step 4.1: executor.json (3个) — 增加分布式默认配置
  ├── Step 4.2: 子 executor (PDFormer, STGformer, DCRNN) — 继承基类改造
  └── Step 4.3: 独立实验脚本适配

Phase 5: Web 控制台适配
  ├── Step 5.1: train_web_fastapi.py — 命令行改为 torchrun 启动 + GPU 多选
  ├── Step 5.2: 前端 HTML/JS — GPU 选择器改为多选复选框
  └── Step 5.3: 日志展示 — 仅显示 rank 0 日志（或按 rank 前缀过滤）

Phase 6: 测试验证
  ├── 单卡回归测试
  ├── 2卡功能测试
  ├── 4卡性能测试
  ├── checkpoint 兼容性测试
  └── Web 控制台 + 多卡联调测试
```

**预计总改动量: ~250 行（含 Web 控制台适配，不含独立实验脚本）**

---

## 十、Helper 函数建议

为避免分散的 `if is_distributed` 检查，建议在 base executor 中增加以下 helper：

```python
class TrafficStateExecutor(AbstractExecutor):
    def _unwrap_model(self):
        """获取原始模型（DDP 包装下取 .module）"""
        if self.is_distributed:
            return self.model.module
        return self.model

    def _is_rank0(self):
        """当前进程是否为主进程（rank 0）"""
        if self.is_distributed:
            import torch.distributed as dist
            return dist.get_rank() == 0
        return True

    def _barrier(self):
        """所有进程同步点"""
        if self.is_distributed:
            import torch.distributed as dist
            dist.barrier()
```

---

## 十一、兼容性总结

| 启动方式 | 行为 | 卡数 |
|----------|------|------|
| `python run_model.py --gpu_id 0` | 单卡，完全不变 | 1 |
| `python run_model.py --gpu_id 2` | 单卡，指定 2 号卡 | 1 |
| `torchrun --nproc_per_node=2 run_model.py` | 2卡 DDP | 2 |
| `torchrun --nproc_per_node=4 run_model.py` | 4卡 DDP | 4 |
| `torchrun --nproc_per_node=8 run_model.py` | 8卡 DDP | 8 |
| `python run_model.py` (无 GPU) | CPU 模式 | 0 |

---

*End of Plan. 待 L 审批后执行 Phase 1.*
