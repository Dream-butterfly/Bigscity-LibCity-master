# eval checkpoint 自动检测 denoiser_layers

**日期**: 2026-05-14
**类型**: 修复
**模型**: new_diffusion_fuzzy_2
**影响文件**: `scripts/run/run_eval_checkpoint.py`

## 问题

评估旧 checkpoint 时，模型结构不匹配导致 `Missing key(s)` 错误：

```
RuntimeError: Error(s) in loading state_dict for NewDiffusion:
    Missing key(s) in state_dict: "noise_predictor.blocks.4.*", "noise_predictor.blocks.5.*", ...
```

根因：checkpoint 以 `denoiser_layers=4` 训练，但 `config.json` 已改为 6。
`load_model_with_epoch()` 调用 `model.load_state_dict()` 时 `strict=True`，新模型有 blocks 4-5 但 checkpoint 没有这些权重。

## 修改

`run_eval_checkpoint.py` 中添加自动检测机制：

1. **新增 `_detect_checkpoint_layers()`** — 在模型创建前，扫描 checkpoint state_dict 中 `noise_predictor.blocks.N.*` 键，推断实际层数 `max(N)+1`
2. **配置覆盖** — 若检测层数 ≠ `config.config["denoiser_layers"]`，自动覆盖配置以匹配 checkpoint，然后创建模型

流程：
```
创建 config → 建 checkpoint 路径 → torch.load 检测层数 → 覆盖 config → 创建模型 → load_state_dict ✅
```

## 兼容性

- 非 new_diffusion 系列模型：`_detect_checkpoint_layers()` 返回 `None`，不做任何覆盖
- checkpoint 不存在：返回 `None`，fallback 到原有错误路径（合理：文件不存在本身就该报错）
- 正常匹配时：零开销，仅多一次 `torch.load(weights_only=True)` 的 IO
