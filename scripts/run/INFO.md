# scripts/run/INFO.md

主流程运行入口层，把 CLI 参数转换为框架调用并触发标准流水线。同时被 Web 后端通过 subprocess 调用。

## 两条流水线

### 数据工件流水线（主流）

| 脚本 | 功能 | 产物 |
|------|------|------|
| `run_data_artifact.py` | 数据处理：构建数据集 → 提取 numpy 数组 → 写入数据工件 | `cache/data_artifacts/<id>/` |
| `run_train_artifact.py` | 训练：加载数据工件 → get_model → executor.train → evaluate | `outputs/<exp_id>/` |
| `run_resume_artifact.py` | 续训：从 effective_config.json + checkpoint 恢复 → 继续训练 | `outputs/<run_id>/` |
| `run_eval_checkpoint.py` | 评估：加载 checkpoint → 仅 evaluate（不训练） | `outputs/<run_id>/evaluate_cache/` |
| `run_hyper_artifact.py` | 调参：基于固定数据工件执行超参数搜索 | `outputs/<exp_id>/artifacts/hyper.result` |

### 旧流水线（兼容保留）

| 脚本 | 功能 | 状态 |
|------|------|------|
| `run_data_prep.py` | 仅构建数据与缓存 | 兼容保留 |
| `run_model.py` | 单模型训练 + 评估 | 兼容保留 |
| `run_resume.py` | 断点续训 | 兼容保留 |
| `run_hyper.py` | 超参数搜索 | 兼容保留 |

## 关键文件

| 文件 | 作用 |
|------|------|
| `run_data_artifact.py` | `build_dataset_runtime()` → `extract_xy_arrays()` → `write_data_artifact()` |
| `run_train_artifact.py` | `build_artifact_runtime(artifact_id)` → `get_model()` → `get_executor()` → `executor.train()` + `evaluate()` |
| `run_resume_artifact.py` | 从 `outputs/<run_id>/` 读取 `effective_config.json` + `run_meta.json` → 恢复 ConfigParser → 续训 |
| `run_eval_checkpoint.py` | 从 checkpoint 加载模型 → 自动检测对齐 denoiser_layers / num_cells → 纯 evaluate |
| `hyper_example.txt` | 调参空间示例配置 |

## 核心调用流程（以 train_artifact 为例）

```
ConfigParser(task, model, dataset, config_file, saved_model, train, other_args)
  ↓ 保存 effective_config.json
build_artifact_runtime(config, artifact_id)
  ↓ load_data_artifact() → 加载 npy → 重新包装 DataLoader → 返回 DataRuntime
get_model(config, data_feature)
get_executor(config, model, data_feature)
  ├── executor.train(train_loader, valid_loader)  → save checkpoint
  └── executor.evaluate(test_loader)              → write metrics + predictions
write_run_meta()
```

## 修改注意事项

1. 参数变更要同步 Web 参数映射与文档示例命令
2. 入口脚本保持”薄层”，业务逻辑优先下沉到 `GNNTP/` 模块
3. 新增入口脚本时，在根 `INFO.md` 与 `README.md` 补充用途
4. `run_resume_artifact.py` 依赖 `outputs/<run_id>/run_meta.json`（由 `run_train_artifact.py` 生成）
5. `run_eval_checkpoint.py` 使用 `force_reuse=True`，纯评估不校验数据签名

