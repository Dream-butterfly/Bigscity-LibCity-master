# GNNTP/data/INFO.md

数据处理核心层：从原始数据到可训练批数据的完整转换，以及**数据工件（data artifact）的读写与签名校验**。

## 关键文件

| 路径 | 作用 |
| --- | --- |
| `factory.py` | 根据配置构建具体数据集对象 |
| `registry.py` | 数据集名称到类的注册与定位 |
| `dataloader.py` | 批加载逻辑封装 |
| `runtime.py` | **DataRuntime**：统一封装 train/valid/test DataLoader + data_feature，供工件流水线消费 |
| `artifact_io.py` | **数据工件核心**：build_data_artifact_id、compute_data_signature、extract_xy_arrays、write/load_data_artifact、serialize/deserialize_scaler、write/load_run_meta |
| `core/` | 底层数据结构（Batch、ListDataset） |
| `dataset/` | 具体数据集实现（TrafficStatePointDataset、PDFormerDataset 等） |

## 数据工件机制（artifact_io.py）

```
run_data_artifact.py 或 Web 数据处理
  ↓
build_dataset_runtime(config) → dataset.get_data() → DataLoaders
  ↓
extract_xy_arrays_from_loader(loader) → numpy arrays: (N, in_window, num_nodes, feat_dim)
  ↓
compute_data_signature(signature_payload) → 确定性哈希
  ↓
write_data_artifact(artifact_dir, train_x, train_y, valid_x, valid_y, test_x, test_y,
                    data_feature, scaler_payload, config_snapshot, ...)

→ cache/data_artifacts/<artifact_id>/
    ├── train_x.npy, train_y.npy
    ├── valid_x.npy, valid_y.npy
    ├── test_x.npy, test_y.npy
    ├── meta.json (data_signature, config_snapshot, scaler, data_feature)
    └── scaler.pkl (序列化的 scaler 对象)

消费端:
build_artifact_runtime(config, artifact_id)
  → load_data_artifact(artifact_id) → validate_artifact_for_config() → 重新生成 DataLoader
```

## 两条数据路径

| 路径 | 函数 | 数据来源 | 场景 |
|------|------|----------|------|
| 传统路径 | `build_dataset_runtime(config)` | 原始 `.dyna/.geo/.rel` 文件 | run_data_prep.py、旧入口 |
| 工件路径 | `build_artifact_runtime(config, artifact_id)` | 预处理的 `.npy` 文件 | run_train_artifact.py、Web 训练 |

## 输入/输出

- **输入**：`resource_data/<dataset>/` 原始文件 + 配置参数（窗口、切分比例、归一化策略）
- **输出**：DataRuntime（train/valid/test DataLoader + data_feature dict）

## 修改注意事项

1. 新增数据集实现后必须接入注册机制（`registry.py` + `factory.py`）
2. 数据签名相关变更（`compute_data_signature`）要评估对已有工件兼容性
3. 字段名与 shape 改动要同步检查模型输入约定和执行器读取逻辑
4. 缓存格式变更需考虑历史缓存兼容或提供重建路径
5. 数据预处理逻辑保持可复现（固定 seed 切分、归一化统计口径稳定）

