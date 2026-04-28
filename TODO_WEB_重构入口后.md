# Web 重构入口改动清单

> Phase 0 (数据-训练解耦) 改动文件汇总，供 Web 前端重构同步参考
> 生成时间: 2026-04-28

---

## Phase 0 完整改动

### 修改 (4)

| 文件 | 改动 |
|------|------|
| `GNNTP/config_parser.py` | `model` 参数改为可选 (默认 None)；无模型时跳过 model/executor/evaluator 配置加载；`dataset_class` 需显式提供 |
| `GNNTP/data/artifact_io.py` | +`list_artifact_metas(dataset, dataset_class)` 扫描工件目录；+`find_latest_artifact(dataset, dataset_class, config)` 自动匹配最新签名一致的工件 |
| `scripts/run/run_data_artifact.py` | +`--dataset_class` CLI 参数；`run_data_artifact()` 函数签名 +`dataset_class`；无 `--model` 时纯数据集驱动 |
| `scripts/run/run_train_artifact.py` | +`--artifact_list` CLI；不传 `--artifact_id` 或 `--artifact_path` 时自动扫描匹配；函数签名 +`artifact_latest`, `artifact_list` |

### 重写 (1)

| 文件 | 改动 |
|------|------|
| `scripts/run/run_data_prep.py` | 重写为 `run_data_artifact` 的薄封装，不再独立实现逻辑 |

### 新建 (2)

| 文件 | 作用 |
|------|------|
| `scripts/experiments/train_new_diffusion_fuzzy.py` | FuzDiff 实验入口，固化 `model_name="new_diffusion_fuzzy"`，自动匹配工件 |
| `ai_logs/change/2026-04/2026-04-28_phase0-data-decouple-v2.md` | Phase 0 变更记录 |

### 文档 (2)

| 文件 | 改动 |
|------|------|
| `PLAN_PROMETHEUS.md` | Phase 0 标记完成 |
| `INFO.md` | 分支名修正，结构重构 |

---

## Web 关联影响分析

### 当前 Web 调用方式

Web 通过**子进程** (`subprocess.Popen`) 调用脚本，非直接 import 函数：

```python
# web/train_web_fastapi.py
RUN_TRAIN_ARTIFACT_ENTRY = str(RUN_SCRIPTS_DIR / "run_train_artifact.py")
RUN_DATA_PREP_ENTRY = str(RUN_SCRIPTS_DIR / "run_data_artifact.py")
```

→ **函数签名变化不影响 Web**

### 新的 CLI 参数（Web 可能需要暴露）

| 脚本 | 新参数 | 说明 |
|------|--------|------|
| `run_data_artifact.py` | `--dataset_class` | dataset 类名，无模型时必须 |
| `run_data_artifact.py` | `--model` (可选) | 改为可选，默认 None |
| `run_train_artifact.py` | `--artifact_list` | 列出可用工件后退出 |

### 新链路流程

```
┌─ 数据预处理 ─────────────────────────────────────┐
│ run_data_prep.py                                   │
│   --task traffic_state_pred                        │
│   --dataset METR_LA                                │
│   --dataset_class TrafficStatePointDataset         │ ← 新增必填
│     → cache/data_artifacts/da_<ts>__METR_LA__.../ │
└────────────────────────────────────────────────────┘
                        │
                        ▼
┌─ 模型训练（自动匹配工件） ────────────────────────┐
│ run_train_artifact.py                              │
│   --task traffic_state_pred                        │
│   --model new_diffusion_fuzzy                      │
│   --dataset METR_LA                                │
│   # 不传 --artifact_id → 自动扫描匹配              │ ← 新行为
└────────────────────────────────────────────────────┘
```

### Web 重构建议

1. 数据预处理页面需新增 `dataset_class` 选择器（下拉：TrafficStatePointDataset 等）
2. 训练页面可新增 artifact 自动匹配开关，替代手动输入 artifact_id
3. 可新增 artifact 列表查看页面（调用 `--artifact_list`）
4. 旧 `--model` 在数据预处理中可改为可选
