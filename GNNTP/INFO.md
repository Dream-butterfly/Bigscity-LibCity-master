# GNNTP/INFO.md

GNNTP（Graph Neural Network for Traffic Prediction）核心框架层，负责将”任务参数 → 配置 → 数据 → 模型 → 执行器 → 评估 → 落盘”串成完整实验流程。支持**两条流水线**：传统 pipeline（直接构建数据集）和**数据工件流水线**（先处理数据为工件再消费）。

## 关键模块

| 路径 | 作用 |
| --- | --- |
| `config_parser.py` | 配置合并（默认配置 + 任务配置 + CLI 覆盖）与运行时参数解析 |
| `pipeline.py` | 传统流水线编排（构建数据 → 模型 → 执行器），供旧入口使用 |
| `common/` | 执行器抽象 + TrafficStateExecutor + TrafficStateEvaluator + 注册机制 + 调参 |
| `data/` | 数据处理核心：数据集实现、DataLoader、**artifact_io（数据工件读写与签名校验）** |
| `data/artifact_io.py` | **数据工件机制**（核心）：npy 读写、签名计算、scalar 序列化、run_meta 管理 |
| `data/runtime.py` | **DataRuntime**：统一封装 DataLoader + data_feature，供工件流水线消费 |
| `models/` | 模型实现：分成 traffic_speed_prediction / traffic_flow_prediction / new / baseline 四层 |
| `models/abstract_model.py` | 模型抽象基类 |
| `models/abstract_traffic_state_model.py` | 交通状态任务模型抽象层 |
| `models/locator.py` | 模型按名称定位与注册 |
| `models/registry.py` | 模型类注册（与 locator 配合） |
| `models/loss.py` | 损失函数与训练辅助损失工具 |
| `utils/` | 工具集：路径管理、logger、exp_id 生成、随机种子、归一化器、CLI 参数定义 |

## 输入/输出

- **输入**：`task/model/dataset`、配置文件（`config.json`）+ CLI 覆盖参数、`resource_data/` 原始数据
- **数据工件输入**：`cache/data_artifacts/<id>/` 中的预训练 npy 数组
- **输出**：`cache/` 中间缓存 + `outputs/<exp_id>/` 实验产物（模型、指标、日志、调参结果）

## 调用关系

```
scripts/run/*.py 或 Web 子进程
       │
       ▼
┌──────────────┐
│ config_parser│ ── 合并配置
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────┐
│  data/build_dataset_runtime()  (传统)  │
│  data/build_artifact_runtime() (工件)  │
│      └── data/dataset/  具体数据集实现  │
│      └── data/artifact_io  工件读写    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  models/locator → get_model()          │
│  common/registry → get_executor()      │
│      └── common/执行器 训练+评估        │
└────────────────────────────────────────┘
                   │
                   ▼
              outputs/<exp_id>/
```

## 修改注意事项

1. 新增模型/执行器/评估器必须补齐注册信息（manifest.json + registry.py 或 locator 扫描），否则运行期找不到
2. 配置键名改动要同时检查：config.json 模板、`config_parser.py`、执行器读取位置
3. 数据工件签名相关变更（`artifact_io.py`）要评估对已有工件的兼容性
4. 工具函数变更需评估跨模块调用影响，`utils/` 中避免引入业务逻辑耦合
5. 输出路径变更需兼容已有 `exp_id` 目录结构

## 相关 INFO.md

- `common/INFO.md` — 执行器/评估器详情
- `data/INFO.md` — 数据处理流水线详情
- `data/artifact_io.py` — 数据工件机制
- `models/INFO.md` — 模型分层与注册详情
- `utils/INFO.md` — 工具函数详情

