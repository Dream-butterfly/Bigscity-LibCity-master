# libcity/INFO.md

## 目录职责

`libcity/` 是项目核心代码目录，负责配置解析、数据加载、模型构建、训练执行和评估流程。

## 关键内容

| 路径 | 说明 |
| --- | --- |
| `pipeline.py` | 训练/评估主流程入口（被 `scripts/run/run_model.py` 等调用） |
| `config_parser.py` | 配置加载与参数整合 |
| `common/` | 执行器、评估器、调参等通用模块 |
| `data/` | 数据集加载、预处理、dataloader 和注册机制 |
| `models/` | 模型抽象类、模型注册与具体模型实现 |
| `utils/` | 参数、数据、归一化、DTW、工具函数等通用能力 |

## 输入/输出

- **输入**：任务参数（task/model/dataset）、配置文件、`resource_data/` 下原子数据文件。  
- **输出**：训练日志、模型缓存、评估结果（写入 `outputs/` 和 `cache/`）。

## 调用关系

- 上层脚本 `scripts/run/run_model.py / scripts/run/run_hyper.py / scripts/run/run_resume.py / scripts/run/run_data_prep.py` 调用 `libcity`。  
- `pipeline.py` 会串联 `data -> models -> common(executor/evaluator)` 完成完整流程。  
- `utils/` 和 `common/registry_*` 提供跨模块复用与注册机制支持。

## 修改注意事项

1. 新增模型或执行器时，需同步注册逻辑（`registry.py`/定位器相关文件）。  
2. 配置键名要和 `config_parser.py`、执行器读取逻辑保持一致。  
3. 尽量复用 `utils/` 现有工具，避免重复实现数据处理或路径逻辑。  
4. 涉及输出路径时，优先遵循现有 `exp_id` 与子目录组织规则。

