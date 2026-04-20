# GNN-TP 项目说明（根目录）

本仓库是基于 **LibCity** 的交通时空预测实验工程，聚焦于图神经网络相关模型的训练、调参、续训和结果分析。

## 快速入口

```bash
# 安装依赖（推荐）
uv sync

# 仅做数据准备/缓存
uv run python scripts/run/run_data_prep.py --task traffic_state_pred --model GRU --dataset METR_LA

# 训练并评估单个模型
uv run python scripts/run/run_model.py --task traffic_state_pred --model GRU --dataset METR_LA

# 超参数搜索
uv run python scripts/run/run_hyper.py --task traffic_state_pred --model GRU --dataset METR_LA --params_file hyper_example.txt

# 续训并评估
uv run python scripts/run/run_resume.py --task traffic_state_pred --model GRU --dataset METR_LA

# 启动 Web 训练控制台
uv run python run_web.py --host 127.0.0.1 --port 7817
```

## 根目录结构

| 路径                  | 类型     | 作用                                                  |
|---------------------|--------|-----------------------------------------------------|
| `libcity/`          | 核心代码目录 | 配置解析、数据加载、模型定义、训练执行与评估流水线                           |
| `web/`              | Web 目录 | FastAPI 后端、模板与静态资源（训练控制台）                           |
| `resource_data/`    | 数据资源目录 | 数据集相关资源（如 `METR_LA/`、`PEMSD4/`）                     |
| `scripts/`          | 脚本目录   | 辅助工具脚本与运行入口脚本                                       |
| `scripts/run/`      | 运行入口目录 | `run_model/run_data_prep/run_hyper/run_resume` 入口脚本 |
| `scripts/experiments/` | 实验脚本目录 | 不走主训练流水线的独立实验入口 |
| `test/`             | 测试目录   | 传统基线模型测试与测试辅助代码                                     |
| `cache/`            | 运行缓存目录 | 实验过程缓存与中间产物（通常由程序自动生成）                              |
| `outputs/`          | 输出目录   | 训练输出、评估结果、模型产物等（通常由程序自动生成）                          |
| `run_web.py`        | Web 入口 | 启动 Web 训练控制台                                        |
| `hyper_example.txt` | 配置文件   | 调参空间示例                                              |
| `pyproject.toml`    | 项目配置   | 依赖、Python 版本、工具链配置                                  |
| `uv.lock`           | 锁文件    | `uv` 依赖锁定结果                                         |
| `readme_zh.md`      | 上游说明   | LibCity 原始中文说明文档（参考）                                |
| `AI_EDIT_LOG.md`    | 修改记录   | 通过 AI 修改的记录日志（供之后进行迭代修改的参考）                         |
| `AI代码准则.md`     | 代码规范   | 供 AI 修改代码时参考的规范准则（供之后进行迭代修改的参考）                     |

## AI/开发者阅读顺序（建议）

1. `scripts/run/run_model.py`：确认训练入口参数与调用方式。
2. `libcity/pipeline.py`：理解训练/评估主流程。
3. `libcity/config_parser.py` 与 `libcity/common/`：理解配置与执行器。
4. `libcity/data/`：理解数据加载、切分、特征构建。
5. `libcity/models/`：定位具体模型实现。

## 目录内说明文件规范（用于后续逐层补充）

建议在每个关键目录下新增 `INFO.md`，统一包含以下 5 个小节，便于 AI 和人类快速理解：

1. 目录职责（做什么）
2. 关键文件（入口与核心模块）
3. 输入/输出（依赖的数据与产物）
4. 调用关系（被谁调用、调用谁）
5. 修改注意事项（易错点与兼容性约束）

