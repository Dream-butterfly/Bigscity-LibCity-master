# GNNTP/utils/INFO.md

跨模块通用工具集：参数解析、随机性控制、路径与日志管理、归一化器、DTW 与调参辅助。

## 关键文件

| 文件 | 字节 | 作用 |
|------|------|------|
| `utils.py` | 10.2K | **核心工具**：ensure_run_id、get_model/get_executor（按名定位）、get_run_dir/subdir、get_cache_root/subdir、get_logger、set_random_seed、build_run_id、align_checkpoint_config |
| `argument_list.py` | 6.1K | CLI 通用参数定义 + `add_general_args()` 函数（供所有 run 入口脚本调用） |
| `dataset.py` | 5.5K | 时间与坐标等数据辅助处理（时间特征、坐标转换） |
| `normalization.py` | 2.6K | 归一化器实现（StandardScalerNormalization、MinMax01Normalization 等） |
| `dtw.py` | 592B | DTW（Dynamic Time Warping）计算与缓存辅助 |
| `tune.py` | 152B | 调参过程辅助逻辑 |
| `disturbance.py` | 1.2K | 扰动/噪声注入工具 |
| `paths.py` | 306B | 路径常量（PROJECT_ROOT、OUTPUT_ROOT、CACHE_ROOT、RESOURCE_DATA_ROOT） |

## 核心函数详解

| 函数 | 位置 | 作用 |
|------|------|------|
| `get_model(config, data_feature)` | utils.py:65 | 通过 locator 定位并实例化模型 |
| `get_executor(config, model, data_feature)` | utils.py:65 | 通过 registry 定位并实例化执行器 |
| `ensure_run_id(config)` | utils.py:21 | 自动生成或校验 exp_id |
| `build_run_id(task, model, dataset)` | utils.py | 格式：`<timestamp>__<task>__<model>__<dataset>` |
| `get_logger(config)` | utils.py | 统一 logger 初始化（写入 outputs/<exp_id>/logs/run.log） |
| `align_checkpoint_config(config, ckpt_path, logger)` | utils.py | 自动检测 checkpoint 中的 num_cells/fuzzy_num_sets，覆盖 config 差异 |
| `add_general_args(parser)` | argument_list.py | 注入通用 CLI 参数（`--gpu`、`--batch_size`、`--lr`、`--max_epoch` 等） |

## 输入/输出

- **输入**：配置对象、路径参数、数据张量/统计信息
- **输出**：统一格式的工具结果（目录路径、logger、模型实例、执行器实例、归一化对象）

## 修改注意事项

1. 公共方法改名或签名变更会影响全局，必须先 grep 所有调用点
2. 路径规则（OUTPUT_ROOT、CACHE_ROOT、exp_id 格式）变更要兼容历史目录结构
3. `utils.py` 中的 get_model/get_executor 是框架核心枢纽，修改需谨慎
4. 新增 CLI 通用参数时需同步更新 `argument_list.py` 和 run 入口脚本

