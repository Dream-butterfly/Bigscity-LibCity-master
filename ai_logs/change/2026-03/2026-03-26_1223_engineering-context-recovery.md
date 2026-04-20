### 更改 2

时间：2026-03-26 12:23:00 +08:00

变更类型：结构重构（refactor） / 实验配置或训练流程变更（experiment） / 环境或工具链或部署变更（infra） / 数据处理或特征工程变更（data）

结论状态：已验证

输入内容：
用户提供丢失的历史对话上下文，涉及 GNNTP 项目中超参数调优（hyperopt/optuna）、任务执行流程、依赖替代（fastdtw、nltk、gensim、torchdata、flake8、idna、ray[tune]）、项目结构重构、缓存与日志管理、最小运行集抽取、注册机制优化等一系列工程化问题，要求恢复关键分析记录。

分析结果：
本阶段对历史对话进行了结构化还原，核心内容可归纳为三条主线：

1）超参数调优机制
项目中 hyperopt 属于历史遗留方案，optuna 或 ray[tune] 为更现代替代，但默认命令（run_model.py）在未开启 hyper_tune=True 时不会触发自动调参；出现 learning_rate choice [0.01, 0.005, 0.001] 且执行 3×2 次，本质是手动 grid search，而非 optuna 生效。

2）任务机制与数据驱动
GNNTP 中仅存在 traffic_state_pred 作为统一任务入口，speed/flow 等差异由 data_col（如 traffic_speed、traffic_flow）、output_dim 以及模型归纳偏置共同决定；当前配置 output_dim=3 且 data_col 包含 flow/occupancy/speed，实际执行的是多变量联合预测任务，而非单一 speed 或 flow 任务。

3）工程结构与依赖优化
项目存在明显工程冗余与历史包袱：

- fastdtw、nltk、gensim 等依赖在交通预测主流程中非核心，可移除或替换
- torchdata 可由 torch.utils.data 完全替代
- flake8 可由 ruff 替代（性能与规则集更优）
- idna 为 requests 依赖链组件，不建议手动删除
- ray[tune] 仅在分布式调参场景有意义，可按需裁剪

同时完成了一系列结构优化建议：

- dataset_cache 迁移至根目录 cache/
- 输出目录由随机 exp_id 改为“时间戳+任务+模型+数据集”
- 删除重复目录（test/tests）
- 精简根目录脚本，整合 run_hyper.py 与 hyper_example
- 构建最小运行子集（仅保留 traffic_state_pred + DCRNN/STGCN/PDFormer/NEW_MODEL）
- 在 newmini 子项目中实现“分层 registry + decorator 自动注册 + 去字符串路径”

关键结论：
GNNTP 项目当前执行流程中未启用 optuna 等自动调参工具，所有参数搜索行为均为显式枚举；任务语义完全由数据字段与模型决定；项目结构存在明显可裁剪空间，可通过“最小运行集 + registry 重构 + 依赖清理”显著降低复杂度并提升可维护性。

启发，或是下一步可以完成的工作：

- [ ] 启用 optuna 并接入 objective_function，实现真正自动调参
- [ ] 建立统一 hyper_tune 配置接口（替代分散配置）
- [ ] 完成 cache/、output/、log/ 三类目录职责统一
- [ ] 为 newmini 子项目补充最小 CI（训练+推理验证）
- [ ] 建立依赖白名单（requirements_min.txt）

补充说明：
风险点：

- 删除依赖（如 gensim/nltk）前需确认是否被隐式调用
- registry 重构可能影响模型动态加载路径

适用范围：
适用于 GNNTP 精简版项目构建、交通预测实验复现、科研原型工程化

限制条件：
仅适用于 traffic_state_pred 单任务场景；多任务扩展（如 OD、demand）需保留部分原始结构
