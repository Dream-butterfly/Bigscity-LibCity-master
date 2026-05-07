# AI 日志索引

> 最新在上。详细内容请进入对应文件。

## 更改（change）

1. `更改 13` | 2026-04-22 12:03:20 +08:00 | 新增功能/结构重构/实验流程变更/数据处理变更 | [数据工件解耦链路第一阶段落地](change/2026-04/2026-04-22_1203_data-artifact-decoupling-phase1.md)
2. `更改 12` | 2026-04-22 11:28:48 +08:00 | 性能优化/实验流程变更 | [new_diffusion_fuzzy 第三阶段容量缩放优化](change/2026-04/2026-04-22_1128_third-stage-capacity-scaling-optimization.md)
3. `更改 11` | 2026-04-21 20:40:00 +08:00 | 性能优化/实验流程变更 | [new_diffusion_fuzzy 第二阶段推理降耗优化](change/2026-04/2026-04-21_2040_second-stage-inference-optimization.md)
4. `更改 10` | 2026-04-21 01:50:00 +08:00 | 性能优化/实验流程变更 | [new_diffusion_fuzzy 第一阶段稳训优化](change/2026-04/2026-04-21_0150_first-stage-stability-optimization.md)
5. `更改 9` | 2026-04-21 01:00:00 +08:00 | 结构重构 | [new_diffusion_fuzzy utils 按功能拆分多文件](change/2026-04/2026-04-21_0100_split-fuzzy-utils-modules.md)
6. `更改 8` | 2026-04-21 00:40:00 +08:00 | 结构重构/性能优化 | [new_diffusion_fuzzy 抽离 utils 公共辅助逻辑](change/2026-04/2026-04-21_0040_new-diffusion-fuzzy-utils-refactor.md)
7. `更改 7` | 2026-04-21 00:20:00 +08:00 | 修复问题/性能优化/结构重构 | [new_diffusion_fuzzy 修订（采样步映射与模糊校验优化）](change/2026-04/2026-04-21_0020_fuzzy-model-revision.md)
8. `更改 6` | 2026-04-21 00:00:00 +08:00 | 新增功能/结构重构 | [new_diffusion_fuzzy 引入模糊数学结构](change/2026-04/2026-04-21_0000_new-diffusion-fuzzy.md)
9. `更改 5` | 2026-04-20 16:39:01 +08:00 | 结构重构 | [dataset mixins 职责拆分](change/2026-04/2026-04-20_1639_split-dataset-mixins.md)
10. `更改 4` | 2026-04-20 16:25:41 +08:00 | 修复问题/结构重构 | [默认模型与文档引用同步修正](change/2026-04/2026-04-20_1625_default-model-and-doc-sync.md)
11. `更改 3` | 2026-04-20 14:55:00 +08:00 | 结构重构/环境部署变更 | [模型资源目录重构](change/2026-04/2026-04-20_1455_refactor-model-layout.md)
12. `更改 2` | 2026-03-26 12:23:00 +08:00 | 结构重构/实验流程变更/环境部署变更/数据处理变更 | [历史上下文恢复与工程优化结论](change/2026-03/2026-03-26_1223_engineering-context-recovery.md)
13. `更改 1` | 2026-03-24 15:23:00 +08:00 | 修复问题/结构重构/环境部署变更 | [移除 torchtext Field 依赖](change/2026-03/2026-03-24_1523_remove-torchtext-field.md)

## 分析（analysis）

1. `分析 10` | 2026-05-07 09:52:00 +08:00 | 方案设计/决策建议 | [Web 前端深度迁移计划](analysis/2026-05/2026-05-07_web-frontend-migration-plan.md)
2. `分析 9` | 2026-04-22 11:58:52 +08:00 | 方案设计/决策建议 | [数据-训练解耦方案与口径确认](analysis/2026-04/2026-04-22_1158_data-training-decoupling-design-and-decisions.md)
2. `分析 8` | 2026-04-21 01:30:00 +08:00 | 方案设计/决策建议 | [不改核心结构下的稳训与降耗方案](analysis/2026-04/2026-04-21_0130_stability-and-efficiency-without-architecture-change.md)
3. `分析 7` | 2026-04-20 16:14:15 +08:00 | 问题定位/事实结论 | [项目结构审计（冗余/缺失/错位）](analysis/2026-04/2026-04-20_1614_project-structure-audit.md)
4. `分析 6` | 2026-03-24 17:00:00 +08:00 | 调研/理解/对比/验证 | [traffic_state_pred 与 speed/flow 关系总结](analysis/2026-03/2026-03-24_1700_task-relation-summary.md)
5. `分析 5` | 2026-03-24 16:23:00 +08:00 | 调研/理解/对比/验证 | [task 层级与调用链定位](analysis/2026-03/2026-03-24_1623_task-hierarchy-and-call-chain.md)
6. `分析 4` | 2026-03-24 16:23:00 +08:00 | 调研/理解/验证/风险 | [torchtext ABI 报错根因分析](analysis/2026-03/2026-03-24_1623_torchtext-abi-root-cause.md)
7. `分析 3` | 2026-03-24 15:47:00 +08:00 | 理解/对比/验证/风险/优化/设计 | [new_model 创新性与改进方向](analysis/2026-03/2026-03-24_1547_new-model-innovation-review.md)
8. `分析 2` | 2026-03-24 15:23:00 +08:00 | 设计/选型/对比/优化 | [论文对比模型组合建议](analysis/2026-03/2026-03-24_1523_paper-model-selection.md)
9. `分析 1` | 2026-03-24 15:23:00 +08:00 | 调研/理解/验证 | [PEMSD4 模型范围与分类](analysis/2026-03/2026-03-24_1523_pemsd4-model-scope.md)
