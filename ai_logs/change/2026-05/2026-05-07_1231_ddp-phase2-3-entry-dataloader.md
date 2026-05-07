### 更改 17
时间：2026-05-07T12:31:14+08:00
来源类型：处理
来源说明：L 的对话 — 开始进行多卡训练改造，先完成入口/配置与数据层分布式接入

更改类型-动作：新增功能/结构重构
更改类型-范围：跨模块
变更状态：草稿

需求/目标：在不破坏单卡训练的前提下，引入 DDP 所需的基础参数语义与分布式采样能力。
变更文件：
- `GNNTP/utils/argument_list.py`
- `GNNTP/config_parser.py`
- `GNNTP/data/dataloader.py`
- `GNNTP/data/dataset/mixins/pipeline_mixin.py`
- `GNNTP/data/runtime.py`

变更摘要：
- 变更内容：为运行脚本新增 `distributed/local_rank/rank/world_size/master_addr/master_port/dist_backend` 参数；配置解析支持从命令行与环境变量推导分布式上下文；DataLoader 在分布式模式下启用 `DistributedSampler`。
- 变更原因：为后续执行器 DDP 化提供稳定前置条件，先完成“入口语义一致 + 数据切分正确”。
- 影响范围：训练/恢复/调参等所有使用 `add_general_args` 与 `generate_dataloader` 的链路。

修改注意事项：
- 当前仅完成到数据层，尚未接入执行器的 DDP 包装与分布式评估聚合。
- 分布式训练正式可用仍需下一步修改 `GNNTP/common/traffic_state_executor.py`（禁区文件，需先审阅 diff）。
