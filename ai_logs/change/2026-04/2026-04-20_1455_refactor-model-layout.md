### 更改 3

时间：2026-04-20 14:55:00 +08:00

变更类型：结构重构（refactor） / 环境或工具链或部署变更（infra）

变更状态：已验证

需求/目标：
将 LibCity 中模型相关资源（model、executor、config）按“模型主题”统一重构至 `libcity/models/<theme>/<Model>/`，减少分散目录依赖，提升模块化与可维护性，并为后续最小化工程与模型扩展提供基础。

变更文件：
libcity/models/locator.py
libcity/config/config_parser.py
libcity/common/registry_executor.py
libcity/model/registry.py
libcity/utils/utils.py
libcity/models/traffic_speed_prediction/DCRNN/model.py
libcity/models/traffic_speed_prediction/DCRNN/executor.py
libcity/models/traffic_speed_prediction/STGCN/model.py
libcity/models/traffic_speed_prediction/STGCN/executor.py
libcity/models/traffic_flow_prediction/PDFormer/model.py
libcity/models/traffic_flow_prediction/PDFormer/executor.py
libcity/models/new/NEW_MODEL/model.py

变更摘要：

- 变更内容：实现模型资源统一收口，将 DCRNN、STGCN、PDFormer、NEW_MODEL 的 model 与 executor 从旧目录迁移至 `libcity/models/<theme>/<Model>/`，新增 locator 模块用于动态定位模型路径；修改 ConfigParser、模型注册与 executor 注册逻辑，使其优先加载新目录结构；删除对应旧模型实现文件及部分旧配置文件。
- 变更原因：原项目模型实现分散在 `libcity/model`、`libcity/executor`、`libcity/config` 等多个目录，耦合严重，不利于扩展与维护；需要建立“模型自洽单元”结构。
- 影响范围：影响模型加载流程、配置解析路径、注册机制；对未迁移模型仍保留旧路径 fallback，不影响当前可运行性。

后续迭代建议：

- [ ] 将 config.json 与 executor.json 合并为单一模型配置文件
- [ ] 批量迁移剩余 traffic_speed_prediction 与 traffic_flow_prediction 模型
- [ ] 清理旧目录（libcity/model、libcity/executor、libcity/config/model）中的冗余实现
- [ ] 引入 theme 级共享模块，减少模型间重复代码
- [ ] 为 models 目录建立自动注册（decorator）机制

修改注意事项：
当前仍存在新旧路径共存状态，删除旧路径前需确认所有模型均已迁移；配置解析优先级已改变，需避免新旧配置冲突导致参数覆盖异常
