# web/INFO.md

## 目录职责

`web/` 提供训练控制台的 Web 层实现：负责参数录入、任务触发、训练状态展示、日志解析和图表渲染。

## 架构版本

当前有两个前端版本并存：

| 版本 | 路由 | 模板 | JS | CSS | 状态 |
|------|------|------|-----|-----|------|
| v1（旧） | `/` | `train_web_fastapi.html` | `main.js` | `train_web_fastapi.css` + `train_web_fastapi_dark.css` | 保留兼容，不再维护 |
| **v2（新）** | `/v2` | `train_web_v2.html` | `train_web_v2.js` | `train_web_v2.css` + `design-tokens.css` | ✅ **当前使用** |

## 关键文件与目录

| 路径 | 作用 |
| --- | --- |
| `train_web_fastapi.py` | FastAPI 主入口，定义路由、子进程调度、日志与结果读取。含 `/v2` 路由 |
| `pyecharts_views.py` | 图表配置拼装（loss、预测对比、指标展示） |
| `templates/train_web_v2.html` | v2 控制台页面模板（sidebar 布局） |
| `templates/train_web_fastapi.html` | v1 旧模板（保留兼容） |
| `static/train_web_v2.js` | v2 前端脚本（~1385 行，全功能迁移） |
| `static/train_web_v2.css` | v2 组件样式（sidebar、cards、terminal、param table 等） |
| `static/design-tokens.css` | 统一设计 token（双主题变量、间距/圆角/阴影） |
| `static/main.js` | v1 旧脚本（保留兼容） |
| `static/i18n/` | 国际化 JSON 文件（zh-CN / en-US / ja-JP） |

## v2 前端功能

- **Sidebar 导航**：5 个 tab（数据处理 / 训练 / 继续训练 / 模型对比 / 运行历史）
- **数据处理**：task/model/dataset 联动、split 预设、数据版本管理（表格）、数据集过滤、数据预览、日志终端
- **训练**：数据版本选择（自动锁定相关参数）、全参数编辑表（config/executor 分栏、类型选择、值编辑、关键字过滤）、CLI 字段、实时日志（级别过滤 + 关键字搜索）、模型参数图（饼图/柱状图）、Loss 曲线、评估指标、预测演示图
- **继续训练**：历史运行选择、checkpoint epoch 选择、续训启动
- **模型对比**：三模型选择、指标对比表（h1/avg/best）、预测曲线叠加
- **运行历史**：表格展示（状态/run/模型/数据集/耗时/指标/输出目录/时间）
- **设计系统**：亮/暗双主题、Tweaks 面板（间距/圆角/密度调节）、中/日/英三语言、响应式断点（1200/1024/768px）

## 输入/输出

- **输入**：前端提交的任务参数（task/model/dataset/训练参数）、历史实验 ID、筛选条件。
- **输出**：运行状态、标准输出日志、图表数据、实验结果查询响应（最终对应 `outputs/<exp_id>/` 的文件）。

## 调用关系

1. `run_web.py` 调用 `web.train_web_fastapi.main()` 启动服务。
2. Web 后端通过子进程执行 `scripts/run/run_*.py`。
3. 训练执行产物回写到 `outputs/`，Web 再读取并渲染给前端。

## 修改注意事项

1. 页面字段名、后端参数名、CLI 参数名必须保持同构，避免”页面可填但后端不识别”。
2. 子进程命令变更时要同步更新路径、参数拼装和错误提示文案。
3. 图表字段口径改动需与 `pyecharts_views.py` 保持一致，防止渲染空图或错图。
4. 任何与日志解析相关的改动都要兼容历史输出格式。
5. v2 使用 `_state`（模块级对象）而非 `state`（全局），新功能添加时应遵循相同模式。
6. v2 ECharts 实例存储在 `_charts` 对象中（`renderEChart` 管理），不要直接操作 `echarts.init`。

