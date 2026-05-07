# 更改 14: Web 前端迁移 Phase 2~4 — 全功能迁移 + 日志过滤 + 文档清理

时间：2026-05-07
文件：`web/` 目录（仅 web/，无跨分支冲突）

---

## 变更内容

### Phase 2: 组件迁移（`train_web_v2.js` 643 → 1405 行）

- **参数表系统**: `inferType()` / `parseValue()` / `renderParamCell()` / `renderParamTable()` / `collectConfigFromTable()` / `loadDefaults()`
  - config/executor 分栏渲染，5 种类型选择器（str/bool/int/float/json），关键字过滤
  - 数据版本自动锁定 seed/train_rate/eval_rate/dataset_class/config_file
- **继续训练**: `loadResumeRuns()` / `startResumeTrain()` — 含 run 选择、epoch 渲染、全参数校验
- **模型对比**: `loadCompareRuns()` / `loadComparison()` / `renderCompareMetrics()` — 三模型 + h1/avg/best 表
- **运行历史**: `loadHistory()` / `renderHistoryTable()` — 含状态/耗时/指标/输出目录
- **ECharts**: `renderEChart()`（formatter 函数反序列化）/ `drawModelParamChart()` / `drawLossChart()` / `refreshPredictionSeries()`
- **实时轮询增强**: `pollTrainLogs()` 现在同时更新模型参数量图、Loss 曲线、评估指标、预测图
- **状态管理**: `_state` 增加 modelPlotType/modelPlot/modelPlotOptionPie/modelPlotOptionBar/lossPlot/predictionRanges/predictionSelection
- `_resumeRuns` / `_compareRuns` / `_charts` 独立变量

### Phase 3 (精简): 日志过滤

- `_trainLogLines` / `_resumeLogLines` 日志缓冲区
- `renderFilteredLogs()` — 按级别 + 关键字过滤重新渲染终端
- `trainLogLevel()` — 从日志行提取级别
- `applyLogFilter()` / `clearLogBuffer()`
- `log_level_filter` + `log_keyword_filter` 输入绑定

### Phase 4: 文档清理

- `web/INFO.md` 重写 — 增加 v1/v2 版本对照、v2 功能清单、修改注意事项（4→6条）
- 旧文件保留兼容（`main.js` / `train_web_fastapi.html` / `train_web_fastapi.css` / `train_web_fastapi_dark.css`）

## 未迁移（跳过 Phase 3）
- Pane-resizer 拖拽调整 → 标记 future work
- 视图布局控制 popup → 非必要
- 磁贴拖拽 → 复杂度高、使用率低
- 图表缩放滑块 → 价值低

## 验收
- [x] v2 路由 `/v2` 正常响应
- [x] 5 个 tab 切换正常
- [x] 参数表加载、编辑、收集正常
- [x] 日志实时显示 + 级别/关键字过滤
- [x] ECharts 图表渲染（参数量/Loss/预测/对比）
- [x] 亮/暗主题切换
- [x] 中/日/英三语言切换
- [x] 继续训练/模型对比/运行历史 tab 功能正常
