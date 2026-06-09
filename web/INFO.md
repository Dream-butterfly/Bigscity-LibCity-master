# web/INFO.md

Web 层提供训练控制台：参数录入、任务触发、训练状态展示、日志实时解析和 ECharts 图表渲染。基于 FastAPI + Jinja2 模板。

## 架构版本

| 版本 | 路由 | 模板 | JS | 状态 |
|------|------|------|-----|------|
| v1（旧） | `/` | `train_web_fastapi.html` | `main.js` | 保留兼容，不再维护 |
| **v2（新）** | `/v2` | `train_web_v2.html` | `train_web_v2.js` | ✅ 当前使用 |

## 关键文件

| 路径 | 作用 |
| --- | --- |
| `train_web_fastapi.py` | FastAPI 主入口（2717行），定义全部路由、子进程调度、stdout 正则解析、结果读取 |
| `pyecharts_views.py` | ECharts 图表配置拼装（loss 线图、参数图、预测对比、指标表） |
| `templates/train_web_v2.html` | v2 控制台 HTML 模板（507行，sidebar + 6 个 tab） |
| `templates/train_web_fastapi.html` | v1 旧模板保留兼容 |
| `static/train_web_v2.js` | v2 前端脚本（108K，全功能） |
| `static/train_web_v2.css` | v2 组件样式 |
| `static/design-tokens.css` | 统一设计 token（双主题、间距/圆角/阴影变量） |
| `static/main.js` | v1 旧脚本保留兼容 |
| `static/i18n/` | 国际化 JSON（zh-CN / en-US / ja-JP） |

## v2 六个页面

| 页面 | 描述 | 核心 API |
|------|------|----------|
| **数据处理** | 生成数据工件 + 版本管理（创建/删除/重命名/激活）+ 数据集原始文件预览 | POST `/api/data/start`, GET `/api/data/versions` |
| **训练** | 选数据版本（锁定签名参数）→ 全参数编辑表 → 异步训练 → 实时日志/Loss 曲线/参数图/预测演示 | POST `/api/start`, GET `/api/status` |
| **继续训练** | 选历史 run + checkpoint epoch → 从 effective_config.json 恢复配置续训 | POST `/api/start_resume` |
| **模型对比** | 选 2-3 个已完成 run → 并排显示 metrics 表 + 预测曲线叠加图 | POST `/api/compare` |
| **模型评估** | 从 checkpoint 加载模型纯评估（EVAL_STATE 与训练独立可并行） | POST `/api/eval/start` |
| **运行历史** | 所有训练/评估记录表（状态/耗时/核心指标），自动扫描旧 outputs 向后兼容 | GET `/api/history` |

## 三状态机

| 状态机 | 变量 | 用途 | 可并行 |
|--------|------|------|--------|
| DataPrepState | `DATA_STATE` | 数据处理 | 与训练互斥 |
| TrainState | `STATE` | 训练/续训 | 与数据处理互斥 |
| TrainState | `EVAL_STATE` | 评估 | ✅ 完全独立可并行 |

## Stdout 实时解析正则

`TrainState._extract_model_log()` 逐行匹配子进程 stdout：

- `MODEL_START_RE`：捕获 `ModelName(` 模型结构打印
- `PARAM_LINE_RE`：`"name torch.Size([...])"` → 参数量行
- `TOTAL_PARAM_RE`：`"Total parameter numbers: N"`
- `EPOCH_LOSS_RE`：`"Epoch [1/100] train_loss:0.xxx, val_loss:0.xxx"`
- `SAVED_EPOCH_RE`：`"Saved model at N"` → checkpoint 保存点

## 数据版本机制

- 版本 ID 格式：`dv_<timestamp>_<uuid[:8]>`
- 元信息存储：`outputs/data_versions/<version_id>/meta.json`
- 数据工件绑定：meta.json → script_meta.artifact_id → `cache/data_artifacts/<artifact_id>/`
- 锁定键：`DATA_LOCKED_CONFIG_KEYS`（dataset/seed/scaler/input_window/output_window 等）训练时不可覆盖

## 修改注意事项

1. 页面字段名 ↔ 后端参数名 ↔ CLI 参数名必须保持同构
2. 子进程命令变更要同步更新路径和参数拼装
3. 图表字段口径改动需与 `pyecharts_views.py` 保持一致
4. 日志解析正则改动要兼容历史输出格式
5. v2 使用 `_state` 模块级对象，ECharts 实例存在 `_charts` 中
6. 状态机操作的锁机制（`with STATE.lock:`）不可遗漏，避免竞态

