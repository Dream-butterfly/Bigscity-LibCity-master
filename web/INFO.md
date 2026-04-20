# web/INFO.md

## 目录职责

`web/` 提供训练控制台的 Web 层实现：负责参数录入、任务触发、训练状态展示、日志解析和图表渲染。

## 关键文件与目录

| 路径 | 作用 |
| --- | --- |
| `train_web_fastapi.py` | FastAPI 主入口，定义路由、子进程调度、日志与结果读取 |
| `pyecharts_views.py` | 图表配置拼装（loss、预测对比、指标展示） |
| `templates/train_web_fastapi.html` | 控制台页面模板 |
| `static/` | 页面脚本、样式和前端资源 |

## 输入/输出

- **输入**：前端提交的任务参数（task/model/dataset/训练参数）、历史实验 ID、筛选条件。
- **输出**：运行状态、标准输出日志、图表数据、实验结果查询响应（最终对应 `outputs/<exp_id>/` 的文件）。

## 调用关系

1. `run_web.py` 调用 `web.train_web_fastapi.main()` 启动服务。
2. Web 后端通过子进程执行 `scripts/run/run_*.py`。
3. 训练执行产物回写到 `outputs/`，Web 再读取并渲染给前端。

## 修改注意事项

1. 页面字段名、后端参数名、CLI 参数名必须保持同构，避免“页面可填但后端不识别”。
2. 子进程命令变更时要同步更新路径、参数拼装和错误提示文案。
3. 图表字段口径改动需与 `pyecharts_views.py` 保持一致，防止渲染空图或错图。
4. 任何与日志解析相关的改动都要兼容历史输出格式。

