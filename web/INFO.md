# web/INFO.md

## 目录职责
`web/` 存放 Web 训练控制台（FastAPI）及其前端资源，是当前主要的人机交互入口。

## 关键文件
- `train_web_fastapi.py`：Web 后端主程序（接口、任务调度、日志与结果解析）。
- `pyecharts_views.py`：图表配置构建（loss/参数/预测可视化）。
- `templates/train_web_fastapi.html`：主页面模板。
- `static/`：前端 JS/CSS 与多语言资源。

## 输入/输出
- 输入：用户在页面配置的任务参数、模型参数、CLI 参数。
- 输出：训练日志、状态、图表数据与历史记录（写入 `outputs/`）。

## 调用关系
- 根目录 `run_web.py` 调用本目录后端启动服务。
- 后端通过 `uv run scripts\run\run_*.py` 触发数据处理、训练与续训。

## 修改注意事项
1. 页面字段与后端参数映射必须保持一致。
2. 调整静态资源路径时需同步检查模板引用。
3. 调整入口脚本路径时需同步更新后端子进程命令。

