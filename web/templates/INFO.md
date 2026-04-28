# web/templates/INFO.md

## 目录职责

Web 控制台 HTML 模板，由 FastAPI Jinja2 模板引擎渲染。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `train_web_fastapi.html` | 训练控制台主页面模板（28.1 KB） |

## 模板渲染

FastAPI 后端通过 Jinja2 渲染 HTML，注入：
- 服务器配置（端口、主机）
- 初始语言设置
- CSRF 令牌（如需要）

## 输入/输出

- **输入**：FastAPI 路由 handler 注入的模板变量
- **输出**：浏览器端完整 HTML 页面

## 调用关系

1. `../train_web_fastapi.py` 通过 `templates.TemplateResponse()` 渲染
2. HTML 引用 `../static/main.js`、`../static/train_web_fastapi.css`
3. HTML 引用 `../static/i18n/zh-CN.json` 等翻译文件

## 修改注意事项

1. 模板变量使用 Jinja2 语法 `{{ variable }}`
2. 静态资源路径使用 `/static/` 前缀（挂载点）
3. 修改模板后无需重启服务器（Jinja2 支持热重载）
