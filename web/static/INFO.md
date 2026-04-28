# web/static/INFO.md

## 目录职责

Web 控制台前端静态资源：JavaScript 主逻辑、CSS 样式、国际化文本。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `main.js` | 前端主逻辑（78.7 KB），训练控制台全部交互 |
| `train_web_fastapi.css` | 亮色主题样式 |
| `train_web_fastapi_dark.css` | 暗色主题样式 |

## 子目录

| 目录 | 作用 |
| --- | --- |
| `i18n/` | 国际化翻译文件（JSON） |

## 输入/输出

- **输入**：FastAPI 后端 API 响应
- **输出**：浏览器渲染的训练控制台页面

## 调用关系

1. `../train_web_fastapi.py` 挂载 `/static/` 路由提供静态文件
2. `main.js` 通过 AJAX/Fetch 调用 FastAPI 后端 `/api/*` 接口
3. `../templates/train_web_fastapi.html` 引入 CSS 和 JS

## 修改注意事项

1. JS/CSS 修改后需强制刷新浏览器缓存（Ctrl+Shift+R）
2. i18n JSON key 需与 JS 中 `t('key')` 调用保持一致
3. 新增 API 调用需同步处理 loading/error/success 三态
