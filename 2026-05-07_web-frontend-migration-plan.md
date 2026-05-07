### 分析 10

时间：2026-05-07 09:52:00 +08:00
来源类型：提问
来源说明：用户要求分析 web/design-ref/style-v1.html 新设计，制定深度迁移计划。

分析类型-目的：方案设计
分析类型-产出：决策建议
结论状态：可用

---

# GNN-TP Web 前端深度迁移计划

## 一、功能清单对照表

### 1. 核心功能模块

| 模块 | 现有功能 | 新设计覆盖 | 差距 | 优先级 |
|-----|---------|----------|-----|-------|
| **导航** | 顶部 Tab 栏 | 左侧 Sidebar | 布局变化 | P0 |
| **主题** | 亮/暗双主题 | 亮/暗双主题 | ✅ | - |
| **语言** | 中/日/英三语言 | 无 | ❌ 缺失 | P0 |
| **数据处理 Tab** | task/model/dataset 选择、split 预设、版本管理、预览、日志 | 有基础 UI | 需补充预设、过滤等 | P1 |
| **训练 Tab** | 数据版本选择、参数编辑、日志、图表、指标 | 有基础 UI | 需补充全参数表 | P1 |
| **继续训练 Tab** | 运行选择、epoch 选择、版本绑定 | 有基础 UI | ✅ | - |
| **模型对比 Tab** | 三模型对比、指标表、曲线图 | 有基础 UI | ✅ | - |
| **运行历史 Tab** | 表格展示、指标显示 | 有基础 UI | ✅ | - |

### 2. 现有功能详单（必须保留）

#### 2.1 数据处理页
- [x] task/model/dataset 下拉联动
- [x] seed/batch_size/dataset_class 输入
- [x] **split 预设选择器**（0.6/0.2/0.2 等 4 种预设）← 新设计缺失
- [x] split 比例实时显示
- [x] extra_args 输入
- [x] 数据版本列表（含 status/task/model/dataset/split/updated）
- [x] 数据集过滤（按 dataset）
- [x] 版本重命名/删除/备注
- [x] 同步到训练参数
- [x] 数据预览（dyna/geo/rel 切换、entity_id 过滤、time 范围、columns 选择）
- [x] 处理日志终端

#### 2.2 训练页
- [x] data_version_id 选择（仅 ready 可用）
- [x] 数据版本上下文显示（task/model/dataset/seed/split）
- [x] 参数锁定提示
- [x] **全参数编辑表**（config/executor 分栏、类型选择、值编辑、筛选）← 新设计缺失完整表格
- [x] 训练控制（max_epoch/batch_size/learning_rate/saved_model/train）
- [x] 运行环境（exp_id/gpu/gpu_id）
- [x] 覆盖参数（executor/evaluator/extra_args）
- [x] 训练日志（级别过滤、关键字过滤、自动滚动）
- [x] 模型参数图（饼图/柱状图切换、TopK 设置）
- [x] Loss 曲线
- [x] 评估指标表
- [x] 预测效果演示（horizon/node/feature 选择器）
- [x] **视图布局控制**（比例调节、图表缩放）← 新设计缺失
- [x] **磁贴拖拽**（卡片可移动）← 新设计缺失
- [x] **Pane 可拖拽调整宽度** ← 新设计缺失

#### 2.3 继续训练页
- [x] 运行选择（显示 latest epoch）
- [x] data_version_id 绑定
- [x] epoch 选择器（动态填充可用 checkpoint）
- [x] 目标 max_epoch
- [x] saved_model 选择
- [x] 运行上下文提示

#### 2.4 模型对比页
- [x] 三模型选择
- [x] 指标对比表（h1/avg/best）
- [x] 预测曲线叠加图

#### 2.5 运行历史页
- [x] 限制条数
- [x] 状态/Run ID/模型/数据集/耗时/指标/输出目录/结束时间

### 3. 新设计缺失功能清单

| 缺失功能 | 重要程度 | 补充方案 |
|---------|---------|---------|
| **i18n 国际化** | 🔴 关键 | 保留现有 data-i18n 属性机制 |
| **split 预设选择器** | 🟡 重要 | 在 param-group 内添加预设下拉 |
| **全参数编辑表** | 🔴 关键 | 新增 card 组件，保留 config/executor 分栏 |
| **视图布局控制** | 🟡 重要 | 保留现有 layout popup 机制 |
| **磁贴拖拽** | 🟢 可选 | 可迁移或用 Tweaks 替代 |
| **Pane 拖拽调整** | 🟡 重要 | 保留现有 pane-resizer 机制 |
| **日志级别/关键字过滤** | 🟡 重要 | 在 terminal 组件中添加 toolbar |
| **预测 horizon/node/feature 选择器** | 🟡 重要 | 在 chart card 中添加 controls |
| **图表 TopK 设置** | 🟢 可选 | 迁移现有 input number |

---

## 二、新设计优化建议

### 2.1 设计系统层面

| 问题 | 现状 | 优化建议 |
|-----|------|---------|
| **CSS 变量未语义化** | `--spacing-xs` 等直接数值 | 建议增加 `--space-inline-sm` 等语义别名 |
| **断点不一致** | 1024px / 768px 两级 | 建议增加 1280px 断点适配大屏 |
| **动画过重** | 每张卡片都有 fadeIn 动画 | 建议仅在首屏或用户交互时触发 |
| **深色侧边栏对比度** | `--text-sidebar: #7c8599` | 建议提升至 `#94a3b8` 改善可读性 |

### 2.2 组件层面

| 组件 | 问题 | 优化建议 |
|-----|------|---------|
| **Terminal** | 无复制按钮 | 添加复制日志按钮 |
| **Table** | 无排序/固定列 | 历史表建议支持排序 |
| **Card** | 悬停效果统一上移 | 建议对 resizable-card 禁用悬停效果 |
| **Badge** | 无 tooltip | 建议添加 tooltip 显示详细信息 |
| **Sidebar** | 无折叠功能 | 建议添加折叠/展开按钮 |
| **Tweaks 面板** | 位置固定右下角 | 建议可拖拽或记住位置 |

### 2.3 交互层面

| 交互 | 问题 | 优化建议 |
|-----|------|---------|
| **Tab 切换** | 无动画 | 保持现有 fadeIn 即可 |
| **表单提交** | 无 loading 状态 | 按钮添加 loading spinner |
| **错误提示** | 仅 alert | 建议增加 toast notification |
| **实时数据** | 无连接状态指示 | 建议在 header 添加 websocket 状态 |

### 2.4 新设计可采纳增强

| 增强 | 描述 |
|-----|------|
| **Tweaks 面板** | ✅ 保留，增加图表缩放滑块 |
| **Stagger 动画** | ✅ 保留，但限制在首屏 |
| **状态徽章** | ✅ 保留，添加 tooltip |
| **毛玻璃卡片** | ✅ 保留，但禁用 resizable-card 的 hover 效果 |

---

## 三、迁移执行计划

### Phase 1: 基础设施层（2 天）

```
Day 1: Design Tokens & 布局骨架
├── 1.1 提取新设计 CSS 变量 → design-tokens.css
├── 1.2 合并现有主题变量 → 统一 --bg-*, --text-*, --border-*
├── 1.3 新建 sidebar-layout.html 测试骨架
├── 1.4 迁移 sidebar 组件（保留折叠逻辑）
└── 1.5 迁移 main.content 布局

Day 2: 响应式 & 主题
├── 2.1 合并断点媒体查询
├── 2.2 亮/暗主题变量完整迁移
├── 2.3 保留 i18n 属性机制（data-i18n）
└── 2.4 验证主题切换功能
```

### Phase 2: 组件迁移（3 天）

```
Day 3: 数据处理页
├── 3.1 迁移数据预处理 card
├── 3.2 添加 split 预设选择器
├── 3.3 迁移数据版本 card
├── 3.4 迁移数据预览 card
└── 3.5 迁移 Terminal 日志组件

Day 4: 训练页上半部分
├── 4.1 迁移训练配置 card
├── 4.2 迁移全参数编辑表（分栏布局）
├── 4.3 迁移训练日志 Terminal
└── 4.4 迁移模型参数图 card

Day 5: 训练页下半部分 + 其他页
├── 5.1 迁移 Loss 曲线 card
├── 5.2 迁移评估指标 + 预测演示 pane
├── 5.3 迁移继续训练页
├── 5.4 迁移模型对比页
└── 5.5 迁移运行历史页
```

### Phase 3: 交互增强（2 天）

```
Day 6: 布局控制 & 拖拽
├── 6.1 迁移视图布局控制 popup
├── 6.2 迁移 pane-resizer 拖拽
├── 6.3 迁移磁贴拖拽（可选）
└── 6.4 添加 Tweaks 面板增强

Day 7: 细节优化 & 验证
├── 7.1 日志过滤功能迁移
├── 7.2 图表选择器迁移
├── 7.3 功能回归测试
└── 7.4 性能优化（动画、重绘）
```

### Phase 4: 清理 & 文档（1 天）

```
Day 8: 收尾
├── 8.1 移除旧 CSS 文件
├── 8.2 更新 web/INFO.md
├── 8.3 添加迁移说明到 ai_logs
└── 8.4 最终验证
```

---

## 四、文件变更清单

### 新建文件

| 文件 | 用途 |
|-----|------|
| `web/static/design-tokens.css` | Design Tokens 变量 |
| `web/templates/train_web_v2.html` | 新模板 |
| `web/static/train_web_v2.css` | 新样式 |
| `web/static/train_web_v2.js` | 新 JS（基于现有 main.js 迁移）|

### 修改文件

| 文件 | 修改内容 |
|-----|---------|
| `web/train_web_fastapi.py` | 添加新模板路由，保留旧路由 |
| `web/INFO.md` | 更新文档 |

### 保留文件（兼容期）

| 文件 | 说明 |
|-----|------|
| `web/templates/train_web_fastapi.html` | 旧模板，标记 deprecated |
| `web/static/train_web_fastapi.css` | 旧样式 |
| `web/static/train_web_fastapi_dark.css` | 暗色样式 |
| `web/static/main.js` | 旧 JS |

---

## 五、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| JS 事件绑定失效 | 中 | 高 | 保留所有 DOM ID，使用相同选择器 |
| 图表渲染异常 | 低 | 中 | ECharts option 格式不变 |
| 样式冲突 | 中 | 低 | 使用 CSS scope 或独立 class 前缀 |
| i18n 功能缺失 | 低 | 高 | 保留 data-i18n 机制不变 |
| 响应式断点不一致 | 中 | 低 | 合并两套断点逻辑 |

---

## 六、验收标准

### 功能验收

- [ ] 所有 API 端点正常调用
- [ ] 数据处理流程完整
- [ ] 训练流程完整（启动/停止/清空）
- [ ] 续训流程完整
- [ ] 模型对比功能正常
- [ ] 运行历史显示正常
- [ ] 中/日/英三语言切换正常
- [ ] 亮/暗主题切换正常
- [ ] 所有图表正常渲染
- [ ] 日志实时滚动正常
- [ ] 布局比例调节正常
- [ ] 磁贴拖拽正常（如保留）

### 性能验收

- [ ] 首屏渲染 < 1s
- [ ] Tab 切换无卡顿
- [ ] 日志渲染不阻塞 UI
- [ ] 图表缩放流畅

---

## 七、后续工作计划

### 7.1 用户可调重要参数显示

**需求描述**：
将现有全参数编辑表中的一部分"重要参数"优先展示，其余参数默认折叠。用户可展开查看/编辑所有参数。

**设计方案**：

#### 方案 A：配置文件标记

在模型配置中新增参数重要性标记：

```json
// GNNTP/models/{model}/param_meta.json
{
  "important_params": [
    "max_epoch",
    "batch_size",
    "learning_rate",
    "dropout",
    "hidden_dim",
    "num_layers"
  ],
  "param_groups": {
    "basic": ["max_epoch", "batch_size", "learning_rate"],
    "model": ["hidden_dim", "num_layers", "dropout", "Ks", "Kt"],
    "training": ["weight_decay", "grad_clip", "early_stop"],
    "advanced": [...]
  }
}
```

#### 方案 B：后端 API 扩展

```python
# 在 /api/default_config 返回中增加
{
  "config": {...},
  "executor_keys": [...],
  "important_keys": ["max_epoch", "batch_size", "learning_rate", "dropout"],
  "param_display_order": ["max_epoch", "batch_size", ...]
}
```

#### 方案 C：前端本地定义

```javascript
// main.js 或配置文件
const IMPORTANT_PARAMS = new Set([
  'max_epoch', 'batch_size', 'learning_rate',
  'dropout', 'hidden_dim', 'num_layers'
]);
```

**推荐方案**：方案 B（后端 API 扩展）
- 理由：参数重要性可随模型变化，由后端控制更灵活

**UI 设计**：

```
┌─────────────────────────────────────────┐
│ 全部参数（42项）              [展开全部] │
├─────────────────────────────────────────┤
│ 重要参数（6项）                          │
│ ┌─────────────────────────────────────┐ │
│ │ max_epoch    │ int  │ 100           │ │
│ │ batch_size   │ int  │ 64            │ │
│ │ learning_rate│ float│ 0.001         │ │
│ │ dropout      │ float│ 0.3           │ │
│ │ hidden_dim   │ int  │ 64            │ │
│ │ num_layers   │ int  │ 2             │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ [▶ 展开其他 36 项参数]                  │
└─────────────────────────────────────────┘
```

**实施步骤**：

1. **Phase 1**：在 `GNNTP/models/{model}/config.json` 中添加 `important_params` 字段
2. **Phase 2**：修改 `/api/default_config` 返回 `important_keys`
3. **Phase 3**：前端参数表渲染逻辑改为分组展示
4. **Phase 4**：添加展开/折叠交互

**预估工作量**：0.5 天

---

关键结论：
1. 新设计在视觉层面有提升，但缺少多项关键交互功能
2. 迁移需 8 天，建议分 4 个 Phase 执行
3. 建议采用渐进式迁移，保留旧版本兼容
4. 后续可增加"重要参数"功能提升用户体验

启发，或是下一步可以完成的工作：
- 执行 Phase 1：创建 design-tokens.css，开始基础设施层迁移
- 并行推进"重要参数"配置文件设计

补充说明：
- 风险点：JS 事件绑定需仔细迁移，保留所有 DOM ID
- 适用范围：GNN-TP Web 控制台前端
- 限制条件：需保证后端 API 完全兼容
