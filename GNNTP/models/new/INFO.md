# new/INFO.md

## 目录职责
`models/new/` 存放本项目新增或实验性模型实现。

## 关键内容
- `NEW_MODEL/`：新模型模板目录。
- `new_diffusion/`、`new_diffusion_2/`：扩散类实验模型实现。

## 输入/输出
- 输入：来自 dataset 的批数据与配置参数。
- 输出：预测值、损失与可训练参数状态。

## 调用关系
- 通过 `models/registry.py` 与 `locator.py` 被主流程加载。

## 修改注意事项
1. 实验模型稳定后建议补充更完整注释与配置说明。
2. 保持目录内 `config.json`、`manifest.json` 与代码实现一致。

