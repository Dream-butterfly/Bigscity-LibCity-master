# STGformer/INFO.md

## 目录职责
实现 STGformer 框架内版本（模型+执行器）。

## 关键文件
- `model.py`：STGformer 模型实现。
- `executor.py`：STGformer 执行器实现。
- `config.json` / `manifest.json` / `executor.json`：配置与注册信息。

## 输入/输出
- 输入：交通速度时序特征与图结构特征。
- 输出：预测结果、训练损失与评估指标。

## 调用关系
- 与通用 pipeline 对接，通过专用 executor 优化训练流程。

## 修改注意事项
1. 与 `STGformer-独立实现` 的代码差异需保持清晰。
2. 训练流程改动需同步检查恢复训练与保存逻辑。

