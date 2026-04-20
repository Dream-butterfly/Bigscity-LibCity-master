# DCRNN/INFO.md

## 目录职责
实现 DCRNN 交通速度预测模型及其执行器。

## 关键文件
- `model.py`：DCRNN 模型定义。
- `executor.py`：DCRNN 训练执行逻辑。
- `config.json` / `manifest.json` / `executor.json`：配置与注册信息。

## 输入/输出
- 输入：交通速度时序与图结构信息。
- 输出：预测序列、训练损失与评估结果。

## 调用关系
- 由 `pipeline -> models locator -> common executor` 链路调用。

## 修改注意事项
1. 图相关参数修改需同时考虑数据预处理输出。
2. executor 行为变更要注意与通用日志/保存流程一致。

