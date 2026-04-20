# STTN/INFO.md

## 目录职责
实现 STTN 交通速度预测模型及专用执行器。

## 关键文件
- `model.py`：STTN 模型实现。
- `executor.py`：STTN 训练执行逻辑。
- `config.json` / `manifest.json` / `executor.json`：配置与注册信息。

## 输入/输出
- 输入：交通速度时空序列特征。
- 输出：预测速度与训练评估结果。

## 调用关系
- 通过模型/执行器注册机制接入主训练流水线。

## 修改注意事项
1. 注意与通用 executor 接口保持一致。
2. 结构性改动后建议同步检查配置兼容。

