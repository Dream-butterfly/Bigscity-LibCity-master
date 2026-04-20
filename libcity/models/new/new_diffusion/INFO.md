# new_diffusion/INFO.md

## 目录职责
该目录实现 `new_diffusion` 实验模型及相关训练辅助脚本。

## 关键文件
- `model.py`：模型结构与前向逻辑。
- `train.py`：局部训练辅助逻辑。
- `config.json` / `manifest.json`：默认配置与注册信息。

## 输入/输出
- 输入：交通时空序列批数据与扩散参数配置。
- 输出：模型预测与训练中间结果。

## 调用关系
- 可由主流程通过注册机制调用，亦可用于局部实验调试。

## 修改注意事项
1. 保持 `train.py` 与框架 executor 行为的一致性。
2. 变更扩散超参数时同步更新 `config.json`。

