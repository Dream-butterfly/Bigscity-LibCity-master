# new_diffusion/INFO.md

## 目录职责
该目录实现 `new_diffusion` 实验模型本体。

## 关键文件
- `model.py`：模型结构与前向逻辑。
- `config.json` / `manifest.json`：默认配置与注册信息。

## 输入/输出
- 输入：交通时空序列批数据与扩散参数配置。
- 输出：模型预测与训练中间结果。

## 调用关系
- 可由主流程通过注册机制调用。
- 若需独立实验训练，使用 `scripts/experiments/train_new_diffusion.py`。

## 修改注意事项
1. 独立实验脚本的输入格式变更需同步更新 `scripts/experiments/train_new_diffusion.py`。
2. 变更扩散超参数时同步更新 `config.json`。

