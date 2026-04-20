# new_diffusion_2/INFO.md

## 目录职责
该目录是 `new_diffusion` 的迭代版本模型实现，用于验证改进策略。

## 关键文件
- `model.py`：模型实现。
- `config.json` / `manifest.json`：配置与注册信息。

## 输入/输出
- 输入：与 diffusion 类模型兼容的时空数据批次。
- 输出：预测结果与训练相关状态。

## 调用关系
- 与 `new_diffusion` 类似，通过模型注册机制接入主流程。
- 若需独立实验训练，使用 `scripts/experiments/train_new_diffusion_2.py`。

## 修改注意事项
1. 与 `new_diffusion` 的差异应尽量保持清晰可追踪。
2. 若接口调整，需同步检查 executor、dataset 和独立实验脚本兼容。

