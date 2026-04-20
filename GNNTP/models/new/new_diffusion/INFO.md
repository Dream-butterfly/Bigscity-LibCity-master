# new_diffusion/INFO.md

## 目录职责

实现 `new_diffusion` 扩散类实验模型（v1）。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | 模型结构与前向逻辑 |
| `config.json` | 默认实验参数 |
| `manifest.json` | 模型注册元信息 |

## 输入/输出

- **输入**：交通时空批数据、扩散相关超参数和训练配置。
- **输出**：预测结果、训练损失和中间状态。

## 调用关系

1. 可通过主流程注册机制加载。
2. 也可通过 `scripts/experiments/train_new_diffusion.py` 独立训练。

## 修改注意事项

1. 输入格式或字段调整时，需同步独立实验脚本与数据准备逻辑。
2. 扩散相关参数改动应在 `config.json` 与代码读取逻辑保持一致。
3. 与 `new_diffusion_2` 的差异点建议在提交记录中显式说明。

