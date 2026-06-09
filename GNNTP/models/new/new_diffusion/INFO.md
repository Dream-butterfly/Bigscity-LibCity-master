# new_diffusion（v1）— Experimental

扩散类实验模型 v1，通过主流程或独立实验脚本运行。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | 扩散模型结构与前向逻辑 |
| `config.json` | 默认实验参数 |
| `manifest.json` | 注册信息（可用 `--model new_diffusion` 通过主流程调用） |

## 参考

- 独立实验入口：`scripts/experiments/train_new_diffusion.py`
- 主流程入口：`run_train_artifact.py --model new_diffusion --dataset <dataset>`

## 修改注意事项

1. 输入格式或字段调整时需同步独立实验脚本
2. 扩散参数改动保持 `config.json` 与代码读取一致性
3. 与 `new_diffusion_2` / `new_diffusion_fuzzy` 的差异建议在提交记录中说明

