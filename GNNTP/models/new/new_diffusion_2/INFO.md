# new_diffusion_2（v2）— Experimental

扩散类实验模型 v2，相对 v1 的改进方案验证。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | 扩散模型 v2 结构与前向逻辑 |
| `config.json` | 默认实验参数 |
| `manifest.json` | 注册信息（可用 `--model new_diffusion_2` 通过主流程调用） |

## 参考

- 独立实验入口：`scripts/experiments/train_new_diffusion_2.py`
- 主流程入口：`run_train_artifact.py --model new_diffusion_2 --dataset <dataset>`

## 修改注意事项

1. 保持与 `new_diffusion` 的差异可追踪
2. 接口调整需同步检查数据、执行器和独立实验脚本兼容性
3. 参数更新优先维护 `config.json`

