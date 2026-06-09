# scripts/experiments/INFO.md

与主流水线解耦的独立实验脚本，用于新模型快速试验和对照验证，不经过 `scripts/run/` 入口。

## 关键脚本

| 文件 | 作用 | 对应模型 |
|------|------|----------|
| `train_new_diffusion.py` | new_diffusion 的独立训练入口 | `GNNTP/models/new/new_diffusion/` |
| `train_new_diffusion_2.py` | new_diffusion_2 的独立训练入口 | `GNNTP/models/new/new_diffusion_2/` |

## 与主流水线的关系

- 这些脚本不经 `scripts/run/`，也不由 Web 控制台触发
- 但也建议：当模型稳定后，优先使用主流程（`run_data_artifact.py` + `run_train_artifact.py`）以获得统一的输出管理和数据工件校验
- 独立实验的结论若落地到主流程，需补齐 manifest.json 注册信息

## 输入/输出

- **输入**：实验专用数据文件、模型配置和命令行参数
- **输出**：独立日志、checkpoint、预测结果

## 修改注意事项

1. 保持与主流程参数体系的最小一致性（seed、设备参数等）
2. 默认路径基于仓库根目录推导，避免依赖当前工作目录
3. 实验脚本输入格式调整后，同步更新对应模型目录 INFO.md
