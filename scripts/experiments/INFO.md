# scripts/experiments/INFO.md

## 目录职责

`scripts/experiments/` 存放与主 pipeline 解耦的实验入口，主要用于新模型快速试验和对照验证。

## 关键脚本

| 文件 | 作用 |
| --- | --- |
| `train_new_diffusion.py` | `new_diffusion` 的独立训练入口 |
| `train_new_diffusion_2.py` | `new_diffusion_2` 的独立训练入口 |

## 输入/输出

- **输入**：实验专用数据文件（如 NPZ）、模型配置和命令行参数。
- **输出**：独立日志、checkpoint、预测结果或采样结果（按脚本定义落盘）。

## 调用关系

1. 不经 `scripts/run/` 主入口，也不默认由 Web 控制台触发。
2. 直接调用 `GNNTP.models.new.*` 中的模型实现及配套逻辑。

## 修改注意事项

1. 保持与主流程参数体系的最小一致性（如 `seed`、设备参数），便于结果对比。
2. 默认路径应基于仓库根目录推导，避免依赖当前工作目录。
3. 实验脚本输入格式调整后，要同步更新对应模型目录 `INFO.md` 和用法说明。
4. 独立实验结论若落地到主流程，需补齐注册、配置和 run 入口接线。
