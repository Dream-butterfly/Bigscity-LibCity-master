# scripts/experiments/INFO.md

## 目录职责
`scripts/experiments/` 存放不走主训练流水线的独立实验入口脚本。

## 关键文件
- `train_new_diffusion.py`：`new_diffusion` 的 NPZ 独立训练入口。
- `train_new_diffusion_2.py`：`new_diffusion_2` 的 NPZ 独立训练入口。

## 输入/输出
- 输入：实验专用数据文件、模型配置和命令行参数。
- 输出：独立训练日志、checkpoint 与采样结果。

## 调用关系
- 仅供开发者手动执行，不被 `scripts/run/` 或 Web 控制台直接调用。
- 会复用 `GNNTP.models` 中的模型定义。

## 修改注意事项
1. 路径默认值应基于项目根目录计算，避免依赖当前工作目录。
2. 独立实验输入格式若变化，需同步更新对应模型目录下的 `INFO.md`。
