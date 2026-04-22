# new_diffusion_fuzzy/INFO.md

## 目录职责

实现 `new_diffusion_fuzzy` 扩散实验模型，在 `new_diffusion_2` 基础上引入模糊数学结构。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | 模型结构与前向逻辑 |
| `utils/adjacency.py` | 邻接矩阵归一化与批量扩展工具 |
| `utils/attention_ops.py` | 时空注意力张量变换辅助逻辑 |
| `utils/time_embedding.py` | 扩散时间步正弦嵌入 |
| `utils/__init__.py` | 工具模块导出聚合 |
| `config.json` | 默认实验参数 |
| `manifest.json` | 模型注册元信息 |

## 模糊化改动

1. 在 `AdaptiveGraphLearner` 中加入模糊隶属度与模糊关系矩阵构造（高斯隶属函数 + 可学习规则权重）。
2. 在交通守恒损失中加入基于状态强度的模糊权重，对高拥堵状态赋予更高约束权重。
3. 通过配置项开关控制模糊图学习与模糊守恒损失，便于消融实验。
4. 一些无状态辅助逻辑已下沉到 `utils/`，减少 `model.py` 的长度与重复。
5. 已加入物理损失预热机制（`physics_warmup_*`），用于降低早期训练震荡风险。
6. 已做第二轮推理降耗：默认 `num_sampling_steps=50`、`num_prediction_samples=2`，并在采样阶段引入时间步调度缓存与并行多样本采样，减少推理 wall-clock 开销。
7. 已做第三轮容量缩放：默认 `hidden_dim=96`、`denoiser_layers=3`、`ffn_hidden_dim=192`、`adaptive_graph_topk=12`，在不改核心结构的前提下降低训练/推理计算与显存压力。

## 输入/输出

- **输入**：扩散模型兼容的时空批数据与训练参数。
- **输出**：预测结果、损失与训练中间状态。

## 调用关系

1. 可通过注册机制接入主流程。
2. 可复用独立实验入口并指定当前目录配置（如 `scripts/experiments/train_new_diffusion_2.py` + `new_diffusion_fuzzy/config.json`）。

## 修改注意事项

1. 保持与 `new_diffusion_2` 的差异可追踪（参数、结构、训练策略）。
2. 接口调整需同步检查数据、执行器和独立实验脚本兼容性。
3. 参数更新优先维护 `config.json`，减少代码硬编码。

