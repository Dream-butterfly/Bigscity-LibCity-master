### 更改 12

时间：2026-04-22 11:28:48 +08:00
来源类型：提问
来源说明：用户要求完成第三轮优化，并同步写入日志。

更改类型-动作：性能优化/实验流程变更
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 在第一轮稳训、第二轮推理降耗之后，按既定顺序落地第三轮容量缩放优化。
- 在不改 `new_diffusion_fuzzy` 核心结构和接口的前提下，进一步降低训练与推理资源开销。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/config.json`
- `GNNTP/models/new/new_diffusion_fuzzy/INFO.md`
- `ai_logs/change/2026-04/2026-04-22_1128_third-stage-capacity-scaling-optimization.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 调整默认容量配置（`config.json`）：
    - `hidden_dim`：`128 -> 96`
    - `denoiser_layers`：`4 -> 3`
    - `ffn_hidden_dim`：`256 -> 192`
    - `adaptive_graph_topk`：`20 -> 12`
  - 在 `INFO.md` 增补第三轮容量缩放说明。
- 变更原因：
  - 第三阶段目标是“在不改核心结构前提下做容量缩放”，优先下调主要计算热点维度与层数，降低显存和时延压力。
  - 相比结构改造，默认参数回退成本更低，便于按数据集做精度-资源折中。
- 影响范围：
  - 仅影响 `new_diffusion_fuzzy` 默认容量与自适应图稀疏度。
  - 模型核心网络组件、输入输出接口、注册方式与训练流程保持不变。

后续迭代建议：
- 在目标数据集上补一组小网格对照（如 `hidden_dim={96,128}`、`denoiser_layers={3,4}`），联合 MAE/RMSE 与 epoch 时长确认最终默认值。
- 若仍需降本，可进入第四阶段工程加速（如数据加载与验证频率策略）。

修改注意事项：
- 容量下调可能带来精度波动，建议在实验模板中明确“轻量默认值”与“高精度回退值”。
