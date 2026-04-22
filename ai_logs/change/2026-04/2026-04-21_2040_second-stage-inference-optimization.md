### 更改 11

时间：2026-04-21 20:40:00 +08:00
来源类型：提问
来源说明：用户要求继续执行第二轮优化，并同步写入日志。

更改类型-动作：性能优化/实验流程变更
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 在第一阶段稳训优化之后，按既定顺序落地第二阶段推理降耗。
- 在不改核心网络结构和接口的前提下，降低 `new_diffusion_fuzzy` 的推理时延。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`
- `GNNTP/models/new/new_diffusion_fuzzy/config.json`
- `GNNTP/models/new/new_diffusion_fuzzy/INFO.md`
- `ai_logs/change/2026-04/2026-04-21_2040_second-stage-inference-optimization.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 调整默认推理预算（`config.json`）：
    - `num_sampling_steps`：`200 -> 50`
    - `num_prediction_samples`：`4 -> 2`
  - 优化采样执行路径（`model.py`）：
    - 新增 `_get_sampling_schedule()`，按 device 缓存采样时间步序列，避免每次推理重复构造 `torch.linspace`。
    - 将多样本采样由 Python 循环改为批内并行采样（扩展 batch 一次反向扩散后 reshape 回 `[samples, B, T, N, C]`），减少调度开销。
    - 增加 `num_prediction_samples` / `num_samples` 的下界校验（必须 `>=1`），防止无效配置导致隐式错误。
  - 在 `INFO.md` 补充第二轮推理降耗说明。
- 变更原因：
  - 第二阶段目标是“先降推理成本，再做更激进容量调参”；优先调整对精度影响可控、可回滚的采样预算与执行路径。
  - 推理时间近似受 `sampling_steps * prediction_samples` 线性影响，本次默认预算可显著降低 wall-clock 开销。
- 影响范围：
  - 仅影响 `new_diffusion_fuzzy` 的推理采样预算与采样执行效率。
  - 训练目标、核心网络结构、模型输入输出接口与注册方式保持不变。

后续迭代建议：
- 若需进一步压缩时延，可继续对比 `num_sampling_steps=25` 与 `num_prediction_samples=1` 的精度-时延折中点。
- 在验证集上补充“采样预算网格”对照记录（MAE/RMSE/MAPE + 推理时长）作为后续固定配置依据。

修改注意事项：
- 预算下调可能带来一定精度波动，建议在目标数据集上做短周期对照后再固化到生产实验模板。
- 并行采样会提升单次推理峰值显存占用，若资源受限可临时降低 `num_prediction_samples`。
