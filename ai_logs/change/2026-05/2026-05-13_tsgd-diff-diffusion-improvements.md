### 更改 17

时间：2026-05-13 +08:00

变更类型：架构增强（enhance）

变更状态：已验证（语法检查 + import + forward pass 通过）

需求/目标：
参考 TSGDiff 论文架构分析，对 new_diffusion_fuzzy_2 的逆向扩散去噪器做两项架构增强：
1. 每个 DenoiserBlock 重新注入扩散时间步信息（scale/shift 调制），解决时间步信号在深层稀释问题
2. Denoiser 内部 U-Net 风格 skip connections，保留浅层空间细节

变更文件：
GNNTP/models/new/new_diffusion_fuzzy_2/denoiser.py

变更摘要：

- 变更内容：
  - **DenoiserBlock**: 新增 `self.time_scale_shift` (SiLU + Linear(hidden_dim, hidden_dim*2))，forward 新增 `timestep_emb` 参数。在 block 入口处通过 scale/shift 调制重新注入扩散时间步信息，防止 4 层串行 block 中初始时间步信号衰减。
  - **AttentionDenoiser.forward**: 实现 U-Net 风格 skip connections — 前 mid 层（编码器）保存 skip 输出，后 (num_layers-mid) 层（解码器）加回镜像 skip。默认 4 层：block 0/1 编码 → block 2(+skips[1]) → block 3(+skips[0])。
  - 时间步注入保留两层：输入层全局 bias 加法（原有）+ 每层 block 入口 scale/shift（新增），类比扩散 UNet 每级重新注入的做法。

- 变更原因：
  - TSGDiff 使用 GAUNet (GAT + U-Net) 保留图结构和空间细节；当前模型纯串行 DenoiserBlock 在深层丢失空间细节，可能是峰值低估的贡献因素。
  - 标准扩散架构（DDPM UNet / DiT）在所有分辨率层级重新注入时间步；当前模型仅在输入层注入一次，4 层 block 中不再接触 t。

- 影响范围：仅 new_diffusion_fuzzy_2 的 denoiser.py；与 model.py / executor.py 接口向后兼容。
- 参数量增加（per block）：~hidden_dim * 2 * 2 ≈ 128*2*2*4 = ~66K (约 5% 增加)

后续迭代建议：

- [ ] 在 PEMSD4 标准基准上验证改进效果（MAE/RMSE/MAPE）
- [ ] 若 skip 增益显著，考虑扩展到 STEncoder 对称结构
- [ ] 若时间步调制增益显著，可升级为完整 adaLN（替换所有 LayerNorm）
