### 更改 20

时间：2026-05-26
来源类型：提问
来源说明：用户要求修复 DDP 多卡训练 val_loss 同步和模型保存竞态，并将扩散训练损失从 MSE 改为 MAE

更改类型-动作：修复问题
更改类型-范围：跨模块
变更状态：已应用

需求/目标：
1. 基类 `_valid_epoch` 缺少 `all_reduce`，导致 STGCN/STTN 在 DDP 下 val_loss 不一致，引发早停错乱
2. `save_model` / `save_model_with_epoch` 缺少 `_is_rank0()` 保护，多 rank 同时写文件存在竞态
3. new_diffusion_fuzzy 训练损失从 MSE 改为 MAE

变更文件：
- `GNNTP/common/traffic_state_executor.py` — 三处修改
- `GNNTP/models/new/new_diffusion_fuzzy/model.py` — 一处修改

变更摘要：
- 变更内容：
  1. `_valid_epoch` (L506-508): 新增 DDP `all_reduce(AVG)` 求全局平均 val_loss
  2. `save_model` (L138): 新增 `if not self._is_rank0(): return` 保护
  3. `save_model_with_epoch` (L163): 新增 `if not self._is_rank0(): return ""` 保护
  4. `calculate_loss` (L802): `F.mse_loss` → `F.l1_loss`
- 变更原因：
  1. 基类 _valid_epoch 无 all_reduce → 各 rank val_loss 不一致 → 早停各 rank 不同步、模型保存决策不一致
  2. 所有 7 个 executor 调用 save_model/save_model_with_epoch 时均无 rank0 保护
  3. 用户偏好 MAE，交通预测领域普遍选择
- 影响范围：
  修改 1: 仅影响 STGCN/STTN（其他模型已有各自 all_reduce）
  修改 2/3: 影响所有模型（基类方法）
  修改 4: 仅影响 new_diffusion_fuzzy
