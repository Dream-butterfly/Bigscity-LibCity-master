### 更改 6

时间：2026-04-21 00:00:00 +08:00
来源类型：提问
来源说明：用户要求在 `new_diffusion_fuzzy` 中实现模糊数学结构，并明确不要修改原 `new_diffusion_2` 目录。

更改类型-动作：新增功能/结构重构
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 仅在 `GNNTP/models/new/new_diffusion_fuzzy/` 目录内完成改造。
- 为扩散模型引入模糊数学结构，重点覆盖动态图学习与物理守恒约束。
- 保持与现有训练执行器和配置体系兼容。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`
- `GNNTP/models/new/new_diffusion_fuzzy/config.json`
- `GNNTP/models/new/new_diffusion_fuzzy/manifest.json`
- `GNNTP/models/new/new_diffusion_fuzzy/__init__.py`
- `GNNTP/models/new/new_diffusion_fuzzy/INFO.md`

变更摘要：
- 变更内容：
  - 在 `AdaptiveGraphLearner` 中新增模糊图学习分支：引入高斯隶属函数、多模糊子集中心、可学习规则权重，将相似度矩阵映射为模糊关系矩阵后参与自适应邻接构建。
  - 在 `AttentionDenoiser` 与 `NewDiffusion` 中增加模糊图相关配置透传与参数合法性校验。
  - 在 `_traffic_conservation_loss` 中增加基于状态强度的模糊拥堵隶属度加权，使高拥堵状态下守恒残差惩罚更强。
  - 修正 `new_diffusion_fuzzy` 的模型注册与包入口，避免误指向 `new_diffusion_2`。
  - 在默认配置中新增 `use_fuzzy_graph`、`fuzzy_graph_num_sets`、`fuzzy_graph_sigma_init`、`use_fuzzy_conservation` 等参数。
- 变更原因：
  - 交通图关系与守恒约束存在不确定性与状态依赖性，采用模糊隶属度可以更自然建模“强-弱关联”和“高-低拥堵”过渡区间。
  - 通过可开关配置保持可消融性，降低引入新机制后的回归定位成本。
- 影响范围：
  - 影响 `new_diffusion_fuzzy` 模型内部动态图构建与损失计算逻辑。
  - 不影响 `new_diffusion_2` 原实现及其目录内容。

后续迭代建议：
- 增加模糊开关消融实验（仅模糊图、仅模糊守恒、两者同时启用）并记录指标对比。
- 将 `scripts/experiments/train_new_diffusion_2.py` 复制为 `train_new_diffusion_fuzzy.py`，减少手动指定配置路径的误用风险。

修改注意事项：
- 当前模糊权重与隶属函数参数为默认可学习方案，建议在不同数据集上复核训练稳定性与采样质量。
