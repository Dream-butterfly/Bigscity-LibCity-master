# GNNTP/models/new/INFO.md

实验性/新增模型目录，用于快速迭代验证新模型架构，不影响稳定模型目录。

## 关键子目录

| 目录 | 状态 | 说明 |
|------|------|------|
| `NEW_MODEL/` | 🟡 模板 | 新模型接入模板，复制并重命名后使用 |
| `new_diffusion/` | 🔴 实验 v1 | 扩散类实验模型 v1（独立实验脚本 + 主流程接入） |
| `new_diffusion_2/` | 🔴 实验 v2 | 扩散类实验模型 v2（v1 改进版） |
| `new_diffusion_fuzzy/` | 🟢 当前工作 | 模糊图学习 + 模糊守恒损失 + 条件扩散（DDIM/DDPM），有专用 Executor |

## new_diffusion_fuzzy 简介

当前核心实验模型，架构：STEncoder(condition) → AttentionDenoiser(ε_θ) → DDIM 50-step 采样。

关键文件：`model.py`(621行) + `executor.py`(DDP-safe) + `utils/`(adjacency/attention/time_embedding)

详见子目录 `INFO.md`。

## 输入/输出

- **输入**：标准交通时空批数据 + 模型配置 + 实验参数
- **输出**：预测结果、训练损失、checkpoint

## 调用关系

1. 通过 manifest.json 由 `models/locator.py` 索引 → 主流程 `run_train_artifact.py` 可调用
2. `new_diffusion_fuzzy` 有自定义 executor（`DiffusionTrafficStateExecutor`）
3. `new_diffusion`/`new_diffusion_2` 也可通过 `scripts/experiments/` 独立脚本运行

## 修改注意事项

1. 实验模型稳定后应沉淀注册信息、配置说明和目录文档
2. 保持 `config.json`、`manifest.json`、代码参数读取名一致
3. 明确”仅实验”与”可上线主流程”状态，避免误用
4. `new_diffusion_fuzzy` 的配置参数（denoiser_layers/diffusion_steps 等）修改需同步 checkpoint 兼容性

