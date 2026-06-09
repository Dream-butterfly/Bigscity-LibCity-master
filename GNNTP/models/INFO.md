# GNNTP/models/INFO.md

模型实现中心：模型抽象基类、注册定位器、四层模型目录、损失函数工具。

## 关键文件

| 路径 | 作用 |
|------|------|
| `abstract_model.py` | 模型抽象基类（533B） |
| `abstract_traffic_state_model.py` | 交通状态任务模型抽象层（895B） |
| `locator.py` | **模型按名称定位**：扫描所有子目录的 manifest.json → 返回模型/执行器/数据集类路径 |
| `registry.py` | 模型类直接注册（157B，与 locator 互补） |
| `loss.py` | 损失函数与训练辅助损失工具（11.7K，含 masked MAE/MSE 等） |

## 四层模型目录

| 目录 | 用途 | 包含模型 |
|------|------|----------|
| `traffic_speed_prediction/` | 交通速度预测 | STGCN、DCRNN、STGformer、STTN、STGformer-独立实现 |
| `traffic_flow_prediction/` | 交通流预测 | PDFormer |
| `new/` | 实验性/新增模型 | new_diffusion(v1)、new_diffusion_2(v2)、new_diffusion_fuzzy、NEW_MODEL(模板) |
| `baseline/` | 迁移的基准模型 | STID、AGCRN、GraphWaveNet、GMAN、STAEformer、ASTGCN、PDFormer、MTGNN |

## 模型接入规范

每个模型目录必须包含：

| 文件 | 必选 | 说明 |
|------|------|------|
| `model.py` | ✅ | 模型结构 + 前向逻辑（继承 AbstractTrafficStateModel） |
| `config.json` | ✅ | 默认超参数配置 |
| `manifest.json` | ✅ | 注册元信息（task/model/dataset_class/executor/evaluator） |
| `executor.py` | ❌ | 有专用执行器时需要（如 DCRNN、PDFormer、new_diffusion_fuzzy） |
| `executor.json` | ❌ | 专用执行器的注册信息 |

## 输入/输出（黑盒约定）

- **输入**：`data_feature`（scaler, num_nodes, feat_dim, output_dim）+ `config`（超参数）
- **前向**：`model(x: [B, in_window, N, feat_dim]) → y: [B, out_window, N, output_dim]`
- **输出**：预测张量（由 executor 调用 scaler.inverse_transform 还原为原始量纲）

## 修改注意事项

1. 新模型必须提供 `model.py` + `config.json` + `manifest.json`，三者缺一不可
2. `manifest.json` 中的 `model` 字段必须与目录名和 CLI `--model` 一致
3. 输入输出 shape 变更须同步检查数据集与执行器逻辑
4. 实验模型（`new/`）稳定后应沉淀到主任务目录或 baseline
5. 新增模型后检查 `locator.py` 能否通过 manifest.json 扫描发现

