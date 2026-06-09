# NEW_MODEL — 新模型接入模板

复制此目录并重命名，快速验证”model.py + config.json + manifest.json”接线。

## 关键文件

| 文件 | 作用 |
|------|------|
| `model.py` | 模型模板实现（继承 AbstractTrafficStateModel） |
| `config.json` | 默认参数模板 |
| `manifest.json` | 注册信息模板 |

## 使用步骤

1. `cp -r NEW_MODEL <your_model_name>`
2. 编辑 `manifest.json`：修改 model 字段为模型名
3. 编辑 `model.py`：实现前向逻辑
4. 编辑 `config.json`：设置默认超参数
5. 使用 `run_train_artifact.py --model <your_model_name> --dataset <dataset>` 验证

## 输入/输出

- **输入**：标准任务批数据 + data_feature + config
- **输出**：预测张量（由 executor 计算损失和评估）

## 修改注意事项

1. 必须同步更新目录名、模型名和 `manifest.json` 的 model 字段，三者一致
2. 参数键名保持 `config.json` 与 `model.py` 读取一致性

