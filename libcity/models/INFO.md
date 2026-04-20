# models/INFO.md

## 目录职责
`models/` 是模型定义中心，包含抽象基类、模型注册器和具体模型实现。

## 关键文件
- `abstract_model.py` / `abstract_traffic_state_model.py`：模型抽象层。
- `registry.py` / `locator.py`：模型查找与注册逻辑。
- `loss.py`：损失函数工具。
- `traffic_speed_prediction/`、`traffic_flow_prediction/`、`new/`：具体模型目录。

## 输入/输出
- 输入：数据特征、配置与批数据。
- 输出：预测结果、训练损失及中间状态。

## 调用关系
- 上游由 `pipeline.py` 创建并交给 executor 训练。
- 下游模型目录按任务细分实现。

## 修改注意事项
1. 新模型需补齐 `manifest/config` 并接入注册器。
2. 模型输入输出形状要与 dataset/executor 保持一致。

