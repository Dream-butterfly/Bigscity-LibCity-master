# STGCN/INFO.md

## 目录职责
实现 STGCN 交通速度预测模型。

## 关键文件
- `model.py`：STGCN 模型结构与前向逻辑。
- `config.json` / `manifest.json`：配置与注册信息。

## 输入/输出
- 输入：交通速度时空序列与图结构。
- 输出：未来时间步速度预测。

## 调用关系
- 由通用 executor 调用模型进行训练和推理。

## 修改注意事项
1. 卷积层输入维度调整时，需校验与数据 shape 对齐。
2. 参数默认值建议维护在 `config.json`。

