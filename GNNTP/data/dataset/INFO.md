# dataset/INFO.md

## 目录职责
`data/dataset/` 存放数据集抽象类与交通状态任务的数据集实现。

## 关键文件
- `abstract_dataset.py`：数据集抽象基类。
- `traffic_state_dataset.py`：交通状态数据集主实现。
- `traffic_state_dataset_mixins.py`：可复用的数据处理 mixin。
- `traffic_state_point_dataset.py`：点位型交通状态数据集。
- `traffic_flow_prediction/`：交通流预测任务专用数据集实现。

## 输入/输出
- 输入：原子文件与数据配置（窗口长度、划分比例、归一化等）。
- 输出：训练/验证/测试数据及特征信息。

## 调用关系
- 被 `data/factory.py` 实例化并交给 `pipeline.py` 使用。

## 修改注意事项
1. 修改字段名或形状要同步模型侧输入约定。
2. 配置项变化需同步 JSON 配置模板。

