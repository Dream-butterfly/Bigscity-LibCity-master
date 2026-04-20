# core/INFO.md

## 目录职责
`data/core/` 提供数据层基础结构，支撑上层 dataset 与 dataloader。

## 关键文件
- `batch.py`：批数据容器与处理逻辑。
- `list_dataset.py`：列表型数据集封装。

## 输入/输出
- 输入：样本列表、特征字段。
- 输出：标准化批数据对象。

## 调用关系
- 被 `data/dataloader.py` 与 `data/dataset/*` 间接使用。

## 修改注意事项
1. 修改批结构前需确认对全部模型输入兼容。
2. 字段命名变更要同步更新下游读取逻辑。

