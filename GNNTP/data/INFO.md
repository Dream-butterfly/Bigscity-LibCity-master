# data/INFO.md

## 目录职责
`data/` 负责数据集构建、切分、dataloader 生成和数据模块注册。

## 关键文件
- `factory.py` / `registry.py`：数据集对象创建与注册。
- `dataloader.py`：批数据加载逻辑。
- `dataset/`：数据集实现与任务相关数据处理。
- `core/`：批处理与列表数据集等基础结构。

## 输入/输出
- 输入：`resource_data/` 中原子文件与配置参数。
- 输出：`train/valid/test` 数据迭代器与数据特征字典。

## 调用关系
- 上游由 `pipeline.py`、`scripts/run/run_data_prep.py` 调用。
- 下游会使用 `dataset/` 子模块完成具体任务数据处理。

## 修改注意事项
1. 新增数据集类要加入注册流程。
2. 变更切分/归一化逻辑时需考虑缓存兼容性。

