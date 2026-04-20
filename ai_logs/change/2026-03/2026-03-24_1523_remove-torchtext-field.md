### 更改 1

时间：2026-03-24 15:23:00 +08:00

变更类型：修复问题 / 结构重构 / 依赖变更
变更状态：已应用

需求/目标：解决运行 LibCity 时因 torchtext 依赖导致的动态库加载失败（libtorchtext.so undefined symbol），确保数据集加载流程可正常执行。

变更文件：
- libcity/data/dataset/dataset_subclass/geosan_dataset.py
- pyproject.toml

变更摘要：
- 变更内容：
  - 移除 `geosan_dataset.py` 中对 `torchtext.data.Field` 的依赖
  - 使用基于 `torch + 原生 Python` 的本地最小实现替代 Field，支持：
    - 词表构建（token → index）
    - padding token 固定为 index=0
    - token 序列数值化（numericalize）
  - 删除 `pyproject.toml` 中 torchtext 依赖声明

- 变更原因：
  - torchtext 已停止活跃维护，且与当前 torch（2.10.0）存在 ABI 不兼容问题
  - dataset_subclass/__init__.py 全量导入导致无关模块（GeoSANDataset）强制触发 torchtext 加载
  - 实际使用中仅依赖 Field 的极少数功能，无需完整库

- 影响范围：
  - 仅影响 GeoSANDataset 数据处理逻辑
  - 不影响 ASTGCN / PEMS 系列数据集
  - 移除 torchtext 后减少环境依赖复杂度
  - 可能影响 GeoSAN 模型的文本处理细节（需验证一致性）

后续迭代建议：
- [ ] 为 GeoSANDataset 增加单元测试（词表一致性 / padding 行为）
- [ ] 考虑重构 dataset_subclass/__init__.py，避免全量导入
- [ ] 检查其他潜在隐式依赖（如 torch_geometric 等）

修改注意事项：
- 本地实现需保证 padding index=0，否则 embedding 行为可能异常
- 若后续使用 GeoSAN 相关模型，需验证结果与原 torchtext 版本一致
- 当前修改未解决“全量导入设计缺陷”，仅规避其副作用
