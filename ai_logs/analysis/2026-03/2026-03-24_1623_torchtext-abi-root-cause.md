### 分析 4
时间：2026-03-24 16:23:00 +08:00

分析类型：调研 / 理解 / 验证 / 风险
结论状态：已验证

输入内容：
用户运行 GNNTP 时出现错误：
- 表面：AttributeError（ASTGCNDataset 不存在）
- 实际：torchtext 动态库加载失败（libtorchtext.so）

分析结果：
- 调用链如下：
  1. `get_dataset()` → 动态加载 dataset_subclass
  2. `dataset_subclass/__init__.py` 执行全量导入
  3. 强制导入 GeoSANDataset
  4. GeoSANDataset 依赖 torchtext.Field
  5. torchtext 加载 C++ 扩展失败（ABI 不兼容）

- 关键问题分层：
  1. 设计问题：
     - dataset_subclass 使用“全量导入”模式
     - 无关数据集也会被加载

  2. 环境问题（致命）：
     - torch==2.10.0
     - torchtext 未锁版本
     - 导致 libtorchtext.so 与 torch ABI 不匹配

- torchtext 使用范围分析：
  - 仅用到：
    - build_vocab()
    - vocab.itos
    - numericalize()
  - 不依赖复杂 NLP 功能

关键结论：
- 报错根因不是 ASTGCNDataset 缺失，而是：
  **torchtext 与 torch 二进制不兼容导致导入链中断**

- 根因可分为两层：
  1. 架构问题：dataset_subclass 全量导入设计不合理
  2. 环境问题：torch / torchtext ABI 不匹配

- 可行修复路径：
  1. 代码修复（推荐）：移除 torchtext 依赖
  2. 环境修复：对齐 torch/torchtext 版本

启发，或是下一步可以完成的工作：
- [ ] 重构 dataset_subclass 为懒加载机制
- [ ] 建立依赖版本锁定策略（避免 ABI 冲突）
- [ ] 对所有 dataset 依赖做最小化审查

补充说明：
- 风险点：类似问题可能在 torch_geometric 等库中复现
- 适用范围：GNNTP 数据集加载机制分析
- 限制条件：仅针对当前 Python 3.12 + torch 2.10 环境
