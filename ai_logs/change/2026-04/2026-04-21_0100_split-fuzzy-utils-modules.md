### 更改 9

时间：2026-04-21 01:00:00 +08:00
来源类型：提问
来源说明：用户要求将 `new_diffusion_fuzzy/utils` 从单一 `__init__.py` 进一步拆分为按功能命名的多个模块文件。

更改类型-动作：结构重构
更改类型-范围：局部模块
变更状态：已应用

需求/目标：
- 不把全部实现堆在 `utils/__init__.py`。
- 按功能拆分为多个 `.py` 文件并保留清晰命名。
- 保持 `new_diffusion_fuzzy` 模型行为与接口不变。

变更文件：
- `GNNTP/models/new/new_diffusion_fuzzy/utils/time_embedding.py`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/adjacency.py`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/attention_ops.py`
- `GNNTP/models/new/new_diffusion_fuzzy/utils/__init__.py`
- `GNNTP/models/new/new_diffusion_fuzzy/model.py`
- `GNNTP/models/new/new_diffusion_fuzzy/INFO.md`
- `ai_logs/index.md`

变更摘要：
- 变更内容：
  - 新增 `time_embedding.py`，承载 `SinusoidalTimeEmbedding`。
  - 新增 `adjacency.py`，承载 `build_normalized_adjacency` 与 `expand_adjacency_batch`。
  - 新增 `attention_ops.py`，承载时空 attention 相关张量辅助函数。
  - `utils/__init__.py` 改为轻量导出层，仅做模块聚合，不再放具体实现。
  - `model.py` 保持 `from .utils import ...` 调用方式，并删除残留同名本地函数，避免导入后被覆盖。
  - 更新 `INFO.md`，明确 `utils` 下按功能拆分后的文件职责。
- 变更原因：
  - 减少单文件拥挤与职责混杂，提升可读性和后续维护效率。
  - 按功能命名模块，便于定位和复用。
- 影响范围：
  - 影响 `new_diffusion_fuzzy` 包内工具函数组织结构。
  - 不改变模型主流程、配置参数或注册入口。

后续迭代建议：
- 若继续拆分，可把 `model.py` 中基础层模块（如注意力、图卷积、FFN）进一步抽到 `layers/` 子包。

修改注意事项：
- `utils/__init__.py` 已变为导出聚合层；后续新增工具建议优先放到功能模块，再在 `__init__.py` 导出。
