# PDFormer

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_flow_prediction/PDFormer.py`
- 论文: PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction (AAAI 2023)

## 重新迁移日期
2026-06-05

## 自定义组件
- **PDFormerDataset**: DTW 矩阵 + 短路径矩阵 (sd_mx/sh_mx) + KShape 模式聚类 (pattern_keys)
- **PDFormerExecutor**: Laplacian PE 预计算 + CosineLRScheduler + gradient accumulation + curriculum learning

## 重新迁移说明
从 Bigscity 源码完整重写三件套 (model/dataset/executor)，仅替换 import 路径和缓存路径。
与原 Bigscity 实现完全一致，使用 fastdtw (非自定义 DTW)。

## 关键配置
- `type_ln: "post"` (后归一化)
- `lr_scheduler: "cosinelr"` (warmup + cosine decay)
- `use_curriculum_learning: true` (逐步增加预测步数)
- `random_flip: true` (Laplacian PE 随机翻转)
