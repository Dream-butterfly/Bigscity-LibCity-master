# STAEformer

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_speed_prediction/STAEformer.py`
- 论文: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA (CIKM 2023)

## 迁移日期
2026-06-05

## 自定义组件
- **STAEformerDataset**: 非 one-hot 的 TOD (float 0-1) / DOW (int 0-6) 编码，适配 `nn.Embedding`
- 无自定义 executor/evaluator

## 迁移说明
- import 路径替换
- 依赖 `add_time_in_day=True`, `add_day_in_week=True`
- `input_dim=3` (value + TOD + DOW)
