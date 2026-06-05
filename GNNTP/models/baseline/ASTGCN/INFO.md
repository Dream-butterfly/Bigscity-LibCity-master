# ASTGCN

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_flow_prediction/ASTGCN.py`
- 论文: Attention Based Spatial-Temporal Graph Convolutional Networks (AAAI 2019)

## 迁移日期
2026-06-05

## 自定义组件
- **ASTGCNDataset**: CPT (closeness/period/trend) 三段采样策略
- 无自定义 executor/evaluator

## 迁移说明
- import 路径替换
- CPT 采样覆盖标准滑动窗口: `len_closeness=3`, `len_period=4`, `len_trend=0`
