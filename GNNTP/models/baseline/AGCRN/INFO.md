# AGCRN

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_flow_prediction/AGCRN.py`
- 论文: Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting (NeurIPS 2020)

## 迁移日期
2026-06-05

## 迁移说明
- 纯 import 路径替换: `libcity.model` → `GNNTP.models`
- 无架构改动，无自定义 dataset/executor
- 核心: AVWGCN + AGCRNCell (自适应图学习 + 门控循环)
