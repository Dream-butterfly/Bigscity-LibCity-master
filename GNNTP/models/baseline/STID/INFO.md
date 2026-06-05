# STID

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_speed_prediction/STID.py`
- 论文: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting (arXiv:2208.05233)

## 迁移日期
2026-06-05

## 迁移说明
- 纯 import 路径替换: `libcity.model` → `GNNTP.models`
- 无架构改动，无自定义 dataset/executor
- 依赖 `add_time_in_day=True` + `add_day_in_week=True` 提供时序嵌入
