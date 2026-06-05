# MTGNN

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_speed_prediction/MTGNN.py`
- 论文: Connecting the Dots: Multivariate Time Series Forecasting with GNNs (KDD 2020)

## 迁移日期
2026-06-05

## 自定义组件
- **MTGNNExecutor**: 节点拆分训练 (node-splitting), num_split 个子图 + step_size2 控制重排

## 迁移说明
- import 路径替换: `libcity.model` → `GNNTP.models`
- 自定义 LayerNorm 接受 idx 参数用于节点索引子集
- GraphConstructor 在子图上构建自适应邻接矩阵
