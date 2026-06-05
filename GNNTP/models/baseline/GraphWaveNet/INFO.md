# Graph WaveNet

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_speed_prediction/GWNET.py`
- 论文: Graph WaveNet for Deep Spatial-Temporal Graph Modeling (IJCAI 2019)

## 迁移日期
2026-06-05

## 迁移说明
- 纯 import 路径替换: `libcity.model` → `GNNTP.models`
- 无架构改动，无自定义 dataset/executor
- 关键: adaptive adjacency matrix + dilated causal convolution + GCN
- 类名从 `GWNET` 改为 `GraphWaveNet` 以更好地识别

## ⚠️ 注意
Bigscity 源文件中类名为 `GWNET`，迁移后改为 `GraphWaveNet`。
manifest.json 中 `model` 字段设为 `GraphWaveNet`，运行时需使用 `--model GraphWaveNet`。
