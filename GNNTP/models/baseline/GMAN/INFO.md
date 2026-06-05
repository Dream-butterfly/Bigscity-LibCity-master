# GMAN

## 来源
- Bigscity-LibCity-master: `libcity/model/traffic_speed_prediction/GMAN.py`
- 论文: GMAN: A Graph Multi-Attention Network for Traffic Prediction (AAAI 2020)

## 迁移日期
2026-06-05

## 自定义组件
- **GMANDataset**: Node2Vec 预计算空间嵌入 (SE)，依赖 `gensim.Word2Vec`
- 无自定义 executor/evaluator

## 迁移说明
- import 路径替换: `libcity.model` → `GNNTP.models`
- 数据集缓存路径: `get_dataset_cache_dir()`
- 需安装 `gensim` 和 `networkx`
