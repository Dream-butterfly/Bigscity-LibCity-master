# final_T2 (MVF-STGFormer) — Architecture Reference

> ⚠️ 本文档已归档。完整架构描述见 `paper/final_T2模型架构.md`（中文详细版）和 `paper/final_T2方法论简述.md`（英文学术稿）。本文件仅保留快速索引。

## 核心组件

| 组件 | 文件 | 配置来源 |
|------|------|---------|
| MV-FRGL 模糊图学习器 | `graph.py` | `config.json` |
| 编码器 (STEncoder) | `encoder.py` | 静态邻接矩阵 |
| 解码器 (FutureDecoder) | `decoder.py` | 模糊关系矩阵 |
| 模糊单元注意力 | `cell_attention.py` | MDI 门控 |
| 主模型 | `model.py` | 整合所有模块 + 损失函数 |
| 注意力/FFN | `attention.py` | 多头注意力 + 前馈网络 |

## 当前配置快照

| 参数 | 值 | 参数 | 值 |
|------|----|------|----|
| `hidden_dim` | 96 | `num_heads` | 4 |
| `encoder_layers` | 2 | `decoder_layers` | 1 |
| `ffn_hidden_dim` | 256 | `fuzzy_num_sets` | 8 |
| `num_cells` | 16 | `graph_k_hop` | 1 |
| `temp_mix_dilation` | 2 | `batch_size` | 32 |

## 输出投影

解码器输出投影为 **temp_mix Conv1D + 逐步投影** 两级结构：
- temp_mix: 两层膨胀 Conv1D (kernel=3, dilation 可配)，约 28K 参数
- 逐步投影: `Linear(D→32) → GELU → Linear(32→O)`，约 3.1K 参数

详见 `paper/final_T2模型架构.md` §4.4。

## 损失函数

9 项组成：MAE + conservation + proto_norm + latent_norm + proto_diversity + delta_diversity + β_entropy + interval_ratio + FOU floor/ceiling。

详见 `paper/final_T2方法论简述.md` §3.5。
