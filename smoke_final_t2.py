"""极简 smoke test — 直接实例化模型并跑前向，不经过数据流水线。

用法:
    uv run python smoke_final_t2.py

CPU 上 <5 秒完成。验证:
    1. 模型可构建
    2. train/eval 前向通过
    3. 输出形状正确
"""

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.models.new.final_T2.model import NewFuzzyCellAttention

B, T, N, F, OUT = 4, 12, 20, 3, 1


def make_config(**overrides):
    cfg = {
        "input_window": T,
        "output_window": T,
        "hidden_dim": 96,
        "num_heads": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "ffn_hidden_dim": 256,
        "graph_k_hop": 1,
        "dropout": 0.1,
        "use_fuzzy_graph": True,
        "use_fuzzy_spatial_attn": True,
        "fuzzy_num_sets": 8,
        "use_cell_attention": True,
        "num_cells": 8,
        "use_hollow_kernel": True,
        "use_spatiotemporal_attention": True,
        "use_temporal_position_embedding": True,
        "use_torch_compile": False,
        "device": torch.device("cpu"),
        "decoder_node_mode": "proto",
        "use_per_node_beta": True,
        "encoder_use_fuzzy_graph": True,
        "conservation_loss_weight": 0.0,  # 跳过守恒 loss 的二次计算
        "t2_entropy_weight": 0.0,
        "t2_interval_weight": 0.0,
        "t2_fou_floor_weight": 0.0,
        "t2_fou_ceiling_weight": 0.0,
        "proto_norm_reg_weight": 0.0,
        "latent_norm_reg_weight": 0.0,
        "proto_diversity_weight": 0.0,
        "delta_diversity_weight": 0.0,
    }
    cfg.update(overrides)
    return cfg


def make_data_feature():
    return {
        "num_nodes": N,
        "feature_dim": F,
        "output_dim": OUT,
        "scaler": None,
        "adj_mx": torch.eye(N, dtype=torch.float32),
    }


def main():
    torch.manual_seed(0)
    t0 = time.time()

    model = NewFuzzyCellAttention(make_config(), make_data_feature())
    print(f"[OK] model built in {time.time()-t0:.2f}s, "
          f"params={sum(p.numel() for p in model.parameters())}")

    X = torch.randn(B, T, N, F)
    y = torch.randn(B, T, N, OUT)
    batch = {"X": X, "y": y}

    # ── eval forward ──
    model.eval()
    with torch.no_grad():
        out = model.predict(batch)
    print(f"[OK] eval forward: {out.shape}")
    assert out.shape == (B, T, N, OUT), f"unexpected shape {out.shape}"

    # ── train forward (calculate_loss) ──
    model.train()
    loss = model.calculate_loss(batch)
    print(f"[OK] train forward: loss={loss.item():.4f}")

    # ── backward ──
    loss.backward()
    has_grad = [p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()]
    print(f"[OK] backward passed, {sum(has_grad)}/{len(has_grad)} params got grad")

    # ── fuzzy diagnostics ──
    diag = model.get_type2_diagnostics()
    print(f"[OK] diagnostics: keys={sorted(diag.keys())[:8]}...")

    print(f"\nALL PASSED in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
