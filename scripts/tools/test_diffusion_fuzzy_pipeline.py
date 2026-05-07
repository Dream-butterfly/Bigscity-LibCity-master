"""
模块化诊断测试：new_diffusion_fuzzy 管线各环节

覆盖 7 个独立测试，每个用合成数据快速验证，无需加载真实数据集。

用法:
  cd /path/to/LibCity
  python scripts/tools/test_diffusion_fuzzy_pipeline.py       # 全部测试
  python scripts/tools/test_diffusion_fuzzy_pipeline.py --idx 3  # 只跑第3个
  python scripts/tools/test_diffusion_fuzzy_pipeline.py --list   # 列出测试
"""

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from GNNTP.models.new.new_diffusion_fuzzy.model import (
    DiffusionScheduler,
    STEncoder,
    AttentionDenoiser,
    NewDiffusion,
)
from GNNTP.utils.normalization import StandardScaler, MinMax01Scaler

# ── helpers ──────────────────────────────────────────────────────────

_DEFAULT_CFG = {
    "input_window": 12,
    "output_window": 12,
    "hidden_dim": 64,
    "num_heads": 4,
    "encoder_layers": 2,
    "denoiser_layers": 2,
    "ffn_hidden_dim": 128,
    "graph_k_hop": 2,
    "dropout": 0.0,
    "diffusion_steps": 20,
    "diffusion_schedule": "linear",
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "num_sampling_steps": 10,
    "num_prediction_samples": 1,
    "sampling_method": "ddim",
    "ddim_eta": 0.0,
    "use_spatiotemporal_attention": False,
    "use_temporal_position_embedding": True,
    "use_gradient_checkpointing": False,
    "use_adaptive_graph": True,
    "adaptive_graph_embed_dim": 16,
    "adaptive_graph_topk": 5,
    "adaptive_graph_blend_init": 0.5,
    "use_fuzzy_graph": True,
    "fuzzy_graph_num_sets": 3,
    "fuzzy_graph_sigma_init": 0.7,
    "physics_loss_weight": 0.0,
    "physics_warmup_steps": 0,
    "use_fuzzy_conservation": False,
    "device": torch.device("cpu"),
}


def make_config(**overrides):
    """Minimal config dict for building components. Accepts kwargs overrides."""
    cfg = dict(_DEFAULT_CFG)
    cfg.update(overrides)
    return cfg


def make_data_feature(num_nodes=10, feature_dim=3, output_dim=1):
    """Minimal data_feature dict matching LibCity format."""
    import numpy as np
    return {
        "num_nodes": num_nodes,
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "adj_mx": np.eye(num_nodes, dtype=np.float32),
        "scaler": None,
    }


def make_synthetic_batch(batch_size=4, input_window=12, output_window=12,
                          num_nodes=10, feature_dim=3, output_dim=1):
    """Generate a synthetic batch dict matching LibCity convention."""
    X = torch.randn(batch_size, input_window, num_nodes, feature_dim)
    y = torch.randn(batch_size, output_window, num_nodes, output_dim)
    return {"X": X, "y": y}


PASS = 0
FAIL = 1
total_pass = 0
total_fail = 0


def run_test(idx, name, fn):
    """Run one test and report pass/fail."""
    global total_pass, total_fail
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  [{idx}] ✅ PASS  {name}  ({elapsed:.3f}s)")
        total_pass += 1
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [{idx}] ❌ FAIL  {name}  ({elapsed:.3f}s)")
        print(f"       {type(e).__name__}: {e}")
        total_fail += 1


# ══════════════════════════════════════════════════════════════════════
# TEST 1: DiffusionScheduler 数学正确性
# ══════════════════════════════════════════════════════════════════════

def test_scheduler_add_noise():
    """验证 add_noise 公式 Y_t = sqrt(ᾱ)·Y_0 + sqrt(1-ᾱ)·ε"""
    scheduler = DiffusionScheduler(diffusion_steps=200, beta_start=1e-4, beta_end=0.02)
    x0 = torch.randn(2, 12, 10, 1)
    t = torch.tensor([199, 199], dtype=torch.long)  # max noise level
    noise = torch.ones_like(x0)
    xt, eps = scheduler.add_noise(x0, t, noise=noise)  # deterministic noise

    assert eps.shape == x0.shape, f"noise shape {eps.shape} != {x0.shape}"
    assert torch.allclose(eps, noise), "noise should be passed through unchanged"
    # At t=max, alpha_bar ≈ 0 → xt ≈ noise
    # Compute diff between xt and noise relative to signal magnitude
    signal_mag = noise.abs().mean()
    residual = (xt - noise).abs().mean()
    # At t=199 with linear schedule, alpha_bar[199] ≈ 0.02 → xt ≈ 0.14*signal + 0.99*noise
    assert residual < signal_mag * 0.5, \
        f"At max t, noisy should be close to noise, |xt-noise|/|noise| = {residual/signal_mag:.4f}"


def test_scheduler_predict_start():
    """验证 predict_start 能从噪声恢复原始信号"""
    scheduler = DiffusionScheduler(diffusion_steps=20)
    x0 = torch.randn(4, 12, 10, 1)
    t = torch.randint(0, 20, (4,))
    xt, noise = scheduler.add_noise(x0, t)
    predicted = scheduler.predict_start_from_noise(xt, t, noise)
    assert torch.allclose(predicted, x0, atol=1e-5), "predict_start should recover x0 exactly"


def test_scheduler_ddim_deterministic():
    """验证 DDIM eta=0 时采样可复现"""
    torch.manual_seed(42)
    scheduler = DiffusionScheduler(diffusion_steps=20)
    xT = torch.randn(4, 12, 10, 1)
    t = torch.tensor([3, 3, 3, 3], dtype=torch.long)  # shape must match batch
    pred_noise = torch.zeros_like(xT)

    out1 = scheduler.ddim_step(xT.clone(), t, pred_noise, eta=0.0)
    out2 = scheduler.ddim_step(xT.clone(), t, pred_noise, eta=0.0)
    assert torch.equal(out1, out2), "DDIM eta=0 should be deterministic"

    # With eta>0, should be stochastic (different each time due to noise)
    torch.manual_seed(42)
    out3 = scheduler.ddim_step(xT.clone(), t, pred_noise, eta=1.0)
    torch.manual_seed(43)
    out4 = scheduler.ddim_step(xT.clone(), t, pred_noise, eta=1.0)
    assert not torch.equal(out3, out4), "DDIM eta=1 should be stochastic with different seeds"


def test_scheduler_no_nan():
    """验证所有中间量无 NaN"""
    scheduler = DiffusionScheduler(diffusion_steps=200)
    for name in ["betas", "alphas", "alphas_cumprod", "posterior_variance"]:
        buf = getattr(scheduler, name)
        assert not buf.isnan().any(), f"{name} contains NaN"
        assert not buf.isinf().any(), f"{name} contains Inf"
    # betas should be monotonically increasing
    assert (scheduler.betas[1:] >= scheduler.betas[:-1]).all(), "betas not monotonic"


def test_scheduler_loss_baseline():
    """验证无信息预测器(ε̂=0)的 loss ≈ 1.0"""
    scheduler = DiffusionScheduler(diffusion_steps=200)
    x0 = torch.randn(100, 12, 10, 1)
    t = scheduler.sample_timesteps(100, device=torch.device("cpu"))
    xt, noise = scheduler.add_noise(x0, t)
    zero_pred = torch.zeros_like(noise)
    loss = F.mse_loss(zero_pred, noise)
    assert 0.8 < loss.item() < 1.2, f"Zero-prediction loss should be ~1.0, got {loss.item():.4f}"


# ══════════════════════════════════════════════════════════════════════
# TEST 2: STEncoder 编码器
# ══════════════════════════════════════════════════════════════════════

def test_encoder_shapes():
    """验证 STEncoder 输入输出形状"""
    cfg = make_config()
    data_feat = make_data_feature(num_nodes=10, feature_dim=3)
    encoder = STEncoder(
        input_dim=data_feat["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["encoder_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_time_steps=cfg["input_window"],
    )
    x = torch.randn(4, cfg["input_window"], 10, 3)
    adj = torch.eye(10)
    out = encoder(x, adj)
    assert out.shape == (4, cfg["input_window"], 10, cfg["hidden_dim"]), \
        f"Encoder output shape {out.shape}"
    assert not out.isnan().any(), "Encoder output contains NaN"
    assert not out.isinf().any(), "Encoder output contains Inf"


def test_encoder_gradient_flow():
    """验证 STEncoder 梯度能传播到所有参数"""
    cfg = make_config()
    data_feat = make_data_feature(num_nodes=10, feature_dim=3)
    encoder = STEncoder(
        input_dim=data_feat["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["encoder_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_time_steps=cfg["input_window"],
    )
    x = torch.randn(4, cfg["input_window"], 10, 3)
    adj = torch.eye(10)
    out = encoder(x, adj)
    loss = out.sum()
    loss.backward()

    no_grad_params = []
    for name, p in encoder.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)
        elif p.requires_grad and p.grad.abs().sum() == 0:
            no_grad_params.append(f"{name} (zero grad)")

    assert len(no_grad_params) == 0, \
        f"Parameters with no/missing gradient: {no_grad_params}"


def test_encoder_different_inputs():
    """验证不同输入产生不同编码"""
    cfg = make_config()
    data_feat = make_data_feature(num_nodes=10, feature_dim=3)
    encoder = STEncoder(
        input_dim=data_feat["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["encoder_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_time_steps=cfg["input_window"],
    )
    x1 = torch.randn(4, cfg["input_window"], 10, 3)
    x2 = torch.randn(4, cfg["input_window"], 10, 3)
    adj = torch.eye(10)
    out1 = encoder(x1, adj)
    out2 = encoder(x2, adj)
    assert not torch.allclose(out1, out2, atol=1e-3), \
        "Different inputs should produce different encodings"


# ══════════════════════════════════════════════════════════════════════
# TEST 3: AttentionDenoiser 噪声预测器
# ══════════════════════════════════════════════════════════════════════

def test_denoiser_shapes():
    """验证 AttentionDenoiser 输入输出形状"""
    cfg = make_config()
    denoiser = AttentionDenoiser(
        output_dim=1,
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        adaptive_graph_enabled=cfg["use_adaptive_graph"],
        adaptive_graph_embed_dim=cfg["adaptive_graph_embed_dim"],
        adaptive_graph_topk=cfg["adaptive_graph_topk"],
        adaptive_graph_blend_init=cfg["adaptive_graph_blend_init"],
        fuzzy_graph_enabled=cfg["use_fuzzy_graph"],
        fuzzy_graph_num_sets=cfg["fuzzy_graph_num_sets"],
        fuzzy_graph_sigma_init=cfg["fuzzy_graph_sigma_init"],
        num_nodes=10,
        static_adjacency=torch.eye(10),
    )
    noisy = torch.randn(4, cfg["output_window"], 10, 1)
    t = torch.randint(0, cfg["diffusion_steps"], (4,))
    cond = torch.randn(4, cfg["input_window"], 10, cfg["hidden_dim"])
    adj = torch.eye(10)
    out = denoiser(noisy, t, cond, adj)
    assert out.shape == (4, cfg["output_window"], 10, 1), \
        f"Denoiser output shape {out.shape}"
    assert not out.isnan().any(), "Denoiser output contains NaN"


def test_denoiser_condition_fusion():
    """验证 condition fusion 层正常工作（注入的条件信号影响输出）"""
    cfg = make_config()
    denoiser = AttentionDenoiser(
        output_dim=1,
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        adaptive_graph_enabled=False,
        fuzzy_graph_enabled=False,
        num_nodes=10,
        static_adjacency=torch.eye(10),
    )
    noisy = torch.randn(4, cfg["output_window"], 10, 1)
    t = torch.zeros(4, dtype=torch.long)  # t=0: minimal noise
    cond1 = torch.zeros(4, cfg["input_window"], 10, cfg["hidden_dim"])
    cond2 = torch.randn(4, cfg["input_window"], 10, cfg["hidden_dim"]) * 10
    adj = torch.eye(10)

    out1 = denoiser(noisy, t, cond1, adj)
    out2 = denoiser(noisy, t, cond2, adj)
    assert not torch.allclose(out1, out2, atol=1e-3), \
        "Different conditions should produce different noise predictions"


def test_denoiser_gradient_flow():
    """验证 AttentionDenoiser 梯度能到达所有参数"""
    cfg = make_config()
    denoiser = AttentionDenoiser(
        output_dim=1,
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        adaptive_graph_enabled=cfg["use_adaptive_graph"],
        adaptive_graph_embed_dim=cfg["adaptive_graph_embed_dim"],
        adaptive_graph_topk=cfg["adaptive_graph_topk"],
        adaptive_graph_blend_init=cfg["adaptive_graph_blend_init"],
        fuzzy_graph_enabled=cfg["use_fuzzy_graph"],
        fuzzy_graph_num_sets=cfg["fuzzy_graph_num_sets"],
        fuzzy_graph_sigma_init=cfg["fuzzy_graph_sigma_init"],
        num_nodes=10,
        static_adjacency=torch.eye(10),
    )
    noisy = torch.randn(4, cfg["output_window"], 10, 1)
    t = torch.randint(0, cfg["diffusion_steps"], (4,))
    cond = torch.randn(4, cfg["input_window"], 10, cfg["hidden_dim"])
    adj = torch.eye(10)
    out = denoiser(noisy, t, cond, adj)
    loss = out.sum()
    loss.backward()

    no_grad_params = []
    for name, p in denoiser.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)
        elif p.requires_grad and p.grad.abs().sum() == 0:
            no_grad_params.append(f"{name} (zero grad)")

    assert len(no_grad_params) == 0, \
        f"Parameters with no/missing gradient: {no_grad_params}"


def test_denoiser_large_t_uniform():
    """验证高噪声下（t=199）输出不坍缩为常数"""
    cfg = make_config(diffusion_steps=200)
    denoiser = AttentionDenoiser(
        output_dim=1,
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"],
        ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"],
        dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        adaptive_graph_enabled=False,
        fuzzy_graph_enabled=False,
        num_nodes=10,
        static_adjacency=torch.eye(10),
    )
    noisy1 = torch.randn(4, cfg["output_window"], 10, 1)
    noisy2 = torch.randn(4, cfg["output_window"], 10, 1)
    t = torch.full((4,), 199, dtype=torch.long)
    cond = torch.randn(4, cfg["input_window"], 10, cfg["hidden_dim"])
    adj = torch.eye(10)

    out1 = denoiser(noisy1, t, cond, adj)
    out2 = denoiser(noisy2, t, cond, adj)

    # 不同噪声输入应产生不同预测（否则条件失效）
    assert not torch.allclose(out1, out2, atol=1e-3), \
        "Different noisy inputs at high t should yield different predictions"


# ══════════════════════════════════════════════════════════════════════
# TEST 4: calculate_loss 训练损失
# ══════════════════════════════════════════════════════════════════════

def test_calculate_loss_shape():
    """验证 calculate_loss 返回标量"""
    cfg = make_config(physics_loss_weight=0.0)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    batch = make_synthetic_batch(
        batch_size=4, input_window=cfg["input_window"],
        output_window=cfg["output_window"], num_nodes=10, feature_dim=3
    )
    loss = model.calculate_loss(batch)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert not loss.isnan(), "Loss is NaN"
    assert not loss.isinf(), "Loss is Inf"


def test_calculate_loss_range():
    """验证初始 loss 在合理范围（~1.0 for noise prediction）"""
    cfg = make_config(physics_loss_weight=0.0)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()  # avoid train_step_count increment

    losses = []
    for _ in range(20):
        batch = make_synthetic_batch(
            batch_size=8, input_window=cfg["input_window"],
            output_window=cfg["output_window"], num_nodes=10, feature_dim=3
        )
        with torch.no_grad():
            loss = model.calculate_loss(batch)
        losses.append(loss.item())

    avg = sum(losses) / len(losses)
    assert 0.5 < avg < 3.0, (
        f"Initial loss should be roughly ~1.0 (zero-pred baseline), got avg={avg:.4f}. "
        f"Values >2 may indicate exploding output or incorrect noise scale.")


def test_calculate_loss_gradient():
    """验证 loss.backward() 后所有参数有梯度"""
    cfg = make_config(physics_loss_weight=0.0)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.train()
    batch = make_synthetic_batch(batch_size=4, input_window=cfg["input_window"],
                                  output_window=cfg["output_window"], num_nodes=10, feature_dim=3)
    loss = model.calculate_loss(batch)
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
            no_grad.append(name)
    assert len(no_grad) == 0, f"No gradient for: {no_grad}"


# ══════════════════════════════════════════════════════════════════════
# TEST 5: sample/predict 采样生成
# ══════════════════════════════════════════════════════════════════════

def test_predict_shape():
    """验证 predict 返回正确形状"""
    cfg = make_config()
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()
    batch = make_synthetic_batch(batch_size=4, input_window=cfg["input_window"],
                                  output_window=cfg["output_window"], num_nodes=10, feature_dim=3)
    with torch.no_grad():
        pred = model.predict(batch)
    expected = (4, cfg["output_window"], 10, 1)
    assert pred.shape == expected, f"predict shape {pred.shape} != {expected}"
    assert not pred.isnan().any(), "predict output contains NaN"


def test_sample_deterministic():
    """验证 DDIM eta=0 时相同输入（相同初始噪声 seed）产生相同输出"""
    cfg = make_config(sampling_method="ddim", ddim_eta=0.0)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()
    x = torch.randn(4, cfg["input_window"], 10, 3)

    torch.manual_seed(42)
    with torch.no_grad():
        pred1 = model.sample(x)
    torch.manual_seed(42)
    with torch.no_grad():
        pred2 = model.sample(x)
    assert torch.allclose(pred1, pred2, atol=1e-4), \
        "DDIM eta=0 with same seed should produce identical outputs"


def test_sample_multi_sample_shape():
    """验证多样本采样的形状"""
    cfg = make_config(num_prediction_samples=3)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()
    x = torch.randn(4, cfg["input_window"], 10, 3)

    with torch.no_grad():
        # return_all=True + num_samples=3 → (3, B, T, N, D)
        all_samples = model.sample(x, num_samples=3, return_all=True)
        assert all_samples.shape == (3, 4, cfg["output_window"], 10, 1), \
            f"All samples shape {all_samples.shape}"
        # default → (B, T, N, D) (mean over samples)
        mean_pred = model.sample(x, num_samples=1)
        assert mean_pred.shape == (4, cfg["output_window"], 10, 1)


def test_sample_output_variance():
    """验证随机采样eta>0时输出有方差"""
    cfg = make_config(sampling_method="ddim", ddim_eta=1.0)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()
    x = torch.randn(2, cfg["input_window"], 10, 3)

    with torch.no_grad():
        samples = torch.stack([model.sample(x) for _ in range(5)])
    std = samples.std(dim=0).mean()
    assert std > 0, "Stochastic sampling should have variance"


# ══════════════════════════════════════════════════════════════════════
# TEST 6: Scaler 归一化回环
# ══════════════════════════════════════════════════════════════════════

def test_standard_scaler_roundtrip():
    """标准归一化：transform → inverse_transform 应恢复原值"""
    import numpy as np
    data = np.random.randn(100, 12, 10, 2).astype(np.float32)
    mean = data.mean(axis=0, keepdims=False)
    std = data.std(axis=0, keepdims=False)
    scaler = StandardScaler(mean, std)
    transformed = scaler.transform(data)
    restored = scaler.inverse_transform(transformed)
    assert np.allclose(restored, data, atol=1e-5), \
        "StandardScaler round-trip failed"


def test_minmax_scaler_roundtrip():
    """MinMax归一化回环"""
    import numpy as np
    data = np.random.uniform(1, 100, (100, 12, 10, 2)).astype(np.float32)
    minn = data.min(axis=0, keepdims=False)
    maxx = data.max(axis=0, keepdims=False)
    scaler = MinMax01Scaler(minn, maxx)
    transformed = scaler.transform(data)
    restored = scaler.inverse_transform(transformed)
    assert np.allclose(restored, data, atol=1e-5), \
        f"MinMaxScaler round-trip failed, max diff: {np.abs(restored - data).max()}"


# ══════════════════════════════════════════════════════════════════════
# TEST 7: 端到端梯度流 + 小样本学习
# ══════════════════════════════════════════════════════════════════════

def test_end_to_end_gradient_flow():
    """端到端：batch → loss → backward → 所有参数有梯度"""
    cfg = make_config(diffusion_steps=20, physics_loss_weight=0.0)
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.train()

    batch = make_synthetic_batch(batch_size=4, input_window=cfg["input_window"],
                                  output_window=cfg["output_window"], num_nodes=10, feature_dim=3)
    loss = model.calculate_loss(batch)
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
            no_grad.append(name)
    assert len(no_grad) == 0, f"Parameters without gradient: {no_grad}"


def test_small_batch_overfit():
    """小批量过拟合测试：带规律数据的 loss 应显著下降

    使用确定性的 sin 模式作为 target，确保模型可以从条件中学到规律。
    如果连简单的周期性模式都无法拟合，说明架构有根本性问题。
    """
    cfg = make_config(
        diffusion_steps=20,
        num_sampling_steps=5,
        hidden_dim=32,
        encoder_layers=1,
        denoiser_layers=1,
        ffn_hidden_dim=64,
        physics_loss_weight=0.0,
        use_adaptive_graph=False,
        use_fuzzy_graph=False,
        use_spatiotemporal_attention=False,
    )
    data_feat = make_data_feature(num_nodes=8, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.train()

    # Build synthetic data with learnable patterns:
    # - X: noisy sine waves (model must denoise to recover the clean signal)
    # - y: clean sine waves
    B, T_in, T_out, N, F = 8, cfg["input_window"], cfg["output_window"], 8, 3
    t_in = torch.linspace(0, 4 * math.pi, T_in)
    t_out = torch.linspace(4 * math.pi, 8 * math.pi, T_out)

    X = torch.zeros(B, T_in, N, F)
    y = torch.zeros(B, T_out, N, 1)
    for b in range(B):
        phase = b * 0.3
        for n in range(N):
            freq = 1.0 + n * 0.2
            # Channel 0: the main traffic signal (deterministic)
            X[b, :, n, 0] = torch.sin(freq * t_in + phase)
            y[b, :, n, 0] = torch.sin(freq * t_out + phase)
            # Channels 1-2: random noise (distractors)
            X[b, :, n, 1] = torch.randn(T_in) * 0.1
            X[b, :, n, 2] = torch.randn(T_in) * 0.1

    batch = {"X": X, "y": y}

    opt = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Initial loss
    with torch.no_grad():
        initial_losses = [model.calculate_loss(batch).item() for _ in range(5)]
    init_loss = sum(initial_losses) / len(initial_losses)

    for epoch in range(50):
        opt.zero_grad()
        loss = model.calculate_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

    with torch.no_grad():
        final_loss = model.calculate_loss(batch).item()

    improvement = (init_loss - final_loss) / max(init_loss, 1e-8)
    print(f"       initial loss: {init_loss:.4f} → final loss: {final_loss:.4f} "
          f"(Δ: {improvement * 100:.1f}%)")

    # For diffusion noise-prediction on structured data:
    # - improvement > 20%: model clearly learns the pattern
    # - 10-20%: model is learning, acceptable for small model
    # - < 10%: fundamental training issue
    if improvement > 0.10:
        if improvement > 0.20:
            print(f"       ✅  Model learns structured patterns well")
        else:
            print(f"       ⚠️   Model shows learning but slowly (acceptable for diffusion)")
    else:
        raise AssertionError(
            f"Small batch overfit: loss should decrease >10% on structured data, "
            f"only {improvement * 100:.1f}%"
        )


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

TESTS = [
    # (name, [subtests])
    ("DiffusionScheduler 数学正确性", [
        test_scheduler_add_noise,
        test_scheduler_predict_start,
        test_scheduler_ddim_deterministic,
        test_scheduler_no_nan,
        test_scheduler_loss_baseline,
    ]),
    ("STEncoder 编码器", [
        test_encoder_shapes,
        test_encoder_gradient_flow,
        test_encoder_different_inputs,
    ]),
    ("AttentionDenoiser 噪声预测器", [
        test_denoiser_shapes,
        test_denoiser_condition_fusion,
        test_denoiser_gradient_flow,
        test_denoiser_large_t_uniform,
    ]),
    ("calculate_loss 训练损失", [
        test_calculate_loss_shape,
        test_calculate_loss_range,
        test_calculate_loss_gradient,
    ]),
    ("sample/predict 采样生成", [
        test_predict_shape,
        test_sample_deterministic,
        test_sample_multi_sample_shape,
        test_sample_output_variance,
    ]),
    ("Scaler 归一化回环", [
        test_standard_scaler_roundtrip,
        test_minmax_scaler_roundtrip,
    ]),
    ("端到端梯度流 + 小样本学习", [
        test_end_to_end_gradient_flow,
        test_small_batch_overfit,
    ]),
]


def main():
    parser = argparse.ArgumentParser(description="new_diffusion_fuzzy pipeline diagnostics")
    parser.add_argument("--idx", type=int, default=None,
                        help="Run only test group N (1-based)")
    parser.add_argument("--list", action="store_true",
                        help="List all test groups")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run tests on (cpu/cuda)")
    args = parser.parse_args()

    if args.list:
        for i, (name, subtests) in enumerate(TESTS, 1):
            print(f"  [{i}] {name}  ({len(subtests)} subtests)")
        return 0

    print("=" * 64)
    print("new_diffusion_fuzzy Pipeline Diagnostic Tests")
    print("=" * 64)
    print(f"Device: {args.device}")
    print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start = time.perf_counter()

    if args.idx is not None:
        idx = args.idx
        if idx < 1 or idx > len(TESTS):
            print(f"ERROR: --idx must be between 1 and {len(TESTS)}")
            return 1
        name, subtests = TESTS[idx - 1]
        print(f"── {idx}. {name} ──")
        for subtest in subtests:
            run_test(idx, subtest.__doc__ or subtest.__name__, subtest)
    else:
        for i, (name, subtests) in enumerate(TESTS, 1):
            print(f"── {i}. {name} ──")
            for subtest in subtests:
                run_test(i, subtest.__doc__ or subtest.__name__, subtest)
            print()

    elapsed = time.perf_counter() - start
    print("=" * 64)
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed  ({elapsed:.1f}s)")
    if total_fail == 0:
        print("  ✅  All tests passed — pipeline is healthy.")
    else:
        print(f"  ❌  {total_fail} test(s) FAILED — check the errors above.")
    print("=" * 64)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
