"""
扩散过程有效性诊断测试

核心问题：条件信号能否穿透扩散噪声，影响最终预测？
如果答案是否定的 → 输出呈正态分布（当前症状）

测试覆盖扩散过程的 6 个关键环节，全部用合成数据，无需训练。

用法:
  python scripts/tools/test_diffusion_fuzzy_process.py          # 全部
  python scripts/tools/test_diffusion_fuzzy_process.py --idx 3  # 只测第3项
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
import numpy as np

from GNNTP.models.new.new_diffusion_fuzzy.model import (
    DiffusionScheduler,
    STEncoder,
    AttentionDenoiser,
    NewDiffusion,
)

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
    "diffusion_steps": 100,
    "diffusion_schedule": "linear",
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "num_sampling_steps": 20,
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
    cfg = dict(_DEFAULT_CFG)
    cfg.update(overrides)
    return cfg


def make_data_feature(num_nodes=10, feature_dim=3, output_dim=1):
    import numpy as np
    return {
        "num_nodes": num_nodes,
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "adj_mx": np.eye(num_nodes, dtype=np.float32),
        "scaler": None,
    }


PASS = 0
FAIL = 1
total_pass = 0
total_fail = 0


def run_test(idx, name, fn):
    global total_pass, total_fail
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        # Use bool() to handle numpy.bool_ vs Python bool
        passed = bool(result) if result is not None else True
        if passed:
            total_pass += 1
            print(f"  [{idx}] ✅ PASS  {name}  ({elapsed:.3f}s)")
        else:
            total_fail += 1
            print(f"  [{idx}] ❌ FAIL  {name}  ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [{idx}] ❌ FAIL  {name}  ({elapsed:.3f}s)")
        print(f"       {type(e).__name__}: {e}")
        total_fail += 1


def header(text):
    print()
    print(f"  ═══ {text} ═══")


# ══════════════════════════════════════════════════════════════════════
# TEST 1: 条件注入信号强度
# ══════════════════════════════════════════════════════════════════════

def test_condition_fusion_magnitude():
    """验证 condition_fusion 层的条件信号在输出中占比

    核心问题：条件注入后，condition 信号是否被时间嵌入或位置嵌入淹没？
    测试方法：分别注入 0 和非 zero condition，比较 denoiser 内部表示的差异。
    如果 condition=0 和 condition=randn 的内部表示几乎一样 → 条件被淹没。
    """
    cfg = make_config(
        use_adaptive_graph=False, use_fuzzy_graph=False,
        use_spatiotemporal_attention=False,
    )
    denoiser = AttentionDenoiser(
        output_dim=1, hidden_dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"], ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"], dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=False,
        adaptive_graph_enabled=False, fuzzy_graph_enabled=False,
        num_nodes=10, static_adjacency=torch.eye(10),
    )
    denoiser.eval()

    noisy = torch.randn(2, cfg["output_window"], 10, 1)
    t = torch.tensor([50, 50], dtype=torch.long)  # mid-noise level

    # Zero condition
    cond_zero = torch.zeros(2, cfg["input_window"], 10, cfg["hidden_dim"])
    # Strong condition (scaled to match typical encoder output magnitude)
    cond_strong = torch.randn(2, cfg["input_window"], 10, cfg["hidden_dim"])

    adj = torch.eye(10)

    with torch.no_grad():
        out_zero = denoiser(noisy, t, cond_zero, adj)
        out_strong = denoiser(noisy, t, cond_strong, adj)

    # Compute relative difference
    diff_magnitude = (out_strong - out_zero).abs().mean()
    signal_magnitude = out_zero.abs().mean().clamp_min(1e-8)
    ratio = (diff_magnitude / signal_magnitude).item()

    header(f"Condition injection strength: Δ/|signal| = {ratio:.4f}")
    if ratio > 0.3:
        print(f"    ✅  Strong condition impact (ratio={ratio:.3f}) — condition signal propagates well")
    elif ratio > 0.1:
        print(f"    ⚠️   Moderate impact (ratio={ratio:.3f}) — condition signal is present but weak")
    else:
        print(f"    ❌  Weak impact (ratio={ratio:.3f}) — condition signal is nearly ignored!")
        print(f"        THIS IS THE LIKELY CAUSE OF ~N(μ,σ) OUTPUT.")
        print(f"        The denoiser sees noisy input but barely uses the condition.")
    return ratio > 0.1


# ══════════════════════════════════════════════════════════════════════
# TEST 2: 不同 t 下的噪声预测行为
# ══════════════════════════════════════════════════════════════════════

def test_noise_prediction_t_dependence():
    """验证噪声预测器对 t 的敏感性

    扩散去噪的本质：不同 t 需要不同策略。
    - t=0: yt≈y0 → 预测 ε≈0
    - t=T: yt≈ε → 预测 ε≈yt（即完全识别出噪声）

    测试方法：相同输入 + 相同条件，仅改变 t。
    测量不同 t 下预测噪声的 cosine similarity。
    如果所有 t 的预测几乎相同 → 时间嵌入未起作用。
    """
    cfg = make_config(
        use_adaptive_graph=False, use_fuzzy_graph=False,
        use_spatiotemporal_attention=False, diffusion_steps=100,
    )
    denoiser = AttentionDenoiser(
        output_dim=1, hidden_dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"], ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"], dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=False,
        adaptive_graph_enabled=False, fuzzy_graph_enabled=False,
        num_nodes=10, static_adjacency=torch.eye(10),
    )
    denoiser.eval()

    noisy = torch.randn(4, cfg["output_window"], 10, 1)
    cond = torch.randn(4, cfg["input_window"], 10, cfg["hidden_dim"])
    adj = torch.eye(10)

    t_values = [0, 10, 30, 60, 90, 99]
    predictions = []
    with torch.no_grad():
        for tv in t_values:
            t = torch.full((4,), tv, dtype=torch.long)
            pred = denoiser(noisy, t, cond, adj)
            predictions.append(pred)

    # Compute pairwise cosine similarity between all t pairs
    # If time embedding works, predictions at different t should differ
    # (lower cosine_sim = better t-separation)
    header("Noise prediction similarity across timesteps")
    print(f"    {'':>8s}", end="")
    for tv in t_values:
        print(f"  t={tv:3d} ", end="")
    print()
    for i, tv_i in enumerate(t_values):
        print(f"    t={tv_i:3d}  ", end="")
        for j, tv_j in enumerate(t_values):
            if i == j:
                print(f"   ───   ", end="")
            else:
                cos = F.cosine_similarity(
                    predictions[i].flatten(1), predictions[j].flatten(1)
                ).mean().item()
                # color code: >0.95 = red flag (identical), <0.8 = green (different)
                marker = "🔴" if cos > 0.95 else ("🟡" if cos > 0.85 else "🟢")
                print(f"{marker}{cos:.3f} ", end="")
        print()

    # Average off-diagonal cosine similarity
    off_diag_cos = []
    for i in range(len(t_values)):
        for j in range(len(t_values)):
            if i != j:
                off_diag_cos.append(
                    F.cosine_similarity(
                        predictions[i].flatten(1), predictions[j].flatten(1)
                    ).mean().item()
                )
    avg_cos = np.mean(off_diag_cos)

    print(f"    Mean cross-t cosine similarity: {avg_cos:.4f}")
    if avg_cos < 0.85:
        print(f"    ✅  Time embedding creates distinct predictions per t")
    elif avg_cos < 0.95:
        print(f"    ⚠️   Weak but present t-dependence")
    else:
        print(f"    ❌  All t → same prediction (cos_sim={avg_cos:.3f})!")
        print(f"        Time embedding signal is drowned out by other components.")
        print(f"        Without t-awareness, denoiser can't learn the noise schedule.")
        print(f"")
        print(f"        ROOT CAUSE: timestep_features added as broadcast [B,1,1,D]")
        print(f"        but input_projection + position_embedding dominate the signal.")
        print(f"        Fix: scale up time_projection output or add t at multiple layers.")
    return avg_cos < 0.95


# ══════════════════════════════════════════════════════════════════════
# TEST 3: 反向扩散过程轨迹检查
# ══════════════════════════════════════════════════════════════════════

def test_reverse_process_trajectory():
    """验证反向扩散过程是否产生结构化输出

    正常 DDIM 反向过程：从纯噪声 xT 开始，每个 step 逐渐去除噪声，
    最终输出应具有与条件相关的结构（而非仍然随机）。

    测试方法：运行完整反向链，检查输出分布的统计特征。
    - 如果输出是正态分布且与条件无关 → 扩散过程失效
    - 如果输出有明确结构（峰度 ≠ 3，偏度 ≠ 0）→ 条件生效
    """
    cfg = make_config(
        diffusion_steps=100, num_sampling_steps=10,
        use_adaptive_graph=False, use_fuzzy_graph=False,
        use_spatiotemporal_attention=False,
    )
    data_feat = make_data_feature(num_nodes=10, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()

    # Use structured condition (sin wave) instead of random
    B, T_in, N, F = 4, cfg["input_window"], 10, 3
    t_vals = torch.linspace(0, 4 * math.pi, T_in)
    X = torch.zeros(B, T_in, N, F)
    for b in range(B):
        phase = b * 0.5
        for n in range(N):
            freq = 1.0 + n * 0.1
            X[b, :, n, 0] = torch.sin(freq * t_vals + phase)
            X[b, :, n, 1] = torch.cos(freq * t_vals + phase) * 0.5
            X[b, :, n, 2] = torch.randn(T_in) * 0.05

    torch.manual_seed(42)
    with torch.no_grad():
        pred = model.sample(X, num_samples=1)

    # Check output distribution statistics
    flat = pred.flatten().numpy()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    skewness = float(np.mean(((flat - mean_val) / max(std_val, 1e-8)) ** 3))
    kurtosis = float(np.mean(((flat - mean_val) / max(std_val, 1e-8)) ** 4))

    header(f"Reverse process output distribution (untrained model)")
    print(f"    Mean:    {mean_val:+.4f}")
    print(f"    Std:     {std_val:.4f}")
    print(f"    Skewness: {skewness:+.4f}  (0 = symmetric)")
    print(f"    Kurtosis: {kurtosis:.4f}  (3 = normal, <3 = platykurtic, >3 = leptokurtic)")

    # NOTE: Untrained model won't match the target distribution!
    # This test establishes a BASELINE. After training, kurtosis should change.
    # Diagnostically important: if a trained model also has kurtosis≈3, it's
    # producing Gaussian noise instead of traffic patterns.

    print(f"    ⚠️   Untrained model — this is a BASELINE measurement.")
    print(f"         After training: kurtosis should deviate from 3 (Gaussian).")
    print(f"         If trained kurtosis ≈ 3 → model predicts noise, not traffic.")

    # Even untrained, we can check: different conditions → different outputs?
    X2 = torch.randn(B, T_in, N, F)  # random condition
    torch.manual_seed(42)
    with torch.no_grad():
        pred2 = model.sample(X2, num_samples=1)

    cond_sensitivity = (pred - pred2).abs().mean().item()
    output_magnitude = pred.abs().mean().clamp_min(1e-8).item()
    cond_ratio = cond_sensitivity / output_magnitude

    print(f"    Condition sensitivity: Δ|pred|/|pred| = {cond_ratio:.4f}")
    if cond_ratio > 0.1:
        print(f"    ✅  Different conditions → different outputs (conditioning works)")
    elif cond_ratio > 0.02:
        print(f"    ⚠️   Weak conditioning — output barely changes with condition")
    else:
        print(f"    ❌  No conditioning — output is independent of history!")
    return cond_ratio > 0.02


# ══════════════════════════════════════════════════════════════════════
# TEST 4: 单步去噪恢复率 vs t
# ══════════════════════════════════════════════════════════════════════

def test_single_step_recovery():
    """验证单步去噪：已知真实噪声，模型能否恢复信号？

    给定: y0 (真值), ε (真实噪声), yt = sqrt(ᾱ)·y0 + sqrt(1-ᾱ)·ε
    模型预测 ε̂ = noise_predictor(yt, t, condition)
    然后恢复 ŷ0 = (yt - sqrt(1-ᾱ)·ε̂) / sqrt(ᾱ)

    比较 ŷ0 与 y0 的相关性：
    - 如果模型学到条件→噪声的映射：低 t 的恢复应该比高 t 好
    - 如果模型完全忽略条件：所有 t 恢复质量差不多（都差）

    即使未训练，这个 test 也可以检测 trivial 策略：
    - 如果 ε̂≈0 → 恢复 = yt/sqrt(ᾱ)，低 t 好高 t 差（trivial 假阳性）
    - 我们需要的是：condition 变化 → 恢复变化（证明条件被使用）
    """
    cfg = make_config(
        diffusion_steps=100,
        use_adaptive_graph=False, use_fuzzy_graph=False,
        use_spatiotemporal_attention=False,
    )
    denoiser = AttentionDenoiser(
        output_dim=1, hidden_dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"], ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"], dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=False,
        adaptive_graph_enabled=False, fuzzy_graph_enabled=False,
        num_nodes=10, static_adjacency=torch.eye(10),
    )
    denoiser.eval()
    scheduler = DiffusionScheduler(diffusion_steps=100)

    # Structured target (sin wave)
    B, T, N = 4, cfg["output_window"], 10
    t_out = torch.linspace(0, 4 * math.pi, T)
    y0 = torch.zeros(B, T, N, 1)
    for b in range(B):
        for n in range(N):
            y0[b, :, n, 0] = torch.sin((1.0 + n * 0.1) * t_out + b * 0.5)

    # Two different conditions: zero vs strong (like Test 1 — maximizes contrast)
    cond_zero = torch.zeros(B, cfg["input_window"], N, cfg["hidden_dim"])
    cond_strong = torch.randn(B, cfg["input_window"], N, cfg["hidden_dim"])
    adj = torch.eye(N)

    header("Single-step recovery: zero-condition vs strong-condition")
    header("  Does the condition difference survive the FULL denoiser pipeline?")
    print(f"    {'t':>6s}  {'cos(zero)':>10s}  {'cos(strong)':>10s}  {'Δ(zero-strong)':>16s}")

    t_list = [0, 5, 10, 20, 40, 70, 99]
    all_cos_z, all_cos_s = [], []

    for tv in t_list:
        t = torch.full((B,), tv, dtype=torch.long)

        # Same noise for both conditions (fair comparison)
        torch.manual_seed(tv)
        noise = torch.randn_like(y0)
        yt, _ = scheduler.add_noise(y0, t, noise=noise)
        alpha_bar_t = scheduler.alphas_cumprod[tv].item()

        with torch.no_grad():
            eps_z = denoiser(yt, t, cond_zero, adj)
            eps_s = denoiser(yt, t, cond_strong, adj)

            sqrt_alpha = math.sqrt(alpha_bar_t)
            sqrt_one_alpha = math.sqrt(1 - alpha_bar_t) if alpha_bar_t < 1 else 0
            y0_z = (yt - sqrt_one_alpha * eps_z) / max(sqrt_alpha, 1e-8)
            y0_s = (yt - sqrt_one_alpha * eps_s) / max(sqrt_alpha, 1e-8)

        cos_z = F.cosine_similarity(y0_z.flatten(1), y0.flatten(1)).mean().item()
        cos_s = F.cosine_similarity(y0_s.flatten(1), y0.flatten(1)).mean().item()
        all_cos_z.append(cos_z)
        all_cos_s.append(cos_s)

        diff_marker = "⚠️ " if abs(cos_z - cos_s) < 0.01 else "   "
        print(f"    t={tv:3d}    {cos_z:+.4f}        {cos_s:+.4f}        {diff_marker}{abs(cos_z - cos_s):.4f}")

    avg_cos_diff = np.mean([abs(z - s) for z, s in zip(all_cos_z, all_cos_s)])

    # Also compare noise predictions directly
    t_last = torch.full((B,), t_list[-1], dtype=torch.long)
    torch.manual_seed(42)
    noise_last = torch.randn_like(y0)
    yt_last, _ = scheduler.add_noise(y0, t_last, noise=noise_last)
    with torch.no_grad():
        eps_z_last = denoiser(yt_last, t_last, cond_zero, adj)
        eps_s_last = denoiser(yt_last, t_last, cond_strong, adj)
    eps_diff = (eps_z_last - eps_s_last).abs().mean().item()
    eps_mag = max(eps_z_last.abs().mean().item(), 1e-8)
    eps_ratio = eps_diff / eps_mag

    print(f"    Mean Δ(zero-strong) across t: {avg_cos_diff:.4f}")
    print(f"    Direct ε̂ difference at t={t_list[-1]}: |ε̂_z-ε̂_s|/|ε̂| = {eps_ratio:.4f}")

    if eps_ratio > 0.3:
        print(f"    ✅  Condition signal survives full pipeline (ratio={eps_ratio:.3f})")
    elif eps_ratio > 0.08:
        print(f"    ⚠️   Condition effect PRESENT but WEAK (ratio={eps_ratio:.3f})")
        print(f"         condition_fusion works → but signal diluted by attention/LayerNorm")
    else:
        print(f"    ❌  Condition signal LOST in denoiser pipeline!")
        print(f"         Test 1 proves condition_fusion injects signal.")
        print(f"         But after {cfg['denoiser_layers']} denoiser blocks → signal VANISHES.")
        print(f"         Root cause: LayerNorm + residual connections dilute condition.")
    return eps_ratio > 0.08


# ══════════════════════════════════════════════════════════════════════
# TEST 5: 预测噪声的零预测倾向检测
# ══════════════════════════════════════════════════════════════════════

def test_predicted_noise_magnitude():
    """检测噪声预测器是否退化到 ≈0 预测

    如果模型预测 ε̂ ≈ 0：
    - 训练 loss ≈ E[ε²] = 1.0（匹配你的症状！）
    - 反向扩散退化为 Y_{t-1} ≈ Y_t / sqrt(α)（纯缩放，无去噪）
    - 输出保持正态分布（也是你的症状！）

    这是扩散模型最常见的失败模式。
    """
    cfg = make_config(
        diffusion_steps=100,
        use_adaptive_graph=False, use_fuzzy_graph=False,
        use_spatiotemporal_attention=False,
    )
    denoiser = AttentionDenoiser(
        output_dim=1, hidden_dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
        num_layers=cfg["denoiser_layers"], ffn_hidden_dim=cfg["ffn_hidden_dim"],
        graph_k_hop=cfg["graph_k_hop"], dropout=cfg["dropout"],
        use_spatiotemporal_attention=cfg["use_spatiotemporal_attention"],
        use_temporal_position_embedding=cfg["use_temporal_position_embedding"],
        max_future_steps=cfg["output_window"],
        use_gradient_checkpointing=False,
        adaptive_graph_enabled=False, fuzzy_graph_enabled=False,
        num_nodes=10, static_adjacency=torch.eye(10),
    )
    denoiser.eval()

    scheduler = DiffusionScheduler(diffusion_steps=100)

    # Use structured data (not random noise) as target
    B, T, N = 8, cfg["output_window"], 10
    y0 = torch.randn(B, T, N, 1) * 2 + 5  # mean=5, std=2 (traffic-like)

    cond = torch.randn(B, cfg["input_window"], N, cfg["hidden_dim"])
    adj = torch.eye(N)

    header("Predicted noise magnitude check")
    avg_norms = []
    for tv in [0, 5, 10, 30, 60, 99]:
        t = torch.full((B,), tv, dtype=torch.long)
        yt, true_noise = scheduler.add_noise(y0, t)
        with torch.no_grad():
            pred_noise = denoiser(yt, t, cond, adj)

        true_norm = true_noise.norm(dim=-1).mean().item()
        pred_norm = pred_noise.norm(dim=-1).mean().item()
        ratio = pred_norm / max(true_norm, 1e-8)
        avg_norms.append(pred_norm)

        print(f"    t={tv:3d}: |ε̂|={pred_norm:.4f}, |ε|={true_norm:.4f}, |ε̂|/|ε|={ratio:.3f}")

    mean_pred_norm = np.mean(avg_norms)
    print(f"    Mean |ε̂| = {mean_pred_norm:.4f}")

    if mean_pred_norm < 0.05:
        print(f"    ❌  CRITICAL: Predicted noise ≈ 0!")
        print(f"         Model has collapsed to predicting zero noise.")
        print(f"         Loss ≈ E[ε²] = 1.0 (matches your symptom)")
        print(f"         Output ≈ pure noise (matches your symptom)")
        print(f"         Fix: check init scale, learning rate, gradient flow")
        return False
    elif mean_pred_norm < 0.3:
        print(f"    ⚠️   Predicted noise is very small (|ε̂|={mean_pred_norm:.3f})")
        print(f"         May partially explain loss ≈ 1.0")
    else:
        print(f"    ✅  Predicted noise has reasonable magnitude")
    return mean_pred_norm > 0.1


# ══════════════════════════════════════════════════════════════════════
# TEST 6: 完整扩散管线：从条件到预测的结构传递
# ══════════════════════════════════════════════════════════════════════

def test_full_pipeline_structure():
    """端到端测试：结构化输入 → 预测是否保持结构？

    给模型 sin 波历史，预测未来。
    即使未训练，不同频率/相位的输入应产生不同的预测。
    如果所有输入的预测都一样 → 条件被完全忽略。
    """
    cfg = make_config(
        diffusion_steps=100, num_sampling_steps=10,
        use_adaptive_graph=False, use_fuzzy_graph=False,
        use_spatiotemporal_attention=False,
    )
    data_feat = make_data_feature(num_nodes=8, feature_dim=3, output_dim=1)
    model = NewDiffusion(cfg, data_feat)
    model.eval()

    B, T_in, T_out, N = 4, cfg["input_window"], cfg["output_window"], 8
    t_in = torch.linspace(0, 4 * math.pi, T_in)

    # Input A: phase=0
    X_a = torch.zeros(B, T_in, N, 3)
    for n in range(N):
        freq = 1.0 + n * 0.2
        X_a[:, :, n, 0] = torch.sin(freq * t_in)
        X_a[:, :, n, 1] = torch.cos(freq * t_in) * 0.5

    # Input B: phase=π/2 (shifted)
    X_b = torch.zeros(B, T_in, N, 3)
    for n in range(N):
        freq = 1.0 + n * 0.2
        X_b[:, :, n, 0] = torch.sin(freq * t_in + math.pi / 2)
        X_b[:, :, n, 1] = torch.cos(freq * t_in + math.pi / 2) * 0.5

    # Input C: random noise
    X_c = torch.randn(B, T_in, N, 3) * 2

    torch.manual_seed(42)
    with torch.no_grad():
        pred_a = model.sample(X_a, num_samples=1)
    torch.manual_seed(42)
    with torch.no_grad():
        pred_b = model.sample(X_b, num_samples=1)
    torch.manual_seed(42)
    with torch.no_grad():
        pred_c = model.sample(X_c, num_samples=1)

    # Pairwise differences
    diff_ab = (pred_a - pred_b).abs().mean().item()
    diff_ac = (pred_a - pred_c).abs().mean().item()
    diff_bc = (pred_b - pred_c).abs().mean().item()
    output_scale = pred_a.abs().mean().clamp_min(1e-8).item()

    header("Full pipeline: structure preservation")
    print(f"    Output scale: {output_scale:.4f}")
    print(f"    Δ(sin_phase0, sin_phaseπ/2) / scale = {diff_ab / output_scale:.4f}")
    print(f"    Δ(sin_phase0, random_noise) / scale = {diff_ac / output_scale:.4f}")
    print(f"    Δ(sin_phaseπ/2, random_noise) / scale = {diff_bc / output_scale:.4f}")

    all_diffs = [diff_ab, diff_ac, diff_bc]
    max_diff = max(all_diffs) / output_scale

    if max_diff > 0.1:
        print(f"    ✅  Different inputs → meaningfully different outputs")
        print(f"         (Untrained: outputs are random but VARIED — condition signal flows)")
    elif max_diff > 0.02:
        print(f"    ⚠️   Weak differentiation — condition has marginal effect")
    else:
        print(f"    ❌  All inputs → same output distribution!")
        print(f"         Condition signal is COMPLETELY ignored by the model.")
        print(f"         This is the root cause of ~N(μ,σ) predictions.")

    return max_diff > 0.02


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

TESTS = [
    ("条件注入信号强度", test_condition_fusion_magnitude),
    ("t 依赖的噪声预测", test_noise_prediction_t_dependence),
    ("反向扩散轨迹", test_reverse_process_trajectory),
    ("单步去噪恢复率", test_single_step_recovery),
    ("零预测倾向检测", test_predicted_noise_magnitude),
    ("完整管线结构传递", test_full_pipeline_structure),
]


def main():
    parser = argparse.ArgumentParser(description="Diffusion process effectiveness tests")
    parser.add_argument("--idx", type=int, default=None, help="Run only test N (1-based)")
    parser.add_argument("--list", action="store_true", help="List all tests")
    args = parser.parse_args()

    if args.list:
        for i, (name, _) in enumerate(TESTS, 1):
            print(f"  [{i}] {name}")
        return 0

    print("=" * 64)
    print("Diffusion Process Effectiveness Tests")
    print("=" * 64)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  Testing: Does condition signal survive the diffusion noise?")
    print("  If NOT → output is ~N(μ,σ) → matches your symptom.")
    print()

    start = time.perf_counter()

    if args.idx is not None:
        idx = args.idx
        if idx < 1 or idx > len(TESTS):
            print(f"ERROR: --idx must be between 1 and {len(TESTS)}")
            return 1
        name, fn = TESTS[idx - 1]
        print(f"── {idx}. {name} ──")
        run_test(idx, name, fn)
    else:
        for i, (name, fn) in enumerate(TESTS, 1):
            print(f"── {i}. {name} ──")
            run_test(i, name, fn)

    elapsed = time.perf_counter() - start
    print()
    print("=" * 64)
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed  ({elapsed:.1f}s)")
    if total_fail == 0:
        print(f"  ✅  All signal pathways are intact.")
        print(f"       If predictions are still ~N(μ,σ), the problem is in training")
        print(f"       dynamics (learning rate, optimizer, batch size, etc.).")
    else:
        print(f"  ❌  {total_fail} signal pathway(s) BROKEN.")
        print(f"       These are ARCHITECTURAL issues, not training issues.")
        print(f"       Fix the broken pathway before attempting training.")
    print("=" * 64)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
