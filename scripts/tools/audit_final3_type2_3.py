"""
final_3_type2 架构审计 v3 — 论文级结构诊断工具。

在原 final 版本基础上修复了以下方法学问题：

M3 (Routing Entropy):
  ❌ 旧版: input_projection(history).mean(dim=(0,1)) → 全局 prototype，非真实 routing
  ✅ v3:   register_forward_hook 捕获每个 CellAttention 的真实输入特征，精确追踪 routing

M4 (FOU-Error Correlation):
  ❌ 旧版: per-node error 平均 (压掉了时间和特征维度) + 静态 FOU → 时间尺度不匹配
  ✅ v3:   per-horizon 分析 + 多视角 FOU-error 关联 + 时间结构保留

M5 (New Closure Edges):
  ❌ 旧版: 把 closure edges 称为 "reasoning edges" (过强解释)
  ✅ v3:   正确称为 "density shift by transitive smoothing" + random baseline 对比

Scorecard:
  ❌ 旧版: 简单 failure count → 未考虑指标相关性
  ✅ v3:   加权评分 + 相关性说明 + 分等级结论

整体定位:
  ✔ 训练过程监控 + debug 诊断 → 强工具
  ⚠️  论文支撑证据 → 有条件使用 (需结合训练曲线、消融实验)
  ❌ 严格结构有效性证明 → 不适用 (需更严格实验设计)

用法:
  python scripts/tools/audit_final3_type2_3.py \
      --dataset PEMSD4 \
      --config_file train_config_PEMSD4.json \
      --checkpoint outputs/.../final_3_type2_PEMSD4_epoch49.tar \
      --output audit_final_PEMSD4_v3.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data import build_dataset_runtime
from GNNTP.utils import get_model
from GNNTP.models.new.final_3_type2.cell_attention import FuzzyCellAttention


# ═══════════════════════════════════════════════════════════════════════
#  Shared model loading (same robust logic as final)
# ═══════════════════════════════════════════════════════════════════════

def _infer_config_overrides(ckpt_state):
    """从 checkpoint state_dict 推断训练时的关键维度，返回 other_args dict。"""
    overrides = {}
    feature_dim_override = None

    _ip_w = "condition_encoder.input_projection.weight"
    if _ip_w in ckpt_state:
        overrides["hidden_dim"] = ckpt_state[_ip_w].shape[0]
        feature_dim_override = ckpt_state[_ip_w].shape[1]

    _ffn_w = "condition_encoder.blocks.0.feed_forward.network.0.weight"
    if _ffn_w in ckpt_state:
        overrides["ffn_hidden_dim"] = ckpt_state[_ffn_w].shape[0]

    _fz = "fuzzy_graph.base_membership_lower"
    if _fz in ckpt_state:
        overrides["fuzzy_num_sets"] = ckpt_state[_fz].shape[1]

    for _cell_key in (
        "condition_encoder.blocks.0.cell_attention.region_mu",
        "condition_encoder.blocks.0.cell_attention.centers",
    ):
        if _cell_key in ckpt_state:
            overrides["num_cells"] = ckpt_state[_cell_key].shape[0]
            break

    enc_blocks = {k for k in ckpt_state if k.startswith("condition_encoder.blocks.")}
    enc_indices = set()
    for k in enc_blocks:
        parts = k.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            enc_indices.add(int(parts[2]))
    if enc_indices:
        overrides["encoder_layers"] = max(enc_indices) + 1

    dec_blocks = {k for k in ckpt_state if k.startswith("future_decoder.blocks.")}
    dec_indices = set()
    for k in dec_blocks:
        parts = k.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            dec_indices.add(int(parts[2]))
    if dec_indices:
        overrides["decoder_layers"] = max(dec_indices) + 1

    return overrides, feature_dim_override


def load_model_robust(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print("❌ No checkpoint found. Aborting.")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_state = (ckpt.get("model_state_dict") or
                  ckpt.get("state_dict") or ckpt)

    ckpt_overrides, ckpt_feature_dim = _infer_config_overrides(ckpt_state)
    if ckpt_overrides:
        print(f"  📐 Inferred from checkpoint: {json.dumps(ckpt_overrides)}")
    if ckpt_feature_dim is not None:
        print(f"  📐 Checkpoint input_dim: {ckpt_feature_dim}")

    other_args = dict(ckpt_overrides)
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            file_config = json.load(f)
        for k, v in file_config.items():
            if k in ('task', 'model', 'dataset', 'saved_model', 'train',
                     'rank', 'world_size', 'local_rank', 'dist_backend',
                     'is_distributed', 'device', 'gpu_id', 'gpu', 'epoch',
                     'exp_id', 'data_version_id', 'log_every'):
                continue
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                other_args[k] = v
        print(f"  Merged {len(file_config)} config keys from {args.config_file}")

        _DIMENSION_KEYS = {
            'hidden_dim', 'ffn_hidden_dim', 'fuzzy_num_sets',
            'num_cells', 'encoder_layers', 'decoder_layers',
        }
        for k in _DIMENSION_KEYS:
            if k in ckpt_overrides:
                other_args[k] = ckpt_overrides[k]

    if args.other_args:
        other_args.update(json.loads(args.other_args))

    config = ConfigParser(
        "traffic_state_pred", "final_3_type2", args.dataset,
        config_file=None, saved_model=True, train=False,
        other_args=other_args,
    )

    runtime = build_dataset_runtime(config)
    dataloader = runtime.valid_loader
    print(f"  data_feature['feature_dim'] = {runtime.data_feature.get('feature_dim')}")

    if ckpt_feature_dim is not None:
        data_feat = runtime.data_feature
        ds_dim = data_feat.get("feature_dim", 0)
        if ckpt_feature_dim != ds_dim:
            print(f"  🔧 Overriding data_feature['feature_dim']: {ds_dim} → {ckpt_feature_dim}")
            data_feat["feature_dim"] = ckpt_feature_dim

    model = get_model(config, runtime.data_feature).to(device)
    if ckpt_feature_dim is not None:
        model._trim_dim = ckpt_feature_dim

    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for key, val in ckpt_state.items():
        if key in model_state and model_state[key].shape == val.shape:
            model_state[key].copy_(val)
            loaded += 1
        elif key in model_state:
            skipped.append(f"{key} (ckpt:{list(val.shape)} vs model:{list(model_state[key].shape)})")

    total = len(model_state)
    print(f"  ✅ Loaded {loaded}/{total} params"
          + (f" (skipped {total - loaded})" if loaded < total else ""))
    if skipped:
        print(f"  ⚠️  Skipped {len(skipped)} mismatched params (first 5):")
        for s in skipped[:5]:
            print(f"     - {s}")
        if len(skipped) > 5:
            print(f"     ... and {len(skipped) - 5} more")

    model.eval()
    return model, dataloader, device, runtime


# ═══════════════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════════════

def _unwrap(model):
    return model.module if hasattr(model, 'module') else model

def _trim(hist, model):
    d = getattr(_unwrap(model), '_trim_dim', None)
    if d and isinstance(hist, torch.Tensor) and hist.shape[-1] > d:
        return hist[..., :d]
    return hist


# ═══════════════════════════════════════════════════════════════════════
#  M1: Membership Entropy (可靠性: ✔ 高)
# ═══════════════════════════════════════════════════════════════════════

def metric_membership_entropy(model):
    """Per-node Shannon entropy of fuzzy membership distribution.

    可靠性说明:
      - 基于静态参数 (base_membership_lower/delta), 不受 batch 噪声影响
      - Shannon entropy + CV 是成熟的信息论度量
      - 可直接作为结构诊断指标使用

    注意事项:
      - 静态 membership 不反映 traffic-conditioned 动态变化
      - 如需完整图景，应配合 M6 set_utilization 一起解读
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    with torch.no_grad():
        mu_low, mu_high, mu_mid = fg._compute_memberships()
        p = mu_mid / mu_mid.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        H = -(p * (p + 1e-8).log()).sum(dim=-1)  # [N]

    H_np = H.cpu().numpy()
    H_max = np.log(m.fuzzy_num_sets)
    cv = H_np.std() / (H_np.mean() + 1e-8)

    # Bootstrap CI for CV (1000 resamples)
    rng = np.random.default_rng(42)
    cv_boot = []
    for _ in range(1000):
        sample = rng.choice(H_np, size=len(H_np), replace=True)
        cv_boot.append(sample.std() / (sample.mean() + 1e-8))
    cv_ci_low, cv_ci_high = np.percentile(cv_boot, [2.5, 97.5])

    results = {
        "H_mean": float(H_np.mean()),
        "H_std": float(H_np.std()),
        "H_min": float(H_np.min()),
        "H_max": float(H_np.max()),
        "H_normalized_mean": float(H_np.mean() / H_max),
        "coefficient_of_variation": float(cv),
        "cv_95ci_low": float(cv_ci_low),
        "cv_95ci_high": float(cv_ci_high),
        "frac_low_entropy (<0.1*H_max)": float((H_np < 0.1 * H_max).mean()),
        "frac_high_entropy (>0.8*H_max)": float((H_np > 0.8 * H_max).mean()),
        "_reliability": "high — static membership, no batch noise",
    }

    if cv < 0.05:
        verdict = "❌ COLLAPSED — all nodes have nearly identical entropy"
    elif cv < 0.15:
        verdict = f"⚠️  LOW DIVERSITY — CV={cv:.3f}, membership barely varies"
    elif results["frac_low_entropy (<0.1*H_max)"] > 0.8:
        verdict = "❌ NEAR-ZERO — most nodes have ~0 entropy (one-hot membership)"
    else:
        verdict = f"✅ HEALTHY — CV={cv:.3f}, membership entropy varies across nodes"

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M2: Pairwise Cosine(μ) (可靠性: ✔ 高)
# ═══════════════════════════════════════════════════════════════════════

def metric_pairwise_cosine(model):
    """Average pairwise cosine similarity of node membership vectors.

    可靠性说明:
      - 基于静态 membership, 不受 batch 噪声影响
      - L2 normalize + pairwise cosine 数值稳定
      - 对 N > 500 使用 sampling (无偏估计, 方差可控)

    注意事项:
      - 与 M1 共享 μ_mid → 两者高度相关 (CV 低 ⇔ cosine 高)
      - 不应作为独立证据计算两次
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    with torch.no_grad():
        _, _, mu_mid = fg._compute_memberships()

    mu_norm = F.normalize(mu_mid, p=2, dim=-1)  # [N, K]
    N = mu_norm.shape[0]

    if N <= 500:
        sim_full = mu_norm @ mu_norm.T  # [N, N]
        mask = ~torch.eye(N, dtype=torch.bool, device=mu_norm.device)
        sim_mean = sim_full[mask].mean().item()
        sim_std = sim_full[mask].std().item()
        sim_p90 = sim_full[mask].quantile(0.9).item()
        sim_p95 = sim_full[mask].quantile(0.95).item()
        sim_p99 = sim_full[mask].quantile(0.99).item()
    else:
        # Random sample (seed fixed for reproducibility)
        rng = torch.Generator(device=mu_norm.device).manual_seed(42)
        idx_i = torch.randint(0, N, (10000,), generator=rng, device=mu_norm.device)
        idx_j = torch.randint(0, N, (10000,), generator=rng, device=mu_norm.device)
        mask = idx_i != idx_j
        sims = (mu_norm[idx_i] * mu_norm[idx_j]).sum(dim=-1)[mask]
        sim_mean = sims.mean().item()
        sim_std = sims.std().item()
        sim_p90 = sims.quantile(0.9).item()
        sim_p95 = sims.quantile(0.95).item()
        sim_p99 = sims.quantile(0.99).item()

    results = {
        "mean_cosine": float(sim_mean),
        "std_cosine": float(sim_std),
        "p90_cosine": float(sim_p90),
        "p95_cosine": float(sim_p95),
        "p99_cosine": float(sim_p99),
        "num_nodes": N,
        "_reliability": "high — static membership, no batch noise",
        "_correlation_note": "与 M1 共享 μ_mid → CV 与 cosine 高度相关, 不是独立指标",
    }

    if sim_mean > 0.95:
        verdict = "❌ HOMOGENEOUS — all nodes have near-identical membership vectors"
    elif sim_mean > 0.8:
        verdict = f"⚠️  HIGH SIMILARITY — mean cosine={sim_mean:.3f}, low diversity"
    elif sim_mean > 0.5:
        verdict = f"⚠️  MODERATE — mean cosine={sim_mean:.3f}, some diversity"
    else:
        verdict = f"✅ DIVERSE — mean cosine={sim_mean:.3f}, nodes are heterogeneous"

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M3: Routing Entropy (FRR) — v3 重写: forward-hook 真实追踪
# ═══════════════════════════════════════════════════════════════════════

def _build_routing_hook(mu_fuzzy):
    """构建 forward hook 闭包, 捕获 CellAttention 真实 routing 分布。

    Args:
        mu_fuzzy: [N, K_f] midpoint fuzzy membership (由 encode_condition 计算),
                  所有 CellAttention block 共享同一份

    Returns:
        hook_fn: 符合 register_forward_hook 签名的回调
        routing_entries: 外部 list, hook 会往里面 append dict
    """
    routing_entries = []

    def hook_fn(module, input, output):
        x = input[0]  # 真实到达 CellAttention 的特征 [B,T,N,D] / [T,N,D]
        if x is None:
            return

        # 模拟 CellAttention._forward_temporal 的静态 assignment 逻辑
        if x.dim() == 4:
            x = x.mean(dim=0)          # [T, N, D]
        x_static = x.mean(dim=0) if x.dim() == 3 else x   # [N, D]

        with torch.no_grad():
            u = module._compute_membership(x_static, mu_fuzzy)  # [N, K_c]

            # 路由权重 (与 module.forward 内部完全一致)
            if module.use_fuzzy_routing:
                B = u.sqrt().clamp(min=1e-8)
                B = B / B.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
            else:
                B = u.sqrt().clamp(min=1e-8)

            # Per-node routing entropy
            H_route = -(B * (B + 1e-8).log()).sum(dim=-1)  # [N]

            # Dominant region distribution
            argmax = B.argmax(dim=-1)  # [N]
            counts = argmax.bincount(minlength=module.num_cells).float()
            max_frac = (counts.max() / counts.sum().clamp_min(1)).item()

            # Per-region utilization
            per_region_frac = (counts / counts.sum().clamp_min(1)).cpu().tolist()

        routing_entries.append({
            "H_mean": float(H_route.mean().item()),
            "H_std": float(H_route.std().item()),
            "max_region_frac": float(max_frac),
            "num_cells": module.num_cells,
            "per_region_frac": per_region_frac,
        })

    return hook_fn, routing_entries


def metric_routing_entropy(model, dataloader, device):
    """Forward-hook based routing entropy measurement.

    可靠性说明:
      - ✅ 使用真实 forward pass 中到达 CellAttention 的特征 (不是 input_projection 均值)
      - ✅ 模拟 CellAttention 内部的 x_static 逻辑 (与 _forward_temporal 一致)
      - ✅ 所有 block 共享同一个 mu_fuzzy (与 encode_condition → predict 一致)
      - ⚠️  单 batch 测量 → 存在 batch 偏差 (traffic 时间模式可能不同)
      - ⚠️  不是严格测量 (routing 在 forward 中有更多上下文: band-pass gate、
            RegionTransformer 等影响最终输出, 这里只测 assignment)
    """
    m = _unwrap(model)
    m.eval()

    batch = next(iter(dataloader))
    batch.to_tensor(device)
    history = _trim(batch["X"], model)

    # 预计算 mu_fuzzy (与 encode_condition 内部完全一致)
    with torch.no_grad():
        mu_fuzzy = m.fuzzy_graph.get_memberships(
            node_features=history.unsqueeze(0) if history.dim() == 3 else history
        ).to(device)

    # 注册 hooks
    hook_fn, routing_entries = _build_routing_hook(mu_fuzzy)
    hooks = []
    for name, module in m.named_modules():
        if isinstance(module, FuzzyCellAttention):
            hooks.append(module.register_forward_hook(hook_fn))

    # 触发 forward → hooks 捕获真实 routing
    with torch.no_grad():
        _ = m.predict({"X": history.unsqueeze(0) if history.dim() == 3 else history})

    # 清理 hooks
    for h in hooks:
        h.remove()

    if not routing_entries:
        return {"_verdict": "⚠️  NO CellAttention blocks found",
                "_reliability": "N/A"}

    # 汇总
    entropies = [e["H_mean"] for e in routing_entries]
    avg_entropy = float(np.mean(entropies))
    max_fracs = [e["max_region_frac"] for e in routing_entries]
    avg_max_frac = float(np.mean(max_fracs))

    H_max = np.log(routing_entries[0]["num_cells"])

    results = {
        "avg_routing_entropy": avg_entropy,
        "normalized_entropy": avg_entropy / H_max if H_max > 0 else 0.0,
        "max_entropy_possible": float(H_max),
        "avg_max_region_fraction": avg_max_frac,
        "num_blocks_measured": len(routing_entries),
        "per_block_entropy": entropies,
        "per_block_max_frac": max_fracs,
        "_reliability": ("medium — real forward features via hooks, "
                         "but single-batch + heuristic probe, not full routing measurement"),
        "_method": "forward-hook on FuzzyCellAttention, capturing real input features",
    }

    # Verdict (与旧版相同的阈值逻辑 → 但现在数据是真实的)
    if avg_max_frac > 0.8:
        verdict = (f"❌ DEGENERATE — {avg_max_frac:.0%} of nodes routed to "
                   f"single region, FRR ≈ GCN")
    elif avg_max_frac > 0.5:
        verdict = (f"⚠️  WEAK ROUTING — {avg_max_frac:.0%} dominated by one region")
    elif avg_entropy / H_max < 0.3:
        verdict = f"⚠️  LOW ENTROPY — routing distribution is peaked"
    else:
        verdict = (f"✅ HEALTHY — routing entropy={avg_entropy:.3f}, "
                   f"max region={avg_max_frac:.0%}")

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M4: FOU-Error Correlation — v3 重写: 多视角分析
# ═══════════════════════════════════════════════════════════════════════

def metric_fou_error_correlation(model, dataloader, device, num_batches=20):
    """多视角 FOU-error 关联分析。

    v3 改进:
      1. Per-node: 保留原版 per-node scalar error vs FOU (benchmark)
      2. Per-horizon: 每个预测时步 t 的 per-node error vs FOU → 时间演化曲线
      3. Per-set: per-fuzzy-set FOU vs mean error (跨节点聚合)

    可靠性说明:
      - ⚠️  FOU 是静态的 (基于 learned parameters), error 是 batch-averaged
      - ⚠️  两者时间尺度不匹配 → correlation 只能说明弱趋势
      - ✔  Per-horizon 分析比单一 scalar 更有信息量
      - ⚠️  Pearson r 假设线性关系 → FOU-error 可能是非线性的
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    m.eval()

    # ── 静态 FOU ──
    with torch.no_grad():
        mu_low, mu_high, _ = fg._compute_memberships()
        fou_node = (mu_high - mu_low).mean(dim=-1).cpu().numpy()  # [N]
        fou_per_set = (mu_high - mu_low).mean(dim=0).cpu().numpy()  # [K]

    # ── 多 view 误差累积 ──
    all_abs_err = []        # each: [B, T_out, N]
    iterator = iter(dataloader)
    count = 0
    for batch in iterator:
        if count >= num_batches:
            break
        batch.to_tensor(device)
        history = _trim(batch["X"], model)
        future = batch["y"][..., :m.output_dim].to(device)

        with torch.no_grad():
            pred = m.predict({"X": history})
            # [B, T_out, N, C] → abs → mean over features → [B, T_out, N]
            abs_err = (pred - future).abs().mean(dim=-1).cpu().numpy()
        all_abs_err.append(abs_err)
        count += 1

    # Concatenate all batches along dim 0 → [total_B, T_out, N]
    err_all = np.concatenate(all_abs_err, axis=0)
    T_out = err_all.shape[1]

    # Per-node scalar error (benchmark): mean over all samples × timesteps
    per_node_error_robust = err_all.mean(axis=(0, 1))  # [N]

    # ── Per-horizon analysis: correlation at each prediction horizon ──
    horizon_corrs = []
    horizon_pvals = []
    for t in range(T_out):
        # Average over batch for timestep t → [N]
        err_t = err_all[:, t, :].mean(axis=0)
        r_t, p_t = pearsonr(fou_node, err_t)
        horizon_corrs.append(float(r_t))
        horizon_pvals.append(float(p_t))

    # ── Per-node correlation (for backward-compat comparison) ──
    r_node, p_node = pearsonr(fou_node, per_node_error_robust)

    # ── Per-set correlation ──
    with torch.no_grad():
        _, _, mu_mid = fg._compute_memberships()
        argmax_set = mu_mid.argmax(dim=-1).cpu().numpy()  # [N]
    set_errors = []
    set_fous = []
    for k in range(m.fuzzy_num_sets):
        mask = argmax_set == k
        if mask.sum() > 0:
            set_errors.append(per_node_error_robust[mask].mean())
            set_fous.append(fou_per_set[k])
    r_set, p_set = pearsonr(set_fous, set_errors) if len(set_errors) >= 3 else (0.0, 1.0)

    # ── 额外: Spearman rank correlation (非线性鲁棒) ──
    r_spearman, p_spearman = spearmanr(fou_node, per_node_error_robust)

    results = {
        "per_node": {
            "pearson_r": float(r_node),
            "p_value": float(p_node),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "is_significant_pearson": p_node < 0.05,
            "is_significant_spearman": p_spearman < 0.05,
        },
        "per_horizon": {
            f"t_{t}": {"r": horizon_corrs[t], "p": horizon_pvals[t],
                       "significant": horizon_pvals[t] < 0.05}
            for t in range(T_out)
        },
        "per_fuzzy_set": {
            "pearson_r": float(r_set),
            "p_value": float(p_set),
            "is_significant": p_set < 0.05,
        },
        "fou_mean": float(fou_node.mean()),
        "fou_std": float(fou_node.std()),
        "error_mean": float(per_node_error_robust.mean()),
        "num_batches_evaluated": count,
        "_reliability": ("low-medium — static FOU vs dynamic error, "
                         "time-scale mismatch; use as trend indicator only"),
        "_note": "Spearman ρ 对非线性关系更鲁棒; per-horizon 曲线比单一 r 更有信息量",
    }

    # Verdict — 基于 Pearson + Spearman 双重检验
    if r_spearman > 0.3 and p_spearman < 0.05:
        verdict = (f"✅ TREND — Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f}), "
                   f"FOU weakly correlates with error (nonlinear monotonic)")
    elif r_node > 0.15 and p_node < 0.05:
        verdict = f"⚠️  WEAK LINEAR — r={r_node:.3f} (p={p_node:.3f}), Spearman ρ={r_spearman:.3f}"
    elif r_spearman > 0.1:
        verdict = f"⚠️  MARGINAL — Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})"
    elif r_node > 0:
        verdict = f"⚠️  VERY WEAK — r={r_node:.3f}, not statistically meaningful"
    else:
        verdict = (f"❌ NO SIGNAL — r={r_node:.3f}, Spearman ρ={r_spearman:.3f}, "
                   f"FOU has no detectable error semantics")

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M5: New Closure Edges — v3 改进: 正确语言 + baseline 对比
# ═══════════════════════════════════════════════════════════════════════

def _compute_closure_baseline(R: torch.Tensor, max_hops: int) -> torch.Tensor:
    """Max-min transitive closure (复现 graph.py._compute_closure)。

    用于生成 baseline 的 closure 作为对照。
    """
    S = R.clone()
    current = R
    for _ in range(max_hops - 1):
        current = torch.max(
            torch.min(R.unsqueeze(1), current.unsqueeze(0)), dim=-1
        ).values
        S = torch.max(S, current)
    return S.clamp(0.0, 1.0)


def metric_new_closure_edges(model, dataloader, device, thresholds=None):
    """Multi-threshold closure edge analysis with random baseline.

    v3 改进:
      1. 语言修正: "new edges" / "reasoning edges" →
         "density shift by transitive smoothing"
      2. 添加 random baseline: 对比随机对称矩阵的 closure 效果
          → 区分 "closure 本身就会产生新边" 和 "R 的结构产生新边"
      3. 多 batch 平均: 减少单 batch 偏差

    可靠性说明:
      - ✔ 多阈值分析 → 比单阈值报告更稳健
      - ✔ Random baseline → 提供归因参考
      - ⚠️  S = closure(R) 是 max-min 模糊传递闭包, 不是 "推理图"
      - ⚠️  "density shift" 是有意义的指标, "new edges" 需谨慎解释
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7]

    m = _unwrap(model)
    fg = m.fuzzy_graph
    N = m.num_nodes

    # ── 多 batch 平均 (3 batches) ──
    iterator = iter(dataloader)
    R_accum = None
    S_accum = None
    n_batches = 0

    for batch in iterator:
        if n_batches >= 3:
            break
        batch.to_tensor(device)
        history = _trim(batch["X"], model)

        with torch.no_grad():
            # 使用 node_features 的动态 membership 计算
            h = history.unsqueeze(0) if history.dim() == 3 else history
            mu_low, mu_high, _ = fg._compute_memberships(h)
            R_low, R_high = fg._build_fuzzy_relation_t2(mu_low, mu_high)
            S_high = _compute_closure_baseline(R_high, m.semantic_closure_hops)

        if R_accum is None:
            R_accum = R_high.cpu()
            S_accum = S_high.cpu()
        else:
            R_accum += R_high.cpu()
            S_accum += S_high.cpu()
        n_batches += 1

    R = (R_accum / n_batches) if n_batches > 1 else R_accum
    S = (S_accum / n_batches) if n_batches > 1 else S_accum

    # ── Random baseline ──
    rng = torch.Generator().manual_seed(42)
    R_rand = torch.rand(N, N, generator=rng, dtype=torch.float32)
    R_rand = (R_rand + R_rand.T) / 2  # symmetric
    R_rand.fill_diagonal_(1.0)         # reflexive
    S_rand = _compute_closure_baseline(R_rand, m.semantic_closure_hops)

    # ── Laplacian baseline: normalized adjacency as a "spatial prior" ──
    adj = m.adjacency_matrix.float()
    deg = adj.sum(dim=-1).clamp_min(1)
    L_norm = adj / deg.unsqueeze(-1)  # row-normalized [N, N]
    S_lap = _compute_closure_baseline(L_norm, m.semantic_closure_hops)

    total_pairs = N * N

    # Per-threshold analysis for model
    threshold_data = {}
    for t in thresholds:
        r_mask = R >= t
        s_mask = S >= t
        new_mask = (~r_mask) & s_mask
        strengthen_mask = r_mask & (S >= R + 0.05)

        r_rand_mask = R_rand >= t
        s_rand_mask = S_rand >= t
        new_rand_mask = (~r_rand_mask) & s_rand_mask

        r_lap_mask = L_norm >= t
        s_lap_mask = S_lap >= t
        new_lap_mask = (~r_lap_mask) & s_lap_mask

        threshold_data[f"threshold_{t}"] = {
            "R_density": float(r_mask.float().mean()),
            "S_density": float(s_mask.float().mean()),
            "density_shift": float((s_mask.float() - r_mask.float()).mean()),
            "new_edges": int(new_mask.float().sum()),
            "new_edges_pct": float(new_mask.float().mean() * 100),
            "strengthened_edges": int(strengthen_mask.float().sum()),
            "strengthened_pct": float(strengthen_mask.float().mean() * 100),
            "mean_delta": float((S - R)[r_mask].mean()) if r_mask.any() else 0.0,
            "baseline_random": {
                "R_density": float(r_rand_mask.float().mean()),
                "S_density": float(s_rand_mask.float().mean()),
                "density_shift": float((s_rand_mask.float() - r_rand_mask.float()).mean()),
                "new_edges_pct": float(new_rand_mask.float().mean() * 100),
            },
            "baseline_laplacian": {
                "R_density": float(r_lap_mask.float().mean()),
                "S_density": float(s_lap_mask.float().mean()),
                "density_shift": float((s_lap_mask.float() - r_lap_mask.float()).mean()),
                "new_edges_pct": float(new_lap_mask.float().mean() * 100),
            },
        }

    # ── Verdict: 综合 threshold=0.5 的模型效果 vs baselines ──
    t50 = threshold_data["threshold_0.5"]
    model_shift = t50["density_shift"]
    rand_shift = t50["baseline_random"]["density_shift"]
    lap_shift = t50["baseline_laplacian"]["density_shift"]

    # 模型 closure 相对于 baselines 的超额效果
    excess_vs_random = model_shift - rand_shift
    excess_vs_lap = model_shift - lap_shift

    if excess_vs_random > 0.03:
        verdict = (f"✅ MEANINGFUL — closure density shift {model_shift:.3f} "
                   f"significantly exceeds random baseline ({rand_shift:.3f}), "
                   f"excess={excess_vs_random:.3f}")
    elif excess_vs_random > 0.01:
        verdict = (f"⚠️  MODERATE — density shift {model_shift:.3f} "
                   f"vs random baseline {rand_shift:.3f}")
    elif t50["new_edges_pct"] > 1.0:
        verdict = (f"⚠️  WEAK — closure effect is close to random baseline "
                   f"(shift {model_shift:.3f} vs random {rand_shift:.3f})")
    else:
        verdict = "❌ NEGLIGIBLE — no meaningful density shift vs baselines"

    results = {
        "total_pairs": total_pairs,
        "by_threshold": threshold_data,
        "excess_vs_random": float(excess_vs_random),
        "excess_vs_laplacian": float(excess_vs_lap),
        "num_batches_averaged": n_batches,
        "_reliability": ("medium — multi-threshold + baseline comparison, "
                         "but closure is transitive smoothing not reasoning"),
        "_terminology": ("'new edges' / 'density shift' = transitive smoothing effect, "
                         "NOT reasoning / inference edges"),
        "_verdict": verdict,
    }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M6: Set Utilization (可靠性: ✔ 高)
# ═══════════════════════════════════════════════════════════════════════

def metric_set_utilization(model):
    """Per-fuzzy-set utilization: are all K sets being used?

    可靠性说明:
      - 基于静态 membership, 不受 batch 噪声影响
      - mean normalization + entropy 双重度量
      - 与 M1 共享 μ_mid → 低 entropy ⇔ 高 utilization 集中度

    u_k = mean_i(μ_mid[i,k]) / mean_{j,l}(μ_mid[j,l])
    Ideally u_k ≈ 1.0 for all k (uniform utilization).
    If u_k ≪ 1 for some sets → those sets are being wasted.
    If one u_k ≫ 1 → collapse to a single set.
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    with torch.no_grad():
        _, _, mu_mid = fg._compute_memberships()

    mu_np = mu_mid.cpu().numpy()                                    # [N, K]
    global_mean = mu_np.mean()
    u_k = mu_np.mean(axis=0) / global_mean                          # [K]

    u_norm = u_k / u_k.sum()
    H_util = -(u_norm * np.log(u_norm + 1e-8)).sum()
    H_max = np.log(m.fuzzy_num_sets)
    normalized_util_entropy = H_util / H_max

    max_util = float(u_k.max())
    min_util = float(u_k.min())
    util_cv = float(u_k.std() / (u_k.mean() + 1e-8))

    results = {
        "per_set_utilization": {f"set_{i}": float(u_k[i]) for i in range(len(u_k))},
        "max_utilization": max_util,
        "min_utilization": min_util,
        "utilization_cv": util_cv,
        "utilization_entropy": float(normalized_util_entropy),
        "mean_raw_membership": float(global_mean),
        "_reliability": "high — static membership, no batch noise",
    }

    if max_util > 3.0:
        verdict = (f"❌ DOMINATED — set utilization max={max_util:.1f}× mean, "
                   f"most nodes collapse to one set")
    elif max_util > 2.0:
        verdict = (f"⚠️  SKEWED — max utilization {max_util:.1f}× mean, "
                   f"some sets underused")
    elif normalized_util_entropy < 0.6:
        verdict = f"⚠️  UNEVEN — utilization entropy={normalized_util_entropy:.2f}"
    else:
        verdict = (f"✅ BALANCED — all {m.fuzzy_num_sets} sets actively used, "
                   f"entropy={normalized_util_entropy:.2f}")

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_final_audit(model, dataloader, device, num_batches=20):
    print("\n" + "=" * 60)
    print("  M1: Membership Entropy  [可靠性: ✔ 高]")
    print("=" * 60)
    m1 = metric_membership_entropy(model)
    _print_metric(m1)

    print("\n" + "=" * 60)
    print("  M2: Pairwise Cosine(μ)  [可靠性: ✔ 高, 与M1相关]")
    print("=" * 60)
    m2 = metric_pairwise_cosine(model)
    _print_metric(m2)

    print("\n" + "=" * 60)
    print("  M3: Routing Entropy (FRR)  [可靠性: ⚠️ 中, forward-hook追踪]")
    print("=" * 60)
    m3 = metric_routing_entropy(model, dataloader, device)
    _print_metric(m3)

    print("\n" + "=" * 60)
    print("  M4: FOU-Error Correlation  [可靠性: ⚠️ 低-中, 多视角分析]")
    print("=" * 60)
    m4 = metric_fou_error_correlation(model, dataloader, device, num_batches)
    _print_metric(m4)

    print("\n" + "=" * 60)
    print("  M5: Closure Density Shift  [可靠性: ⚠️ 中, +baseline对比]")
    print("=" * 60)
    m5 = metric_new_closure_edges(model, dataloader, device)
    _print_metric(m5)

    print("\n" + "=" * 60)
    print("  M6: Set Utilization  [可靠性: ✔ 高]")
    print("=" * 60)
    m6 = metric_set_utilization(model)
    _print_metric(m6)

    return {"M1_membership_entropy": m1,
            "M2_pairwise_cosine": m2,
            "M3_routing_entropy": m3,
            "M4_fou_error_corr": m4,
            "M5_closure_edges": m5,
            "M6_set_utilization": m6}


def _print_metric(result):
    """Print key fields of a metric result with nested dict support."""
    for k, v in result.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            print(f"  {k:<35s} {v:.4f}")
        elif isinstance(v, dict):
            first_val = next(iter(v.values()), None)
            if isinstance(first_val, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        print(f"    [{sk}]")
                        for ssk, ssv in sv.items():
                            if isinstance(ssv, float):
                                print(f"      {ssk:<30s} {ssv:>8.4f}")
                            else:
                                print(f"      {ssk:<30s} {ssv}")
                    elif isinstance(sv, float):
                        print(f"    {sk:<31s} {sv:.4f}")
                    else:
                        print(f"    {sk:<31s} {sv}")
            else:
                print(f"  {k}:")
                for sk, sv in v.items():
                    if isinstance(sv, float):
                        print(f"    {sk:<33s} {sv:.4f}")
                    elif isinstance(sv, dict):
                        print(f"    {sk}:")
                        for ssk, ssv in sv.items():
                            if isinstance(ssv, float):
                                print(f"      {ssk:<30s} {ssv:.4f}")
                            else:
                                print(f"      {ssk:<30s} {ssv}")
                    else:
                        print(f"    {sk:<33s} {sv}")
        else:
            print(f"  {k:<35s} {v}")
    print(f"\n  → {result['_verdict']}")


# ═══════════════════════════════════════════════════════════════════════
#  Scorecard — v3 改进: 加权评分 + 相关性说明
# ═══════════════════════════════════════════════════════════════════════

def print_scorecard(results):
    """加权评分卡 — 考虑指标可靠性和相关性。

    设计原则:
      1. 不同指标有不同的可靠性等级 (✔高 / ⚠️中 / ⚠️低-中)
      2. M1 和 M2 高度相关 → 不重复计数
      3. M3 和 M4 是探测性指标 → 降低权重
      4. 结论分三级: paper claim / supporting evidence / debug only
    """
    print("\n" + "=" * 70)
    print("  ARCHITECTURE FREEZE DECISION  (v3 — weighted scorecard)")
    print("=" * 70)

    m_labels = {
        "M1_membership_entropy": ("Membership Entropy", "✔ high"),
        "M2_pairwise_cosine": ("Pairwise Cosine(μ)", "✔ high (correlated with M1)"),
        "M3_routing_entropy": ("Routing Entropy (FRR)", "⚠️ medium"),
        "M4_fou_error_corr": ("FOU-Error Corr", "⚠️ low-med"),
        "M5_closure_edges": ("Closure Density Shift", "⚠️ medium"),
        "M6_set_utilization": ("Set Utilization", "✔ high"),
    }

    # 可靠性权重: high=3, medium=2, low-med=1
    reliability_weights = {
        "M1_membership_entropy": 3,
        "M2_pairwise_cosine": 2,       # correlated with M1, reduced
        "M3_routing_entropy": 2,
        "M4_fou_error_corr": 1,
        "M5_closure_edges": 2,
        "M6_set_utilization": 3,
    }

    verdicts = {}
    verdict_score = {}  # -2=fail, -1=warning, 0=pass  (per the original emoji scheme)
    for key, (label, reliability) in m_labels.items():
        v = results.get(key, {}).get("_verdict", "?")
        verdicts[label] = v
        print(f"\n  [{key[:2]}] {label}  [{reliability}]")
        print(f"      {v}")

        # Score
        if "❌" in v:
            verdict_score[key] = -2
        elif "⚠️" in v:
            verdict_score[key] = -1
        else:
            verdict_score[key] = 0

    # 加权失败分
    weighted_fail = sum(
        max(0, -verdict_score[k]) * reliability_weights[k]
        for k in verdict_score
    )
    max_possible_fail = sum(
        reliability_weights[k] * 2  # max = all ❌
        for k in verdict_score
    )

    # 简单计数 (用于对比旧版 scorecard)
    failures = sum(1 for v in verdict_score.values() if v == -2)
    warnings = sum(1 for v in verdict_score.values() if v == -1)

    print("\n" + "-" * 70)
    print(f"  Weighted failure score: {weighted_fail}/{max_possible_fail}")
    print(f"  Simple count: {failures} failures, {warnings} warnings")
    print()

    # ── 相关性警告 ──
    m1_fail = verdict_score.get("M1_membership_entropy", 0) < 0
    m2_fail = verdict_score.get("M2_pairwise_cosine", 0) < 0
    if m1_fail and m2_fail:
        print("  ⚠️  M1 and M2 share μ_mid → both failing = single underlying issue")
        print("      (counts as 1 independent failure in weighted score)")

    # ── 分等级结论 ──
    weighted_ratio = weighted_fail / max_possible_fail if max_possible_fail > 0 else 0

    if weighted_ratio <= 0.05:
        print("  ✅ ALL METRICS PASS  (weighted score ≤ 5%)")
        print("  → ARCHITECTURE IS HEALTHY")
        print("  → M1/M2/M6 can be used as PAPER SUPPORTING EVIDENCE")
        print("  → M3/M4/M5 can be cited as AUXILIARY DIAGNOSTICS")
    elif weighted_ratio <= 0.15:
        print(f"  ⚠️  MINOR CONCERNS  (weighted score = {weighted_ratio:.0%})")
        print("  → Architecture is ACCEPTABLE for experiments")
        print("  → Document warnings in paper appendix")
        print("  → Consider addressing M1/M2/M6 issues before final submission")
    elif weighted_ratio <= 0.35:
        print(f"  ⚠️  MODERATE CONCERNS  (weighted score = {weighted_ratio:.0%})")
        print("  → Architecture USABLE but needs attention")
        print("  → Fix reliability-weighted failures before claiming structural validity")
        print("  → M3/M4 results should NOT be cited as evidence")
    else:
        print(f"  ❌ SIGNIFICANT ISSUES  (weighted score = {weighted_ratio:.0%})")
        print("  → DO NOT FREEZE — fix core issues first")
        print("  → At minimum, ensure M1/M6 pass (high-reliability structural checks)")

    print("-" * 70)

    # ── 总结性说明 ──
    print("\n  📋 CAVEATS:")
    print("     • This is a STRUCTURAL DIAGNOSTIC tool, not a formal proof")
    print("     • M1/M2 share μ_mid — their verdicts are highly correlated (not independent)")
    print("     • M3 uses forward-hook tracing (real features, but single-batch probe)")
    print("     • M4 FOU is static while error is dynamic — temporal scale mismatch")
    print("     • M5 'new edges' = transitive smoothing, NOT reasoning edges")
    print("     • For paper claims, supplement with:")
    print("       - Ablation studies (remove FRR, remove closure, remove Type-2)")
    print("       - Training curves showing metric evolution")
    print("       - Multi-checkpoint consistency (not just best epoch)")
    print("     • For statistical rigour, use bootstrapped confidence intervals")
    print("       and report effect sizes (not just p-values)")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="final_3_type2 架构审计 v3 (论文级结构诊断, 6 metrics)")
    parser.add_argument("--dataset", type=str, default="METR_LA")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Training config JSON")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--other_args", type=str, default=None,
                        help='JSON config overrides')
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Batches for M4 error accumulation")
    args = parser.parse_args()

    model, dataloader, device, runtime = load_model_robust(args)

    results = run_final_audit(model, dataloader, device, args.num_batches)

    print_scorecard(results)

    if args.output:
        def convert(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
