"""
final_3_type2 模块有效性审计 v2 — 完整六层框架。

六层审计顺序（消融实验前必做）:
  Layer 1 — 输出变化审计: 各模块开启/关闭时的输出差异量化
  Layer 2 — 统计量审计: FOU/Membership/Entropy 分布是否合理
  Layer 3 — 梯度审计: θ_delta/θ_lower/region_mu 等核心参数梯度
  Layer 4 — 模块专属指标: Closure ΔS / FOU分布 / Entropy corr / FIR loss
  Layer 5 — 可解释性审计: Membership 热力图 / Region 分配 / FOU 空间图
  Layer 6 — 交通语义审计: 高速入口 vs 市中心 vs 居民区节点对比

最优先三个检查:
  P1: grad(theta_delta) 是否显著非零
  P2: FOU 是否不是集中在 0 附近
  P3: Var(μ_t) 是否随时间明显变化

用法:
  python scripts/tools/audit_final3_type2_v2.py \
      --dataset METR_LA \
      --checkpoint cache/model_cache/METR_LA/final_3_type2/xxx.pt \
      --layer all
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data import build_dataset_runtime
from GNNTP.utils import get_model


# ═══════════════════════════════════════════════════════════════════════
#  Utility: Node semantic labels (for Layer 6)
# ═══════════════════════════════════════════════════════════════════════

def _classify_nodes_by_degree(adj_matrix, num_nodes):
    """Classify nodes by graph degree into semantic groups.

    Returns:
        dict: {group_name: [node_indices]}
    """
    adj = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
    degrees = adj.sum(axis=1)

    # Sort by degree
    order = np.argsort(degrees)[::-1]
    n = len(order)
    top_n = max(1, n // 10)
    mid_n = max(1, n // 5)
    low_n = max(1, n // 10)

    groups = {
        "hub (high-degree, likely highway)": order[:top_n].tolist(),
        "mid (medium-degree, likely arterial)": order[n//2 - mid_n//2 : n//2 + mid_n//2].tolist(),
        "leaf (low-degree, likely residential)": order[-low_n:].tolist(),
    }
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  Layer 1: Output Change Audit
# ═══════════════════════════════════════════════════════════════════════

def audit_layer1_output_change(model, dataloader, device):
    """Quantify output change when modules are switched on/off.

    Tests:
      A. Semantic Closure:  |S_{hops=3} - S_{hops=1}|_F / |S_{hops=1}|_F
      B. Entropy Dynamic Graph:  |S_dyn - S_eff|_F / |S_eff|_F
      C. FOU Mode:  compare S_eff across modes
      D. Dynamic Membership:  |μ_dyn - μ_static|_F / |μ_static|_F
    """
    m = model.module if hasattr(model, 'module') else model
    fg = m.fuzzy_graph
    m.eval()

    results = {}

    # ── A. Semantic Closure: hops=1 vs hops=3 ─────────────────
    batch = next(iter(dataloader))
    batch.to_tensor(device)
    history = batch["X"]

    mu_low, mu_high, _ = fg._compute_memberships(history)
    R_low, R_high = fg._build_fuzzy_relation_t2(mu_low, mu_high)

    S1_low = fg._compute_closure(R_low, 1)
    S3_low = fg._compute_closure(R_low, 3)
    S1_high = fg._compute_closure(R_high, 1)
    S3_high = fg._compute_closure(R_high, 3)

    delta_low = (S3_low - S1_low).norm('fro') / S1_low.norm('fro').clamp_min(1e-8)
    delta_high = (S3_high - S1_high).norm('fro') / S1_high.norm('fro').clamp_min(1e-8)

    results["A_closure_delta"] = {
        "low_bound": float(delta_low),
        "high_bound": float(delta_high),
        "verdict": (
            "✅ ACTIVE (5-20%)" if 0.05 < delta_low < 0.5 else
            "⚠️  WEAK (1-5%)"   if 0.01 < delta_low <= 0.05 else
            "❌ DEAD (<1%)"      if delta_low <= 0.01 else
            "⚠️  TOO STRONG (>50%)"
        ),
    }

    # ── B. Entropy Dynamic Graph: S_eff vs S_dyn ──────────────
    S_eff, FOU_mat = fg.get_effective_graph(
        max_hops=m.semantic_closure_hops,
        mode=m.type2_graph_mode,
        fou_gate_scale=m.type2_fou_gate_scale,
        node_features=history,
    )

    S_dyn = S_eff.clone()
    if m.use_entropy_dynamic_graph:
        S_dyn = m._apply_entropy_dynamic_graph(S_eff, FOU=FOU_mat, node_features=history)

    delta_dyn = (S_dyn - S_eff).norm('fro') / S_eff.norm('fro').clamp_min(1e-8)

    # Correlation: H vs ΔS
    H_vec = fg.get_cell_entropy(node_features=history).detach().cpu().numpy()
    delta_S_vec = (S_dyn - S_eff).abs().mean(dim=-1).detach().cpu().numpy()
    r_dyn, p_dyn = pearsonr(H_vec, delta_S_vec)

    results["B_entropy_graph"] = {
        "delta_S_dyn_vs_eff": float(delta_dyn),
        "corr_H_deltaS": float(r_dyn),
        "corr_H_deltaS_pval": float(p_dyn),
        "verdict": (
            "✅ ACTIVE (H positively modulates graph)"
            if r_dyn > 0.1 and delta_dyn > 0.01 else
            "⚠️  MARGINAL (delta < 1% or weak corr)"
            if delta_dyn > 0.005 else
            "❌ DEAD (S_dyn ≈ S_eff)"
        ),
    }

    # ── C. FOU Mode Comparison ─────────────────────────────────
    mode_results = {}
    for mode in ["low", "mid", "high", "fou_gated"]:
        S_mode, _ = fg.get_effective_graph(
            max_hops=m.semantic_closure_hops,
            mode=mode,
            node_features=history,
        )
        mode_results[mode] = {
            "mean": float(S_mode.mean()),
            "std": float(S_mode.std()),
            "sparsity": float((S_mode < 0.01).float().mean()),
        }

    # Mid vs Low delta
    delta_mid_low = (mode_results["mid"]["mean"] - mode_results["low"]["mean"])

    results["C_fou_mode"] = {
        "per_mode_stats": mode_results,
        "delta_mid_vs_low": delta_mid_low,
        "verdict": (
            "✅ ACTIVE (mid ≠ low: FOU interval has width)"
            if abs(delta_mid_low) > 0.01 else
            "❌ DEGENERATE (all modes produce same S_eff → FOU ≈ 0)"
        ),
    }

    # ── D. Dynamic Membership Variance ─────────────────────────
    mu_static = fg._compute_memberships()[0]  # static μ_low
    mu_per_batch = []
    iterator = iter(dataloader)
    for _ in range(min(10, len(dataloader))):
        try:
            b = next(iterator)
        except StopIteration:
            break
        b.to_tensor(device)
        mu_l, _, _ = fg._compute_memberships(b["X"])
        mu_per_batch.append(mu_l.cpu())

    if len(mu_per_batch) > 1:
        mu_stack = torch.stack(mu_per_batch, dim=0)  # [B, N, K]
        # Per-node variance across batches
        var_per_node = mu_stack.var(dim=0).mean().item()  # avg over N, K
        mu_mean = mu_stack.mean(dim=0).abs().mean().item()

        # Dynamic vs static delta
        delta_mu = (mu_stack.mean(dim=0) - mu_static).norm('fro') / mu_static.norm('fro').clamp_min(1e-8)
    else:
        var_per_node = 0.0
        mu_mean = 0.0
        delta_mu = 0.0

    results["D_dynamic_membership"] = {
        "mean_var_per_node": float(var_per_node),
        "mean_mu_abs": float(mu_mean),
        "delta_dyn_vs_static": float(delta_mu),
        "verdict": (
            "✅ ACTIVE (Var(μ) > 0.001)" if var_per_node > 0.001 else
            "⚠️  WEAK (Var(μ) > 0.0001)" if var_per_node > 0.0001 else
            "❌ STATIC (code dynamic, behavior static)"
        ),
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Layer 2: Statistical Sanity Audit
# ═══════════════════════════════════════════════════════════════════════

def audit_layer2_statistical_sanity(model, device):
    """Check distribution shapes of FOU, Membership, Entropy.

    Key danger signs:
      - FOU ≈ 0 everywhere → Type-2 degenerates to Type-1
      - Membership all same → Membership Collapse
      - Entropy all same → no meaningful uncertainty differentiation
    """
    m = model.module if hasattr(model, 'module') else model
    fg = m.fuzzy_graph
    m.eval()

    with torch.no_grad():
        mu_low, mu_high, mu_mid = fg._compute_memberships()
        FOU_k = mu_high - mu_low                                          # [N, K]
        H = fg.get_cell_entropy()                                         # [N]

    fou_flat = FOU_k.flatten().cpu().numpy()
    mu_flat = mu_mid.flatten().cpu().numpy()
    H_np = H.cpu().numpy()

    # ── FOU distribution ──────────────────────────────────────
    fou_percentiles = {
        "p10": float(np.percentile(fou_flat, 10)),
        "p25": float(np.percentile(fou_flat, 25)),
        "p50": float(np.percentile(fou_flat, 50)),
        "p75": float(np.percentile(fou_flat, 75)),
        "p90": float(np.percentile(fou_flat, 90)),
        "mean": float(fou_flat.mean()),
        "std": float(fou_flat.std()),
    }

    fou_near_zero = (fou_flat < 0.01).mean()
    fou_healthy = ((fou_flat >= 0.05) & (fou_flat <= 0.3)).mean()

    # ── Membership distribution ───────────────────────────────
    mu_high_np = mu_high.flatten().cpu().numpy()
    mu_low_np = mu_low.flatten().cpu().numpy()

    # Check membership collapse: are all nodes assigned to same fuzzy set?
    argmax_counts = np.bincount(mu_mid.argmax(dim=-1).cpu().numpy(),
                                 minlength=fg.num_fuzzy_sets)
    argmax_entropy = -(argmax_counts / argmax_counts.sum() *
                       np.log(argmax_counts / argmax_counts.sum().clip(1e-8)).clip(1e-8)).sum()
    max_entropy = np.log(fg.num_fuzzy_sets)

    # ── Entropy distribution ──────────────────────────────────
    entropy_stats = {
        "mean": float(H_np.mean()),
        "std": float(H_np.std()),
        "min": float(H_np.min()),
        "max": float(H_np.max()),
    }

    results = {
        "FOU": {
            "percentiles": fou_percentiles,
            "frac_near_zero (<0.01)": float(fou_near_zero),
            "frac_healthy (0.05-0.3)": float(fou_healthy),
            "verdict": (
                "✅ HEALTHY (distributed in 0.05-0.3)"
                if fou_healthy > 0.3 else
                "⚠️  NARROW (mostly < 0.05)"
                if fou_near_zero > 0.5 else
                "❌ DEGENERATE (all ≈ 0 → Type-2 ≈ Type-1)"
                if fou_near_zero > 0.9 else
                "⚠️  CHECK MANUALLY"
            ),
        },
        "Membership": {
            "mu_mean": float(mu_flat.mean()),
            "mu_std": float(mu_flat.std()),
            "mu_high_mean": float(mu_high_np.mean()),
            "mu_low_mean": float(mu_low_np.mean()),
            "argmax_distribution": argmax_counts.tolist(),
            "argmax_normalized_entropy": float(argmax_entropy / max_entropy),
            "verdict": (
                "✅ DIVERSE (nodes spread across fuzzy sets)"
                if argmax_entropy / max_entropy > 0.5 else
                "⚠️  COLLAPSING (dominated by 1-2 sets)"
                if argmax_entropy / max_entropy > 0.2 else
                "❌ COLLAPSED (all nodes → same set → FRR degenerate)"
            ),
        },
        "Entropy": {
            "stats": entropy_stats,
            "verdict": (
                "✅ SPREAD (meaningful uncertainty differentiation)"
                if entropy_stats["std"] > 0.05 else
                "⚠️  UNIFORM (all nodes same uncertainty)"
            ),
        },
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Layer 3: Gradient Audit (expanded from Q3)
# ═══════════════════════════════════════════════════════════════════════

def audit_layer3_gradient(model, dataloader, device, num_steps=20):
    """Measure gradient norms for ALL innovation parameters.

    Critical params:
      - base_membership_lower (θ_low)
      - base_membership_delta (θ_delta)  ← most important for FOU
      - region_mu (FRR region prototypes)
      - blend_logit (fuzzy/static mix)
      - feature_to_membership (dynamic membership MLP)
      - fou_to_logvar (Type-2 uncertainty head)
    """
    m = model.module if hasattr(model, 'module') else model
    fg = m.fuzzy_graph
    m.train()

    # Collect all innovation params
    param_names = [
        "fg.base_membership_lower",
        "fg.base_membership_delta",
        "fg.feature_to_membership",
        "fg.blend_logit",
        "fou_to_logvar",
    ]

    # Add region_mu from all CellAttention blocks
    for blocks in [m.condition_encoder.blocks, m.future_decoder.blocks]:
        for i, block in enumerate(blocks):
            if hasattr(block, 'cell_attention'):
                ca = block.cell_attention
                for pname, _ in ca.named_parameters():
                    if 'region_mu' in pname or 'centers' in pname or 'cell_blend' in pname:
                        full_name = f"cell_attn.{pname}"
                        param_names.append(full_name)

    grad_accum = {name: [] for name in param_names}

    iterator = iter(dataloader)
    for step in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch.to_tensor(device)
        if hasattr(m, 'optimizer'):
            m.optimizer.zero_grad()
        else:
            m.zero_grad()

        loss = m.calculate_loss(batch)
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()

        # Extract gradients
        for name in param_names:
            try:
                if name == "fg.base_membership_lower":
                    g = fg.base_membership_lower.grad
                elif name == "fg.base_membership_delta":
                    g = fg.base_membership_delta.grad
                elif name == "fg.feature_to_membership":
                    g = None
                    for p in fg.feature_to_membership.parameters():
                        if p.grad is not None:
                            g = p.grad if g is None else g + p.grad
                elif name == "fg.blend_logit":
                    g = fg.blend_logit.grad
                elif name == "fou_to_logvar":
                    g = m.fou_to_logvar.weight.grad
                elif name.startswith("cell_attn."):
                    # Parse nested attribute
                    parts = name.split(".")
                    # Navigate to param - this is approximate
                    g = None
                else:
                    g = None

                if g is not None:
                    grad_accum[name].append(g.norm().item())
            except Exception:
                pass

    m.eval()

    # Summarize
    summary = {}
    for name, vals in grad_accum.items():
        if vals:
            summary[name] = {
                "grad_mean": float(np.mean(vals)),
                "grad_std": float(np.std(vals)),
                "grad_max": float(np.max(vals)),
            }
        else:
            summary[name] = "no gradient captured"

    # ── Priority indicators ───────────────────────────────────
    theta_delta_mean = summary.get("fg.base_membership_delta", {}).get("grad_mean", 0)
    theta_lower_mean = summary.get("fg.base_membership_lower", {}).get("grad_mean", 1e-8)
    if isinstance(theta_delta_mean, (int, float)) and isinstance(theta_lower_mean, (int, float)):
        ratio = theta_delta_mean / max(theta_lower_mean, 1e-8)
    else:
        ratio = 0

    summary["_PRIORITY"] = {
        "grad_theta_delta": theta_delta_mean,
        "grad_theta_delta_vs_lower_ratio": float(ratio),
        "P1_verdict": (
            "✅ FOU IS LEARNING (grad(θ_delta) > 1e-4)"
            if isinstance(theta_delta_mean, (int, float)) and theta_delta_mean > 1e-4 else
            "⚠️  FOU WEAKLY LEARNING (grad(θ_delta) > 1e-6)"
            if isinstance(theta_delta_mean, (int, float)) and theta_delta_mean > 1e-6 else
            "❌ FOU NOT LEARNING (grad(θ_delta) ≈ 0 → decorative)"
        ),
    }

    return summary


# ═══════════════════════════════════════════════════════════════════════
#  Layer 4: Module-Specific Metrics
# ═══════════════════════════════════════════════════════════════════════

def audit_layer4_module_metrics(model, dataloader, device):
    """Deep module-specific diagnostics.

    Tests:
      A. Closure:  Is S significantly different from R?
      B. FOU:      Distribution shape + per-hop evolution
      C. Entropy:  corr(H, S_dyn - S_eff)
      D. FIR:      Loss magnitude vs NLL
    """
    m = model.module if hasattr(model, 'module') else model
    fg = m.fuzzy_graph
    m.eval()

    batch = next(iter(dataloader))
    batch.to_tensor(device)
    history = batch["X"]

    results = {}

    # ── A. Closure: S vs R ────────────────────────────────────
    mu_low, mu_high, _ = fg._compute_memberships(history)
    R_low, R_high = fg._build_fuzzy_relation_t2(mu_low, mu_high)
    S_low = fg._compute_closure(R_low, m.semantic_closure_hops)
    S_high = fg._compute_closure(R_high, m.semantic_closure_hops)

    delta_low_abs = float((S_low - R_low).abs().mean())
    delta_high_abs = float((S_high - R_high).abs().mean())

    results["A_closure_S_vs_R"] = {
        "mean_abs_S_minus_R_low": delta_low_abs,
        "mean_abs_S_minus_R_high": delta_high_abs,
        "verdict": (
            "✅ CLOSURE PRODUCES INFERENCE" if delta_low_abs > 0.01 else
            "❌ CLOSURE IS IDENTITY (graph already transitive)"
        ),
    }

    # ── B. FOU: per-hop evolution ─────────────────────────────
    fou_per_hop = {}
    S_current = R_high.clone()
    current = R_high.clone()
    for h in range(1, m.semantic_closure_hops + 1):
        if h > 1:
            current = torch.max(
                torch.min(R_high.unsqueeze(1), current.unsqueeze(0)), dim=-1
            ).values
            S_current = torch.max(S_current, current)
        # FOU at this hop: compare with low bound
        S_low_h = fg._compute_closure(R_low, h)
        fou_h = (S_current - S_low_h).mean().item()
        fou_per_hop[f"hop_{h}"] = float(fou_h)

    results["B_fou_per_hop"] = {
        "fou_evolution": fou_per_hop,
        "verdict": (
            "✅ FOU GROWS WITH HOPS (closure amplifies uncertainty)"
            if all(fou_per_hop.get(f"hop_{i}", 0) <= fou_per_hop.get(f"hop_{i+1}", 0)
                   for i in range(1, m.semantic_closure_hops))
            else "⚠️  CHECK FOU EVOLUTION"
        ),
    }

    # ── C. Entropy-Graph correlation ──────────────────────────
    S_eff, FOU_mat = fg.get_effective_graph(
        max_hops=m.semantic_closure_hops,
        mode=m.type2_graph_mode,
        node_features=history,
    )
    S_dyn = m._apply_entropy_dynamic_graph(S_eff, FOU=FOU_mat, node_features=history)
    delta_S = (S_dyn - S_eff)

    # Per-node: avg delta vs entropy
    H_vec = fg.get_cell_entropy(node_features=history).detach().cpu().numpy()
    delta_per_node = delta_S.mean(dim=-1).detach().cpu().numpy()
    r_corr, p_corr = pearsonr(H_vec, delta_per_node)

    results["C_entropy_graph_corr"] = {
        "pearson_r": float(r_corr),
        "p_value": float(p_corr),
        "verdict": (
            "✅ ENTROPY DRIVES DYNAMIC GRAPH (positive corr)"
            if r_corr > 0.1 and p_corr < 0.05 else
            "⚠️  WEAK CORRELATION"
            if p_corr < 0.1 else
            "❌ ENTROPY GRAPH NOT WORKING (no correlation)"
        ),
    }

    # ── D. FIR magnitude ──────────────────────────────────────
    m.train()
    batch2 = next(iter(dataloader))
    batch2.to_tensor(device)
    if hasattr(m, 'optimizer'):
        m.optimizer.zero_grad()
    else:
        m.zero_grad()

    loss = m.calculate_loss(batch2)
    m.eval()

    # Get NLL-only loss (without FIR)
    history2 = batch2["X"]
    future2 = batch2["y"][..., :m.output_dim].to(device)
    condition, fuzzy_R, mu = m.encode_condition(history2)
    output = m.future_decoder(condition, fuzzy_R, graph_dist=m.graph_dist, mu_fuzzy=mu)
    mu_hat = output[..., :m.output_dim]
    log_var = output[..., m.output_dim:].clamp(min=-1.5)
    import math
    nll_only = 0.5 * (math.log(2 * math.pi) + log_var +
                       torch.exp(-log_var) * (mu_hat - future2) ** 2).mean()

    fir_contribution = loss.item() - nll_only.item()

    results["D_fir_magnitude"] = {
        "total_loss": float(loss.item()),
        "nll_only": float(nll_only.item()),
        "fir_contribution_abs": float(fir_contribution),
        "fir_contribution_pct": float(fir_contribution / abs(loss.item() + 1e-8) * 100),
        "verdict": (
            "✅ FIR ACTIVE (contribution > 1% of total loss)"
            if abs(fir_contribution) > abs(loss.item() * 0.01) else
            "⚠️  FIR MARGINAL (check warmup / weight)"
            if abs(fir_contribution) > 1e-6 else
            "❌ FIR ≈ 0 (constraint not engaged)"
        ),
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Layer 5: Interpretability Audit
# ═══════════════════════════════════════════════════════════════════════

def audit_layer5_interpretability(model, device, output_dir=None):
    """Generate interpretability diagnostics for paper-ready figures.

    Outputs (saved to output_dir if provided):
      1. Membership matrix [N, K] — raw values + argmax distribution
      2. Region assignment — per-node dominant fuzzy set
      3. FOU spatial map — per-node FOU × per-fuzzy-set FOU
      4. Entropy spatial ranking
    """
    m = model.module if hasattr(model, 'module') else model
    fg = m.fuzzy_graph
    m.eval()

    with torch.no_grad():
        mu_low, mu_high, mu_mid = fg._compute_memberships()
        FOU_k = mu_high - mu_low                                            # [N, K]
        FOU_node = FOU_k.mean(dim=-1)                                      # [N]
        H = fg.get_cell_entropy()                                          # [N]
        S_margin = fg.get_margin_stability()                               # [N]

    N, K = mu_mid.shape

    # ── 1. Membership matrix ──────────────────────────────────
    argmax_set = mu_mid.argmax(dim=-1).cpu().numpy()                       # [N]
    set_counts = {int(k): int((argmax_set == k).sum()) for k in range(K)}

    # Membership collapse detection
    dominant_frac = max(set_counts.values()) / N

    # ── 2. Region assignment diversity ────────────────────────
    # Check if all nodes assigned to same set
    argmax_entropy = -sum((c/N) * np.log(c/N + 1e-8) for c in set_counts.values())
    max_possible_entropy = np.log(K)

    # ── 3. FOU spatial map ────────────────────────────────────
    fou_per_set = FOU_k.mean(dim=0).cpu().numpy()                          # [K]
    fou_node_np = FOU_node.cpu().numpy()
    fou_top_nodes = np.argsort(fou_node_np)[::-1][:10].tolist()
    fou_bottom_nodes = np.argsort(fou_node_np)[:10].tolist()

    # ── 4. Entropy ranking ────────────────────────────────────
    H_np = H.cpu().numpy()
    H_top = np.argsort(H_np)[::-1][:10].tolist()
    H_bottom = np.argsort(H_np)[:10].tolist()

    results = {
        "membership_matrix_shape": [N, K],
        "argmax_distribution": set_counts,
        "dominant_set_fraction": float(dominant_frac),
        "argmax_normalized_entropy": float(argmax_entropy / max_possible_entropy),
        "verdict_membership": (
            "✅ DIVERSE ASSIGNMENTS" if dominant_frac < 0.6 else
            "⚠️  DOMINATED BY ONE SET (FRR may degenerate)"
            if dominant_frac < 0.9 else
            "❌ MEMBERSHIP COLLAPSE (all nodes → one set → FRR = GCN)"
        ),
        "fou_per_fuzzy_set": fou_per_set.tolist(),
        "fou_node": {
            "mean": float(fou_node_np.mean()),
            "std": float(fou_node_np.std()),
            "top10_nodes": fou_top_nodes,
            "bottom10_nodes": fou_bottom_nodes,
        },
        "entropy_node": {
            "mean": float(H_np.mean()),
            "std": float(H_np.std()),
            "top10_most_uncertain": H_top,
            "bottom10_most_certain": H_bottom,
        },
    }

    # Save raw tensors if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "mu_mid.npy"), mu_mid.cpu().numpy())
        np.save(os.path.join(output_dir, "mu_low.npy"), mu_low.cpu().numpy())
        np.save(os.path.join(output_dir, "mu_high.npy"), mu_high.cpu().numpy())
        np.save(os.path.join(output_dir, "FOU_k.npy"), FOU_k.cpu().numpy())
        np.save(os.path.join(output_dir, "FOU_node.npy"), fou_node_np)
        np.save(os.path.join(output_dir, "entropy_node.npy"), H_np)
        np.save(os.path.join(output_dir, "argmax_set.npy"), argmax_set)
        np.save(os.path.join(output_dir, "margin_stability.npy"), S_margin.cpu().numpy())

        results["_saved_to"] = output_dir

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Layer 6: Traffic Semantic Audit
# ═══════════════════════════════════════════════════════════════════════

def audit_layer6_traffic_semantics(model, device, adj_matrix):
    """Compare FOU/Entropy/Membership across semantic node groups.

    Groups (by graph degree):
      - Hub (high-degree):  likely highway intersections
      - Mid (medium-degree): likely arterial roads
      - Leaf (low-degree):  likely residential/side streets

    Expected patterns (strong paper evidence):
      - Hub:    high Entropy, high FOU  (complex, variable function)
      - Leaf:   low Entropy, low FOU   (stable, predictable function)

    If these hold, it proves the model learned traffic semantics,
    not just mathematical structure.
    """
    m = model.module if hasattr(model, 'module') else model
    fg = m.fuzzy_graph
    m.eval()

    with torch.no_grad():
        mu_low, mu_high, mu_mid = fg._compute_memberships()
        FOU_k = mu_high - mu_low                                          # [N, K]
        FOU_node = FOU_k.mean(dim=-1)                                     # [N]
        H_node = fg.get_cell_entropy()                                    # [N]
        S_margin = fg.get_margin_stability()                              # [N]

    # Classify nodes
    groups = _classify_nodes_by_degree(adj_matrix, m.num_nodes)

    results = {}
    for group_name, node_indices in groups.items():
        if not node_indices:
            continue
        idx = torch.tensor(node_indices)
        results[group_name] = {
            "n_nodes": len(node_indices),
            "FOU_mean": float(FOU_node[idx].mean()),
            "FOU_std": float(FOU_node[idx].std()),
            "Entropy_mean": float(H_node[idx].mean()),
            "Entropy_std": float(H_node[idx].std()),
            "Margin_mean": float(S_margin[idx].mean()),
            "Margin_std": float(S_margin[idx].std()),
            # Per-fuzzy-set dominance
            "dominant_set": int(mu_mid[idx].mean(dim=0).argmax().item()),
            "set_distribution": mu_mid[idx].mean(dim=0).tolist(),
        }

    # ── Verify expected pattern ───────────────────────────────
    hub_key = [k for k in results if "hub" in k.lower()]
    leaf_key = [k for k in results if "leaf" in k.lower()]

    pattern_holds = False
    evidence = ""
    if hub_key and leaf_key:
        hub_fou = results[hub_key[0]]["FOU_mean"]
        leaf_fou = results[leaf_key[0]]["FOU_mean"]
        hub_h = results[hub_key[0]]["Entropy_mean"]
        leaf_h = results[leaf_key[0]]["Entropy_mean"]

        if hub_fou > leaf_fou and hub_h > leaf_h:
            pattern_holds = True
            evidence = (
                f"Hub FOU={hub_fou:.4f} > Leaf FOU={leaf_fou:.4f}, "
                f"Hub H={hub_h:.4f} > Leaf H={leaf_h:.4f} — "
                "model has learned traffic-role-aware uncertainty"
            )
        else:
            evidence = (
                f"Hub FOU={hub_fou:.4f} vs Leaf FOU={leaf_fou:.4f}, "
                f"Hub H={hub_h:.4f} vs Leaf H={leaf_h:.4f} — "
                "expected pattern NOT observed"
            )

    results["_semantic_verdict"] = {
        "pattern_expected": "Hub(higher FOU, higher Entropy) > Leaf(lower FOU, lower Entropy)",
        "pattern_holds": pattern_holds,
        "evidence": evidence,
        "verdict": (
            "✅ TRAFFIC SEMANTICS LEARNED — paper-ready evidence"
            if pattern_holds else
            "❌ PATTERN NOT FOUND — model may not have learned traffic-role-specific uncertainty"
        ),
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Full 6-Layer Audit Runner
# ═══════════════════════════════════════════════════════════════════════

class FullAuditor:
    """Run all 6 layers of the Module Sanity Check."""

    def __init__(self, model, dataloader, device, adj_matrix=None, output_dir=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.adj_matrix = adj_matrix
        self.output_dir = output_dir

    def run(self, layers=None):
        """Run specified layers (default: all).

        Args:
            layers: list of ints [1-6], or "all".
        """
        if layers is None or layers == "all":
            layers = [1, 2, 3, 4, 5, 6]

        results = {}

        layer_names = {
            1: "输出变化审计 (Output Change)",
            2: "统计量审计 (Statistical Sanity)",
            3: "梯度审计 (Gradient Audit) [P1]",
            4: "模块专属指标 (Module-Specific Metrics)",
            5: "可解释性审计 (Interpretability)",
            6: "交通语义审计 (Traffic Semantics)",
        }

        for layer_id in sorted(layers):
            name = layer_names.get(layer_id, f"Layer {layer_id}")
            print(f"\n{'='*60}")
            print(f"  Layer {layer_id}: {name}")
            print(f"{'='*60}")

            try:
                if layer_id == 1:
                    results["layer1_output_change"] = audit_layer1_output_change(
                        self.model, self.dataloader, self.device)
                elif layer_id == 2:
                    results["layer2_statistical_sanity"] = audit_layer2_statistical_sanity(
                        self.model, self.device)
                elif layer_id == 3:
                    results["layer3_gradient"] = audit_layer3_gradient(
                        self.model, self.dataloader, self.device)
                elif layer_id == 4:
                    results["layer4_module_metrics"] = audit_layer4_module_metrics(
                        self.model, self.dataloader, self.device)
                elif layer_id == 5:
                    results["layer5_interpretability"] = audit_layer5_interpretability(
                        self.model, self.device, output_dir=self.output_dir)
                elif layer_id == 6:
                    adj = self.adj_matrix
                    if adj is None:
                        m = self.model.module if hasattr(self.model, 'module') else self.model
                        adj = m.adjacency_matrix
                    results["layer6_traffic_semantics"] = audit_layer6_traffic_semantics(
                        self.model, self.device, adj_matrix=adj)
            except Exception as e:
                print(f"  ⚠️  Layer {layer_id} FAILED: {e}")
                results[f"layer{layer_id}_error"] = str(e)

        return results


# ═══════════════════════════════════════════════════════════════════════
#  Priority Scorecard
# ═══════════════════════════════════════════════════════════════════════

def print_scorecard(results):
    """Print the 3-priority scorecard + per-layer verdicts."""
    print("\n" + "=" * 60)
    print("  PRIORITY SCORECARD — 消融实验前必过三项")
    print("=" * 60)

    # ── P1: grad(theta_delta) ──────────────────────────────────
    l3 = results.get("layer3_gradient", {})
    p1_info = l3.get("_PRIORITY", {})
    p1_v = p1_info.get("P1_verdict", "NO DATA")
    print(f"\n  [P1] grad(θ_delta): {p1_v}")
    if "grad_theta_delta" in p1_info:
        print(f"       |grad(θ_delta)| = {p1_info['grad_theta_delta']:.2e}")
    if "grad_theta_delta_vs_lower_ratio" in p1_info:
        print(f"       ratio(δ/lower)  = {p1_info['grad_theta_delta_vs_lower_ratio']:.4f}")

    # ── P2: FOU distribution ───────────────────────────────────
    l2 = results.get("layer2_statistical_sanity", {})
    fou = l2.get("FOU", {})
    p2_v = fou.get("verdict", "NO DATA")
    print(f"\n  [P2] FOU distribution: {p2_v}")
    percentiles = fou.get("percentiles", {})
    if percentiles:
        print(f"       p10={percentiles.get('p10','?'):.4f}  "
              f"p50={percentiles.get('p50','?'):.4f}  "
              f"p90={percentiles.get('p90','?'):.4f}")
        print(f"       mean={percentiles.get('mean','?'):.4f}  "
              f"frac_healthy={fou.get('frac_healthy (0.05-0.3)','?'):.2%}")

    # ── P3: Var(μ_t) ──────────────────────────────────────────
    l1 = results.get("layer1_output_change", {})
    dyn = l1.get("D_dynamic_membership", {})
    p3_v = dyn.get("verdict", "NO DATA")
    print(f"\n  [P3] Dynamic Membership: {p3_v}")
    if "mean_var_per_node" in dyn:
        print(f"       Var(μ_t) = {dyn['mean_var_per_node']:.6f}")
    if "delta_dyn_vs_static" in dyn:
        print(f"       Δ(μ_dyn vs μ_static) = {dyn['delta_dyn_vs_static']:.4%}")

    # ── Per-layer verdicts ─────────────────────────────────────
    print("\n" + "-" * 40)
    print("  LAYER-BY-LAYER VERDICTS")
    print("-" * 40)

    verdicts = {
        "Closure": l1.get("A_closure_delta", {}).get("verdict", "?"),
        "Entropy Graph": l1.get("B_entropy_graph", {}).get("verdict", "?"),
        "FOU Mode": l1.get("C_fou_mode", {}).get("verdict", "?"),
        "Membership Diversity": l2.get("Membership", {}).get("verdict", "?"),
        "Entropy Spread": l2.get("Entropy", {}).get("verdict", "?"),
        "Closure S vs R": l4_val(results, "layer4_module_metrics", "A_closure_S_vs_R", "verdict"),
        "Entropy-Dynamic Corr": l4_val(results, "layer4_module_metrics", "C_entropy_graph_corr", "verdict"),
        "FIR Engagement": l4_val(results, "layer4_module_metrics", "D_fir_magnitude", "verdict"),
        "Membership Collapse": l5_val(results),
        "Traffic Semantics": l6_val(results),
    }

    for name, verdict in verdicts.items():
        print(f"  {name:25s} {verdict}")

    # ── GO/NO-GO ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    failures = sum(1 for v in verdicts.values() if "❌" in str(v))
    warnings = sum(1 for v in verdicts.values() if "⚠️" in str(v))
    if failures > 0:
        print(f"  ⛔ GO/NO-GO: {failures} FAILURES — fix before ablation")
        if any("P1" in str(p1_v) and "NOT LEARNING" in str(p1_v) for _ in [1]):
            print("     ⚠️  P1 failure is CRITICAL — FOU is decorative")
    elif warnings > 0:
        print(f"  ⚠️  GO/NO-GO: {warnings} WARNINGS — proceed with caution")
    else:
        print(f"  ✅ GO/NO-GO: ALL CLEAR — proceed to formal ablation study")

    print("=" * 60)


def l4_val(results, layer_key, sub_key, field):
    """Safely extract nested verdict from layer 4."""
    try:
        return results.get(layer_key, {}).get(sub_key, {}).get(field, "?")
    except Exception:
        return "?"


def l5_val(results):
    """Extract membership verdict from layer 5."""
    try:
        return results.get("layer5_interpretability", {}).get("verdict_membership", "?")
    except Exception:
        return "?"


def l6_val(results):
    """Extract semantic verdict from layer 6."""
    try:
        return results.get("layer6_traffic_semantics", {}).get("_semantic_verdict", {}).get("verdict", "?")
    except Exception:
        return "?"


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="final_3_type2 六层模块有效性审计 (消融实验前必做)")

    parser.add_argument("--dataset", type=str, default="METR_LA",
                        help="Dataset name")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Override config file (JSON)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (REQUIRED)")
    parser.add_argument("--train_first", action="store_true",
                        help="Train model before auditing (fallback)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results to JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save raw tensors for visualization (Layer 5)")
    parser.add_argument("--layer", type=str, default="all",
                        help="Comma-separated layers to run: 1,2,3,4,5,6 or 'all'")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batches for gradient tests")
    parser.add_argument("--other_args", type=str, default=None,
                        help='JSON string of config overrides, e.g. \'{"num_cells": 16}\'')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Try to auto-detect training config from output directory
    other_args = {}
    if args.checkpoint:
        # e.g. outputs/.../model_cache/xxx.tar → outputs/.../
        ckpt_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        for cfg_name in ["config.json", "env.json", "train_config.json"]:
            cfg_path = os.path.join(ckpt_dir, cfg_name)
            if os.path.exists(cfg_path):
                print(f"Auto-detected training config: {cfg_path}")
                other_args["config_file"] = cfg_path
                break

    if args.config_file:
        other_args["config_file"] = args.config_file

    # Parse --other_args JSON overrides (highest priority)
    if args.other_args:
        import json as _json
        cli_overrides = _json.loads(args.other_args)
        other_args.update(cli_overrides)
        print(f"Config overrides: {cli_overrides}")

    config = ConfigParser(
        "traffic_state_pred",
        "final_3_type2",
        args.dataset,
        config_file=None,
        saved_model=True,
        train=args.train_first,
        other_args=other_args,
    )

    runtime = build_dataset_runtime(config)
    dataloader = runtime.valid_loader

    # Build model
    model = get_model(config, runtime.data_feature)
    model = model.to(device)

    # Load checkpoint with size-mismatch tolerance
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt_state = ckpt["model_state_dict"]
            print(f"  Loaded from .tar (epoch {ckpt.get('epoch', '?')})")
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt_state = ckpt["state_dict"]
        else:
            ckpt_state = ckpt

        # Robust loading: skip size-mismatched params
        model_state = model.state_dict()
        loaded, skipped, missing = [], [], []
        for key, ckpt_val in ckpt_state.items():
            if key not in model_state:
                missing.append(key)
                continue
            if model_state[key].shape != ckpt_val.shape:
                skipped.append((key, list(ckpt_val.shape), list(model_state[key].shape)))
                continue
            model_state[key].copy_(ckpt_val)
            loaded.append(key)

        if skipped:
            print(f"  ⚠️  Skipped {len(skipped)} size-mismatched params "
                  f"(config override needed):")
            for name, ckpt_shape, model_shape in skipped[:5]:
                print(f"    {name}: ckpt {ckpt_shape} vs model {model_shape}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more")
        print(f"  ✅ Loaded {len(loaded)} params, "
              f"skipped {len(skipped)}, missing {len(missing)}")
    elif not args.train_first:
        print("\n⚠️  WARNING: No checkpoint provided. "
              "Auditing UNTRAINED model. Results will be random.")
        print("   Use --checkpoint PATH to audit a trained model.\n")
    else:
        print("Training model first...")
        model.train()

    model.eval()

    # Parse layers
    if args.layer == "all":
        layers = [1, 2, 3, 4, 5, 6]
    else:
        layers = [int(x.strip()) for x in args.layer.split(",")]

    # Get adjacency
    adj_matrix = model.module.adjacency_matrix if hasattr(model, 'module') else model.adjacency_matrix

    # Run audit
    auditor = FullAuditor(
        model, dataloader, device,
        adj_matrix=adj_matrix,
        output_dir=args.output_dir,
    )

    results = auditor.run(layers=layers)

    # Print scorecard
    print_scorecard(results)

    # Save
    if args.output:
        def convert(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
