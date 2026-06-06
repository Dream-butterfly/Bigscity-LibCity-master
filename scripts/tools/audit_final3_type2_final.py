"""
final_3_type2 最终架构冻结审计 — 7 项终极指标。

在大规模消融实验前执行的最后一轮检查。全部通过 → 冻结架构 → 进入正式实验。

七项指标:
  M1: Membership Entropy — 隶属度是否真塌缩？
  M2: Pairwise Cosine(μ) — 节点是否同质化？
  M3: Routing Entropy — FRR 是否退化？
  M4: FOU-Error Correlation — FOU 是否有预测语义？
  M5: New Closure Edges — Closure 是否产生新的推理边？
  M6: Set Utilization — 模糊集合是否都被使用？
  M7: Membership Peak — 节点是否真正 committed 到特定集合？

用法:
  python scripts/tools/audit_final3_type2_final.py \
      --dataset PEMSD4 \
      --config_file train_config_PEMSD4.json \
      --checkpoint outputs/.../model_cache/final_3_type2_PEMSD4_epoch49.tar \
      --output audit_final_PEMSD4.json
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
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data import build_dataset_runtime
from GNNTP.utils import get_model


# ═══════════════════════════════════════════════════════════════════════
#  Shared model loading (same robust logic as v2)
# ═══════════════════════════════════════════════════════════════════════

def load_model_robust(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    other_args = {}
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
        print(f"  Merged {len(other_args)} config keys from {args.config_file}")

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

    # Build model
    model = get_model(config, runtime.data_feature).to(device)

    # Load checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print("❌ No checkpoint found. Aborting.")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_state = (ckpt.get("model_state_dict") or
                  ckpt.get("state_dict") or ckpt)

    # Auto-patch input_dim mismatch
    _ip_key = "condition_encoder.input_projection.weight"
    if _ip_key in ckpt_state:
        ckpt_in_dim = ckpt_state[_ip_key].shape[1]
        model_in_dim = model.condition_encoder.input_projection.weight.shape[1]
        if ckpt_in_dim != model_in_dim:
            print(f"  🔧 Patching input_dim: {model_in_dim} → {ckpt_in_dim}")
            # Patch input_projection
            model.condition_encoder.input_projection = torch.nn.Linear(
                ckpt_in_dim,
                model.condition_encoder.input_projection.out_features,
                bias=True,
            ).to(device)
            model.condition_encoder.input_projection.weight.data.copy_(
                ckpt_state[_ip_key])
            model.condition_encoder.input_projection.bias.data.copy_(
                ckpt_state.get(_ip_key.replace(".weight", ".bias"),
                               torch.zeros(ckpt_in_dim, device=device)))

            # Patch raw_projection
            _rp_key = "fuzzy_graph.raw_projection.weight"
            if _rp_key in ckpt_state:
                model.fuzzy_graph.raw_projection = torch.nn.Linear(
                    ckpt_in_dim,
                    model.fuzzy_graph.raw_projection.out_features,
                    bias=True,
                ).to(device)
                model.fuzzy_graph.raw_projection.weight.data.copy_(
                    ckpt_state[_rp_key])
                model.fuzzy_graph.raw_projection.bias.data.copy_(
                    ckpt_state.get(_rp_key.replace(".weight", ".bias"),
                                   torch.zeros(ckpt_in_dim, device=device)))

            # Patch encode_condition to auto-trim
            _orig_encode = model.encode_condition
            def _trimmed_encode(hist):
                if hist.shape[-1] > ckpt_in_dim:
                    hist = hist[..., :ckpt_in_dim]
                return _orig_encode(hist)
            model.encode_condition = _trimmed_encode
            model._trim_dim = ckpt_in_dim

    # Load all params
    model_state = model.state_dict()
    loaded = 0
    skipped_shape = []
    skipped_missing = []
    for key, val in ckpt_state.items():
        if key in model_state:
            if model_state[key].shape == val.shape:
                model_state[key].copy_(val)
                loaded += 1
            else:
                skipped_shape.append(
                    f"    {key}: ckpt{tuple(val.shape)} vs model{tuple(model_state[key].shape)}")
        else:
            skipped_missing.append(f"    {key}")

    # Also check: params in model but NOT in checkpoint
    ckpt_keys = set(ckpt_state.keys())
    model_keys = set(model_state.keys())
    missing_from_ckpt = model_keys - ckpt_keys - {
        'fuzzy_graph.static_adjacency'}  # buffer, not param

    total = len(model_state)
    print(f"  ✅ Loaded {loaded}/{total} params"
          + (f" (skipped {total - loaded})" if loaded < total else ""))

    if skipped_shape:
        print(f"  ⚠️  {len(skipped_shape)} params SKIPPED (shape mismatch):")
        for s in skipped_shape:
            print(s)

    if skipped_missing:
        print(f"  ⚠️  {len(skipped_missing)} ckpt keys NOT in model:")
        for s in skipped_missing:
            print(s)

    if missing_from_ckpt:
        print(f"  ⚠️  {len(missing_from_ckpt)} model keys NOT in checkpoint (random init):")
        for k in sorted(missing_from_ckpt)[:10]:
            print(f"    {k}: shape={tuple(model_state[k].shape)}")
        if len(missing_from_ckpt) > 10:
            print(f"    ... and {len(missing_from_ckpt)-10} more")

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
#  M1: Membership Entropy
# ═══════════════════════════════════════════════════════════════════════

def metric_membership_entropy(model):
    """Per-node Shannon entropy of fuzzy membership distribution.

    If ALL nodes have the SAME entropy → membership collapse is real.
    If entropy varies widely across nodes → model has learned heterogeneous
    memberships.
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    with torch.no_grad():
        mu_low, mu_high, mu_mid = fg._compute_memberships()
        # Shannon entropy on (static) midpoint membership
        p = mu_mid / mu_mid.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        H = -(p * (p + 1e-8).log()).sum(dim=-1)  # [N]

    H_np = H.cpu().numpy()
    H_max = np.log(m.fuzzy_num_sets)

    # Diversity metrics
    cv = H_np.std() / (H_np.mean() + 1e-8)  # coefficient of variation

    results = {
        "H_mean": float(H_np.mean()),
        "H_std": float(H_np.std()),
        "H_min": float(H_np.min()),
        "H_max": float(H_np.max()),
        "H_normalized_mean": float(H_np.mean() / H_max),
        "coefficient_of_variation": float(cv),
        "frac_low_entropy (<0.1*H_max)": float((H_np < 0.1 * H_max).mean()),
        "frac_high_entropy (>0.8*H_max)": float((H_np > 0.8 * H_max).mean()),
    }

    # Verdict
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
#  M2: Pairwise Cosine(μ)
# ═══════════════════════════════════════════════════════════════════════

def metric_pairwise_cosine(model):
    """Average pairwise cosine similarity of node membership vectors.

    If ≈1.0 → all nodes have identical membership → collapse.
    If ≪0.5 → nodes are genuinely heterogeneous.
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    with torch.no_grad():
        _, _, mu_mid = fg._compute_memberships()

    # Normalize rows
    mu_norm = F.normalize(mu_mid, p=2, dim=-1)  # [N, K]

    # Compute all-pairs cosine (memory-efficient for N < 500)
    N = mu_norm.shape[0]
    if N <= 500:
        sim_full = mu_norm @ mu_norm.T  # [N, N]
        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(N, dtype=torch.bool, device=mu_norm.device)
        sim_mean = sim_full[mask].mean().item()
        sim_std = sim_full[mask].std().item()
        sim_p90 = sim_full[mask].quantile(0.9).item()
    else:
        # Random sample of 5000 pairs for large graphs
        idx_i = torch.randint(0, N, (5000,), device=mu_norm.device)
        idx_j = torch.randint(0, N, (5000,), device=mu_norm.device)
        mask = idx_i != idx_j
        sims = (mu_norm[idx_i] * mu_norm[idx_j]).sum(dim=-1)[mask]
        sim_mean = sims.mean().item()
        sim_std = sims.std().item()
        sim_p90 = sims.quantile(0.9).item()

    results = {
        "mean_cosine": float(sim_mean),
        "std_cosine": float(sim_std),
        "p90_cosine": float(sim_p90),
        "num_nodes": N,
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
#  M3: Routing Entropy (FRR)
# ═══════════════════════════════════════════════════════════════════════

def metric_routing_entropy(model, dataloader, device):
    """Entropy of routing weights in FRR's CellAttention blocks.

    Captures intermediate routing weights B = sqrt(u)/||sqrt(u)|| from the
    first encoder block on a single batch.

    If low entropy → all nodes routed to same region → FRR ≈ GCN.
    """
    m = _unwrap(model)
    m.eval()

    # Get one batch
    batch = next(iter(dataloader))
    batch.to_tensor(device)
    history = _trim(batch["X"], model)

    # Get mu_fuzzy for routing
    fg = m.fuzzy_graph
    with torch.no_grad():
        mu_fuzzy = fg.get_memberships(history).to(device)

    # Collect routing weights from each CellAttention block
    routing_entropies = []
    routing_max_counts = []

    for block_list in [m.condition_encoder.blocks, m.future_decoder.blocks]:
        for blk in block_list:
            if not hasattr(blk, 'cell_attention'):
                continue
            ca = blk.cell_attention
            with torch.no_grad():
                # Get node representations for this block
                # We need to run a partial forward to get the pre-FRR features
                # Use the mean feature as proxy
                x = m.condition_encoder.input_projection(history)
                node_repr = x.mean(dim=(0, 1))  # [N, D]

                # Compute routing membership u = _compute_membership(x)
                if ca.use_fuzzy_routing and ca.fuzzy_to_cell is not None:
                    u_fuzzy = ca._compute_membership(node_repr, mu_fuzzy)
                else:
                    u_fuzzy = ca._compute_membership(node_repr)

                # B = sqrt(u) / ||sqrt(u)||
                B = u_fuzzy.sqrt()
                B = B / B.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)  # [N, K_c]

                # Per-node routing entropy
                H_route = -(B * (B + 1e-8).log()).sum(dim=-1)  # [N]
                routing_entropies.append(H_route.mean().item())

                # Which region gets the most nodes?
                argmax_counts = B.argmax(dim=-1).bincount(minlength=ca.num_cells).float()
                max_frac = (argmax_counts.max() / argmax_counts.sum()).item()
                routing_max_counts.append(max_frac)

    if not routing_entropies:
        return {"_verdict": "⚠️  NO CellAttention blocks found"}

    H_max = np.log(m.num_cells)
    avg_entropy = float(np.mean(routing_entropies))
    avg_max_frac = float(np.mean(routing_max_counts))

    results = {
        "avg_routing_entropy": avg_entropy,
        "normalized_entropy": avg_entropy / H_max,
        "max_entropy_possible": float(H_max),
        "avg_max_region_fraction": avg_max_frac,
        "num_blocks_measured": len(routing_entropies),
        "per_block_entropy": [float(e) for e in routing_entropies],
        "per_block_max_frac": [float(f) for f in routing_max_counts],
    }

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
#  M4: FOU-Error Correlation
# ═══════════════════════════════════════════════════════════════════════

def metric_fou_error_correlation(model, dataloader, device, num_batches=20):
    """Per-node FOU vs per-node prediction error correlation.

    If FOU is positively correlated with error → model knows where it's
    uncertain → FOU has genuine predictive semantics.
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    m.eval()

    # Per-node FOU from static membership
    with torch.no_grad():
        mu_low, mu_high, _ = fg._compute_memberships()
        fou_node = (mu_high - mu_low).mean(dim=-1).cpu().numpy()  # [N]

    # Per-node error accumulation
    all_errors = []
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
            abs_err = (pred - future).abs().mean(dim=(0, 1, -1)).cpu().numpy()
        all_errors.append(abs_err)
        count += 1

    mean_error = np.stack(all_errors, axis=0).mean(axis=0)  # [N]
    r, p = pearsonr(fou_node, mean_error)

    # Also check per-fuzzy-set FOU vs per-set prediction error
    fou_per_set = (mu_high - mu_low).mean(dim=0).cpu().numpy()  # [K]
    # For per-set, we correlate with the mean error of nodes dominated by that set
    with torch.no_grad():
        _, _, mu_mid = fg._compute_memberships()
        argmax_set = mu_mid.argmax(dim=-1).cpu().numpy()  # [N]
    set_errors = []
    set_fous = []
    for k in range(m.fuzzy_num_sets):
        mask = argmax_set == k
        if mask.sum() > 0:
            set_errors.append(mean_error[mask].mean())
            set_fous.append(fou_per_set[k])
    r_set, p_set = pearsonr(set_fous, set_errors) if len(set_errors) >= 3 else (0.0, 1.0)

    results = {
        "per_node": {
            "pearson_r": float(r),
            "p_value": float(p),
            "is_significant": p < 0.05,
        },
        "per_fuzzy_set": {
            "pearson_r": float(r_set),
            "p_value": float(p_set),
            "is_significant": p_set < 0.05,
        },
        "fou_mean": float(fou_node.mean()),
        "fou_std": float(fou_node.std()),
        "error_mean": float(mean_error.mean()),
        "num_batches_evaluated": count,
    }

    # Verdict
    if r > 0.3 and p < 0.05:
        verdict = (f"✅ STRONG — r={r:.3f} (p={p:.3f}), "
                   f"FOU captures prediction uncertainty")
    elif r > 0.1 and p < 0.1:
        verdict = f"⚠️  WEAK — r={r:.3f} (p={p:.3f}), marginal signal"
    elif r > 0:
        verdict = f"⚠️  VERY WEAK — r={r:.3f} (p={p:.3f}), not statistically reliable"
    else:
        verdict = f"❌ NEGATIVE or ZERO — r={r:.3f}, FOU has no error semantics"

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M5: New Closure Edges
# ═══════════════════════════════════════════════════════════════════════

def metric_new_closure_edges(model, dataloader, device,
                              thresholds=None):
    """Multi-threshold closure edge analysis.

    Single-threshold reporting (e.g. R > 0.1) is misleading for sigmoid-based
    fuzzy relations since R[i,j] > 0 always.  Instead, report density and new
    edges at multiple thresholds (0.1, 0.3, 0.5, 0.7) to reveal the true
    contribution of semantic closure.

    Key insight: if closure is working, we expect:
      density(S, 0.5) > density(R, 0.5)  ← closure lifts edges above threshold
    even if density(R, 0.1) ≈ 100%.
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7]

    m = _unwrap(model)
    fg = m.fuzzy_graph

    batch = next(iter(dataloader))
    batch.to_tensor(device)
    history = _trim(batch["X"], model)

    with torch.no_grad():
        mu_low, mu_high, _ = fg._compute_memberships(history)
        R_low, R_high = fg._build_fuzzy_relation_t2(mu_low, mu_high)
        S_high = fg._compute_closure(R_high, m.semantic_closure_hops)

    N = R_high.shape[0]
    total_pairs = N * N
    R = R_high.cpu()
    S = S_high.cpu()

    # Per-threshold analysis
    threshold_data = {}
    for t in thresholds:
        r_mask = R >= t
        s_mask = S >= t
        new_mask = (~r_mask) & s_mask          # R below but S above
        strengthen_mask = r_mask & (S >= R + 0.05)  # already above, significantly lifted

        threshold_data[f"threshold_{t}"] = {
            "R_density": float(r_mask.float().mean()),
            "S_density": float(s_mask.float().mean()),
            "new_edges": int(new_mask.float().sum()),
            "new_edges_pct": float(new_mask.float().mean() * 100),
            "strengthened_edges": int(strengthen_mask.float().sum()),
            "strengthened_pct": float(strengthen_mask.float().mean() * 100),
            "mean_delta": float((S - R)[r_mask].mean()) if r_mask.any() else 0.0,
        }

    # Verdict based on threshold=0.5 (most informative for sigmoid relations)
    t50 = threshold_data["threshold_0.5"]
    if t50["new_edges_pct"] > 5.0:
        verdict = (f"✅ SIGNIFICANT — {t50['new_edges_pct']:.1f}% new edges "
                   f"at threshold=0.5, closure produces genuine inference")
    elif t50["new_edges_pct"] > 1.0:
        verdict = (f"⚠️  MARGINAL — {t50['new_edges_pct']:.1f}% new edges "
                   f"at threshold=0.5")
    elif any(threshold_data[f"threshold_{t}"]["new_edges_pct"] > 5.0
             for t in thresholds if t >= 0.3):
        verdict = "⚠️  WEAK — new edges only appear at low thresholds"
    else:
        verdict = "❌ NEGLIGIBLE — no meaningful new edges at any threshold"

    results = {
        "total_pairs": total_pairs,
        "by_threshold": threshold_data,
        "_verdict": verdict,
    }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  M6: Set Utilization
# ═══════════════════════════════════════════════════════════════════════

def metric_set_utilization(model):
    """Per-fuzzy-set utilization: are all K sets being used?

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

    # Diversity: how uniform is set utilization?
    u_norm = u_k / u_k.sum()
    H_util = -(u_norm * np.log(u_norm + 1e-8)).sum()
    H_max = np.log(m.fuzzy_num_sets)
    normalized_util_entropy = H_util / H_max

    # Worst-case dominance
    max_util = float(u_k.max())
    min_util = float(u_k.min())

    results = {
        "per_set_utilization": {f"set_{i}": float(u_k[i]) for i in range(len(u_k))},
        "max_utilization": max_util,
        "min_utilization": min_util,
        "utilization_entropy": float(normalized_util_entropy),
        "mean_raw_membership": float(global_mean),
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
#  M7: Membership Peak Analysis
# ═══════════════════════════════════════════════════════════════════════

def metric_membership_peak(model):
    """Per-node peak membership analysis.

    Answers the key question raised by M6 vs M1/M2 tension:
    "Are all K sets being used, but each node spreads thin across ALL of them?"

    Two diagnostics:
      1. top1_mean: mean of max_k p(k|i).  With K=8:
         - uniform baseline ≈ 0.125  → no commitment
         - weak              ≈ 0.25   → thin spread (current suspected state)
         - moderate          ≈ 0.50   → emerging clusters
         - strong             > 0.70   → genuine fuzzy clustering
      2. argmax distribution: how nodes distribute across K peaks.
         - Uniform across K  → sets are all used (confirms M6)
         - Skewed            → some sets dead

    Combined with M6 utilization_entropy, this reveals whether
    "balanced utilization" is genuine clustering or just uniform
    thin membership spread.
    """
    m = _unwrap(model)
    fg = m.fuzzy_graph
    with torch.no_grad():
        _, _, mu_mid = fg._compute_memberships()

    # Normalize to probability simplex (same as M1)
    p = mu_mid / mu_mid.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    p_np = p.cpu().numpy()
    N, K = p_np.shape

    # ── Top-1 statistics ──
    top1_vals = p_np.max(axis=-1)  # [N]
    top1_mean = float(top1_vals.mean())
    top1_std = float(top1_vals.std())
    top1_min = float(top1_vals.min())
    top1_max = float(top1_vals.max())

    # Histogram (10 equal-width bins over [0, 1])
    hist, bin_edges = np.histogram(top1_vals, bins=10, range=(0.0, 1.0))

    # ── Argmax distribution ──
    argmax_set = p_np.argmax(axis=-1)  # [N]
    argmax_counts = np.bincount(argmax_set, minlength=K)  # [K]
    argmax_dist = argmax_counts / N

    # Gini of argmax distribution (low = uniform, high = skewed)
    argmax_gini = 1.0 - (argmax_dist ** 2).sum()
    argmax_gini_max = 1.0 - 1.0 / K
    argmax_gini_norm = float(argmax_gini / argmax_gini_max)

    # Per-set details
    per_set = {}
    for k in range(K):
        mask = argmax_set == k
        n_k = int(mask.sum())
        per_set[f"set_{k}"] = {
            "count": n_k,
            "pct": round(n_k / N * 100, 1),
            "mean_top1": round(float(top1_vals[mask].mean()), 4) if n_k > 0 else 0.0,
        }

    results = {
        "N_nodes": N,
        "K_sets": K,
        "top1_mean": top1_mean,
        "top1_std": top1_std,
        "top1_min": top1_min,
        "top1_max": top1_max,
        "top1_histogram": {
            "bin_edges": [round(float(e), 2) for e in bin_edges],
            "counts": [int(c) for c in hist],
        },
        "argmax_counts": {f"set_{k}": int(argmax_counts[k]) for k in range(K)},
        "argmax_gini_normalized": argmax_gini_norm,
        "per_set": per_set,
    }

    uniform_baseline = 1.0 / K

    if top1_mean > 0.7:
        verdict = (f"✅ STRONG PEAKS — top1_mean={top1_mean:.3f}, "
                   f"nodes are committed to specific fuzzy sets")
    elif top1_mean > 0.4:
        verdict = (f"⚠️  MODERATE — top1_mean={top1_mean:.3f}, "
                   f"some commitment but membership still diffuse")
    elif top1_mean > uniform_baseline * 1.5:
        verdict = (f"⚠️  WEAK — top1_mean={top1_mean:.3f}, "
                   f"barely above uniform baseline ({uniform_baseline:.3f})")
    else:
        verdict = (f"❌ UNIFORM — top1_mean={top1_mean:.3f}, "
                   f"membership is effectively flat across {K} sets")

    results["_verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_final_audit(model, dataloader, device, num_batches=20):
    print("\n" + "=" * 60)
    print("  M1: Membership Entropy")
    print("=" * 60)
    m1 = metric_membership_entropy(model)
    _print_metric(m1)

    print("\n" + "=" * 60)
    print("  M2: Pairwise Cosine(μ)")
    print("=" * 60)
    m2 = metric_pairwise_cosine(model)
    _print_metric(m2)

    print("\n" + "=" * 60)
    print("  M3: Routing Entropy (FRR)")
    print("=" * 60)
    m3 = metric_routing_entropy(model, dataloader, device)
    _print_metric(m3)

    print("\n" + "=" * 60)
    print("  M4: FOU-Error Correlation")
    print("=" * 60)
    m4 = metric_fou_error_correlation(model, dataloader, device, num_batches)
    _print_metric(m4)

    print("\n" + "=" * 60)
    print("  M5: New Closure Edges (multi-threshold)")
    print("=" * 60)
    m5 = metric_new_closure_edges(model, dataloader, device)
    _print_metric(m5)

    print("\n" + "=" * 60)
    print("  M6: Set Utilization")
    print("=" * 60)
    m6 = metric_set_utilization(model)
    _print_metric(m6)

    print("\n" + "=" * 60)
    print("  M7: Membership Peak Analysis")
    print("=" * 60)
    m7 = metric_membership_peak(model)
    _print_metric(m7)

    return {"M1_membership_entropy": m1,
            "M2_pairwise_cosine": m2,
            "M3_routing_entropy": m3,
            "M4_fou_error_corr": m4,
            "M5_closure_edges": m5,
            "M6_set_utilization": m6,
            "M7_membership_peak": m7}


def _print_metric(result):
    """Print key fields of a metric result with nested dict support."""
    for k, v in result.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            print(f"  {k:<35s} {v:.4f}")
        elif isinstance(v, dict):
            # Check if it's a nested dict-of-dicts (e.g. M5 by_threshold)
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
#  Scorecard
# ═══════════════════════════════════════════════════════════════════════

def print_scorecard(results):
    print("\n" + "=" * 60)
    print("  ARCHITECTURE FREEZE DECISION")
    print("=" * 60)

    m_labels = {
        "M1_membership_entropy": "Membership Entropy",
        "M2_pairwise_cosine": "Pairwise Cosine",
        "M3_routing_entropy": "Routing Entropy",
        "M4_fou_error_corr": "FOU-Error Corr",
        "M5_closure_edges": "New Closure Edges",
        "M6_set_utilization": "Set Utilization",
        "M7_membership_peak": "Membership Peak",
    }

    verdicts = {}
    for key, label in m_labels.items():
        v = results.get(key, {}).get("_verdict", "?")
        verdicts[label] = v
        print(f"\n  [{key[:2]}] {label}")
        print(f"      {v}")

    failures = sum(1 for v in verdicts.values() if "❌" in v)
    warnings = sum(1 for v in verdicts.values() if "⚠️" in v)

    print("\n" + "-" * 40)
    if failures == 0 and warnings == 0:
        print("  ✅ ALL 7 METRICS PASSED")
        print("  → FREEZE ARCHITECTURE")
        print("  → ENTER FORMAL EXPERIMENT PHASE")
    elif failures == 0 and warnings <= 2:
        print(f"  ⚠️  {warnings} WARNING(S), 0 FAILURES")
        print("  → ARCHITECTURE ACCEPTABLE")
        print("  → Proceed to experiments, note warnings in paper")
    elif failures >= 2:
        print(f"  ❌ {failures} FAILURES")
        print("  → DO NOT FREEZE — fix issues before experiments")
    else:
        print(f"  ⚠️  {failures} FAILURE, {warnings} WARNING(S)")
        print("  → Fix failures before freezing")
    print("-" * 40)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="final_3_type2 最终架构冻结审计 (7 metrics)")
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
