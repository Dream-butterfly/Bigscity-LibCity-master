#!/usr/bin/env python3
"""final_T2 Model Audit Script

Loads a trained checkpoint and runs comprehensive diagnostics on:
  1. Fuzzy set structure (membership, prototypes, σ/r distribution)
  2. Type-2 interval validity (R-gap, stability, FOU)
  3. Module marginal contributions (ablation ΔMAE)
  4. Soft ablation curves (λ-interpolation)
  5. Parameter utilization audit

Usage:
    uv run scripts/audit/audit_final_T2.py --run_dir outputs/20260609_023341__traffic_state_pred__final_T2__METR_LA
    uv run scripts/audit/audit_final_T2.py --run_dir <path> --epoch 28 --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_run(run_dir: str, epoch: int | None, device: str):
    """Load model, config, and data from a training run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # ── 1. Load config ──
    config_path = run_path / "effective_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"effective_config.json not found in {run_dir}")
    with open(config_path) as f:
        config = json.load(f)
    config["device"] = torch.device(device)

    task   = config.get("task", "traffic_state_pred")
    model_name = config.get("model", "final_T2")
    dataset_name = config.get("dataset", "")

    # ── 2. Load run_meta for artifact binding ──
    run_meta_path = run_path / "run_meta.json"
    artifact_id = None
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        artifact_id = run_meta.get("artifact_id")

    # ── 3. Import model class ──
    from GNNTP.models.new.final_T2.model import NewFuzzyCellAttention

    # ── 4. Build minimal data_feature (needed for model init) ──
    from GNNTP.data.artifact_io import load_data_artifact as _load_artifact
    if artifact_id:
        artifact = _load_artifact(artifact_id)
        data_feature = artifact.get("data_feature", {})
    else:
        # Fallback: build from config
        num_nodes = {"PEMSD4": 307, "PEMSD7": 883, "METR_LA": 207,
                     "PEMS_BAY": 325, "PEMSD3": 358, "PEMSD8": 170}.get(dataset_name, 300)
        data_feature = {
            "feature_dim": config.get("input_window", 12),
            "output_dim": config.get("output_dim", 1),
            "adj_mx": np.eye(num_nodes, dtype=np.float32),
            "num_nodes": num_nodes,
            "scaler": None,
        }

    # ── 5. Create model ──
    model = NewFuzzyCellAttention(config, data_feature)
    model.to(device)
    model.eval()

    # ── 6. Load checkpoint ──
    cache_dir = run_path / "model_cache"
    if epoch is not None:
        ckpt_name = f"{model_name}_{dataset_name}_epoch{epoch}.tar"
    else:
        ckpt_name = f"{model_name}_{dataset_name}.m"
    ckpt_path = cache_dir / ckpt_name
    if not ckpt_path.exists():
        # Try the other format
        alt_ckpt = cache_dir / f"{model_name}_{dataset_name}.m"
        if alt_ckpt.exists():
            ckpt_path = alt_ckpt
        else:
            # List available checkpoints
            available = list(cache_dir.glob("*.tar")) + list(cache_dir.glob("*.m"))
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Available: {[p.name for p in available]}")
    state_dict, _ = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {ckpt_path}")

    return model, config, data_feature, run_path


def load_test_data(config: dict, data_feature: dict):
    """Load test data using data artifact or config."""
    from GNNTP.common import ConfigParser

    # Use the same config but in eval mode
    config_parser = ConfigParser(
        config.get("task", "traffic_state_pred"),
        config.get("model", "final_T2"),
        config.get("dataset", ""),
        config_file=None,
        saved_model=False,
        train=False,
    )
    # Merge our loaded config on top
    eval_config = dict(config_parser.config)
    eval_config.update(config)

    from GNNTP.data.artifact_io import load_data_artifact
    from GNNTP.data.runtime import build_artifact_runtime

    try:
        artifact_id = config.get("artifact_id") or ""
        runtime = build_artifact_runtime(
            eval_config,
            task=eval_config["task"],
            model_name=eval_config["model"],
            artifact_id=artifact_id,
            force_reuse=True,
        )
        return runtime.test_loader, runtime.artifact_meta
    except Exception as e:
        raise RuntimeError(
            f"Failed to load test data: {e}\n"
            f"Make sure the data artifact exists or use --dummy_data flag."
        )


def evaluate(model, dataloader, device) -> float:
    """Compute MAE on the given dataloader."""
    model.eval()
    total_err = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch.to_tensor(device)
            pred = model.predict(batch)
            target = batch["y"][..., : model.output_dim]
            total_err += (pred - target).abs().sum().item()
            count += target.numel()
    return total_err / count


# ═══════════════════════════════════════════════════════════════════
#  Audit Functions
# ═══════════════════════════════════════════════════════════════════

def audit_membership(model, test_loader, device):
    """Audit fuzzy membership structure."""
    model.eval()
    g = model.fuzzy_graph
    if g is None:
        return {"error": "no fuzzy graph"}

    with torch.no_grad():
        batch = next(iter(test_loader))
        batch.to_tensor(device)
        history = batch["X"]

        mu_lower, mu_upper, mu_mid = g._compute_memberships(history)
        mu = mu_mid  # [N, K]

    # Per-node entropy and margin
    ent = -(mu * (mu + 1e-8).log()).sum(dim=-1)
    top2 = mu.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]   # [N]

    # Dead sets (mean membership < 0.01)
    dead_mask = mu.mean(dim=0) < 0.01
    dead_sets = dead_mask.sum().item()

    # Hard assignment distribution
    hard_assign = mu.argmax(dim=-1)
    set_sizes = torch.bincount(hard_assign, minlength=g.num_fuzzy_sets)

    return {
        "entropy_mean": ent.mean().item(),
        "entropy_std": ent.std().item(),
        "margin_mean": margin.mean().item(),
        "margin_std": margin.std().item(),
        "margin_min": margin.min().item(),
        "dead_sets": dead_sets,
        "set_sizes": set_sizes.tolist(),
    }


def audit_prototypes(model):
    """Audit prototype quality."""
    g = model.fuzzy_graph
    if g is None:
        return {"error": "no fuzzy graph"}

    proto = g.prototype_center.detach()  # [K, D]
    K = proto.size(0)
    dist = torch.cdist(proto, proto)     # [K, K]

    eye = torch.eye(K, device=dist.device, dtype=torch.bool)
    inter = dist[~eye].mean().item()
    dist_no_diag = dist + torch.eye(K, device=dist.device) * 1e9
    nearest = dist_no_diag.min(dim=-1).values
    intra = nearest.mean().item()
    ratio = inter / (intra + 1e-8)

    return {
        "inter_mean": inter,
        "intra_mean": intra,
        "scatter_ratio": ratio,
        "min_proto_dist": nearest.min().item(),
        "max_proto_dist": dist.max().item(),
    }


def audit_type2_relations(model, test_loader, device):
    """Audit Type-2 interval relations: R-gap and stability."""
    g = model.fuzzy_graph
    if g is None:
        return {"error": "no fuzzy graph"}

    with torch.no_grad():
        batch = next(iter(test_loader))
        batch.to_tensor(device)
        history = batch["X"]

        mu_lower, mu_upper, mu_mid = g._compute_memberships(history)
        R_low  = g._build_fuzzy_relation(mu_lower)
        R_mid  = g._build_fuzzy_relation(mu_mid)
        R_high = g._build_fuzzy_relation(mu_upper)

    diff_lm = (R_low - R_mid).abs().mean().item()
    diff_hm = (R_high - R_mid).abs().mean().item()
    R_gap   = (R_high - R_low).abs().mean().item() / (R_mid.abs().mean().item() + 1e-8)

    def flat_cos(A, B):
        return F.cosine_similarity(A.flatten(), B.flatten(), dim=0).item()

    cos_lm = flat_cos(R_low, R_mid)
    cos_hm = flat_cos(R_high, R_mid)
    cos_lh = flat_cos(R_low, R_high)
    stability = (cos_lm + cos_hm + cos_lh) / 3.0

    sigma = F.softplus(g.log_sigma) + 1e-3
    r = torch.sigmoid(g.log_radius_ratio)

    beta = F.softmax(g.relation_mix_logits, dim=0)
    h_beta = -(beta * (beta + 1e-8).log()).sum().item()

    fou = (mu_upper - mu_lower).clamp(min=0.0).mean(dim=-1)

    # ── Edge analysis ──
    # R_mixed is the actual graph used in propagation
    beta = F.softmax(g.relation_mix_logits, dim=0)
    R_mixed = beta[0] * R_low + beta[1] * R_mid + beta[2] * R_high
    N = g.num_nodes
    total_edges = N * N

    def edge_stats(R, name):
        """Count effective edges above thresholds, per-node degree stats."""
        thresholds = [0.01, 0.05, 0.10, 0.30, 0.50]
        stats = {"name": name}
        # Effective edges (excluding self-loops, diag≈1)
        R_nodiag = R.clone()
        R_nodiag[range(N), range(N)] = 0
        for th in thresholds:
            count = (R_nodiag > th).sum().item()
            stats[f"edges>{th}"] = count
            stats[f"sparsity>{th}"] = count / (total_edges - N)  # exclude diag
        # Per-node degree (edges > 0.05)
        degree = (R_nodiag > 0.05).sum(dim=1).float()
        stats["degree_mean"] = degree.mean().item()
        stats["degree_std"] = degree.std().item()
        stats["degree_min"] = degree.min().item()
        stats["degree_max"] = degree.max().item()
        # Edge weight distribution
        stats["edge_mean"] = R_nodiag.mean().item()
        stats["edge_std"] = R_nodiag.std().item()
        return stats

    edge_info = {
        "R_mid": edge_stats(R_mid, "R_mid"),
        "R_mixed": edge_stats(R_mixed, "R_mixed"),
        "total_edges": total_edges,
    }

    return {
        "R_diff_lm": diff_lm,
        "R_diff_hm": diff_hm,
        "R_gap": R_gap,
        "stability": stability,
        "cos_lm": cos_lm,
        "cos_hm": cos_hm,
        "cos_lh": cos_lh,
        "sigma_mean": sigma.mean().item(),
        "sigma_std": sigma.std().item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
        "r_mean": r.mean().item(),
        "r_std": r.std().item(),
        "r_min": r.min().item(),
        "r_max": r.max().item(),
        "beta": [round(b.item(), 4) for b in beta],
        "beta_entropy": h_beta,
        "fou_mean": fou.mean().item(),
        "fou_std": fou.std().item(),
        "edge_info": edge_info,
    }


def audit_ablation(model, test_loader, device):
    """Hard ablation: marginal contribution of each module."""
    baseline = evaluate(model, test_loader, device)
    results = {"baseline": baseline}

    g = model.fuzzy_graph

    # ── Ablate Type-2 (r→0) ──
    if g is not None:
        orig_r = g.log_radius_ratio.data.clone()
        g.log_radius_ratio.data.fill_(-100)  # sigmoid ≈ 0
        results["no_type2"] = evaluate(model, test_loader, device)
        g.log_radius_ratio.data.copy_(orig_r)

    # ── Ablate CellAttention ──
    blocks = []
    enc = model.condition_encoder
    dec = model.future_decoder
    for b in list(enc.blocks) + list(dec.blocks):
        if hasattr(b, 'cell_attention') and b.use_cell_attention:
            blocks.append(b)
    if blocks:
        saved_blends = [b.cell_attention.cell_blend.data.clone() for b in blocks]
        for b in blocks:
            b.cell_attention.cell_blend.data.fill_(-100)
        results["no_cell"] = evaluate(model, test_loader, device)
        for b, saved in zip(blocks, saved_blends):
            b.cell_attention.cell_blend.data.copy_(saved)

    # ── Ablate FuzzyGraph (use static adjacency) ──
    if g is not None and g._has_static:
        # Use static adjacency as the graph
        orig_forward = g.get_type2_info
        def static_graph(h):
            N = g.num_nodes
            R = g.static_adjacency.to(device=device)
            return R, torch.zeros(N, device=device)
        g.get_type2_info = static_graph
        results["no_fuzzy"] = evaluate(model, test_loader, device)
        g.get_type2_info = orig_forward

    # ── Marginal contributions ──
    for key in list(results.keys()):
        if key != "baseline":
            results[f"Δ{key.replace('no_','')}"] = results[key] - baseline

    return results


def audit_soft_ablation(model, test_loader, device, lambdas=None):
    """Soft ablation: λ-interpolation of CellAttention contribution."""
    if lambdas is None:
        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    blocks = []
    enc = model.condition_encoder
    dec = model.future_decoder
    for b in list(enc.blocks) + list(dec.blocks):
        if hasattr(b, 'cell_attention') and b.use_cell_attention:
            blocks.append(b)

    if not blocks:
        return {"error": "no CellAttention blocks"}

    saved_blends = [b.cell_attention.cell_blend.data.clone() for b in blocks]
    curve = {}
    for lam in lambdas:
        for b in blocks:
            b.cell_attention.cell_blend.data.copy_(saved_blends[0] * lam)
        curve[f"λ={lam:.1f}"] = evaluate(model, test_loader, device)

    # Restore
    for b, saved in zip(blocks, saved_blends):
        b.cell_attention.cell_blend.data.copy_(saved)

    return curve


def audit_params(model):
    """Parameter utilization audit."""
    rows = []
    total_params = 0
    dead_params = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.grad is not None:
            gnorm = p.grad.abs().mean().item()
            pnorm = p.data.abs().mean().item()
            ratio = gnorm / (pnorm + 1e-8)
        else:
            gnorm = 0.0
            ratio = 0.0
        if ratio < 1e-6:
            dead_params += n
        rows.append((name, n, gnorm, ratio))
    rows.sort(key=lambda r: r[3])  # sort by grad/param ratio
    return {
        "total": total_params,
        "dead": dead_params,
        "dead_pct": dead_params / total_params * 100 if total_params else 0,
        "details": rows[:20] + rows[-10:],  # top dead + top active
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="final_T2 Model Audit")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training output directory")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch checkpoint (default: best .m)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dummy_data", action="store_true",
                        help="Skip data loading, use dummy test (for quick audit)")
    args = parser.parse_args()

    device = args.device
    print(f"=== final_T2 Audit ===")
    print(f"Run dir: {args.run_dir}")
    print(f"Device:  {device}")

    # ── Load model ──
    model, config, data_feature, run_path = load_run(args.run_dir, args.epoch, device)
    print(f"Model:   {config['model']} on {config['dataset']}")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Load test data ──
    if args.dummy_data:
        class DummyLoader:
            def __iter__(self):
                N = data_feature.get("num_nodes", 300)
                D = config.get("input_window", 12)
                hdim = config.get("hidden_dim", 64)
                batch = type('Batch', (), {})()
                batch.X = torch.randn(4, D, N, hdim, device=device)
                batch.y = torch.randn(4, D, N, 1, device=device)
                def to_tensor(dev):
                    batch.X = batch.X.to(dev)
                    batch.y = batch.y.to(dev)
                batch.to_tensor = to_tensor
                yield batch
        test_loader = DummyLoader()
    else:
        try:
            test_loader, _ = load_test_data(config, data_feature)
            print(f"Test data loaded OK")
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to dummy data. Use --dummy_data to skip this warning.")
            args.dummy_data = True
            from types import SimpleNamespace
            N = data_feature.get("num_nodes", 300)
            D = config.get("input_window", 12)
            hdim = config.get("hidden_dim", 64)
            class Dummy:
                def __iter__(s):
                    b = SimpleNamespace()
                    b.X = torch.randn(4, D, N, hdim, device=device)
                    b.y = torch.randn(4, D, N, 1, device=device)
                    b.to_tensor = lambda dev: None
                    yield b
            test_loader = Dummy()
    print()

    # ═══════════════════════════════════════════════════
    # 1. Membership Structure
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("1. FUZZY MEMBERSHIP STRUCTURE")
    print("=" * 70)
    mem = audit_membership(model, test_loader, device)
    if "error" not in mem:
        print(f"  Entropy:     {mem['entropy_mean']:.4f} ± {mem['entropy_std']:.4f}")
        print(f"  Margin:      {mem['margin_mean']:.4f} ± {mem['margin_std']:.4f}  (min={mem['margin_min']:.4f})")
        print(f"  Dead sets:   {mem['dead_sets']} / {len(mem['set_sizes'])}")
        print(f"  Set sizes:   {mem['set_sizes']}")
        if mem['margin_mean'] < 0.05:
            print("  ⚠️  WARNING: Very low margin → fuzzy collapse")
        elif mem['margin_mean'] < 0.2:
            print("  ⚠️  Low margin → weak cluster separation")
        else:
            print("  ✓  Good margin → strong clustering")
        if mem['dead_sets'] > 0:
            print(f"  ⚠️  WARNING: {mem['dead_sets']} dead fuzzy sets")
    print()

    # ═══════════════════════════════════════════════════
    # 2. Prototype Quality
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("2. PROTOTYPE QUALITY")
    print("=" * 70)
    proto = audit_prototypes(model)
    if "error" not in proto:
        print(f"  Inter-cluster:  {proto['inter_mean']:.4f}")
        print(f"  Intra-cluster:  {proto['intra_mean']:.4f}")
        print(f"  Scatter ratio:  {proto['scatter_ratio']:.2f}")
        print(f"  Min proto dist: {proto['min_proto_dist']:.4f}")
        if proto['scatter_ratio'] < 1.5:
            print("  ⚠️  Low scatter ratio → prototypes possibly redundant")
        else:
            print("  ✓  Good prototype separation")
    print()

    # ═══════════════════════════════════════════════════
    # 3. Type-2 Interval Validity
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("3. TYPE-2 INTERVAL VALIDITY")
    print("=" * 70)
    t2 = audit_type2_relations(model, test_loader, device)
    if "error" not in t2:
        print(f"  σ:      {t2['sigma_mean']:.4f} ± {t2['sigma_std']:.4f}  [{t2['sigma_min']:.2f}, {t2['sigma_max']:.2f}]")
        print(f"  r:      {t2['r_mean']:.4f} ± {t2['r_std']:.4f}  [{t2['r_min']:.4f}, {t2['r_max']:.4f}]")
        print(f"  β:      {t2['beta']}  H={t2['beta_entropy']:.4f}")
        print(f"  FOU:    {t2['fou_mean']:.4f} ± {t2['fou_std']:.4f}")
        print(f"  R_gap:  {t2['R_gap']:.4f}")
        print(f"  Stability: {t2['stability']:.4f}  (cos: LM={t2['cos_lm']:.3f} HM={t2['cos_hm']:.3f} LH={t2['cos_lh']:.3f})")
        print(f"  Δ(L,M): {t2['R_diff_lm']:.4f}  Δ(H,M): {t2['R_diff_hm']:.4f}")

        # Interpretations
        if t2['sigma_std'] < 0.2:
            print("  ⚠️  Low σ diversity → all fuzzy sets have similar width")
        if t2['R_gap'] < 0.01:
            print("  ⚠️  CRITICAL: R_gap < 0.01 → Type-2 has collapsed to Type-1")
        elif t2['R_gap'] < 0.05:
            print("  ⚠️  Low R_gap → Type-2 interval is weak")
        else:
            print("  ✓  Healthy R_gap → Type-2 interval is active")
        if t2['stability'] > 0.99:
            print("  ⚠️  Very high stability → three graphs nearly identical (Type-2 dead)")
        elif t2['stability'] < 0.95:
            print("  ✓  Good stability diversity")
        if t2['r_std'] < 0.01:
            print("  ⚠️  r not differentiated across fuzzy sets")

        # ── Edge analysis ──
        if 'edge_info' in t2:
            ei = t2['edge_info']
            print()
            print("  ── Graph sparsity ──")
            print(f"  Total possible edges: {ei['total_edges']:,} ({int(ei['total_edges']**0.5)}×{int(ei['total_edges']**0.5)})")
            for name in ['R_mid', 'R_mixed']:
                if name in ei:
                    e = ei[name]
                    print(f"  {name}:")
                    print(f"    edges >0.01: {e['edges>0.01']:,} ({e['sparsity>0.01']:.1%})  >0.05: {e['edges>0.05']:,} ({e['sparsity>0.05']:.1%})  >0.10: {e['edges>0.10']:,} ({e['sparsity>0.10']:.1%})")
                    print(f"    degree: {e['degree_mean']:.1f}±{e['degree_std']:.1f}  [{e['degree_min']:.0f}, {e['degree_max']:.0f}]")
                    print(f"    weight: μ={e['edge_mean']:.4f} σ={e['edge_std']:.4f}")
            if ei['R_mid']['edges>0.05'] < 10:
                print("  ⚠️  CRITICAL: <10 effective edges → graph almost dead")
            elif ei['R_mixed']['sparsity>0.05'] < 0.01:
                print("  ⚠️  Very sparse graph (<1% edges active)")
    print()

    # ═══════════════════════════════════════════════════
    # 4. Module Ablation
    # ═══════════════════════════════════════════════════
    if not args.dummy_data:
        print("=" * 70)
        print("4. MODULE ABLATION (ΔMAE)")
        print("=" * 70)
        abl = audit_ablation(model, test_loader, device)
        print(f"  Baseline MAE:    {abl['baseline']:.4f}")
        for key in sorted(abl.keys()):
            if key.startswith("Δ"):
                val = abl[key]
                tag = "✓" if val > 0 else "✗"
                print(f"  {tag} {key:20s}: {abl[key.replace('Δ','no_')]:.4f}  (Δ={val:+.4f})")

        # Soft ablation
        print()
        print("  CellAttention soft ablation curve:")
        curve = audit_soft_ablation(model, test_loader, device)
        for key, val in curve.items():
            bar = "█" * int(val * 50 / max(curve.values()))
            print(f"    {key}: {val:.4f}  {bar}")
        print()

    # ═══════════════════════════════════════════════════
    # 5. Parameter Audit
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("5. PARAMETER UTILIZATION")
    print("=" * 70)
    params = audit_params(model)
    print(f"  Total: {params['total']:,}")
    print(f"  Dead:  {params['dead']:,} ({params['dead_pct']:.1f}%)")
    print()
    print("  Top 5 dead params (lowest |∇|/|θ|):")
    for name, n, gn, ratio in params['details'][:5]:
        print(f"    {name:50s} {n:>6d}  |∇|/|θ|={ratio:.2e}")
    print("  Top 5 active params (highest |∇|/|θ|):")
    for name, n, gn, ratio in params['details'][-5:]:
        print(f"    {name:50s} {n:>6d}  |∇|/|θ|={ratio:.2e}")
    print()

    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
