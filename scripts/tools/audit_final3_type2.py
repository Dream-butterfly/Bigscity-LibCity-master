"""
final_3_type2 Innovation Audit — 第二轮审计脚本。

收集以下指标的完整实验数据：

  Q1:  S_low / S_mid / S_high 进入哪些模块的统计
  Q3:  theta_lower vs theta_delta 梯度范数
  Q4:  Dynamic Membership 时间方差 Var(μ_t)
  Q5:  Static vs Dynamic Membership 性能对比
  Q6:  Closure Δ = |S-R|₁ / |R|₁
  Q7:  hop 阶数对 S 的边际贡献
  Q8:  corr(FOU, Entropy)
  Q9:  corr(FOU, prediction error)
  Q10: FRR ablation (无μ / 全模型)

用法：
  python scripts/tools/audit_final3_type2.py \
      --dataset METR_LA \
      --checkpoint cache/model_cache/METR_LA/final_3_type2/xxx.pt
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


# ═══════════════════════════════════════════════════════════════════
#  Hook registry for intermediate tensor collection
# ═══════════════════════════════════════════════════════════════════

class AuditCollector:
    """Collect intermediate tensors via forward hooks."""

    def __init__(self):
        self.data = defaultdict(list)
        self._handles = []

    def _make_hook(self, key):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.data[key].append(out.detach().cpu())
            elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                self.data[key].append(out[0].detach().cpu())
        return hook

    def register(self, module, key):
        h = module.register_forward_hook(self._make_hook(key))
        self._handles.append(h)

    def clear(self):
        for key in list(self.data.keys()):
            self.data[key] = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stack(self, key):
        vals = self.data.get(key, [])
        if not vals:
            return None
        # Handle list-of-scalars or list-of-tensors
        if vals[0].dim() == 0:
            return torch.tensor([v.item() for v in vals])
        return torch.stack(vals, dim=0)


# ═══════════════════════════════════════════════════════════════════
#  Audit runner
# ═══════════════════════════════════════════════════════════════════

class Final3Type2Auditor:
    """Instruments final_3_type2 for innovation audit."""

    def __init__(self, model, dataloader, device, num_batches=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_batches = num_batches or len(dataloader)
        self.collector = AuditCollector()
        self._registered = False

    def _unwrap(self):
        """Handle DDP wrapper."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    # ── Hook registration ─────────────────────────────────────

    def _register_hooks(self):
        if self._registered:
            return
        m = self._unwrap()
        fg = m.fuzzy_graph

        # graph.py hooks
        self.collector.register(fg, "fuzzy_graph_out")

        # model.py hooks for loss components
        self.collector.register(m.condition_encoder, "encoder_out")
        self.collector.register(m.future_decoder, "decoder_out")

        self._registered = True

    def _remove_hooks(self):
        self.collector.remove()
        self._registered = False

    # ── Core capture methods ───────────────────────────────────

    @torch.no_grad()
    def _capture_static_graphs(self):
        """Capture S_low, S_high, FOU, S_eff, S_dyn on a single batch."""
        m = self._unwrap()
        fg = m.fuzzy_graph

        # Static memberships (no features)
        mu_low_s, mu_high_s, mu_mid_s = fg._compute_memberships()
        R_low_s, R_high_s = fg._build_fuzzy_relation_t2(mu_low_s, mu_high_s)

        # With features (dynamic) — use first batch
        batch = next(iter(self.dataloader))
        batch.to_tensor(self.device)
        history = batch["X"]

        mu_low_d, mu_high_d, mu_mid_d = fg._compute_memberships(history)
        R_low_d, R_high_d = fg._build_fuzzy_relation_t2(mu_low_d, mu_high_d)

        # Closures
        S_low = fg._compute_closure(R_low_d, m.semantic_closure_hops)
        S_high = fg._compute_closure(R_high_d, m.semantic_closure_hops)
        FOU_mat = S_high - S_low

        # Effective graph
        S_eff, FOU_out = fg.get_effective_graph(
            max_hops=m.semantic_closure_hops,
            mode=m.type2_graph_mode,
            fou_gate_scale=m.type2_fou_gate_scale,
            node_features=history,
        )

        # S_dyn
        if m.use_entropy_dynamic_graph:
            S_dyn = m._apply_entropy_dynamic_graph(S_eff, FOU=FOU_out,
                                                    node_features=history)
        else:
            S_dyn = S_eff

        return {
            "mu_low_static": mu_low_s,
            "mu_high_static": mu_high_s,
            "mu_mid_static": mu_mid_s,
            "mu_low_dynamic": mu_low_d,
            "mu_high_dynamic": mu_high_d,
            "mu_mid_dynamic": mu_mid_d,
            "R_low": R_low_d,
            "R_high": R_high_d,
            "S_low": S_low,
            "S_high": S_high,
            "FOU": FOU_mat,
            "S_eff": S_eff,
            "S_dyn": S_dyn,
        }

    # ═══════════════════════════════════════════════════════════
    #  Q1: Graph → Module flow
    # ═══════════════════════════════════════════════════════════

    def audit_q1_graph_flow(self):
        """Which graphs enter which modules?"""
        caps = self._capture_static_graphs()
        S_eff = caps["S_eff"]
        S_dyn = caps["S_dyn"]
        FOU_mat = caps["FOU"]
        N = S_eff.shape[0]

        results = {
            "graphs_available": {
                "S_low": caps["S_low"].shape,
                "S_high": caps["S_high"].shape,
                "FOU": FOU_mat.shape,
                "S_eff": S_eff.shape,
                "S_dyn": S_dyn.shape,
            },
            "module_usage": {
                "GCN": "S_dyn" if torch.equal(S_dyn, S_dyn) else "N/A",
                "FRR": "mu_mid (midpoint membership only)",
                "FIR": "S_dyn",
                "Decoder_GCN": "S_dyn",
                "Encoder_GCN": "S_dyn",
            },
            "effective_graphs_per_module": {
                "GCN": "S_dyn (single graph)",
                "FRR": "mu_fuzzy (midpoint vector, not graph matrix)",
                "FIR": "S_dyn (single graph)",
            },
            "s_diff_stats": {
                "|S_low - S_high|_mean": float((caps["S_low"] - caps["S_high"]).abs().mean()),
                "|R_low - R_high|_mean": float((caps["R_low"] - caps["R_high"]).abs().mean()),
                "FOU_mean": float(FOU_mat.mean()),
                "FOU_max": float(FOU_mat.max()),
            },
        }
        return results

    # ═══════════════════════════════════════════════════════════
    #  Q3: Gradient norms (theta_lower vs theta_delta)
    # ═══════════════════════════════════════════════════════════

    def audit_q3_gradient_norms(self, num_steps=20):
        """Measure per-step gradient norms for Type-2 parameters."""
        m = self._unwrap()
        fg = m.fuzzy_graph
        m.train()

        grad_stats = {"theta_lower": [], "theta_delta": [], "ratio_delta_lower": []}

        iterator = iter(self.dataloader)
        for step in range(num_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                batch = next(iterator)

            batch.to_tensor(self.device)
            m.optimizer.zero_grad() if hasattr(m, 'optimizer') else m.zero_grad()

            loss = m.calculate_loss(batch)
            if isinstance(loss, torch.Tensor):
                loss.backward()

            g_lower = fg.base_membership_lower.grad
            g_delta = fg.base_membership_delta.grad

            if g_lower is not None and g_delta is not None:
                n_lower = g_lower.norm().item()
                n_delta = g_delta.norm().item()
                grad_stats["theta_lower"].append(n_lower)
                grad_stats["theta_delta"].append(n_delta)
                grad_stats["ratio_delta_lower"].append(
                    n_delta / n_lower if n_lower > 1e-8 else float("inf"))

        m.eval()
        summary = {}
        for k, vals in grad_stats.items():
            if vals:
                summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            else:
                summary[k] = "no gradients captured"

        return summary

    # ═══════════════════════════════════════════════════════════
    #  Q4: Dynamic Membership variance across time
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def audit_q4_membership_variance(self, num_batches=10, sample_nodes=(10, 50, 100)):
        """Var(μ_t) for sampled nodes across different input batches."""
        m = self._unwrap()
        fg = m.fuzzy_graph
        N = m.num_nodes
        K = m.fuzzy_num_sets

        # Validate sample nodes
        sample_nodes = [n for n in sample_nodes if n < N]
        if not sample_nodes:
            return {"error": f"no valid sample nodes (N={N})"}

        memberships_by_node = {n: [] for n in sample_nodes}
        iterator = iter(self.dataloader)

        for _ in range(min(num_batches, len(self.dataloader))):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch.to_tensor(self.device)
            history = batch["X"]
            _, _, mu_mid = fg._compute_memberships(history)
            for n in sample_nodes:
                memberships_by_node[n].append(mu_mid[n].cpu())

        results = {}
        for n in sample_nodes:
            if not memberships_by_node[n]:
                results[f"node_{n}"] = "no data"
                continue
            stacked = torch.stack(memberships_by_node[n], dim=0)  # [B, K]
            var_per_set = stacked.var(dim=0)  # [K]
            results[f"node_{n}"] = {
                "membership_shape": list(stacked.shape),
                "var_per_fuzzy_set": var_per_set.tolist(),
                "mean_var": float(var_per_set.mean()),
                "max_var": float(var_per_set.max()),
                "is_effectively_static": bool(var_per_set.mean() < 0.001),
            }

        return results

    # ═══════════════════════════════════════════════════════════
    #  Q5: Performance delta (static vs dynamic)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def audit_q5_static_vs_dynamic(self, num_batches=20):
        """Compare static membership performance vs dynamic."""
        m = self._unwrap()
        fg = m.fuzzy_graph
        m.eval()

        losses_dynamic = []
        losses_static = []
        iterator = iter(self.dataloader)

        for _ in range(min(num_batches, len(self.dataloader))):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch.to_tensor(self.device)
            history = batch["X"]
            future = batch["y"][..., :m.output_dim].to(self.device)

            # Dynamic path (normal)
            pred_dyn = m.predict({"X": history})
            l_dyn = F.l1_loss(pred_dyn, future).item()
            losses_dynamic.append(l_dyn)

        return {
            "dynamic_mean_loss": float(np.mean(losses_dynamic)),
            "dynamic_std_loss": float(np.std(losses_dynamic)),
            "note": "static vs dynamic comparison on eval data. "
                     "Static variant requires separate training run "
                     "with node_features=None hardcoded.",
        }

    # ═══════════════════════════════════════════════════════════
    #  Q6: Closure Δ = |S - R|₁ / |R|₁
    # ═══════════════════════════════════════════════════════════

    def audit_q6_closure_diff(self):
        """How much does closure change the base relation?"""
        caps = self._capture_static_graphs()
        R = caps["R_high"]  # using upper bound
        S = caps["S_high"]

        R_abs = R.abs().sum()
        delta = (S - R).abs().sum()
        ratio = (delta / R_abs.clamp_min(1e-8)).item()

        # Per-hop decomposition
        m = self._unwrap()
        fg = m.fuzzy_graph
        max_hops = m.semantic_closure_hops

        hops_data = {}
        S = R.clone()         # closure accumulator
        current = R.clone()    # R^k
        for h in range(1, max_hops + 1):
            cumul_delta = (S - R).abs().sum() / R_abs.clamp_min(1e-8)
            hops_data[f"hop_{h}_cumulative_delta"] = float(cumul_delta)
            if h < max_hops:
                # R^(k+1) = max-min(R, R^k)
                next_power = torch.max(
                    torch.min(R.unsqueeze(1), current.unsqueeze(0)), dim=-1
                ).values
                current = next_power
                S = torch.max(S, current)  # S = max(R, R², ..., R^{k+1})

        return {
            "|S - R| / |R|": float(ratio),
            "is_closure_working": ratio > 0.01,
            "hop_contribution": hops_data,
            "interpretation": (
                "Closure is significantly modifying the graph"
                if ratio > 0.05 else
                "Closure has marginal effect — graph is nearly transitive already"
                if ratio > 0.01 else
                "Closure is effectively identity — graph is already transitively closed"
            ),
        }

    # ═══════════════════════════════════════════════════════════
    #  Q7: Per-hop contribution to S (order analysis)
    # ═══════════════════════════════════════════════════════════

    def audit_q7_hop_analysis(self):
        """How much does each hop order contribute to the final S?"""
        caps = self._capture_static_graphs()
        m = self._unwrap()
        fg = m.fuzzy_graph
        R = caps["R_high"]

        contributions = []
        current = R.clone()
        N = R.shape[0]
        prev_sum = R.abs().sum().item()

        for h in range(1, m.semantic_closure_hops + 1):
            contributions.append({
                "hop": h,
                "graph_sum": float(current.abs().sum()),
                "graph_mean": float(current.mean()),
                "graph_nonzero_ratio": float((current > 0.01).float().mean()),
            })
            if h < m.semantic_closure_hops:
                R_k = torch.max(
                    torch.min(R.unsqueeze(1), current.unsqueeze(0)), dim=-1
                ).values
                current = R_k

        # Marginal: how much new information each hop adds
        marginal = []
        for i in range(len(contributions) - 1):
            delta_sum = contributions[i+1]["graph_sum"] - contributions[i]["graph_sum"]
            marginal.append({
                "from_hop": i + 1,
                "to_hop": i + 2,
                "delta_sum": delta_sum,
                "delta_mean": float(contributions[i+1]["graph_mean"] - contributions[i]["graph_mean"]),
            })

        return {
            "contributions": contributions,
            "marginal_gains": marginal,
            "effective_hops": next(
                (i+1 for i, m in enumerate(marginal)
                 if abs(m["delta_mean"]) < 0.001),
                len(marginal) + 1
            ),
        }

    # ═══════════════════════════════════════════════════════════
    #  Q8: corr(FOU, Entropy)
    # ═══════════════════════════════════════════════════════════

    def audit_q8_fou_entropy_corr(self):
        """Correlation between per-node FOU and membership entropy.

        Computes TWO correlations:
          - corr(FOU, H_full):  H_full = H_mid * (1 + FOU_avg)
            Note: this has a built-in structural correlation because
            H_full shares the FOU_avg term.  Interpret with caution.
          - corr(FOU, H_mid):   H_mid = -(p·log p) on midpoint
            This is the pure coupling: does a node with ambiguous
            membership (high H_mid) ALSO have a wide interval?
        """
        m = self._unwrap()
        fg = m.fuzzy_graph

        mu_low, mu_high, _ = fg._compute_memberships()
        mu_mid = (mu_low + mu_high) / 2.0
        FOU_node = (mu_high - mu_low).mean(dim=-1).detach().cpu().numpy()

        # H_mid: pure Shannon entropy on midpoint membership
        p = mu_mid / mu_mid.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        H_mid = -(p * (p + 1e-8).log()).sum(dim=-1).detach().cpu().numpy()

        # H_full: with FOU amplification (same as get_cell_entropy)
        H_full = H_mid * (1.0 + FOU_node)

        r_full, p_full = pearsonr(FOU_node, H_full)
        r_mid, p_mid = pearsonr(FOU_node, H_mid)

        return {
            "corr_FOU_H_full": {
                "pearson_r": float(r_full),
                "p_value": float(p_full),
                "note": "⚠️  H_full = H_mid*(1+FOU_avg) — structural coupling, "
                        "not a learned property.  High r is expected.",
            },
            "corr_FOU_H_mid": {
                "pearson_r": float(r_mid),
                "p_value": float(p_mid),
                "is_significant": p_mid < 0.05,
                "interpretation": (
                    "FOU and H_mid strongly coupled — interval width "
                    "reflects genuine membership ambiguity"
                    if abs(r_mid) > 0.5 else
                    "FOU and H_mid weakly coupled — interval width is "
                    "not aligned with membership ambiguity"
                    if abs(r_mid) > 0.2 else
                    "FOU and H_mid uncorrelated — interval width is "
                    "arbitrary, not driven by membership structure"
                ),
            },
            "mean_fou": float(FOU_node.mean()),
            "mean_h_mid": float(H_mid.mean()),
            "std_fou": float(FOU_node.std()),
            "std_h_mid": float(H_mid.std()),
        }

    # ═══════════════════════════════════════════════════════════
    #  Q9: corr(FOU, prediction error)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def audit_q9_fou_error_corr(self, num_batches=20):
        """Does the model know where it's uncertain?"""
        m = self._unwrap()
        fg = m.fuzzy_graph
        m.eval()

        all_errors = []
        all_fou = []

        # Per-node average FOU (static approximation)
        mu_low, mu_high, _ = fg._compute_memberships()
        fou_node = (mu_high - mu_low).mean(dim=-1).cpu().numpy()  # [N]

        iterator = iter(self.dataloader)
        for _ in range(min(num_batches, len(self.dataloader))):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch.to_tensor(self.device)
            history = batch["X"]
            future = batch["y"][..., :m.output_dim].to(self.device)

            pred = m.predict({"X": history})  # [B, T, N, C]
            abs_err = (pred - future).abs().mean(dim=(0, 1, -1)).cpu().numpy()  # [N]
            all_errors.append(abs_err)

        # Average error per node across batches
        mean_error = np.stack(all_errors, axis=0).mean(axis=0)  # [N]
        r, p = pearsonr(fou_node, mean_error)

        return {
            "pearson_r": float(r),
            "p_value": float(p),
            "is_significant": p < 0.05,
            "interpretation": (
                "Model KNOWS where it's uncertain — FOU ↑ → error ↑. "
                "This is the strongest possible evidence for Type-2 validity."
                if r > 0.3 else
                "FOU and error weakly correlated — model has some self-awareness"
                if abs(r) > 0.1 else
                "FOU and error uncorrelated — Type-2 uncertainty is not "
                "calibrated to actual prediction difficulty"
            ),
            "mean_node_error": float(mean_error.mean()),
            "std_node_error": float(mean_error.std()),
        }

    # ═══════════════════════════════════════════════════════════
    #  Q10: FRR ablation (with/without mu_fuzzy)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def audit_q10_frr_ablation(self, num_batches=20):
        """Compare FRR performance with and without fuzzy membership conditioning."""
        m = self._unwrap()
        m.eval()

        # We need to temporarily disable mu_fuzzy in FRR
        # Approach: monkey-patch FRR's _compute_membership to ignore mu_fuzzy
        losses_full = []
        losses_no_mu = []

        iterator = iter(self.dataloader)
        for _ in range(min(num_batches, len(self.dataloader))):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch.to_tensor(self.device)
            history = batch["X"]
            future = batch["y"][..., :m.output_dim].to(self.device)

            # Full model
            pred_full = m.predict({"X": history})
            losses_full.append(F.l1_loss(pred_full, future).item())

            # FRR without mu_fuzzy: temporarily nullify fuzzy_to_cell
            saved_ftc = {}
            for blocks in [m.condition_encoder.blocks, m.future_decoder.blocks]:
                for block in blocks:
                    if hasattr(block, 'cell_attention'):
                        ca = block.cell_attention
                        saved_ftc[ca] = ca.fuzzy_to_cell
                        ca.fuzzy_to_cell = None

            pred_no_mu = m.predict({"X": history})
            losses_no_mu.append(F.l1_loss(pred_no_mu, future).item())

            # Restore
            for ca, ftc in saved_ftc.items():
                ca.fuzzy_to_cell = ftc

        lf_mean = float(np.mean(losses_full)) if losses_full else 0
        ln_mean = float(np.mean(losses_no_mu)) if losses_no_mu else 0
        delta_pct = (ln_mean - lf_mean) / lf_mean * 100 if lf_mean > 0 else 0

        return {
            "full_model_MAE": lf_mean,
            "no_mu_fuzzy_MAE": ln_mean,
            "delta_pct": delta_pct,
            "interpretation": (
                "FRR does NOT actually use fuzzy membership — removing "
                "mu_fuzzy has negligible impact"
                if abs(delta_pct) < 0.5 else
                "FRR uses fuzzy membership moderately — removing it has "
                "measurable but not critical impact"
                if abs(delta_pct) < 2.0 else
                "FRR relies on fuzzy membership significantly — removing "
                "it causes substantial performance drop"
            ),
        }

    # ═══════════════════════════════════════════════════════════
    #  Full audit runner
    # ═══════════════════════════════════════════════════════════

    def run_full_audit(self):
        """Run all audits and return structured results."""
        results = {}

        print("=" * 60)
        print("final_3_type2 Innovation Audit")
        print("=" * 60)

        print("\n[Q1] Graph → Module flow analysis...")
        results["Q1_graph_flow"] = self.audit_q1_graph_flow()

        print("[Q3] Gradient norms (theta_lower vs theta_delta)...")
        results["Q3_gradient_norms"] = self.audit_q3_gradient_norms()

        print("[Q4] Membership time-variance...")
        results["Q4_membership_variance"] = self.audit_q4_membership_variance()

        print("[Q5] Static vs Dynamic performance...")
        results["Q5_static_vs_dynamic"] = self.audit_q5_static_vs_dynamic()

        print("[Q6] Closure Δ analysis...")
        results["Q6_closure_diff"] = self.audit_q6_closure_diff()

        print("[Q7] Hop-order analysis...")
        results["Q7_hop_analysis"] = self.audit_q7_hop_analysis()

        print("[Q8] FOU-Entropy correlation...")
        results["Q8_fou_entropy_corr"] = self.audit_q8_fou_entropy_corr()

        print("[Q9] FOU-Error correlation...")
        results["Q9_fou_error_corr"] = self.audit_q9_fou_error_corr()

        print("[Q10] FRR ablation...")
        results["Q10_frr_ablation"] = self.audit_q10_frr_ablation()

        return results


def print_summary(results):
    """Print a condensed summary."""
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    # Q1
    q1 = results.get("Q1_graph_flow", {})
    fou_mean = q1.get("s_diff_stats", {}).get("FOU_mean", "N/A")
    print(f"\n  Q1: FOU matrix mean = {fou_mean:.4f}" if isinstance(fou_mean, float)
          else f"\n  Q1: FOU matrix mean = {fou_mean}")

    # Q3
    q3 = results.get("Q3_gradient_norms", {})
    tl = q3.get("theta_lower", {}).get("mean", "N/A")
    td = q3.get("theta_delta", {}).get("mean", "N/A")
    ratio = q3.get("ratio_delta_lower", {}).get("mean", "N/A")
    print(f"  Q3: grad(theta_lower)={tl:.6f}" if isinstance(tl, float) else f"  Q3: {tl}")
    print(f"      grad(theta_delta)={td:.6f}" if isinstance(td, float) else f"      {td}")
    print(f"      ratio(delta/lower)={ratio:.4f}" if isinstance(ratio, float) else f"      {ratio}")
    if isinstance(ratio, float) and ratio < 0.1:
        print("      ⚠️  FOU gradient is very weak — FOU may be decorative")

    # Q4
    q4 = results.get("Q4_membership_variance", {})
    for k, v in q4.items():
        if isinstance(v, dict):
            mv = v.get("mean_var", "N/A")
            static = v.get("is_effectively_static", False)
            tag = " ⚠️ STATIC" if static else ""
            print(f"  Q4: {k} Var(μ) = {mv:.6f}{tag}" if isinstance(mv, float)
                  else f"  Q4: {k} = {v}")

    # Q6
    q6 = results.get("Q6_closure_diff", {})
    delta = q6.get("|S - R| / |R|", "N/A")
    interp = q6.get("interpretation", "")
    print(f"  Q6: |S-R|/|R| = {delta:.4f}" if isinstance(delta, float)
          else f"  Q6: {delta}")
    print(f"      {interp}")

    # Q7
    q7 = results.get("Q7_hop_analysis", {})
    eff_hops = q7.get("effective_hops", "N/A")
    print(f"  Q7: effective hops = {eff_hops}")

    # Q8
    q8 = results.get("Q8_fou_entropy_corr", {})
    r8_full = q8.get("corr_FOU_H_full", {}).get("pearson_r", "N/A")
    r8_mid = q8.get("corr_FOU_H_mid", {}).get("pearson_r", "N/A")
    interp8 = q8.get("corr_FOU_H_mid", {}).get("interpretation", "")
    r8_str = f"{r8_mid:.4f}" if isinstance(r8_mid, float) else str(r8_mid)
    r8f_str = f"{r8_full:.4f}" if isinstance(r8_full, float) else str(r8_full)
    print(f"  Q8: corr(FOU, H_mid) = {r8_str}  |  corr(FOU, H_full) = {r8f_str}")
    print(f"      {interp8}")

    # Q9
    q9 = results.get("Q9_fou_error_corr", {})
    r9 = q9.get("pearson_r", "N/A")
    interp9 = q9.get("interpretation", "")
    print(f"  Q9: corr(FOU, error) = {r9:.4f}" if isinstance(r9, float)
          else f"  Q9: {r9}")
    print(f"      {interp9}")

    # Q10
    q10 = results.get("Q10_frr_ablation", {})
    delta_pct = q10.get("delta_pct", "N/A")
    interp10 = q10.get("interpretation", "")
    print(f"  Q10: FRR Δ(no_μ - full) = {delta_pct:.2f}%" if isinstance(delta_pct, float)
          else f"  Q10: {q10}")
    print(f"       {interp10}")

    # ── Final Innovation Scorecard ──
    print("\n" + "-" * 40)
    print("INNOVATION SCORECARD")
    print("-" * 40)

    scores = {}
    # Dynamic Membership
    has_var = False
    for k, v in q4.items():
        if isinstance(v, dict):
            if v.get("mean_var", 0) > 0.001:
                has_var = True
                break
    scores["Dynamic Membership"] = "✅ ACTIVE" if has_var else "⚠️ STATIC (code dynamic, behavior static)"

    # Type-2 Graph
    if isinstance(ratio, float) and ratio > 0.05:
        scores["Type-2 FOU"] = "✅ ACTIVE"
    elif isinstance(ratio, float) and ratio > 0.01:
        scores["Type-2 FOU"] = "⚠️ WEAK GRADIENT"
    else:
        scores["Type-2 FOU"] = "❌ DECORATIVE" if isinstance(ratio, float) else "?"

    # Closure
    if isinstance(delta, float) and delta > 0.05:
        scores["Semantic Closure"] = "✅ ACTIVE"
    elif isinstance(delta, float) and delta > 0.01:
        scores["Semantic Closure"] = "⚠️ MARGINAL"
    else:
        scores["Semantic Closure"] = "⚠️ IDENTITY (graph already transitive)" if isinstance(delta, float) else "?"

    # FOU-H_mid coupling (learned, not structural)
    if isinstance(r8_mid, float) and r8_mid > 0.5:
        scores["FOU-H Coupling"] = "✅ STRONG (interval width ∝ ambiguity)"
    elif isinstance(r8_mid, float) and r8_mid > 0.2:
        scores["FOU-H Coupling"] = "⚠️ WEAK"
    else:
        scores["FOU-H Coupling"] = "❌ NONE (FOU not driven by membership structure)" if isinstance(r8_mid, float) else "?"

    # FOU-Error calibration
    if isinstance(r9, float) and r9 > 0.3:
        scores["FOU Calibration"] = "✅ STRONG (model knows its uncertainty)"
    elif isinstance(r9, float) and r9 > 0.1:
        scores["FOU Calibration"] = "⚠️ WEAK"
    else:
        scores["FOU Calibration"] = "❌ UNCALIBRATED" if isinstance(r9, float) else "?"

    # FRR
    if isinstance(delta_pct, float) and abs(delta_pct) > 2:
        scores["FRR Fuzzy Gate"] = "✅ SIGNIFICANT"
    elif isinstance(delta_pct, float) and abs(delta_pct) > 0.5:
        scores["FRR Fuzzy Gate"] = "⚠️ MARGINAL"
    else:
        scores["FRR Fuzzy Gate"] = "❌ NEGLIGIBLE" if isinstance(delta_pct, float) else "?"

    for name, score in scores.items():
        print(f"  {name:25s} {score}")


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="final_3_type2 Innovation Audit")
    parser.add_argument("--dataset", type=str, default="METR_LA",
                        help="Dataset name")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Override config file (JSON)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--train_first", action="store_true",
                        help="Train model before auditing")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batches for statistical tests")
    parser.add_argument("--grad_steps", type=int, default=20,
                        help="Gradient measurement steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build config and dataset
    other_args = {}
    if args.config_file:
        other_args["type2_graph_mode"] = args.config_file

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

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
    elif not args.train_first:
        print("⚠️  No checkpoint provided and --train_first not set. "
              "Running audit on untrained model (gradients will be random).")

    model.eval()

    # Run audit
    auditor = Final3Type2Auditor(model, dataloader, device, args.num_batches)
    results = auditor.run_full_audit()

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
