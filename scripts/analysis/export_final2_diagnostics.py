"""
导出 final_T2 / MVF-STGFormer 的诊断数据用于论文制图。

用法:
    uv run scripts/analysis/export_final2_diagnostics.py \
        --run_id 20260615_120000__traffic_state_pred__final_T2__METR_LA \
        --epoch 42

输出:
    outputs/<run_id>/evaluate_cache/diagnostics.npz    ← 空间数据（矩阵/向量/隶属度/三视角R图）
    outputs/<run_id>/evaluate_cache/diagnostics_curves.csv ← 收敛曲线（从run.log解析）
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data import build_artifact_runtime, build_dataset_runtime
from GNNTP.data.artifact_io import load_run_meta
from GNNTP.utils import (
    add_general_args,
    align_checkpoint_config,
    get_executor,
    get_logger,
    get_model,
    get_output_root,
    set_random_seed,
    str2bool,
)
from GNNTP.utils.utils import get_run_dir

# ══════════════════════════════════════════════════════════════════════
#  run.log 解析：从文本日志提取收敛曲线
# ══════════════════════════════════════════════════════════════════════

_PER_EPOCH_PATTERNS = {
    "epoch": re.compile(
        r"Epoch\s*\[(\d+)/\d+\]\s+train_loss:\s*([\d.e+\-]+),\s*"
        r"val_loss:\s*([\d.e+\-]+),\s*lr:\s*([\d.e+\-]+)"
    ),
    "t2_beta": re.compile(
        r"\[T2\]\s+β=\[([\d.e+\-]+),\s*([\d.e+\-]+),\s*([\d.e+\-]+)\]"
        r"(?:\s+H=([\d.e+\-]+))?"
    ),
    "fou_stats": re.compile(
        r"FOU\(μ=([\d.e+\-]+),\s*σ=([\d.e+\-]+)\)"
    ),
    "t2_sigma": re.compile(
        r"σ_low=([\d.e+\-]+)\s+σ_mid=([\d.e+\-]+)\s+"
        r"σ_high=([\d.e+\-]+)\s+δ=([\d.e+\-]+).*"
        r"τ=([\d.e+\-]+)/([\d.e+\-]+)/([\d.e+\-]+).*"
        r"r=([\d.e+\-]+)"
    ),
    "losses": re.compile(
        r"L=\[mae=([\d.e+\-]+)\s+ent=([\d.e+\-]+)\s+gap=([\d.e+\-]+)\s+"
        r"fou=([\d.e+\-]+)\s+fce=([\d.e+\-]+)\s+consv=([\d.e+\-]+)\s+"
        r"pn=([\d.e+\-]+)\s+ln=([\d.e+\-]+)\s+dv=([\d.e+\-]+)\s+dd=([\d.e+\-]+)"
    ),
    "r_corr": re.compile(
        r"ρ\(L,M\)=([\d.e+\-]+)\s+ρ\(L,H\)=([\d.e+\-]+)\s+ρ\(M,H\)=([\d.e+\-]+)"
    ),
    "grads": re.compile(
        r"∇β=([\d.e+\-]+)\s+σ_low=([\d.e+\-]+)\s+"
        r"δ=([\d.e+\-]+)\s+proto=([\d.e+\-]+)"
    ),
    "r_gap": re.compile(r"R_gap=([\d.e+\-]+)\s+w=([\d.e+\-]+)"),
    "gcn_view": re.compile(
        r"Δ/sh=([\d.e+\-]+).*GCN=([\d.e+\-]+).*"
        r"∇z\(LM=([\d.e+\-]+),LH=([\d.e+\-]+),MH=([\d.e+\-]+)"
    ),
}


def parse_run_log(run_id: str) -> dict[str, list]:
    log_path = get_run_dir(run_id) / "logs" / "run.log"
    if not log_path.exists():
        print(f"[WARN] run.log not found: {log_path}")
        return {}
    curves: dict[str, list] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            for name, pat in _PER_EPOCH_PATTERNS.items():
                m = pat.search(line)
                if m:
                    curves.setdefault(name, []).append(
                        [float(g) for g in m.groups() if g is not None]
                    )
    return curves


def curves_to_csv(curves: dict[str, list], output_path: Path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, rows in curves.items():
            writer.writerow([f"# {name}", f"({len(rows)} rows)"])
            ncols = max(len(r) for r in rows) if rows else 0
            writer.writerow([f"col_{i}" for i in range(ncols)])
            for row in rows:
                writer.writerow(row)
            writer.writerow([])


# ══════════════════════════════════════════════════════════════════════
#  诊断数据收集
# ══════════════════════════════════════════════════════════════════════

def _safe_cpu(obj):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return None


def collect_diagnostics(model, test_loader) -> dict:
    """收集所有论文需要的空间诊断数据。

    返回 dict 包含：
      - 静态参数: beta_global, beta_per_node, sigma_*, tau_*, prototypes
      - 测试集聚合: predictions, truth, fou_mean/std, R_final_mean
      - 第一个batch快照: R_low/mid/high (三视角原始图), mu_* (隶属度矩阵)
      - 最后一个batch标量: R_corr, R_gap, view_dist, sigma_* 等
    """
    g = model.fuzzy_graph
    diag: dict = {}

    # ── 静态参数 (无需前向) ──
    diag["beta_global"]    = _safe_cpu(torch.softmax(g.relation_mix_logits, dim=0))
    diag["beta_per_node"]  = _safe_cpu(torch.softmax(g.node_beta_logits, dim=-1))
    diag["blend_alpha"]    = float(torch.sigmoid(g.blend_logit).cpu().item())
    diag["sigma_low"]      = _safe_cpu(torch.nn.functional.softplus(g.log_sigma_low) + 1e-3)
    sd = _safe_cpu(torch.nn.functional.softplus(g.log_sigma_delta) + 1e-3)
    diag["sigma_delta"]    = sd
    diag["sigma_high"]     = diag["sigma_low"] + sd
    diag["sigma_mid"]      = diag["sigma_low"] + sd * 0.5
    diag["tau_low"]        = _safe_cpu(torch.nn.functional.softplus(g.log_tau_low) + 0.1)
    diag["tau_mid"]        = _safe_cpu(torch.nn.functional.softplus(g.log_tau_mid) + 0.1)
    diag["tau_high"]       = _safe_cpu(torch.nn.functional.softplus(g.log_tau_high) + 0.1)
    diag["proto_low"]      = _safe_cpu(g.prototype_center_low)
    diag["proto_mid"]      = _safe_cpu(g.prototype_center_mid)
    diag["proto_high"]     = _safe_cpu(g.prototype_center_high)

    # ── 逐 batch 前向，收集聚合统计 + 第一个 batch 快照 ──
    model.eval()
    all_pred, all_truth, all_fou, all_R_final = [], [], [], []
    all_mu_raw_means = []
    first = {}  # 第一个 batch 的完整矩阵
    got_first = False

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            history, future = batch["X"], batch["y"][..., :model.output_dim]

            condition, R_final, fou, _ = model.encode_condition(history)
            pred = model.future_decoder(
                condition, R_final, graph_uncertainty=fou,
                powers=None,
                mu_low=getattr(model, "_current_mu_low_raw", None),
                mu_mid=getattr(model, "_current_mu_mid_raw", None),
                mu_high=getattr(model, "_current_mu_high_raw", None),
                beta=getattr(model, "_current_beta", None),
            )
            all_pred.append(pred.cpu().numpy())
            all_truth.append(future.cpu().numpy())
            all_fou.append(fou.cpu().numpy())
            all_R_final.append(R_final.cpu().numpy())
            if hasattr(g, "_current_mu_raw_low_mean"):
                all_mu_raw_means.append([
                    g._current_mu_raw_low_mean.cpu().item(),
                    g._current_mu_raw_mid_mean.cpu().item(),
                    g._current_mu_raw_high_mean.cpu().item(),
                ])

            if not got_first:
                # 重新计算以获取未混合的 mu 和 R_v
                ml, mu, mm, mlr, mmr, mhr = g._compute_memberships(history)
                def _bm(t):
                    return t.mean(dim=0) if t.dim() == 3 else t
                first["mu_upper"]    = _safe_cpu(_bm(mu))
                first["mu_lower"]    = _safe_cpu(_bm(ml))
                first["mu_mid"]      = _safe_cpu(_bm(mm))
                first["mu_low_raw"]  = _safe_cpu(_bm(mlr))
                first["mu_mid_raw"]  = _safe_cpu(_bm(mmr))
                first["mu_high_raw"] = _safe_cpu(_bm(mhr))
                for vn, vm in [("low", mlr), ("mid", mmr), ("high", mhr)]:
                    Rv = g._build_fuzzy_relation(vm)
                    first[f"R_{vn}"] = _safe_cpu(_bm(Rv))
                first["R_final"] = _safe_cpu(
                    R_final[0] if R_final.dim() == 3 else R_final)
                got_first = True

    # ── 聚合 ──
    diag["predictions"]    = np.concatenate(all_pred, axis=0)
    diag["truth"]          = np.concatenate(all_truth, axis=0)
    diag["fou_batches"]    = np.stack(all_fou, axis=0)
    diag["fou_mean"]       = diag["fou_batches"].mean(axis=0)
    diag["fou_std"]        = diag["fou_batches"].std(axis=0)
    diag["R_final_batches"] = np.stack(all_R_final, axis=0)
    diag["R_final_mean"]   = diag["R_final_batches"].mean(axis=0)
    for k, v in first.items():
        diag[k] = v
    if all_mu_raw_means:
        diag["mu_raw_mean"] = np.array(all_mu_raw_means).mean(axis=0)

    # ── 最后一个 batch 的标量快照 ──
    for attr in [
        "d2_mean", "d2_std", "proto_norm", "latent_norm",
        "center_dist", "transform_weight",
        "mu_diff_mean", "mu_diff_max",
        "R_diff_lm", "R_diff_hm", "R_gap",
        "R_corr_lm", "R_corr_lh", "R_corr_mh",
        "eff_width", "sigma_low", "sigma_mid", "sigma_high",
        "sigma_delta_mean", "tau_low", "tau_mid", "tau_high",
        "adapter_ratio",
        "view_dist_lm", "view_dist_lh", "view_dist_mh",
        "mu_raw_low_mean", "mu_raw_mid_mean", "mu_raw_high_mean",
    ]:
        val = getattr(g, f"_current_{attr}", None)
        if val is not None:
            diag[attr] = val.cpu().item() if isinstance(val, torch.Tensor) else float(val)

    return diag


# ══════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════

def export_diagnostics(
    *,
    run_id: str,
    epoch: int,
    task: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    config_file: str | None = None,
    artifact_id: str | None = None,
    other_args: dict | None = None,
):
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("--run_id is required")

    try:
        run_meta = load_run_meta(run_id)
    except FileNotFoundError:
        run_meta = {}
    resolved_task = str(task or run_meta.get("task") or "traffic_state_pred")
    resolved_model = str(model_name or run_meta.get("model") or "final_T2")
    resolved_dataset = str(dataset_name or run_meta.get("dataset") or "")
    if not resolved_dataset:
        raise ValueError("Cannot determine dataset. Provide --dataset or ensure run_meta.json exists.")

    bound_aid = str(run_meta.get("artifact_id", "")).strip()
    effective_aid = str(artifact_id or "").strip() or bound_aid

    saved_cfg = PROJECT_ROOT / "outputs" / run_id / "effective_config.json"
    base = {}
    if saved_cfg.exists():
        with open(saved_cfg, "r", encoding="utf-8") as f:
            base = json.load(f)
        for k in ("task", "model", "dataset", "exp_id"):
            base.pop(k, None)

    merged = dict(base)
    merged.update(other_args or {})
    merged["exp_id"] = run_id
    merged["epoch"] = int(epoch)

    config = ConfigParser(
        resolved_task, resolved_model, resolved_dataset,
        config_file, saved_model=False, train=False, other_args=merged,
    )
    logger = get_logger(config)
    logger.info("Export diag: run_id=%s epoch=%d model=%s dataset=%s",
                run_id, epoch, resolved_model, resolved_dataset)
    set_random_seed(config.get("seed", 0))

    if effective_aid:
        runtime = build_artifact_runtime(
            config, task=resolved_task, model_name=resolved_model,
            artifact_id=effective_aid, force_reuse=True,
        )
        for msg in runtime.warnings:
            logger.warning("[FORCE_REUSE] %s", msg)
    else:
        logger.info("No artifact_id found, using old dataset pipeline")
        runtime = build_dataset_runtime(config)

    ckpt_path = os.path.join(
        get_output_root(), run_id, "model_cache",
        f"{resolved_model}_{resolved_dataset}_epoch{int(epoch)}.tar",
    )
    align_checkpoint_config(config.config, ckpt_path, logger)

    model = get_model(config, runtime.data_feature)
    executor = get_executor(config, model, runtime.data_feature)

    logger.info("Collecting spatial diagnostics...")
    diag = collect_diagnostics(model, runtime.test_loader)

    out_dir = get_run_dir(run_id) / "evaluate_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "diagnostics.npz"
    np.savez_compressed(npz_path, **diag)
    logger.info("Saved diagnostics.npz  (%d arrays, %.1f KB)",
                len(diag), npz_path.stat().st_size / 1024)

    curves = parse_run_log(run_id)
    if curves:
        csv_path = out_dir / "diagnostics_curves.csv"
        curves_to_csv(curves, csv_path)
        logger.info("Saved diagnostics_curves.csv  (%d groups, %.1f KB)",
                    len(curves), csv_path.stat().st_size / 1024)
    else:
        logger.warning("No curves extracted from run.log")

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export final_T2 diagnostics for paper figures"
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--artifact_id", type=str, default=None)
    add_general_args(parser)
    args = parser.parse_args()

    d = vars(args)
    other = {k: v for k, v in d.items()
             if k not in ["run_id", "epoch", "task", "model", "dataset",
                          "config_file", "artifact_id"] and v is not None}

    export_diagnostics(
        run_id=args.run_id, epoch=args.epoch,
        task=args.task, model_name=args.model, dataset_name=args.dataset,
        config_file=args.config_file, artifact_id=args.artifact_id,
        other_args=other,
    )
