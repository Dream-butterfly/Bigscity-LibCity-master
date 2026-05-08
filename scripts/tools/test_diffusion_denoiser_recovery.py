"""new_diffusion_fuzzy 去噪器恢复能力诊断。

测试三个关键维度:
1. 不同噪声水平的单步恢复精度 (t=10, 50, 100, 150, 199)
2. 完整 DDIM 逆向链路的误差累积
3. 条件信号对不同噪声水平的影响强度

用法: uv run scripts/tools/test_diffusion_denoiser_recovery.py --artifact_id <id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data.runtime import build_artifact_runtime
from GNNTP.models.locator import get_model_component
from GNNTP.utils.paths import OUTPUT_ROOT
from GNNTP.utils.normalization import StandardScaler


def load_trained_model(artifact_id: str):
    """Load latest trained model checkpoint for new_diffusion_fuzzy."""
    config = ConfigParser(
        task="traffic_state_pred",
        model_name="new_diffusion_fuzzy",
        dataset_name="PEMSD4",
    )
    runtime = build_artifact_runtime(config, task="traffic_state_pred", model_name="new_diffusion_fuzzy",
                                     artifact_id=artifact_id)

    # Find latest checkpoint
    outputs = sorted(OUTPUT_ROOT.glob("*__traffic_state_pred__new_diffusion_fuzzy__*"), reverse=True)
    if not outputs:
        raise FileNotFoundError("No output directory found for new_diffusion_fuzzy")
    ckpt_dir = outputs[0] / "model_cache"
    ckpt_files = sorted(ckpt_dir.glob("new_diffusion_fuzzy_PEMSD4_epoch*.tar"), reverse=True)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    model_cls = get_model_component("traffic_state_pred", "new_diffusion_fuzzy", "model")
    model = model_cls(config, runtime.data_feature)
    ckpt = torch.load(str(ckpt_files[0]), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_files[0].name}")

    return model, runtime


def test_single_step_recovery(model, runtime, device="cuda"):
    """测试不同 t 下单步去噪恢复精度。

    取真实 Y_0 → 加噪到 Y_t → 预测噪声 → 计算恢复 Ŷ_0 → 对比 Y_0。
    """
    model = model.to(device)
    dataloader = runtime.test_loader
    batch = next(iter(dataloader))
    batch.to_tensor(device)

    Y_true = batch["y"][..., :1].float()  # [B, T, N, 1]
    condition = model.encode_condition(batch["X"].float())
    adj = model.adjacency_matrix.to(device)

    test_timesteps = [10, 30, 50, 100, 150, 199]
    print("\n=== 单步恢复精度 (t → Ŷ₀ vs Y₀) ===")
    print(f"{'t':>5}  {'MSE(ε̂,ε)':>12}  {'MSE(Ŷ₀,Y₀)':>12}  {'R²(Ŷ₀,Y₀)':>12}")

    scheduler = model.diffusion_scheduler
    batch_size = Y_true.shape[0]

    with torch.no_grad():
        for t_val in test_timesteps:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(Y_true)
            Y_noisy = scheduler.add_noise(Y_true, t, noise=noise)
            eps_pred = model.noise_predictor(Y_noisy, t, condition, adj)

            mse_noise = F.mse_loss(eps_pred, noise).item()
            Y_recovered = scheduler.predict_start_from_noise(Y_noisy, t, eps_pred)
            mse_recovery = F.mse_loss(Y_recovered, Y_true).item()

            ss_res = ((Y_true - Y_recovered) ** 2).sum()
            ss_tot = ((Y_true - Y_true.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot).item()

            print(f"{t_val:>5}  {mse_noise:>12.6f}  {mse_recovery:>12.6f}  {r2:>12.4f}")

    return Y_true, condition, adj


def test_ddim_trajectory(model, Y_true, condition, adj, device="cuda"):
    """测试完整 DDIM 逆向链路的误差累积。

    从 Y_true 出发，加满噪声 Y_199，再 50 步 DDIM 去噪，对比原始 Y_0。
    """
    print("\n=== DDIM 完整逆向链路 (50步) ===")

    scheduler = model.diffusion_scheduler
    batch_size = Y_true.shape[0]

    # 从真实 Y_0 加噪到 t=199
    t_max = torch.full((batch_size,), 199, device=device, dtype=torch.long)
    Y_noisy = scheduler.add_noise(Y_true, t_max)

    # 记录逆向轨迹中的中间结果
    schedule = model._get_sampling_schedule(device)
    print(f"采样调度 ({len(schedule)}步): {schedule.cpu().tolist()}")

    # DDIM 逆向
    Y_current = Y_noisy.clone()
    with torch.no_grad():
        for i, step in enumerate(schedule):
            t = step.expand(batch_size)
            t_next = schedule[i + 1].expand(batch_size) if i < len(schedule) - 1 else None
            eps_pred = model.noise_predictor(Y_current, t, condition, adj)
            Y_current = scheduler.ddim_step(Y_current, t, eps_pred, eta=0.0, t_next=t_next)

    mse_final = F.mse_loss(Y_current, Y_true).item()
    ss_res = ((Y_true - Y_current) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()

    # 对比: 不加条件 (condition=zeros)
    zero_condition = torch.zeros_like(condition)
    Y_unc = Y_noisy.clone()
    with torch.no_grad():
        for i, step in enumerate(schedule):
            t = step.expand(batch_size)
            t_next = schedule[i + 1].expand(batch_size) if i < len(schedule) - 1 else None
            eps_pred = model.noise_predictor(Y_unc, t, zero_condition, adj)
            Y_unc = scheduler.ddim_step(Y_unc, t, eps_pred, eta=0.0, t_next=t_next)

    mse_uncond = F.mse_loss(Y_unc, Y_true).item()
    print(f"完整 DDIM 恢复: MSE={mse_final:.4f},  R²={r2:.4f}")
    print(f"无条件 DDIM:    MSE={mse_uncond:.4f}")
    print(f"条件改善倍数:   {(mse_uncond / max(mse_final, 1e-8)):.1f}x  ← (>1 表示条件有效)")

    # 统计预测值范围 vs 真实值范围
    print(f"\n真实值范围:  [{Y_true.min().item():.3f}, {Y_true.max().item():.3f}]  std={Y_true.std().item():.4f}")
    print(f"有条件下预测: [{Y_current.min().item():.3f}, {Y_current.max().item():.3f}]  std={Y_current.std().item():.4f}")
    print(f"无条件下预测: [{Y_unc.min().item():.3f}, {Y_unc.max().item():.3f}]  std={Y_unc.std().item():.4f}")


def test_prediction_vs_baseline(model, runtime, device="cuda"):
    """对比模型预测 vs 简单基线 (历史均值, 最后一步重复)。"""
    print("\n=== 预测对比 ===")
    model = model.to(device)
    dataloader = runtime.test_loader
    all_mse_model = []
    all_mse_last = []
    all_mse_mean = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # 只测 10 个 batch
                break
            batch.to_tensor(device)
            X = batch["X"].float()  # [B, T_in, N, C]
            Y = batch["y"][..., :1].float()  # [B, T_out, N, 1]

            pred = model.predict(batch)
            last_step = X[:, -1:, :, :1].expand(-1, Y.shape[1], -1, -1)
            history_mean = X[:, :, :, :1].mean(dim=1, keepdim=True).expand(-1, Y.shape[1], -1, -1)

            all_mse_model.append(F.mse_loss(pred, Y).item())
            all_mse_last.append(F.mse_loss(last_step, Y).item())
            all_mse_mean.append(F.mse_loss(history_mean, Y).item())

    print(f"模型预测 MSE:         {np.mean(all_mse_model):.4f}")
    print(f"最后步重复 MSE:       {np.mean(all_mse_last):.4f}")
    print(f"历史均值重复 MSE:     {np.mean(all_mse_mean):.4f}")
    print(f"模型 vs 最后步:       {np.mean(all_mse_last)/max(np.mean(all_mse_model), 1e-8):.1f}x  ← (<1 说明比重复最后一步还差)")
    print(f"模型 vs 历史均值:     {np.mean(all_mse_mean)/max(np.mean(all_mse_model), 1e-8):.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_id", required=True, help="Data artifact ID")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, runtime = load_trained_model(args.artifact_id)
    Y_true, condition, adj = test_single_step_recovery(model, runtime, args.device)
    test_ddim_trajectory(model, Y_true, condition, adj, args.device)
    test_prediction_vs_baseline(model, runtime, args.device)
