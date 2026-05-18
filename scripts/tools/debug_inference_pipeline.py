"""诊断推理管线：用训练数据跑预测，检查中间值和最终输出。

用法：
  uv run scripts/tools/debug_inference_pipeline.py \
      --model new_diffusion_fuzzy_2 \
      --dataset PEMSD4 \
      --artifact_id da_20260421_143022__PEMSD4__TrafficStateDataset__abc12345

不加 --artifact_id 则从 dataset 构建 DataRuntime。
新增 --disable_time_scale_shift 临时禁掉 per-block timestep 调制，
用于对比排查新加 time_scale_shift 是否为噪声来源。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data import build_dataset_runtime, build_artifact_runtime
from GNNTP.models.registry import get_model_class
from GNNTP.utils.paths import OUTPUT_ROOT


def _seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarise(tensor, name, channel=0):
    """打印 tensor 的统计信息。"""
    if tensor is None:
        print(f"  {name}: None")
        return
    t = tensor.detach().float()
    if t.dim() >= 4:  # [B, T, N, C]
        t = t[..., channel]
    elif t.dim() == 3:  # [B, N, C]
        t = t[..., channel]
    shape_str = str(list(tensor.shape))
    print(
        f"  {name:30s}  shape={shape_str:22s}  "
        f"min={t.min().item():.4f}  max={t.max().item():.4f}  "
        f"mean={t.mean().item():.4f}  std={t.std().item():.4f}  "
        f"abs_mean={t.abs().mean().item():.4f}"
    )


def _check_noise_level(y_pred, y_true, scaler, threshold=0.5):
    """判断预测值是否接近纯噪声。

    返回 (is_noise, r2_approx, snr_approx)。
    r2_approx ≈ 1 - Var(residual)/Var(truth) 粗略衡量。
    """
    yp = y_pred.cpu().numpy()
    yt = y_true.cpu().numpy()
    # 反归一化
    if scaler is not None:
        yp = scaler.inverse_transform(yp)
        yt = scaler.inverse_transform(yt)
    # 逐样本 R² 近似
    ss_res = ((yt - yp) ** 2).sum(axis=(1, 2, 3))
    ss_tot = ((yt - yt.mean(axis=(1, 2, 3), keepdims=True)) ** 2).sum(axis=(1, 2, 3))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    r2_mean = float(np.mean(r2))

    # SNR
    signal_power = (yp ** 2).mean()
    noise_power = ((yp - yt) ** 2).mean()
    snr = signal_power / (noise_power + 1e-8)

    is_noise = r2_mean < threshold
    return is_noise, r2_mean, float(snr)


def main():
    parser = argparse.ArgumentParser(description="诊断扩散模型推理管线")
    parser.add_argument("--model", default="new_diffusion_fuzzy_2")
    parser.add_argument("--dataset", default="PEMSD4")
    parser.add_argument("--dataset_class", default="TrafficStateDataset")
    parser.add_argument("--task", default="traffic_state_pred")
    parser.add_argument("--artifact_id", default="", help="指定则用数据工件，否则实时构建 dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_time_scale_shift", action="store_true",
                        help="临时禁掉 DenoiserBlock 的 time_scale_shift 调制用于对比")
    parser.add_argument("--num_sampling_steps_override", type=int, default=0,
                        help="覆盖 num_sampling_steps (0=不改)")
    parser.add_argument("--run_id", default="", help="加载训练好的 checkpoint（需配合 --epoch）")
    parser.add_argument("--epoch", type=int, default=0, help="要加载的 checkpoint epoch")
    args = parser.parse_args()

    _seed()
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model} | Dataset: {args.dataset}")

    # ═══════════════════════════════════════════════════════════
    # 1. 构建运行时
    # ═══════════════════════════════════════════════════════════
    config = ConfigParser(args.task, args.model, args.dataset)
    config["device"] = device

    if args.artifact_id:
        runtime = build_artifact_runtime(
            config, task=args.task, model_name=args.model,
            artifact_id=args.artifact_id, force_reuse=True,
        )
    else:
        runtime = build_dataset_runtime(config)

    if args.num_sampling_steps_override > 0:
        config["num_sampling_steps"] = args.num_sampling_steps_override

    data_feature = runtime.data_feature
    scaler = data_feature.get("scaler")
    adjacency = data_feature.get("adj_mx")
    num_nodes = data_feature.get("num_nodes")
    output_dim = data_feature.get("output_dim", 1)
    input_window = config.get("input_window", 12)
    output_window = config.get("output_window", 12)
    sampling_steps = config.get("num_sampling_steps", 50)
    diffusion_steps = config.get("diffusion_steps", 200)

    print(f"\n── 数据信息 ──────────────────────────────────────────────")
    print(f"  num_nodes={num_nodes}  input_window={input_window}  "
          f"output_window={output_window}  output_dim={output_dim}")
    print(f"  diffusion_steps={diffusion_steps}  num_sampling_steps={sampling_steps}")
    print(f"  scaler={type(scaler).__name__}  adj_shape={adjacency.shape}")

    # ═══════════════════════════════════════════════════════════
    # 2. 加载模型
    # ═══════════════════════════════════════════════════════════
    ModelClass = get_model_class(args.task, args.model)
    model = ModelClass(config, data_feature).to(device)
    model.eval()

    # ── 加载训练好的 checkpoint ────────────────────────────
    if args.run_id and args.epoch > 0:
        ckpt_path = os.path.join(
            OUTPUT_ROOT, args.run_id, "model_cache",
            f"{args.model}_{args.dataset}_epoch{args.epoch}.tar",
        )
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Checkpoint not found: {ckpt_path}")
        else:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\n✓ Loaded checkpoint: epoch={args.epoch}  "
                  f"val_loss={checkpoint.get('best_val_loss', 'N/A')}")
    elif not args.run_id:
        print(f"\n⚠️  未指定 --run_id，使用随机初始化模型（仅用于检查管线结构）")

    # ── 临时禁掉 time_scale_shift ──────────────────────────
    if args.disable_time_scale_shift:
        if hasattr(model, 'noise_predictor') and hasattr(model.noise_predictor, 'blocks'):
            n_disabled = 0
            for block in model.noise_predictor.blocks:
                if hasattr(block, 'time_scale_shift'):
                    block._orig_time_scale_shift = block.time_scale_shift
                    block.time_scale_shift = None
                    n_disabled += 1
            print(f"\n⚠️  已禁掉 {n_disabled} 个 DenoiserBlock 的 time_scale_shift")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n── 模型信息 ──────────────────────────────────────────────")
    print(f"  total_params={total_params:,}  trainable={trainable_params:,}")

    # ═══════════════════════════════════════════════════════════
    # 3. 取一个训练 batch 跑推理
    # ═══════════════════════════════════════════════════════════
    train_loader = runtime.train_loader
    batch = next(iter(train_loader))
    batch.to_tensor(device)

    X = batch["X"]       # [B, Tin, N, Cin]
    y_true = batch["y"]  # [B, Tout, N, Cout]
    B = X.shape[0]

    print(f"\n── 输入 batch ────────────────────────────────────────────")
    _summarise(X, "X (history)", channel=0)
    _summarise(y_true, "y_true (future)", channel=0)

    # ═══════════════════════════════════════════════════════════
    # 3a. 编码器输出
    # ═══════════════════════════════════════════════════════════
    print(f"\n── Encoder 中间值 ────────────────────────────────────────")
    with torch.no_grad():
        condition = model.encode_condition(X)
    _summarise(condition, "condition H", channel=0)
    # 条件特征的激活饱和检查
    cond_saturated = (condition.abs() > 2.0).float().mean().item()
    print(f"  condition 饱和率 (|x|>2): {cond_saturated:.4f}  "
          f"{'⚠️  接近饱和' if cond_saturated > 0.5 else '✓'}")

    # ═══════════════════════════════════════════════════════════
    # 3b. 去噪器单步输出（t=最大值）
    # ═══════════════════════════════════════════════════════════
    print(f"\n── Denoiser 单步输出 (t={diffusion_steps-1}) ─────────────")
    with torch.no_grad():
        noisy_future = torch.randn(B, output_window, num_nodes, output_dim, device=device)
        t_max = torch.full((B,), diffusion_steps - 1, device=device, dtype=torch.long)
        adj = model.adjacency_matrix.to(device)
        predicted_noise = model.noise_predictor(noisy_future, t_max, condition, adj)

    _summarise(noisy_future, "noisy_future Y_t", channel=0)
    _summarise(predicted_noise, "predicted_noise ε̂", channel=0)
    noise_mean_abs = predicted_noise.abs().mean().item()
    status = "✓ 正常" if noise_mean_abs > 0.1 else \
             "⚠️  接近零 (梯度消失?)" if noise_mean_abs > 0.001 else \
             "🔴 几乎为零！去噪器输出坍缩！"
    print(f"  |ε̂|_mean = {noise_mean_abs:.6f}  →  {status}")

    # ═══════════════════════════════════════════════════════════
    # 3c. 检查每个 block 内部中间值的衰减/爆炸
    # ═══════════════════════════════════════════════════════════
    print(f"\n── Denoiser Block 逐层中间值 ─────────────────────────────")
    with torch.no_grad():
        denoiser_input = model.noise_predictor.input_projection(noisy_future)
        if model.noise_predictor.future_position_embedding is not None:
            denoiser_input = denoiser_input + \
                model.noise_predictor.future_position_embedding[:, :output_window]
        timestep_features = model.noise_predictor.time_projection(
            model.noise_predictor.time_embedding(t_max)
        ).to(dtype=denoiser_input.dtype)

        # 条件注入
        temporal_weights = torch.softmax(
            model.noise_predictor.condition_temporal_weight, dim=0
        )
        condition_pooled = (condition * temporal_weights[None, :, None, None]).sum(
            dim=1, keepdim=True
        )
        condition_pooled = condition_pooled.expand(-1, output_window, -1, -1)
        denoiser_input = model.noise_predictor.condition_fusion(
            torch.cat([denoiser_input, condition_pooled], dim=-1)
        )

        _summarise(denoiser_input, "denoiser_input (after fusion)", channel=0)
        _summarise(timestep_features, "timestep_features", channel=0)

        current_adj = adj
        for i, block in enumerate(model.noise_predictor.blocks):
            _summarise(denoiser_input, f"  block[{i}] input", channel=0)
            if model.noise_predictor.adaptive_graph_learner is not None:
                current_adj = model.noise_predictor.adaptive_graph_learner(
                    denoiser_input,
                    timestep_embedding=timestep_features,
                )
            denoiser_input = block(
                denoiser_input, condition, current_adj, timestep_features
            )
            _summarise(denoiser_input, f"  block[{i}] output", channel=0)

    # ═══════════════════════════════════════════════════════════
    # 3d. 完整逆向扩散
    # ═══════════════════════════════════════════════════════════
    print(f"\n── 完整逆向扩散 ({sampling_steps} steps) ────────────────")
    with torch.no_grad():
        y_pred = model.predict(batch)

    _summarise(y_pred, "y_pred (预测)", channel=0)
    _summarise(y_true, "y_true (真实)", channel=0)
    _summarise(y_pred - y_true, "residual (pred - true)", channel=0)

    # 判断
    is_noise, r2, snr = _check_noise_level(y_pred, y_true, scaler)
    print(f"\n── 诊断结论 ──────────────────────────────────────────────")
    print(f"  R² (近似) = {r2:.4f}")
    print(f"  SNR       = {snr:.2f}")
    if is_noise:
        print(f"  🔴 预测接近噪声！R² < 0.5，推理管线或模型权重有问题。")
        print(f"     建议: 1) 检查 loss 曲线  2) 检查 checkpoint 是否正确加载")
    elif r2 < 0.0:
        print(f"  🔴 R² < 0 (比均值预测还差)，模型完全失效。")
    elif r2 < 0.5:
        print(f"  🟡 R² 偏低，模型学到了部分信号但远不够。")
    else:
        print(f"  🟢 R² 正常，推理管线正常。如果 loss 不降，问题在训练阶段。")

    # ═══════════════════════════════════════════════════════════
    # 3e. 恢复 time_scale_shift
    # ═══════════════════════════════════════════════════════════
    if args.disable_time_scale_shift:
        for block in model.noise_predictor.blocks:
            if hasattr(block, '_orig_time_scale_shift'):
                block.time_scale_shift = block._orig_time_scale_shift
                del block._orig_time_scale_shift

    print(f"\nDone.")


if __name__ == "__main__":
    main()
