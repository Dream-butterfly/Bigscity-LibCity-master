"""
过拟合测试：new_diffusion_fuzzy

用极小数据子集（默认 256 样本 = 8 batches × 32）训练模型，验证架构能否学习。

如果 loss 能从 ~1.0 降到 <0.3 → 架构可行，只需调参。
如果 loss 纹丝不动 → 有根本性 bug。

用法:
  cd /path/to/LibCity
  python scripts/tools/overfit_test_diffusion_fuzzy.py \
      --dataset PEMSD4 \
      --batches 8 \
      --epochs 200 \
      --lr 1e-3
"""

import argparse
import sys
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data.dataloader import DataLoader

from GNNTP.config_parser import ConfigParser
from GNNTP.data import build_dataset_runtime
from GNNTP.data.core.batch import Batch
from GNNTP.data.core.list_dataset import ListDataset
from GNNTP.utils import get_model, set_random_seed, get_logger


def main():
    parser = argparse.ArgumentParser(description="Diffusion Fuzzy overfitting test")
    parser.add_argument("--dataset", type=str, default="PEMSD4")
    parser.add_argument("--batches", type=int, default=8,
                        help="Number of batches to use (batch_size=32 → ~256 samples)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.seed)

    # ── 1. Build config ──────────────────────────────────────────
    config = ConfigParser(
        task="traffic_state_pred",
        model="new_diffusion_fuzzy",
        dataset=args.dataset,
        config_file="GNNTP/models/new/new_diffusion_fuzzy/config",
        train=True,
    )

    # Override for overfitting test: disable complexity, boost LR
    config.config["max_epoch"] = args.epochs
    config.config["learning_rate"] = args.lr
    config.config["use_amp"] = False
    config.config["use_gradient_checkpointing"] = False
    config.config["use_early_stop"] = False
    config.config["patience"] = args.epochs + 1
    config.config["clip_grad_norm"] = True
    config.config["max_grad_norm"] = 3.0
    config.config["gpu"] = not args.no_cuda
    config.config["gpu_id"] = args.gpu_id
    config._init_device()

    device = config["device"]
    logger = get_logger(config)
    batch_size = config.config.get("batch_size", 32)
    n_samples = args.batches * batch_size
    logger.info("Overfitting test: dataset=%s batches=%d epochs=%d lr=%g device=%s",
                args.dataset, args.batches, args.epochs, args.lr, device)

    # ── 2. Build full dataset ────────────────────────────────────
    runtime = build_dataset_runtime(config)
    data_feature = runtime.data_feature
    feature_name = runtime.feature_name
    feature_name = dict(feature_name) if hasattr(feature_name, "items") else feature_name

    # Grab first N samples from the raw training data
    full_train_data = runtime.train_loader.dataset
    if hasattr(full_train_data, "__len__"):
        logger.info("Full training set: %d samples", len(full_train_data))

    # Collect only the first N samples
    subset_items = []
    for i, item in enumerate(full_train_data):
        if i >= n_samples:
            break
        subset_items.append(item)

    # Manually build a Batch-based DataLoader (same as LibCity's collator)
    def _collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(item)
        return batch

    tiny_dataset = ListDataset(subset_items)
    tiny_loader = DataLoader(
        dataset=tiny_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collator,
        shuffle=True,
        drop_last=False,
    )

    logger.info("Training on %d samples, %d batches/epoch",
                len(subset_items), len(tiny_loader))

    # ── 3. Build model ───────────────────────────────────────────
    model = get_model(config, data_feature).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", n_params)

    # ── 4. Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.config.get("weight_decay", 1e-4),
    )

    # ── 5. Training loop ─────────────────────────────────────────
    model.train()
    best_loss = float("inf")
    loss_history = []

    print("\n--- Training (loss < 0.3 = PASS) ---")
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        grad_norm = 0.0
        for batch in tiny_loader:
            batch.to_tensor(device)
            optimizer.zero_grad()

            loss = model.calculate_loss(batch)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.config["max_grad_norm"]
            )
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            pct = (loss_history[0] - avg_loss) / max(loss_history[0], 1e-8) * 100
            logger.info(
                "epoch %4d | loss=%.6f  best=%.6f  Δ=%.1f%%  |g|=%.2f  step=%d",
                epoch, avg_loss, best_loss, pct, grad_norm,
                model._train_step_count,
            )

    # ── 6. Verdict ───────────────────────────────────────────────
    initial = loss_history[0]
    final = loss_history[-1]
    improvement = (initial - final) / max(initial, 1e-8)

    print("\n" + "=" * 60)
    print("OVERFITTING TEST RESULT")
    print("=" * 60)
    print(f"  Dataset:         {args.dataset}")
    print(f"  Samples:         {len(subset_items)} ({args.batches} batches × {batch_size})")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Model params:    {n_params:,}")
    print(f"  Initial loss:    {initial:.6f}")
    print(f"  Final loss:      {final:.6f}")
    print(f"  Best loss:       {best_loss:.6f}")
    print(f"  Improvement:     {improvement * 100:.1f}%")
    print()

    if final < 0.3:
        verdict = "✅  PASS — model CAN learn (loss < 0.3)"
    elif improvement > 0.5:
        verdict = "⚠️   PARTIAL — model IS learning but slowly (architecture OK, tune hyperparams)"
    elif improvement > 0.15:
        verdict = "⚠️   WEAK — marginal improvement (check gradient flow / data)"
    else:
        verdict = "❌  FAIL — model is NOT learning (fundamental bug in architecture or data pipeline)"

    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    return 0 if improvement > 0.15 else 1


if __name__ == "__main__":
    sys.exit(main())
