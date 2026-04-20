"""Standalone PyTorch training script for new_diffusion."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.models.new.new_diffusion.model import NewDiffusion


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "GNNTP" / "models" / "new" / "new_diffusion" / "config.json"


class TrafficNpzDataset(Dataset):
    """Traffic dataset wrapper for NPZ files with X and y arrays."""

    def __init__(self, npz_path):
        raw_data = np.load(npz_path)
        self.history = torch.tensor(raw_data["X"], dtype=torch.float32)
        self.future = torch.tensor(raw_data["y"], dtype=torch.float32)
        if "A" in raw_data:
            self.adjacency = torch.tensor(raw_data["A"], dtype=torch.float32)
        else:
            node_count = self.history.shape[2]
            self.adjacency = torch.eye(node_count, dtype=torch.float32)

    def __len__(self):
        """Return dataset length."""
        return self.history.size(0)

    def __getitem__(self, index):
        """Return one sample in framework-like batch format."""
        return {"X": self.history[index], "y": self.future[index]}


def build_model(config_path, adjacency_matrix, device, feature_dim, output_dim):
    """Create NewDiffusion model from config JSON."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["device"] = device

    data_feature = {
        "adj_mx": adjacency_matrix.cpu().numpy(),
        "num_nodes": adjacency_matrix.size(0),
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "scaler": None,
    }
    return NewDiffusion(config, data_feature).to(device), config


def train(args):
    """Run model training and save trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = TrafficNpzDataset(args.data_npz)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model, model_config = build_model(
        args.config_json,
        dataset.adjacency,
        device,
        feature_dim=dataset.history.size(-1),
        output_dim=dataset.future.size(-1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate if args.learning_rate is not None else model_config.get("learning_rate", 1e-3),
        weight_decay=model_config.get("weight_decay", 1e-4),
    )

    model.train()
    for epoch_index in range(args.epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mean_loss = epoch_loss / max(len(data_loader), 1)
        print(f"Epoch {epoch_index + 1}/{args.epochs} - diffusion_loss: {mean_loss:.6f}")

    torch.save({"state_dict": model.state_dict(), "config": model_config}, args.output_ckpt)
    print(f"Saved checkpoint to {args.output_ckpt}")

    model.eval()
    with torch.no_grad():
        first_batch = next(iter(data_loader))
        first_batch = {key: value.to(device) for key, value in first_batch.items()}
        sampled_prediction = model.sample(
            first_batch["X"], num_samples=args.num_samples, return_all=False
        )
    print(f"Sampled prediction tensor shape: {tuple(sampled_prediction.shape)}")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train new_diffusion with NPZ data.")
    parser.add_argument("--data_npz", type=str, required=True, help="Path to NPZ file containing X, y, and optional A.")
    parser.add_argument(
        "--config_json",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to model config JSON.",
    )
    parser.add_argument("--output_ckpt", type=str, default="new_diffusion.pth", help="Checkpoint output path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--num_samples", type=int, default=4, help="Sampling count for prediction demo.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
