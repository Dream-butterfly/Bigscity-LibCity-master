"""Minimal smoke test for the LibCity STGformer wrapper."""

import torch

from libcity.models.traffic_speed_prediction.STGformer.model import STGformer


class _DummyScaler:
    def inverse_transform(self, x):
        return x


if __name__ == "__main__":
    config = {
        "device": torch.device("cpu"),
        "input_window": 12,
        "output_window": 12,
        "input_dim": 1,
        "output_dim": 1,
        "steps_per_day": 288,
        "input_embedding_dim": 24,
        "tod_embedding_dim": 12,
        "dow_embedding_dim": 12,
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 12,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "mlp_ratio": 2,
        "use_mixed_proj": True,
        "dropout_a": 0.05,
        "kernel_size": [1],
    }
    data_feature = {
        "scaler": _DummyScaler(),
        "adj_mx": torch.eye(307).numpy(),
        "num_nodes": 307,
        "feature_dim": 9,
        "output_dim": 1,
    }
    model = STGformer(config, data_feature)
    batch_x = torch.zeros(2, 12, 307, 9)
    batch_x[..., 0] = torch.randn(2, 12, 307)
    batch_x[..., 1] = torch.rand(2, 12, 307)
    dow_index = torch.randint(0, 7, (2, 12, 307))
    batch_x[..., 2:9] = torch.nn.functional.one_hot(dow_index, num_classes=7).float()
    batch = {
        "X": batch_x,
        "y": torch.randn(2, 12, 307, 1),
    }
    output = model(batch)
    print(output.shape)

    minimal_batch = {
        "X": torch.randn(2, 12, 307, 1),
        "y": torch.randn(2, 12, 307, 1),
    }
    minimal_output = model(minimal_batch)
    print(minimal_output.shape)
