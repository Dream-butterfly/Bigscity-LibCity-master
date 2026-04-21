"""Time embedding utilities for diffusion models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """Encode integer timesteps into sinusoidal embeddings."""
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding
