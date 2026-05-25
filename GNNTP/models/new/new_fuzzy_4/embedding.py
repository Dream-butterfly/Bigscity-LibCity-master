"""Time embedding utilities (unused in new_fuzzy — kept for reference).

NOTE: ``SinusoidalTimeEmbedding`` is NOT imported or used by the current
new_fuzzy model. It is a leftover from an earlier diffusion-based design.
Other packages (new_diffusion_fuzzy, new_diffusion_fuzzy_2) have their
own independent copies.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding.

    Encodes integer timesteps t ∈ [0, T-1] into a fixed-dimensional vector
    using sin/cos of exponentially scaled frequencies, as in the Transformer
    and DDPM papers.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """Encode integer timesteps into sinusoidal embeddings.

        Args:
            timesteps: [B] long tensor of integer timestep indices.

        Returns:
            [B, embedding_dim] float32 embedding.
        """
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
