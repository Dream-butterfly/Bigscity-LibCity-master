"""Fuzzy-guided denoiser for conditional diffusion — new_fuzzy_3 (Phase C).

Predicts noise ε̂ = ε_θ(Y_t, t, H, R) from noisy future, timestep,
encoded history condition, and fuzzy relational graph R.

Key differences from old AttentionDenoiser:
  - Uses ``FuzzyGraphConvolution`` (when use_fuzzy_graph=True) instead of
    standard ``GraphConvolution``, so the fuzzy relation R guides spatial
    message passing during denoising.
  - Receives pre-computed fuzzy relation R from outside (no per-block
    ``AdaptiveGraphLearner`` call).  R is shared across all blocks.
  - Otherwise preserves proven patterns: concat+linear condition fusion,
    blend gate (anti-shortcut), per-block timestep re-injection,
    U-Net skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .embedding import SinusoidalTimeEmbedding
from .graph import GraphConvolution, FuzzyGraphConvolution
from .utils import (
    apply_temporal_attention,
    apply_node_temporal_cross_attention,
    apply_spatiotemporal_attention,
)


# ═══════════════════════════════════════════════════════════════════
#  Denoiser Block
# ═══════════════════════════════════════════════════════════════════

class DenoiserBlock(nn.Module):
    """Denoising block: temporal self-attn → graph conv → cross-attn → [ST-attn] → FFN.

    Uses Pre-LN residual connections.  Per-block timestep re-injection
    via scale/shift modulation prevents timestep signal dilution across
    deep stacks.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ffn_hidden_dim,
        graph_k_hop,
        dropout=0.1,
        use_spatiotemporal_attention=False,
        use_fuzzy_graph=False,
    ):
        super().__init__()
        self.temporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        if use_fuzzy_graph:
            self.graph_convolution = FuzzyGraphConvolution(hidden_dim, graph_k_hop)
        else:
            self.graph_convolution = GraphConvolution(hidden_dim, graph_k_hop)
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.use_spatiotemporal_attention = use_spatiotemporal_attention
        if self.use_spatiotemporal_attention:
            self.spatiotemporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        if self.use_spatiotemporal_attention:
            self.norm_spatiotemporal = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Per-block timestep re-injection (adaLN-style scale/shift)
        self.time_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, noisy_future, condition_features, graph_matrix, timestep_emb=None):
        """Run one denoiser block.

        Args:
            noisy_future:    [B, T_out, N, D] current noisy state.
            condition_features: [B, T_in, N, D] encoded history.
            graph_matrix:    [N, N] adjacency or fuzzy relation.
            timestep_emb:    Optional [B, D] timestep embedding for per-block
                             re-injection.

        Returns:
            [B, T_out, N, D] updated features.
        """
        # Per-block timestep modulation
        if timestep_emb is not None:
            scale, shift = self.time_scale_shift(timestep_emb).chunk(2, dim=-1)
            noisy_future = noisy_future * (1.0 + scale.unsqueeze(1).unsqueeze(2)) \
                + shift.unsqueeze(1).unsqueeze(2)

        # 1. Temporal self-attention
        temporal_output = apply_temporal_attention(noisy_future, self.temporal_attention)
        noisy_future = self.norm_temporal(noisy_future + self.dropout(temporal_output))

        # 2. Graph convolution
        batch_size, time_steps, num_nodes, hidden_dim = noisy_future.shape
        graph_input = noisy_future.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        graph_output = self.graph_convolution(graph_input, graph_matrix)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        noisy_future = self.norm_graph(noisy_future + self.dropout(graph_output))

        # 3. Cross-attention over history
        cross_output = apply_node_temporal_cross_attention(
            noisy_future, condition_features, self.cross_attention
        )
        noisy_future = self.norm_cross(noisy_future + self.dropout(cross_output))

        # 4. (Optional) Spatiotemporal attention
        if self.use_spatiotemporal_attention:
            st_output = apply_spatiotemporal_attention(
                noisy_future, self.spatiotemporal_attention,
                context_sequence=condition_features,
            )
            noisy_future = self.norm_spatiotemporal(noisy_future + self.dropout(st_output))

        # 5. Feed-forward
        ffn_output = self.feed_forward(noisy_future)
        noisy_future = self.norm_ffn(noisy_future + ffn_output)
        return noisy_future


# ═══════════════════════════════════════════════════════════════════
#  Fuzzy-Guided Denoiser
# ═══════════════════════════════════════════════════════════════════

class FuzzyGuidedDenoiser(nn.Module):
    """Fuzzy-relation-guided noise predictor ε_θ(Y_t, t, H, R).

    Condition injection:
      Learnable temporal weighted pooling over history → concat + linear
      fusion BEFORE attention/graph ops.  Blend gate prevents the
      denoiser from learning an unconditional shortcut.

    Fuzzy graph:
      When ``use_fuzzy_graph=True``, the fuzzy relation R (pre-computed
      by ``FuzzyRelationalGraphLearner``) is passed through each block's
      ``FuzzyGraphConvolution``, providing a mathematically grounded
      spatial prior for the denoising process.

    U-Net skips:
      First half of blocks save features, second half add mirrored skips.
      This preserves spatial details that would be lost in deep sequential
      denoising.
    """

    def __init__(
        self,
        output_dim,
        hidden_dim,
        num_heads,
        num_layers,
        ffn_hidden_dim,
        graph_k_hop,
        dropout=0.1,
        use_spatiotemporal_attention=False,
        max_future_steps=None,
        input_window=None,
        use_gradient_checkpointing=True,
        use_fuzzy_graph=False,
    ):
        super().__init__()
        self.max_future_steps = max_future_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ── Input embedding ─────────────────────────────────────
        self.input_projection = nn.Linear(output_dim, hidden_dim)

        self.future_position_embedding = None
        if max_future_steps is not None and max_future_steps >= 1:
            self.future_position_embedding = nn.Parameter(
                torch.zeros(1, max_future_steps, 1, hidden_dim)
            )
            nn.init.trunc_normal_(self.future_position_embedding, std=0.02)

        # ── Timestep embedding ──────────────────────────────────
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Condition fusion (concat+linear, proven) ────────────
        if input_window is None or input_window < 1:
            raise ValueError("input_window must be >= 1 for condition temporal weighting.")
        self.condition_temporal_weight = nn.Parameter(torch.zeros(input_window))
        self.condition_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        # Blend gate: σ(0)=0.5 → equal weight. Model learns optimal balance.
        self.condition_blend = nn.Parameter(torch.tensor(0.0))

        # ── Denoiser blocks ─────────────────────────────────────
        self.blocks = nn.ModuleList([
            DenoiserBlock(
                hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop, dropout,
                use_spatiotemporal_attention=use_spatiotemporal_attention,
                use_fuzzy_graph=use_fuzzy_graph,
            )
            for _ in range(num_layers)
        ])

        # ── Output ──────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, noisy_future, timesteps, condition_features, graph_matrix):
        """Predict noise from noisy future given condition and timestep.

        Args:
            noisy_future:        [B, T_out, N, C_out] noisy future state.
            timesteps:           [B] diffusion timestep indices.
            condition_features:  [B, T_in, N, D] encoded history.
            graph_matrix:        [N, N] fuzzy relation or adjacency.

        Returns:
            [B, T_out, N, C_out] predicted noise ε_θ.
        """
        # ── 1. Embed noisy future ───────────────────────────────
        denoiser_input = self.input_projection(noisy_future)
        if self.future_position_embedding is not None:
            future_steps = denoiser_input.shape[1]
            if future_steps > self.max_future_steps:
                raise ValueError(
                    f"future sequence length {future_steps} exceeds "
                    f"max_future_steps={self.max_future_steps}."
                )
            denoiser_input = denoiser_input + self.future_position_embedding[:, :future_steps]

        # ── 2. Timestep embedding ───────────────────────────────
        timestep_features = self.time_projection(
            self.time_embedding(timesteps)
        ).to(dtype=denoiser_input.dtype)
        denoiser_input = denoiser_input + timestep_features.unsqueeze(1).unsqueeze(2)

        # ── 3. Condition injection (concat+linear + blend gate) ─
        temporal_weights = F.softmax(self.condition_temporal_weight, dim=0)
        condition_pooled = (
            condition_features * temporal_weights[None, :, None, None]
        ).sum(dim=1, keepdim=True)
        condition_pooled = condition_pooled.expand(
            -1, denoiser_input.size(1), -1, -1
        )
        noisy_only = denoiser_input
        fused = self.condition_fusion(
            torch.cat([denoiser_input, condition_pooled], dim=-1)
        )
        alpha = torch.sigmoid(self.condition_blend)
        denoiser_input = (1.0 - alpha) * noisy_only + alpha * fused

        # ── 4. Denoiser blocks (with U-Net skips) ───────────────
        num_layers = len(self.blocks)
        mid = num_layers // 2
        skips = []

        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                denoiser_input = checkpoint(
                    block, denoiser_input, condition_features, graph_matrix,
                    timestep_features, use_reentrant=False,
                )
            else:
                denoiser_input = block(
                    denoiser_input, condition_features, graph_matrix, timestep_features,
                )

            # Encoder half: save skip; decoder half: add mirrored skip
            if i < mid:
                skips.append(denoiser_input)
            else:
                skip_idx = num_layers - 1 - i
                if skip_idx < len(skips):
                    denoiser_input = denoiser_input + skips[skip_idx]

        # ── 5. Output ───────────────────────────────────────────
        denoiser_input = self.final_norm(denoiser_input)
        return self.output_projection(denoiser_input)
