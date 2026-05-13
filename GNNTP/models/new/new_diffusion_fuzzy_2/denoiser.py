"""Attention-based denoiser for conditional diffusion.

Predicts noise ε̂ = ε_θ(Y_t, t, H, A) from noisy future, timestep,
condition features, and (optionally adaptive) adjacency matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .embedding import SinusoidalTimeEmbedding
from .graph import GraphConvolution, AdaptiveGraphLearner
from .utils import (
    apply_temporal_attention,
    apply_node_temporal_cross_attention,
    apply_spatiotemporal_attention,
)


class DenoiserBlock(nn.Module):
    """Denoising block: temporal self-attn → graph conv → cross-attn → [ST-attn] → FFN.

    Operates with Pre-LN residual connections on [B, T_out, N, D] tensors,
    with cross-attention attending over encoded history [B, T_in, N, D].
    """

    def __init__(
            self,
            hidden_dim,
            num_heads,
            ffn_hidden_dim,
            graph_k_hop,
            dropout=0.1,
            use_spatiotemporal_attention=False,
    ):
        super().__init__()
        self.temporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
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

        # Per-block timestep re-injection via scale/shift modulation.
        # Analogous to adaLN in DiT but simpler: applies once at block entry
        # rather than replacing every LayerNorm, trading some expressivity
        # for fewer parameters and training stability.
        self.time_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, noisy_future, condition_features, adjacency_matrix, timestep_emb=None):
        """Run one denoiser block.

        Args:
            noisy_future: [B, T_out, N, D] current noisy state.
            condition_features: [B, T_in, N, D] encoded history.
            adjacency_matrix: [N, N] or [B, N, N].
            timestep_emb: Optional [B, D] timestep embedding for per-block
                re-injection (scale/shift modulation).

        Returns:
            [B, T_out, N, D] updated features.
        """
        # Timestep re-injection: scale/shift modulates features per block.
        # Prevents timestep signal dilution across deep denoiser stacks.
        if timestep_emb is not None:
            scale, shift = self.time_scale_shift(timestep_emb).chunk(2, dim=-1)
            # [B, D] → [B, 1, 1, D] for broadcast over time and node dims
            noisy_future = noisy_future * (1.0 + scale.unsqueeze(1).unsqueeze(2)) + shift.unsqueeze(1).unsqueeze(2)

        # 1. Temporal self-attention
        temporal_output = apply_temporal_attention(noisy_future, self.temporal_attention)
        noisy_future = self.norm_temporal(noisy_future + self.dropout(temporal_output))

        # 2. Graph convolution
        batch_size, time_steps, num_nodes, hidden_dim = noisy_future.shape
        graph_input = noisy_future.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        graph_output = self.graph_convolution(graph_input, adjacency_matrix)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        noisy_future = self.norm_graph(noisy_future + self.dropout(graph_output))

        # 3. Cross-attention over history
        cross_output = apply_node_temporal_cross_attention(
            noisy_future, condition_features, self.cross_attention
        )
        noisy_future = self.norm_cross(noisy_future + self.dropout(cross_output))

        # 4. (Optional) Spatiotemporal attention
        if self.use_spatiotemporal_attention:
            spatiotemporal_output = apply_spatiotemporal_attention(
                noisy_future, self.spatiotemporal_attention, context_sequence=condition_features
            )
            noisy_future = self.norm_spatiotemporal(
                noisy_future + self.dropout(spatiotemporal_output)
            )

        # 5. Feed-forward
        ffn_output = self.feed_forward(noisy_future)
        noisy_future = self.norm_ffn(noisy_future + ffn_output)
        return noisy_future


class AttentionDenoiser(nn.Module):
    """Attention-based noise predictor ε_θ(Y_t, t, H, A).

    Condition injection: learnable temporal weighted pooling over all history steps →
    concat + linear fusion. (FiLM was tried but zero-init caused condition signal
    starvation early in training.)

    When adaptive graph is enabled, the graph structure is updated per denoiser
    block based on the current noisy features and timestep embedding.
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
            use_temporal_position_embedding=True,
            max_future_steps=None,
            input_window=None,
            use_gradient_checkpointing=True,
            adaptive_graph_enabled=False,
            adaptive_graph_embed_dim=32,
            adaptive_graph_topk=None,
            adaptive_graph_blend_init=0.5,
            fuzzy_graph_enabled=False,
            fuzzy_graph_num_sets=3,
            fuzzy_graph_sigma_init=0.7,
            num_nodes=None,
            static_adjacency=None,
    ):
        super().__init__()
        self.max_future_steps = max_future_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_projection = nn.Linear(output_dim, hidden_dim)
        self.future_position_embedding = None
        if use_temporal_position_embedding:
            if max_future_steps is None or max_future_steps < 1:
                raise ValueError("max_future_steps must be >= 1 when temporal position embedding is enabled.")
            self.future_position_embedding = nn.Parameter(
                torch.zeros(1, max_future_steps, 1, hidden_dim)
            )
            nn.init.trunc_normal_(self.future_position_embedding, std=0.02)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if input_window is None or input_window < 1:
            raise ValueError("input_window must be >= 1 for condition temporal weighting.")
        self.condition_temporal_weight = nn.Parameter(torch.zeros(input_window))
        self.condition_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DenoiserBlock(
                    hidden_dim,
                    num_heads,
                    ffn_hidden_dim,
                    graph_k_hop,
                    dropout,
                    use_spatiotemporal_attention=use_spatiotemporal_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.adaptive_graph_learner = None
        if adaptive_graph_enabled:
            if num_nodes is None or static_adjacency is None:
                raise ValueError("num_nodes and static_adjacency are required when adaptive graph is enabled.")
            self.adaptive_graph_learner = AdaptiveGraphLearner(
                num_nodes=num_nodes,
                hidden_dim=hidden_dim,
                static_adjacency=static_adjacency,
                embed_dim=adaptive_graph_embed_dim,
                top_k=adaptive_graph_topk,
                blend_init=adaptive_graph_blend_init,
                fuzzy_enabled=fuzzy_graph_enabled,
                fuzzy_num_sets=fuzzy_graph_num_sets,
                fuzzy_sigma_init=fuzzy_graph_sigma_init,
            )

    def forward(self, noisy_future, timesteps, condition_features, adjacency_matrix, return_last_adjacency=False):
        """Predict noise from noisy future given condition and timestep.

        Args:
            noisy_future: [B, T_out, N, C_out] noisy future state.
            timesteps: [B] diffusion timestep indices.
            condition_features: [B, T_in, N, D] encoded history conditions.
            adjacency_matrix: [N, N] or [B, N, N] initial adjacency.
            return_last_adjacency: If True, return (noise, final_adjacency) tuple.

        Returns:
            [B, T_out, N, C_out] predicted noise, or tuple with adjacency.
        """
        denoiser_input = self.input_projection(noisy_future)
        if self.future_position_embedding is not None:
            future_steps = denoiser_input.shape[1]
            if future_steps > self.max_future_steps:
                raise ValueError(
                    f"future sequence length {future_steps} exceeds max_future_steps={self.max_future_steps}."
                )
            denoiser_input = denoiser_input + self.future_position_embedding[:, :future_steps]
        timestep_features = self.time_projection(self.time_embedding(timesteps)).to(dtype=denoiser_input.dtype)
        denoiser_input = denoiser_input + timestep_features.unsqueeze(1).unsqueeze(2)

        # Condition injection: learnable temporal weighted pooling + concat+linear
        temporal_weights = F.softmax(self.condition_temporal_weight, dim=0)  # [T_in]
        condition_pooled = (condition_features * temporal_weights[None, :, None, None]).sum(dim=1, keepdim=True)
        condition_pooled = condition_pooled.expand(-1, denoiser_input.size(1), -1, -1)
        denoiser_input = self.condition_fusion(torch.cat([denoiser_input, condition_pooled], dim=-1))

        # U-Net style skip connections: first half blocks encode (save skips),
        # second half decode (add mirrored skip). Preserves spatial details
        # that would otherwise be lost in deep sequential denoising.
        num_layers = len(self.blocks)
        mid = num_layers // 2
        skips = []

        current_adjacency = adjacency_matrix
        for i, block in enumerate(self.blocks):
            if self.adaptive_graph_learner is not None:
                current_adjacency = self.adaptive_graph_learner(
                    denoiser_input, timestep_embedding=timestep_features
                )
            if self.use_gradient_checkpointing and self.training:
                denoiser_input = checkpoint(
                    block, denoiser_input, condition_features, current_adjacency, timestep_features,
                    use_reentrant=False
                )
            else:
                denoiser_input = block(
                    denoiser_input, condition_features, current_adjacency, timestep_features
                )

            # Encoder half: save skip; Decoder half: add mirrored skip
            if i < mid:
                skips.append(denoiser_input)
            else:
                skip_idx = num_layers - 1 - i
                if skip_idx < len(skips):
                    denoiser_input = denoiser_input + skips[skip_idx]

        denoiser_input = self.final_norm(denoiser_input)
        denoised_output = self.output_projection(denoiser_input)
        if return_last_adjacency:
            return denoised_output, current_adjacency
        return denoised_output
