"""Spatio-temporal encoder — final_new (Fuzzy Region Transformer).

Converts historical traffic sequences [B, Tin, N, Cin] into condition
features H [B, Tin, N, D].  Three-scale processing per block:
  Temporal: per-node self-attention
  Spatial-Local: FuzzyGCN (K-hop topology propagation)
  Spatial-Global: FRR (Fuzzy Region Routing, replaces spatial self-attn)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .graph import GraphConvolution, FuzzyGraphConvolution
from .cell_attention import FuzzyCellAttention
from .utils import apply_temporal_attention


class STEncoderBlock(nn.Module):
    """Spatio-temporal encoder block: Temporal → Local(GCN) → Global(FRR) → FFN.

    Operates on [B, T, N, D] tensors with Pre-LN residual connections.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ffn_hidden_dim,
        graph_k_hop,
        dropout=0.1,
        use_fuzzy_graph=False,
        use_cell_attention=False,
        num_cells=8,
        cell_blend_init=0.3,
        band_center_init=1.1,
        band_width_init=0.7,
        region_transformer_layers=1,
    ):
        super().__init__()
        self.use_cell_attention = use_cell_attention

        self.temporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        if use_fuzzy_graph:
            self.graph_convolution = FuzzyGraphConvolution(hidden_dim, graph_k_hop)
        else:
            self.graph_convolution = GraphConvolution(hidden_dim, graph_k_hop)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── Fuzzy Region Routing ──
        if use_cell_attention:
            self.cell_attention = FuzzyCellAttention(
                hidden_dim=hidden_dim,
                num_cells=num_cells,
                cell_blend_init=cell_blend_init,
                band_center_init=band_center_init,
                band_width_init=band_width_init,
                region_transformer_layers=region_transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.norm_cell = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_features, graph_matrix,
                graph_dist=None, mu_fuzzy=None):
        """Run one encoder block.

        Args:
            sequence_features: [B, T, N, D].
            graph_matrix: [N, N] adjacency or fuzzy relation R.
            graph_dist: [N, N] hop distance for band-pass gate.
            mu_fuzzy: [N, K_f] fuzzy memberships for conditioning.
        """
        # 1. Temporal self-attention (per-node)
        temporal_out = apply_temporal_attention(sequence_features, self.temporal_attention)
        sequence_features = self.norm_temporal(
            sequence_features + self.dropout(temporal_out))

        # 2. Graph convolution (shared across time)
        B, T, N, D = sequence_features.shape
        g_in = sequence_features.reshape(B * T, N, D)
        g_out = self.graph_convolution(g_in, graph_matrix)
        sequence_features = self.norm_graph(
            sequence_features + self.dropout(g_out.reshape(B, T, N, D)))

        # 3. Fuzzy Region Routing — global spatial interaction
        if self.use_cell_attention:
            # FRR handles 4D internally (dispatch + temporal aggregation)
            cell_out = self.cell_attention(
                sequence_features,
                graph_dist=graph_dist,
                mu_fuzzy=mu_fuzzy,
            )  # [N, D]
            cell_out = cell_out.unsqueeze(0).unsqueeze(0)   # [1, 1, N, D]
            sequence_features = self.norm_cell(
                sequence_features + cell_out)

        # 4. Feed-forward
        ffn_out = self.feed_forward(sequence_features)
        sequence_features = self.norm_ffn(sequence_features + ffn_out)
        return sequence_features


class STEncoder(nn.Module):
    """Condition encoder: X [B, Tin, N, Cin] → H [B, Tin, N, D].

    Input projection + temporal position encoding
    → N × STEncoderBlock → LayerNorm.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        ffn_hidden_dim,
        graph_k_hop,
        dropout=0.1,
        use_temporal_position_embedding=True,
        max_time_steps=None,
        use_gradient_checkpointing=True,
        use_fuzzy_graph=False,
        use_cell_attention=False,
        num_cells=8,
        cell_blend_init=0.3,
        band_center_init=1.1,
        band_width_init=0.7,
        region_transformer_layers=1,
    ):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            STEncoderBlock(
                hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop,
                dropout,
                use_fuzzy_graph=use_fuzzy_graph,
                use_cell_attention=use_cell_attention,
                num_cells=num_cells,
                cell_blend_init=cell_blend_init,
                band_center_init=band_center_init,
                band_width_init=band_width_init,
                region_transformer_layers=region_transformer_layers,
            )
            for _ in range(num_layers)
        ])
        self.temporal_position_embedding = None
        if use_temporal_position_embedding:
            if max_time_steps is None or max_time_steps < 1:
                raise ValueError(
                    "max_time_steps must be >= 1 when temporal position "
                    "embedding is enabled."
                )
            self.temporal_position_embedding = nn.Parameter(
                torch.zeros(1, max_time_steps, 1, hidden_dim))
            nn.init.trunc_normal_(self.temporal_position_embedding, std=0.02)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_sequence, graph_matrix,
                graph_dist=None, mu_fuzzy=None):
        """Encode X [B, Tin, N, Cin] → H [B, Tin, N, D].

        Args:
            history_sequence: [B, Tin, N, Cin].
            graph_matrix: [N, N].
            graph_dist: [N, N] hop distance.
            mu_fuzzy: [N, K_f].

        Returns:
            [B, Tin, N, D].
        """
        x = self.input_projection(history_sequence)
        if self.temporal_position_embedding is not None:
            T = x.shape[1]
            if T > self.max_time_steps:
                raise ValueError(
                    f"history length {T} exceeds max_time_steps={self.max_time_steps}")
            x = x + self.temporal_position_embedding[:, :T]

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, graph_matrix,
                               graph_dist, mu_fuzzy, use_reentrant=False)
            else:
                x = block(x, graph_matrix, graph_dist=graph_dist, mu_fuzzy=mu_fuzzy)

        return self.final_norm(x)
