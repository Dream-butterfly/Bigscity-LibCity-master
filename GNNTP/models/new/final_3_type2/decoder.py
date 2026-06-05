"""Future decoder — final_new (Fuzzy Region Transformer).

Refines learnable future queries through cross-attention over encoded history.
Three-scale processing per block:
  Temporal → Local(GCN) → Global(FRR) → Cross-Attn(history) → FFN
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .graph import GraphConvolution, FuzzyGraphConvolution
from .cell_attention import FuzzyCellAttention
from .utils import (
    apply_temporal_attention,
    apply_node_temporal_cross_attention,
)


class DecoderBlock(nn.Module):
    """Decoder block: T→L→G→Cross→FFN with Pre-LN residuals.

    Sub-layer order:
        temporal self-attn → graph conv → FRR → cross-attn(history) → FFN
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
        use_fuzzy_routing=False,
    ):
        super().__init__()
        self.use_cell_attention = use_cell_attention

        self.temporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        if use_fuzzy_graph:
            self.graph_convolution = FuzzyGraphConvolution(hidden_dim, graph_k_hop)
        else:
            self.graph_convolution = GraphConvolution(hidden_dim, graph_k_hop)
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
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
                use_fuzzy_routing=use_fuzzy_routing,
            )
            self.norm_cell = nn.LayerNorm(hidden_dim)

    def forward(self, queries, condition_features, graph_matrix,
                graph_dist=None, mu_fuzzy=None):
        """Run one decoder block.

        Args:
            queries: [B, T_out, N, D].
            condition_features: [B, T_in, N, D] encoded history.
            graph_matrix: [N, N].
            graph_dist: [N, N] hop distance.
            mu_fuzzy: [N, K_f].
        """
        # 1. Temporal self-attention
        t_out = apply_temporal_attention(queries, self.temporal_attention)
        queries = self.norm_temporal(queries + self.dropout(t_out))

        # 2. Graph convolution
        B, T, N, D = queries.shape
        g_in = queries.reshape(B * T, N, D)
        g_out = self.graph_convolution(g_in, graph_matrix)
        queries = self.norm_graph(
            queries + self.dropout(g_out.reshape(B, T, N, D)))

        # 3. Fuzzy Region Routing — global spatial interaction
        if self.use_cell_attention:
            cell_out = self.cell_attention(
                queries, graph_dist=graph_dist, mu_fuzzy=mu_fuzzy)
            cell_out = cell_out.unsqueeze(0).unsqueeze(0)   # [1, 1, N, D]
            queries = self.norm_cell(queries + cell_out)

        # 4. Cross-attention over encoded history
        cross_out = apply_node_temporal_cross_attention(
            queries, condition_features, self.cross_attention)
        queries = self.norm_cross(queries + self.dropout(cross_out))

        # 5. Feed-forward
        ffn_out = self.feed_forward(queries)
        queries = self.norm_ffn(queries + ffn_out)
        return queries


class FutureDecoder(nn.Module):
    """Decoder: learnable queries → blocks → predictions.

    Learnable future_queries [1, T_out, 1, D] broadcast to N nodes,
    refined through N × DecoderBlock, then projected to output dimension.
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
        output_window=12,
        num_nodes=1,
        use_gradient_checkpointing=False,
        use_fuzzy_graph=False,
        use_cell_attention=False,
        num_cells=8,
        cell_blend_init=0.3,
        band_center_init=1.1,
        band_width_init=0.7,
        region_transformer_layers=1,
        use_fuzzy_routing=False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.output_window = output_window
        self.num_nodes = num_nodes

        self.future_queries = nn.Parameter(
            torch.zeros(1, output_window, 1, hidden_dim))
        nn.init.trunc_normal_(self.future_queries, std=0.02)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop,
                dropout,
                use_fuzzy_graph=use_fuzzy_graph,
                use_cell_attention=use_cell_attention,
                num_cells=num_cells,
                cell_blend_init=cell_blend_init,
                band_center_init=band_center_init,
                band_width_init=band_width_init,
                region_transformer_layers=region_transformer_layers,
                use_fuzzy_routing=use_fuzzy_routing,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, condition_features, graph_matrix,
                graph_dist=None, mu_fuzzy=None):
        """Decode future from condition.

        Args:
            condition_features: [B, T_in, N, D].
            graph_matrix: [N, N].
            graph_dist: [N, N] hop distance.
            mu_fuzzy: [N, K_f].

        Returns:
            [B, T_out, N, C_out].
        """
        B = condition_features.shape[0]
        queries = self.future_queries.expand(B, -1, self.num_nodes, -1)

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                queries = checkpoint(
                    block, queries, condition_features, graph_matrix,
                    graph_dist, mu_fuzzy, use_reentrant=False)
            else:
                queries = block(queries, condition_features, graph_matrix,
                                graph_dist=graph_dist, mu_fuzzy=mu_fuzzy)

        queries = self.final_norm(queries)
        return self.output_projection(queries)
