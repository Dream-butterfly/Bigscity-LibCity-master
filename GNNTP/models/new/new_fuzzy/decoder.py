"""Future decoder for deterministic fuzzy-graph traffic prediction.

Refines learnable future queries through cross-attention over encoded history.
Encoder-decoder architecture with cross-attention decoding — no diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .graph import GraphConvolution, AdaptiveGraphLearner
from .utils import (
    apply_temporal_attention,
    apply_node_temporal_cross_attention,
    apply_spatiotemporal_attention,
)


class DecoderBlock(nn.Module):
    """Decoder block: temporal self-attn → graph conv → cross-attn → [ST-attn] → FFN.

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
        if use_spatiotemporal_attention:
            self.spatiotemporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        if use_spatiotemporal_attention:
            self.norm_spatiotemporal = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, condition_features, adjacency_matrix):
        """Run one decoder block.

        Args:
            queries: [B, T_out, N, D] current future queries.
            condition_features: [B, T_in, N, D] encoded history.
            adjacency_matrix: [N, N] or [B, N, N].

        Returns:
            [B, T_out, N, D] updated queries.
        """
        # 1. Temporal self-attention
        temporal_output = apply_temporal_attention(queries, self.temporal_attention)
        queries = self.norm_temporal(queries + self.dropout(temporal_output))

        # 2. Graph convolution
        batch_size, time_steps, num_nodes, hidden_dim = queries.shape
        graph_input = queries.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        graph_output = self.graph_convolution(graph_input, adjacency_matrix)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        queries = self.norm_graph(queries + self.dropout(graph_output))

        # 3. Cross-attention over history
        cross_output = apply_node_temporal_cross_attention(
            queries, condition_features, self.cross_attention
        )
        queries = self.norm_cross(queries + self.dropout(cross_output))

        # 4. (Optional) Spatiotemporal attention
        if self.use_spatiotemporal_attention:
            spatiotemporal_output = apply_spatiotemporal_attention(
                queries, self.spatiotemporal_attention, context_sequence=condition_features
            )
            queries = self.norm_spatiotemporal(queries + self.dropout(spatiotemporal_output))

        # 5. Feed-forward
        ffn_output = self.feed_forward(queries)
        queries = self.norm_ffn(queries + ffn_output)
        return queries


class FutureDecoder(nn.Module):
    """Deterministic decoder: learnable queries → cross-attention blocks → predictions.

    Architecture:
        learnable future_queries [1, T_out, N, D]
        → N × DecoderBlock (temporal attn + graph conv + cross-attn + FFN)
        → LayerNorm → Linear projection → [B, T_out, N, C_out]
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
        output_window=12,
        num_nodes=1,
        use_gradient_checkpointing=False,
        adaptive_graph_enabled=False,
        adaptive_graph_embed_dim=32,
        adaptive_graph_topk=None,
        adaptive_graph_blend_init=0.5,
        fuzzy_graph_enabled=False,
        fuzzy_graph_num_sets=3,
        fuzzy_graph_sigma_init=0.7,
        static_adjacency=None,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.output_window = output_window
        self.num_nodes = num_nodes

        # Learnable future queries (shared across batch)
        self.future_queries = nn.Parameter(torch.zeros(1, output_window, num_nodes, hidden_dim))
        nn.init.trunc_normal_(self.future_queries, std=0.02)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop, dropout,
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

    def forward(self, condition_features, adjacency_matrix):
        """Decode future from condition.

        Args:
            condition_features: [B, T_in, N, D] encoded history.
            adjacency_matrix: [N, N] or [B, N, N] initial adjacency.

        Returns:
            [B, T_out, N, C_out] predicted future.
        """
        batch_size = condition_features.shape[0]
        queries = self.future_queries.expand(batch_size, -1, -1, -1)

        current_adjacency = adjacency_matrix
        for block in self.blocks:
            if self.adaptive_graph_learner is not None:
                current_adjacency = self.adaptive_graph_learner(queries)
            if self.use_gradient_checkpointing and self.training:
                queries = checkpoint(
                    block, queries, condition_features, current_adjacency, use_reentrant=False
                )
            else:
                queries = block(queries, condition_features, current_adjacency)

        queries = self.final_norm(queries)
        return self.output_projection(queries)
