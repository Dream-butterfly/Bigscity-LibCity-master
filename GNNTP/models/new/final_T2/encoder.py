"""Spatio-temporal encoder (final_T2) — accepts optional per-node MDI uncertainty.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .graph import GraphConvolution, FuzzyGraphConvolution, FuzzySpatialAttention
from .cell_attention import FuzzyCellAttention, CellAttentionPool
from .utils import apply_temporal_attention


class STEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        ffn_hidden_dim,
        graph_k_hop,
        dropout=0.1,
        use_fuzzy_graph=False,
        use_fuzzy_spatial_attn=False,
        use_cell_attention=False,
        num_cells=8,
        use_hollow_kernel=True,
        cell_blend_init=0.3,
    ):
        super().__init__()
        self.use_cell_attention = use_cell_attention
        self.use_fuzzy_spatial_attn = use_fuzzy_spatial_attn

        self.temporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        if use_fuzzy_spatial_attn:
            self.spatial_mixer = FuzzySpatialAttention(hidden_dim, num_heads, dropout)
            self.norm_spatial = nn.LayerNorm(hidden_dim)
        elif use_fuzzy_graph:
            self.graph_convolution = FuzzyGraphConvolution(hidden_dim, graph_k_hop)
        else:
            self.graph_convolution = GraphConvolution(hidden_dim, graph_k_hop)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        if not use_fuzzy_spatial_attn:
            self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if use_cell_attention:
            self.cell_attention = FuzzyCellAttention(
                hidden_dim=hidden_dim,
                num_cells=num_cells,
                use_hollow_kernel=use_hollow_kernel,
                cell_blend_init=cell_blend_init,
            )
            self.norm_cell = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_features, graph_matrix, graph_uncertainty=None, powers=None):
        temporal_output = apply_temporal_attention(sequence_features, self.temporal_attention)
        sequence_features = self.norm_temporal(sequence_features + self.dropout(temporal_output))

        batch_size, time_steps, num_nodes, hidden_dim = sequence_features.shape
        graph_input = sequence_features.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        if self.use_fuzzy_spatial_attn:
            graph_output = self.spatial_mixer(graph_input, R=graph_matrix)
        else:
            graph_output = self.graph_convolution(graph_input, graph_matrix, powers=powers)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        if self.use_fuzzy_spatial_attn:
            sequence_features = self.norm_spatial(sequence_features + self.dropout(graph_output))
        else:
            sequence_features = self.norm_graph(sequence_features + self.dropout(graph_output))

        if self.use_cell_attention:
            node_repr = CellAttentionPool.time_mean_pool(sequence_features)  # (B,N,D)
            cell_out = self.cell_attention(node_repr, node_uncertainty=graph_uncertainty)  # (B,N,D)
            cell_out = cell_out.unsqueeze(1)  # (B,1,N,D)
            blend = self.cell_attention.cell_blend.sigmoid()
            sequence_features = self.norm_cell(
                sequence_features + blend * cell_out
            )

        ffn_output = self.feed_forward(sequence_features)
        sequence_features = self.norm_ffn(sequence_features + ffn_output)
        return sequence_features


class STEncoder(nn.Module):
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
        use_fuzzy_spatial_attn=False,
        use_cell_attention=False,
        num_cells=8,
        use_hollow_kernel=True,
        cell_blend_init=0.3,
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
                use_fuzzy_spatial_attn=use_fuzzy_spatial_attn,
                use_cell_attention=use_cell_attention,
                num_cells=num_cells,
                use_hollow_kernel=use_hollow_kernel,
                cell_blend_init=cell_blend_init,
            )
            for _ in range(num_layers)
        ])
        self.temporal_position_embedding = None
        if use_temporal_position_embedding:
            if max_time_steps is None or max_time_steps < 1:
                raise ValueError(
                    "max_time_steps must be >= 1 when temporal position embedding is enabled."
                )
            self.temporal_position_embedding = nn.Parameter(
                torch.zeros(1, max_time_steps, 1, hidden_dim)
            )
            nn.init.trunc_normal_(self.temporal_position_embedding, std=0.02)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_sequence, graph_matrix, graph_uncertainty=None, powers=None):
        encoded_features = self.input_projection(history_sequence)
        if self.temporal_position_embedding is not None:
            history_steps = encoded_features.shape[1]
            if history_steps > self.max_time_steps:
                raise ValueError(
                    f"history sequence length {history_steps} exceeds "
                    f"max_time_steps={self.max_time_steps}."
                )
            encoded_features = encoded_features + self.temporal_position_embedding[:, :history_steps]
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                encoded_features = checkpoint(
                    block, encoded_features, graph_matrix, graph_uncertainty, powers, use_reentrant=False
                )
            else:
                encoded_features = block(encoded_features, graph_matrix, graph_uncertainty, powers=powers)
        encoded_features = self.final_norm(encoded_features)
        return encoded_features

