"""Spatio-temporal encoder — new_fuzzy_2 (fuzzy graph support).

Converts historical traffic sequences [B, Tin, N, Cin] into rich condition
features H [B, Tin, N, D].  Supports both standard GCN and fuzzy graph
convolution modes.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .graph import GraphConvolution, FuzzyGraphConvolution
from .utils import apply_temporal_attention


class STEncoderBlock(nn.Module):
    """Spatio-temporal encoder block: temporal self-attn → graph conv → FFN.

    Operates on [B, T, N, D] tensors with Pre-LN residual connections.
    Supports both ``GraphConvolution`` and ``FuzzyGraphConvolution``.
    """

    def __init__(self, hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop,
                 dropout=0.1, use_fuzzy_graph=False):
        super().__init__()
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

    def forward(self, sequence_features, graph_matrix):
        """Run one encoder block on [B, T, N, D].

        ``graph_matrix`` is an adjacency (standard mode) or a fuzzy
        relation matrix (fuzzy mode).
        """
        # 1. Temporal self-attention (per-node)
        temporal_output = apply_temporal_attention(sequence_features, self.temporal_attention)
        sequence_features = self.norm_temporal(sequence_features + self.dropout(temporal_output))

        # 2. Graph convolution (shared across time)
        batch_size, time_steps, num_nodes, hidden_dim = sequence_features.shape
        graph_input = sequence_features.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        graph_output = self.graph_convolution(graph_input, graph_matrix)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        sequence_features = self.norm_graph(sequence_features + self.dropout(graph_output))

        # 3. Feed-forward
        ffn_output = self.feed_forward(sequence_features)
        sequence_features = self.norm_ffn(sequence_features + ffn_output)
        return sequence_features


class STEncoder(nn.Module):
    """Condition encoder mapping history X to time-aware condition H.

    Architecture:
        input_projection(Cin → D) + temporal_position_embedding
        → N × STEncoderBlock (temporal attn + graph conv + FFN)
        → LayerNorm
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
    ):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            STEncoderBlock(hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop,
                           dropout, use_fuzzy_graph=use_fuzzy_graph)
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

    def forward(self, history_sequence, graph_matrix):
        """Encode X [B, Tin, N, Cin] → H [B, Tin, N, D].

        Args:
            history_sequence: [B, Tin, N, Cin] historical traffic data.
            graph_matrix: [N, N] adjacency or fuzzy relation.

        Returns:
            [B, Tin, N, D] condition features.
        """
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
                    block, encoded_features, graph_matrix, use_reentrant=False
                )
            else:
                encoded_features = block(encoded_features, graph_matrix)
        encoded_features = self.final_norm(encoded_features)
        return encoded_features
