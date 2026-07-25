"""Future decoder — final_T2 (copied).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, FeedForwardNetwork
from .graph import GraphConvolution, FuzzyGraphConvolution, FuzzySpatialAttention
from .cell_attention import FuzzyCellAttention, CellAttentionPool
from .utils import (
    apply_temporal_attention,
    apply_node_temporal_cross_attention,
    apply_spatiotemporal_attention,
)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        ffn_hidden_dim,
        graph_k_hop,
        dropout=0.1,
        use_spatiotemporal_attention=False,
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
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.use_spatiotemporal_attention = use_spatiotemporal_attention
        if use_spatiotemporal_attention:
            self.spatiotemporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        if not use_fuzzy_spatial_attn:
            self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        if use_spatiotemporal_attention:
            self.norm_spatiotemporal = nn.LayerNorm(hidden_dim)
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

    def forward(self, queries, condition_features, graph_matrix, graph_uncertainty=None, powers=None):
        temporal_output = apply_temporal_attention(queries, self.temporal_attention)
        queries = self.norm_temporal(queries + self.dropout(temporal_output))

        batch_size, time_steps, num_nodes, hidden_dim = queries.shape
        graph_input = queries.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        if self.use_fuzzy_spatial_attn:
            graph_output = self.spatial_mixer(graph_input, R=graph_matrix)
        else:
            graph_output = self.graph_convolution(graph_input, graph_matrix, powers=powers)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        if self.use_fuzzy_spatial_attn:
            queries = self.norm_spatial(queries + self.dropout(graph_output))
        else:
            queries = self.norm_graph(queries + self.dropout(graph_output))

        if self.use_cell_attention:
            node_repr = CellAttentionPool.time_mean_pool(queries)  # (B,N,D)
            cell_out = self.cell_attention(node_repr, node_uncertainty=graph_uncertainty)  # (B,N,D)
            cell_out = cell_out.unsqueeze(1)  # (B,1,N,D)
            blend = self.cell_attention.cell_blend.sigmoid()
            queries = self.norm_cell(queries + blend * cell_out)

        cross_output = apply_node_temporal_cross_attention(
            queries, condition_features, self.cross_attention
        )
        queries = self.norm_cross(queries + self.dropout(cross_output))

        if self.use_spatiotemporal_attention:
            spatiotemporal_output = apply_spatiotemporal_attention(
                queries, self.spatiotemporal_attention, context_sequence=condition_features
            )
            queries = self.norm_spatiotemporal(queries + self.dropout(spatiotemporal_output))

        ffn_output = self.feed_forward(queries)
        queries = self.norm_ffn(queries + ffn_output)
        return queries


class FutureDecoder(nn.Module):
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
        use_fuzzy_graph=False,
        use_fuzzy_spatial_attn=False,
        use_cell_attention=False,
        num_cells=8,
        use_hollow_kernel=True,
        cell_blend_init=0.3,
        node_mode="embed",
        proto_embed_low=None,
        proto_embed_mid=None,
        proto_embed_high=None,
        temp_dilation=1,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.output_window = output_window
        self.num_nodes = num_nodes
        self.node_mode = node_mode

        # Factorized future queries: horizon × node → T×N×D
        self.horizon_embed = nn.Parameter(
            torch.zeros(1, output_window, 1, hidden_dim))
        nn.init.trunc_normal_(self.horizon_embed, std=0.02)

        # Node identity component — routed through prototype membership
        # when node_mode="proto" or "both", avoiding the per-node shortcut.
        if node_mode in ("embed", "both"):
            self.node_embed = nn.Parameter(
                torch.zeros(1, 1, num_nodes, hidden_dim))
            nn.init.trunc_normal_(self.node_embed, std=0.02)
        else:
            self.node_embed = None

        # Per-view prototype embeddings (shared from model layer)
        self.proto_embed_low  = proto_embed_low
        self.proto_embed_mid  = proto_embed_mid
        self.proto_embed_high = proto_embed_high
        # Backward-compat
        self.proto_embed = proto_embed_mid

        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop, dropout,
                use_spatiotemporal_attention=use_spatiotemporal_attention,
                use_fuzzy_graph=use_fuzzy_graph,
                use_fuzzy_spatial_attn=use_fuzzy_spatial_attn,
                use_cell_attention=use_cell_attention,
                num_cells=num_cells,
                use_hollow_kernel=use_hollow_kernel,
                cell_blend_init=cell_blend_init,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # ── Temporal refinement: pure time attention after spatial ops ──
        self.temporal_refine = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm_temporal_refine = nn.LayerNorm(hidden_dim)

        # Per-step projection: D → bottleneck → output_dim
        # Cross-timestep interaction is handled by temporal_refine attention,
        # so a simple per-step linear projection suffices.
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )
        # Lightweight temporal mixing before per-step projection (Conv1D, ~7K params)
        # Provides local cross-step interaction without O(T²D²) cost of full mixed projection
        # temp_dilation: 1=RF5, 2=RF7, 4=RF11  (RF ≈ 1 + 2*kernel*dilation per layer)
        padding = (3 // 2) * temp_dilation
        self.temp_mix = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3,
                      padding=padding, dilation=temp_dilation),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3,
                      padding=padding, dilation=temp_dilation),
        )
        self.output_window = output_window  # needed in forward

    def forward(self, condition_features, graph_matrix, graph_uncertainty=None,
                powers=None, mu_low=None, mu_mid=None, mu_high=None, beta=None):
        batch_size = condition_features.shape[0]
        # Detect whether mu is per-sample (3D) or batch-mean (2D)
        mu_3d = mu_low is not None and mu_low.dim() == 3

        # ── Node identity component ──
        if self.node_mode == "embed":
            node_comp = self.node_embed
        elif self.node_mode == "proto":
            # β-weighted per-view routing: each view's membership × its own proto embed
            # Supports both per-node β (N,3) and global β [3]
            if mu_low is None or beta is None:
                raise ValueError("mu_low/mu_mid/mu_high/beta required for node_mode='proto'")
            def _route(mu, pe):
                return torch.matmul(mu, pe.squeeze(0))  # (*,N,K)@(K,D) → (*,N,D)
            if beta.dim() == 2:
                node_comp = (beta[:, 0:1] * _route(mu_low,  self.proto_embed_low) +
                             beta[:, 1:2] * _route(mu_mid,  self.proto_embed_mid) +
                             beta[:, 2:3] * _route(mu_high, self.proto_embed_high))
            else:
                node_comp = (beta[0] * _route(mu_low,  self.proto_embed_low) +
                             beta[1] * _route(mu_mid,  self.proto_embed_mid) +
                             beta[2] * _route(mu_high, self.proto_embed_high))
            if mu_3d:
                node_comp = node_comp.unsqueeze(1)    # (B,N,D) → (B,1,N,D)
            else:
                node_comp = node_comp.unsqueeze(0).unsqueeze(0)  # (N,D) → (1,1,N,D)
        elif self.node_mode == "both":
            node_comp = self.node_embed
            if mu_low is not None and beta is not None:
                def _route(mu, pe):
                    return torch.matmul(mu, pe.squeeze(0))
                if beta.dim() == 2:
                    routing = (beta[:, 0:1] * _route(mu_low,  self.proto_embed_low) +
                               beta[:, 1:2] * _route(mu_mid,  self.proto_embed_mid) +
                               beta[:, 2:3] * _route(mu_high, self.proto_embed_high))
                else:
                    routing = (beta[0] * _route(mu_low,  self.proto_embed_low) +
                               beta[1] * _route(mu_mid,  self.proto_embed_mid) +
                               beta[2] * _route(mu_high, self.proto_embed_high))
                if mu_3d:
                    node_comp = node_comp + routing.unsqueeze(1)    # (B,N,D) → (B,1,N,D)
                else:
                    node_comp = node_comp + routing.unsqueeze(0).unsqueeze(0)
        else:
            node_comp = self.node_embed

        queries = (self.horizon_embed + node_comp).expand(
            batch_size, -1, -1, -1)

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                queries = checkpoint(
                    block, queries, condition_features, graph_matrix, graph_uncertainty, powers, use_reentrant=False
                )
            else:
                queries = block(queries, condition_features, graph_matrix, graph_uncertainty, powers=powers)

        # ── Temporal refinement: pure time attention across output steps ──
        B, T, N, D = queries.shape
        q_t = queries.permute(0, 2, 1, 3).reshape(B * N, T, D)  # merge batch & node
        q_t_refined = self.temporal_refine(q_t, q_t)  # query, context (no mask)
        q_t = q_t + q_t_refined  # residual
        queries = self.norm_temporal_refine(q_t).reshape(B, N, T, D).permute(0, 2, 1, 3)

        queries = self.final_norm(queries)  # (B, T, N, D)

        # Temporal mixing → per-step projection
        B, T, N, D = queries.shape
        q_t = queries.permute(0, 2, 1, 3).reshape(B * N, T, D).transpose(1, 2)  # (BN, D, T)
        q_t = self.temp_mix(q_t).transpose(1, 2)                                  # Conv1d over time
        q_flat = q_t.reshape(B * N * T, D)                                        # (BNT, D)
        out_flat = self.output_projection(q_flat)
        return out_flat.reshape(B, N, T, -1).permute(0, 2, 1, 3)   # (B, T, N, out)

