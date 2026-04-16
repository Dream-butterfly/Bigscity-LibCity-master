"""Conditional graph-attention diffusion model for traffic forecasting."""

import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from libcity.models.abstract_traffic_state_model import AbstractTrafficStateModel


def build_normalized_adjacency(adjacency_matrix, add_self_loop=True):
    """Build symmetric normalized adjacency matrix D^{-1/2} A D^{-1/2}."""
    if adjacency_matrix.dim() == 2:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
    if add_self_loop:
        num_nodes = adjacency_matrix.size(-1)
        identity = torch.eye(num_nodes, device=adjacency_matrix.device, dtype=adjacency_matrix.dtype).unsqueeze(0)
        adjacency_matrix = adjacency_matrix + identity
    degree = adjacency_matrix.sum(dim=-1)
    degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
    normalized_adjacency = torch.bmm(torch.bmm(degree_inv_sqrt, adjacency_matrix), degree_inv_sqrt)
    return normalized_adjacency


def expand_adjacency_batch(adjacency_matrix, target_batch_size):
    """Expand or tile adjacency matrix batch dimension to a target batch size."""
    if adjacency_matrix.dim() == 2:
        return adjacency_matrix.unsqueeze(0).expand(target_batch_size, -1, -1)
    if adjacency_matrix.dim() == 3:
        if adjacency_matrix.size(0) == target_batch_size:
            return adjacency_matrix
        if adjacency_matrix.size(0) == 1:
            return adjacency_matrix.expand(target_batch_size, -1, -1)
        if target_batch_size % adjacency_matrix.size(0) != 0:
            raise ValueError(
                f"adjacency batch size {adjacency_matrix.size(0)} is incompatible with "
                f"target batch size {target_batch_size}."
            )
        repeat_factor = target_batch_size // adjacency_matrix.size(0)
        return adjacency_matrix.repeat_interleave(repeat_factor, dim=0)
    raise ValueError("adjacency_matrix must be 2D or 3D tensor.")


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


class MultiHeadAttention(nn.Module):
    """Batch-first multi-head attention with optional cross-attention."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context=None, mask=None):
        """Apply multi-head attention on query and context."""
        if context is None:
            context = query
        batch_size, query_len, _ = query.shape
        context_len = context.size(1)

        query = self.query_projection(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        key = self.key_projection(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        value = self.value_projection(context).view(batch_size, context_len, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_mask = None
        if mask is not None:
            attention_mask = mask
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() != 4:
                raise ValueError("mask must have 2, 3, or 4 dimensions.")
            if attention_mask.dtype == torch.bool:
                # Keep old behavior where True meant a masked-out position.
                attention_mask = ~attention_mask
            elif not torch.is_floating_point(attention_mask):
                raise ValueError("mask must be a bool or floating-point tensor.")
            attention_mask = attention_mask.to(device=query.device)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        return self.output_projection(attention_output)


class GraphConvolution(nn.Module):
    """K-hop graph convolution X' = sum_k A_hat^k X W_k."""

    def __init__(self, hidden_dim, k_hop=2):
        super().__init__()
        self.k_hop = k_hop
        self.projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)])

    def forward(self, node_features, adjacency_matrix):
        """Apply normalized adjacency propagation on node features."""
        batch_size, num_nodes, _ = node_features.shape
        adjacency_matrix = expand_adjacency_batch(adjacency_matrix, batch_size).to(
            device=node_features.device, dtype=node_features.dtype
        )
        adjacency_norm = build_normalized_adjacency(adjacency_matrix)
        adjacency_power = torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        output = torch.zeros_like(node_features)
        for hop_index, projection in enumerate(self.projections):
            if hop_index > 0:
                adjacency_power = torch.bmm(adjacency_power, adjacency_norm)
            propagated = torch.bmm(adjacency_power, node_features)
            output = output + projection(propagated)
        return output


class AdaptiveGraphLearner(nn.Module):
    """Learn and update graph structure from node states during diffusion."""

    def __init__(self, num_nodes, hidden_dim, static_adjacency, embed_dim=32, top_k=None, blend_init=0.5):
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1.")
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.top_k = top_k

        static_adjacency = static_adjacency.to(dtype=torch.float32)
        if static_adjacency.dim() != 2 or static_adjacency.shape[0] != static_adjacency.shape[1]:
            raise ValueError("static_adjacency must be a square 2D tensor.")
        identity = torch.eye(static_adjacency.size(0), device=static_adjacency.device)
        static_adjacency = torch.relu(static_adjacency) + identity
        static_adjacency = self._row_normalize(static_adjacency)
        self.register_buffer("static_adjacency", static_adjacency)
        self.register_buffer("identity", identity)

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.feature_projection = nn.Linear(hidden_dim, embed_dim)
        self.time_projection = nn.Linear(hidden_dim, embed_dim)
        blend_init = float(min(max(blend_init, 1e-4), 1 - 1e-4))
        self.blend_logit = nn.Parameter(
            torch.tensor(math.log(blend_init / (1.0 - blend_init)), dtype=torch.float32)
        )

    @staticmethod
    def _row_normalize(adjacency_matrix):
        return adjacency_matrix / adjacency_matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, sequence_features, timestep_embedding=None):
        if sequence_features.dim() == 4:
            # [B, T, N, D] -> [B, N, D]
            node_features = sequence_features.mean(dim=1)
        elif sequence_features.dim() == 3:
            node_features = sequence_features
        else:
            raise ValueError("sequence_features must be 3D or 4D tensor.")

        batch_size = node_features.size(0)
        node_representation = self.feature_projection(node_features) + self.node_embeddings.unsqueeze(0)
        if timestep_embedding is not None:
            node_representation = node_representation + self.time_projection(timestep_embedding).unsqueeze(1)
        node_representation = torch.tanh(node_representation)

        similarity = torch.matmul(node_representation, node_representation.transpose(1, 2))
        similarity = similarity / math.sqrt(self.embed_dim)
        dynamic_adjacency = torch.softmax(similarity, dim=-1)

        if self.top_k is not None and 0 < self.top_k < self.num_nodes:
            top_values, top_indices = torch.topk(dynamic_adjacency, k=self.top_k, dim=-1)
            sparse_dynamic = torch.zeros_like(dynamic_adjacency)
            sparse_dynamic.scatter_(-1, top_indices, top_values)
            dynamic_adjacency = self._row_normalize(sparse_dynamic)

        dynamic_adjacency = dynamic_adjacency + self.identity.to(dynamic_adjacency.dtype).unsqueeze(0)
        dynamic_adjacency = self._row_normalize(dynamic_adjacency)

        static_adjacency = self.static_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        static_adjacency = static_adjacency.to(dtype=dynamic_adjacency.dtype)

        blend = torch.sigmoid(self.blend_logit)
        adaptive_adjacency = (1.0 - blend) * static_adjacency + blend * dynamic_adjacency
        return self._row_normalize(adaptive_adjacency)


def apply_temporal_attention(sequence_features, attention_module):
    """Apply attention along time dimension for each node independently."""
    batch_size, time_steps, num_nodes, hidden_dim = sequence_features.shape
    reshaped_features = sequence_features.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, hidden_dim)
    attended_features = attention_module(reshaped_features)
    return attended_features.reshape(batch_size, num_nodes, time_steps, hidden_dim).permute(0, 2, 1, 3)


def apply_node_temporal_cross_attention(query_sequence, context_sequence, attention_module):
    """Apply cross-attention over history time for each node independently."""
    if query_sequence.dim() != 4 or context_sequence.dim() != 4:
        raise ValueError("query_sequence and context_sequence must be 4D tensors [B, T, N, D].")
    if query_sequence.shape[0] != context_sequence.shape[0]:
        raise ValueError("Batch size mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[2] != context_sequence.shape[2]:
        raise ValueError("Node count mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[3] != context_sequence.shape[3]:
        raise ValueError("Hidden dimension mismatch between query_sequence and context_sequence.")
    batch_size, query_steps, num_nodes, hidden_dim = query_sequence.shape
    context_steps = context_sequence.shape[1]

    query = query_sequence.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, query_steps, hidden_dim)
    context = context_sequence.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, context_steps, hidden_dim)
    attended_output = attention_module(query, context=context)
    return attended_output.reshape(batch_size, num_nodes, query_steps, hidden_dim).permute(0, 2, 1, 3)


def apply_spatiotemporal_attention(query_sequence, attention_module, context_sequence=None):
    """Apply attention on flattened spatio-temporal tokens (self or cross)."""
    if context_sequence is None:
        context_sequence = query_sequence
    if query_sequence.dim() != 4 or context_sequence.dim() != 4:
        raise ValueError("query_sequence and context_sequence must be 4D tensors [B, T, N, D].")
    if query_sequence.shape[0] != context_sequence.shape[0]:
        raise ValueError("Batch size mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[2] != context_sequence.shape[2]:
        raise ValueError("Node count mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[3] != context_sequence.shape[3]:
        raise ValueError("Hidden dimension mismatch between query_sequence and context_sequence.")

    batch_size, query_steps, num_nodes, hidden_dim = query_sequence.shape
    context_steps = context_sequence.shape[1]
    query_tokens = query_sequence.reshape(batch_size, query_steps * num_nodes, hidden_dim)
    context_tokens = context_sequence.reshape(batch_size, context_steps * num_nodes, hidden_dim)
    attended_tokens = attention_module(query_tokens, context=context_tokens)
    return attended_tokens.reshape(batch_size, query_steps, num_nodes, hidden_dim)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_dim, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        """Apply MLP on the last dimension."""
        return self.network(features)


class STEncoderBlock(nn.Module):
    """Spatio-temporal encoder block with temporal attention and graph convolution."""

    def __init__(self, hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop, dropout=0.1):
        super().__init__()
        self.temporal_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.graph_convolution = GraphConvolution(hidden_dim, graph_k_hop)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ffn_hidden_dim, dropout)

        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_features, adjacency_matrix):
        """Run one encoder block on [B, T, N, D]."""
        temporal_output = apply_temporal_attention(sequence_features, self.temporal_attention)
        sequence_features = self.norm_temporal(sequence_features + self.dropout(temporal_output))

        batch_size, time_steps, num_nodes, hidden_dim = sequence_features.shape
        graph_input = sequence_features.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        graph_output = self.graph_convolution(graph_input, adjacency_matrix)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        sequence_features = self.norm_graph(sequence_features + self.dropout(graph_output))

        ffn_output = self.feed_forward(sequence_features)
        sequence_features = self.norm_ffn(sequence_features + ffn_output)
        return sequence_features


class STEncoder(nn.Module):
    """Condition encoder that maps historical sequence to time-aware conditions."""

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
    ):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                STEncoderBlock(hidden_dim, num_heads, ffn_hidden_dim, graph_k_hop, dropout)
                for _ in range(num_layers)
            ]
        )
        self.temporal_position_embedding = None
        if use_temporal_position_embedding:
            if max_time_steps is None or max_time_steps < 1:
                raise ValueError("max_time_steps must be >= 1 when temporal position embedding is enabled.")
            self.temporal_position_embedding = nn.Parameter(
                torch.zeros(1, max_time_steps, 1, hidden_dim)
            )
            nn.init.trunc_normal_(self.temporal_position_embedding, std=0.02)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_sequence, adjacency_matrix):
        """Encode X in shape [B, T_in, N, C] to H in shape [B, T_in, N, D]."""
        encoded_features = self.input_projection(history_sequence)
        if self.temporal_position_embedding is not None:
            history_steps = encoded_features.shape[1]
            if history_steps > self.max_time_steps:
                raise ValueError(
                    f"history sequence length {history_steps} exceeds max_time_steps={self.max_time_steps}."
                )
            encoded_features = encoded_features + self.temporal_position_embedding[:, :history_steps]
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                encoded_features = checkpoint(
                    block, encoded_features, adjacency_matrix, use_reentrant=False
                )
            else:
                encoded_features = block(encoded_features, adjacency_matrix)
        encoded_features = self.final_norm(encoded_features)
        return encoded_features


class DenoiserBlock(nn.Module):
    """Denoising block: temporal self-attention, graph conv, cross-attention, FFN."""

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

    def forward(self, noisy_future, condition_features, adjacency_matrix):
        """Run one denoiser block with Q=[B,T_out,N,D], K/V=[B,T_in,N,D]."""
        temporal_output = apply_temporal_attention(noisy_future, self.temporal_attention)
        noisy_future = self.norm_temporal(noisy_future + self.dropout(temporal_output))

        batch_size, time_steps, num_nodes, hidden_dim = noisy_future.shape
        graph_input = noisy_future.reshape(batch_size * time_steps, num_nodes, hidden_dim)
        graph_output = self.graph_convolution(graph_input, adjacency_matrix)
        graph_output = graph_output.reshape(batch_size, time_steps, num_nodes, hidden_dim)
        noisy_future = self.norm_graph(noisy_future + self.dropout(graph_output))

        cross_output = apply_node_temporal_cross_attention(
            noisy_future, condition_features, self.cross_attention
        )
        noisy_future = self.norm_cross(noisy_future + self.dropout(cross_output))

        if self.use_spatiotemporal_attention:
            spatiotemporal_output = apply_spatiotemporal_attention(
                noisy_future, self.spatiotemporal_attention, context_sequence=condition_features
            )
            noisy_future = self.norm_spatiotemporal(
                noisy_future + self.dropout(spatiotemporal_output)
            )

        ffn_output = self.feed_forward(noisy_future)
        noisy_future = self.norm_ffn(noisy_future + ffn_output)
        return noisy_future


class AttentionDenoiser(nn.Module):
    """Attention-based noise predictor epsilon_theta(Y_t, t, H, A)."""

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
            use_gradient_checkpointing=True,
            adaptive_graph_enabled=False,
            adaptive_graph_embed_dim=32,
            adaptive_graph_topk=None,
            adaptive_graph_blend_init=0.5,
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
            )

    def forward(self, noisy_future, timesteps, condition_features, adjacency_matrix, return_last_adjacency=False):
        """Predict noise with condition H in shape [B, T_in, N, D]."""
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

        current_adjacency = adjacency_matrix
        for block in self.blocks:
            if self.adaptive_graph_learner is not None:
                current_adjacency = self.adaptive_graph_learner(
                    denoiser_input, timestep_embedding=timestep_features
                )
            if self.use_gradient_checkpointing and self.training:
                denoiser_input = checkpoint(
                    block, denoiser_input, condition_features, current_adjacency, use_reentrant=False
                )
            else:
                denoiser_input = block(denoiser_input, condition_features, current_adjacency)
        denoiser_input = self.final_norm(denoiser_input)
        denoised_output = self.output_projection(denoiser_input)
        if return_last_adjacency:
            return denoised_output, current_adjacency
        return denoised_output


class DiffusionScheduler(nn.Module):
    """Diffusion schedule and forward/backward transition utilities."""

    def __init__(self, diffusion_steps, schedule="linear", beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        betas = self._build_betas(diffusion_steps, schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp_min(1e-12)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    @staticmethod
    def _build_betas(diffusion_steps, schedule, beta_start, beta_end):
        """Build beta schedule by linear or cosine strategy."""
        if schedule.lower() == "linear":
            return torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        if schedule.lower() == "cosine":
            s = 0.008
            steps = diffusion_steps + 1
            x = torch.linspace(0, diffusion_steps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / diffusion_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1e-6, 0.999)
        raise ValueError(f"Unsupported diffusion schedule: {schedule}")

    def sample_timesteps(self, batch_size, device):
        """Sample timesteps uniformly for training."""
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

    @staticmethod
    def _extract(schedule_values, timesteps, target_shape):
        """Gather scheduler values by timestep and reshape for broadcasting."""
        batch_size = timesteps.shape[0]
        extracted = schedule_values.gather(0, timesteps)
        return extracted.reshape(batch_size, *([1] * (len(target_shape) - 1)))

    def add_noise(self, clean_future, timesteps, noise=None):
        """Forward diffusion q(Y_t | Y_0)."""
        if noise is None:
            noise = torch.randn_like(clean_future)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_future.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean_future.shape
        )
        noisy_future = sqrt_alpha_bar * clean_future + sqrt_one_minus_alpha_bar * noise
        return noisy_future, noise

    def predict_start_from_noise(self, noisy_future, timesteps, predicted_noise):
        """Recover clean sample estimate Y_0 from noisy Y_t and predicted epsilon."""
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_future.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_future.shape
        )
        return (noisy_future - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar.clamp_min(1e-12)

    def ddpm_step(self, current_state, timesteps, predicted_noise):
        """Single DDPM reverse step p(Y_{t-1} | Y_t)."""
        beta_t = self._extract(self.betas, timesteps, current_state.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, timesteps, current_state.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, current_state.shape
        )
        posterior_variance_t = self._extract(self.posterior_variance, timesteps, current_state.shape)

        model_mean = sqrt_recip_alpha_t * (
                current_state - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
        )
        noise = torch.randn_like(current_state)
        nonzero_mask = (timesteps > 0).float().reshape(current_state.shape[0], 1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    def ddim_step(self, current_state, timesteps, predicted_noise, eta=0.0):
        """Single DDIM reverse step for faster deterministic or stochastic sampling."""
        alpha_bar_t = self._extract(self.alphas_cumprod, timesteps, current_state.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, timesteps, current_state.shape)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt((1.0 - alpha_bar_t).clamp_min(1e-12))
        predicted_start = (current_state - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        sigma_t = eta * torch.sqrt(
            ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp_min(1e-12))
            * (1.0 - alpha_bar_t / alpha_bar_prev.clamp_min(1e-12))
        )
        direction = torch.sqrt((1.0 - alpha_bar_prev - sigma_t ** 2).clamp_min(1e-12)) * predicted_noise
        noise = torch.randn_like(current_state)
        nonzero_mask = (timesteps > 0).float().reshape(current_state.shape[0], 1, 1, 1)
        return torch.sqrt(alpha_bar_prev) * predicted_start + direction + nonzero_mask * sigma_t * noise


class NewDiffusion(AbstractTrafficStateModel):
    """Graph + Attention + Conditional Diffusion with adaptive graph and conservation prior."""

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self._scaler = data_feature.get("scaler")

        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_heads = config.get("num_heads", 4)
        self.encoder_layers = config.get("encoder_layers", 2)
        self.denoiser_layers = config.get("denoiser_layers", 4)
        self.ffn_hidden_dim = config.get("ffn_hidden_dim", self.hidden_dim * 2)
        self.graph_k_hop = config.get("graph_k_hop", 2)
        self.dropout = config.get("dropout", 0.1)
        self.diffusion_steps = config.get("diffusion_steps", 200)
        self.diffusion_schedule = config.get("diffusion_schedule", "linear")
        self.beta_start = config.get("beta_start", 1e-4)
        self.beta_end = config.get("beta_end", 2e-2)
        self.num_sampling_steps = max(1, min(
            config.get("num_sampling_steps", self.diffusion_steps), self.diffusion_steps
        ))
        self.num_prediction_samples = config.get("num_prediction_samples", 1)
        self.sampling_method = config.get("sampling_method", "ddpm").lower()
        self.ddim_eta = config.get("ddim_eta", 0.0)
        self.use_spatiotemporal_attention = config.get("use_spatiotemporal_attention", True)
        self.use_temporal_position_embedding = config.get("use_temporal_position_embedding", True)
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", True)
        self.use_adaptive_graph = config.get("use_adaptive_graph", True)
        self.adaptive_graph_embed_dim = config.get("adaptive_graph_embed_dim", 32)
        self.adaptive_graph_topk = config.get("adaptive_graph_topk", None)
        self.adaptive_graph_blend_init = config.get("adaptive_graph_blend_init", 0.5)
        self.physics_loss_weight = config.get("physics_loss_weight", 0.05)
        self.flow_conservation_coeff = config.get("flow_conservation_coeff", 1.0)
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self.device = config.get("device", torch.device("cpu"))

        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)
        if self.sampling_method not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampling_method: {self.sampling_method}")
        if self.physics_loss_weight < 0:
            raise ValueError("physics_loss_weight must be >= 0.")
        if self.physics_channel_idx < 0:
            raise ValueError("physics_channel_idx must be >= 0.")
        if self.physics_channel_idx >= self.output_dim:
            raise ValueError(
                f"physics_channel_idx={self.physics_channel_idx} is out of range for output_dim={self.output_dim}."
            )

        adjacency_matrix = data_feature.get("adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        self.condition_encoder = STEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.encoder_layers,
            ffn_hidden_dim=self.ffn_hidden_dim,
            graph_k_hop=self.graph_k_hop,
            dropout=self.dropout,
            use_temporal_position_embedding=self.use_temporal_position_embedding,
            max_time_steps=self.input_window,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        self.noise_predictor = AttentionDenoiser(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.denoiser_layers,
            ffn_hidden_dim=self.ffn_hidden_dim,
            graph_k_hop=self.graph_k_hop,
            dropout=self.dropout,
            use_spatiotemporal_attention=self.use_spatiotemporal_attention,
            use_temporal_position_embedding=self.use_temporal_position_embedding,
            max_future_steps=self.output_window,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            adaptive_graph_enabled=self.use_adaptive_graph,
            adaptive_graph_embed_dim=self.adaptive_graph_embed_dim,
            adaptive_graph_topk=self.adaptive_graph_topk,
            adaptive_graph_blend_init=self.adaptive_graph_blend_init,
            num_nodes=self.num_nodes,
            static_adjacency=adjacency_matrix,
        )
        self.diffusion_scheduler = DiffusionScheduler(
            diffusion_steps=self.diffusion_steps,
            schedule=self.diffusion_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

    def encode_condition(self, history_sequence):
        """Encode historical traffic into condition feature H."""
        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        return self.condition_encoder(history_sequence, adjacency_matrix)

    def forward(self, batch):
        """Forward inference entry, delegates to diffusion sampling."""
        return self.predict(batch)

    def calculate_loss(self, batch):
        """Compute diffusion objective E[||epsilon - epsilon_theta||^2]."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., : self.output_dim]
        condition_features = self.encode_condition(history_sequence)

        batch_size = future_sequence.shape[0]
        timesteps = self.diffusion_scheduler.sample_timesteps(batch_size, history_sequence.device)
        noisy_future, true_noise = self.diffusion_scheduler.add_noise(future_sequence, timesteps)

        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        need_physics = self.physics_loss_weight > 0
        if need_physics:
            predicted_noise, adaptive_adjacency = self.noise_predictor(
                noisy_future, timesteps, condition_features, adjacency_matrix, return_last_adjacency=True
            )
        else:
            predicted_noise = self.noise_predictor(
                noisy_future, timesteps, condition_features, adjacency_matrix, return_last_adjacency=False
            )
        diffusion_loss = F.mse_loss(predicted_noise, true_noise)
        if not need_physics:
            return diffusion_loss

        predicted_future = self.diffusion_scheduler.predict_start_from_noise(
            noisy_future, timesteps, predicted_noise
        )
        conservation_loss = self._traffic_conservation_loss(predicted_future, adaptive_adjacency)
        return diffusion_loss + self.physics_loss_weight * conservation_loss

    def _traffic_conservation_loss(self, future_sequence, adjacency_matrix):
        """Penalize mismatch between temporal state change and graph net-flow."""
        if future_sequence.size(1) < 2:
            return future_sequence.new_tensor(0.0)

        node_state = future_sequence[..., self.physics_channel_idx]  # [B, T, N]
        current_state = node_state[:, :-1, :]
        next_state = node_state[:, 1:, :]
        temporal_delta = next_state - current_state  # [B, T-1, N]

        adjacency_matrix = expand_adjacency_batch(adjacency_matrix, node_state.size(0)).to(
            device=node_state.device, dtype=node_state.dtype
        )

        outflow = current_state * adjacency_matrix.sum(dim=-1).unsqueeze(1)
        inflow = torch.einsum("bij,btj->bti", adjacency_matrix.transpose(1, 2), current_state)
        net_flow = inflow - outflow

        residual = temporal_delta - self.flow_conservation_coeff * net_flow
        return residual.pow(2).mean()

    def _sample_once(self, condition_features):
        """Generate one future trajectory by reverse diffusion."""
        batch_size = condition_features.shape[0]
        future_state = torch.randn(
            batch_size,
            self.output_window,
            self.num_nodes,
            self.output_dim,
            device=condition_features.device,
        )
        adjacency_matrix = self.adjacency_matrix.to(condition_features.device)
        for step in reversed(range(self.num_sampling_steps)):
            timestep = torch.full((batch_size,), step, device=condition_features.device, dtype=torch.long)
            predicted_noise = self.noise_predictor(
                future_state, timestep, condition_features, adjacency_matrix
            )
            if self.sampling_method == "ddim":
                future_state = self.diffusion_scheduler.ddim_step(
                    future_state, timestep, predicted_noise, eta=self.ddim_eta
                )
            else:
                future_state = self.diffusion_scheduler.ddpm_step(
                    future_state, timestep, predicted_noise
                )
        return future_state

    def sample(self, history_sequence, num_samples=1, return_all=False):
        """Sample future trajectories for uncertainty-aware prediction."""
        condition_features = self.encode_condition(history_sequence)
        sampled_futures = [self._sample_once(condition_features) for _ in range(num_samples)]
        sampled_futures = torch.stack(sampled_futures, dim=0)
        if return_all:
            return sampled_futures
        return sampled_futures.mean(dim=0)

    def predict(self, batch):
        """Predict future traffic as mean of multi-sample diffusion trajectories."""
        history_sequence = batch["X"]
        return self.sample(
            history_sequence, num_samples=self.num_prediction_samples, return_all=False
        )
