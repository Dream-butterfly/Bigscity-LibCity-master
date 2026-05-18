"""Fuzzy-graph deterministic traffic prediction model.

NewFuzzy = STEncoder (condition) + FutureDecoder (regression) + fuzzy conservation loss.

Core contributions:
1. Fuzzy graph learning via Gaussian membership functions
2. Fuzzy conservation loss for physics-informed training
3. Learnable future queries with cross-attention decoding
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel

from .encoder import STEncoder
from .decoder import FutureDecoder
from .utils import expand_adjacency_batch


class NewFuzzy(AbstractTrafficStateModel):
    """Fuzzy graph + attention regression model for traffic forecasting.

    Training: forward() → calculate_loss() → scalar.
    Inference: forward() → predict() → [B, T_out, N, C_out].
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self._scaler = data_feature.get("scaler")

        # ── Geometry ──────────────────────────────────────────────
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)

        # ── Architecture ──────────────────────────────────────────
        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_heads = config.get("num_heads", 2)
        self.encoder_layers = config.get("encoder_layers", 2)
        self.decoder_layers = config.get("decoder_layers", 2)
        self.ffn_hidden_dim = config.get("ffn_hidden_dim", 128)
        self.graph_k_hop = config.get("graph_k_hop", 2)
        self.dropout = config.get("dropout", 0.1)
        self.use_spatiotemporal_attention = config.get("use_spatiotemporal_attention", True)
        self.use_temporal_position_embedding = config.get("use_temporal_position_embedding", True)
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)

        # ── Adaptive / Fuzzy Graph ────────────────────────────────
        self.use_adaptive_graph = config.get("use_adaptive_graph", True)
        self.adaptive_graph_embed_dim = config.get("adaptive_graph_embed_dim", 32)
        self.adaptive_graph_topk = config.get("adaptive_graph_topk", None)
        self.adaptive_graph_blend_init = config.get("adaptive_graph_blend_init", 0.5)
        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_graph_num_sets = config.get("fuzzy_graph_num_sets", 3)
        self.fuzzy_graph_sigma_init = config.get("fuzzy_graph_sigma_init", 0.7)

        # ── Conservation Loss ─────────────────────────────────────
        self.conservation_loss_weight = config.get("conservation_loss_weight", 0.1)
        self.conservation_warmup_epochs = int(max(0, config.get("conservation_warmup_epochs", 5)))
        # Steps-per-epoch for warmup ramp; must be set to match the
        # actual number of training batches per epoch (depends on dataset
        # size, batch_size, and DDP world_size).
        self.conservation_steps_per_epoch = int(
            config.get("conservation_steps_per_epoch", 80)
        )
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self.use_fuzzy_conservation = config.get("use_fuzzy_conservation", True)
        self.fuzzy_conservation_threshold = config.get("fuzzy_conservation_threshold", 0.6)
        self.fuzzy_conservation_temperature = config.get("fuzzy_conservation_temperature", 8.0)
        self._train_step_count = 0

        # ── Device ────────────────────────────────────────────────
        self.device = config.get("device", torch.device("cpu"))

        # ── Adjacency ─────────────────────────────────────────────
        adjacency_matrix = data_feature.get("adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        # ── Submodules ────────────────────────────────────────────
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
        self.future_decoder = FutureDecoder(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.decoder_layers,
            ffn_hidden_dim=self.ffn_hidden_dim,
            graph_k_hop=self.graph_k_hop,
            dropout=self.dropout,
            use_spatiotemporal_attention=self.use_spatiotemporal_attention,
            output_window=self.output_window,
            num_nodes=self.num_nodes,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            adaptive_graph_enabled=self.use_adaptive_graph,
            adaptive_graph_embed_dim=self.adaptive_graph_embed_dim,
            adaptive_graph_topk=self.adaptive_graph_topk,
            adaptive_graph_blend_init=self.adaptive_graph_blend_init,
            fuzzy_graph_enabled=self.use_fuzzy_graph,
            fuzzy_graph_num_sets=self.fuzzy_graph_num_sets,
            fuzzy_graph_sigma_init=self.fuzzy_graph_sigma_init,
            static_adjacency=adjacency_matrix,
        )

    def encode_condition(self, history_sequence):
        """Encode historical traffic into condition feature H."""
        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        return self.condition_encoder(history_sequence, adjacency_matrix)

    # ═══════════════════════════════════════════════════════════════
    #  Forward / Predict
    # ═══════════════════════════════════════════════════════════════

    def forward(self, batch):
        """Forward entry. Training → returns loss. Inference → predicts."""
        if self.training:
            self._train_step_count += 1
            return self.calculate_loss(batch)
        return self.predict(batch)

    def predict(self, batch):
        """Predict future traffic from history.

        Returns:
            [B, T_out, N, C_out] predicted future.
        """
        history_sequence = batch["X"]
        condition_features = self.encode_condition(history_sequence)
        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        return self.future_decoder(condition_features, adjacency_matrix)

    # ═══════════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute L1 loss + optional fuzzy conservation loss."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]
        condition_features = self.encode_condition(history_sequence)
        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        predicted_future = self.future_decoder(condition_features, adjacency_matrix)

        regression_loss = F.l1_loss(predicted_future, future_sequence)

        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, adjacency_matrix)
            return regression_loss + effective_weight * conservation_loss
        return regression_loss

    def _get_effective_conservation_weight(self):
        """Linearly ramp conservation loss weight over warmup epochs.

        Uses ``conservation_steps_per_epoch`` to convert epoch-based warmup
        to step-based ramp.  Must match the actual number of training batches
        per epoch for the current dataset / batch-size / DDP-world-size
        combination.
        """
        if self.conservation_loss_weight <= 0:
            return 0.0
        if self.conservation_warmup_epochs <= 0:
            return self.conservation_loss_weight
        steps_per_epoch = self.conservation_steps_per_epoch
        total_warmup_steps = self.conservation_warmup_epochs * steps_per_epoch
        if total_warmup_steps <= 0:
            return self.conservation_loss_weight
        if self._train_step_count >= total_warmup_steps:
            return self.conservation_loss_weight
        return self.conservation_loss_weight * (self._train_step_count / total_warmup_steps)

    def _fuzzy_conservation_loss(self, future_sequence, adjacency_matrix):
        """Fuzzy traffic conservation: soft penalty on flow imbalance.

        Uses fuzzy congestion membership to up-weight high-traffic nodes
        where conservation violations matter most.
        """
        if future_sequence.size(1) < 2:
            return future_sequence.new_tensor(0.0)

        node_state = future_sequence[..., self.physics_channel_idx]  # [B, T, N]
        current_state = node_state[:, :-1, :]
        next_state = node_state[:, 1:, :]
        temporal_delta = next_state - current_state  # [B, T-1, N]

        adjacency_matrix = expand_adjacency_batch(adjacency_matrix, node_state.size(0)).to(
            device=node_state.device, dtype=node_state.dtype
        )
        spatial_flow = torch.einsum("bnm,btm->btn", adjacency_matrix, current_state)
        net_flow = spatial_flow - current_state
        residual = temporal_delta - net_flow

        if self.use_fuzzy_conservation:
            node_scale = current_state.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            normalized_state = current_state.abs() / node_scale
            high_congestion_membership = torch.sigmoid(
                self.fuzzy_conservation_temperature * (normalized_state - self.fuzzy_conservation_threshold)
            )
            fuzzy_weight = 0.5 + high_congestion_membership
            return (fuzzy_weight * residual.pow(2)).mean()

        return residual.pow(2).mean()
