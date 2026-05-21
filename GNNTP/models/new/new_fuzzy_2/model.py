"""FuzDiff — Fuzzy Relational Graph + Diffusion traffic prediction model.

NewFuzzy2 = STEncoder (condition) + FuzzyRelationalGraph + FutureDecoder.

Core innovations (Phase A):
1. Fuzzy Relational Graph: max-min composition of node membership vectors
2. Fuzzy Graph Convolution: K-hop propagation via fuzzy relational powers
3. Łukasiewicz Fuzzy Conservation loss (T-norm based)
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel

from .encoder import STEncoder
from .decoder import FutureDecoder
from .graph import FuzzyRelationalGraphLearner


class NewFuzzy2(AbstractTrafficStateModel):
    """Fuzzy relational graph + attention regression model for traffic forecasting.

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

        # ── Fuzzy Relational Graph (Phase A) ──────────────────────
        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_num_sets = config.get("fuzzy_num_sets", 3)

        # ── Conservation Loss ─────────────────────────────────────
        self.conservation_loss_weight = config.get("conservation_loss_weight", 0.1)
        self.conservation_warmup_epochs = int(max(0, config.get("conservation_warmup_epochs", 5)))
        self.conservation_steps_per_epoch = int(
            config.get("conservation_steps_per_epoch", 80)
        )
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self.use_fuzzy_conservation = config.get("use_fuzzy_conservation", True)
        self._train_step_count = 0

        # ── Device ────────────────────────────────────────────────
        self.device = config.get("device", torch.device("cpu"))

        # ── Adjacency ─────────────────────────────────────────────
        adjacency_matrix = data_feature.get("adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        # ── Fuzzy Relational Graph Learner ─────────────────────────
        if self.use_fuzzy_graph:
            self.fuzzy_graph = FuzzyRelationalGraphLearner(
                num_nodes=self.num_nodes,
                hidden_dim=self.hidden_dim,
                num_fuzzy_sets=self.fuzzy_num_sets,
                static_adjacency=adjacency_matrix,
                input_dim=self.feature_dim,
            )
        else:
            self.fuzzy_graph = None

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
            use_fuzzy_graph=self.use_fuzzy_graph,
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
            use_fuzzy_graph=self.use_fuzzy_graph,
        )

    def encode_condition(self, history_sequence):
        """Encode historical traffic into condition feature H."""
        if self.use_fuzzy_graph and self.fuzzy_graph is not None:
            fuzzy_R = self.fuzzy_graph(history_sequence).to(history_sequence.device)
        else:
            fuzzy_R = self.adjacency_matrix.to(history_sequence.device)
        return self.condition_encoder(history_sequence, fuzzy_R)

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
        # encode_condition already computed fuzzy_R, but it's not returned.
        # Re-use the graph_matrix from the encoder flow.
        # For simplicity, re-derive (the fuzzy graph is cheap compared to encoder).
        if self.use_fuzzy_graph and self.fuzzy_graph is not None:
            fuzzy_R = self.fuzzy_graph(history_sequence).to(history_sequence.device)
        else:
            fuzzy_R = self.adjacency_matrix.to(history_sequence.device)
        return self.future_decoder(condition_features, fuzzy_R)

    # ═══════════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute L1 loss + optional fuzzy conservation loss."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition_features = self.encode_condition(history_sequence)

        if self.use_fuzzy_graph and self.fuzzy_graph is not None:
            fuzzy_R = self.fuzzy_graph(history_sequence).to(history_sequence.device)
        else:
            fuzzy_R = self.adjacency_matrix.to(history_sequence.device)

        predicted_future = self.future_decoder(condition_features, fuzzy_R)

        regression_loss = F.l1_loss(predicted_future, future_sequence)

        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, fuzzy_R)
            return regression_loss + effective_weight * conservation_loss
        return regression_loss

    def _get_effective_conservation_weight(self):
        """Linearly ramp conservation loss weight over warmup epochs."""
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

    def _fuzzy_conservation_loss(self, future_sequence, fuzzy_relation):
        """Łukasiewicz T-norm fuzzy conservation loss.

        Uses fuzzy logic for physics-informed flow conservation:
          FlowPressure[i→j] = max(0, congestion[i] + R[i,j] - 1)

        This is the Łukasiewicz T-norm (standard in fuzzy logic):
          T_L(x,y) = max(0, x+y-1)

        Intuition: "IF node i is congested AND relation(i,j) is strong,
        THEN there exists flow pressure from i to j."
        """
        if future_sequence.size(1) < 2:
            return future_sequence.new_tensor(0.0)

        node_state = future_sequence[..., self.physics_channel_idx]  # [B, T, N]
        current_state = node_state[:, :-1, :]   # [B, T-1, N]
        next_state = node_state[:, 1:, :]        # [B, T-1, N]
        temporal_delta = next_state - current_state

        # Normalize state to [0, 1] for T-norm compatibility
        s_min = current_state.amin(dim=(0, 1), keepdim=True)
        s_max = current_state.amax(dim=(0, 1), keepdim=True).clamp_min(s_min + 1e-6)
        congestion = ((current_state - s_min) / (s_max - s_min)).clamp(0.0, 1.0)

        # Łukasiewicz T-norm flow pressure
        # R is [N, N] in [0,1], congestion is [B, T-1, N]
        R = fuzzy_relation.to(device=congestion.device, dtype=congestion.dtype)
        # pressure[i→j] = max(0, congestion[i] + R[i,j] - 1)
        c = congestion.unsqueeze(-1)          # [B, T-1, N, 1]
        r = R.unsqueeze(0).unsqueeze(0)        # [1, 1, N, N]
        flow_pressure = (c + r - 1.0).clamp(min=0.0)  # [B, T-1, N, N]

        # Net pressure change: inflow - outflow
        inflow = flow_pressure.sum(dim=-2)     # Σ_j pressure[j→i]
        outflow = flow_pressure.sum(dim=-1)     # Σ_j pressure[i→j]
        net_pressure = inflow - outflow         # [B, T-1, N]

        # Re-scale to original magnitude
        net_pressure = net_pressure * (s_max - s_min)

        residual = temporal_delta - net_pressure
        return residual.pow(2).mean()
