"""FuzDiff — Fuzzy Relational Graph + Conditional Diffusion (Phase C).

NewFuzzy3 supports two paths controlled by ``use_diffusion``:

  use_diffusion = True  → DDIM conditional diffusion with fuzzy-guided denoiser
  use_diffusion = False → deterministic Encoder-Decoder regression (new_fuzzy_2)

All other ablation controls (``use_fuzzy_graph``, ``use_fuzzy_conservation``)
work identically in both paths, enabling an 8-configuration ablation matrix.
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
from .diffusion import DiffusionScheduler
from .denoiser import FuzzyGuidedDenoiser


class NewFuzzy3(AbstractTrafficStateModel):
    """Fuzzy relational graph + optional diffusion for traffic forecasting.

    Training: forward() → calculate_loss() → scalar.
    Inference: forward() → predict() → [B, T_out, N, C_out].

    DDP: ``_ddp_loss_through_forward = True`` ensures the executor calls
    ``self.model(batch)`` which triggers the DDP backward hook correctly
    for both diffusion and deterministic paths.
    中文翻译版：Fuzzy 相关图 +
    """

    # DDP hook — forward() returns loss when training
    _ddp_loss_through_forward = True

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

        # ── FCM Regularization ────────────────────────────────────
        self.fcm_loss_weight = config.get("fcm_loss_weight", 0.01)
        self.fcm_warmup_epochs = int(max(0, config.get("fcm_warmup_epochs", 3)))
        self.fcm_steps_per_epoch = int(config.get("fcm_steps_per_epoch", 80))

        self._train_step_count = 0

        # ── Diffusion (Phase C) ───────────────────────────────────
        self.use_diffusion = config.get("use_diffusion", True)
        if self.use_diffusion:
            self.diffusion_steps = config.get("diffusion_steps", 50)
            self.num_sampling_steps = config.get("num_sampling_steps", 20)
            self.ddim_eta = config.get("ddim_eta", 0.0)
            self.prediction_clamp_min = config.get("prediction_clamp_min", -3.0)
            self.prediction_clamp_max = config.get("prediction_clamp_max", 3.0)
            self.denoiser_layers = config.get("denoiser_layers", 2)
            self.diffusion = DiffusionScheduler(
                diffusion_steps=self.diffusion_steps,
                schedule=config.get("diffusion_schedule", "linear"),
                beta_start=config.get("beta_start", 1e-4),
                beta_end=config.get("beta_end", 0.01),
            )

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

        # ── Encoder (shared by both paths) ────────────────────────
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

        # ── Prediction head: diffusion or deterministic ───────────
        if self.use_diffusion:
            self.denoiser = FuzzyGuidedDenoiser(
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.denoiser_layers,
                ffn_hidden_dim=self.ffn_hidden_dim,
                graph_k_hop=self.graph_k_hop,
                dropout=self.dropout,
                use_spatiotemporal_attention=self.use_spatiotemporal_attention,
                max_future_steps=self.output_window,
                input_window=self.input_window,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_fuzzy_graph=self.use_fuzzy_graph,
            )
        else:
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

    # ═══════════════════════════════════════════════════════════════
    #  Shared helpers
    # ═══════════════════════════════════════════════════════════════

    def encode_condition(self, history_sequence):
        """Encode historical traffic into condition feature H."""
        graph_matrix = self._get_graph_matrix(history_sequence)
        return self.condition_encoder(history_sequence, graph_matrix)

    def _get_graph_matrix(self, history_sequence):
        """Return fuzzy relation R or fallback adjacency [N, N]."""
        if self.use_fuzzy_graph and self.fuzzy_graph is not None:
            return self.fuzzy_graph(history_sequence).to(history_sequence.device)
        return self.adjacency_matrix.to(history_sequence.device)

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
        """Predict future traffic from history. 从历史数据预测未来的交通

        Diffusion path: DDIM reverse sampling.
        Deterministic path: single forward pass through decoder.

        Returns:
            [B, T_out, N, C_out] predicted future.
        """
        history_sequence = batch["X"]
        condition_features = self.encode_condition(history_sequence)
        graph_matrix = self._get_graph_matrix(history_sequence)

        if self.use_diffusion:
            return self._diffusion_predict(condition_features, graph_matrix)
        else:
            return self.future_decoder(condition_features, graph_matrix)

    def _diffusion_predict(self, condition_features, graph_matrix):
        """DDIM reverse sampling from noise to clean prediction."""
        B = condition_features.size(0)
        device = condition_features.device
        dtype = condition_features.dtype

        Y_t = torch.randn(
            B, self.output_window, self.num_nodes, self.output_dim,
            device=device, dtype=dtype,
        )

        timesteps = torch.linspace(
            self.diffusion_steps - 1, 0, self.num_sampling_steps,
            dtype=torch.long, device=device,
        )

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)
            epsilon_theta = self.denoiser(Y_t, t_curr, condition_features, graph_matrix)
            Y_t = self.diffusion.ddim_step(
                Y_t, t_curr, epsilon_theta, eta=self.ddim_eta, t_next=t_next,
            )
            Y_t = Y_t.clamp(self.prediction_clamp_min, self.prediction_clamp_max)

        return Y_t

    # ═══════════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute loss: main (diffusion or deterministic) + FCM regularization."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition_features = self.encode_condition(history_sequence)
        graph_matrix = self._get_graph_matrix(history_sequence)

        # ── Main loss ────────────────────────────────────────────
        if self.use_diffusion:
            total_loss = self._diffusion_loss(future_sequence, condition_features, graph_matrix)
        else:
            total_loss = self._deterministic_loss(future_sequence, condition_features, graph_matrix)

        # ── FCM membership–prototype regularization ──────────────
        effective_fcm = self._get_effective_fcm_weight()
        if effective_fcm > 0 and self.use_fuzzy_graph and self.fuzzy_graph is not None:
            fcm_loss = self.fuzzy_graph.fcm_loss(history_sequence)
            total_loss = total_loss + effective_fcm * fcm_loss

        return total_loss

    def _deterministic_loss(self, future_sequence, condition_features, graph_matrix):
        """L1 regression + optional fuzzy conservation."""
        predicted_future = self.future_decoder(condition_features, graph_matrix)
        regression_loss = F.l1_loss(predicted_future, future_sequence)

        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, graph_matrix)
            return regression_loss + effective_weight * conservation_loss
        return regression_loss

    def _diffusion_loss(self, future_sequence, condition_features, graph_matrix):
        """SNR-weighted MSE diffusion loss + optional conservation."""
        B = future_sequence.size(0)
        device = future_sequence.device

        # 1. Sample random timesteps
        t = torch.randint(0, self.diffusion_steps, (B,), device=device)

        # 2. Forward diffusion: Y_t = √ᾱ_t · Y_0 + √(1-ᾱ_t) · ε
        Y_t, epsilon = self.diffusion.add_noise(future_sequence, t)

        # 3. Predict noise
        epsilon_theta = self.denoiser(Y_t, t, condition_features, graph_matrix)

        # 4. SNR-weighted MSE (Improved DDPM)
        alpha_bar_t = self.diffusion.alphas_cumprod[t]
        snr = alpha_bar_t / (1.0 - alpha_bar_t).clamp_min(1e-8)
        loss_weight = (snr + 1.0).clamp(max=10.0)
        loss_weight = loss_weight[:, None, None, None]  # [B] → [B, 1, 1, 1]
        diffusion_loss = (
            loss_weight * F.mse_loss(epsilon_theta, epsilon, reduction='none')
        ).mean()

        # 5. Conservation loss (only for low-noise steps where
        #    one-step reconstruction is meaningful)
        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            low_noise_mask = t < int(0.7 * self.diffusion_steps)
            if low_noise_mask.any():
                Y_0_pred = self.diffusion.predict_start_from_noise(
                    Y_t[low_noise_mask], t[low_noise_mask],
                    epsilon_theta[low_noise_mask],
                )
                conservation_loss = self._fuzzy_conservation_loss(
                    Y_0_pred, graph_matrix,
                )
                return diffusion_loss + effective_weight * conservation_loss

        return diffusion_loss

    # ═══════════════════════════════════════════════════════════════
    #  Conservation loss (shared by both paths)
    # ═══════════════════════════════════════════════════════════════

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

    def _get_effective_fcm_weight(self):
        """Linearly ramp FCM loss weight over warmup epochs.

        Symmetric to ``_get_effective_conservation_weight`` so both
        regularisation terms follow the same ramp-in schedule.
        """
        if self.fcm_loss_weight <= 0:
            return 0.0
        if self.fcm_warmup_epochs <= 0:
            return self.fcm_loss_weight
        steps_per_epoch = self.fcm_steps_per_epoch
        total_warmup_steps = self.fcm_warmup_epochs * steps_per_epoch
        if total_warmup_steps <= 0:
            return self.fcm_loss_weight
        if self._train_step_count >= total_warmup_steps:
            return self.fcm_loss_weight
        return self.fcm_loss_weight * (self._train_step_count / total_warmup_steps)

    def _fuzzy_conservation_loss(self, future_sequence, fuzzy_relation):
        """Łukasiewicz T-norm fuzzy conservation loss.

        FlowPressure[i→j] = max(0, congestion[i] + R[i,j] - 1)

        Intuition: "IF node i is congested AND relation(i,j) is strong,
        THEN there exists flow pressure from i to j."
        """
        if future_sequence.size(1) < 2:
            return future_sequence.new_tensor(0.0)

        node_state = future_sequence[..., self.physics_channel_idx]
        current_state = node_state[:, :-1, :]
        next_state = node_state[:, 1:, :]
        temporal_delta = next_state - current_state

        s_min = current_state.amin(dim=(0, 1), keepdim=True)
        s_max = current_state.amax(dim=(0, 1), keepdim=True).clamp_min(s_min + 1e-6)
        congestion = ((current_state - s_min) / (s_max - s_min)).clamp(0.0, 1.0)

        R = fuzzy_relation.to(device=congestion.device, dtype=congestion.dtype)
        c = congestion.unsqueeze(-1)
        r = R.unsqueeze(0).unsqueeze(0)
        flow_pressure = (c + r - 1.0).clamp(min=0.0)

        inflow = flow_pressure.sum(dim=-2)
        outflow = flow_pressure.sum(dim=-1)
        net_pressure = inflow - outflow
        net_pressure = net_pressure * (s_max - s_min)

        residual = temporal_delta - net_pressure
        return residual.pow(2).mean()
