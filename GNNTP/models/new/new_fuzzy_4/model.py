"""FuzDiff Phase D — simplified fuzzy graph + conditional diffusion.

NewFuzzy4 drops all auxiliary losses (conservation, FCM) in favor of
a single pure-MSE training signal, following the TSGDiff/DDPM recipe.

Two paths controlled by ``use_diffusion``:

  use_diffusion = True  → DDIM conditional diffusion (pure MSE training)
  use_diffusion = False → deterministic encoder-decoder regression
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


class NewFuzzy4(AbstractTrafficStateModel):
    """Fuzzy relational graph + optional diffusion (simplified training).

    Training loss is pure:
      - Diffusion path:  ``MSE(ε̂, ε)``  (standard DDPM)
      - Deterministic:   ``L1(Y_pred, Y_true)``

    No auxiliary conservation / FCM losses — the fuzzy graph is
    learned end-to-end from the single prediction signal alone.
    """

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

        # ── Fuzzy Relational Graph ─────────────────────────────────
        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_num_sets = config.get("fuzzy_num_sets", 3)

        # ── Diffusion ──────────────────────────────────────────────
        self.use_diffusion = config.get("use_diffusion", True)
        if self.use_diffusion:
            self.diffusion_steps = config.get("diffusion_steps", 200)
            self.num_sampling_steps = config.get("num_sampling_steps", 50)
            self.ddim_eta = config.get("ddim_eta", 0.0)
            self.prediction_clamp_min = config.get("prediction_clamp_min", -3.0)
            self.prediction_clamp_max = config.get("prediction_clamp_max", 3.0)
            self.denoiser_layers = config.get("denoiser_layers", 4)
            self.diffusion = DiffusionScheduler(
                diffusion_steps=self.diffusion_steps,
                schedule=config.get("diffusion_schedule", "cosine"),
                beta_start=config.get("beta_start", 1e-4),
                beta_end=config.get("beta_end", 0.02),
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
        """Training → returns loss.  Inference → predicts."""
        if self.training:
            return self.calculate_loss(batch)
        return self.predict(batch)

    def predict(self, batch):
        """Predict future traffic from history.

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
        """Compute loss: pure MSE (diffusion) or L1 (deterministic)."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition_features = self.encode_condition(history_sequence)
        graph_matrix = self._get_graph_matrix(history_sequence)

        if self.use_diffusion:
            return self._diffusion_loss(future_sequence, condition_features, graph_matrix)
        else:
            return self._deterministic_loss(future_sequence, condition_features, graph_matrix)

    def _deterministic_loss(self, future_sequence, condition_features, graph_matrix):
        """Pure L1 regression."""
        predicted_future = self.future_decoder(condition_features, graph_matrix)
        return F.l1_loss(predicted_future, future_sequence)

    def _diffusion_loss(self, future_sequence, condition_features, graph_matrix):
        """Pure MSE noise-prediction loss (standard DDPM).

        Simplified from the SNR-weighted + conservation version in
        new_fuzzy_3 — follows the TSGDiff recipe: a single clean
        loss signal avoids gradient conflicts.
        """
        B = future_sequence.size(0)
        device = future_sequence.device

        # 1. Sample random timesteps
        t = torch.randint(0, self.diffusion_steps, (B,), device=device)

        # 2. Forward diffusion: Y_t = √ᾱ_t · Y_0 + √(1-ᾱ_t) · ε
        Y_t, epsilon = self.diffusion.add_noise(future_sequence, t)

        # 3. Predict noise
        epsilon_theta = self.denoiser(Y_t, t, condition_features, graph_matrix)

        # 4. Pure MSE (standard DDPM / TSGDiff formulation)
        return F.mse_loss(epsilon_theta, epsilon)
