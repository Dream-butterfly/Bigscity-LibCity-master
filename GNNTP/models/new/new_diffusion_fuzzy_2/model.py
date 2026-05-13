"""Conditional graph-attention diffusion model for traffic forecasting.

NewDiffusion = STEncoder (condition) + AttentionDenoiser (noise prediction)
            + DiffusionScheduler (noise schedule & sampling).

Core contributions:
1. Fuzzy graph learning via Gaussian membership functions
2. Fuzzy conservation loss for physics-informed training
3. Conditional diffusion with SNR+1 noise weighting
"""

import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel

from .encoder import STEncoder
from .denoiser import AttentionDenoiser
from .diffusion import DiffusionScheduler
from .utils import expand_adjacency_batch


class NewDiffusion(AbstractTrafficStateModel):
    """Graph + Attention + Conditional Diffusion with adaptive graph and conservation prior.

    Training: forward() → calculate_loss() → scalar (DDP-safe via _ddp_loss_through_forward).
    Inference: forward() → predict() → sample() → [B, T_out, N, C_out].
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
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_heads = config.get("num_heads", 4)
        self.encoder_layers = config.get("encoder_layers", 2)
        self.denoiser_layers = config.get("denoiser_layers", 4)
        self.ffn_hidden_dim = config.get("ffn_hidden_dim", self.hidden_dim * 2)
        self.graph_k_hop = config.get("graph_k_hop", 2)
        self.dropout = config.get("dropout", 0.1)
        self.use_spatiotemporal_attention = config.get("use_spatiotemporal_attention", True)
        self.use_temporal_position_embedding = config.get("use_temporal_position_embedding", True)
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", True)

        # ── Adaptive / Fuzzy Graph ────────────────────────────────
        self.use_adaptive_graph = config.get("use_adaptive_graph", True)
        self.adaptive_graph_embed_dim = config.get("adaptive_graph_embed_dim", 32)
        self.adaptive_graph_topk = config.get("adaptive_graph_topk", None)
        self.adaptive_graph_blend_init = config.get("adaptive_graph_blend_init", 0.5)
        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_graph_num_sets = config.get("fuzzy_graph_num_sets", 3)
        self.fuzzy_graph_sigma_init = config.get("fuzzy_graph_sigma_init", 0.7)
        if self.use_fuzzy_graph:
            if self.fuzzy_graph_num_sets < 2:
                raise ValueError("fuzzy_graph_num_sets must be >= 2 when use_fuzzy_graph is enabled.")
            if self.fuzzy_graph_sigma_init <= 0:
                raise ValueError("fuzzy_graph_sigma_init must be > 0 when use_fuzzy_graph is enabled.")

        # ── Diffusion ─────────────────────────────────────────────
        self.diffusion_steps = config.get("diffusion_steps", 200)
        self.diffusion_schedule = config.get("diffusion_schedule", "linear")
        self.beta_start = config.get("beta_start", 1e-4)
        self.beta_end = config.get("beta_end", 2e-2)
        self.num_sampling_steps = max(1, min(
            config.get("num_sampling_steps", self.diffusion_steps), self.diffusion_steps
        ))
        self.num_prediction_samples = int(config.get("num_prediction_samples", 1))
        self.sampling_method = config.get("sampling_method", "ddpm").lower()
        self.ddim_eta = config.get("ddim_eta", 0.0)
        if self.sampling_method not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampling_method: {self.sampling_method}")
        if self.num_prediction_samples < 1:
            raise ValueError("num_prediction_samples must be >= 1.")

        # ── Clamp ─────────────────────────────────────────────────
        _raw_min = config.get("prediction_clamp_min", None)
        _raw_max = config.get("prediction_clamp_max", None)
        self.prediction_clamp_min: float | None = float(_raw_min) if _raw_min is not None else None
        self.prediction_clamp_max: float | None = float(_raw_max) if _raw_max is not None else None

        # ── Physics Loss ──────────────────────────────────────────
        self.physics_loss_weight = config.get("physics_loss_weight", 0.05)
        self.physics_warmup_steps = int(max(0, config.get("physics_warmup_steps", 0)))
        self.physics_warmup_start_ratio = float(config.get("physics_warmup_start_ratio", 0.2))
        self.physics_warmup_mode = str(config.get("physics_warmup_mode", "linear")).lower()
        self.flow_conservation_coeff = config.get("flow_conservation_coeff", 1.0)
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self.use_fuzzy_conservation = config.get("use_fuzzy_conservation", True)
        self.fuzzy_conservation_threshold = config.get("fuzzy_conservation_threshold", 0.6)
        self.fuzzy_conservation_temperature = config.get("fuzzy_conservation_temperature", 8.0)
        if self.physics_loss_weight < 0:
            raise ValueError("physics_loss_weight must be >= 0.")
        if not 0.0 <= self.physics_warmup_start_ratio <= 1.0:
            raise ValueError("physics_warmup_start_ratio must be in [0, 1].")
        if self.physics_warmup_mode not in {"linear", "cosine"}:
            raise ValueError("physics_warmup_mode must be either `linear` or `cosine`.")
        if self.physics_channel_idx < 0:
            raise ValueError("physics_channel_idx must be >= 0.")
        if self.physics_channel_idx >= self.output_dim:
            raise ValueError(
                f"physics_channel_idx={self.physics_channel_idx} is out of range for output_dim={self.output_dim}."
            )
        if self.use_fuzzy_conservation and self.fuzzy_conservation_temperature <= 0:
            raise ValueError("fuzzy_conservation_temperature must be > 0 when use_fuzzy_conservation is enabled.")

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
            input_window=self.input_window,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            adaptive_graph_enabled=self.use_adaptive_graph,
            adaptive_graph_embed_dim=self.adaptive_graph_embed_dim,
            adaptive_graph_topk=self.adaptive_graph_topk,
            adaptive_graph_blend_init=self.adaptive_graph_blend_init,
            fuzzy_graph_enabled=self.use_fuzzy_graph,
            fuzzy_graph_num_sets=self.fuzzy_graph_num_sets,
            fuzzy_graph_sigma_init=self.fuzzy_graph_sigma_init,
            num_nodes=self.num_nodes,
            static_adjacency=adjacency_matrix,
        )
        self.diffusion_scheduler = DiffusionScheduler(
            diffusion_steps=self.diffusion_steps,
            schedule=self.diffusion_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

        # ── Training state ────────────────────────────────────────
        self._train_step_count = 0
        self._physics_warmup_start_logged = False
        self._physics_warmup_end_logged = False
        self._sampling_schedule_cache = {}

    # ═══════════════════════════════════════════════════════════════
    #  Physics warmup
    # ═══════════════════════════════════════════════════════════════

    def _get_effective_physics_weight(self):
        """Gradually ramp up physics loss weight to stabilize early-stage training."""
        if self.physics_loss_weight <= 0:
            return 0.0
        if self.physics_warmup_steps <= 0:
            return float(self.physics_loss_weight)

        progress = min(1.0, self._train_step_count / max(self.physics_warmup_steps, 1))
        if self.physics_warmup_mode == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        scale = self.physics_warmup_start_ratio + (1.0 - self.physics_warmup_start_ratio) * progress
        return float(self.physics_loss_weight) * scale

    # ═══════════════════════════════════════════════════════════════
    #  Encoding
    # ═══════════════════════════════════════════════════════════════

    def encode_condition(self, history_sequence):
        """Encode historical traffic into condition feature H.

        Args:
            history_sequence: [B, Tin, N, Cin] historical data.

        Returns:
            [B, Tin, N, D] condition features.
        """
        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        return self.condition_encoder(history_sequence, adjacency_matrix)

    # ═══════════════════════════════════════════════════════════════
    #  Forward
    # ═══════════════════════════════════════════════════════════════

    def forward(self, batch):
        """Forward entry. Training → returns loss (DDP-synced). Inference → predicts."""
        if self.training:
            return self.calculate_loss(batch)
        return self.predict(batch)

    # ═══════════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute diffusion objective E[||ε - ε_θ(Y_t, t, H)||^2] + optional conservation."""
        if self.training:
            self._train_step_count += 1
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., : self.output_dim]
        condition_features = self.encode_condition(history_sequence)

        batch_size = future_sequence.shape[0]
        timesteps = self.diffusion_scheduler.sample_timesteps(batch_size, history_sequence.device)
        noisy_future, true_noise = self.diffusion_scheduler.add_noise(future_sequence, timesteps)

        adjacency_matrix = self.adjacency_matrix.to(history_sequence.device)
        effective_physics_weight = self._get_effective_physics_weight()
        need_physics = effective_physics_weight > 0

        # Physics warmup logging (first epoch only)
        if self.training and self.physics_warmup_steps > 0:
            if not self._physics_warmup_start_logged:
                self._logger.info(
                    "Enable physics warmup: steps=%d start_ratio=%.3f mode=%s.",
                    self.physics_warmup_steps,
                    self.physics_warmup_start_ratio,
                    self.physics_warmup_mode,
                )
                self._physics_warmup_start_logged = True
            if self._train_step_count >= self.physics_warmup_steps and not self._physics_warmup_end_logged:
                self._logger.info("Physics warmup reached full weight at step=%d.", self._train_step_count)
                self._physics_warmup_end_logged = True

        if need_physics:
            predicted_noise, adaptive_adjacency = self.noise_predictor(
                noisy_future, timesteps, condition_features, adjacency_matrix, return_last_adjacency=True
            )
        else:
            predicted_noise = self.noise_predictor(
                noisy_future, timesteps, condition_features, adjacency_matrix, return_last_adjacency=False
            )

        # SNR+1 weighted MSE (Improved DDPM)
        loss_per_element = F.mse_loss(predicted_noise, true_noise, reduction='none')
        alpha_bar_t = self.diffusion_scheduler._extract(
            self.diffusion_scheduler.alphas_cumprod, timesteps, true_noise.shape
        )
        snr = alpha_bar_t / (1.0 - alpha_bar_t).clamp_min(1e-8)
        loss_weight = (snr + 1.0).clamp(max=10.0)
        diffusion_loss = (loss_weight * loss_per_element).mean()

        if not need_physics:
            return diffusion_loss

        # Conservation loss branch
        predicted_future = self.diffusion_scheduler.predict_start_from_noise(
            noisy_future, timesteps, predicted_noise
        )
        conservation_loss = self._traffic_conservation_loss(predicted_future, adaptive_adjacency)
        return diffusion_loss + effective_physics_weight * conservation_loss

    def _traffic_conservation_loss(self, future_sequence, adjacency_matrix):
        """Penalize mismatch between temporal state change and graph net-flow.

        Uses fuzzy congestion membership to up-weight high-traffic nodes.
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

        outflow = current_state * adjacency_matrix.sum(dim=-1).unsqueeze(1)
        inflow = torch.einsum("bij,btj->bti", adjacency_matrix.transpose(1, 2), current_state)
        net_flow = inflow - outflow

        residual = temporal_delta - self.flow_conservation_coeff * net_flow
        if not self.use_fuzzy_conservation:
            return residual.pow(2).mean()

        # Fuzzy congestion membership
        node_scale = current_state.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized_state = current_state.abs() / node_scale
        high_congestion_membership = torch.sigmoid(
            self.fuzzy_conservation_temperature * (normalized_state - self.fuzzy_conservation_threshold)
        )
        fuzzy_weight = 0.5 + high_congestion_membership
        return (fuzzy_weight * residual.pow(2)).mean()

    # ═══════════════════════════════════════════════════════════════
    #  Inference / Sampling
    # ═══════════════════════════════════════════════════════════════

    def _sample_once(self, condition_features, sampling_schedule):
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
        for i, step in enumerate(sampling_schedule):
            timestep = step.expand(batch_size)
            predicted_noise = self.noise_predictor(
                future_state, timestep, condition_features, adjacency_matrix
            )
            if self.sampling_method == "ddim":
                t_next = None
                if i < len(sampling_schedule) - 1:
                    t_next = sampling_schedule[i + 1].expand(batch_size)
                future_state = self.diffusion_scheduler.ddim_step(
                    future_state, timestep, predicted_noise,
                    eta=self.ddim_eta, t_next=t_next,
                )
            else:
                future_state = self.diffusion_scheduler.ddpm_step(
                    future_state, timestep, predicted_noise
                )
            # Clamp to prevent DDIM/DDPM divergence and peak underestimation
            if self.prediction_clamp_min is not None or self.prediction_clamp_max is not None:
                future_state = future_state.clamp(
                    min=-float('inf') if self.prediction_clamp_min is None else self.prediction_clamp_min,
                    max=float('inf') if self.prediction_clamp_max is None else self.prediction_clamp_max,
                )
        return future_state

    def _get_sampling_schedule(self, device):
        """Cache reverse diffusion schedule per device."""
        cache_key = str(device)
        if cache_key not in self._sampling_schedule_cache:
            self._sampling_schedule_cache[cache_key] = torch.linspace(
                self.diffusion_steps - 1,
                0,
                self.num_sampling_steps,
                device=device,
            ).long()
        return self._sampling_schedule_cache[cache_key]

    def sample(self, history_sequence, num_samples=1, return_all=False):
        """Sample future trajectories for uncertainty-aware prediction.

        Args:
            history_sequence: [B, Tin, N, Cin] historical data.
            num_samples: Number of trajectories to sample.
            return_all: If True, return [K, B, Tout, N, Cout].
                        If False, return mean [B, Tout, N, Cout].

        Returns:
            Predicted future with shape depending on return_all.
        """
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        condition_features = self.encode_condition(history_sequence)
        sampling_schedule = self._get_sampling_schedule(condition_features.device)
        batch_size = condition_features.shape[0]
        expanded_condition = condition_features.repeat_interleave(num_samples, dim=0)
        sampled_futures = self._sample_once(expanded_condition, sampling_schedule).view(
            num_samples, batch_size, self.output_window, self.num_nodes, self.output_dim
        )
        if return_all:
            return sampled_futures
        return sampled_futures.mean(dim=0)

    def predict(self, batch):
        """Predict future traffic as mean of multi-sample diffusion trajectories.

        Args:
            batch: dict with 'X' key [B, Tin, N, Cin].

        Returns:
            [B, Tout, N, Cout] predicted future.
        """
        history_sequence = batch["X"]
        return self.sample(
            history_sequence, num_samples=self.num_prediction_samples, return_all=False
        )
