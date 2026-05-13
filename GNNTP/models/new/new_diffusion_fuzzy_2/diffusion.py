"""Diffusion noise scheduler with DDPM/DDIM reverse sampling.

Manages β/α/ᾱ schedules and provides:
- Forward: add_noise(Y₀, t) → Y_t, ε
- Reverse: ddpm_step / ddim_step for sampling
- Recovery: predict_start_from_noise(Y_t, t, ε̂) → Ŷ₀
"""

import math

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """Diffusion schedule and forward/backward transition utilities.

    Pre-computes all buffers (betas, alphas, cumulative products, etc.)
    for efficient indexing during training and sampling.
    """

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
        """Sample timesteps uniformly for training.

        Returns:
            [B] long tensor with values in [0, diffusion_steps).
        """
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

    @staticmethod
    def _extract(schedule_values, timesteps, target_shape):
        """Gather scheduler values by timestep and reshape for broadcasting.

        Args:
            schedule_values: [T] or [T, ...] buffer.
            timesteps: [B] indices.
            target_shape: Shape of the tensor to broadcast to.

        Returns:
            [B, 1, 1, 1] (reshaped to match target_shape trailing dims).
        """
        batch_size = timesteps.shape[0]
        extracted = schedule_values.gather(0, timesteps)
        return extracted.reshape(batch_size, *([1] * (len(target_shape) - 1)))

    def add_noise(self, clean_future, timesteps, noise=None):
        """Forward diffusion q(Y_t | Y_0).

        Y_t = √ᾱ_t · Y_0 + √(1-ᾱ_t) · ε

        Args:
            clean_future: [B, ...] clean target (Y_0).
            timesteps: [B] integer timestep indices.
            noise: Optional pre-sampled noise tensor.

        Returns:
            (Y_t, ε) tuple — noisy future and the noise added.
        """
        if noise is None:
            noise = torch.randn_like(clean_future)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_future.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean_future.shape
        )
        noisy_future = sqrt_alpha_bar * clean_future + sqrt_one_minus_alpha_bar * noise
        return noisy_future, noise

    def predict_start_from_noise(self, noisy_future, timesteps, predicted_noise):
        """Recover clean sample estimate Ŷ_0 from noisy Y_t and predicted ε̂.

        Ŷ_0 = (Y_t - √(1-ᾱ_t) · ε̂) / √ᾱ_t

        Args:
            noisy_future: [B, ...] noisy sample Y_t.
            timesteps: [B] integer timestep indices.
            predicted_noise: [B, ...] predicted noise ε̂.

        Returns:
            [B, ...] estimated clean sample Ŷ_0.
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_future.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_future.shape
        )
        return (noisy_future - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar.clamp_min(1e-12)

    def ddpm_step(self, current_state, timesteps, predicted_noise):
        """Single DDPM reverse step p(Y_{t-1} | Y_t).

        Uses the posterior mean + variance formula from the DDPM paper.
        Adds noise for t > 0 only (zero for t = 0).

        Args:
            current_state: [B, ...] current state Y_t.
            timesteps: [B] current timestep indices.
            predicted_noise: [B, ...] predicted noise ε̂.

        Returns:
            [B, ...] next state Y_{t-1}.
        """
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

    def ddim_step(self, current_state, timesteps, predicted_noise, eta=0.0, t_next=None):
        """Single DDIM reverse step for faster deterministic/stochastic sampling.

        When sampling with non-consecutive timesteps (num_sampling_steps < diffusion_steps),
        pass `t_next` to use the correct ᾱ of the previous sampling step.

        Args:
            current_state: [B, ...] current state Y_t.
            timesteps: [B] current timestep indices.
            predicted_noise: [B, ...] predicted noise ε̂.
            eta: 0 = deterministic, 1 = stochastic (DDPM-like).
            t_next: [B] next timestep indices (required for stride > 1).

        Returns:
            [B, ...] next state Y_{t-1}.
        """
        alpha_bar_t = self._extract(self.alphas_cumprod, timesteps, current_state.shape)
        if t_next is not None:
            alpha_bar_prev = self._extract(self.alphas_cumprod, t_next, current_state.shape)
        else:
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
