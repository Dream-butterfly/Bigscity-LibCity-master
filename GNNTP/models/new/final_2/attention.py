"""Multi-head attention and feed-forward network components."""

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Batch-first multi-head attention with optional cross-attention.

    Uses PyTorch 2.x native F.scaled_dot_product_attention with flash attention
    backend when available.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self._logger = getLogger(__name__)

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context=None, mask=None):
        """Apply multi-head attention on query and context.

        Args:
            query: [B, Lq, D] query tensor.
            context: [B, Lkv, D] key/value tensor (None → self-attention).
            mask: Optional attention mask (2D/3D/4D, bool or float).

        Returns:
            [B, Lq, D] attended output.
        """
        if context is None:
            context = query
        batch_size, query_len, _ = query.shape
        context_len = context.size(1)

        query = self.query_projection(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        key = self.key_projection(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        value = self.value_projection(context).view(batch_size, context_len, self.num_heads, self.head_dim)

        # ── Numerical safeguard: clamp Q/K to ~±3σ to prevent softmax overflow ──
        # head_dim=48, scale=0.144 → max safe Q·K element < ln(FP32_max) ≈ 87
        #   Q_i×K_i×scale bounded to ~87/48≈1.8 → |Q_i|,|K_i| < √(1.8×√48) ≈ 3.5
        # We use ±10 which is conservative but well within the ~3σ range of
        # LayerNorm-conditioned activations.
        qk_clamp_val = 10.0
        query = query.clamp(-qk_clamp_val, qk_clamp_val)
        key = key.clamp(-qk_clamp_val, qk_clamp_val)
        value = value.clamp(-qk_clamp_val, qk_clamp_val)

        # ── NaN detection (one-shot, logged at most once per forward) ──
        if not torch.isfinite(query).all():
            self._logger.warning(
                "NaN/Inf in attention query! shape=%s", tuple(query.shape))
        if not torch.isfinite(key).all():
            self._logger.warning(
                "NaN/Inf in attention key! shape=%s", tuple(key.shape))

        # [B, L, H, D/H] → [B, H, L, D/H]
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
        # [B, H, L, D/H] → [B, L, H, D/H] → [B, L, D]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)

        # ── Post-attention NaN guard: silently replace NaN/Inf with zeros ──
        if not torch.isfinite(attention_output).all():
            self._logger.warning(
                "NaN/Inf in attention output! shape=%s, "
                "replacing with zeros.", tuple(attention_output.shape))
            attention_output = torch.nan_to_num(
                attention_output, nan=0.0, posinf=0.0, neginf=0.0)

        return self.output_projection(attention_output)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Dropout → Linear → Dropout."""

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
