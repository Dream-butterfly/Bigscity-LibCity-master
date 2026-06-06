"""Multi-head attention and feed-forward network components (copied).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Batch-first multi-head attention with optional cross-attention.
    """

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


class FeedForwardNetwork(nn.Module):
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
        return self.network(features)

