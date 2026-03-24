import math
import numpy as np
import torch
import torch.nn as nn

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


def _row_normalize(adj, eps=1e-6):
    denom = adj.sum(-1, keepdim=True) + eps
    return adj / denom


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (B, N, F), adj: (B, N, N)
        adj = _row_normalize(adj)
        agg = torch.einsum("bnm,bmf->bnf", adj, x)
        return self.proj(agg)


class DynamicAdjacency(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, proj_dim)
        self.k = nn.Linear(in_dim, proj_dim)
        self.scale = math.sqrt(proj_dim)

    def forward(self, x):
        # x: (B, N, F)
        q = self.q(x)
        k = self.k(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        return torch.softmax(scores, dim=-1)


class BiasMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.num_heads = num_heads

    def forward(self, x, bias=None):
        # x: (B, L, D), bias: (B, L, L)
        attn_mask = None
        if bias is not None:
            attn_mask = bias.repeat_interleave(self.num_heads, dim=0)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out


class SpatialBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout, ffn_dim):
        super().__init__()
        self.attn = BiasMultiheadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        x = x + self.drop(self.attn(self.norm1(x), bias))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GSHFlow(AbstractTrafficStateModel):
    """
    Graph-Spectral Hierarchical Transformer for traffic flow prediction.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = data_feature.get("scaler")
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.device = config.get("device", torch.device("cpu"))

        self.d_model = config.get("d_model", 64)
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 2)
        self.ffn_dim = config.get("ffn_dim", self.d_model * 4)
        self.dropout = config.get("dropout", 0.1)
        self.graph_bias_scale = config.get("graph_bias_scale", 1.0)
        self.use_dynamic_adj = config.get("use_dynamic_adj", True)
        self.num_supernodes = min(config.get("num_supernodes", 16), self.num_nodes)
        self.spectral_k = min(config.get("spectral_k", 8), self.num_nodes)

        self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        self.node_emb = nn.Embedding(self.num_nodes, self.d_model)
        self.pos_emb = nn.Embedding(self.input_window, self.d_model)
        self.input_drop = nn.Dropout(self.dropout)

        self.graph_conv = GraphConv(self.d_model, self.d_model)
        self.dynamic_adj = DynamicAdjacency(self.d_model, self.d_model)
        self.alpha_static = nn.Parameter(torch.tensor(0.0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.spatial_blocks = nn.ModuleList(
            [SpatialBlock(self.d_model, self.num_heads, self.dropout, self.ffn_dim)
             for _ in range(self.num_layers)]
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid(),
        )

        self.supernode_assign = nn.Parameter(
            0.1 * torch.randn(self.num_nodes, self.num_supernodes)
        )

        self.temporal_proj = nn.Linear(self.input_window, self.output_window)
        self.out_proj = nn.Linear(self.d_model, self.output_dim)

        adj_mx = data_feature.get("adj_mx")
        adj_static = np.array(adj_mx, dtype=np.float32)
        adj_static[np.isinf(adj_static)] = 0.0
        adj_static[np.isnan(adj_static)] = 0.0
        adj_static = adj_static + np.eye(self.num_nodes, dtype=np.float32)
        adj_static = adj_static / (adj_static.sum(axis=1, keepdims=True) + 1e-6)
        self.register_buffer("adj_static_norm", torch.tensor(adj_static))
        spec_basis = self._compute_spectral_basis(adj_static, self.spectral_k)
        self.register_buffer("spectral_basis", torch.tensor(spec_basis))

    @staticmethod
    def _compute_spectral_basis(adj, k):
        if k <= 0:
            return np.zeros((adj.shape[0], 0), dtype=np.float32)
        d = np.sum(adj, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-6))
        d_mat = np.diag(d_inv_sqrt)
        lap = np.eye(adj.shape[0], dtype=np.float32) - d_mat @ adj @ d_mat
        eigvals, eigvecs = np.linalg.eigh(lap)
        idx = np.argsort(eigvals)[:k]
        return eigvecs[:, idx].astype(np.float32)

    def _build_adj(self, x_last):
        # x_last: (B, N, D)
        batch_size = x_last.size(0)
        adj_static = self.adj_static_norm.unsqueeze(0).expand(batch_size, -1, -1)
        if not self.use_dynamic_adj:
            return adj_static
        adj_dyn = self.dynamic_adj(x_last)
        w = torch.sigmoid(self.alpha_static)
        return w * adj_static + (1.0 - w) * adj_dyn

    def _graph_encode(self, x, adj):
        # x: (B, T, N, D)
        outs = []
        for t in range(x.size(1)):
            h_t = self.graph_conv(x[:, t], adj) + x[:, t]
            outs.append(h_t)
        return torch.stack(outs, dim=1)

    def _temporal_encode(self, h):
        # h: (B, T, N, D)
        b, t, n, d = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, d)
        h = self.temporal_encoder(h)
        h = h.reshape(b, n, t, d).permute(0, 2, 1, 3)
        return h

    def _spatial_encode(self, h, adj):
        # h: (B, T, N, D)
        b, t, n, d = h.shape
        assign = torch.softmax(self.supernode_assign, dim=0)  # (N, K)
        outputs = []
        for t_idx in range(t):
            h_t = h[:, t_idx]  # (B, N, D)
            h_super = torch.einsum("bnf,nk->bkf", h_t, assign)
            if self.spectral_k > 0:
                h_spec = torch.einsum("nk,bnf->bkf", self.spectral_basis, h_t)
                tokens = torch.cat([h_t, h_super, h_spec], dim=1)
            else:
                tokens = torch.cat([h_t, h_super], dim=1)

            l = tokens.size(1)
            bias = torch.zeros(b, l, l, device=tokens.device)
            bias[:, :n, :n] = self.graph_bias_scale * adj
            for block in self.spatial_blocks:
                tokens = block(tokens, bias)
            outputs.append(tokens[:, :n, :])
        return torch.stack(outputs, dim=1)

    def forward(self, batch):
        x = batch["X"]  # (B, T, N, F)
        b, t, n, _ = x.shape
        x = self.input_proj(x)

        node_ids = torch.arange(self.num_nodes, device=x.device)
        pos_ids = torch.arange(t, device=x.device) % self.input_window
        x = x + self.node_emb(node_ids)[None, None, :, :] + self.pos_emb(pos_ids)[None, :, None, :]
        x = self.input_drop(x)

        adj = self._build_adj(x[:, -1])
        h_g = self._graph_encode(x, adj)
        h_t = self._temporal_encode(h_g)
        h_s = self._spatial_encode(h_g, adj)

        gate = self.fusion_gate(torch.cat([h_t, h_s], dim=-1))
        h = gate * h_t + (1.0 - gate) * h_s

        if h.size(1) != self.input_window:
            if h.size(1) > self.input_window:
                h = h[:, -self.input_window:]
            else:
                pad_len = self.input_window - h.size(1)
                pad = torch.zeros(b, pad_len, n, self.d_model, device=h.device, dtype=h.dtype)
                h = torch.cat([pad, h], dim=1)

        h = h.permute(0, 2, 3, 1)  # (B, N, D, T)
        h = self.temporal_proj(h)
        h = h.permute(0, 3, 1, 2)  # (B, T_out, N, D)
        out = self.out_proj(h)
        return out

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch["y"]
        y_pred = self.predict(batch)
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        else:
            y_true = y_true[..., :self.output_dim]
            y_pred = y_pred[..., :self.output_dim]
        return loss.masked_mae_torch(y_pred, y_true, 0)
