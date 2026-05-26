"""Fuzzy Region Routing (FRR) — low-rank latent region routing.

Replaces spatial self-attention with region-mediated global interaction:
  Node features → soft region assignment → region token aggregation
  → Region Transformer (self-attn among K region tokens)
  → node readback

Complexity: O(NK + (T·K)²) vs O((T·N)²) for full spatiotemporal attention.
K regions serve as a low-rank bottleneck capturing macro-scale
traffic organization patterns.

Stability diagnostics:
  - H(i) = -Σ_k u_k log u_k       Region entropy (high → boundary node)
  - S(i) = u_{(1)} - u_{(2)}       Assignment margin (low → unstable assignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, FeedForwardNetwork


# ═══════════════════════════════════════════════════════════════════
#  Region Transformer
# ═══════════════════════════════════════════════════════════════════

class RegionTransformer(nn.Module):
    """Self-attention Transformer on region tokens [K, D] or [T×K, D].

    Performs standard Transformer interaction in region space.
    Static mode: [K, D].  Temporal mode: [T×K, D] — K regions × T
    timesteps jointly attend.

    Complexity: O(L²·D) where L = K or T×K. With T=12, K=8:
    L²=9216 attn weights vs (T·N)²=6M for full spatiotemporal.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim * 2, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, region_tokens: torch.Tensor) -> torch.Tensor:
        """Pre-LN Transformer on region tokens.

        Args:
            region_tokens: [K, D] or [T×K, D].

        Returns:
            Same shape as input.
        """
        # MHA expects [B, L, D] → add batch dim
        need_squeeze = region_tokens.dim() == 2
        if need_squeeze:
            region_tokens = region_tokens.unsqueeze(0)      # [1, L, D]

        x = self.norm1(region_tokens + self.dropout(
            self.self_attn(region_tokens)))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        if need_squeeze:
            x = x.squeeze(0)
        return x


# ═══════════════════════════════════════════════════════════════════
#  Fuzzy Region Routing (FRR)
# ═══════════════════════════════════════════════════════════════════

class FuzzyCellAttention(nn.Module):
    """Fuzzy Region Routing: node → region → node propagation.

    Forms a local-global spatial dual with graph convolution:
        H' = λ₁ · GCN(H) + λ₂ · FRR(H)

    GCN handles local topology propagation.  FRR provides global
    functional routing through a low-rank region bottleneck.

    Architecture (static mode, [N, D]):
      ① Soft region assignment via learnable prototypes
      ② sqrt routing weights
      ③ Message encoding
      ④ Region token aggregation  [N, D] → [K, D]
      ⑤ Topology-aware band-pass gate
      ⑥ Region Transformer (self-attn among region tokens)
      ⑦ Node readback  [K, D] → [N, D]
      ⑧ Output projection + residual blend
    """

    def __init__(
        self,
        hidden_dim: int,
        num_cells: int = 8,
        cell_blend_init: float = 0.3,
        band_center_init: float = 1.1,
        band_width_init: float = 0.7,
        region_transformer_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Node feature dimension.
            num_cells: Number of region prototypes K.
            cell_blend_init: Initial FRR blend weight λ₂ (sigmoid-transformed).
            band_center_init: Initial band-pass center μ (log-hop-space).
            band_width_init: Initial band-pass width σ.
            region_transformer_layers: 1 → RegionTransformer, 0 → identity.
            num_heads: MHA heads for RegionTransformer.
            dropout: Dropout rate.
        """
        super().__init__()
        if num_cells < 2:
            raise ValueError("num_cells must be >= 2.")
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim

        # ── Region prototypes ──
        self.centers = nn.Parameter(torch.randn(num_cells, hidden_dim) * 0.1)

        # ── Assignment variance (controls fuzziness) ──
        self.log_sigma_sq = nn.Parameter(torch.zeros(1))

        # ── Message encoding / decoding ──
        self.cell_transform = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # ── Learnable blend weight λ₂  ──
        self.cell_blend = nn.Parameter(torch.tensor(cell_blend_init))

        # ── Topological band-pass gate ──
        self.band_center = nn.Parameter(torch.tensor(band_center_init))
        self.band_width_raw = nn.Parameter(torch.tensor(band_width_init))

        # ── Region Transformer ──
        self.region_transformer = None
        if region_transformer_layers > 0:
            self.region_transformer = RegionTransformer(
                hidden_dim, num_heads=num_heads, dropout=dropout,
            )

        # ── Fuzzy graph conditioning (set externally by model.py) ──
        self.fuzzy_to_cell: nn.Module | None = None

    # ── ① Soft region assignment ─────────────────────────────

    def _compute_membership(
        self, x: torch.Tensor, mu_fuzzy: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Soft region assignment via learnable prototypes.

        u_ik = softmax( -||x_i - c_k||² / 2σ²  [+ Linear(μ_fuzzy_i)_k] )

        Args:
            x: [N, D] node features.
            mu_fuzzy: [N, K_f] optional fuzzy memberships for conditioning.

        Returns:
            u: [N, K] region membership, rows sum to 1.
        """
        dist_sq = torch.cdist(x, self.centers).pow(2)           # [N, K]
        sigma_sq = F.softplus(self.log_sigma_sq) + 0.01
        logits = -dist_sq / (2 * sigma_sq)

        if mu_fuzzy is not None and self.fuzzy_to_cell is not None:
            logits = logits + self.fuzzy_to_cell(mu_fuzzy)

        return F.softmax(logits, dim=-1)

    # ── ⑤ Topological band-pass gate ─────────────────────────

    def _bandpass_gate(
        self, graph_dist: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """Pure log-Gaussian band-pass on hop distance.

        d_ik = Σ_j B²_jk · dist[i,j] / Σ_j B²_jk      effective distance
        gate_ik = exp( -(log d_ik - μ)² / 2σ² )       log-Gaussian

        Properties:
        - d→0  (self/neighbor): log d→-∞, gate→0  (GCN covers local)
        - d≈e^μ (target range): gate peaks          (FRR target)
        - d→∞  (far nodes):     gate→0              (irrelevant)

        Args:
            graph_dist: [N, N] hop distance matrix.
            B: [N, K] routing weights (sqrt of membership).

        Returns:
            gate: [N, K] per-(node, region) gate values, row-normalized to [0,1].
        """
        B_sq = B.pow(2)
        region_weight = B_sq / B_sq.sum(dim=0, keepdim=True).clamp_min(1e-8)
        d_ik = graph_dist @ region_weight                         # [N, K]

        eps = 1e-6
        log_d = torch.log(d_ik.clamp(min=eps))
        mu = self.band_center
        sigma = F.softplus(self.band_width_raw) + 0.1
        gate = torch.exp(-(log_d - mu).pow(2) / (2 * sigma.pow(2)))  # [N, K]

        # Per-region normalization to [0, 1]
        gate_max = gate.max(dim=0, keepdim=True).values.clamp(min=eps)
        return gate / gate_max

    # ── Static forward ───────────────────────────────────────

    def _forward_static(
        self,
        x: torch.Tensor,
        graph_dist: torch.Tensor | None = None,
        mu_fuzzy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full FRR pipeline for static [N, D] input."""
        N, D = x.shape

        # ① Soft region assignment
        u = self._compute_membership(x, mu_fuzzy)                # [N, K]
        # ② Routing weights (sqrt preserves simplex geometry)
        B = u.sqrt().clamp(min=1e-8)                             # [N, K]

        # ③ Message encoding
        X_tilde = self.cell_transform(x)                         # [N, D]

        # ④ Region token aggregation
        M = B.T @ X_tilde                                        # [K, D]

        # ⑤ Topological band-pass gate
        if graph_dist is not None:
            gate = self._bandpass_gate(graph_dist, B)            # [N, K]
            gw_num = (gate * B.pow(2)).sum(dim=0)                # [K]
            gw_den = B.pow(2).sum(dim=0).clamp_min(1e-8)         # [K]
            gw = (gw_num / gw_den).unsqueeze(-1)                # [K, 1]
            M = M * gw                                         # [K, D]

        # ⑥ Region Transformer
        if self.region_transformer is not None:
            M = self.region_transformer(M)                       # [K, D]

        # ⑦ Node readback
        H = B @ M                                                # [N, D]

        # ⑧ Output projection + residual blend
        output = self.output_projection(H)
        blend = self.cell_blend.sigmoid()                        # λ₂ ∈ (0, 1)
        return blend * output

    # ── Temporal-aware forward ───────────────────────────────

    def _forward_temporal(
        self,
        x: torch.Tensor,
        graph_dist: torch.Tensor | None = None,
        mu_fuzzy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Temporal-aware FRR: [T, N, D] → [N, D].

        Region assignment is computed from time-averaged features
        (city-wide functional zones are shared), but region token
        aggregation operates per-timestep to preserve temporal dynamics.

        Region tokens: [T, K, D] → flattened to [T×K, D] for Transformer.
        """
        T, N, D = x.shape

        # ① Stable region assignment (time-averaged)
        x_static = x.mean(dim=0)                                 # [N, D]
        u = self._compute_membership(x_static, mu_fuzzy)         # [N, K]
        B = u.sqrt().clamp(min=1e-8)                             # [N, K]

        # ③④ Per-timestep aggregation → time-aware region tokens
        X_tilde = self.cell_transform(x)                         # [T, N, D]
        M = torch.einsum("nk,tnd->tkd", B, X_tilde)              # [T, K, D]

        # ⑤ Topological band-pass gate (static assignment based)
        if graph_dist is not None:
            gate = self._bandpass_gate(graph_dist, B)            # [N, K]
            gw_num = (gate * B.pow(2)).sum(dim=0)                # [K]
            gw_den = B.pow(2).sum(dim=0).clamp_min(1e-8)         # [K]
            gw = (gw_num / gw_den).view(1, -1, 1)               # [1, K, 1]
            M = M * gw                                         # [T, K, D]

        # ⑥ Region Transformer on flattened [T×K, D]
        if self.region_transformer is not None:
            M = M.reshape(T * self.num_cells, D)
            M = self.region_transformer(M)
            M = M.reshape(T, self.num_cells, D)

        # ⑦ Per-timestep readback → temporal mean
        H = torch.einsum("tkd,nk->tnd", M, B)                   # [T, N, D]
        H_mean = H.mean(dim=0)                                   # [N, D]

        # ⑧ Output projection + residual blend
        output = self.output_projection(H_mean)
        blend = self.cell_blend.sigmoid()
        return blend * output

    # ── Forward dispatch ─────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        graph_dist: torch.Tensor | None = None,
        mu_fuzzy: torch.Tensor | None = None,
        return_stability: bool = False,
    ):
        """Fuzzy Region Routing forward pass.

        Supports three input shapes:
        - [N, D]:       Static mode (single timestep or globally pooled)
        - [T, N, D]:    Temporal mode (per-timestep region tokens)
        - [B, T, N, D]: Batch-temporal mode (mean over batch, keep time)

        Args:
            x: Node features.
            graph_dist: [N, N] hop distance matrix for band-pass gate.
            mu_fuzzy: [N, K_f] fuzzy memberships for conditioning.
            return_stability: If True, returns (output, (entropy, margin)).

        Returns:
            output: [N, D] FRR output.
            (output, (H, S)): if return_stability=True.
        """
        # Dispatch by dimensionality
        if x.dim() == 4:
            x = x.mean(dim=0)                                    # [B,T,N,D] → [T,N,D]
        if x.dim() == 3:
            output = self._forward_temporal(x, graph_dist, mu_fuzzy)
        elif x.dim() == 2:
            output = self._forward_static(x, graph_dist, mu_fuzzy)
        else:
            raise ValueError(
                f"FuzzyCellAttention expects 2D/3D/4D input, got shape {x.shape}"
            )

        if return_stability:
            # Compute stability from the time-averaged assignment
            x_for_metrics = x.mean(dim=0) if x.dim() == 3 else x
            u = self._compute_membership(x_for_metrics, mu_fuzzy)
            H = -(u * (u + 1e-8).log()).sum(dim=-1)              # [N] entropy
            top2 = u.topk(2, dim=-1).values                       # [N, 2]
            S = top2[:, 0] - top2[:, 1]                           # [N] margin
            return output, (H, S)

        return output

    # ── Stability diagnostics ────────────────────────────────

    def get_stability_metrics(
        self, x: torch.Tensor, mu_fuzzy: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute region assignment stability metrics.

        Args:
            x: [N, D] node features.
            mu_fuzzy: [N, K_f] optional fuzzy conditioning.

        Returns:
            H: [N] region entropy — high → boundary / uncertain nodes.
            S: [N] assignment margin — low → assignment easily flipped.
        """
        u = self._compute_membership(x, mu_fuzzy)
        H = -(u * (u + 1e-8).log()).sum(dim=-1)
        top2 = u.topk(2, dim=-1).values
        S = top2[:, 0] - top2[:, 1]
        return H, S


# ═══════════════════════════════════════════════════════════════════
#  Pooling utilities (kept for backward-compatible diagnostics)
# ═══════════════════════════════════════════════════════════════════

class CellAttentionPool:
    """Pooling helpers for converting [B,T,N,D] features to [N,D].

    Kept for diagnostic and backward-compatible use.  The main FRR
    forward() now accepts 4D input directly and handles pooling
    internally.
    """

    @staticmethod
    def mean_pool(sequence_features: torch.Tensor) -> torch.Tensor:
        """Global mean pool: [B,T,N,D] → mean(B,T) → [N,D]."""
        return sequence_features.mean(dim=(0, 1))

    @staticmethod
    def batch_mean_pool(sequence_features: torch.Tensor) -> torch.Tensor:
        """Batch mean pool: [B,T,N,D] → mean(T) → [B,N,D]."""
        return sequence_features.mean(dim=1)

    @staticmethod
    def time_last(sequence_features: torch.Tensor) -> torch.Tensor:
        """Last timestep + batch mean: [B,T,N,D] → [-1] → mean(B) → [N,D]."""
        return sequence_features[:, -1].mean(dim=0)
