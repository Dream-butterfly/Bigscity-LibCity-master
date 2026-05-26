"""Graph convolution and fuzzy relational graph learning components.

Phase A — Fuzzy Relational Graph (new_fuzzy_2):
  ``FuzzyRelationalGraphLearner`` learns a fuzzy similarity relation R
  via max-min composition of node membership vectors μ ∈ [0,1]^K.
  ``FuzzyGraphConvolution`` does K-hop propagation using R^(k) (computed
  via max-min relational powers) × feature matmul.

Key distinction:
  - Relation propagation → max-min composition  (fuzzy relation algebra)
  - Feature propagation  → R^(k) @ X            (linear, R as weights)

``GraphConvolution`` and ``AdaptiveGraphLearner`` are kept for
backward compatibility / fallback.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_normalized_adjacency, expand_adjacency_batch


# ═══════════════════════════════════════════════════════════════════
#  Standard Graph Convolution (unchanged)
# ═══════════════════════════════════════════════════════════════════

class GraphConvolution(nn.Module):
    """K-hop graph convolution: X' = Σ_{k=0}^{K} Â^k X W_k."""

    def __init__(self, hidden_dim, k_hop=2):
        super().__init__()
        self.k_hop = k_hop
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)]
        )

    def forward(self, node_features, adjacency_matrix):
        batch_size, num_nodes, _ = node_features.shape
        adjacency_matrix = expand_adjacency_batch(adjacency_matrix, batch_size).to(
            device=node_features.device, dtype=node_features.dtype
        )
        adjacency_norm = build_normalized_adjacency(adjacency_matrix)
        adjacency_power = (
            torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype)
            .unsqueeze(0).expand(batch_size, -1, -1)
        )
        output = torch.zeros_like(node_features)
        for hop_index, projection in enumerate(self.projections):
            if hop_index > 0:
                adjacency_power = torch.bmm(adjacency_power, adjacency_norm)
            propagated = torch.bmm(adjacency_power, node_features)
            output = output + projection(propagated)
        return output


# ═══════════════════════════════════════════════════════════════════
#  Fuzzy Graph Convolution  (Phase A)
# ═══════════════════════════════════════════════════════════════════

class FuzzyGraphConvolution(nn.Module):
    """K-hop propagation via fuzzy relational powers × feature matmul.

    R^(k) computed via max-min composition (fuzzy relation algebra).
    Features propagated via R^(k) @ X (linear mixing in feature space).
    """

    def __init__(self, hidden_dim, k_hop=2):
        super().__init__()
        self.k_hop = k_hop
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)]
        )

    @staticmethod
    def _max_min_compose_2d(R, S):
        """Max-min composition on [N, N] matrices (no batch dim).

        (R∘S)[i,j] = max_k min(R[i,k], S[k,j])

        Intermediate: [N, N, N] instead of [B, N, N, N]
        """
        # R: [N, N] → [N, 1, N]; S: [N, N] → [1, N, N]
        return torch.max(
            torch.min(R.unsqueeze(1), S.unsqueeze(0)), dim=-1
        ).values  # [N, N]

    def forward(self, node_features, fuzzy_relation):
        """K-hop fuzzy propagation.

        All max-min compositions run on [N, N] — the fuzzy relation is
        shared across batch elements.  Only the final feature propagation
        (bmm) expands to batch dimension.

        Args:
            node_features:  [B, N, D]
            fuzzy_relation: [N, N] or [B, N, N] in [0, 1]
        Returns:
            [B, N, D]
        """
        batch_size, num_nodes, _ = node_features.shape

        # ── Extract base [N, N] relation (all batch elements identical) ──
        if fuzzy_relation.dim() == 3:
            R_base = fuzzy_relation[0].to(
                device=node_features.device, dtype=node_features.dtype
            )
        else:
            R_base = fuzzy_relation.to(
                device=node_features.device, dtype=node_features.dtype
            )
        # R_base: [N, N]

        # ── Pre-compute k-hop powers on [N, N] (tiny, shared) ──
        I = torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype)
        R_powers = [I]  # R^(0)
        current = R_base
        for _ in range(self.k_hop):
            R_powers.append(current)
            current = self._max_min_compose_2d(R_base, current)  # R^(k+1)

        # ── Feature propagation (memory-efficient: avoids [B,N,N] intermediates) ──
        # Hop 0: I @ X = X → projection directly
        output = self.projections[0](node_features)
        for hop_index in range(1, self.k_hop + 1):
            R_k = R_powers[hop_index]  # [N, N] — never expanded to batch
            # R_k @ X via reshape trick: [N,N] @ [N, B*D] → [N, B*D] → [B, N, D]
            # Saves ~N/D × memory vs. expand([B,N,N])
            x_flat = node_features.permute(1, 0, 2).reshape(num_nodes, -1)
            propagated_flat = torch.mm(R_k, x_flat)
            propagated = propagated_flat.reshape(num_nodes, batch_size, -1).permute(1, 0, 2)
            output = output + self.projections[hop_index](propagated)

        return output


# ═══════════════════════════════════════════════════════════════════
#  Fuzzy Relational Graph Learner  (Phase A — core innovation)
# ═══════════════════════════════════════════════════════════════════

class FuzzyRelationalGraphLearner(nn.Module):
    """Learn a fuzzy similarity relation R: N×N → [0,1].

    Mathematical construction:
      1. Node i has membership vector μ_i ∈ [0,1]^K over K fuzzy sets.
      2. Fuzzy similarity:  R[i,j] = max_k min(μ_k(i), μ_k(j))
         This is the standard max-min construction of a fuzzy
         tolerance relation from membership vectors.
      3. Enforced reflexivity:  R[i,i] = 1  by construction.
      4. K-hop powers via max-min composition:
           R^(k) = R ∘ R^(k-1)
           (R∘S)[i,j] = max_n min(R[i,n], S[n,j])

    Properties:
      - Reflexive:  R[i,i] = 1
      - Symmetric:  R[i,j] = R[j,i]  (max of min is symmetric)
      - Values always in [0, 1] without extra normalization
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_fuzzy_sets: int = 3,
        static_adjacency: torch.Tensor | None = None,
        input_dim: int | None = None,
    ):
        super().__init__()
        if num_fuzzy_sets < 2:
            raise ValueError("num_fuzzy_sets must be >= 2.")
        self.num_nodes = num_nodes
        self.num_fuzzy_sets = num_fuzzy_sets

        # ── Base node membership vectors μ ∈ [0,1]^K ──
        self.base_memberships = nn.Parameter(
            torch.zeros(num_nodes, num_fuzzy_sets)
        )
        nn.init.trunc_normal_(self.base_memberships, std=0.05)

        # ── Raw → hidden projection (for pre-encoder feature conditioning) ──
        if input_dim is not None and input_dim != hidden_dim:
            self.raw_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.raw_projection = None

        # ── Feature-conditioned membership modulation ──
        self.feature_to_membership = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_fuzzy_sets),
        )

        # ── Fuzzy set prototypes (for FCM regularization) ──
        self.fuzzy_prototypes = nn.Parameter(
            torch.randn(num_fuzzy_sets, hidden_dim) * 0.02
        )

        # ── Static adjacency (optional, for fuzzy union) ──
        if static_adjacency is not None:
            static_adj = static_adjacency.to(dtype=torch.float32)
            static_adj = torch.relu(static_adj) + torch.eye(num_nodes, device=static_adj.device)
            static_adj = static_adj / static_adj.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("static_adjacency", static_adj)
            self._has_static = True
        else:
            self.register_buffer("static_adjacency", torch.eye(num_nodes))
            self._has_static = False

        # Blend between fuzzy relation and static adjacency
        # init: sigmoid(0.5) ≈ 0.62 weight on fuzzy relation
        self.blend_logit = nn.Parameter(torch.tensor(0.5))

    # ── Membership computation ──────────────────────────────────

    def _compute_memberships(self, node_features=None):
        """μ = 0.7·μ_base + 0.3·μ_feat (sigmoid → [0,1]^K)."""
        mu_base = torch.sigmoid(self.base_memberships)  # [N, K]

        if node_features is not None:
            if node_features.dim() == 4:   # [B, T, N, D]
                node_feat = node_features.mean(dim=(0, 1))   # → [N, D]
            elif node_features.dim() == 3:  # [B, N, D]
                node_feat = node_features.mean(dim=0)         # → [N, D]
            else:
                node_feat = node_features
            # Project if feature dim != hidden_dim (e.g. raw 3-dim input)
            if self.raw_projection is not None:
                node_feat = self.raw_projection(node_feat)
            mu_feat = torch.sigmoid(self.feature_to_membership(node_feat))
            mu = 0.7 * mu_base + 0.3 * mu_feat
        else:
            mu = mu_base

        return mu  # [N, K]

    # ── Relation construction ───────────────────────────────────

    def _build_fuzzy_relation(self, memberships):
        """R[i,j] = max_k min(μ_k(i), μ_k(j)), with enforced reflexivity."""
        mu_i = memberships.unsqueeze(1)  # [N, 1, K]
        mu_j = memberships.unsqueeze(0)  # [1, N, K]
        R = torch.max(torch.min(mu_i, mu_j), dim=-1).values  # [N, N]

        # Enforce reflexivity: R[i,i] = 1
        diag = torch.eye(self.num_nodes, device=R.device, dtype=R.dtype)
        R = R + diag * (1.0 - R.diag().unsqueeze(-1))
        return R.clamp(0.0, 1.0)

    # ── Forward ─────────────────────────────────────────────────

    def forward(self, node_features=None):
        """Build fuzzy relational graph R ∈ [0,1]^(N×N).

        Args:
            node_features: Optional [B, T, N, D] / [B, N, D] for conditioning.
        Returns:
            R: [N, N] fuzzy similarity relation.
        """
        memberships = self._compute_memberships(node_features)
        R_fuzzy = self._build_fuzzy_relation(memberships)

        if self._has_static:
            blend = torch.sigmoid(self.blend_logit)
            R_static = self.static_adjacency.to(
                device=R_fuzzy.device, dtype=R_fuzzy.dtype
            )
            # Fuzzy union via max (not linear interpolation)
            R_union = torch.max(R_fuzzy * blend, R_static * (1.0 - blend))
            # Re-enforce reflexivity after union
            diag = torch.eye(self.num_nodes, device=R_union.device, dtype=R_union.dtype)
            R_union = R_union + diag * (1.0 - R_union.diag().unsqueeze(-1))
            return R_union.clamp(0.0, 1.0)

        return R_fuzzy

    # ── Accessors ───────────────────────────────────────────────

    def get_memberships(self):
        """Return learned membership vectors (for interpretability vis)."""
        return torch.sigmoid(self.base_memberships)  # [N, K]

    def get_prototypes(self):
        """Return fuzzy set prototypes (for FCM regularization)."""
        return self.fuzzy_prototypes  # [K, D]

    # ── Stability Diagnostics (路线 A) ─────────────────────────────

    def get_cell_entropy(self):
        """模糊胞熵: H(i) = -Σ_k μ_k(i) log μ_k(i)。

        基于 base_memberships（全局可学习参数，不依赖输入特征）。

        Returns:
            [N] 熵值。高值 → 节点位于模糊 Voronoi 边界 (交通相变边界候选)。
        """
        mu = torch.sigmoid(self.base_memberships)  # [N, K] → [0,1]
        mu = mu / mu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(mu * mu.log()).sum(dim=-1)  # [N]

    def get_margin_stability(self):
        """归属稳定度: S(i) = μ_{(1)}(i) - μ_{(2)}(i)。

        基于 base_memberships（全局可学习参数）。

        Returns:
            [N] 稳定度。低值 → 最大和第二大模糊集归属几乎相等 → 归属易翻转。
        """
        mu = torch.sigmoid(self.base_memberships)  # [N, K]
        top2 = mu.topk(2, dim=-1).values  # [N, 2]
        return top2[:, 0] - top2[:, 1]    # [N]


# ═══════════════════════════════════════════════════════════════════
#  Adaptive Graph Learner  (kept for reference)
# ═══════════════════════════════════════════════════════════════════

class AdaptiveGraphLearner(nn.Module):
    """(DEPRECATED in new_fuzzy_2) Legacy adaptive graph learner.

    Use ``FuzzyRelationalGraphLearner`` instead.
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim,
        static_adjacency,
        embed_dim=32,
        top_k=None,
        blend_init=0.5,
        fuzzy_enabled=False,
        fuzzy_num_sets=3,
        fuzzy_sigma_init=0.7,
    ):
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1.")
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.fuzzy_enabled = fuzzy_enabled

        static_adjacency = static_adjacency.to(dtype=torch.float32)
        if static_adjacency.dim() != 2 or static_adjacency.shape[0] != static_adjacency.shape[1]:
            raise ValueError("static_adjacency must be a square 2D tensor.")
        identity = torch.eye(static_adjacency.size(0), device=static_adjacency.device)
        static_adjacency = torch.relu(static_adjacency) + identity
        static_adjacency = self._row_normalize(static_adjacency)
        self.register_buffer("static_adjacency", static_adjacency)
        self.register_buffer("identity", identity)

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.feature_projection = nn.Linear(hidden_dim, embed_dim)
        self.time_projection = nn.Linear(hidden_dim, embed_dim)
        blend_init = float(min(max(blend_init, 1e-4), 1 - 1e-4))
        self.blend_logit = nn.Parameter(
            torch.tensor(math.log(blend_init / (1.0 - blend_init)), dtype=torch.float32)
        )

        if self.fuzzy_enabled:
            if fuzzy_num_sets < 2:
                raise ValueError("fuzzy_num_sets must be >= 2 when fuzzy graph is enabled.")
            if fuzzy_sigma_init <= 0:
                raise ValueError("fuzzy_sigma_init must be > 0 when fuzzy graph is enabled.")
            fuzzy_centers = torch.linspace(-1.0, 1.0, fuzzy_num_sets, dtype=torch.float32)
            self.fuzzy_centers = nn.Parameter(fuzzy_centers)
            self.fuzzy_log_sigmas = nn.Parameter(
                torch.full((fuzzy_num_sets,), math.log(fuzzy_sigma_init), dtype=torch.float32)
            )
            self.fuzzy_rule_logits = nn.Parameter(torch.zeros(fuzzy_num_sets, dtype=torch.float32))

    @staticmethod
    def _row_normalize(adjacency_matrix):
        return adjacency_matrix / adjacency_matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _fuzzy_similarity_to_adjacency(self, similarity):
        bounded_similarity = torch.tanh(similarity).unsqueeze(-1)
        centers = self.fuzzy_centers.view(1, 1, 1, -1).type_as(similarity)
        sigmas = self.fuzzy_log_sigmas.exp().view(1, 1, 1, -1).type_as(similarity)
        gaussian_membership = torch.exp(
            -0.5 * ((bounded_similarity - centers) / sigmas.clamp_min(1e-4)) ** 2
        )
        rule_weights = torch.softmax(self.fuzzy_rule_logits, dim=0).view(1, 1, 1, -1).type_as(similarity)
        fuzzy_relation = (gaussian_membership * rule_weights).sum(dim=-1)
        return fuzzy_relation.clamp_min(1e-12)

    def forward(self, sequence_features, timestep_embedding=None):
        if sequence_features.dim() == 4:
            node_features = sequence_features.mean(dim=1)
        elif sequence_features.dim() == 3:
            node_features = sequence_features
        else:
            raise ValueError("sequence_features must be 3D or 4D tensor.")

        batch_size = node_features.size(0)
        node_representation = (
            self.feature_projection(node_features) + self.node_embeddings.unsqueeze(0)
        )
        if timestep_embedding is not None:
            node_representation = node_representation + self.time_projection(
                timestep_embedding
            ).unsqueeze(1)
        node_representation = torch.tanh(node_representation)

        similarity = torch.matmul(node_representation, node_representation.transpose(1, 2))
        similarity = similarity / math.sqrt(self.embed_dim)
        if self.fuzzy_enabled:
            dynamic_adjacency = self._row_normalize(
                self._fuzzy_similarity_to_adjacency(similarity)
            )
        else:
            dynamic_adjacency = torch.softmax(similarity, dim=-1)

        if self.top_k is not None and 0 < self.top_k < self.num_nodes:
            top_values, top_indices = torch.topk(dynamic_adjacency, k=self.top_k, dim=-1)
            sparse_dynamic = torch.zeros_like(dynamic_adjacency)
            sparse_dynamic.scatter_(-1, top_indices, top_values)
            dynamic_adjacency = self._row_normalize(sparse_dynamic)

        dynamic_adjacency = (
            dynamic_adjacency + self.identity.to(dynamic_adjacency.dtype).unsqueeze(0)
        )
        dynamic_adjacency = self._row_normalize(dynamic_adjacency)

        static_adjacency = self.static_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        static_adjacency = static_adjacency.to(dtype=dynamic_adjacency.dtype)

        blend = torch.sigmoid(self.blend_logit)
        adaptive_adjacency = (1.0 - blend) * static_adjacency + blend * dynamic_adjacency
        return self._row_normalize(adaptive_adjacency)
