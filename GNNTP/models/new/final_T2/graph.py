"""Graph components for final_T2 with a light-weight Type-2 interface.

Design notes:
- We compute membership logits (theta) and expose a Type-2 FOU per node
  (mean over fuzzy sets). The mid (expectation) graph is built from
  mu = sigmoid(theta) and any closure is applied only on this mid graph.
- FOU is intended solely for gating/attention; it is NOT used in
  relation propagation (avoids structure collapse).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_normalized_adjacency, expand_adjacency_batch


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, k_hop=2):
        super().__init__()
        self.k_hop = k_hop
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)]
        )

    def forward(self, node_features, adjacency_matrix, **kwargs):
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


class FuzzyGraphConvolution(nn.Module):
    def __init__(self, hidden_dim, k_hop=2, topk=None):
        super().__init__()
        self.k_hop = k_hop
        self.topk = topk
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)]
        )

    @staticmethod
    def _max_min_compose_2d(R, S, topk=None):
        if topk is not None and topk < R.size(0) - 1:
            return FuzzyGraphConvolution._sparse_max_min_compose(R, S, topk)
        return torch.max(
            torch.min(R.unsqueeze(1), S.unsqueeze(0)), dim=-1
        ).values

    @staticmethod
    def _sparse_max_min_compose(R, S, topk):
        """Top-K sparse max-min compose: O(N^2*topk) instead of O(N^3)."""
        _, idx = R.topk(topk, dim=-1)            # [N, topk]
        R_topk = R.gather(1, idx)                 # [N, topk]
        S_topk = S[idx]                           # [N, topk, N]
        return torch.max(
            torch.min(R_topk.unsqueeze(-1), S_topk),
            dim=1
        ).values  # [N, N]

    @staticmethod
    def precompute_powers(R_base, k_hop, topk=None):
        device = R_base.device
        dtype = R_base.dtype
        N = R_base.size(0)
        I = torch.eye(N, device=device, dtype=dtype)
        powers = [I]
        current = R_base
        for _ in range(k_hop):
            powers.append(current)
            current = FuzzyGraphConvolution._max_min_compose_2d(R_base, current, topk=topk)
        return powers

    def forward(self, node_features, fuzzy_relation, powers=None):
        batch_size, num_nodes, _ = node_features.shape

        if powers is not None:
            R_powers = [p.to(device=node_features.device, dtype=node_features.dtype)
                        for p in powers]
        else:
            if fuzzy_relation.dim() == 3:
                R_base = fuzzy_relation[0].to(
                    device=node_features.device, dtype=node_features.dtype
                )
            else:
                R_base = fuzzy_relation.to(
                    device=node_features.device, dtype=node_features.dtype
                )
            R_powers = self.precompute_powers(R_base, self.k_hop, topk=self.topk)

        output = self.projections[0](node_features)
        for hop_index in range(1, self.k_hop + 1):
            R_k = R_powers[hop_index]
            x_flat = node_features.permute(1, 0, 2).reshape(num_nodes, -1)
            propagated_flat = torch.mm(R_k, x_flat)
            propagated = propagated_flat.reshape(num_nodes, batch_size, -1).permute(1, 0, 2)
            output = output + self.projections[hop_index](propagated)

        return output


class FuzzyRelationalGraphLearner(nn.Module):
    """Interval Type-2 Fuzzy Relational Graph Learner.

    Uses Gaussian IT2 membership functions with uncertain standard deviation:
        μ̃_k(x) = exp(−||x−c_k||² / 2σ̃_k²),   σ̃_k ∈ [σ_k·(1−r_k), σ_k·(1+r_k)]

    Upper membership (narrower, optimistic): σ_low  = σ·(1−r)
    Lower membership (wider,   pessimistic): σ_high = σ·(1+r)

    Degeneration theorem:
        r_k → 0  ⇒  μ_lower → μ_upper  ⇒  Type-2 collapses to Type-1
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_fuzzy_sets: int = 3,
        static_adjacency: torch.Tensor | None = None,
        input_dim: int | None = None,
        closure_steps: int = 0,
        topk: int | None = None,
    ):
        super().__init__()
        if num_fuzzy_sets < 2:
            raise ValueError("num_fuzzy_sets must be >= 2.")
        self.num_nodes = num_nodes
        self.num_fuzzy_sets = num_fuzzy_sets
        self.closure_steps = int(max(0, closure_steps))
        self.topk = topk

        # ── Prototype centers c_k ∈ R^D ──────────────────────────
        self.prototype_center = nn.Parameter(
            torch.randn(num_fuzzy_sets, hidden_dim) * 0.1)

        # ── Interval width: σ_k (base), r_k (ratio) ──────────────
        #   σ_low  = σ·(1−r)  → narrower  → optimistic (upper MF)
        #   σ_high = σ·(1+r)  → wider     → pessimistic (lower MF)
        self.log_sigma = nn.Parameter(torch.full((num_fuzzy_sets,), 3.5))
        self.log_radius_ratio = nn.Parameter(
            torch.zeros(num_fuzzy_sets))  # sigmoid(0)=0.5，明显区间宽度

        # ── Input projection ──────────────────────────────────────
        if input_dim is not None and input_dim != hidden_dim:
            self.raw_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.raw_projection = None

        # ── Feature transform (maps node repr to prototype space) ─
        self.node_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Static adjacency ──────────────────────────────────────
        if static_adjacency is not None:
            static_adj = static_adjacency.to(dtype=torch.float32)
            static_adj = torch.relu(static_adj) + torch.eye(num_nodes, device=static_adj.device)
            static_adj = static_adj / static_adj.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("static_adjacency", static_adj)
            self._has_static = True
        else:
            self.register_buffer("static_adjacency", torch.eye(num_nodes))
            self._has_static = False

        self.blend_logit = nn.Parameter(torch.tensor(0.5))

        # Interval relation mixing: β = softmax(logits)
        self.relation_mix_logits = nn.Parameter(torch.zeros(3))

    # ═══════════════════════════════════════════════════════════════
    #  Interval Type-2 Membership Computation
    # ═══════════════════════════════════════════════════════════════

    def _compute_memberships(self, node_features):
        """Compute IT2 Gaussian memberships [μ_lower, μ_upper, μ_mid].

        Returns:
            mu_lower:  [N, K]  pessimistic (wider Gaussian)
            mu_upper:  [N, K]  optimistic  (narrower Gaussian)
            mu_mid:    [N, K]  midpoint
        """
        # 1. Node representation extraction
        if node_features is not None:
            if node_features.dim() == 4:          # [B, T, N, D]
                node_repr = node_features.mean(dim=(0, 1))
            elif node_features.dim() == 3:        # [B, N, D]
                node_repr = node_features.mean(dim=0)
            else:
                node_repr = node_features
        else:
            raise ValueError(
                "node_features is required for prototype-based membership")

        # 2. Project to hidden space + transform for prototype matching
        if self.raw_projection is not None:
            node_repr = self.raw_projection(node_repr)    # → [N, D]
        node_latent = self.node_transform(node_repr)      # → [N, D]

        # 3. Interval width parameterization
        sigma = F.softplus(self.log_sigma) + 1e-3           # [K], base width
        r = torch.sigmoid(self.log_radius_ratio)             # [K], ratio ∈ (0,1)
        sigma_low  = sigma * (1 - r)                         # narrower  → optimistic
        sigma_high = sigma * (1 + r)                         # wider     → pessimistic
        sigma_mid  = sigma

        # 4. Gaussian membership: exp(−d² / 2σ²)
        #    Narrow Gaussian: high near center, fast decay
        #    Wide Gaussian:   low near center, slow decay
        #    → the two curves cross. Upper/lower envelopes use max/min.
        d2 = torch.cdist(node_latent, self.prototype_center).pow(2)  # [N, K]
        mu_narrow = torch.exp(-d2 / (2 * sigma_low.pow(2)))
        mu_wide   = torch.exp(-d2 / (2 * sigma_high.pow(2)))
        mu_upper  = torch.maximum(mu_narrow, mu_wide)
        mu_lower  = torch.minimum(mu_narrow, mu_wide)
        mu_mid    = torch.exp(-d2 / (2 * sigma_mid.pow(2)))

        return mu_lower, mu_upper, mu_mid

    # ═══════════════════════════════════════════════════════════════
    #  Fuzzy Relation Construction
    # ═══════════════════════════════════════════════════════════════

    def _build_fuzzy_relation(self, memberships):
        mu_i = memberships.unsqueeze(1)  # [N, 1, K]
        mu_j = memberships.unsqueeze(0)  # [1, N, K]
        R = torch.max(torch.min(mu_i, mu_j), dim=-1).values  # [N, N]
        diag = torch.eye(self.num_nodes, device=R.device, dtype=R.dtype)
        R = R + diag * (1.0 - R.diag().unsqueeze(-1))
        return R.clamp(0.0, 1.0)

    def _apply_mid_closure(self, R_mid):
        if self.closure_steps <= 0:
            return R_mid
        R_current = R_mid
        R_list = [R_current]
        for _ in range(self.closure_steps):
            R_current = FuzzyGraphConvolution._max_min_compose_2d(R_mid, R_current, topk=self.topk)
            R_list.append(R_current)
        S = torch.stack(R_list, dim=0).amax(dim=0)
        diag = torch.eye(self.num_nodes, device=S.device, dtype=S.dtype)
        S = S + diag * (1.0 - S.diag().unsqueeze(-1))
        return S.clamp(0.0, 1.0)

    # ═══════════════════════════════════════════════════════════════
    #  Graph Construction
    # ═══════════════════════════════════════════════════════════════

    def get_type2_info(self, node_features):
        """Build interval-valued fuzzy graph with per-node FOU.

        Returns:
            R_with_closure: [N, N] effective fuzzy relation
            fou_node:       [N]    per-node FOU scalar
        """
        mu_lower, mu_upper, mu_mid = self._compute_memberships(node_features)

        # Per-node FOU (for CellAttention, backward compat)
        fou_node = (mu_upper - mu_lower).clamp(min=0.0).mean(dim=-1)  # [N]

        # ── Interval-valued relations ──
        R_low  = self._build_fuzzy_relation(mu_lower)     # pessimistic
        R_mid  = self._build_fuzzy_relation(mu_mid)       # midpoint
        R_high = self._build_fuzzy_relation(mu_upper)     # optimistic

        # Learnable interval mix
        beta = F.softmax(self.relation_mix_logits, dim=0)  # [3]
        R_mixed = (beta[0] * R_low + beta[1] * R_mid + beta[2] * R_high)

        # Static adjacency blend
        R_final = R_mixed
        if self._has_static:
            blend = torch.sigmoid(self.blend_logit)
            R_static = self.static_adjacency.to(device=R_final.device, dtype=R_final.dtype)
            R_final = torch.max(R_final * blend, R_static * (1.0 - blend))
            diag = torch.eye(self.num_nodes, device=R_final.device, dtype=R_final.dtype)
            R_final = R_final + diag * (1.0 - R_final.diag().unsqueeze(-1))

        # Closure (if configured)
        R_with_closure = self._apply_mid_closure(R_final)

        return R_with_closure, fou_node

    # ── Backward-compatible interface ─────────────────────────────

    def forward(self, node_features):
        R, _ = self.get_type2_info(node_features)
        return R

    def get_memberships(self, node_features=None):
        """Return midpoint memberships for backward compatibility."""
        if node_features is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.num_nodes,
                                    self.prototype_center.size(1),
                                    device=self.prototype_center.device)
            _, _, mu_mid = self._compute_memberships(dummy)
        else:
            _, _, mu_mid = self._compute_memberships(node_features)
        return mu_mid

    def get_prototypes(self):
        return self.prototype_center

    def get_cell_entropy(self, node_features=None):
        mu = self.get_memberships(node_features)
        mu = mu / mu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(mu * mu.log()).sum(dim=-1)

    def get_margin_stability(self, node_features=None):
        mu = self.get_memberships(node_features)
        top2 = mu.topk(2, dim=-1).values
        return top2[:, 0] - top2[:, 1]

