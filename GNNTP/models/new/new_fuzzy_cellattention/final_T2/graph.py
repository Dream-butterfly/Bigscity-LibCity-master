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


class FuzzyGraphConvolution(nn.Module):
    def __init__(self, hidden_dim, k_hop=2):
        super().__init__()
        self.k_hop = k_hop
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)]
        )

    @staticmethod
    def _max_min_compose_2d(R, S):
        return torch.max(
            torch.min(R.unsqueeze(1), S.unsqueeze(0)), dim=-1
        ).values

    def forward(self, node_features, fuzzy_relation):
        batch_size, num_nodes, _ = node_features.shape

        if fuzzy_relation.dim() == 3:
            R_base = fuzzy_relation[0].to(
                device=node_features.device, dtype=node_features.dtype
            )
        else:
            R_base = fuzzy_relation.to(
                device=node_features.device, dtype=node_features.dtype
            )

        I = torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype)
        R_powers = [I]
        current = R_base
        for _ in range(self.k_hop):
            R_powers.append(current)
            current = self._max_min_compose_2d(R_base, current)

        output = self.projections[0](node_features)
        for hop_index in range(1, self.k_hop + 1):
            R_k = R_powers[hop_index]
            x_flat = node_features.permute(1, 0, 2).reshape(num_nodes, -1)
            propagated_flat = torch.mm(R_k, x_flat)
            propagated = propagated_flat.reshape(num_nodes, batch_size, -1).permute(1, 0, 2)
            output = output + self.projections[hop_index](propagated)

        return output


class FuzzyRelationalGraphLearner(nn.Module):
    """Learns fuzzy relation and exposes Type-2 FOU for gating.

    Forward keeps backward-compatible behavior (returns mid graph), and
    a separate method `get_type2_info` returns (R_mid, fou_vector).
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_fuzzy_sets: int = 3,
        static_adjacency: torch.Tensor | None = None,
        input_dim: int | None = None,
        closure_steps: int = 0,
    ):
        super().__init__()
        if num_fuzzy_sets < 2:
            raise ValueError("num_fuzzy_sets must be >= 2.")
        self.num_nodes = num_nodes
        self.num_fuzzy_sets = num_fuzzy_sets
        self.closure_steps = int(max(0, closure_steps))

        # raw logits θ (we will apply sigmoid when forming expectations)
        self.base_memberships = nn.Parameter(
            torch.zeros(num_nodes, num_fuzzy_sets)
        )
        nn.init.trunc_normal_(self.base_memberships, std=0.05)

        if input_dim is not None and input_dim != hidden_dim:
            self.raw_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.raw_projection = None

        self.feature_to_membership = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_fuzzy_sets),
        )

        self.fuzzy_prototypes = nn.Parameter(
            torch.randn(num_fuzzy_sets, hidden_dim) * 0.02
        )

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

        # Type-2 scale per fuzzy set (log space for positivity)
        self.log_epsilon = nn.Parameter(torch.log(torch.full((num_fuzzy_sets,), 0.08)))

    # ── Internal: membership logits (θ) ──────────────────────────

    def _compute_membership_logits(self, node_features=None):
        theta_base = self.base_memberships  # raw logits [N, K]
        if node_features is not None:
            if node_features.dim() == 4:   # [B, T, N, D]
                node_feat = node_features.mean(dim=(0, 1))   # → [N, D]
            elif node_features.dim() == 3:  # [B, N, D]
                node_feat = node_features.mean(dim=0)         # → [N, D]
            else:
                node_feat = node_features
            if self.raw_projection is not None:
                node_feat = self.raw_projection(node_feat)
            theta_feat = self.feature_to_membership(node_feat)  # [N, K]
            theta = theta_base + theta_feat
        else:
            theta = theta_base
        return theta

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
            R_current = FuzzyGraphConvolution._max_min_compose_2d(R_mid, R_current)
            R_list.append(R_current)
        S = torch.stack(R_list, dim=0).amax(dim=0)
        diag = torch.eye(self.num_nodes, device=S.device, dtype=S.dtype)
        S = S + diag * (1.0 - S.diag().unsqueeze(-1))
        return S.clamp(0.0, 1.0)

    # ── Public Type-2 info: mid graph + per-node FOU scalar ──────

    def get_type2_info(self, node_features=None):
        theta = self._compute_membership_logits(node_features)  # [N, K]
        eps = self.log_epsilon.exp().clamp_min(1e-6)  # [K]
        # Broadcast eps over nodes: [N, K]
        eps_node = eps.unsqueeze(0)

        mu_mid = torch.sigmoid(theta)
        mu_L = torch.sigmoid(theta - eps_node)
        mu_U = torch.sigmoid(theta + eps_node)

        fou_per_set = (mu_U - mu_L).clamp(min=0.0)  # [N, K]
        fou_node = fou_per_set.mean(dim=-1)  # aggregate → [N]

        R_mid = self._build_fuzzy_relation(mu_mid)
        R_final = R_mid
        if self._has_static:
            blend = torch.sigmoid(self.blend_logit)
            R_static = self.static_adjacency.to(device=R_mid.device, dtype=R_mid.dtype)
            R_final = torch.max(R_mid * blend, R_static * (1.0 - blend))
            diag = torch.eye(self.num_nodes, device=R_final.device, dtype=R_final.dtype)
            R_final = R_final + diag * (1.0 - R_final.diag().unsqueeze(-1))
        # apply closure only on mid/expectation graph (if configured)
        R_with_closure = self._apply_mid_closure(R_final)

        return R_with_closure, fou_node

    # Keep backward-compatible forward (returns mid graph only)
    def forward(self, node_features=None):
        R, _ = self.get_type2_info(node_features)
        return R

    def get_memberships(self):
        return torch.sigmoid(self.base_memberships)

    def get_prototypes(self):
        return self.fuzzy_prototypes

    def get_cell_entropy(self):
        mu = torch.sigmoid(self.base_memberships)
        mu = mu / mu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(mu * mu.log()).sum(dim=-1)

    def get_margin_stability(self):
        mu = torch.sigmoid(self.base_memberships)
        top2 = mu.topk(2, dim=-1).values
        return top2[:, 0] - top2[:, 1]

