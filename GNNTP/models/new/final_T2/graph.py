"""Graph components for final_T2 with a light-weight Type-2 interface.

Design notes:
- We compute membership logits (theta) and expose a Type-2 FOU per node
  (mean over fuzzy sets). The mid (expectation) graph is built from
  mu = sigmoid(theta) and any closure is applied only on this mid graph.
- FOU is intended solely for gating/attention; it is NOT used in
  relation propagation (avoids structure collapse).
"""

import math
import os
import time

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
        _, idx = R.topk(topk, dim=-1)  # [N, topk]
        R_topk = R.gather(1, idx)  # [N, topk]
        S_topk = S[idx]  # [N, topk, N]
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
            beta_init_random: bool = False,
    ):
        super().__init__()
        if num_fuzzy_sets < 2:
            raise ValueError("num_fuzzy_sets must be >= 2.")
        self.num_nodes = num_nodes
        self.num_fuzzy_sets = num_fuzzy_sets
        self.closure_steps = int(max(0, closure_steps))
        self.topk = topk
        self.graph_sparsify_topk = None  # set via model config if needed

        # ── Random init with time+pid seed (different per run) ─────
        _seed = int((time.time() * 1e6) % (2 ** 31)) ^ (os.getpid() % (2 ** 16))
        _g = torch.Generator()
        _g.manual_seed(_seed)

        # ── Independent prototype centers per view ─────────────────
        #   Three views → three geometry templates → genuinely different relations
        self.prototype_center_low  = nn.Parameter(
            torch.randn(num_fuzzy_sets, hidden_dim, generator=_g) * 0.1)
        self.prototype_center_mid  = nn.Parameter(
            torch.randn(num_fuzzy_sets, hidden_dim, generator=_g) * 0.1)
        self.prototype_center_high = nn.Parameter(
            torch.randn(num_fuzzy_sets, hidden_dim, generator=_g) * 0.1)
        # Backward-compat alias
        self.prototype_center = self.prototype_center_mid

        # ── Independent dual widths: σ_low, σ_high (genuine Type-2) ─
        #   Each fuzzy set k has its own lower and upper Gaussian width,
        #   independently learned from data (not a symmetric perturbation).
        #   Ordering enforced: σ_low ≤ σ_high at compute time.
        _sigma_low_init   = torch.rand(num_fuzzy_sets, generator=_g) * 1.0 + 0.5   # U(0.5,1.5)
        _sigma_delta_init = torch.rand(num_fuzzy_sets, generator=_g) * 0.3 + 0.2   # U(0.2,0.5)
        self.log_sigma_low = nn.Parameter(
            torch.log(torch.exp(_sigma_low_init) - 1))
        self.log_sigma_delta = nn.Parameter(
            torch.log(torch.exp(_sigma_delta_init) - 1))

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
        # Temperature for membership sharpness annealing
        self.register_buffer('tau', torch.tensor(3.0))  # updated by model each step

        # Interval relation mixing: β = softmax(logits)
        self.relation_mix_logits = nn.Parameter(
            torch.randn(3, generator=_g) * 1.0 if beta_init_random
            else torch.zeros(3))

    # ═══════════════════════════════════════════════════════════════
    #  Interval Type-2 Membership Computation
    # ═══════════════════════════════════════════════════════════════

    def _compute_memberships(self, node_features):
        """Compute IT2 Gaussian memberships with independent prototype geometry per view.

        Three views → three prototype sets → three distance fields → three
        structurally different relation matrices (not just scaled copies).

        Returns:
            mu_lower_raw:  [N, K]  pessimistic (wider Gaussian, proto_low)
            mu_mid_raw:    [N, K]  midpoint     (mid Gaussian,  proto_mid)
            mu_high_raw:   [N, K]  optimistic   (narrow Gaussian, proto_high)
            mu_upper:      [N, K]  max over three raw
            mu_lower:      [N, K]  min over three raw
            mu_mid:        [N, K]  (upper+lower)/2
        """
        # 1. Node representation extraction
        if node_features is not None:
            if node_features.dim() == 4:  # [B, T, N, D]
                node_repr = node_features.mean(dim=(0, 1))
            elif node_features.dim() == 3:  # [B, N, D]
                node_repr = node_features.mean(dim=0)
            else:
                node_repr = node_features
        else:
            raise ValueError(
                "node_features is required for prototype-based membership")

        # 2. Project to hidden space + transform for prototype matching
        if self.raw_projection is not None:
            node_repr = self.raw_projection(node_repr)  # → [N, D]
        node_latent = self.node_transform(node_repr)  # → [N, D]

        # ── Hyperspherical projection: normalize to unit sphere ──
        node_latent_norm = F.normalize(node_latent, dim=-1)          # [N, D], ||·||=1
        proto_low_norm  = F.normalize(self.prototype_center_low, dim=-1)   # [K, D]
        proto_mid_norm  = F.normalize(self.prototype_center_mid, dim=-1)   # [K, D]
        proto_high_norm = F.normalize(self.prototype_center_high, dim=-1)  # [K, D]

        # 3. Three sigma levels (single delta parameterization, three views)
        sigma_low   = F.softplus(self.log_sigma_low) + 1e-3   # [K], narrowest
        sigma_delta = F.softplus(self.log_sigma_delta) + 1e-3 # [K], δ ≥ 0
        sigma_mid   = sigma_low + sigma_delta * 0.5           # [K], middle
        sigma_high  = sigma_low + sigma_delta                  # [K], widest
        # Cache for diagnostics
        self._current_sigma_low = sigma_low.detach()
        self._current_sigma_mid = sigma_mid.detach()
        self._current_sigma_high = sigma_high.detach()

        # 4. Three independent distance fields → genuinely different geometries
        d2_low  = torch.cdist(node_latent_norm, proto_low_norm).pow(2)   # [N, K]
        d2_mid  = torch.cdist(node_latent_norm, proto_mid_norm).pow(2)   # [N, K]
        d2_high = torch.cdist(node_latent_norm, proto_high_norm).pow(2)  # [N, K]

        # Gaussian membership on each view's own geometry
        mu_low_raw  = torch.exp(-d2_low  / (2 * sigma_high.pow(2)))   # wide  σ → pessimistic
        mu_mid_raw  = torch.exp(-d2_mid  / (2 * sigma_mid.pow(2)))    # mid   σ → midpoint
        mu_high_raw = torch.exp(-d2_high / (2 * sigma_low.pow(2)))    # narrow σ → optimistic

        # Type-2 envelope: upper/lower across three independent views
        mu_upper = torch.maximum(torch.maximum(mu_low_raw, mu_mid_raw), mu_high_raw)
        mu_lower = torch.minimum(torch.minimum(mu_low_raw, mu_mid_raw), mu_high_raw)
        mu_mid   = (mu_lower + mu_upper) / 2

        # Cache diagnostics (detached — no gradient)
        _d2_avg = d2_mid.detach()  # use mid as representative
        self._current_d2_mean = _d2_avg.mean()
        self._current_d2_std  = _d2_avg.std()
        # Proto norm: average across all three sets
        _p_low  = self.prototype_center_low.norm(dim=-1).mean().detach()
        _p_mid  = self.prototype_center_mid.norm(dim=-1).mean().detach()
        _p_high = self.prototype_center_high.norm(dim=-1).mean().detach()
        self._current_proto_norm = ((_p_low + _p_mid + _p_high) / 3)
        self._current_latent_norm = node_latent.detach().norm(dim=-1).mean()
        self._current_transform_weight = self.node_transform[-1].weight.norm().detach()
        self._node_latent_for_reg = node_latent  # keep grad: for latent norm regularization
        # Per-step movement tracking (track mid prototype as representative)
        _pc = self.prototype_center_mid.detach()
        _nl = node_latent.detach()
        if hasattr(self, '_prev_proto'):
            self._current_proto_update = (_pc - self._prev_proto).norm()
            self._current_latent_update = (_nl.mean(dim=0) - self._prev_latent_mean).norm()
        else:
            self._current_proto_update = torch.tensor(0.0)
            self._current_latent_update = torch.tensor(0.0)
        self._prev_proto = _pc.clone()
        self._prev_latent_mean = _nl.mean(dim=0).clone()
        _pc_mean = self.prototype_center_mid.mean(dim=0)
        _nl_mean = node_latent.detach().mean(dim=0)
        self._current_center_dist = (_nl_mean - _pc_mean).norm()
        _diff = (mu_upper - mu_lower).detach()
        self._current_mu_diff_mean = _diff.mean()
        self._current_mu_diff_max = _diff.max()
        self._current_mu_raw_low_mean  = mu_low_raw.detach().mean()
        self._current_mu_raw_mid_mean  = mu_mid_raw.detach().mean()
        self._current_mu_raw_high_mean = mu_high_raw.detach().mean()
        self._current_sigma_delta_mean = sigma_delta.detach().mean()

        return mu_lower, mu_upper, mu_mid, mu_low_raw, mu_mid_raw, mu_high_raw

    # ═══════════════════════════════════════════════════════════════
    #  Relation Sparsification
    # ═══════════════════════════════════════════════════════════════

    def _sparsify_relation(self, R):
        """Top-K sparsification per node: keep K strongest edges, zero rest.

        Uses Straight-Through Estimator: forward = sparse, backward grad
        flows through full dense R, allowing zeroed-out edges to recover.
        """
        if self.graph_sparsify_topk is None or self.graph_sparsify_topk <= 0:
            return R
        N = R.size(0)
        K = min(self.graph_sparsify_topk, N)
        # Keep self-loops + top-K (excluding diag)
        R_nodiag = R.clone()
        R_nodiag[range(N), range(N)] = 0
        _, idx = R_nodiag.topk(K, dim=-1)
        mask = torch.zeros_like(R)
        mask.scatter_(-1, idx, 1.0)
        mask[range(N), range(N)] = 1.0  # always keep self-loops
        # STE: forward = R ⊙ mask, backward gradients pass through full R
        sparse = R * mask
        return (sparse - R).detach() + R

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
        mu_lower, mu_upper, mu_mid, mu_low_raw, mu_mid_raw, mu_high_raw = \
            self._compute_memberships(node_features)

        # Per-node FOU (for CellAttention, backward compat)
        fou_node = (mu_upper - mu_lower).clamp(min=0.0).mean(dim=-1)  # [N]

        # ── Interval-valued relations — each from its OWN geometry ──
        R_low  = self._build_fuzzy_relation(mu_low_raw)   # proto_low  + σ_high (pessimistic)
        R_mid  = self._build_fuzzy_relation(mu_mid_raw)   # proto_mid  + σ_mid  (midpoint)
        R_high = self._build_fuzzy_relation(mu_high_raw)  # proto_high + σ_low  (optimistic)

        # Per-relation top-K sparsification (independent per view)
        R_low  = self._sparsify_relation(R_low)
        R_mid  = self._sparsify_relation(R_mid)
        R_high = self._sparsify_relation(R_high)

        # Cache relation diffs for diagnostics
        self._current_R_diff_lm = (R_low - R_mid).abs().mean().detach()
        self._current_R_diff_hm = (R_high - R_mid).abs().mean().detach()
        # R_gap: relative interval size — directly measures Type-2 collapse
        _R_mid_norm = R_mid.abs().mean().clamp_min(1e-8)
        self._current_R_gap = (R_high - R_low).abs().mean().detach() / _R_mid_norm

        # Cache relation correlations — are the three views structurally different?
        R_flat = lambda R: R.flatten().detach()
        _rl, _rm, _rh = R_flat(R_low), R_flat(R_mid), R_flat(R_high)
        self._current_R_corr_lm = torch.corrcoef(torch.stack([_rl, _rm]))[0, 1]
        self._current_R_corr_lh = torch.corrcoef(torch.stack([_rl, _rh]))[0, 1]
        self._current_R_corr_mh = torch.corrcoef(torch.stack([_rm, _rh]))[0, 1]

        # Cache effective Type-2 width (membership interval)
        self._current_eff_width = (mu_upper - mu_lower).abs().mean().detach()

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

        return R_with_closure, fou_node, mu_mid

    # ── Backward-compatible interface ─────────────────────────────

    def forward(self, node_features):
        R, _, _ = self.get_type2_info(node_features)
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
