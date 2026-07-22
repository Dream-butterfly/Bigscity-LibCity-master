"""Graph components for final_T2 — multi-view fuzzy relational graph learner.

Design notes:
- Three structurally independent fuzzy perspectives (Low/Mid/High), each with its own
  prototype geometry, node representation, Gaussian width, and temperature.
- The upper/lower envelope (max/min over three views) quantifies cross-view disagreement
  rather than classical IT2 parametric uncertainty.
- The mid (expected) graph is built from mu_mid = (upper+lower)/2 and any closure
  is applied only on this mid graph.
- The Membership Disagreement Interval (MDI, code variable 'fou') is intended solely
  for gating/attention; it is NOT used in relation propagation (avoids structure collapse).
"""

import math
import os
import time
import warnings

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
    def __init__(self, hidden_dim, k_hop=2, topk=None, relation_mode="inner"):
        super().__init__()
        self.k_hop = k_hop
        self.topk = topk
        self.relation_mode = relation_mode
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
    def precompute_powers(R_base, k_hop, topk=None, relation_mode="maxmin"):
        device = R_base.device
        dtype = R_base.dtype
        N = R_base.size(-1)  # works for both (N,N) and (B,N,N)
        I = torch.eye(N, device=device, dtype=dtype)
        if R_base.dim() == 3:
            I = I.unsqueeze(0).expand(R_base.size(0), -1, -1)  # (B,N,N)
        powers = [I]
        current = R_base
        for _ in range(k_hop):
            powers.append(current)
            if relation_mode == "inner":
                if current.dim() == 3:
                    current = torch.bmm(current, R_base)
                else:
                    current = torch.mm(current, R_base)
            else:
                current = FuzzyGraphConvolution._max_min_compose_2d(R_base, current, topk=topk)
        return powers

    def forward(self, node_features, fuzzy_relation, powers=None):
        batch_size, num_nodes, _ = node_features.shape

        if powers is not None:
            R_powers = [p.to(device=node_features.device, dtype=node_features.dtype)
                        for p in powers]
        else:
            if fuzzy_relation.dim() == 3:
                R_base = fuzzy_relation.to(
                    device=node_features.device, dtype=node_features.dtype)
            else:
                R_base = fuzzy_relation.to(
                    device=node_features.device, dtype=node_features.dtype)
            R_powers = self.precompute_powers(R_base, self.k_hop,
                                                topk=self.topk, relation_mode=self.relation_mode)

        output = self.projections[0](node_features)
        graph_contrib = torch.zeros_like(output)
        for hop_index in range(1, self.k_hop + 1):
            R_k = R_powers[hop_index]
            # Match batch dim: R may be (B,N,N), node_features may be (B*T,N,D)
            if R_k.dim() == 3:
                T = node_features.size(0) // R_k.size(0)
                if T > 1:
                    R_k = R_k.repeat_interleave(T, dim=0)
            # Row normalization: each row sums to 1 for stable message passing
            R_k = R_k / (R_k.sum(dim=-1, keepdim=True).clamp_min(1e-8))
            propagated = R_k @ node_features
            contrib = self.projections[hop_index](propagated)
            output = output + contrib
            graph_contrib = graph_contrib + contrib

        self._current_graph_energy = (graph_contrib.detach().norm() /
                                      (node_features.detach().norm() + 1e-8))
        return output


class FuzzyRelationalGraphLearner(nn.Module):
    """Multi-View Fuzzy Relational Graph Learner (MV-FRGL).

    Constructs three structurally independent fuzzy perspectives (Low / Mid / High)
    on the sensor network, each with:
      - independent prototype centers C_v
      - independent node representations z_v via view-specific Δ-networks
      - independent Gaussian widths (ordered: σ_L ≤ σ_M ≤ σ_H)
      - independent temperature τ_v

    The three views produce three T1 membership matrices μ_L, μ_M, μ_H.
    The upper envelope μ⁺ = max(μ_L,μ_M,μ_H) and lower envelope μ⁻ = min(·)
    form a membership interval. The midpoint μ̄ = (μ⁺+μ⁻)/2 is used for
    relation construction; the width δ = mean(μ⁺−μ⁻) is the Membership
    Disagreement Interval (MDI) — a learned spatial uncertainty signal.

    Inspired by Interval Type-2 Fuzzy Sets (Mendel 2007), but this is a
    multi-view fuzzy system where uncertainty emerges from cross-view
    structural disagreement rather than from parametric uncertainty within
    a single MF.

    Degeneration:
        Δ_v→0, C_v→C, σ_L→σ_H, τ_v→τ  ⇒  three views identical
        ⇒  μ⁺=μ⁻  ⇒  δ(n)=0  ⇒  envelope collapses to single-view T1
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
            relation_mode: str = "maxmin",
    ):
        super().__init__()
        if num_fuzzy_sets < 2:
            raise ValueError("num_fuzzy_sets must be >= 2.")
        self.num_nodes = num_nodes
        self.relation_mode = relation_mode
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
        #   Init directly on unit hypersphere (not randn*0.1 → normalize).
        self.prototype_center_low  = nn.Parameter(
            F.normalize(torch.randn(num_fuzzy_sets, hidden_dim, generator=_g), dim=-1))
        self.prototype_center_mid  = nn.Parameter(
            F.normalize(torch.randn(num_fuzzy_sets, hidden_dim, generator=_g), dim=-1))
        self.prototype_center_high = nn.Parameter(
            F.normalize(torch.randn(num_fuzzy_sets, hidden_dim, generator=_g), dim=-1))
        # Backward-compat alias
        self.prototype_center = self.prototype_center_mid

        # ── Independent dual widths (view granularity control) ─
        #   Each fuzzy set k has its own base width and width delta,
        #   producing three ordered widths σ_L ≤ σ_M ≤ σ_H.
        #   Cross-assigned to views: Low→σ_H (coarse), High→σ_L (fine).
        _sigma_low_init   = torch.rand(num_fuzzy_sets, generator=_g) * 1.0 + 0.5   # U(0.5,1.5)
        _sigma_delta_init = torch.rand(num_fuzzy_sets, generator=_g) * 0.3 + 0.2   # U(0.2,0.5)
        self.log_sigma_low = nn.Parameter(
            torch.log(torch.exp(_sigma_low_init) - 1))
        self.log_sigma_delta = nn.Parameter(
            torch.log(torch.exp(_sigma_delta_init) - 1))

        # Per-view learnable temperature: sharpens/flattens each view's
        # membership independently → different R even with inner product.
        # τ init: low=sharp(0.3), mid=normal(0.8), high=flat(1.5)
        # Different sharpness → different R even with inner product.
        # τ init: low=sharp(0.2), mid=normal(0.8), high=flat(1.5)
        # 6x range → immediate R differentiation even with inner product.
        self.log_tau_low  = nn.Parameter(torch.tensor(-2.0))
        self.log_tau_mid  = nn.Parameter(torch.tensor(0.0))
        self.log_tau_high = nn.Parameter(torch.tensor(1.5))

        # ── Input projection ──────────────────────────────────────
        if input_dim is not None and input_dim != hidden_dim:
            self.raw_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.raw_projection = None

        # ── Feature transforms (independent per view) ──────────────
        #   Each view has its own 2-layer MLP + LayerNorm. Independent
        #   gradients prevent view collapse observed with shared backbone.
        self.transform_low = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transform_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transform_high = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Backward-compat alias
        self.node_transform = self.transform_mid

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
        # Per-node β parameters: each node learns its own view preference
        self.node_beta_logits = nn.Parameter(torch.zeros(num_nodes, 3))

    # ═══════════════════════════════════════════════════════════════
    #  Multi-View Membership Computation
    # ═══════════════════════════════════════════════════════════════

    def _compute_memberships(self, node_features):
        """Compute multi-view Gaussian T1 memberships with independent geometry per view.

        Three views → three prototype sets → three distance fields → three
        structurally different membership matrices (not just scaled copies).

        Cross-assigned widths: Low view uses σ_H (coarse), High view uses σ_L (fine),
        maximizing structural differentiation.

        Returns:
            mu_lower_raw:  [N, K]  Low-view  T1 membership (coarse, wide Gaussian)
            mu_mid_raw:    [N, K]  Mid-view  T1 membership (reference)
            mu_high_raw:   [N, K]  High-view T1 membership (fine, narrow Gaussian)
            mu_upper:      [N, K]  upper envelope  μ⁺ = max over three views
            mu_lower:      [N, K]  lower envelope  μ⁻ = min over three views
            mu_expected:   [N, K]  expected μ̄ = (μ⁺+μ⁻)/2  (NOT mid-view!)
        """
        # 1. Node representation from latest traffic state + short-term trend
        if node_features is not None:
            if node_features.dim() == 4:  # [B, T, N, D]
                latest = node_features[:, -1, :, :]         # [B, N, D] — latest traffic
                trend = latest - node_features[:, 0, :, :]  # [B, N, D] — T-step change
                node_repr = latest + trend                   # [B, N, D]
            elif node_features.dim() == 3:  # [B, N, D]
                node_repr = node_features  # keep batch
            else:
                node_repr = node_features
        else:
            raise ValueError(
                "node_features is required for prototype-based membership")

        # 2. Independent transforms per view (no shared bottleneck)
        if self.raw_projection is not None:
            node_repr = self.raw_projection(node_repr)  # → [B, N, D]
        z_low  = self.transform_low(node_repr)   # [B, N, D]
        z_mid  = self.transform_mid(node_repr)
        z_high = self.transform_high(node_repr)
        self._z_mid = z_mid  # backward-compat alias

        # ── Per-view hyperspherical projection ──
        z_low_norm   = F.normalize(z_low, dim=-1)
        z_mid_norm   = F.normalize(z_mid, dim=-1)
        z_high_norm  = F.normalize(z_high, dim=-1)
        # Cache for z-diversity loss (keep grad)
        self._z_low_norm  = z_low_norm
        self._z_mid_norm  = z_mid_norm
        self._z_high_norm = z_high_norm
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
        d2_low  = torch.cdist(z_low_norm,  proto_low_norm).pow(2)   # [N, K]
        d2_mid  = torch.cdist(z_mid_norm,  proto_mid_norm).pow(2)   # [N, K]
        d2_high = torch.cdist(z_high_norm, proto_high_norm).pow(2)  # [N, K]

        # Per-view temperature: τ ≠ 1 → different membership sharpness per view
        tau_low  = F.softplus(self.log_tau_low) + 0.1   # τ ∈ (0.1, ∞), init=1
        tau_mid  = F.softplus(self.log_tau_mid) + 0.1
        tau_high = F.softplus(self.log_tau_high) + 0.1

        # Gaussian membership on each view's own geometry × own temperature
        mu_low_raw  = torch.exp(-d2_low  / (2 * sigma_high.pow(2) * tau_low))
        mu_mid_raw  = torch.exp(-d2_mid  / (2 * sigma_mid.pow(2)  * tau_mid))
        mu_high_raw = torch.exp(-d2_high / (2 * sigma_low.pow(2)  * tau_high))

        # Multi-view envelope: upper/lower across three independent views
        mu_upper = torch.maximum(torch.maximum(mu_low_raw, mu_mid_raw), mu_high_raw)
        mu_lower = torch.minimum(torch.minimum(mu_low_raw, mu_mid_raw), mu_high_raw)
        mu_expected = (mu_lower + mu_upper) / 2  # expected membership μ̄ (not mid-view)

        # Cache diagnostics (detached — no gradient)
        _d2_avg = d2_mid.detach()  # use mid as representative
        self._current_d2_mean = _d2_avg.mean()
        self._current_d2_std  = _d2_avg.std()
        # Proto norm: average across all three sets
        _p_low  = self.prototype_center_low.norm(dim=-1).mean().detach()
        _p_mid  = self.prototype_center_mid.norm(dim=-1).mean().detach()
        _p_high = self.prototype_center_high.norm(dim=-1).mean().detach()
        self._current_proto_norm = ((_p_low + _p_mid + _p_high) / 3)
        self._current_latent_norm = z_mid.detach().norm(dim=-1).mean()
        self._current_transform_weight = self.node_transform[-1].weight.norm().detach()
        # Latent norm reg uses batch-mean for stable gradient
        self._node_latent_for_reg = z_mid.mean(dim=0) if z_mid.dim() == 3 else z_mid
        # Per-step movement tracking (track mid prototype as representative)
        _pc = self.prototype_center_mid.detach()
        _nl = z_mid.detach().mean(dim=0) if z_mid.dim() == 3 else z_mid.detach()
        if hasattr(self, '_prev_proto'):
            self._current_proto_update = (_pc - self._prev_proto).norm()
            self._current_latent_update = (_nl.mean(dim=0) - self._prev_latent_mean).norm()
        else:
            self._current_proto_update = torch.tensor(0.0)
            self._current_latent_update = torch.tensor(0.0)
        self._prev_proto = _pc.clone()
        self._prev_latent_mean = _nl.mean(dim=0).clone()
        _pc_mean = self.prototype_center_mid.mean(dim=0)
        _nl_mean = z_mid.detach().mean(dim=0)
        self._current_center_dist = (_nl_mean - _pc_mean).norm()
        _diff = (mu_upper - mu_lower).detach()
        self._current_mu_diff_mean = _diff.mean()
        self._current_mu_diff_max = _diff.max()
        self._current_mu_raw_low_mean  = mu_low_raw.detach().mean()
        self._current_mu_raw_mid_mean  = mu_mid_raw.detach().mean()
        self._current_mu_raw_high_mean = mu_high_raw.detach().mean()
        self._current_sigma_delta_mean = sigma_delta.detach().mean()
        self._current_tau_low  = tau_low.detach().mean()
        self._current_tau_mid  = tau_mid.detach().mean()
        self._current_tau_high = tau_high.detach().mean()

        # ── Adapter ratio: Δ contribution vs shared latent ──
        _s_norm = z_mid.detach().norm(dim=-1).mean()
        _d_low  = (z_low.detach() - z_mid.detach()).norm(dim=-1).mean()
        _d_mid  = (z_mid.detach() - z_low.detach()).norm(dim=-1).mean()   # z_mid vs z_low divergence
        _d_high = (z_high.detach() - z_mid.detach()).norm(dim=-1).mean()
        self._current_adapter_ratio = ((_d_low + _d_mid + _d_high) / 3) / (_s_norm + 1e-8)

        # ── View distance: pairwise ||z_v - z_w|| ──
        self._current_view_dist_lm = (z_low.detach() - z_mid.detach()).norm(dim=-1).mean()
        self._current_view_dist_lh = (z_low.detach() - z_high.detach()).norm(dim=-1).mean()
        self._current_view_dist_mh = (z_mid.detach() - z_high.detach()).norm(dim=-1).mean()

        return mu_lower, mu_upper, mu_expected, mu_low_raw, mu_mid_raw, mu_high_raw

    # ═══════════════════════════════════════════════════════════════
    #  Relation Sparsification
    # ═══════════════════════════════════════════════════════════════

    def _sparsify_relation(self, R, topk=None):
        """Top-K sparsification per node: keep K strongest edges, zero rest.

        Uses Straight-Through Estimator: forward = sparse, backward grad
        flows through full dense R, allowing zeroed-out edges to recover.
        """
        k = topk if topk is not None else self.graph_sparsify_topk
        if k is None or k <= 0:
            return R
        # Handle 3D (B,N,N): iterate batch
        if R.dim() == 3:
            return torch.stack([self._sparsify_relation(r, topk=k) for r in R])
        N = R.size(0)
        K = min(k, N)
        R_nodiag = R.clone()
        R_nodiag[range(N), range(N)] = 0
        _, idx = R_nodiag.topk(K, dim=-1)
        mask = torch.zeros_like(R)
        mask.scatter_(-1, idx, 1.0)
        mask[range(N), range(N)] = 1.0  # always keep self-loops
        sparse = R * mask
        return (sparse - R).detach() + R

    # ═══════════════════════════════════════════════════════════════
    #  Fuzzy Relation Construction
    # ═══════════════════════════════════════════════════════════════

    def _build_fuzzy_relation(self, memberships):
        # memberships: [N, K] or [B, N, K]
        if self.relation_mode == "inner":
            K = memberships.size(-1)
            R = (memberships @ memberships.transpose(-2, -1)) / K  # [*B, N, N]
        else:  # "maxmin"
            if memberships.dim() == 3:
                mu_i = memberships.unsqueeze(2)  # [B, N, 1, K]
                mu_j = memberships.unsqueeze(1)  # [B, 1, N, K]
            else:
                mu_i = memberships.unsqueeze(1)  # [N, 1, K]
                mu_j = memberships.unsqueeze(0)  # [1, N, K]
            R = torch.max(torch.min(mu_i, mu_j), dim=-1).values
        N = memberships.size(-2)
        I = torch.eye(N, device=R.device, dtype=R.dtype)
        # Broadcast self-loop across batch if needed
        if R.dim() == 3:
            I = I.unsqueeze(0)
            R = R + I * (1.0 - R.diagonal(dim1=-2, dim2=-1).unsqueeze(-1))
        else:
            R = R + I * (1.0 - R.diag().unsqueeze(-1))
        return R.clamp(0.0, 1.0)

    def _apply_mid_closure(self, R_mid):
        if self.closure_steps <= 0:
            return R_mid
        if R_mid.dim() == 3:
            warnings.warn(
                f"closure_steps={self.closure_steps} > 0 but R is batched (dim=3); "
                f"closure is only supported for 2D (single sample). Skipping."
            )
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
        """Build multi-view fuzzy graph with per-node disagreement interval.

        Returns:
            R_with_closure: [N, N] effective fuzzy relation
            fou_node:       [N]    per-node MDI scalar δ(n)
            mu_expected:    [B,N,K] expected memberships μ̄ (envelope midpoint)
            mu_low_raw:     [B,N,K] Low-view T1 memberships
            mu_mid_raw:     [B,N,K] Mid-view T1 memberships
            mu_high_raw:    [B,N,K] High-view T1 memberships
            beta:           [3]     global view mixing coefficients
            beta_node:      [N,3]   per-node view preferences
        """
        mu_lower, mu_upper, mu_expected, mu_low_raw, mu_mid_raw, mu_high_raw = \
            self._compute_memberships(node_features)

        # Per-node MDI δ(n) (mean over batch if per-sample)
        if mu_upper.dim() == 3:
            fou_node = (mu_upper - mu_lower).clamp(min=0.0).mean(dim=-1).mean(dim=0)  # [N]
        else:
            fou_node = (mu_upper - mu_lower).clamp(min=0.0).mean(dim=-1)  # [N]

        # ── Per-sample relations → per-view sparsification → batch-mean → (N,N) ──
        R_low  = self._build_fuzzy_relation(mu_low_raw)
        R_mid  = self._build_fuzzy_relation(mu_mid_raw)
        R_high = self._build_fuzzy_relation(mu_high_raw)
        # Per-view top-K: different sparsity → structurally different graphs
        R_low  = self._sparsify_relation(R_low,  topk=getattr(self, 't2_topk_low', None))
        R_mid  = self._sparsify_relation(R_mid,  topk=getattr(self, 't2_topk_mid', None))
        R_high = self._sparsify_relation(R_high, topk=getattr(self, 't2_topk_high', None))
        # Diagnostics on batch-mean R (keep per-sample R for GCN)
        if R_low.dim() == 3:
            R_low_diag  = R_low.mean(dim=0)
            R_mid_diag  = R_mid.mean(dim=0)
            R_high_diag = R_high.mean(dim=0)
        else:
            R_low_diag, R_mid_diag, R_high_diag = R_low, R_mid, R_high

        # Cache relation diffs for diagnostics
        self._current_R_diff_lm = (R_low_diag - R_mid_diag).abs().mean().detach()
        self._current_R_diff_hm = (R_high_diag - R_mid_diag).abs().mean().detach()
        _R_mid_norm = R_mid_diag.abs().mean().clamp_min(1e-8)
        self._current_R_gap = (R_high_diag - R_low_diag).abs().mean().detach() / _R_mid_norm

        # Cache relation correlations (manual Pearson for numerical stability)
        def _pearson(x, y):
            xc = x - x.mean()
            yc = y - y.mean()
            denom = (xc.norm() * yc.norm()).clamp_min(1e-12)
            return (xc @ yc) / denom
        _rl = R_low_diag.flatten().detach()
        _rm = R_mid_diag.flatten().detach()
        _rh = R_high_diag.flatten().detach()
        self._current_R_corr_lm = _pearson(_rl, _rm)
        self._current_R_corr_lh = _pearson(_rl, _rh)
        self._current_R_corr_mh = _pearson(_rm, _rh)

        self._current_eff_width = (mu_upper - mu_lower).abs().mean().detach()

        # Learnable interval mix — global β for R_mixed
        beta = F.softmax(self.relation_mix_logits, dim=0)  # [3]
        # Per-node β for decoder routing: each node learns its own view preference
        beta_node = F.softmax(self.node_beta_logits, dim=-1)  # (N,3) per-node
        R_mixed = (beta[0] * R_low + beta[1] * R_mid + beta[2] * R_high)

        # Static adjacency blend
        R_final = R_mixed
        if self._has_static:
            blend = torch.sigmoid(self.blend_logit)
            R_static = self.static_adjacency.to(device=R_final.device, dtype=R_final.dtype)
            R_final = torch.max(R_final * blend, R_static * (1.0 - blend))
            diag = torch.eye(self.num_nodes, device=R_final.device, dtype=R_final.dtype)
            if R_final.dim() == 3:
                diag = diag.unsqueeze(0)
                R_final = R_final + diag * (1.0 - R_final.diagonal(dim1=-2, dim2=-1).unsqueeze(-1))
            else:
                R_final = R_final + diag * (1.0 - R_final.diag().unsqueeze(-1))

        # Closure (if configured)
        R_with_closure = self._apply_mid_closure(R_final)

        return R_with_closure, fou_node, mu_expected, mu_low_raw, mu_mid_raw, mu_high_raw, beta, beta_node

    # ── Backward-compatible interface ─────────────────────────────

    def forward(self, node_features):
        R, *_ = self.get_type2_info(node_features)
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
