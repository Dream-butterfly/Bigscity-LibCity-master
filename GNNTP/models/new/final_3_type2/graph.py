"""Graph convolution and fuzzy relational graph learning components.

Phase B — Interval Type-2 Fuzzy Relational Graph (final_3_type2):
  ``FuzzyRelationalGraphLearner`` learns interval membership
    μ(i) = [μ_low(i), μ_high(i)] ∈ [0,1]^K  per node,
  producing an interval fuzzy relation
    [R_low, R_high]  with Footprint of Uncertainty (FOU):
    FOU_R = R_high − R_low ∈ [0,1].

  This is the key contribution of final_3_type2 over final_2:
    Type-1 (final_2):  μ ∈ [0,1]           → R ∈ [0,1]
    Type-2 (final_3):  [μ_low, μ_high]     → [R_low, R_high] + FOU

  ``FuzzyGraphConvolution`` does K-hop propagation using the effective
  graph (midpoint, upper-bound, or FOU-gated) × feature matmul.

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
#  Fuzzy Graph Convolution  (unchanged from final_2)
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
        output = self.projections[0](node_features)
        for hop_index in range(1, self.k_hop + 1):
            R_k = R_powers[hop_index]  # [N, N] — never expanded to batch
            x_flat = node_features.permute(1, 0, 2).reshape(num_nodes, -1)
            propagated_flat = torch.mm(R_k, x_flat)
            propagated = propagated_flat.reshape(num_nodes, batch_size, -1).permute(1, 0, 2)
            output = output + self.projections[hop_index](propagated)

        return output


# ═══════════════════════════════════════════════════════════════════
#  Interval Type-2 Fuzzy Relational Graph Learner  (Phase B)
# ═══════════════════════════════════════════════════════════════════

class FuzzyRelationalGraphLearner(nn.Module):
    """Interval Type-2 Fuzzy Relational Graph Learner.

    Mathematical construction (Type-2 extension):

    1. Node i has interval membership over K fuzzy sets:
         μ(i,k) = [μ_low(i,k), μ_high(i,k)]  with 0 ≤ μ_low ≤ μ_high ≤ 1

       Parametrization:
         μ_low  = σ(θ_lower)           ∈ [0, 1]
         μ_delta = σ(θ_delta)          ∈ [0, 1]
         μ_high = μ_low + μ_delta·(1−μ_low)  ∈ [μ_low, 1]

       This guarantees the interval ordering and keeps FOU in [0,1].
       FOU_k(i) = μ_high(i,k) − μ_low(i,k)  is the Footprint of
       Uncertainty — epistemic uncertainty about node i's membership
       in fuzzy set k.

    2. Interval fuzzy relation via max-min composition:
         R_low[i,j]  = max_k min(μ_low_i[k], μ_low_j[k])
         R_high[i,j] = max_k min(μ_high_i[k], μ_high_j[k])

       By monotonicity of max-min: R_low ≤ R_high (elementwise).
       Structural FOU: FOU_R[i,j] = R_high[i,j] − R_low[i,j]

    3. Interval semantic closure (separate per bound):
         S_low  = max(R_low,  R_low²,  R_low³,  ...)
         S_high = max(R_high, R_high², R_high³, ...)

       By monotonicity: S_low ≤ S_high.
       Closure FOU: FOU_S = S_high − S_low — quantifies structural
       uncertainty in the inferred relational graph.

    4. Effective graph for GCN propagation:
         S_eff = combine(S_low, S_high, FOU_S) → [N, N]
       The combination mode controls how uncertainty affects propagation.

    Physical significance in traffic:
      - CBD core nodes:   tight interval (low FOU — clear functional role)
      - Transition zones:  wide interval (high FOU — ambiguous role,
                           e.g. residential area becoming commercial)
      - The FOU naturally captures zones undergoing functional transition,
        making the model robust to urban evolution.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_fuzzy_sets: int = 3,
        static_adjacency: torch.Tensor | None = None,
        input_dim: int | None = None,
        sparsification_epsilon: float = 0.0,
    ):
        super().__init__()
        if num_fuzzy_sets < 2:
            raise ValueError("num_fuzzy_sets must be >= 2.")
        self.num_nodes = num_nodes
        self.num_fuzzy_sets = num_fuzzy_sets
        self.sparsification_epsilon = sparsification_epsilon
        self.membership_temperature = 1.0  # annealed during training

        # ── Type-2 interval membership parameters ─────────────────
        # μ_low  = σ(θ_lower)           ∈ [0, 1]
        # μ_delta = σ(θ_delta)          ∈ [0, 1]
        # μ_high = μ_low + μ_delta·(1−μ_low)  ⇒ 0 ≤ μ_low ≤ μ_high ≤ 1
        self.base_membership_lower = nn.Parameter(
            torch.zeros(num_nodes, num_fuzzy_sets)
        )
        self.base_membership_delta = nn.Parameter(
            torch.zeros(num_nodes, num_fuzzy_sets)
        )
        # Large init std → sigmoid outputs spread across [0.05, 0.95].
        # With K=8 fuzzy sets, this gives 307 nodes enough initial diversity
        # across the 8-dimensional simplex before gradient kicks in.
        nn.init.trunc_normal_(self.base_membership_lower, std=1.2)
        nn.init.trunc_normal_(self.base_membership_delta, std=0.6)

        # ── Raw → hidden projection (for pre-encoder feature conditioning) ──
        if input_dim is not None and input_dim != hidden_dim:
            self.raw_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.raw_projection = None

        # ── Feature-conditioned membership modulation ──
        # Adjusts the center of the interval; preserves FOU proportion
        self.feature_to_membership = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_fuzzy_sets),
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

    # ═══════════════════════════════════════════════════════════════
    #  Interval Membership Computation
    # ═══════════════════════════════════════════════════════════════

    def _compute_memberships(self, node_features=None):
        """Compute interval membership [μ_low, μ_high] for all nodes.

        μ_low  = softmax(θ_lower / τ)            ∈ Δ^{K-1} (simplex)
        μ_delta = σ(θ_delta / τ)                 ∈ [0, 1]^K
        μ_high = μ_low + μ_delta·(1−μ_low)       ∈ [μ_low, 1]^K

        Softmax on μ_low enforces competition across the K fuzzy sets:
        if one set gains membership, another must lose it.  This directly
        attacks the "uniform spread" problem where sigmoid allows all K
        sets to activate independently at ~0.5.

        μ_delta stays sigmoid: interval width is a per-dimension
        independent property (FOU), not a zero-sum resource.

        Feature conditioning (when node_features is provided):
          Blends traffic-conditioned softmax μ_feat into the interval
          center while preserving the FOU half-width.

        Args:
            node_features: Optional [B,T,N,D] / [B,N,D] / [N,D].

        Returns:
            mu_low:  [N, K] lower membership bound (softmax, Σ_k ≈ 1).
            mu_high: [N, K] upper membership bound, ≥ mu_low elementwise.
            mu_mid:  [N, K] midpoint = (mu_low + mu_high) / 2.
        """
        tau = max(self.membership_temperature, 0.05)  # prevent div by zero
        mu_low = F.softmax(self.base_membership_lower / tau, dim=-1)   # [N, K]
        mu_delta = torch.sigmoid(self.base_membership_delta / tau)     # [N, K]
        mu_high = mu_low + mu_delta * (1.0 - mu_low)                  # [N, K]

        if node_features is not None:
            if node_features.dim() == 4:       # [B, T, N, D]
                node_feat = node_features.mean(dim=(0, 1))
            elif node_features.dim() == 3:     # [B, N, D]
                node_feat = node_features.mean(dim=0)
            else:
                node_feat = node_features

            if self.raw_projection is not None:
                node_feat = self.raw_projection(node_feat)

            mu_feat = F.softmax(self.feature_to_membership(node_feat) / tau, dim=-1)  # [N, K]

            # Blend feature into center with stronger feature modulation.
            # Higher feature weight (60%) ensures traffic-conditioned
            # membership provides meaningful node differentiation.
            center = (mu_low + mu_high) / 2.0
            half_width = (mu_high - mu_low) / 2.0
            center_blended = 0.8 * center + 0.2 * mu_feat
            mu_low = (center_blended - half_width).clamp(0.0, 1.0)
            mu_high = (center_blended + half_width).clamp(0.0, 1.0)

        mu_mid = (mu_low + mu_high) / 2.0
        return mu_low, mu_high, mu_mid

    def set_temperature(self, tau: float):
        """Update membership temperature for annealing.

        Lower τ → sigmoid outputs pushed toward 0/1 extremes,
        encouraging sharper fuzzy set membership.
        τ=1.0 is identity (default), τ→0 is near-hard assignment.
        """
        self.membership_temperature = max(tau, 0.05)

    # ═══════════════════════════════════════════════════════════════
    #  Interval Relation Construction
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _max_min_relation(memberships: torch.Tensor) -> torch.Tensor:
        """Compute R[i,j] = max_k min(μ_k(i), μ_k(j)) from [N, K].

        Standard max-min construction of fuzzy tolerance relation.
        """
        mu_i = memberships.unsqueeze(1)   # [N, 1, K]
        mu_j = memberships.unsqueeze(0)   # [1, N, K]
        return torch.max(torch.min(mu_i, mu_j), dim=-1).values  # [N, N]

    @staticmethod
    def _enforce_reflexivity(R: torch.Tensor) -> torch.Tensor:
        """Set R[i,i] = 1 and clamp to [0, 1]."""
        N = R.size(0)
        diag = torch.eye(N, device=R.device, dtype=R.dtype)
        R = R + diag * (1.0 - R.diag().unsqueeze(-1))
        return R.clamp(0.0, 1.0)

    def _build_fuzzy_relation_t2(self, mu_low, mu_high):
        """Build interval fuzzy relation from interval memberships.

        R_low[i,j]  = max_k min(μ_low_i[k],  μ_low_j[k])
        R_high[i,j] = max_k min(μ_high_i[k], μ_high_j[k])

        By monotonicity of max and min:
            R_low[i,j] ≤ R_high[i,j]  for all i,j.

        The structural Footprint of Uncertainty:
            FOU[i,j] = R_high[i,j] − R_low[i,j]

        quantifies how much the relation strength varies under
        epistemic uncertainty about node memberships.

        Returns:
            R_low:  [N, N] pessimistic relation.
            R_high: [N, N] optimistic relation, ≥ R_low elementwise.
        """
        R_low = self._max_min_relation(mu_low)
        R_high = self._max_min_relation(mu_high)
        R_low = self._enforce_reflexivity(R_low)
        R_high = self._enforce_reflexivity(R_high)
        return R_low, R_high

    # ═══════════════════════════════════════════════════════════════
    #  Sparsification
    # ═══════════════════════════════════════════════════════════════

    def _sparsify_relation(self, R: torch.Tensor) -> torch.Tensor:
        """ε-threshold sparsification: suppress noise-level fuzzy relations.

        Soft approach (preserves gradients):
            R' = ReLU(R − ε) / (1 − ε)

        Physical intuition: fuzzy tolerance relations should only be
        non-zero when nodes genuinely share functional features.

        Args:
            R: [N, N] dense fuzzy relation, values ∈ [0,1].

        Returns:
            [N, N] sparsified relation, values ∈ [0,1].
        """
        if self.sparsification_epsilon <= 0:
            return R
        eps = self.sparsification_epsilon
        R_sp = F.relu(R - eps) / (1.0 - eps)
        diag = torch.eye(self.num_nodes, device=R_sp.device, dtype=R_sp.dtype)
        R_sp = R_sp + diag * (1.0 - R_sp.diag().unsqueeze(-1))
        return R_sp.clamp(0.0, 1.0)

    def _sparsify_interval(self, R_low: torch.Tensor, R_high: torch.Tensor):
        """Sparsify both interval bounds.

        Ensures R_low ≤ R_high is preserved after sparsification
        (monotonicity of ReLU + rescaling guarantees this).
        """
        return self._sparsify_relation(R_low), self._sparsify_relation(R_high)

    # ═══════════════════════════════════════════════════════════════
    #  Static Adjacency Blend
    # ═══════════════════════════════════════════════════════════════

    def _blend_static_interval(self, R_low, R_high):
        """Blend static adjacency into both interval bounds via fuzzy union."""
        blend = torch.sigmoid(self.blend_logit)
        R_static = self.static_adjacency.to(device=R_low.device, dtype=R_low.dtype)

        # Fuzzy union via max for both bounds
        R_low_u = torch.max(R_low * blend, R_static * (1.0 - blend))
        R_high_u = torch.max(R_high * blend, R_static * (1.0 - blend))

        # Re-enforce reflexivity
        R_low_u = self._enforce_reflexivity(R_low_u)
        R_high_u = self._enforce_reflexivity(R_high_u)
        return R_low_u, R_high_u

    # ═══════════════════════════════════════════════════════════════
    #  Semantic Closure (Type-2)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_closure(R: torch.Tensor, max_hops: int) -> torch.Tensor:
        """Compute fuzzy semantic closure S = max(R, R², ..., R^K).

        Uses max-min composition: (R∘S)[i,j] = max_n min(R[i,n], S[n,j]).

        Args:
            R: [N, N] base fuzzy relation.
            max_hops: maximum transitive depth K.

        Returns:
            S: [N, N] closure, S[i,j] ≥ R[i,j] for all i,j.
        """
        S = R.clone()
        current = R
        for _ in range(max_hops - 1):
            current = torch.max(
                torch.min(R.unsqueeze(1), current.unsqueeze(0)), dim=-1
            ).values
            S = torch.max(S, current)
        return S.clamp(0.0, 1.0)

    def get_semantic_closure_t2(self, max_hops: int = 3, node_features=None):
        """Compute Type-2 interval semantic closure.

        Pipeline:
          1. Compute interval membership  [μ_low, μ_high]
          2. Build interval relation       [R_low, R_high]
          3. Sparsify both bounds          (noise suppression)
          4. Blend static adjacency
          5. Compute closure per bound     [S_low, S_high]
          6. Compute structural FOU        FOU = S_high − S_low

        By monotonicity of max-min composition:
            S_low[i,j] ≤ S_high[i,j]  for all i,j.

        The closure FOU quantifies structural uncertainty:
          - Low FOU →  relation is stable regardless of membership choice
          - High FOU → relation is uncertain, depends on membership ambiguity

        Args:
            max_hops: maximum transitive depth (K). Default 3.
            node_features: Optional [B,T,N,D] for dynamic membership conditioning.

        Returns:
            S_low:  [N, N] pessimistic closure.
            S_high: [N, N] optimistic closure.
            FOU:    [N, N] structural Footprint of Uncertainty.
        """
        mu_low, mu_high, _ = self._compute_memberships(node_features)
        R_low, R_high = self._build_fuzzy_relation_t2(mu_low, mu_high)

        # Sparsify noise
        if self.sparsification_epsilon > 0:
            R_low, R_high = self._sparsify_interval(R_low, R_high)

        # Blend static adjacency into both bounds
        if self._has_static:
            R_low, R_high = self._blend_static_interval(R_low, R_high)

        # Compute closures independently
        S_low = self._compute_closure(R_low, max_hops)
        S_high = self._compute_closure(R_high, max_hops)

        FOU = S_high - S_low  # guaranteed ≥ 0 by monotonicity
        return S_low, S_high, FOU

    # ═══════════════════════════════════════════════════════════════
    #  Effective Graph Combination
    # ═══════════════════════════════════════════════════════════════

    def get_effective_graph(
        self,
        max_hops: int = 3,
        mode: str = "mid",
        fou_gate_scale: float = 1.0,
        node_features=None,
    ):
        """Combine interval bounds into a single effective graph.

        Modes:
          "low"       → S_eff = S_low
                         Conservative: only well-established relations.
          "high"      → S_eff = S_high
                         Optimistic: all possible relations included.
          "mid"       → S_eff = (S_low + S_high) / 2
                         Balanced: midpoint of the uncertainty interval.
          "fou_gated" → S_eff = S_high / (1 + FOU·scale)
                         Uncertainty-gated: maximise connectivity but
                         dampen uncertain relations. Lower FOU → higher
                         effective weight (model is confident).

        Args:
            max_hops: transitive depth for semantic closure.
            mode: combination strategy (see above).
            fou_gate_scale: scaling factor for FOU gating (mode="fou_gated").
            node_features: Optional [B,T,N,D] for dynamic membership.

        Returns:
            S_eff: [N, N] effective graph for GCN propagation.
            FOU:   [N, N] structural Footprint of Uncertainty.
        """
        S_low, S_high, FOU = self.get_semantic_closure_t2(
            max_hops, node_features=node_features)

        if mode == "low":
            S_eff = S_low
        elif mode == "high":
            S_eff = S_high
        elif mode == "mid":
            S_eff = (S_low + S_high) / 2.0
        elif mode == "fou_gated":
            # High-uncertainty relations → dampened propagation
            confidence = 1.0 / (1.0 + FOU * fou_gate_scale)
            S_eff = (S_high * confidence).clamp(0.0, 1.0)
            # Re-enforce reflexivity after gating
            S_eff = self._enforce_reflexivity(S_eff)
        else:
            raise ValueError(
                f"Unknown type2_graph_mode: '{mode}'. "
                f"Expected one of: low, high, mid, fou_gated."
            )

        return S_eff, FOU

    # ═══════════════════════════════════════════════════════════════
    #  Forward (backward-compatible midpoint)
    # ═══════════════════════════════════════════════════════════════

    def forward(self, node_features=None):
        """Build midpoint fuzzy relation for backward compatibility.

        Returns the midpoint of the interval relation as a single [N,N]
        graph. For the full Type-2 pipeline, use get_effective_graph()
        or get_semantic_closure_t2().

        Args:
            node_features: Optional [B,T,N,D] / [B,N,D] for conditioning.

        Returns:
            R_mid: [N, N] midpoint fuzzy relation, values in [0, 1].
        """
        mu_low, mu_high, mu_mid = self._compute_memberships(node_features)
        R_fuzzy = self._max_min_relation(mu_mid)
        R_fuzzy = self._enforce_reflexivity(R_fuzzy)

        if self._has_static:
            blend = torch.sigmoid(self.blend_logit)
            R_static = self.static_adjacency.to(
                device=R_fuzzy.device, dtype=R_fuzzy.dtype
            )
            R_union = torch.max(R_fuzzy * blend, R_static * (1.0 - blend))
            R_union = self._enforce_reflexivity(R_union)
            return self._sparsify_relation(R_union)

        return self._sparsify_relation(R_fuzzy)

    # ═══════════════════════════════════════════════════════════════
    #  Accessors
    # ═══════════════════════════════════════════════════════════════

    def get_memberships(self, node_features=None):
        """Return midpoint memberships for FRR conditioning.

        Uses the midpoint μ_mid = (μ_low + μ_high) / 2 —
        the most representative single-valued membership.

        Args:
            node_features: Optional [B,T,N,D] for dynamic membership.
        """
        _, _, mu_mid = self._compute_memberships(node_features)
        return mu_mid  # [N, K]

    def get_interval_memberships(self):
        """Return full interval memberships [μ_low, μ_high].

        Returns:
            mu_low:  [N, K] lower membership.
            mu_high: [N, K] upper membership.
            FOU_k:   [N, K] per-set Footprint of Uncertainty.
        """
        mu_low, mu_high, _ = self._compute_memberships()
        return mu_low, mu_high, mu_high - mu_low

    # ═══════════════════════════════════════════════════════════════
    #  Stability Diagnostics (Type-2 aware)
    # ═══════════════════════════════════════════════════════════════

    def get_cell_entropy(self, node_features=None):
        """Type-2 fuzzy cell entropy.

        H(i) = H_mid(i) · (1 + FOU_avg(i))

        where:
          H_mid(i)  = −Σ_k p_mid(k|i) log p_mid(k|i)
          FOU_avg(i) = mean_k (μ_high(i,k) − μ_low(i,k))

        The FOU amplification term captures the additional uncertainty
        from the Type-2 interval structure — nodes with wide membership
        intervals receive higher entropy, reflecting genuine epistemic
        uncertainty about their functional zone.

        Args:
            node_features: Optional [B,T,N,D] for dynamic membership.

        Returns:
            [N] Type-2 entropy. High → boundary node with uncertain membership.
        """
        mu_low, mu_high, _ = self._compute_memberships(node_features)
        mu_mid = (mu_low + mu_high) / 2.0
        # Normalize to probability-like
        p = mu_mid / mu_mid.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        H_mid = -(p * p.log()).sum(dim=-1)                       # [N]
        # FOU amplification: wider interval → higher uncertainty
        FOU_avg = (mu_high - mu_low).mean(dim=-1)                # [N]
        return H_mid * (1.0 + FOU_avg)                           # [N]

    def get_margin_stability(self):
        """归属稳定度 (based on midpoint membership).

        S(i) = μ_mid_{(1)}(i) − μ_mid_{(2)}(i)

        Returns:
            [N] 稳定度。低值 → 归属易翻转。
        """
        _, _, mu_mid = self._compute_memberships()
        top2 = mu_mid.topk(2, dim=-1).values                     # [N, 2]
        return top2[:, 0] - top2[:, 1]                           # [N]

    def get_fou_stats(self):
        """Return aggregate FOU statistics for monitoring.

        Returns:
            fou_node: [N] average FOU across fuzzy sets per node.
            fou_global: scalar, mean FOU across all nodes and sets.
        """
        mu_low, mu_high, _ = self._compute_memberships()
        FOU = mu_high - mu_low                                  # [N, K]
        fou_node = FOU.mean(dim=-1)                              # [N]
        fou_global = FOU.mean()                                  # scalar
        return fou_node, fou_global

    # ═══════════════════════════════════════════════════════════════
    #  Semantic Closure (Type-1 backward-compatible, for reference)
    # ═══════════════════════════════════════════════════════════════

    def get_semantic_closure(self, max_hops: int = 3):
        """Compute midpoint semantic closure (backward-compatible).

        Uses the midpoint membership for the closure, providing a
        single [N,N] graph compatible with Type-1 interfaces.
        For the full Type-2 closure, use get_semantic_closure_t2().

        Args:
            max_hops: Maximum transitive depth (K). Default 3.

        Returns:
            S: [N, N] fuzzy semantic closure on midpoint.
        """
        R = self.forward()  # midpoint relation [N, N]
        S = R.clone()
        current = R
        for _ in range(max_hops - 1):
            current = torch.max(
                torch.min(R.unsqueeze(1), current.unsqueeze(0)), dim=-1
            ).values
            S = torch.max(S, current)
        return S.clamp(0.0, 1.0)
