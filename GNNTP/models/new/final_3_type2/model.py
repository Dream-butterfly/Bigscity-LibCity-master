"""final_3_type2 — Interval Type-2 Fuzzy Relational Reasoning System.

Enhanced from final_2 with Type-2 Fuzzy Sets (Route 2):
  Instead of single membership μ ∈ [0,1], each node has an interval
  membership [μ_low, μ_high]. This yields an interval relation
  [R_low, R_high] with Footprint of Uncertainty (FOU).

  This is the key conceptual upgrade:
    Type-1 (final_2):  "node belongs to CBD with μ=0.8"
    Type-2 (final_3):  "node belongs to CBD with μ∈[0.6, 0.9]"

  The FOU captures epistemic uncertainty — particularly meaningful
  for traffic where functional zones undergo transitions (congestion
  onset, dissipation, land-use change).

Full fuzzy reasoning chain (7-stage):
  μ_low, μ_high → sparsify(R_low,R_high) → S_low,S_high = closure →
  S_eff = combine(S_low,S_high,FOU) → S_dyn(S_eff,H_t2) → FHR → FIR

Carried forward from final_2:
  Route 1 — Fuzzy Semantic Closure (interval-aware)
  Route 3 — Entropy-Driven Dynamic Graph (Type-2 entropy + FOU dampening)
  Route 3b— Entropy-Weighted FIR (with FOU-modulated H_t2 weights)
  Route 4 — Fuzzy Hierarchical Routing
  Route 6 — Fuzzy Relation Sparsification

Local-Global Spatial Dual architecture:
  Spatial-Local:  FuzzyGCN (K-hop topology propagation via S_dyn)
  Spatial-Global: FHR (Fuzzy Hierarchical Routing)
  Temporal:       Per-node Transformer
"""

import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)

from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel

from .encoder import STEncoder
from .decoder import FutureDecoder
from .graph import FuzzyRelationalGraphLearner
from .utils import compute_hop_distance


class NewFuzzyCellAttention3_Type2(AbstractTrafficStateModel):
    """Interval Type-2 Fuzzy Relational Reasoning System.

    Core innovation over final_2: replaces Type-1 fuzzy membership μ∈[0,1]
    with interval Type-2 membership [μ_low, μ_high], producing an interval
    fuzzy relation with quantifiable Footprint of Uncertainty (FOU).

    The fuzzy relation forms a complete 7-stage reasoning pipeline:

      μ_low,μ_high → R_low,R_high → S_low,S_high=closure → S_eff+mode → S_dyn → FHR → FIR

    Graph combination modes for S_eff:
      - "mid":       S_eff = (S_low + S_high) / 2        (balanced)
      - "high":      S_eff = S_high                       (optimistic)
      - "low":       S_eff = S_low                        (conservative)
      - "fou_gated": S_eff = S_high / (1 + FOU·scale)    (uncertainty-gated)

    Architecture:
        H' = λ₁·FuzzyGCN(H, S_dyn) + λ₂·FHR(H, C)
        S_dyn = S_eff + α·EntropyModulation (Type-2 with FOU dampening)
        S_eff = combine(S_low, S_high, FOU, mode)
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self._scaler = data_feature.get("scaler")

        # ── Geometry ──────────────────────────────────────────
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)

        # ── Architecture ──────────────────────────────────────
        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_heads = config.get("num_heads", 2)
        self.encoder_layers = config.get("encoder_layers", 2)
        self.decoder_layers = config.get("decoder_layers", 2)
        self.ffn_hidden_dim = config.get("ffn_hidden_dim", 128)
        self.graph_k_hop = config.get("graph_k_hop", 2)
        self.dropout = config.get("dropout", 0.1)
        self.use_temporal_position_embedding = config.get(
            "use_temporal_position_embedding", True)
        self.use_gradient_checkpointing = config.get(
            "use_gradient_checkpointing", False)

        # ── FRR configuration ─────────────────────────────────
        self.fuzzy_num_sets = config.get("fuzzy_num_sets", 4)
        self.num_cells = config.get("num_cells", 8)
        self.cell_blend_init = config.get("cell_blend_init", 0.3)
        self.band_center_init = config.get("band_center_init", 1.1)
        self.band_width_init = config.get("band_width_init", 0.7)
        self.region_transformer_layers = config.get(
            "region_transformer_layers", 1)

        # ── FIR (Fuzzy Interaction Regularization) ────────────
        self.fir_mode = config.get("fir_mode", "lukasiewicz")
        self.conservation_loss_weight = config.get(
            "conservation_loss_weight", 0.1)
        self.conservation_warmup_epochs = int(max(
            0, config.get("conservation_warmup_epochs", 5)))
        self.conservation_steps_per_epoch = int(
            config.get("conservation_steps_per_epoch", 80))
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self._train_step_count = 0

        # ── Route 1: Fuzzy Semantic Closure ─────────────────────
        self.use_semantic_closure = config.get(
            "use_semantic_closure", True)
        self.semantic_closure_hops = config.get(
            "semantic_closure_hops", 3)

        # ── Route 2: Interval Type-2 Fuzzy Sets ────────────────
        self.use_type2_fuzzy = config.get(
            "use_type2_fuzzy", True)
        self.type2_graph_mode = config.get(
            "type2_graph_mode", "mid")  # low | high | mid | fou_gated
        self.type2_fou_gate_scale = config.get(
            "type2_fou_gate_scale", 1.0)

        # ── Route 3: Entropy-Driven Dynamic Graph ───────────────
        self.use_entropy_dynamic_graph = config.get(
            "use_entropy_dynamic_graph", True)
        self.entropy_scale = config.get("entropy_scale", 0.1)
        self.use_entropy_fir_weight = config.get(
            "use_entropy_fir_weight", True)
        # Type-2: FOU dampens entropy contribution for uncertain pairs
        self.use_fou_entropy_modulation = config.get(
            "use_fou_entropy_modulation", True)

        # ── Route 4: Fuzzy Hierarchical Routing ──────────────────
        self.use_fuzzy_routing = config.get(
            "use_fuzzy_routing", True)

        # ── Route 6: Fuzzy Relation Sparsification ───────────────
        self.use_fuzzy_sparsification = config.get(
            "use_fuzzy_sparsification", True)
        self.sparsification_epsilon = config.get(
            "sparsification_epsilon", 0.05)

        # ── FOU-Entropy Alignment (Type-2 regularisation) ──────
        self.fou_entropy_align_weight = config.get(
            "fou_entropy_align_weight", 0.0)

        # ── Device ────────────────────────────────────────────
        self.device = config.get("device", torch.device("cpu"))

        # ── Adjacency + Graph Distance ────────────────────────
        adjacency_matrix = data_feature.get(
            "adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        graph_dist = compute_hop_distance(adjacency_matrix)
        self.register_buffer("graph_dist", graph_dist)

        # ── Type-2 Fuzzy Relational Graph Learner ─────────────
        # Same interface as final_2, but internally learns interval
        # memberships [μ_low, μ_high] via dual parameter sets
        self.fuzzy_graph = FuzzyRelationalGraphLearner(
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            num_fuzzy_sets=self.fuzzy_num_sets,
            static_adjacency=adjacency_matrix,
            input_dim=self.feature_dim,
            sparsification_epsilon=(
                self.sparsification_epsilon if self.use_fuzzy_sparsification else 0.0),
        )

        # ── Submodules ────────────────────────────────────────
        block_kwargs = dict(
            use_fuzzy_graph=True,
            use_cell_attention=True,
            num_cells=self.num_cells,
            cell_blend_init=self.cell_blend_init,
            band_center_init=self.band_center_init,
            band_width_init=self.band_width_init,
            region_transformer_layers=self.region_transformer_layers,
            use_fuzzy_routing=self.use_fuzzy_routing,
        )
        self.condition_encoder = STEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.encoder_layers,
            ffn_hidden_dim=self.ffn_hidden_dim,
            graph_k_hop=self.graph_k_hop,
            dropout=self.dropout,
            use_temporal_position_embedding=self.use_temporal_position_embedding,
            max_time_steps=self.input_window,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            **block_kwargs,
        )
        self.future_decoder = FutureDecoder(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.decoder_layers,
            ffn_hidden_dim=self.ffn_hidden_dim,
            graph_k_hop=self.graph_k_hop,
            dropout=self.dropout,
            output_window=self.output_window,
            num_nodes=self.num_nodes,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            **block_kwargs,
        )

        # ── Connect FuzzyGraph μ (midpoint) → CellAttention conditioning ─
        self._connect_fuzzy_conditioning()

        # ── Cache FOU for uncertainty quantification ─
        self._current_fou: torch.Tensor | None = None

        # ── Type-2 FOU → log-var (NLL uncertainty head) ──
        self.fou_to_logvar = nn.Linear(self.fuzzy_num_sets, 1)

        # ── Init output_projection bias: log_var ≈ 0 (σ² ≈ 1) ─
        nn.init.zeros_(self.future_decoder.output_projection.bias)
        # offset bias for log_var half to start near log(1) = 0
        with torch.no_grad():
            b = self.future_decoder.output_projection.bias
            b[self.output_dim:] = 0.0  # already 0 from init

    def _connect_fuzzy_conditioning(self):
        """Wire fuzzy_to_cell Linear(K_f→K_c) to all CellAttention blocks."""
        for blocks in [self.condition_encoder.blocks, self.future_decoder.blocks]:
            for block in blocks:
                if hasattr(block, 'cell_attention'):
                    block.cell_attention.fuzzy_to_cell = nn.Linear(
                        self.fuzzy_num_sets, self.num_cells)

    # ═══════════════════════════════════════════════════════════
    #  Route 2: Type-2 Fuzzy Graph Construction
    # ═══════════════════════════════════════════════════════════

    def _build_type2_effective_graph(self, device=None, node_features=None):
        """Build effective graph from Type-2 interval relation.

        Pipeline:
          μ_low, μ_high → [R_low, R_high] → sparsify →
          [S_low, S_high, FOU] → S_eff = combine(mode)

        The combination mode controls how epistemic uncertainty (FOU)
        affects the propagation graph:
          - mid/high/low: static blend strategies
          - fou_gated: FOU inversely gates relation strength

        Args:
            device: target device for tensors.
            node_features: Optional [B,T,N,D] for dynamic membership.

        Returns:
            S_eff: [N, N] effective graph for GCN propagation.
            FOU:   [N, N] structural Footprint of Uncertainty.
        """
        if not self.use_type2_fuzzy:
            # Fallback to Type-1 midpoint closure
            fuzzy_relation = self.fuzzy_graph.get_semantic_closure(
                max_hops=self.semantic_closure_hops)
            return fuzzy_relation, None

        # Type-2 enabled: closures hops depend on semantic_closure switch.
        #   use_semantic_closure=True  → full transitive closure (default 3 hops)
        #   use_semantic_closure=False → no closure, S_eff from raw interval relation
        closure_hops = self.semantic_closure_hops if self.use_semantic_closure else 1
        S_eff, FOU = self.fuzzy_graph.get_effective_graph(
            max_hops=closure_hops,
            mode=self.type2_graph_mode,
            fou_gate_scale=self.type2_fou_gate_scale,
            node_features=node_features,
        )

        if device is not None:
            S_eff = S_eff.to(device)
            if FOU is not None:
                FOU = FOU.to(device)

        return S_eff, FOU

    def get_type2_fou(self) -> torch.Tensor | None:
        """Return the FOU from the last encode_condition call.

        FOU[i,j] quantifies structural uncertainty in the relation
        between nodes i and j. Can be visualised as a heatmap to
        show which urban connections are most uncertain.

        Returns:
            [N, N] FOU matrix, or None if Type-2 is disabled.
        """
        return self._current_fou

    # ═══════════════════════════════════════════════════════════
    #  Route 3: Entropy-Driven Dynamic Graph (Type-2 aware)
    # ═══════════════════════════════════════════════════════════

    def _apply_entropy_dynamic_graph(self, fuzzy_R, FOU=None, node_features=None):
        """Modulate fuzzy relation by Type-2 fuzzy entropy.

        H_t2(i) = H_mid(i) · (1 + FOU_avg(i))

        where H_mid is the standard membership entropy and FOU_avg
        amplifies it based on the interval width — nodes with wide
        membership intervals receive higher entropy.

        S_dyn[i,j] = min(1.0, R[i,j] + α·H_norm[i]·H_norm[j])

        Type-2 enhancement: when FOU is available, the outer product
        is dampened for structurally uncertain node pairs — if the
        relation itself is uncertain, the entropy boost should be
        conservative.

        Args:
            fuzzy_R: [N, N] base effective graph (S_eff).
            FOU:     [N, N] optional structural Footprint of Uncertainty.
            node_features: Optional [B,T,N,D] for dynamic membership.

        Returns:
            [N, N] entropy-modulated relation, values in [0, 1].
        """
        H = self.fuzzy_graph.get_cell_entropy(
            node_features=node_features).to(device=fuzzy_R.device,
                                            dtype=fuzzy_R.dtype)
        H_max = H.max().clamp_min(1e-8)
        H_norm = H / H_max                                       # [N], ∈ [0, 1]

        # Outer product: amplify relations between high-entropy node pairs
        H_outer = H_norm.unsqueeze(-1) * H_norm.unsqueeze(-2)    # [N, N]

        # Type-2: FOU dampens entropy contribution for uncertain pairs
        if FOU is not None and self.use_fou_entropy_modulation:
            FOU_norm = FOU / FOU.max().clamp_min(1e-8)           # [N, N]
            fou_damp = 1.0 / (1.0 + FOU_norm)                    # [N, N]
            H_outer = H_outer * fou_damp

        H_outer.fill_diagonal_(0.0)                               # no self-boost
        S_dyn = (fuzzy_R + self.entropy_scale * H_outer).clamp(0.0, 1.0)
        return S_dyn

    def _apply_entropy_weighted_fir(self, temporal_delta, net_pressure,
                                      node_features=None):
        """Weight FIR residual by Type-2 fuzzy entropy.

        Boundary nodes (high Type-2 entropy) receive higher FIR penalty.
        When FOU is available, the weight is modulated by structural
        uncertainty — nodes with wide membership intervals get
        proportionally adjusted penalty.

        Args:
            temporal_delta: [B, T-1, N] state change.
            net_pressure:   [B, T-1, N] FIR net pressure.
            node_features: Optional [B,T,N,D] for dynamic membership.

        Returns:
            scalar weighted MSE loss.
        """
        H = self.fuzzy_graph.get_cell_entropy(
            node_features=node_features).to(
            device=temporal_delta.device, dtype=temporal_delta.dtype)
        H_weight = 1.0 + H / H.max().clamp_min(1e-8)             # [N], ∈ [1, 2]

        # FOU modulation: structurally uncertain nodes get adjusted weight
        if self._current_fou is not None:
            fou_node = self._current_fou.mean(dim=-1).to(
                device=H_weight.device, dtype=H_weight.dtype)    # [N]
            H_weight = H_weight / (1.0 + fou_node * 0.5)          # higher FOU → reduce

        H_weight = H_weight.unsqueeze(0).unsqueeze(0)             # [1, 1, N]
        residual_sq = (temporal_delta - net_pressure).pow(2)     # [B, T-1, N]
        return (residual_sq * H_weight).mean()

    # ═══════════════════════════════════════════════════════════
    #  Encoding
    # ═══════════════════════════════════════════════════════════

    def encode_condition(self, history_sequence):
        """Encode historical traffic → (H, S_eff, μ_mid).

        Type-2 enhancement over final_2:
          Route 2: Interval [μ_low, μ_high] → [S_low, S_high, FOU] → S_eff
          Route 1: Interval-aware semantic closure (per-bound max-min)
          Route 3: Type-2 entropy + FOU-dampened dynamic modulation

        Returns:
            condition_features: [B, Tin, N, D]
            fuzzy_relation:     [N, N] effective graph S_dyn
            mu_fuzzy:           [N, K_f] midpoint memberships (for FRR)
        """
        device = history_sequence.device

        # ── Route 2 + Route 1: Type-2 Semantic Closure ──────────
        S_eff, FOU = self._build_type2_effective_graph(
            device=device, node_features=history_sequence)
        fuzzy_relation = S_eff

        # Cache FOU for downstream use (entropy, FIR)
        self._current_fou = FOU

        # ── Route 3: Entropy-Driven Dynamic Graph (Type-2 aware) ─
        if self.use_entropy_dynamic_graph:
            fuzzy_relation = self._apply_entropy_dynamic_graph(
                fuzzy_relation, FOU=FOU, node_features=history_sequence)

        # Midpoint memberships for FRR conditioning
        mu_fuzzy = self.fuzzy_graph.get_memberships(
            node_features=history_sequence).to(device)

        condition_features = self.condition_encoder(
            history_sequence, fuzzy_relation,
            graph_dist=self.graph_dist,
            mu_fuzzy=mu_fuzzy,
        )
        return condition_features, fuzzy_relation, mu_fuzzy

    # ═══════════════════════════════════════════════════════════
    #  Forward / Predict
    # ═══════════════════════════════════════════════════════════

    def forward(self, batch):
        """Forward entry. Training → returns loss. Inference → predicts."""
        if self.training:
            self._train_step_count += 1
            return self.calculate_loss(batch)
        return self.predict(batch)

    def predict(self, batch):
        """Predict future traffic from history.

        Returns:
            [B, T_out, N, C_out] — only μ (point estimate).
        """
        history_sequence = batch["X"]
        condition, fuzzy_R, mu = self.encode_condition(history_sequence)
        output = self.future_decoder(
            condition, fuzzy_R, graph_dist=self.graph_dist, mu_fuzzy=mu)
        return output[..., :self.output_dim]

    # ═══════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute NLL + FIR + FOU-uncertainty alignment.

        NLL:        μ, log_var from decoder → Gaussian NLL.
        FIR:        Łukasiewicz fuzzy interaction regularization.
        FOU-align:  FOU width → log_var projection → aligned with learned log_var.
        """
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition, fuzzy_R, mu = self.encode_condition(history_sequence)
        output = self.future_decoder(
            condition, fuzzy_R, graph_dist=self.graph_dist, mu_fuzzy=mu)

        # ── Split μ and log σ² ──
        mu_hat = output[..., :self.output_dim]
        log_var = output[..., self.output_dim:]

        # ── NLL for diagonal Gaussian ──
        nll = 0.5 * (LOG_2PI + log_var + torch.exp(-log_var) * (mu_hat - future_sequence) ** 2)
        total = nll.mean()

        # ── FIR: Fuzzy Interaction Regularization ──
        eff_weight = self._get_effective_reg_weight()
        if eff_weight > 0 and self.fir_mode != "none":
            if self.fir_mode == "lukasiewicz":
                reg_loss = self._fuzzy_interaction_regularization(
                    mu_hat, fuzzy_R, node_features=history_sequence)
            elif self.fir_mode == "simple":
                reg_loss = self._simple_consistency_loss(mu_hat, fuzzy_R)
            else:
                reg_loss = 0.0
            total = total + eff_weight * reg_loss

        # ── Type-2 FOU → log_var alignment ──
        if self.use_type2_fuzzy and self.fou_entropy_align_weight > 0:
            total = total + self.fou_entropy_align_weight * self._fou_entropy_alignment(log_var)

        return total

    # ═══════════════════════════════════════════════════════════
    #  FIR: Fuzzy Interaction Regularization
    # ═══════════════════════════════════════════════════════════

    def _get_effective_reg_weight(self):
        """Linearly ramp FIR weight over warmup steps."""
        if self.conservation_loss_weight <= 0:
            return 0.0
        if self.conservation_warmup_epochs <= 0:
            return self.conservation_loss_weight
        total = self.conservation_warmup_epochs * self.conservation_steps_per_epoch
        if total <= 0:
            return self.conservation_loss_weight
        if self._train_step_count >= total:
            return self.conservation_loss_weight
        return self.conservation_loss_weight * (self._train_step_count / total)

    def _fuzzy_interaction_regularization(self, future, fuzzy_relation,
                                            node_features=None):
        """FIR via Łukasiewicz T-norm.

        FlowPressure[i→j] = max(0, congestion[i] + R[i,j] − 1)

        This T-norm expresses: "IF node i is congested
        AND relation(i,j) is strong, THEN there exists flow
        pressure from i to j."  The residual between temporal
        change and net pressure is penalised.

        Type-2 enhancement: when use_entropy_fir_weight=True,
        the residual is weighted by Type-2 fuzzy entropy
        H_t2(i) = H_mid(i)·(1+FOU_avg(i)), additionally
        modulated by structural FOU.

        Args:
            future: [B, T, N, C] prediction.
            fuzzy_relation: [N, N] effective graph (S_dyn).
            node_features: Optional [B,T,N,D] for dynamic membership.
        """
        if future.size(1) < 2:
            return future.new_tensor(0.0)

        node_state = future[..., self.physics_channel_idx]   # [B, T, N]
        current_state = node_state[:, :-1, :]                 # [B, T-1, N]
        next_state = node_state[:, 1:, :]                     # [B, T-1, N]
        temporal_delta = next_state - current_state

        # Normalize to [0, 1] for T-norm
        s_min = current_state.amin(dim=(0, 1), keepdim=True)
        s_max = current_state.amax(dim=(0, 1), keepdim=True).clamp_min(s_min + 1e-6)
        congestion = ((current_state - s_min) / (s_max - s_min)).clamp(0.0, 1.0)

        R = fuzzy_relation.to(device=congestion.device, dtype=congestion.dtype)
        c = congestion.unsqueeze(-1)                          # [B, T-1, N, 1]
        r = R.unsqueeze(0).unsqueeze(0)                       # [1, 1, N, N]
        flow_pressure = (c + r - 1.0).clamp(min=0.0)          # [B, T-1, N, N]

        inflow = flow_pressure.sum(dim=-2)                    # [B, T-1, N]
        outflow = flow_pressure.sum(dim=-1)                   # [B, T-1, N]
        net_pressure = (inflow - outflow) * (s_max - s_min)

        # ── Entropy-weighted FIR (Type-2 aware) ─────────────────
        if self.use_entropy_fir_weight:
            return self._apply_entropy_weighted_fir(
                temporal_delta, net_pressure, node_features=node_features)
        return (temporal_delta - net_pressure).pow(2).mean()

    def _simple_consistency_loss(self, future, fuzzy_relation):
        """Simple consistency baseline for FIR ablation.

        Penalises state-change divergence between strongly related nodes.
        """
        if future.size(1) < 2:
            return future.new_tensor(0.0)

        delta = future[:, 1:] - future[:, :-1]                # [B, T-1, N, C]
        diff = (delta.unsqueeze(-2) - delta.unsqueeze(-3)).pow(2).mean(dim=-1)
        R = fuzzy_relation.unsqueeze(0).unsqueeze(0)          # [1, 1, N, N]
        return (R * diff).mean()

    # ═══════════════════════════════════════════════════════════
    #  FOU-Entropy Alignment (Type-2 regularisation)
    # ═══════════════════════════════════════════════════════════

    def _fou_entropy_alignment(self, learned_log_var):
        """Align FOU width with learned prediction variance.

        FOU width → fou_to_logvar → FOU-derived log_var [N]
        MSE against learned log_var averaged over (B, T, C).

        This gives the Type-2 FOU direct uncertainty semantics:
          wider FOU ↔ higher predictive variance ↔ larger error.
        """
        mu_low, mu_high, _ = self.fuzzy_graph._compute_memberships()
        fou_width = mu_high - mu_low                               # [N, K_f]
        fou_log_var = self.fou_to_logvar(fou_width).squeeze(-1)    # [N]

        # Collapse learned log_var across batch, time, channel → per-node
        target = learned_log_var.mean(dim=(0, 1, 3))               # [N]
        return F.mse_loss(fou_log_var, target)

    # ═══════════════════════════════════════════════════════════
    #  Stability Diagnostics
    # ═══════════════════════════════════════════════════════════

    def get_fuzzy_graph_stability(self):
        """Fuzzy graph stability metrics (Type-2 aware).

        Returns:
            H:   [N] Type-2 cell entropy (FOU-amplified).
            S:   [N] assignment margin (based on midpoint).
            FOU: [N] per-node average FOU.
        """
        if self.fuzzy_graph is None:
            return None, None, None
        fou_node, _ = self.fuzzy_graph.get_fou_stats()
        return (self.fuzzy_graph.get_cell_entropy(),
                self.fuzzy_graph.get_margin_stability(),
                fou_node)

    def get_cell_attention_stability(self, history_sequence):
        """FRR stability metrics from each encoder block.

        Args:
            history_sequence: [B, Tin, N, Cin].

        Returns:
            metrics: list of (H, S) tuples, one per block.
        """
        metrics = []
        device = history_sequence.device
        with torch.no_grad():
            fuzzy_R = self.fuzzy_graph(history_sequence).to(device)
            mu_fuzzy = self.fuzzy_graph.get_memberships().to(device)

            x = self.condition_encoder.input_projection(history_sequence)
            if self.condition_encoder.temporal_position_embedding is not None:
                x = x + self.condition_encoder.temporal_position_embedding[:, :x.shape[1]]

            for block in self.condition_encoder.blocks:
                if hasattr(block, 'cell_attention'):
                    node_repr = x.mean(dim=(0, 1))
                    H, S = block.cell_attention.get_stability_metrics(
                        node_repr, mu_fuzzy)
                    metrics.append((H.cpu(), S.cpu()))
                # Simplified forward for feature collection
                from .utils import apply_temporal_attention as _ta
                t_out = _ta(x, block.temporal_attention)
                x = block.norm_temporal(x + block.dropout(t_out))
                B, T, N, D = x.shape
                g_in = x.reshape(B * T, N, D)
                g_out = block.graph_convolution(g_in, fuzzy_R)
                x = block.norm_graph(x + block.dropout(g_out.reshape(B, T, N, D)))
                if hasattr(block, 'feed_forward'):
                    x = block.norm_ffn(x + block.dropout(block.feed_forward(x)))

        return metrics

    # ═══════════════════════════════════════════════════════════
    #  Type-2 specific diagnostics
    # ═══════════════════════════════════════════════════════════

    def get_fou_heatmap(self):
        """Return the cached FOU matrix for visualization.

        Returns:
            [N, N] FOU matrix from last encode_condition,
            or None if not available.
        """
        if self._current_fou is not None:
            return self._current_fou.detach().cpu()
        return None

    def get_type2_summary(self):
        """Return Type-2 summary statistics for logging.

        Returns:
            dict with keys:
              - fou_mean:   scalar, mean FOU across all node pairs.
              - fou_max:    scalar, max FOU.
              - mode:       str, effective graph combination mode.
              - enabled:    bool, whether Type-2 is active.
        """
        stats = {
            "enabled": self.use_type2_fuzzy,
            "mode": self.type2_graph_mode,
            "fou_gate_scale": self.type2_fou_gate_scale,
        }
        if self._current_fou is not None:
            stats["fou_mean"] = float(self._current_fou.mean().cpu())
            stats["fou_max"] = float(self._current_fou.max().cpu())
        return stats
