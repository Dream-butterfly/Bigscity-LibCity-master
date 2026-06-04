"""final_2 — Fuzzy Relational Reasoning System for traffic forecasting.

Enhanced from final_new with fuzzy-first inductive biases (6 routes):
  Route 1 — Fuzzy Semantic Closure:       R → S = max(R, R², R³)
  Route 3 — Entropy-Driven Dynamic Graph: H(i) modulates relation strengths
  Route 3b— Entropy-Weighted FIR:         boundary nodes get higher penalty
  Route 4 — Fuzzy Hierarchical Routing:   true fuzzy membership (no softmax)
                                          + region-region fuzzy relations
  Route 6 — Fuzzy Relation Sparsification:  ε-threshold noise suppression

Local-Global Spatial Dual architecture:
  Spatial-Local:  FuzzyGCN (K-hop topology propagation via fuzzy closure S)
  Spatial-Global: FHR (Fuzzy Hierarchical Routing, replaces spatial self-attn)
  Temporal:       Per-node Transformer

Core innovations:
1. Fuzzy Region Routing → Fuzzy Hierarchical Routing (Route 4)
2. Topology-aware band-pass gate
3. FIR: Łukasiewicz T-norm interaction consistency
4. Fuzzy Semantic Closure via max-min composition (Route 1)
5. Entropy-Driven Dynamic Graph (Route 3)
6. Entropy-Weighted FIR (Route 3b)
7. Fuzzy Relation Sparsification (Route 6)
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel

from .encoder import STEncoder
from .decoder import FutureDecoder
from .graph import FuzzyRelationalGraphLearner
from .utils import compute_hop_distance


class NewFuzzyCellAttention2(AbstractTrafficStateModel):
    """Fuzzy Relational Reasoning System for traffic forecasting.

    The fuzzy relation forms a complete reasoning pipeline:

        μ → R → sparsify(R) → S = closure(R) → S_dyn(S, H) → FHR → FIR(H)

    Not just "fuzzy adjacency" — a fuzzy relational reasoning system.

    Training: forward() → calculate_loss() → scalar.
    Inference: forward() → predict() → [B, T_out, N, C_out].

    Architecture:
        H' = λ₁·FuzzyGCN(H, S_dyn) + λ₂·FHR(H, C)
        S_dyn = sparsify(closure(R)) + α·EntropyModulation
        R = FuzzyRelationalGraph(X), C = fuzzy region prototypes
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

        # ── Route 3: Entropy-Driven Dynamic Graph ───────────────
        self.use_entropy_dynamic_graph = config.get(
            "use_entropy_dynamic_graph", True)
        self.entropy_scale = config.get("entropy_scale", 0.1)
        self.use_entropy_fir_weight = config.get(
            "use_entropy_fir_weight", True)

        # ── Route 4: Fuzzy Hierarchical Routing ──────────────────
        self.use_fuzzy_routing = config.get(
            "use_fuzzy_routing", True)

        # ── Route 6: Fuzzy Relation Sparsification ───────────────
        self.use_fuzzy_sparsification = config.get(
            "use_fuzzy_sparsification", True)
        self.sparsification_epsilon = config.get(
            "sparsification_epsilon", 0.05)

        # ── Device ────────────────────────────────────────────
        self.device = config.get("device", torch.device("cpu"))

        # ── Adjacency + Graph Distance ────────────────────────
        adjacency_matrix = data_feature.get(
            "adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        graph_dist = compute_hop_distance(adjacency_matrix)
        self.register_buffer("graph_dist", graph_dist)

        # ── Fuzzy Relational Graph Learner ────────────────────
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

        # ── Connect FuzzyGraph μ → CellAttention conditioning ─
        self._connect_fuzzy_conditioning()

    def _connect_fuzzy_conditioning(self):
        """Wire fuzzy_to_cell Linear(K_f→K_c) to all CellAttention blocks."""
        for blocks in [self.condition_encoder.blocks, self.future_decoder.blocks]:
            for block in blocks:
                if hasattr(block, 'cell_attention'):
                    block.cell_attention.fuzzy_to_cell = nn.Linear(
                        self.fuzzy_num_sets, self.num_cells)

    # ═══════════════════════════════════════════════════════════
    #  Route 3: Entropy-Driven Dynamic Graph
    # ═══════════════════════════════════════════════════════════

    def _apply_entropy_dynamic_graph(self, fuzzy_R):
        """Modulate fuzzy relation by per-node membership entropy.

        H(i) = -Σ_k μ_k(i)·log μ_k(i)  (fuzzy cell entropy)

        High H(i) → node straddles multiple fuzzy sets
                  → traffic phase-transition boundary
                  → deserves amplified connectivity.

        S_dyn[i,j] = min(1.0, R[i,j] + α·H_norm[i]·H_norm[j])

        This creates a dynamic component: boundary nodes that are both
        uncertain in their functional-class membership receive an
        additional learned-relation boost, helping the model track
        congestion onset/dissipation.

        Args:
            fuzzy_R: [N, N] base fuzzy relation (or semantic closure).

        Returns:
            [N, N] entropy-modulated relation, values in [0, 1].
        """
        H = self.fuzzy_graph.get_cell_entropy().to(device=fuzzy_R.device,
                                                    dtype=fuzzy_R.dtype)
        H_max = H.max().clamp_min(1e-8)
        H_norm = H / H_max                                       # [N], ∈ [0, 1]

        # Outer product: amplify relations between high-entropy node pairs
        H_outer = H_norm.unsqueeze(-1) * H_norm.unsqueeze(-2)    # [N, N]
        H_outer.fill_diagonal_(0.0)                               # no self-boost

        # Add entropy modulation and clamp
        S_dyn = (fuzzy_R + self.entropy_scale * H_outer).clamp(0.0, 1.0)
        return S_dyn

    def _apply_entropy_weighted_fir(self, temporal_delta, net_pressure):
        """Weight FIR residual by fuzzy entropy.

        Boundary nodes (high H) receive higher FIR penalty → the model
        must maintain stronger physical consistency where its fuzzy
        membership is most uncertain.

        Args:
            temporal_delta: [B, T-1, N] state change.
            net_pressure:   [B, T-1, N] FIR net pressure.

        Returns:
            scalar weighted MSE loss.
        """
        H = self.fuzzy_graph.get_cell_entropy().to(
            device=temporal_delta.device, dtype=temporal_delta.dtype)
        H_weight = 1.0 + H / H.max().clamp_min(1e-8)             # [N], ∈ [1, 2]
        H_weight = H_weight.unsqueeze(0).unsqueeze(0)             # [1, 1, N]
        residual_sq = (temporal_delta - net_pressure).pow(2)     # [B, T-1, N]
        return (residual_sq * H_weight).mean()

    # ═══════════════════════════════════════════════════════════
    #  Encoding
    # ═══════════════════════════════════════════════════════════

    def encode_condition(self, history_sequence):
        """Encode historical traffic → (H, R, μ).

        Enhancement over final_new:
          Route 1: If use_semantic_closure, the fuzzy relation R is
                   replaced by its semantic closure S = max(R, R², R³)
                   — explicit transitive fuzzy relational reasoning.
          Route 3: If use_entropy_dynamic_graph, the relation is
                   further modulated by per-node fuzzy entropy H(i)
                   — boundary nodes receive amplified connectivity.

        Returns:
            condition_features: [B, Tin, N, D]
            fuzzy_relation:     [N, N] (closure-entropy-modulated)
            mu_fuzzy:           [N, K_f] fuzzy memberships
        """
        device = history_sequence.device

        # ── Route 1: Fuzzy Semantic Closure ─────────────────────
        if self.use_semantic_closure:
            # S = max(R, R², ..., R^K) via max-min composition
            # This embodies transitive fuzzy relational reasoning:
            #   IF i∼k AND k∼j THEN i∼j  (in the max-min sense)
            fuzzy_relation = self.fuzzy_graph.get_semantic_closure(
                max_hops=self.semantic_closure_hops).to(device)
        else:
            fuzzy_relation = self.fuzzy_graph(history_sequence).to(device)

        # ── Route 3: Entropy-Driven Dynamic Graph ───────────────
        if self.use_entropy_dynamic_graph:
            fuzzy_relation = self._apply_entropy_dynamic_graph(fuzzy_relation)

        mu_fuzzy = self.fuzzy_graph.get_memberships().to(device)
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
            [B, T_out, N, C_out].
        """
        history_sequence = batch["X"]
        condition, fuzzy_R, mu = self.encode_condition(history_sequence)
        return self.future_decoder(
            condition, fuzzy_R, graph_dist=self.graph_dist, mu_fuzzy=mu)

    # ═══════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute L1 loss + optional FIR regularization."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition, fuzzy_R, mu = self.encode_condition(history_sequence)
        pred = self.future_decoder(
            condition, fuzzy_R, graph_dist=self.graph_dist, mu_fuzzy=mu)

        regression_loss = F.l1_loss(pred, future_sequence)

        eff_weight = self._get_effective_reg_weight()
        if eff_weight > 0 and self.fir_mode != "none":
            if self.fir_mode == "lukasiewicz":
                reg_loss = self._fuzzy_interaction_regularization(
                    pred, fuzzy_R)
            elif self.fir_mode == "simple":
                reg_loss = self._simple_consistency_loss(pred, fuzzy_R)
            else:
                reg_loss = 0.0
            return regression_loss + eff_weight * reg_loss
        return regression_loss

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

    def _fuzzy_interaction_regularization(self, future, fuzzy_relation):
        """FIR via Łukasiewicz T-norm.

        FlowPressure[i→j] = max(0, congestion[i] + R[i,j] - 1)

        This T-norm expresses: "IF node i is congested
        AND relation(i,j) is strong, THEN there exists flow
        pressure from i to j."  The residual between temporal
        change and net pressure is penalized.

        When use_entropy_fir_weight=True (final_2), the residual
        is weighted by fuzzy entropy H(i) — boundary nodes at
        traffic phase transitions get higher FIR penalty.
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

        # ── Entropy-weighted FIR (Route 3) ────────────────────────
        if self.use_entropy_fir_weight:
            return self._apply_entropy_weighted_fir(
                temporal_delta, net_pressure)
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
    #  Stability Diagnostics
    # ═══════════════════════════════════════════════════════════

    def get_fuzzy_graph_stability(self):
        """Fuzzy graph stability metrics.

        Returns:
            H: [N] cell entropy (high → boundary nodes).
            S: [N] assignment margin (low → unstable assignment).
        """
        if self.fuzzy_graph is None:
            return None, None
        return self.fuzzy_graph.get_cell_entropy(), self.fuzzy_graph.get_margin_stability()

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
