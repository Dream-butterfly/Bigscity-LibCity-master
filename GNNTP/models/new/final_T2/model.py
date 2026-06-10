"""final_T2 model: integrates Type-2 FOU as a local uncertainty modulator.

Key engineering choices implemented here follow your specification:
- Type-2 models membership uncertainty but only gates attention.
- Closure (if enabled) is applied only on the mid/expected fuzzy graph.
"""

from logging import getLogger

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

from GNNTP.models.abstract_traffic_state_model import AbstractTrafficStateModel

from .encoder import STEncoder
from .decoder import FutureDecoder
from .graph import FuzzyGraphConvolution, FuzzyRelationalGraphLearner
from .utils import apply_temporal_attention


class NewFuzzyCellAttention(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self._scaler = data_feature.get("scaler")

        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)

        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_heads = config.get("num_heads", 2)
        self.encoder_layers = config.get("encoder_layers", 2)
        self.decoder_layers = config.get("decoder_layers", 2)
        self.ffn_hidden_dim = config.get("ffn_hidden_dim", 128)
        self.graph_k_hop = config.get("graph_k_hop", 2)
        self.dropout = config.get("dropout", 0.1)
        self.use_spatiotemporal_attention = config.get("use_spatiotemporal_attention", True)
        self.use_temporal_position_embedding = config.get("use_temporal_position_embedding", True)
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
        self.use_torch_compile = config.get("use_torch_compile", False)

        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_num_sets = config.get("fuzzy_num_sets", 3)
        self.graph_topk = config.get("graph_topk", 32)
        self.graph_sparsify_topk = config.get("graph_sparsify_topk", 0)  # 0=off
        self.beta_init_random = config.get("beta_init_random", False)
        self.use_proto_adaptive_embed = config.get("use_proto_adaptive_embed", False)
        self.proto_norm_reg_weight = config.get("proto_norm_reg_weight", 0.01)
        self._latent_norm_reg_weight = config.get("latent_norm_reg_weight", 0.01)
        self.decoder_node_mode = config.get("decoder_node_mode", "embed")
        self.use_static_blend = config.get("use_static_blend", True)
        self._proto_diversity_weight = config.get("proto_diversity_weight", 0.0)

        self.use_cell_attention = config.get("use_cell_attention", True)
        self.num_cells = config.get("num_cells", 8)
        self.use_hollow_kernel = config.get("use_hollow_kernel", True)
        self.cell_blend_init = config.get("cell_blend_init", 0.3)

        self.conservation_loss_weight = config.get("conservation_loss_weight", 0.1)
        self.conservation_warmup_epochs = int(max(0, config.get("conservation_warmup_epochs", 5)))
        self.conservation_steps_per_epoch = int(
            config.get("conservation_steps_per_epoch", 80)
        )
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self.use_fuzzy_conservation = config.get("use_fuzzy_conservation", True)
        self.loss_mode = config.get("loss_mode", "mae")  # mae | mse | huber
        self.huber_delta = config.get("huber_delta", 1.0)
        # Type-2 anti-collapse losses (all default 0 → backward compatible)
        self.t2_entropy_weight   = config.get("t2_entropy_weight", 0.0)    # β entropy (keep β diverse)
        self.t2_interval_weight          = config.get("t2_interval_weight", 0.0)
        self.t2_interval_ratio_threshold  = config.get("t2_interval_ratio_threshold", 0.90)
        self.t2_fou_floor_weight = config.get("t2_fou_floor_weight", 0.0)  # FOU floor (keep μ interval)
        # Type-2 gradient boost (default 1=off)
        self.t2_lr_boost = float(config.get("t2_lr_boost", 1.0))
        self._train_step_count = 0

        self.device = config.get("device", torch.device("cpu"))

        adjacency_matrix = data_feature.get("adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        if self.use_fuzzy_graph:
            self.fuzzy_graph = FuzzyRelationalGraphLearner(
                num_nodes=self.num_nodes,
                hidden_dim=self.hidden_dim,
                num_fuzzy_sets=self.fuzzy_num_sets,
                static_adjacency=adjacency_matrix if self.use_static_blend else None,
                input_dim=self.feature_dim,
                closure_steps=int(max(0, config.get("graph_closure_steps", 0))),
                topk=self.graph_topk,
                beta_init_random=self.beta_init_random,
            )
            self.fuzzy_graph.graph_sparsify_topk = (
                self.graph_sparsify_topk if self.graph_sparsify_topk > 0 else None)
        else:
            self.fuzzy_graph = None

        # Prototype-aware adaptive embedding: E_node = μ_mid @ E_proto
        # Each of K fuzzy prototypes learns a D-dim spatial signature.
        # Node embedding = membership-weighted mixture (N×K @ K×D → N×D).
        # 3×hidden_dim parameters total — cannot become a shortcut.
        if self.use_proto_adaptive_embed:
            self.proto_embed = nn.Parameter(
                torch.zeros(1, self.fuzzy_num_sets, self.hidden_dim))
            nn.init.trunc_normal_(self.proto_embed, std=0.02)

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
            use_fuzzy_graph=self.use_fuzzy_graph,
            use_cell_attention=self.use_cell_attention,
            num_cells=self.num_cells,
            use_hollow_kernel=self.use_hollow_kernel,
            cell_blend_init=self.cell_blend_init,
        )
        self.future_decoder = FutureDecoder(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.decoder_layers,
            ffn_hidden_dim=self.ffn_hidden_dim,
            graph_k_hop=self.graph_k_hop,
            dropout=self.dropout,
            use_spatiotemporal_attention=self.use_spatiotemporal_attention,
            output_window=self.output_window,
            num_nodes=self.num_nodes,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_fuzzy_graph=self.use_fuzzy_graph,
            use_cell_attention=self.use_cell_attention,
            num_cells=self.num_cells,
            use_hollow_kernel=self.use_hollow_kernel,
            cell_blend_init=self.cell_blend_init,
            node_mode=self.decoder_node_mode,
            proto_embed=self.proto_embed if self.use_proto_adaptive_embed else None,
        )

        # ── torch.compile each block (standard Transformer kernels fuse well) ──
        if self.use_torch_compile:
            torch.set_float32_matmul_precision('high')  # enable TF32, ~2× speed on Ampere+
            for i, block in enumerate(self.condition_encoder.blocks):
                self.condition_encoder.blocks[i] = torch.compile(
                    block, mode="default")
            for i, block in enumerate(self.future_decoder.blocks):
                self.future_decoder.blocks[i] = torch.compile(
                    block, mode="default")

    def encode_condition(self, history_sequence):
        if self.use_fuzzy_graph and self.fuzzy_graph is not None:
            graph_matrix, fou, mu_mid = self.fuzzy_graph.get_type2_info(history_sequence)
            graph_matrix = graph_matrix.to(history_sequence.device)
            fou = fou.to(history_sequence.device)
            mu_mid = mu_mid.to(history_sequence.device)
            self._current_fou = fou  # cache for diagnostics
            self._current_mu_mid = mu_mid  # cache for diagnostics
        else:
            graph_matrix = self.adjacency_matrix.to(history_sequence.device)
            fou = None
            mu_mid = None
            self._current_fou = None
            self._current_mu_mid = None
        graph_powers = FuzzyGraphConvolution.precompute_powers(
            graph_matrix, k_hop=self.graph_k_hop, topk=self.graph_topk)
        condition_features = self.condition_encoder(
            history_sequence, graph_matrix, graph_uncertainty=fou, powers=graph_powers)

        # Prototype-aware spatial embedding: route through Type-2 membership
        if self.use_proto_adaptive_embed and mu_mid is not None:
            # E_node = μ_mid @ E_proto  (N,K) @ (K,D) → (N,D)
            node_adaptive = mu_mid @ self.proto_embed  # (N, K) @ (1, K, D) → (N, D)
            condition_features = condition_features + node_adaptive  # broadcast (B,T)

        return condition_features, graph_matrix, fou, graph_powers

    def forward(self, batch):
        if self.training:
            self._train_step_count += 1
            return self.calculate_loss(batch)
        return self.predict(batch)

    def predict(self, batch):
        history_sequence = batch["X"]
        condition_features, graph_matrix, fou, graph_powers = self.encode_condition(history_sequence)
        mu_mid = getattr(self, '_current_mu_mid', None)
        return self.future_decoder(
            condition_features, graph_matrix, graph_uncertainty=fou,
            powers=graph_powers, mu_mid=mu_mid)

    def calculate_loss(self, batch):
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition_features, graph_matrix, fou, graph_powers = self.encode_condition(history_sequence)
        mu_mid = getattr(self, '_current_mu_mid', None)
        predicted_future = self.future_decoder(
            condition_features, graph_matrix, graph_uncertainty=fou,
            powers=graph_powers, mu_mid=mu_mid)

        # ── Regression loss ──
        if self.loss_mode == "mse":
            regression_loss = F.mse_loss(predicted_future, future_sequence)
        elif self.loss_mode == "huber":
            regression_loss = F.smooth_l1_loss(
                predicted_future, future_sequence, beta=self.huber_delta)
        else:  # "mae" (default, backward compatible)
            regression_loss = F.l1_loss(predicted_future, future_sequence)
        total = regression_loss
        self._loss_mae = regression_loss.detach()  # cache for diagnostics

        # ── Conservation (if enabled) ──
        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, graph_matrix)
            contrib = effective_weight * conservation_loss
            self._loss_consv = contrib.detach()
            total = total + contrib
        else:
            self._loss_consv = torch.tensor(0.0)

        # ── Type-2 anti-collapse losses ──
        self._loss_ent = torch.tensor(0.0)
        self._loss_gap = torch.tensor(0.0)
        self._loss_fou = torch.tensor(0.0)
        self._raw_ent = torch.tensor(0.0)
        self._raw_gap = torch.tensor(0.0)
        self._raw_fou = torch.tensor(0.0)
        if self.fuzzy_graph is not None:
            g = self.fuzzy_graph
            beta = F.softmax(g.relation_mix_logits, dim=0)

            # ① β entropy: maximize → keep all three views active
            if self.t2_entropy_weight > 0:
                h_beta = -(beta * (beta + 1e-8).log()).sum()
                self._raw_ent = (-h_beta).detach()
                self._loss_ent = (self.t2_entropy_weight * (-h_beta)).detach()
                total = total + self.t2_entropy_weight * (-h_beta)

            # ② Interval active: ensure σ_low < σ_high (functionally different)
            if self.t2_interval_weight > 0:
                sl = F.softplus(g.log_sigma_low) + 1e-3
                sh = sl + F.softplus(g.log_sigma_delta) + 1e-3
                s_ratio = (sl / (sh + 1e-8)).clamp(0, 1)
                s_gap = F.relu(s_ratio - self.t2_interval_ratio_threshold)
                self._raw_gap = s_gap.mean().detach()
                self._loss_gap = (self.t2_interval_weight * s_gap.mean()).detach()
                total = total + self.t2_interval_weight * s_gap.mean()

            # ③ FOU floor: keep membership interval from collapsing to 0
            if self.t2_fou_floor_weight > 0 and self._current_fou is not None:
                fou_gap = F.relu(0.01 - self._current_fou.mean())
                self._raw_fou = fou_gap.detach()
                self._loss_fou = (self.t2_fou_floor_weight * fou_gap).detach()
                total = total + self.t2_fou_floor_weight * fou_gap

        # ── Prototype norm regularization: prevent |prototype_center| → 0 ──
        self._loss_proto_norm = torch.tensor(0.0)
        if self.proto_norm_reg_weight > 0 and self.fuzzy_graph is not None:
            # Three independent prototype sets → average norm across all
            proto_norms = torch.stack([
                self.fuzzy_graph.prototype_center_low.norm(dim=-1).mean(),
                self.fuzzy_graph.prototype_center_mid.norm(dim=-1).mean(),
                self.fuzzy_graph.prototype_center_high.norm(dim=-1).mean(),
            ])
            proto_norm_loss = (proto_norms.mean() - 1.0).pow(2)
            self._loss_proto_norm = (self.proto_norm_reg_weight * proto_norm_loss).detach()
            total = total + self.proto_norm_reg_weight * proto_norm_loss

        # ── Latent norm regularization: prevent node_transform weights → 0 ──
        # LayerNorm kills gradient on magnitude; this compensates.
        self._loss_latent_norm = torch.tensor(0.0)
        if (hasattr(self, '_latent_norm_reg_weight')
                and self._latent_norm_reg_weight > 0
                and self.fuzzy_graph is not None
                and hasattr(self.fuzzy_graph, '_node_latent_for_reg')):
            latent_norms = self.fuzzy_graph._node_latent_for_reg.norm(dim=-1).mean()
            latent_norm_loss = (latent_norms - 2.0).pow(2)
            self._loss_latent_norm = (self._latent_norm_reg_weight * latent_norm_loss).detach()
            total = total + self._latent_norm_reg_weight * latent_norm_loss

        # ── Cross-view prototype diversity: force three views to diverge ──
        self._loss_proto_div = torch.tensor(0.0)
        if (hasattr(self, '_proto_diversity_weight')
                and self._proto_diversity_weight > 0
                and self.fuzzy_graph is not None
                and hasattr(self.fuzzy_graph, 'prototype_center_low')):
            g = self.fuzzy_graph
            p_low  = F.normalize(g.prototype_center_low, dim=-1)   # (K, D)
            p_mid  = F.normalize(g.prototype_center_mid, dim=-1)
            p_high = F.normalize(g.prototype_center_high, dim=-1)
            # Max inter-view cosine — trigger if ANY prototype pair correlates
            # Mean() drowns the signal (256 pairs avg → ~0±0.005 in 128D).
            # Max() catches even a single correlated pair.
            sim_lm_mat = p_low @ p_mid.T    # (K, K) cosine similarity matrix
            sim_mh_mat = p_mid @ p_high.T
            sim_lh_mat = p_low @ p_high.T
            cross_sim = torch.max(torch.stack([
                sim_lm_mat.max(), sim_mh_mat.max(), sim_lh_mat.max()
            ]))
            cross_sim_clamped = F.relu(cross_sim - 0.3)  # allow up to 0.3, penalize above
            self._loss_proto_div = (self._proto_diversity_weight * cross_sim_clamped).detach()
            total = total + self._proto_diversity_weight * cross_sim_clamped

        # ── Type-2 gradient boost: amplify σ/r/β gradients post-backward ──
        if (self.t2_lr_boost != 1.0 and self.fuzzy_graph is not None
                and total.requires_grad):
            _params = [
                self.fuzzy_graph.log_sigma_low,
                self.fuzzy_graph.log_sigma_delta,
                self.fuzzy_graph.relation_mix_logits,
                self.fuzzy_graph.prototype_center_low,
                self.fuzzy_graph.prototype_center_mid,
                self.fuzzy_graph.prototype_center_high,
            ]
            _boost = self.t2_lr_boost
            def _amp_grad(_grad):
                for p in _params:
                    if p.grad is not None:
                        p.grad.mul_(_boost)
            total.register_hook(_amp_grad)

        return total

    def get_fuzzy_graph_stability(self, history_sequence=None):
        if self.fuzzy_graph is None:
            return None, None
        H = self.fuzzy_graph.get_cell_entropy(history_sequence)
        S = self.fuzzy_graph.get_margin_stability(history_sequence)
        return H, S

    def get_type2_diagnostics(self):
        """Return Type-2 diagnostic values for epoch-end logging.

        Returns:
            dict with keys:
              beta:         [β_low, β_mid, β_high] interval mix weights
              blend:        sigmoid(blend_logit) fuzzy-static mix
              fou_norm:     mean FOU norm (per-node), or None
        """
        diag = {}
        if self.fuzzy_graph is not None:
            # Interval mix weights
            beta = F.softmax(self.fuzzy_graph.relation_mix_logits, dim=0).detach()
            diag['beta'] = [round(b.item(), 3) for b in beta]
            # Fuzzy-static blend
            diag['blend'] = round(
                torch.sigmoid(self.fuzzy_graph.blend_logit).item(), 3)
            # FOU statistics (from last forward)
            if hasattr(self, '_current_fou') and self._current_fou is not None:
                fou = self._current_fou.detach()
                diag['fou_mean'] = fou.mean().item()
                diag['fou_std'] = fou.std().item()
            # Independent dual widths (genuine Type-2)
            if hasattr(self.fuzzy_graph, '_current_sigma_low'):
                sl = self.fuzzy_graph._current_sigma_low
                sh = self.fuzzy_graph._current_sigma_high
                diag['sigma_low_mean']  = round(sl.mean().item(), 4)
                diag['sigma_low_std']   = round(sl.std().item(), 4)
                diag['sigma_high_mean'] = round(sh.mean().item(), 4)
                diag['sigma_high_std']  = round(sh.std().item(), 4)
                diag['sigma_ratio']     = round(
                    (sl / (sh + 1e-8)).clamp(0, 1).mean().item(), 4)
            if hasattr(self.fuzzy_graph, '_current_sigma_mid'):
                diag['sigma_mid_mean'] = round(
                    self.fuzzy_graph._current_sigma_mid.mean().item(), 4)
            # Anti-collapse loss values
            # Weighted contributions for L=[] display
            if hasattr(self, '_loss_mae'):
                diag['loss_mae']  = round(self._loss_mae.item(), 4)
                diag['loss_consv'] = round(self._loss_consv.item(), 4)
                diag['loss_ent']   = round(self._loss_ent.item(), 4)
                diag['loss_gap']   = round(self._loss_gap.item(), 4)
                diag['loss_fou']   = round(self._loss_fou.item(), 4)
                diag['loss_proto_norm'] = round(self._loss_proto_norm.item(), 4)
                diag['loss_latent_norm'] = round(self._loss_latent_norm.item(), 4)
                diag['loss_proto_div'] = round(self._loss_proto_div.item(), 4)
            # Raw (unweighted) anti-collapse loss values
            if hasattr(self, '_raw_ent'):
                diag['raw_ent'] = round(self._raw_ent.item(), 4)
                diag['raw_gap'] = round(self._raw_gap.item(), 4)
                diag['raw_fou'] = round(self._raw_fou.item(), 4)
            # Membership-level interval stats
            if hasattr(self.fuzzy_graph, '_current_mu_diff_mean'):
                diag['mu_diff_mean'] = round(
                    self.fuzzy_graph._current_mu_diff_mean.item(), 4)
                diag['mu_diff_max'] = round(
                    self.fuzzy_graph._current_mu_diff_max.item(), 4)
            # Distance & raw membership stats
            if hasattr(self.fuzzy_graph, '_current_d2_mean'):
                diag['d2_mean'] = round(
                    self.fuzzy_graph._current_d2_mean.item(), 2)
                diag['d2_std'] = round(
                    self.fuzzy_graph._current_d2_std.item(), 2)
            if hasattr(self.fuzzy_graph, '_current_mu_raw_low_mean'):
                diag['mu_raw_low']  = round(
                    self.fuzzy_graph._current_mu_raw_low_mean.item(), 4)
                diag['mu_raw_mid']  = round(
                    self.fuzzy_graph._current_mu_raw_mid_mean.item(), 4)
                diag['mu_raw_high'] = round(
                    self.fuzzy_graph._current_mu_raw_high_mean.item(), 4)
            # Prototype drift diagnostics
            if hasattr(self.fuzzy_graph, '_current_proto_norm'):
                diag['proto_norm'] = round(
                    self.fuzzy_graph._current_proto_norm.item(), 2)
                diag['latent_norm'] = round(
                    self.fuzzy_graph._current_latent_norm.item(), 2)
                diag['center_dist'] = round(
                    self.fuzzy_graph._current_center_dist.item(), 2)
            if hasattr(self.fuzzy_graph, '_current_transform_weight'):
                diag['transform_w'] = round(
                    self.fuzzy_graph._current_transform_weight.item(), 2)
            if hasattr(self.fuzzy_graph, '_current_proto_update'):
                diag['proto_up'] = round(
                    self.fuzzy_graph._current_proto_update.item(), 4)
                diag['latent_up'] = round(
                    self.fuzzy_graph._current_latent_update.item(), 2)
            if hasattr(self.fuzzy_graph, '_current_sigma_delta_mean'):
                diag['sigma_delta_mean'] = round(
                    self.fuzzy_graph._current_sigma_delta_mean.item(), 4)
            # Gradient norms for key Type-2 parameters
            g = self.fuzzy_graph
            for pname, grad_key in [
                ('|∇β|', 'relation_mix_logits'),
                ('|∇σ_low|', 'log_sigma_low'),
                ('|∇δ|', 'log_sigma_delta'),
                ('|∇proto|', 'prototype_center'),
            ]:
                param = getattr(g, grad_key, None)
                # For prototype, max gradient across all three sets
                if grad_key == 'prototype_center' and hasattr(g, 'prototype_center_low'):
                    gn = max(
                        g.prototype_center_low.grad.detach().abs().mean().item() if g.prototype_center_low.grad is not None else 0,
                        g.prototype_center_mid.grad.detach().abs().mean().item() if g.prototype_center_mid.grad is not None else 0,
                        g.prototype_center_high.grad.detach().abs().mean().item() if g.prototype_center_high.grad is not None else 0,
                    )
                elif param is not None and param.grad is not None:
                    gn = param.grad.detach().abs().mean().item()
                else:
                    gn = 0.0
                diag[grad_key + '_grad'] = gn
            # Relation diffs (are the three graphs actually different?)
            if hasattr(g, '_current_R_diff_lm'):
                diag['R_diff_lm'] = round(g._current_R_diff_lm.item(), 4)
                diag['R_diff_hm'] = round(g._current_R_diff_hm.item(), 4)
            # R_gap: |R_high - R_low| / |R_mid| — Type-2 collapse indicator
            if hasattr(g, '_current_R_gap'):
                diag['R_gap'] = round(g._current_R_gap.item(), 4)
            # Effective Type-2 width
            if hasattr(g, '_current_eff_width'):
                diag['eff_width'] = round(g._current_eff_width.item(), 4)
            # R correlations: are the three views structurally different?
            if hasattr(g, '_current_R_corr_lm'):
                diag['R_corr_lm'] = round(g._current_R_corr_lm.item(), 4)
                diag['R_corr_lh'] = round(g._current_R_corr_lh.item(), 4)
                diag['R_corr_mh'] = round(g._current_R_corr_mh.item(), 4)
            # β delta from previous epoch
            if hasattr(self, '_last_beta'):
                cur_beta = torch.tensor(diag['beta'])
                diag['beta_delta'] = round(
                    (cur_beta - self._last_beta).abs().mean().item(), 6)
            self._last_beta = torch.tensor(diag['beta'])
            # β entropy (interval mix diversity)
            beta = F.softmax(self.fuzzy_graph.relation_mix_logits, dim=0)
            h = -(beta * (beta + 1e-8).log()).sum().item()
            diag['beta_entropy'] = round(h, 4)
            # Raw logits std — tracks if router is converging
            diag['logits_std'] = round(
                self.fuzzy_graph.relation_mix_logits.detach().std().item(), 4)
        if self.use_cell_attention and hasattr(self, 'condition_encoder'):
            # Cell blend from first encoder block
            first_block = self.condition_encoder.blocks[0]
            if hasattr(first_block, 'cell_attention'):
                cb = torch.sigmoid(
                    first_block.cell_attention.cell_blend).detach().item()
                diag['cell_blend'] = round(cb, 3)
        return diag

    def get_cell_attention_stability(self, history_sequence):
        metrics = []
        with torch.no_grad():
            if self.use_fuzzy_graph and self.fuzzy_graph is not None:
                graph_matrix, fou, _ = self.fuzzy_graph.get_type2_info(history_sequence)
                graph_matrix = graph_matrix.to(history_sequence.device)
                fou = fou.to(history_sequence.device)
            else:
                graph_matrix = self.adjacency_matrix.to(history_sequence.device)
                fou = None

            x = self.condition_encoder.input_projection(history_sequence)
            if self.condition_encoder.temporal_position_embedding is not None:
                x = x + self.condition_encoder.temporal_position_embedding[:, :x.shape[1]]

            for block in self.condition_encoder.blocks:
                if hasattr(block, 'cell_attention'):
                    node_repr = x.mean(dim=(0, 1))
                    H, S = block.cell_attention.get_stability_metrics(node_repr)
                    metrics.append((H.cpu(), S.cpu()))
                temporal_out = apply_temporal_attention(x, block.temporal_attention)
                x = block.norm_temporal(x + block.dropout(temporal_out))

                bt, t, n, d = x.shape
                g_in = x.reshape(bt * t, n, d)
                g_out = block.graph_convolution(g_in, graph_matrix)
                x = block.norm_graph(x + block.dropout(g_out.reshape(bt, t, n, d)))
                x = block.norm_ffn(x + block.dropout(block.feed_forward(x)))

        return metrics

    def _get_effective_conservation_weight(self):
        if self.conservation_loss_weight <= 0:
            return 0.0
        if self.conservation_warmup_epochs <= 0:
            return self.conservation_loss_weight
        steps_per_epoch = self.conservation_steps_per_epoch
        total_warmup_steps = self.conservation_warmup_epochs * steps_per_epoch
        if total_warmup_steps <= 0:
            return self.conservation_loss_weight
        if self._train_step_count >= total_warmup_steps:
            return self.conservation_loss_weight
        return self.conservation_loss_weight * (self._train_step_count / total_warmup_steps)

    def _fuzzy_conservation_loss(self, future_sequence, fuzzy_relation):
        if future_sequence.size(1) < 2:
            return future_sequence.new_tensor(0.0)

        node_state = future_sequence[..., self.physics_channel_idx]
        current_state = node_state[:, :-1, :]
        next_state = node_state[:, 1:, :]
        temporal_delta = next_state - current_state

        s_min = current_state.amin(dim=(0, 1), keepdim=True)
        s_max = current_state.amax(dim=(0, 1), keepdim=True).clamp_min(s_min + 1e-6)
        congestion = ((current_state - s_min) / (s_max - s_min)).clamp(0.0, 1.0)

        R = fuzzy_relation.to(device=congestion.device, dtype=congestion.dtype)
        c = congestion.unsqueeze(-1)
        r = R.unsqueeze(0).unsqueeze(0)
        flow_pressure = (c + r - 1.0).clamp(min=0.0)

        inflow = flow_pressure.sum(dim=-2)
        outflow = flow_pressure.sum(dim=-1)
        net_pressure = inflow - outflow
        net_pressure = net_pressure * (s_max - s_min)

        residual = temporal_delta - net_pressure
        return residual.pow(2).mean()

