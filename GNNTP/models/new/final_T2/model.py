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
        self.beta_init_random = config.get("beta_init_random", False)

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
                static_adjacency=adjacency_matrix,
                input_dim=self.feature_dim,
                closure_steps=int(max(0, config.get("graph_closure_steps", 0))),
                topk=self.graph_topk,
                beta_init_random=self.beta_init_random,
            )
        else:
            self.fuzzy_graph = None

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
            graph_matrix, fou = self.fuzzy_graph.get_type2_info(history_sequence)
            graph_matrix = graph_matrix.to(history_sequence.device)
            fou = fou.to(history_sequence.device)
            self._current_fou = fou  # cache for diagnostics
        else:
            graph_matrix = self.adjacency_matrix.to(history_sequence.device)
            fou = None
            self._current_fou = None
        graph_powers = FuzzyGraphConvolution.precompute_powers(
            graph_matrix, k_hop=self.graph_k_hop, topk=self.graph_topk)
        condition_features = self.condition_encoder(
            history_sequence, graph_matrix, graph_uncertainty=fou, powers=graph_powers)
        return condition_features, graph_matrix, fou, graph_powers

    def forward(self, batch):
        if self.training:
            self._train_step_count += 1
            return self.calculate_loss(batch)
        return self.predict(batch)

    def predict(self, batch):
        history_sequence = batch["X"]
        condition_features, graph_matrix, fou, graph_powers = self.encode_condition(history_sequence)
        return self.future_decoder(
            condition_features, graph_matrix, graph_uncertainty=fou, powers=graph_powers)

    def calculate_loss(self, batch):
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition_features, graph_matrix, fou, graph_powers = self.encode_condition(history_sequence)
        predicted_future = self.future_decoder(
            condition_features, graph_matrix, graph_uncertainty=fou, powers=graph_powers)

        regression_loss = F.l1_loss(predicted_future, future_sequence)
        total = regression_loss

        # ── Conservation (if enabled) ──
        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, graph_matrix)
            total = total + effective_weight * conservation_loss

        # ── Type-2 anti-collapse losses ──
        if self.fuzzy_graph is not None:
            g = self.fuzzy_graph
            beta = F.softmax(g.relation_mix_logits, dim=0)

            # ① β entropy: maximize → keep all three views active
            if self.t2_entropy_weight > 0:
                h_beta = -(beta * (beta + 1e-8).log()).sum()
                self._t2_entropy_val = (-h_beta).detach()  # cache for diagnostics
                total = total + self.t2_entropy_weight * (-h_beta)

            # ② Interval active: ensure σ_low < σ_high (functionally different)
            if self.t2_interval_weight > 0:
                sl = F.softplus(g.log_sigma_low) + 1e-3
                sh = F.softplus(g.log_sigma_high) + 1e-3
                s_ratio = (sl / (sh + 1e-8)).clamp(0, 1)
                s_gap = F.relu(s_ratio - self.t2_interval_ratio_threshold)
                self._t2_gap_val = s_gap.mean().detach()
                total = total + self.t2_interval_weight * s_gap.mean()

            # ③ FOU floor: keep membership interval from collapsing to 0
            if self.t2_fou_floor_weight > 0 and self._current_fou is not None:
                fou_gap = F.relu(0.01 - self._current_fou.mean())
                self._t2_fou_val = fou_gap.detach()
                total = total + self.t2_fou_floor_weight * fou_gap

        # ── Type-2 gradient boost: amplify σ/r/β gradients post-backward ──
        if (self.t2_lr_boost != 1.0 and self.fuzzy_graph is not None
                and total.requires_grad):
            _params = [
                self.fuzzy_graph.log_sigma_low,
                self.fuzzy_graph.log_sigma_high,
                self.fuzzy_graph.relation_mix_logits,
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
                diag['fou_mean'] = round(fou.mean().item(), 4)
                diag['fou_std'] = round(fou.std().item(), 4)
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
            # Anti-collapse loss values
            if hasattr(self, '_t2_entropy_val'):
                diag['t2_entropy_val']  = round(self._t2_entropy_val.item(), 4)
                diag['t2_entropy_w']    = self.t2_entropy_weight
            if hasattr(self, '_t2_gap_val'):
                diag['t2_gap_val']      = round(self._t2_gap_val.item(), 4)
                diag['t2_interval_w']   = self.t2_interval_weight
            if hasattr(self, '_t2_fou_val'):
                diag['t2_fou_val']      = round(self._t2_fou_val.item(), 4)
                diag['t2_fou_floor_w']  = self.t2_fou_floor_weight
            # Gradient norms for key Type-2 parameters
            g = self.fuzzy_graph
            for pname, grad_key in [
                ('|∇β|', 'relation_mix_logits'),
                ('|∇σ_low|', 'log_sigma_low'),
                ('|∇σ_high|', 'log_sigma_high'),
                ('|∇proto|', 'prototype_center'),
            ]:
                param = getattr(g, grad_key, None)
                if param is not None and param.grad is not None:
                    gn = param.grad.detach().abs().mean().item()
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
                graph_matrix, fou = self.fuzzy_graph.get_type2_info(history_sequence)
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

