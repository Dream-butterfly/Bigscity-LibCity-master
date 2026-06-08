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

        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_num_sets = config.get("fuzzy_num_sets", 3)

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
            graph_matrix, k_hop=self.graph_k_hop)
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

        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, graph_matrix)
            return regression_loss + effective_weight * conservation_loss
        return regression_loss

    def get_fuzzy_graph_stability(self, history_sequence=None):
        if self.fuzzy_graph is None:
            return None, None
        H = self.fuzzy_graph.get_cell_entropy()
        S = self.fuzzy_graph.get_margin_stability()
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

