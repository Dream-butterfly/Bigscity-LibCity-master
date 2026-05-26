"""FuzDiff + Fuzzy Cell Attention — 模糊胞型注意力交通预测模型。

NewFuzzyCellAttention = STEncoder (condition) + FuzzyRelationalGraph
                       + FuzzyCellAttention + FutureDecoder.

Core innovations:
1. Fuzzy Relational Graph: max-min composition of node membership vectors
2. Fuzzy Graph Convolution: K-hop propagation via fuzzy relational powers
3. Fuzzy Cell Attention: node→region→node propagation with hollow kernel
4. Łukasiewicz Fuzzy Conservation loss (T-norm based)

Cell attention complements graph convolution — not replaces it.
Architecture: H' = λ₁·FuzzyGCN(H, R) + λ₂·CellAttention(H, C)
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
from .utils import apply_temporal_attention


class NewFuzzyCellAttention(AbstractTrafficStateModel):
    """Fuzzy relational graph + cell attention model for traffic forecasting.

    Training: forward() → calculate_loss() → scalar.
    Inference: forward() → predict() → [B, T_out, N, C_out].

    Architecture:
        H' = λ₁·FuzzyGCN(H, R) + λ₂·CellAttention(H, C)
        where R = FuzzyRelationalGraph(X), C = learnable centers
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self._scaler = data_feature.get("scaler")

        # ── Geometry ──────────────────────────────────────────────
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)

        # ── Architecture ──────────────────────────────────────────
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

        # ── Fuzzy Relational Graph (Phase A) ──────────────────────
        self.use_fuzzy_graph = config.get("use_fuzzy_graph", True)
        self.fuzzy_num_sets = config.get("fuzzy_num_sets", 3)

        # ── Fuzzy Cell Attention (Phase B) ────────────────────────
        self.use_cell_attention = config.get("use_cell_attention", True)
        self.num_cells = config.get("num_cells", 8)
        self.use_hollow_kernel = config.get("use_hollow_kernel", True)
        self.cell_blend_init = config.get("cell_blend_init", 0.3)

        # ── Conservation Loss ─────────────────────────────────────
        self.conservation_loss_weight = config.get("conservation_loss_weight", 0.1)
        self.conservation_warmup_epochs = int(max(0, config.get("conservation_warmup_epochs", 5)))
        self.conservation_steps_per_epoch = int(
            config.get("conservation_steps_per_epoch", 80)
        )
        self.physics_channel_idx = config.get("physics_channel_idx", 0)
        self.use_fuzzy_conservation = config.get("use_fuzzy_conservation", True)
        self._train_step_count = 0

        # ── Device ────────────────────────────────────────────────
        self.device = config.get("device", torch.device("cpu"))

        # ── Adjacency ─────────────────────────────────────────────
        adjacency_matrix = data_feature.get("adj_mx", np.eye(self.num_nodes, dtype=np.float32))
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        # ── Fuzzy Relational Graph Learner ─────────────────────────
        if self.use_fuzzy_graph:
            self.fuzzy_graph = FuzzyRelationalGraphLearner(
                num_nodes=self.num_nodes,
                hidden_dim=self.hidden_dim,
                num_fuzzy_sets=self.fuzzy_num_sets,
                static_adjacency=adjacency_matrix,
                input_dim=self.feature_dim,
            )
        else:
            self.fuzzy_graph = None

        # ── Submodules ────────────────────────────────────────────
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
        """Encode historical traffic → (H, R).

        Returns:
            condition_features: [B, Tin, N, D]
            graph_matrix: [N, N] fuzzy relation or adjacency
        """
        if self.use_fuzzy_graph and self.fuzzy_graph is not None:
            fuzzy_R = self.fuzzy_graph(history_sequence).to(history_sequence.device)
        else:
            fuzzy_R = self.adjacency_matrix.to(history_sequence.device)
        condition_features = self.condition_encoder(history_sequence, fuzzy_R)
        return condition_features, fuzzy_R

    # ═══════════════════════════════════════════════════════════════
    #  Forward / Predict
    # ═══════════════════════════════════════════════════════════════

    def forward(self, batch):
        """Forward entry. Training → returns loss. Inference → predicts."""
        if self.training:
            self._train_step_count += 1
            return self.calculate_loss(batch)
        return self.predict(batch)

    def predict(self, batch):
        """Predict future traffic from history.

        Returns:
            [B, T_out, N, C_out] predicted future.
        """
        history_sequence = batch["X"]
        condition_features, graph_matrix = self.encode_condition(history_sequence)
        return self.future_decoder(condition_features, graph_matrix)

    # ═══════════════════════════════════════════════════════════════
    #  Training loss
    # ═══════════════════════════════════════════════════════════════

    def calculate_loss(self, batch):
        """Compute L1 loss + optional fuzzy conservation loss."""
        history_sequence = batch["X"]
        future_sequence = batch["y"][..., :self.output_dim]

        condition_features, graph_matrix = self.encode_condition(history_sequence)
        predicted_future = self.future_decoder(condition_features, graph_matrix)

        regression_loss = F.l1_loss(predicted_future, future_sequence)

        effective_weight = self._get_effective_conservation_weight()
        if effective_weight > 0:
            conservation_loss = self._fuzzy_conservation_loss(predicted_future, graph_matrix)
            return regression_loss + effective_weight * conservation_loss
        return regression_loss

    # ═══════════════════════════════════════════════════════════════
    #  Stability Diagnostics (路线 A)
    # ═══════════════════════════════════════════════════════════════

    def get_fuzzy_graph_stability(self, history_sequence=None):
        """获取模糊关系图的稳定性指标（路线A）。

        基于 FuzzyRelationalGraphLearner 的成员隶属度:
          - 模糊胞熵 H ∈ [0, log K] → 高值=模糊边界节点
          - 归属稳定度 S ∈ [0, 1]  → 低值=归属易翻转

        Returns:
            (H, S): 各 [N] 张量，或 (None, None) 如果未启用模糊图。
        """
        if self.fuzzy_graph is None:
            return None, None
        H = self.fuzzy_graph.get_cell_entropy()
        S = self.fuzzy_graph.get_margin_stability()
        return H, S

    def get_cell_attention_stability(self, history_sequence):
        """获取模糊胞型注意力的稳定性指标。

        从每个 Encoder/Decoder Block 的 FuzzyCellAttention 中聚合:
          - 模糊胞熵 H → 高值=交通功能区边界
          - 归属稳定度 S → 低值=归属易翻转（交通相变边界候选）

        Args:
            history_sequence: [B, Tin, N, Cin] 用于计算节点特征。

        Returns:
            metrics: list of (H, S) tuples, 每个 block 一个。
        """
        metrics = []
        with torch.no_grad():
            # 通过 encoder 前向获取中间节点表示
            if self.use_fuzzy_graph and self.fuzzy_graph is not None:
                graph_matrix = self.fuzzy_graph(history_sequence).to(history_sequence.device)
            else:
                graph_matrix = self.adjacency_matrix.to(history_sequence.device)

            x = self.condition_encoder.input_projection(history_sequence)
            if self.condition_encoder.temporal_position_embedding is not None:
                x = x + self.condition_encoder.temporal_position_embedding[:, :x.shape[1]]

            for block in self.condition_encoder.blocks:
                # 收集该 block 的 cell attention 稳定性
                if hasattr(block, 'cell_attention'):
                    node_repr = x.mean(dim=(0, 1))  # global pool [N, D]
                    H, S = block.cell_attention.get_stability_metrics(node_repr)
                    metrics.append((H.cpu(), S.cpu()))
                # 继续前向（不做残差，只收集后续 block 的节点特征）
                temporal_out = apply_temporal_attention(x, block.temporal_attention)
                x = block.norm_temporal(x + block.dropout(temporal_out))

                bt, t, n, d = x.shape
                g_in = x.reshape(bt * t, n, d)
                g_out = block.graph_convolution(g_in, graph_matrix)
                x = block.norm_graph(x + block.dropout(g_out.reshape(bt, t, n, d)))
                # FFN (简化，不做残差因为只需要收集特征)
                x = block.norm_ffn(x + block.dropout(block.feed_forward(x)))

        return metrics

    def _get_effective_conservation_weight(self):
        """Linearly ramp conservation loss weight over warmup epochs."""
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
        """Łukasiewicz T-norm fuzzy conservation loss.

        Uses fuzzy logic for physics-informed flow conservation:
          FlowPressure[i→j] = max(0, congestion[i] + R[i,j] - 1)

        This is the Łukasiewicz T-norm (standard in fuzzy logic):
          T_L(x,y) = max(0, x+y-1)

        Intuition: "IF node i is congested AND relation(i,j) is strong,
        THEN there exists flow pressure from i to j."
        """
        if future_sequence.size(1) < 2:
            return future_sequence.new_tensor(0.0)

        node_state = future_sequence[..., self.physics_channel_idx]  # [B, T, N]
        current_state = node_state[:, :-1, :]   # [B, T-1, N]
        next_state = node_state[:, 1:, :]        # [B, T-1, N]
        temporal_delta = next_state - current_state

        # Normalize state to [0, 1] for T-norm compatibility
        s_min = current_state.amin(dim=(0, 1), keepdim=True)
        s_max = current_state.amax(dim=(0, 1), keepdim=True).clamp_min(s_min + 1e-6)
        congestion = ((current_state - s_min) / (s_max - s_min)).clamp(0.0, 1.0)

        # Łukasiewicz T-norm flow pressure
        # R is [N, N] in [0,1], congestion is [B, T-1, N]
        R = fuzzy_relation.to(device=congestion.device, dtype=congestion.dtype)
        # pressure[i→j] = max(0, congestion[i] + R[i,j] - 1)
        c = congestion.unsqueeze(-1)          # [B, T-1, N, 1]
        r = R.unsqueeze(0).unsqueeze(0)        # [1, 1, N, N]
        flow_pressure = (c + r - 1.0).clamp(min=0.0)  # [B, T-1, N, N]

        # Net pressure change: inflow - outflow
        inflow = flow_pressure.sum(dim=-2)     # Σ_j pressure[j→i]
        outflow = flow_pressure.sum(dim=-1)     # Σ_j pressure[i→j]
        net_pressure = inflow - outflow         # [B, T-1, N]

        # Re-scale to original magnitude
        net_pressure = net_pressure * (s_max - s_min)

        residual = temporal_delta - net_pressure
        return residual.pow(2).mean()
