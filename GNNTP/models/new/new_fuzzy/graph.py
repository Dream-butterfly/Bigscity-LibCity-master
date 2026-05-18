"""Graph convolution and adaptive graph learning components."""

import math

import torch
import torch.nn as nn

from .utils import build_normalized_adjacency, expand_adjacency_batch


class GraphConvolution(nn.Module):
    """K-hop graph convolution: X' = Σ_{k=0}^{K} Â^k X W_k.

    Each hop has its own learned linear projection, and results are summed.
    """

    def __init__(self, hidden_dim, k_hop=2):
        super().__init__()
        self.k_hop = k_hop
        self.projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(k_hop + 1)])

    def forward(self, node_features, adjacency_matrix):
        """Apply normalized adjacency propagation on node features.

        Args:
            node_features: [B, N, D] node feature tensor.
            adjacency_matrix: [N, N] or [B, N, N] adjacency matrix.

        Returns:
            [B, N, D] convolved features.
        """
        batch_size, num_nodes, _ = node_features.shape
        adjacency_matrix = expand_adjacency_batch(adjacency_matrix, batch_size).to(
            device=node_features.device, dtype=node_features.dtype
        )
        adjacency_norm = build_normalized_adjacency(adjacency_matrix)
        adjacency_power = torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        output = torch.zeros_like(node_features)
        for hop_index, projection in enumerate(self.projections):
            if hop_index > 0:
                adjacency_power = torch.bmm(adjacency_power, adjacency_norm)
            propagated = torch.bmm(adjacency_power, node_features)
            output = output + projection(propagated)
        return output


class AdaptiveGraphLearner(nn.Module):
    """Learn and update graph structure from node states during diffusion.

    Core formula: A_adapt = (1 - blend) × A_static + blend × A_dynamic.

    When fuzzy graph is enabled, dynamic adjacency is computed via Gaussian
    fuzzy membership functions over pairwise node similarity, producing a
    soft relational matrix instead of sharp softmax assignments.
    """

    def __init__(
            self,
            num_nodes,
            hidden_dim,
            static_adjacency,
            embed_dim=32,
            top_k=None,
            blend_init=0.5,
            fuzzy_enabled=False,
            fuzzy_num_sets=3,
            fuzzy_sigma_init=0.7,
    ):
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1.")
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.fuzzy_enabled = fuzzy_enabled

        static_adjacency = static_adjacency.to(dtype=torch.float32)
        if static_adjacency.dim() != 2 or static_adjacency.shape[0] != static_adjacency.shape[1]:
            raise ValueError("static_adjacency must be a square 2D tensor.")
        identity = torch.eye(static_adjacency.size(0), device=static_adjacency.device)
        static_adjacency = torch.relu(static_adjacency) + identity
        static_adjacency = self._row_normalize(static_adjacency)
        self.register_buffer("static_adjacency", static_adjacency)
        self.register_buffer("identity", identity)

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.feature_projection = nn.Linear(hidden_dim, embed_dim)
        self.time_projection = nn.Linear(hidden_dim, embed_dim)
        blend_init = float(min(max(blend_init, 1e-4), 1 - 1e-4))
        self.blend_logit = nn.Parameter(
            torch.tensor(math.log(blend_init / (1.0 - blend_init)), dtype=torch.float32)
        )

        if self.fuzzy_enabled:
            if fuzzy_num_sets < 2:
                raise ValueError("fuzzy_num_sets must be >= 2 when fuzzy graph is enabled.")
            if fuzzy_sigma_init <= 0:
                raise ValueError("fuzzy_sigma_init must be > 0 when fuzzy graph is enabled.")
            fuzzy_centers = torch.linspace(-1.0, 1.0, fuzzy_num_sets, dtype=torch.float32)
            self.fuzzy_centers = nn.Parameter(fuzzy_centers)
            self.fuzzy_log_sigmas = nn.Parameter(
                torch.full((fuzzy_num_sets,), math.log(fuzzy_sigma_init), dtype=torch.float32)
            )
            self.fuzzy_rule_logits = nn.Parameter(torch.zeros(fuzzy_num_sets, dtype=torch.float32))

    @staticmethod
    def _row_normalize(adjacency_matrix):
        return adjacency_matrix / adjacency_matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _fuzzy_similarity_to_adjacency(self, similarity):
        """Convert pairwise similarity to fuzzy relation strength in [0, 1].

        Uses learnable Gaussian membership functions over tanh-bounded similarity,
        weighted by learned rule weights.
        """
        bounded_similarity = torch.tanh(similarity).unsqueeze(-1)
        centers = self.fuzzy_centers.view(1, 1, 1, -1).type_as(similarity)
        sigmas = self.fuzzy_log_sigmas.exp().view(1, 1, 1, -1).type_as(similarity)
        gaussian_membership = torch.exp(-0.5 * ((bounded_similarity - centers) / sigmas.clamp_min(1e-4)) ** 2)
        rule_weights = torch.softmax(self.fuzzy_rule_logits, dim=0).view(1, 1, 1, -1).type_as(similarity)
        fuzzy_relation = (gaussian_membership * rule_weights).sum(dim=-1)
        return fuzzy_relation.clamp_min(1e-12)

    def forward(self, sequence_features, timestep_embedding=None):
        """Build adaptive adjacency from node features.

        Args:
            sequence_features: [B, T, N, D] or [B, N, D] node features.
            timestep_embedding: Optional [B, 1, D] time-conditioned embedding.

        Returns:
            [B, N, N] row-normalized adaptive adjacency matrix.
        """
        if sequence_features.dim() == 4:
            node_features = sequence_features.mean(dim=1)
        elif sequence_features.dim() == 3:
            node_features = sequence_features
        else:
            raise ValueError("sequence_features must be 3D or 4D tensor.")

        batch_size = node_features.size(0)
        node_representation = self.feature_projection(node_features) + self.node_embeddings.unsqueeze(0)
        if timestep_embedding is not None:
            node_representation = node_representation + self.time_projection(timestep_embedding).unsqueeze(1)
        node_representation = torch.tanh(node_representation)

        similarity = torch.matmul(node_representation, node_representation.transpose(1, 2))
        similarity = similarity / math.sqrt(self.embed_dim)
        if self.fuzzy_enabled:
            dynamic_adjacency = self._row_normalize(self._fuzzy_similarity_to_adjacency(similarity))
        else:
            dynamic_adjacency = torch.softmax(similarity, dim=-1)

        if self.top_k is not None and 0 < self.top_k < self.num_nodes:
            top_values, top_indices = torch.topk(dynamic_adjacency, k=self.top_k, dim=-1)
            sparse_dynamic = torch.zeros_like(dynamic_adjacency)
            sparse_dynamic.scatter_(-1, top_indices, top_values)
            dynamic_adjacency = self._row_normalize(sparse_dynamic)

        dynamic_adjacency = dynamic_adjacency + self.identity.to(dynamic_adjacency.dtype).unsqueeze(0)
        dynamic_adjacency = self._row_normalize(dynamic_adjacency)

        static_adjacency = self.static_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        static_adjacency = static_adjacency.to(dtype=dynamic_adjacency.dtype)

        blend = torch.sigmoid(self.blend_logit)
        adaptive_adjacency = (1.0 - blend) * static_adjacency + blend * dynamic_adjacency
        return self._row_normalize(adaptive_adjacency)
