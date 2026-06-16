"""Fuzzy Cell Attention with MDI uncertainty gating (final_T2).

This variant accepts an optional per-node Membership Disagreement Interval
(MDI) vector δ and modulates region affinity by (1 - α·U_ij) where
U_ij = (δ_i + δ_j) / 2. The uncertainty only gates attention — it
never alters graph propagation.

The MDI signal δ comes from the multi-view fuzzy graph learner and
quantifies cross-view membership disagreement. High δ → the node's
fuzzy identity is disputed across views → attenuate its CellAttention
affinity to prevent unreliable structural information from dominating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyCellAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_cells: int = 8,
        use_hollow_kernel: bool = True,
        membership_temperature: float = 1.0,
        cell_blend_init: float = 0.3,
    ):
        super().__init__()
        if num_cells < 2:
            raise ValueError("num_cells must be >= 2.")
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim

        self.centers = nn.Parameter(
            torch.randn(num_cells, hidden_dim) * 0.1
        )

        self.log_temperature = nn.Parameter(
            torch.tensor(membership_temperature).log()
        )

        self.cell_transform = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        self.cell_blend = nn.Parameter(torch.tensor(cell_blend_init))

        self.use_hollow_kernel = use_hollow_kernel
        if use_hollow_kernel:
            self.log_sigma_excite = nn.Parameter(torch.tensor(2.0).log())
            self.log_sigma_inhibit = nn.Parameter(torch.tensor(0.5).log())
            self.inhibit_weight = nn.Parameter(torch.tensor(0.5))

        # alpha controls how strongly uncertainty attenuates attention
        self.uncertainty_alpha = nn.Parameter(torch.tensor(0.5))

    def _compute_membership(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, self.centers)
        tau = self.log_temperature.exp().clamp(min=0.05)
        return F.softmax(-dist / tau, dim=-1)

    def _hollow_kernel(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        s1 = self.log_sigma_excite.exp().clamp(min=0.1)
        s2 = self.log_sigma_inhibit.exp().clamp(min=0.05)
        lam = self.inhibit_weight.sigmoid()

        gauss_excite = torch.exp(-dist_matrix ** 2 / (2 * s1 ** 2))
        gauss_inhibit = torch.exp(-dist_matrix ** 2 / (2 * s2 ** 2))

        kernel = gauss_excite - lam * gauss_inhibit
        return kernel.clamp(min=0.0)

    def get_stability_metrics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._compute_membership(x)
        H = -(u * (u + 1e-8).log()).sum(dim=-1)
        top2 = u.topk(2, dim=-1).values
        S = top2[:, 0] - top2[:, 1]
        return H, S

    def forward(
        self,
        x: torch.Tensor,
        return_stability: bool = False,
        node_uncertainty: torch.Tensor | None = None,
    ):
        if x.dim() != 2:
            raise ValueError(
                f"FuzzyCellAttention expects [N, D] input, got shape {x.shape}"
            )

        u = self._compute_membership(x)  # [N, K_c]
        u_norm = F.normalize(u, p=2, dim=-1, eps=1e-8)
        region_affinity = u_norm @ u_norm.T  # [N, N]

        if self.use_hollow_kernel:
            dist = torch.cdist(x, x)
            hollow = self._hollow_kernel(dist)
            region_affinity = region_affinity * hollow

        # If provided, modulate by MDI mask U_ij = (δ_i + δ_j) / 2
        if node_uncertainty is not None:
            fou = node_uncertainty.view(-1).to(device=region_affinity.device, dtype=region_affinity.dtype)
            U_ij = (fou.unsqueeze(0) + fou.unsqueeze(1)) / 2.0
            alpha = self.uncertainty_alpha.sigmoid()
            region_affinity = region_affinity * (1.0 - alpha * U_ij)

        region_affinity = region_affinity / (
            region_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )

        cell_output = region_affinity @ self.cell_transform(x)
        output = self.output_projection(cell_output)

        if return_stability:
            H = -(u * (u + 1e-8).log()).sum(dim=-1)
            top2 = u.topk(2, dim=-1).values
            S = top2[:, 0] - top2[:, 1]
            return output, (H, S)
        return output


class CellAttentionPool:
    @staticmethod
    def mean_pool(sequence_features: torch.Tensor) -> torch.Tensor:
        return sequence_features.mean(dim=(0, 1))

    @staticmethod
    def batch_mean_pool(sequence_features: torch.Tensor) -> torch.Tensor:
        return sequence_features.mean(dim=1)

    @staticmethod
    def time_last(sequence_features: torch.Tensor) -> torch.Tensor:
        return sequence_features[:, -1].mean(dim=0)

