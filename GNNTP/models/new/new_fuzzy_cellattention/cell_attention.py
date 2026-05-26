"""模糊胞型注意力（Fuzzy Cell Attention）模块。

节点→模糊区域→节点的注意力传播，复杂度 O(NK) 替代 O(N²)。

核心思路:
  1. 学习 K 个注意力中心（prototypes / fuzzy anchors）
  2. 节点对中心的模糊归属 u_{ik} = softmax(-d(x_i, c_k)/τ)
  3. 区域内节点注意力传播（region_affinity @ x）
  4. [可选] 空心注意力核（Mexican Hat / lateral inhibition）

稳定性指标:
  - H(i) = -Σ_k u_k log u_k        模糊胞熵（高 → 边界节点）
  - S(i) = u_{(1)} - u_{(2)}        归属稳定度（低 → 易切换归属）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyCellAttention(nn.Module):
    """模糊胞型注意力：节点 → 区域 → 节点 的传播。

    在每个 Encoder/Decoder Block 内部，与图卷积并行叠加：
        H' = λ₁ · GCN(H) + λ₂ · CellAttention(H)

    不替代图卷积，而是补充跨区域的远距离信息路由。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_cells: int = 8,
        use_hollow_kernel: bool = True,
        membership_temperature: float = 1.0,
        cell_blend_init: float = 0.3,
    ):
        """
        Args:
            hidden_dim: 节点特征维度。
            num_cells: 注意力中心/原型数量 K_c。
            use_hollow_kernel: 是否启用空心注意力核（Mexican hat）。
            membership_temperature: 模糊归属的温度参数 τ（可学习）。
            cell_blend_init: 胞型注意力输出的初始混合权重 λ₂。
        """
        super().__init__()
        if num_cells < 2:
            raise ValueError("num_cells must be >= 2.")
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim

        # ── 注意力中心（prototypes） ──
        self.centers = nn.Parameter(
            torch.randn(num_cells, hidden_dim) * 0.1
        )

        # ── 温度参数（模糊程度的控制） ──
        self.log_temperature = nn.Parameter(
            torch.tensor(membership_temperature).log()
        )

        # ── 区域内变换 ──
        self.cell_transform = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # ── 可学习混合权重 λ₂ ──
        self.cell_blend = nn.Parameter(torch.tensor(cell_blend_init))

        # ── 空心注意力核 ──
        self.use_hollow_kernel = use_hollow_kernel
        if use_hollow_kernel:
            # 宽高斯（兴奋）σ₁ > 窄高斯（抑制）σ₂
            self.log_sigma_excite = nn.Parameter(torch.tensor(2.0).log())   # σ₁
            self.log_sigma_inhibit = nn.Parameter(torch.tensor(0.5).log())  # σ₂
            self.inhibit_weight = nn.Parameter(torch.tensor(0.5))           # λ

    # ── 模糊归属 ──────────────────────────────────────────────

    def _compute_membership(self, x: torch.Tensor) -> torch.Tensor:
        """计算节点到各注意力中心的模糊归属。

        Args:
            x: [N, D] 节点特征。

        Returns:
            u: [N, K] 模糊归属矩阵，每行求和为 1。
        """
        # L2 距离: [N, K]
        dist = torch.cdist(x, self.centers)
        tau = self.log_temperature.exp().clamp(min=0.05)
        return F.softmax(-dist / tau, dim=-1)

    # ── 空心核 ──────────────────────────────────────────────────

    def _hollow_kernel(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """空心注意力核: K(r) = G(r; σ₁) - λ · G(r; σ₂)。

        Mexican Hat / Difference of Gaussians:
        中心附近注意力被抑制，中远距离被增强。

        Args:
            dist_matrix: [N, N] 节点间特征距离。

        Returns:
            kernel: [N, N]，clamped to [0, +∞)。
        """
        s1 = self.log_sigma_excite.exp().clamp(min=0.1)
        s2 = self.log_sigma_inhibit.exp().clamp(min=0.05)
        lam = self.inhibit_weight.sigmoid()  # λ ∈ (0, 1)

        gauss_excite = torch.exp(-dist_matrix ** 2 / (2 * s1 ** 2))
        gauss_inhibit = torch.exp(-dist_matrix ** 2 / (2 * s2 ** 2))

        kernel = gauss_excite - lam * gauss_inhibit
        return kernel.clamp(min=0.0)

    # ── 稳定性指标 ──────────────────────────────────────────────

    def get_stability_metrics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算模糊胞型稳定性双指标。

        Args:
            x: [N, D] 节点特征。

        Returns:
            H: [N] 模糊胞熵 — 高值 → 边界节点 / 归属不确定。
            S: [N] 归属稳定度 — 低值 → 归属易翻转。
        """
        u = self._compute_membership(x)  # [N, K]
        # 模糊胞熵: H(i) = -Σ_k u_k log u_k
        H = -(u * (u + 1e-8).log()).sum(dim=-1)
        # 归属稳定度: S(i) = u_{(1)} - u_{(2)}
        top2 = u.topk(2, dim=-1).values  # [N, 2]
        S = top2[:, 0] - top2[:, 1]
        return H, S

    # ── 前向传播 ────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_stability: bool = False,
    ):
        """模糊胞型注意力前向传播。

        Args:
            x: [N, D] 节点特征（通常是时间+批次平均后的全局表示）。
            return_stability: 是否返回稳定性指标。

        Returns:
            output: [N, D] 经胞型注意力传播后的节点特征。
            (H, S): 可选，模糊胞熵和归属稳定度。
        """
        if x.dim() != 2:
            raise ValueError(
                f"FuzzyCellAttention expects [N, D] input, got shape {x.shape}"
            )

        # Step 1: 模糊归属
        u = self._compute_membership(x)  # [N, K_c]

        # Step 2: 区域亲和度 = 归属向量的余弦相似度
        u_norm = F.normalize(u, p=2, dim=-1, eps=1e-8)   # [N, K_c]
        region_affinity = u_norm @ u_norm.T                 # [N, N]

        # Step 3: [可选] 空心核调制
        if self.use_hollow_kernel:
            dist = torch.cdist(x, x)                        # [N, N]
            hollow = self._hollow_kernel(dist)
            region_affinity = region_affinity * hollow

        # Step 4: 行归一化 → 注意力权重
        region_affinity = region_affinity / (
            region_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )

        # Step 5: 区域内传播
        cell_output = region_affinity @ self.cell_transform(x)  # [N, D]
        output = self.output_projection(cell_output)

        if return_stability:
            H = -(u * (u + 1e-8).log()).sum(dim=-1)
            top2 = u.topk(2, dim=-1).values
            S = top2[:, 0] - top2[:, 1]
            return output, (H, S)
        return output


class CellAttentionPool:
    """将 [B, T, N, D] 特征池化为全局 [N, D] 的工具函数集合。

    在 Encoder/Decoder Block 中调用 FuzzyCellAttention 前，
    需将序列特征池化为全局节点表示。本类提供不同的池化策略。
    """

    @staticmethod
    def mean_pool(sequence_features: torch.Tensor) -> torch.Tensor:
        """全局均值池化: [B, T, N, D] → mean(B, T) → [N, D]"""
        return sequence_features.mean(dim=(0, 1))

    @staticmethod
    def batch_mean_pool(sequence_features: torch.Tensor) -> torch.Tensor:
        """批次均值池化: [B, T, N, D] → mean(T) → [B, N, D]"""
        return sequence_features.mean(dim=1)

    @staticmethod
    def time_last(sequence_features: torch.Tensor) -> torch.Tensor:
        """取最后一个时间步 + 批次均值: [B, T, N, D] → [:, -1] → mean(B) → [N, D]"""
        return sequence_features[:, -1].mean(dim=0)
