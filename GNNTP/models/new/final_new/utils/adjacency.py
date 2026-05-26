"""Adjacency matrix helper utilities."""

import torch


def build_normalized_adjacency(adjacency_matrix, add_self_loop=True):
    """Build symmetric normalized adjacency matrix D^{-1/2} A D^{-1/2}."""
    if adjacency_matrix.dim() == 2:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
    if add_self_loop:
        num_nodes = adjacency_matrix.size(-1)
        identity = torch.eye(num_nodes, device=adjacency_matrix.device, dtype=adjacency_matrix.dtype).unsqueeze(0)
        adjacency_matrix = adjacency_matrix + identity
    degree = adjacency_matrix.sum(dim=-1)
    degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
    normalized_adjacency = torch.bmm(torch.bmm(degree_inv_sqrt, adjacency_matrix), degree_inv_sqrt)
    return normalized_adjacency


def expand_adjacency_batch(adjacency_matrix, target_batch_size):
    """Expand or tile adjacency matrix batch dimension to a target batch size."""
    if adjacency_matrix.dim() == 2:
        return adjacency_matrix.unsqueeze(0).expand(target_batch_size, -1, -1)
    if adjacency_matrix.dim() == 3:
        if adjacency_matrix.size(0) == target_batch_size:
            return adjacency_matrix
        if adjacency_matrix.size(0) == 1:
            return adjacency_matrix.expand(target_batch_size, -1, -1)
        if target_batch_size % adjacency_matrix.size(0) != 0:
            raise ValueError(
                f"adjacency batch size {adjacency_matrix.size(0)} is incompatible with "
                f"target batch size {target_batch_size}."
            )
        repeat_factor = target_batch_size // adjacency_matrix.size(0)
        return adjacency_matrix.repeat_interleave(repeat_factor, dim=0)
    raise ValueError("adjacency_matrix must be 2D or 3D tensor.")


def compute_hop_distance(adjacency, max_hops=20):
    """Iterative hop distance propagation.  O(K·N²), <1s for N≤1000.

    For each hop k, compute which node pairs become reachable via
    k-step random walk on the binary adjacency, and assign distance=k.

    Traffic graphs do not need exact Euclidean shortest paths;
    hop distance is sufficient and is a proper metric.

    Args:
        adjacency: [N, N] binary adjacency (edges > 0).
        max_hops: distance for unreachable pairs.

    Returns:
        [N, N] float distance matrix.
    """
    N = adjacency.shape[0]
    dist = torch.full((N, N), float(max_hops), dtype=torch.float32)
    dist.fill_diagonal_(0.0)
    mask_edges = (adjacency > 0) & ~torch.eye(N, dtype=torch.bool, device=adjacency.device)
    dist[mask_edges] = 1.0

    A_k = adjacency.float()
    for hop in range(2, max_hops + 1):
        A_k = (A_k @ adjacency.float()).clamp(min=0, max=1)
        mask = (A_k > 0) & (dist == max_hops)
        if not mask.any():
            break
        dist[mask] = float(hop)
    return dist
