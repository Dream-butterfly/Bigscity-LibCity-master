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
