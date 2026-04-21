"""Attention-related tensor reshape utilities."""


def apply_temporal_attention(sequence_features, attention_module):
    """Apply attention along time dimension for each node independently."""
    batch_size, time_steps, num_nodes, hidden_dim = sequence_features.shape
    reshaped_features = sequence_features.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, hidden_dim)
    attended_features = attention_module(reshaped_features)
    return attended_features.reshape(batch_size, num_nodes, time_steps, hidden_dim).permute(0, 2, 1, 3)



def apply_node_temporal_cross_attention(query_sequence, context_sequence, attention_module):
    """Apply cross-attention over history time for each node independently."""
    if query_sequence.dim() != 4 or context_sequence.dim() != 4:
        raise ValueError("query_sequence and context_sequence must be 4D tensors [B, T, N, D].")
    if query_sequence.shape[0] != context_sequence.shape[0]:
        raise ValueError("Batch size mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[2] != context_sequence.shape[2]:
        raise ValueError("Node count mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[3] != context_sequence.shape[3]:
        raise ValueError("Hidden dimension mismatch between query_sequence and context_sequence.")
    batch_size, query_steps, num_nodes, hidden_dim = query_sequence.shape
    context_steps = context_sequence.shape[1]

    query = query_sequence.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, query_steps, hidden_dim)
    context = context_sequence.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, context_steps, hidden_dim)
    attended_output = attention_module(query, context=context)
    return attended_output.reshape(batch_size, num_nodes, query_steps, hidden_dim).permute(0, 2, 1, 3)



def apply_spatiotemporal_attention(query_sequence, attention_module, context_sequence=None):
    """Apply attention on flattened spatio-temporal tokens (self or cross)."""
    if context_sequence is None:
        context_sequence = query_sequence
    if query_sequence.dim() != 4 or context_sequence.dim() != 4:
        raise ValueError("query_sequence and context_sequence must be 4D tensors [B, T, N, D].")
    if query_sequence.shape[0] != context_sequence.shape[0]:
        raise ValueError("Batch size mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[2] != context_sequence.shape[2]:
        raise ValueError("Node count mismatch between query_sequence and context_sequence.")
    if query_sequence.shape[3] != context_sequence.shape[3]:
        raise ValueError("Hidden dimension mismatch between query_sequence and context_sequence.")

    batch_size, query_steps, num_nodes, hidden_dim = query_sequence.shape
    context_steps = context_sequence.shape[1]
    query_tokens = query_sequence.reshape(batch_size, query_steps * num_nodes, hidden_dim)
    context_tokens = context_sequence.reshape(batch_size, context_steps * num_nodes, hidden_dim)
    attended_tokens = attention_module(query_tokens, context=context_tokens)
    return attended_tokens.reshape(batch_size, query_steps, num_nodes, hidden_dim)
