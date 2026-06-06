"""final_T2: Type-2 aware variant of new_fuzzy_cellattention.

This package mirrors the original implementation but adds a stable
Type-2 uncertainty estimator (FOU) and uncertainty-modulated
cell-attention gating. The FOU is only used for gating/attention and
not for graph propagation/closure (closure applied only on mid graph).
"""

from .model import NewFuzzyCellAttention

__all__ = ["NewFuzzyCellAttention"]

