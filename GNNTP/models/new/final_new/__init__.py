"""new_fuzzy_cellattention — Fuzzy Cell Attention model package.

FuzDiff + Fuzzy Cell Attention:
  H' = λ₁·FuzzyGCN(H, R) + λ₂·CellAttention(H, C)
"""

from GNNTP.models.new.new_fuzzy_cellattention.model import NewFuzzyCellAttention

__all__ = ["NewFuzzyCellAttention"]
