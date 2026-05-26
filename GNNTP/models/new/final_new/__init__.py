"""final_new — Fuzzy Region Transformer for traffic forecasting.

Local-Global Spatial Dual architecture:
  Spatial-Local:  FuzzyGCN (K-hop topology propagation)
  Spatial-Global: FRR (Fuzzy Region Routing, replaces spatial self-attn)
  Temporal:       Per-node Transformer
"""

from GNNTP.models.new.final_new.model import NewFuzzyCellAttention

__all__ = ["NewFuzzyCellAttention"]
