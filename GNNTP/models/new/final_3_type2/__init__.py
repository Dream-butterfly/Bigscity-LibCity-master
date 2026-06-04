"""final_3_type2 — Interval Type-2 Fuzzy Relational Reasoning System.

Enhanced from final_2 with Type-2 Fuzzy Sets (Route 2):
  - Interval membership [μ_low, μ_high] instead of single μ
  - Interval relation [R_low, R_high] with Footprint of Uncertainty (FOU)
  - FOU-gated effective graph combination (mid/high/low/fou_gated)
  - Type-2 aware entropy (FOU-amplified)
  - FOU-modulated dynamic graph and FIR

Local-Global Spatial Dual architecture:
  Spatial-Local:  FuzzyGCN (K-hop topology propagation)
  Spatial-Global: FRR (Fuzzy Region Routing)
  Temporal:       Per-node Transformer
"""

from GNNTP.models.new.final_3_type2.model import NewFuzzyCellAttention3_Type2

__all__ = ["NewFuzzyCellAttention3_Type2"]
