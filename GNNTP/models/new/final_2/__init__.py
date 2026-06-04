"""final_2 — Fuzzy Relational Reasoning System for traffic forecasting.

Enhanced from final_new with:
  - Fuzzy Semantic Closure: transitive relational reasoning via max-min composition.
  - Entropy-Driven Dynamic Graph: fuzzy entropy modulates relation strengths.

Local-Global Spatial Dual architecture:
  Spatial-Local:  FuzzyGCN (K-hop topology propagation)
  Spatial-Global: FRR (Fuzzy Region Routing, replaces spatial self-attn)
  Temporal:       Per-node Transformer
"""

from GNNTP.models.new.final_2.model import NewFuzzyCellAttention2

__all__ = ["NewFuzzyCellAttention2"]
