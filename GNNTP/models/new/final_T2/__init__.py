"""final_T2: multi-view fuzzy graph transformer (MVF-STGFormer).

This package implements the Multi-View Fuzzy Relational Graph Learner:
three structurally independent fuzzy perspectives (Low/Mid/High) on the
traffic sensor graph. The Membership Disagreement Interval (MDI) — the width
of the cross-view membership envelope — serves as a per-node uncertainty
signal for attention gating. MDI is only used for gating/attention and not
for graph propagation/closure (closure applied only on the mid/expected graph).
"""

from .model import NewFuzzyCellAttention

__all__ = ["NewFuzzyCellAttention"]

