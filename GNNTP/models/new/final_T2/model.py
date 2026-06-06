"""Registration shim for final_T2 — re-export implementation from the copied folder.

This module keeps a thin import shim so the locator can reference
GNNTP.models.new.final_T2.model:NewFuzzyCellAttention while the actual
implementation lives under new_fuzzy_cellattention/final_T2.
"""

from GNNTP.models.new.new_fuzzy_cellattention.final_T2.model import NewFuzzyCellAttention

__all__ = ["NewFuzzyCellAttention"]

