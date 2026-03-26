"""Lazy exports for traffic_speed_prediction models.

This module exposes symbols like ``DCRNN`` and ``STGCN`` but only imports
their implementation modules when the attributes are actually accessed.
This avoids heavy imports at package import time.
"""
from typing import Dict, TYPE_CHECKING
import importlib

__all__ = ["DCRNN", "STGCN"]

if TYPE_CHECKING:
    # type: ignore - for static type checkers only
    from libcity.tasks.traffic_speed_prediction.DCRNN import DCRNN  # noqa: F401
    from libcity.tasks.traffic_speed_prediction.STGCN import STGCN  # noqa: F401

# map exported attribute -> module path that defines it
_exports: Dict[str, str] = {
    "DCRNN": "libcity.tasks.traffic_speed_prediction.DCRNN.model",
    "STGCN": "libcity.tasks.traffic_speed_prediction.STGCN.model",
}


def __getattr__(name: str):
    if name in _exports:
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_exports.keys()))
