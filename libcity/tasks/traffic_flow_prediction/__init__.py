"""Lazy exports for traffic_flow_prediction models."""
from typing import Dict
import importlib

__all__ = ["PDFormer"]

_exports: Dict[str, str] = {
    "PDFormer": "libcity.model.traffic_flow_prediction.PDFormer",
}


def __getattr__(name: str):
    if name in _exports:
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_exports.keys()))
