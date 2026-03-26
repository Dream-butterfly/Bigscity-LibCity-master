"""Lazy exports for evaluator implementations."""
from typing import Dict, TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from libcity.evaluator.traffic_state_evaluator import TrafficStateEvaluator  # noqa: F401

__all__ = ["TrafficStateEvaluator"]

_exports: Dict[str, str] = {
    "TrafficStateEvaluator": "libcity.evaluator.traffic_state_evaluator",
}


def __getattr__(name: str):
    if name in _exports:
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_exports.keys()))
