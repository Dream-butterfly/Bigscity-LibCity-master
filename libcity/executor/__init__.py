"""Lazy exports for executor implementations."""
from typing import Dict, TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    # type: ignore - for static analyzers only
    from libcity.executor.dcrnn_executor import DCRNNExecutor  # noqa: F401
    from libcity.executor.pdformer_executor import PDFormerExecutor  # noqa: F401
    from libcity.executor.traffic_state_executor import TrafficStateExecutor  # noqa: F401

__all__ = ["TrafficStateExecutor", "DCRNNExecutor", "PDFormerExecutor"]

_exports: Dict[str, str] = {
    "DCRNNExecutor": "libcity.executor.dcrnn_executor",
    "PDFormerExecutor": "libcity.executor.pdformer_executor",
    "TrafficStateExecutor": "libcity.executor.traffic_state_executor",
}


def __getattr__(name: str):
    if name in _exports:
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_exports.keys()))
