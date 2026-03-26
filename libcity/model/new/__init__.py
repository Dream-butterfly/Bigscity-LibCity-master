"""Lazy exports for new-model package."""
from typing import Dict, TYPE_CHECKING
import importlib

# Help static analyzers know NEW_MODEL exists when checking __all__
if TYPE_CHECKING:
    # type: ignore - for static type checkers only
    from libcity.model.new.new_model import NEW_MODEL  # noqa: F401

__all__ = ["NEW_MODEL"]

_exports: Dict[str, str] = {
    "NEW_MODEL": "libcity.model.new.new_model",
}


def __getattr__(name: str):
    if name in _exports:
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_exports.keys()))
