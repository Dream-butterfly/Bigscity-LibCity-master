"""
Deprecated compatibility shim for hyper_tuning.

This module used to provide Optuna-based hyper-parameter tuning. The project
no longer exposes a top-level run_hyper entrypoint. Keep a shim here to
emit a deprecation warning and forward callers to the new utilities in
`libcity.utils.tune` (if available).
"""
import warnings

warnings.warn(
    "libcity.executor.hyper_tuning is deprecated and will be removed. "
    "Use libcity.utils.tune instead.",
    DeprecationWarning,
)

try:
    # try to import convenience helpers from the new location
    from libcity.utils.tune import *  # noqa: F401,F403
except Exception:
    # If the target module does not exist, provide safe no-op placeholders
    def tune_study(*args, **kwargs):
        raise RuntimeError("Hyper tuning utility not available in this installation.")

    __all__ = ["tune_study"]

