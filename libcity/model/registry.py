from libcity.core.registry import Registry
import pkgutil
import importlib
import logging
from typing import Set


TASK_MODEL_REGISTRY = {
    "traffic_state_pred": Registry("traffic_state_pred.model"),
}
_BOOTSTRAPPED: Set[str] = set()


def register_model(task, name=None):
    if task not in TASK_MODEL_REGISTRY:
        raise AttributeError("task is not found")
    return TASK_MODEL_REGISTRY[task].register(name=name)


def _import_submodules(package_name: str):
    logger = logging.getLogger(__name__)
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        logger.warning("Failed to import package %s: %s", package_name, e)
        return
    if not hasattr(pkg, '__path__'):
        return
    prefix = pkg.__name__ + '.'
    for finder, full_name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.warning("Failed to import %s: %s", full_name, e)


def _bootstrap_traffic_speed_models():
    if "traffic_state_pred" in _BOOTSTRAPPED:
        return
    # Import the top-level model package and the task-specific subpackage so
    # model modules placed under `libcity.model.<task>` are discovered.
    _import_submodules('libcity.model')
    task_pkg = 'libcity.model.traffic_state_pred'
    _import_submodules(task_pkg)
    # Also try to import the traffic_state_pred package explicitly (best-effort)
    try:
        importlib.import_module(task_pkg)
    except Exception:
        # It's okay if the explicit import fails; other submodules may still register models.
        pass

    _BOOTSTRAPPED.add("traffic_state_pred")


# Backwards-compatible mapping for bootstrapping specific models per task.
# New models that register themselves on import will be discovered by the
# bootstrappers which import submodules under `libcity.model`.
MODEL_BOOTSTRAPPERS = {
    "traffic_state_pred": {
        "DCRNN": _bootstrap_traffic_speed_models,
        "STGCN": _bootstrap_traffic_speed_models,
        "PDFormer": _bootstrap_traffic_speed_models,
        "NEW_MODEL": _bootstrap_traffic_speed_models,
    }
}


def get_model_class(task, model_name):
    task_bootstrappers = MODEL_BOOTSTRAPPERS.get(task, {})
    bootstrap = task_bootstrappers.get(model_name)
    if bootstrap is not None:
        # If there's a specific bootstrap function for this model, call it
        bootstrap()
    else:
        # Fallback behavior: try to call a generic bootstrapper for the task if available,
        # otherwise attempt to import submodules under the task package to discover registrations.
        try:
            # example: for task 'traffic_state_pred' try calling _bootstrap_traffic_state_pred_models
            generic_name = f"_bootstrap_{task}_models"
            generic_bootstrap = globals().get(generic_name)
            if callable(generic_bootstrap):
                generic_bootstrap()
            else:
                # last-resort: import submodules under libcity.model and libcity.model.<task>
                _import_submodules('libcity.model')
                _import_submodules(f'libcity.model.{task}')
        except Exception:
            # Do not fail here; registry.get below will raise a meaningful error if model is missing
            pass
    task_registry = TASK_MODEL_REGISTRY.get(task)
    if task_registry is None:
        raise AttributeError("task is not found")
    return task_registry.get(model_name)
