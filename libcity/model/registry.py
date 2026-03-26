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


def get_model_class(task, model_name):
    # Always attempt to discover model modules under libcity.model and the task subpackage.
    try:
        _import_submodules('libcity.model')
        _import_submodules(f'libcity.model.{task}')
    except Exception:
        # Discovery errors are logged in _import_submodules; continue to registry lookup below
        pass
    task_registry = TASK_MODEL_REGISTRY.get(task)
    if task_registry is None:
        raise AttributeError("task is not found")
    return task_registry.get(model_name)
