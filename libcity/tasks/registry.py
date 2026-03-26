"""Central registries for models, executors and evaluators.

This module provides decorators and lookup helpers that match the old
`libcity.model.registry` / `libcity.executor.registry` / `libcity.evaluator.registry`
APIs but live under `libcity.tasks` so the new per-model-folder layout can
register classes at import-time.
"""
from typing import Dict
import pkgutil
import importlib
import logging
import os

from libcity.core.registry import Registry

LOGGER = logging.getLogger(__name__)

# model registries are organized per-task (task -> Registry)
TASK_MODEL_REGISTRY: Dict[str, Registry] = {}


def register_model(task_name: str):
    def decorator(cls):
        if task_name not in TASK_MODEL_REGISTRY:
            TASK_MODEL_REGISTRY[task_name] = Registry(f"model:{task_name}")
        # use the Registry.register decorator to keep behavior consistent
        TASK_MODEL_REGISTRY[task_name].register(cls)
        return cls

    return decorator


def get_model_class(task_name: str, model_name: str):
    # try direct lookup first
    reg = TASK_MODEL_REGISTRY.get(task_name)
    if reg is not None:
        try:
            return reg.get(model_name)
        except Exception:
            pass

    # fallback: import all modules under libcity.tasks to trigger registrations
    _import_submodules('libcity.tasks')

    reg = TASK_MODEL_REGISTRY.get(task_name)
    if reg is None:
        # last-resort: scan the libcity/tasks tree for directories matching model_name
        try:
            _import_model_modules_by_name(model_name)
        except Exception:
            pass
        reg = TASK_MODEL_REGISTRY.get(task_name)
        if reg is None:
            raise AttributeError(f"No models registered for task {task_name}")
    return reg.get(model_name)


# executor / evaluator are single registries (flat)
EXECUTOR_REGISTRY = Registry('executor')
EVALUATOR_REGISTRY = Registry('evaluator')


def register_executor(name=None):
    return EXECUTOR_REGISTRY.register(name=name)


def register_evaluator(name=None):
    return EVALUATOR_REGISTRY.register(name=name)


def get_executor_class(name: str):
    # try lookup
    try:
        return EXECUTOR_REGISTRY.get(name)
    except Exception:
        # import tasks to bootstrap
        _import_submodules('libcity.tasks')
        try:
            # try importing executor modules explicitly (scan for executor.py)
            _import_executor_and_evaluator_modules()
        except Exception:
            pass
        return EXECUTOR_REGISTRY.get(name)


def get_evaluator_class(name: str):
    try:
        return EVALUATOR_REGISTRY.get(name)
    except Exception:
        _import_submodules('libcity.tasks')
        try:
            _import_executor_and_evaluator_modules()
        except Exception:
            pass
        return EVALUATOR_REGISTRY.get(name)


def _import_submodules(package_name: str):
    """Import all submodules under the given package name to trigger registration."""
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        LOGGER.warning("Failed to import package %s: %s", package_name, e)
        return

    if not hasattr(pkg, '__path__'):
        return
    prefix = pkg.__name__ + '.'
    for finder, full_name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
        try:
            importlib.import_module(full_name)
        except Exception as e:
            LOGGER.warning("Failed to import %s: %s", full_name, e)
            continue
        # If the discovered name is itself a package, recurse to import its children
        if ispkg:
            try:
                _import_submodules(full_name)
            except Exception:
                # recursion errors already logged in _import_submodules
                pass


def _import_model_modules_by_name(model_name: str):
    """Scan the libcity/tasks tree for directories named after model_name and import their .model modules."""
    tasks_dir = os.path.dirname(__file__)  # libcity/tasks
    for root, dirs, files in os.walk(tasks_dir):
        for d in list(dirs):
            if d == model_name:
                model_path = os.path.join(root, d, 'model.py')
                if os.path.exists(model_path):
                    rel = os.path.relpath(model_path, tasks_dir)
                    module_name = 'libcity.tasks.' + rel.replace(os.sep, '.')[:-3]
                    try:
                        importlib.import_module(module_name)
                        LOGGER.debug("Imported model module by path: %s", module_name)
                    except Exception as e:
                        LOGGER.warning("Failed to import model module %s: %s", module_name, e)


def _import_executor_and_evaluator_modules():
    """Import any executor.py / evaluator.py modules under libcity/tasks so their
    registration decorators are executed."""
    tasks_dir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(tasks_dir):
        for fname in ('executor.py', 'evaluator.py'):
            if fname in files:
                full_path = os.path.join(root, fname)
                rel = os.path.relpath(full_path, tasks_dir)
                module_name = 'libcity.tasks.' + rel.replace(os.sep, '.')[:-3]
                try:
                    importlib.import_module(module_name)
                    LOGGER.debug("Imported module: %s", module_name)
                except Exception as e:
                    LOGGER.warning("Failed to import %s: %s", module_name, e)


__all__ = [
    'TASK_MODEL_REGISTRY',
    'register_model',
    'get_model_class',
    'EXECUTOR_REGISTRY',
    'EVALUATOR_REGISTRY',
    'register_executor',
    'register_evaluator',
    'get_executor_class',
    'get_evaluator_class',
    '_import_submodules',
]
