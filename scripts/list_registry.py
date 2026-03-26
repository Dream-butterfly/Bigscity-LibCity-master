"""
List registered models and datasets discovered by the registry bootstrap.

Usage:
    python scripts/list_registry.py

This prints registered dataset classes and model classes (per task).
"""
import pprint

from libcity import model as model_pkg
from libcity import data as data_pkg

# Import registries and bootstrappers from tasks-based registry
from libcity.tasks import registry as model_registry
from libcity.data import registry as data_registry
from libcity.tasks import registry as executor_registry
from libcity.tasks import registry as evaluator_registry
import importlib


def list_datasets():
    # Trigger dataset bootstrapping for known bootstrappers
    # Try to recursively import dataset modules to trigger registrations
    try:
        if hasattr(data_registry, '_import_submodules'):
            data_registry._import_submodules('libcity.data.dataset')
        else:
            importlib.import_module('libcity.data.dataset')
    except Exception:
        pass

    items = {}
    try:
        items = data_registry.DATASET_REGISTRY.items()
    except Exception:
        items = {}
    print("Registered datasets:")
    pprint.pprint(items)


def list_models():
    # Trigger recursive import of model modules to ensure registrations
    try:
        # import all model modules under libcity.tasks to trigger registrations
        if hasattr(model_registry, '_import_submodules'):
            model_registry._import_submodules('libcity.tasks')
        else:
            importlib.import_module('libcity.tasks')
    except Exception:
        pass
    # Also try task-level packages
    try:
        for task in getattr(model_registry, 'TASK_MODEL_REGISTRY', {}).keys():
            try:
                model_registry._import_submodules(f'libcity.tasks.{task}')
            except Exception:
                pass
    except Exception:
        pass

    print("\nRegistered models (by task):")
    try:
        for task, reg in getattr(model_registry, 'TASK_MODEL_REGISTRY', {}).items():
            try:
                items = reg.items()
            except Exception:
                items = {}
            print(f"- Task: {task}")
            pprint.pprint(items)
    except Exception:
        pass


def list_executors_and_evaluators():
    try:
        if hasattr(executor_registry, '_import_submodules'):
            executor_registry._import_submodules('libcity.tasks')
    except Exception:
        pass
    try:
        if hasattr(evaluator_registry, '_import_submodules'):
            evaluator_registry._import_submodules('libcity.tasks')
    except Exception:
        pass
    print('\nRegistered executors:')
    try:
        pprint.pprint(executor_registry.EXECUTOR_REGISTRY.items())
    except Exception:
        pass
    print('\nRegistered evaluators:')
    try:
        pprint.pprint(evaluator_registry.EVALUATOR_REGISTRY.items())
    except Exception:
        pass


if __name__ == '__main__':
    list_datasets()
    list_models()

