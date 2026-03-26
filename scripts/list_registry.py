"""
List registered models and datasets discovered by the registry bootstrap.

Usage:
    python scripts/list_registry.py

This prints registered dataset classes and model classes (per task).
"""
import pprint

from libcity import model as model_pkg
from libcity import data as data_pkg

# Import registries and bootstrappers
from libcity.model import registry as model_registry
from libcity.data import registry as data_registry


def list_datasets():
    # Trigger dataset bootstrapping for known bootstrappers
    try:
        for name, fn in getattr(data_registry, 'DATASET_BOOTSTRAPPERS', {}).items():
            try:
                fn()
            except Exception:
                # best-effort
                pass
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
    # Trigger model bootstrapping for known bootstrappers
    try:
        for task, mapping in getattr(model_registry, 'MODEL_BOOTSTRAPPERS', {}).items():
            for model_name, fn in mapping.items():
                try:
                    fn()
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


if __name__ == '__main__':
    list_datasets()
    list_models()

