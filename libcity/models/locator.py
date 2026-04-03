import importlib
import importlib.util
import json
import os
from functools import lru_cache


_MODELS_ROOT = os.path.dirname(__file__)
_MANIFEST_NAME = "manifest.json"


@lru_cache(maxsize=1)
def _load_manifest_index():
    manifest_index = []
    for root, _, files in os.walk(_MODELS_ROOT):
        if _MANIFEST_NAME not in files:
            continue
        manifest_path = os.path.join(root, _MANIFEST_NAME)
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            metadata = json.load(manifest_file)
        metadata["directory"] = root
        metadata["package"] = "libcity.models." + os.path.relpath(root, _MODELS_ROOT).replace(os.sep, ".")
        manifest_index.append(metadata)
    return tuple(manifest_index)


def _iter_model_manifests():
    return _load_manifest_index()


def get_model_metadata(task, model_name):
    for metadata in _iter_model_manifests():
        if metadata.get("model") != model_name:
            continue
        if task is not None and metadata.get("task") != task:
            continue
        return metadata
    raise FileNotFoundError(f"Model metadata for task={task}, model={model_name} is not found under {_MODELS_ROOT}.")


def get_model_dir(task, model_name):
    return get_model_metadata(task, model_name)["directory"]


def get_model_package(task, model_name):
    return get_model_metadata(task, model_name)["package"]


def get_model_resource_path(task, model_name, resource_name):
    return os.path.join(get_model_dir(task, model_name), resource_name)


def has_model_resource(task, model_name, resource_name):
    return os.path.exists(get_model_resource_path(task, model_name, resource_name))


def import_model_resource(task, model_name, module_name):
    try:
        model_dir = get_model_dir(task, model_name)
    except FileNotFoundError:
        return False
    if not os.path.isdir(model_dir):
        return False
    package = get_model_package(task, model_name)
    full_name = f"{package}.{module_name}"
    if importlib.util.find_spec(full_name) is None:
        return False
    importlib.import_module(full_name)
    return True
