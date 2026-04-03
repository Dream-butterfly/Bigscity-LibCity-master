import importlib
import json
import os
from functools import lru_cache


_MODELS_ROOT = os.path.dirname(__file__)
_MANIFEST_NAME = "manifest.json"
_ENTRY_KEYS = (
    "model_entry",
    "dataset_entry",
    "executor_entry",
    "evaluator_entry",
)


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
        _normalize_manifest(metadata)
        _validate_manifest(metadata)
        manifest_index.append(metadata)
    return tuple(manifest_index)


def _normalize_manifest(metadata):
    package = metadata["package"]
    model_name = metadata.get("model")
    if model_name:
        metadata.setdefault("model_entry", f"{package}.model:{model_name}")


def _validate_manifest(metadata):
    model_name = metadata.get("model", "<unknown>")
    missing_fields = []
    for field_name in ("task", "model", "dataset_class", "executor", "evaluator"):
        if not metadata.get(field_name):
            missing_fields.append(field_name)
    for entry_key in _ENTRY_KEYS:
        if not metadata.get(entry_key):
            missing_fields.append(entry_key)
    if missing_fields:
        raise KeyError(
            f"Manifest for model {model_name} is missing required fields: {', '.join(missing_fields)}."
        )


def _iter_model_manifests():
    return _load_manifest_index()


def _split_entrypoint(entrypoint):
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint '{entrypoint}'. Expected '<module>:<attribute>'.")
    module_name, attribute_name = entrypoint.split(":", 1)
    return module_name, attribute_name


def import_entrypoint(entrypoint):
    module_name, attribute_name = _split_entrypoint(entrypoint)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Entrypoint '{entrypoint}' could not resolve attribute '{attribute_name}'."
        ) from exc


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


def get_model_component_entry(task, model_name, component_type):
    metadata = get_model_metadata(task, model_name)
    try:
        return metadata[f"{component_type}_entry"]
    except KeyError as exc:
        raise KeyError(
            f"Manifest for task={task}, model={model_name} does not define {component_type}_entry."
        ) from exc


def get_model_component(task, model_name, component_type):
    return import_entrypoint(get_model_component_entry(task, model_name, component_type))


def find_component_entry(component_type, component_name, task=None, model_name=None):
    entry_key = f"{component_type}_entry"
    name_key = {
        "dataset": "dataset_class",
        "executor": "executor",
        "evaluator": "evaluator",
        "model": "model",
    }[component_type]
    if task is not None and model_name is not None:
        metadata = get_model_metadata(task, model_name)
        manifest_name = metadata.get(name_key)
        if manifest_name != component_name:
            raise AttributeError(
                f"{component_type} {component_name} does not match manifest entry {manifest_name} "
                f"for task={task}, model={model_name}"
            )
        return metadata[entry_key]

    matched_entries = set()
    for metadata in _iter_model_manifests():
        if metadata.get(name_key) == component_name and entry_key in metadata:
            matched_entries.add(metadata[entry_key])
    if not matched_entries:
        raise AttributeError(f"{component_type} {component_name} is not declared in any manifest")
    if len(matched_entries) > 1:
        raise AttributeError(
            f"{component_type} {component_name} is ambiguous across manifests: {sorted(matched_entries)}"
        )
    return matched_entries.pop()


def get_component(component_type, component_name, task=None, model_name=None):
    return import_entrypoint(find_component_entry(component_type, component_name, task=task, model_name=model_name))


def validate_manifest_index():
    for metadata in _iter_model_manifests():
        _validate_manifest(metadata)
