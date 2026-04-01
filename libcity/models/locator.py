import importlib
import importlib.util
import os


_MODELS_ROOT = os.path.dirname(__file__)


def get_model_dir(task, model_name):
    direct_path = os.path.join(_MODELS_ROOT, task, model_name)
    if os.path.isdir(direct_path):
        return direct_path
    for entry in os.listdir(_MODELS_ROOT):
        candidate = os.path.join(_MODELS_ROOT, entry, model_name)
        if os.path.isdir(candidate):
            return candidate
    return direct_path


def get_model_package(task, model_name):
    model_dir = get_model_dir(task, model_name)
    relative_dir = os.path.relpath(model_dir, _MODELS_ROOT)
    package_suffix = relative_dir.replace(os.sep, ".")
    return f"libcity.models.{package_suffix}"


def get_model_resource_path(task, model_name, resource_name):
    return os.path.join(get_model_dir(task, model_name), resource_name)


def has_model_resource(task, model_name, resource_name):
    return os.path.exists(get_model_resource_path(task, model_name, resource_name))


def import_model_resource(task, model_name, module_name):
    model_dir = get_model_dir(task, model_name)
    if not os.path.isdir(model_dir):
        return False
    package = get_model_package(task, model_name)
    full_name = f"{package}.{module_name}"
    if importlib.util.find_spec(full_name) is None:
        return False
    importlib.import_module(full_name)
    return True
