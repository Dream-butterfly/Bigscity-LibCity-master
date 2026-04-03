from libcity.utils.registry import Registry
from libcity.models.locator import import_model_resource


TASK_MODEL_REGISTRY = {
    "traffic_state_pred": Registry("traffic_state_pred.model"),
}

def register_model(task, name=None):
    if task not in TASK_MODEL_REGISTRY:
        raise AttributeError("task is not found")
    return TASK_MODEL_REGISTRY[task].register(name=name)


def get_model_class(task, model_name):
    imported = import_model_resource(task, model_name, "model")
    if not imported:
        raise AttributeError(f"model {model_name} for task {task} is not found")
    task_registry = TASK_MODEL_REGISTRY.get(task)
    if task_registry is None:
        raise AttributeError("task is not found")
    return task_registry.get(model_name)
