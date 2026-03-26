from libcity.core.registry import Registry


TASK_MODEL_REGISTRY = {
    "traffic_state_pred": Registry("traffic_state_pred.model"),
}
_BOOTSTRAPPED = set()


def register_model(task, name=None):
    if task not in TASK_MODEL_REGISTRY:
        raise AttributeError("task is not found")
    return TASK_MODEL_REGISTRY[task].register(name=name)


def _bootstrap_traffic_speed_models():
    if "traffic_state_pred" in _BOOTSTRAPPED:
        return
    import libcity.model.traffic_state_pred  # noqa: F401

    _BOOTSTRAPPED.add("traffic_state_pred")


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
        bootstrap()
    task_registry = TASK_MODEL_REGISTRY.get(task)
    if task_registry is None:
        raise AttributeError("task is not found")
    return task_registry.get(model_name)
