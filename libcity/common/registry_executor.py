from libcity.utils.registry import Registry
from libcity.models.locator import import_model_resource


EXECUTOR_REGISTRY = Registry("executor")
_BOOTSTRAPPED = set()


def register_executor(name=None):
    return EXECUTOR_REGISTRY.register(name=name)


def _bootstrap_traffic_state_executor():
    if "TrafficStateExecutor" in _BOOTSTRAPPED:
        return
    import libcity.common.traffic_state_executor  # noqa: F401

    _BOOTSTRAPPED.add("TrafficStateExecutor")


EXECUTOR_BOOTSTRAPPERS = {
    "TrafficStateExecutor": _bootstrap_traffic_state_executor,
}


def get_executor_class(executor_name, task=None, model_name=None):
    if task is not None and model_name is not None:
        imported = import_model_resource(task, model_name, "executor")
        if imported:
            try:
                return EXECUTOR_REGISTRY.get(executor_name)
            except AttributeError:
                pass
    bootstrap = EXECUTOR_BOOTSTRAPPERS.get(executor_name)
    if bootstrap is not None:
        bootstrap()
    return EXECUTOR_REGISTRY.get(executor_name)
