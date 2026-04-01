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


def _bootstrap_dcrnn_executor():
    if "DCRNNExecutor" in _BOOTSTRAPPED:
        return
    import libcity.models.traffic_speed_prediction.DCRNN.executor  # noqa: F401

    _BOOTSTRAPPED.add("DCRNNExecutor")


def _bootstrap_pdformer_executor():
    if "PDFormerExecutor" in _BOOTSTRAPPED:
        return
    import libcity.models.traffic_flow_prediction.PDFormer.executor  # noqa: F401

    _BOOTSTRAPPED.add("PDFormerExecutor")


EXECUTOR_BOOTSTRAPPERS = {
    "TrafficStateExecutor": _bootstrap_traffic_state_executor,
    "DCRNNExecutor": _bootstrap_dcrnn_executor,
    "PDFormerExecutor": _bootstrap_pdformer_executor,
}


def get_executor_class(executor_name, task=None, model_name=None):
    if task is not None and model_name is not None:
        imported = import_model_resource(task, model_name, "executor")
        if imported:
            return EXECUTOR_REGISTRY.get(executor_name)
    bootstrap = EXECUTOR_BOOTSTRAPPERS.get(executor_name)
    if bootstrap is not None:
        bootstrap()
    return EXECUTOR_REGISTRY.get(executor_name)
