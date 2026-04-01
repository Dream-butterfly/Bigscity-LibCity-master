from libcity.utils.registry import Registry


EVALUATOR_REGISTRY = Registry("evaluator")
_BOOTSTRAPPED = set()


def register_evaluator(name=None):
    return EVALUATOR_REGISTRY.register(name=name)


def _bootstrap_traffic_state_evaluator():
    if "TrafficStateEvaluator" in _BOOTSTRAPPED:
        return
    import libcity.common.traffic_state_evaluator  # noqa: F401

    _BOOTSTRAPPED.add("TrafficStateEvaluator")


EVALUATOR_BOOTSTRAPPERS = {
    "TrafficStateEvaluator": _bootstrap_traffic_state_evaluator,
}


def get_evaluator_class(evaluator_name):
    bootstrap = EVALUATOR_BOOTSTRAPPERS.get(evaluator_name)
    if bootstrap is not None:
        bootstrap()
    return EVALUATOR_REGISTRY.get(evaluator_name)
