from libcity.models.locator import get_component


def get_evaluator_class(evaluator_name, task=None, model_name=None):
    return get_component("evaluator", evaluator_name, task=task, model_name=model_name)
