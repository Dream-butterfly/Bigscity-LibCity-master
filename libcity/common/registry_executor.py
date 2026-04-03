from libcity.models.locator import get_component


def get_executor_class(executor_name, task=None, model_name=None):
    return get_component("executor", executor_name, task=task, model_name=model_name)
