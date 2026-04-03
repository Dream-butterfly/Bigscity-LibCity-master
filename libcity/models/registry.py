from libcity.models.locator import get_model_component


def get_model_class(task, model_name):
    return get_model_component(task, model_name, "model")
