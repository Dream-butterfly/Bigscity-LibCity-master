from GNNTP.models.locator import get_component


def get_dataset_class(dataset_name, task=None, model_name=None):
    return get_component("dataset", dataset_name, task=task, model_name=model_name)
