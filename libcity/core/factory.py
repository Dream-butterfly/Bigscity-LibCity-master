def create_dataset(config):
    from libcity.data.registry import get_dataset_class

    dataset_class = get_dataset_class(config["dataset_class"])
    return dataset_class(config)


def create_model(config, data_feature):
    from libcity.tasks.registry import get_model_class

    model_class = get_model_class(config["task"], config["model"])
    return model_class(config, data_feature)


def create_executor(config, model, data_feature):
    from libcity.tasks.registry import get_executor_class

    executor_class = get_executor_class(config["executor"])
    return executor_class(config, model, data_feature)


def create_evaluator(config):
    from libcity.tasks.registry import get_evaluator_class

    evaluator_class = get_evaluator_class(config["evaluator"])
    return evaluator_class(config)
