def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    from libcity.data.registry import get_dataset_class

    dataset_class = get_dataset_class(config['dataset_class'])
    return dataset_class(config)
