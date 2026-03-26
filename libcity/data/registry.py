from libcity.utils.registry import Registry


DATASET_REGISTRY = Registry("dataset")
_BOOTSTRAPPED = set()


def register_dataset(name=None):
    return DATASET_REGISTRY.register(name=name)


def _bootstrap_point_datasets():
    if "point" in _BOOTSTRAPPED:
        return
    import libcity.data.dataset  # noqa: F401

    _BOOTSTRAPPED.add("point")


def _bootstrap_pdformer_dataset():
    if "pdformer" in _BOOTSTRAPPED:
        return
    import libcity.data.dataset.dataset_subclass  # noqa: F401

    _BOOTSTRAPPED.add("pdformer")


DATASET_BOOTSTRAPPERS = {
    "TrafficStatePointDataset": _bootstrap_point_datasets,
    "PDFormerDataset": _bootstrap_pdformer_dataset,
}


def get_dataset_class(dataset_name):
    bootstrap = DATASET_BOOTSTRAPPERS.get(dataset_name)
    if bootstrap is not None:
        bootstrap()
    return DATASET_REGISTRY.get(dataset_name)
