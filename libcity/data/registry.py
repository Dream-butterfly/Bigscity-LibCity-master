from libcity.core.registry import Registry
import pkgutil
import importlib
import logging
from typing import Set


DATASET_REGISTRY = Registry("dataset")
_BOOTSTRAPPED: Set[str] = set()


def register_dataset(name=None):
    return DATASET_REGISTRY.register(name=name)


def _import_submodules(package_name: str):
    """Import all submodules under the given package name (non-recursive for subpackages).

    This ensures that modules which call `register_dataset()` at import time are executed.
    Import errors are caught and logged so a single bad module won't break bootstrapping.
    """
    logger = logging.getLogger(__name__)
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        logger.warning("Failed to import package %s: %s", package_name, e)
        return

    # Recursively import all submodules under the package so registration
    # performed at import-time will be triggered even for nested modules.
    if not hasattr(pkg, '__path__'):
        return
    prefix = pkg.__name__ + '.'
    for finder, full_name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.warning("Failed to import %s: %s", full_name, e)


def _bootstrap_point_datasets():
    """Bootstrap importer for point datasets.

    It imports modules under `libcity.data.dataset` so that dataset classes
    which register themselves at import time are discovered automatically.
    """
    if "point" in _BOOTSTRAPPED:
        return
    _import_submodules('libcity.data.dataset')
    _BOOTSTRAPPED.add("point")


def _bootstrap_pdformer_dataset():
    """Bootstrap importer for PDFormer-related datasets.

    Imports the `dataset_subclass` package (if exists) to discover registrations.
    """
    if "pdformer" in _BOOTSTRAPPED:
        return
    _import_submodules('libcity.data.dataset.dataset_subclass')
    _BOOTSTRAPPED.add("pdformer")


# Map dataset class names (as used in config) to bootstrap functions. New dataset
# types following the convention of registering during module import will be
# automatically discovered by these bootstrappers, avoiding manual edits here.
DATASET_BOOTSTRAPPERS = {
    "TrafficStatePointDataset": _bootstrap_point_datasets,
    "PDFormerDataset": _bootstrap_pdformer_dataset,
}


def get_dataset_class(dataset_name):
    bootstrap = DATASET_BOOTSTRAPPERS.get(dataset_name)
    if bootstrap is not None:
        bootstrap()
    return DATASET_REGISTRY.get(dataset_name)
