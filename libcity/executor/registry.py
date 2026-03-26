from libcity.core.registry import Registry
import pkgutil
import importlib
import logging
from typing import Set


EXECUTOR_REGISTRY = Registry("executor")
_BOOTSTRAPPED: Set[str] = set()


def register_executor(name=None):
    return EXECUTOR_REGISTRY.register(name=name)


def _import_submodules(package_name: str):
    logger = logging.getLogger(__name__)
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        logger.warning("Failed to import package %s: %s", package_name, e)
        return
    if not hasattr(pkg, '__path__'):
        return
    prefix = pkg.__name__ + '.'
    for finder, full_name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.warning("Failed to import %s: %s", full_name, e)


def _bootstrap_traffic_state_executor():
    if "TrafficStateExecutor" in _BOOTSTRAPPED:
        return
    _import_submodules('libcity.executor')
    try:
        importlib.import_module('libcity.executor.traffic_state_executor')
    except Exception:
        pass
    _BOOTSTRAPPED.add("TrafficStateExecutor")


def _bootstrap_dcrnn_executor():
    if "DCRNNExecutor" in _BOOTSTRAPPED:
        return
    _import_submodules('libcity.executor')
    try:
        importlib.import_module('libcity.executor.dcrnn_executor')
    except Exception:
        pass
    _BOOTSTRAPPED.add("DCRNNExecutor")


def _bootstrap_pdformer_executor():
    if "PDFormerExecutor" in _BOOTSTRAPPED:
        return
    _import_submodules('libcity.executor')
    try:
        importlib.import_module('libcity.executor.pdformer_executor')
    except Exception:
        pass
    _BOOTSTRAPPED.add("PDFormerExecutor")


def get_executor_class(executor_name):
    # Always attempt discovery under libcity.executor
    try:
        _import_submodules('libcity.executor')
    except Exception:
        pass
    return EXECUTOR_REGISTRY.get(executor_name)
