import os

import torch

from GNNTP.common import ConfigParser
from GNNTP.data import build_dataset_runtime
from GNNTP.utils import get_executor, get_model, get_logger, get_run_subdir, ensure_run_id, set_random_seed


def _maybe_wrap_ddp(config, model):
    """DDP 初始化 + 模型包装。非 DDP 时原样返回模型"""
    is_distributed = config.get('is_distributed', False)
    if not is_distributed:
        return model

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.get('dist_backend', 'nccl'),
            init_method='env://'
        )
    local_rank = config.get('local_rank', 0)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    return model


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = ensure_run_id(config)
    is_distributed = config.get('is_distributed', False)
    rank = config.get('rank', 0)
    logger = get_logger(config)
    if rank == 0:
        logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                    format(str(task), str(model_name), str(dataset_name), str(exp_id)))
        logger.info(config.config)
    seed = config.get('seed', 0)
    set_random_seed(seed)
    runtime = build_dataset_runtime(config)
    model_cache_file = os.path.join(
        get_run_subdir(exp_id, 'model_cache'),
        '{}_{}.m'.format(model_name, dataset_name)
    )
    model = get_model(config, runtime.data_feature)
    model = _maybe_wrap_ddp(config, model)
    executor = get_executor(config, model, runtime.data_feature)
    if train or not os.path.exists(model_cache_file):
        executor.train(runtime.train_loader, runtime.valid_loader)
        if saved_model and rank == 0:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    if rank == 0 or not is_distributed:
        executor.evaluate(runtime.test_loader)
    if is_distributed:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            dist.destroy_process_group()


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    runtime = build_dataset_runtime(config)
    model = get_model(config, runtime.data_feature)
    model = _maybe_wrap_ddp(config, model)
    executor = get_executor(config, model, runtime.data_feature)
    best_valid_score = executor.train(runtime.train_loader, runtime.valid_loader)
    test_result = executor.evaluate(runtime.test_loader)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
