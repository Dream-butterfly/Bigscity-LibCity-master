import os

from GNNTP.common import ConfigParser
from GNNTP.data import build_dataset_runtime
from GNNTP.utils import get_executor, get_model, get_logger, get_run_subdir, ensure_run_id, set_random_seed


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
    logger = get_logger(config)
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
    executor = get_executor(config, model, runtime.data_feature)
    if train or not os.path.exists(model_cache_file):
        executor.train(runtime.train_loader, runtime.valid_loader)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    executor.evaluate(runtime.test_loader)


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    runtime = build_dataset_runtime(config)
    model = get_model(config, runtime.data_feature)
    executor = get_executor(config, model, runtime.data_feature)
    best_valid_score = executor.train(runtime.train_loader, runtime.valid_loader)
    test_result = executor.evaluate(runtime.test_loader)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
