"""
继续训练并评估单一模型（独立于 run_model.py 的入口脚本）
"""

import os
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libcity.common import ConfigParser
from libcity.data import get_dataset
from libcity.utils import (
    add_general_args,
    ensure_run_id,
    get_executor,
    get_logger,
    get_model,
    get_run_subdir,
    set_random_seed,
    str2bool,
)


def run_resume(task=None, model_name=None, dataset_name=None, config_file=None, saved_model=True, other_args=None):
    config = ConfigParser(task, model_name, dataset_name, config_file, saved_model, True, other_args)
    exp_id = ensure_run_id(config)
    logger = get_logger(config)
    logger.info(
        "Begin resume pipeline, task={}, model_name={}, dataset_name={}, exp_id={}".format(
            str(task), str(model_name), str(dataset_name), str(exp_id)
        )
    )
    logger.info(config.config)

    epoch = int(config.get("epoch", 0) or 0)
    max_epoch = int(config.get("max_epoch", 0) or 0)
    if epoch < 0:
        raise ValueError("Resume epoch must be >= 0.")
    if max_epoch <= epoch:
        raise ValueError("Resume max_epoch must be greater than epoch.")

    seed = config.get("seed", 0)
    set_random_seed(seed)

    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    executor.train(train_data, valid_data)
    if saved_model:
        model_cache_file = os.path.join(
            get_run_subdir(exp_id, "model_cache"),
            "{}_{}.m".format(model_name, dataset_name),
        )
        executor.save_model(model_cache_file)
    executor.evaluate(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="GRU", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=True, help="whether save the trained model")
    parser.add_argument("--exp_id", type=str, default=None, help="id of experiment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    add_general_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key not in ["task", "model", "dataset", "config_file", "saved_model"] and val is not None
    }
    run_resume(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        other_args=other_args,
    )
