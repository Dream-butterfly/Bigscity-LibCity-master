"""
独立数据集处理脚本：仅执行数据集构建/缓存，不进行模型训练。
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libcity.common import ConfigParser
from libcity.data import get_dataset
from libcity.utils import add_general_args, ensure_run_id, get_logger, set_random_seed, str2bool


def run_data_prep(task=None, model_name=None, dataset_name=None, config_file=None, other_args=None):
    config = ConfigParser(task, model_name, dataset_name, config_file, saved_model=False, train=False, other_args=other_args)
    exp_id = ensure_run_id(config)
    logger = get_logger(config)
    logger.info(
        "Begin data prep, task={}, model_name={}, dataset_name={}, exp_id={}".format(
            str(task), str(model_name), str(dataset_name), str(exp_id)
        )
    )
    seed = config.get("seed", 0)
    set_random_seed(seed)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    return {
        "task": task,
        "model": model_name,
        "dataset": dataset_name,
        "exp_id": str(exp_id),
        "cache_file_name": str(getattr(dataset, "cache_file_name", "") or ""),
        "train_batches": len(train_data) if train_data is not None else 0,
        "valid_batches": len(valid_data) if valid_data is not None else 0,
        "test_batches": len(test_data) if test_data is not None else 0,
        "data_feature_keys": sorted([str(k) for k in data_feature.keys()]) if isinstance(data_feature, dict) else [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="GRU", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=False, help="unused for data prep")
    parser.add_argument("--train", type=str2bool, default=False, help="unused for data prep")
    parser.add_argument("--version_meta", type=str, default=None, help="path to write json metadata")
    parser.add_argument("--exp_id", type=str, default=None, help="id of experiment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    add_general_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key not in ["task", "model", "dataset", "config_file", "saved_model", "train", "version_meta"] and val is not None
    }
    meta = run_data_prep(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        other_args=other_args,
    )
    if args.version_meta:
        out = Path(args.version_meta)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
