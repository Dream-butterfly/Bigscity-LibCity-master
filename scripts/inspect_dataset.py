import argparse
import json
import random

from libcity.tasks.config_parser import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_logger, set_random_seed, str2bool


def _shape_of(item):
    if hasattr(item, "shape"):
        return tuple(item.shape)
    if hasattr(item, "__len__"):
        return len(item)
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset outputs without training a model.")
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="GRU", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=False, help="whether save the trained model")
    parser.add_argument("--train", type=str2bool, default=True, help="whether build training pipeline")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for dataset inspection")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    config = ConfigParser(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        other_args={"batch_size": args.batch_size, "seed": args.seed},
    )
    exp_id = config.get("exp_id", None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config["exp_id"] = exp_id

    logger = get_logger(config)
    set_random_seed(args.seed)

    dataset = get_dataset(config)
    train_data, eval_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    summary = {
        "dataset_class": dataset.__class__.__name__,
        "train_batches": len(train_data) if train_data is not None else 0,
        "eval_batches": len(eval_data) if eval_data is not None else 0,
        "test_batches": len(test_data) if test_data is not None else 0,
        "data_feature_keys": sorted(data_feature.keys()),
        "data_feature_shapes": {
            key: _shape_of(value) for key, value in data_feature.items() if _shape_of(value) is not None
        },
    }
    logger.info("Dataset inspection summary:\n%s", json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
