import argparse
import random
from pathlib import Path

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_logger, set_random_seed, str2bool


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CACHE = PROJECT_ROOT / "libcity" / "cache" / "dataset_cache"


DTW_CACHE_PATTERNS = [
    "dtw_{dataset}.npy",
    "dtw_distance_index_{dataset}.npz",
    "dtw_graph_{dataset}.npz",
    "dtw_edge_index_{dataset}.npz",
]


def _remove_existing(dataset_name, dry_run):
    removed = []
    for pattern in DTW_CACHE_PATTERNS:
        path = DATASET_CACHE / pattern.format(dataset=dataset_name)
        if path.exists():
            removed.append(path)
            if dry_run:
                print(f"[dry-run] remove {path}")
            else:
                path.unlink()
                print(f"removed {path}")
    if not removed:
        print("no existing DTW cache files found")


def main():
    parser = argparse.ArgumentParser(description="Remove and optionally rebuild DTW cache for a dataset.")
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, required=True, help="the name of model")
    parser.add_argument("--dataset", type=str, required=True, help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size when loading dataset")
    parser.add_argument(
        "--rebuild",
        type=str2bool,
        default=True,
        help="whether rebuild DTW cache immediately after deletion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show actions without deleting or rebuilding",
    )
    args = parser.parse_args()

    _remove_existing(args.dataset, args.dry_run)
    if args.dry_run or not args.rebuild:
        return

    other_args = {"seed": args.seed, "batch_size": args.batch_size}
    config = ConfigParser(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        config_file=args.config_file,
        saved_model=False,
        train=True,
        other_args=other_args,
    )
    exp_id = config.get("exp_id", None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config["exp_id"] = exp_id

    logger = get_logger(config)
    set_random_seed(args.seed)
    dataset = get_dataset(config)
    dataset.get_data()
    logger.info("DTW cache rebuild completed for dataset %s with model %s.", args.dataset, args.model)


if __name__ == "__main__":
    main()
