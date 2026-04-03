import argparse
import json

from libcity.config_parser import ConfigParser
from libcity.utils import str2bool


def main():
    parser = argparse.ArgumentParser(description="Inspect the merged runtime config.")
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="GRU", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=True, help="whether save the trained model")
    parser.add_argument("--train", type=str2bool, default=True, help="whether build train config")
    args = parser.parse_args()

    config = ConfigParser(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        other_args={},
    )
    print(json.dumps(config.config, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
