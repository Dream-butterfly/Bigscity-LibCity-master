import argparse
import random

from libcity.config_parser import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_logger, get_model, set_random_seed, str2bool


def main():
    parser = argparse.ArgumentParser(description="Run a one-batch smoke test for a model.")
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="RNN", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for smoke test")
    parser.add_argument("--saved_model", type=str2bool, default=False, help="whether save the trained model")
    parser.add_argument("--train", type=str2bool, default=True, help="whether build training pipeline")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    other_args = {"batch_size": args.batch_size, "seed": args.seed}
    config = ConfigParser(
        task=args.task,
        model=args.model,
        dataset=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        other_args=other_args,
    )
    exp_id = config.get("exp_id", None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config["exp_id"] = exp_id

    logger = get_logger(config)
    set_random_seed(args.seed)

    dataset = get_dataset(config)
    train_data, _, _ = dataset.get_data()
    data_feature = dataset.get_data_feature()
    batch = next(iter(train_data))

    model = get_model(config, data_feature).to(config["device"])
    get_executor(config, model, data_feature)

    batch.to_tensor(config["device"])
    output = model.predict(batch)
    logger.info("Smoke test output shape: %s", tuple(output.shape))
    logger.info("Smoke test finished successfully.")


if __name__ == "__main__":
    main()
