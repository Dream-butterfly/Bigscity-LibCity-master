"""
基于数据工件执行超参数搜索（Optuna）。
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser, HyperTuning
from GNNTP.data import build_artifact_runtime
from GNNTP.utils import (
    add_general_args,
    build_run_id,
    get_executor,
    get_logger,
    get_model,
    get_run_subdir,
    set_random_seed,
    str2bool,
)


def objective_function_artifact(
    task=None,
    model_name=None,
    dataset_name=None,
    config_file=None,
    saved_model=True,
    train=True,
    other_args=None,
    hyper_config_dict=None,
    artifact_id=None,
    artifact_path=None,
    force_reuse=False,
):
    config = ConfigParser(
        task,
        model_name,
        dataset_name,
        config_file,
        saved_model,
        train,
        other_args,
        hyper_config_dict,
    )
    resolved_task = str(config.get("task", task))
    resolved_model = str(config.get("model", model_name))
    set_random_seed(config.get("seed", 0))

    runtime = build_artifact_runtime(
        config,
        task=resolved_task,
        model_name=resolved_model,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
        force_reuse=force_reuse,
    )
    model = get_model(config, runtime.data_feature)
    executor = get_executor(config, model, runtime.data_feature)
    best_valid_score = executor.train(runtime.train_loader, runtime.valid_loader)
    test_result = executor.evaluate(runtime.test_loader)
    return {"best_valid_score": best_valid_score, "test_result": test_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="STGCN", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=True, help="whether save the trained model")
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        help="whether re-train model if the model is trained before",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default=str(Path(__file__).resolve().with_name("hyper_example.txt")),
        help="the file which specify the hyper-parameters and ranges to be adjusted",
    )
    parser.add_argument(
        "--hyper_algo",
        type=str,
        default="grid_search",
        help="hyper-parameters search algorithm: grid_search, random_search, tpe",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=100,
        help="Allow up to this many function evaluations before returning.",
    )
    parser.add_argument("--artifact_id", type=str, default=None, help="data artifact id")
    parser.add_argument("--artifact_path", type=str, default=None, help="data artifact directory path")
    parser.add_argument("--force_reuse", type=str2bool, default=False, help="force reuse even if signature mismatch")
    parser.add_argument("--exp_id", type=str, default=None, help="id of experiment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    add_general_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key
        not in [
            "task",
            "model",
            "dataset",
            "config_file",
            "saved_model",
            "train",
            "params_file",
            "hyper_algo",
            "artifact_id",
            "artifact_path",
            "force_reuse",
        ]
        and val is not None
    }
    exp_id = dict_args.get("exp_id", None) or build_run_id(args.task, args.model, args.dataset)
    other_args["exp_id"] = exp_id
    logger = get_logger({"model": args.model, "dataset": args.dataset, "exp_id": exp_id})
    seed = dict_args.get("seed", 0)
    set_random_seed(seed)
    other_args["seed"] = seed

    objective = partial(
        objective_function_artifact,
        artifact_id=args.artifact_id,
        artifact_path=args.artifact_path,
        force_reuse=args.force_reuse,
    )
    hp = HyperTuning(
        objective,
        params_file=args.params_file,
        algo=args.hyper_algo,
        max_evals=args.max_evals,
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        other_args=other_args,
    )
    hp.start()
    hp.save_result(filename=os.path.join(get_run_subdir(exp_id, "artifacts"), "hyper.result"))
    logger.info("best params: " + str(hp.best_params))
    logger.info("best result: ")
    logger.info(str(hp.params2result[hp.params2str(hp.best_params)]))
