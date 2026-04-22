"""
基于已保存 run 目录和数据工件继续训练并评估。
"""

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data.artifact_io import (
    deserialize_scaler,
    load_data_artifact,
    load_run_meta,
    validate_artifact_for_config,
    write_run_meta,
)
from GNNTP.data.dataloader import generate_dataloader
from GNNTP.models.locator import get_model_metadata
from GNNTP.utils import (
    add_general_args,
    ensure_run_id,
    get_executor,
    get_logger,
    get_model,
    get_run_subdir,
    set_random_seed,
    str2bool,
)


def run_resume_artifact(
    *,
    run_id: str,
    task: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    config_file: str | None = None,
    saved_model: bool = True,
    artifact_id: str | None = None,
    artifact_path: str | None = None,
    force_reuse: bool = False,
    other_args: dict | None = None,
):
    text_run_id = str(run_id or "").strip()
    if not text_run_id:
        raise ValueError("run_id is required.")
    run_meta = load_run_meta(text_run_id)
    resolved_task = str(task or run_meta.get("task") or "").strip()
    resolved_model = str(model_name or run_meta.get("model") or "").strip()
    resolved_dataset = str(dataset_name or run_meta.get("dataset") or "").strip()
    if not resolved_task or not resolved_model or not resolved_dataset:
        raise ValueError("run_meta.json missing task/model/dataset; cannot resume.")
    precheck_warnings = []
    for key, given, bound in [
        ("task", str(task or "").strip(), str(run_meta.get("task", "")).strip()),
        ("model", str(model_name or "").strip(), str(run_meta.get("model", "")).strip()),
        ("dataset", str(dataset_name or "").strip(), str(run_meta.get("dataset", "")).strip()),
    ]:
        if given and bound and given != bound:
            msg = "run_meta {} mismatch: bound={}, given={}".format(key, bound, given)
            if force_reuse:
                precheck_warnings.append(msg)
            else:
                raise ValueError(msg)

    bound_artifact_id = str(run_meta.get("artifact_id", "")).strip()
    selected_artifact_id = str(artifact_id or "").strip()
    if selected_artifact_id and bound_artifact_id and selected_artifact_id != bound_artifact_id:
        if not force_reuse:
            raise ValueError(
                "artifact_id mismatch with run binding: bound={}, given={}".format(
                    bound_artifact_id, selected_artifact_id
                )
            )
    effective_artifact_id = selected_artifact_id or bound_artifact_id
    if not effective_artifact_id and not artifact_path:
        raise ValueError("No artifact_id provided and run_meta has no bound artifact_id.")

    merged_other_args = dict(other_args or {})
    merged_other_args["exp_id"] = text_run_id
    config = ConfigParser(
        resolved_task,
        resolved_model,
        resolved_dataset,
        config_file,
        saved_model,
        True,
        merged_other_args,
    )
    exp_id = ensure_run_id(config)
    logger = get_logger(config)
    logger.info(
        "Begin resume artifact pipeline, run_id=%s, task=%s, model_name=%s, dataset_name=%s",
        text_run_id,
        resolved_task,
        resolved_model,
        resolved_dataset,
    )
    logger.info(config.config)
    for msg in precheck_warnings:
        logger.warning("[FORCE_REUSE] %s", msg)

    epoch = int(config.get("epoch", 0) or 0)
    max_epoch = int(config.get("max_epoch", 0) or 0)
    if epoch < 0:
        raise ValueError("Resume epoch must be >= 0.")
    if max_epoch <= epoch:
        raise ValueError("Resume max_epoch must be greater than epoch.")

    seed = config.get("seed", 0)
    set_random_seed(seed)

    bundle = load_data_artifact(
        artifact_id=effective_artifact_id if not artifact_path else None,
        artifact_path=artifact_path,
    )
    artifact_meta = dict(bundle["meta"])
    if bound_artifact_id and str(artifact_meta.get("artifact_id", "")).strip() != bound_artifact_id:
        msg = "Resolved artifact differs from run binding: bound={}, resolved={}".format(
            bound_artifact_id, str(artifact_meta.get("artifact_id", "")).strip()
        )
        if force_reuse:
            logger.warning("[FORCE_REUSE] %s", msg)
        else:
            raise ValueError(msg)

    model_metadata = get_model_metadata(resolved_task, resolved_model)
    warnings = validate_artifact_for_config(
        artifact_meta=artifact_meta,
        config_like=config,
        model_dataset_class=str(model_metadata.get("dataset_class", "")),
        force_reuse=bool(force_reuse),
    )
    for msg in warnings:
        logger.warning("[FORCE_REUSE] %s", msg)

    train_data = bundle["train"]
    valid_data = bundle["valid"]
    test_data = bundle["test"]
    feature_name = artifact_meta.get("feature_name", {"X": "float", "y": "float"})
    if not isinstance(feature_name, dict):
        feature_name = {"X": "float", "y": "float"}
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0))
    pad_with_last_sample = bool(config.get("pad_with_last_sample", True))
    train_loader, valid_loader, test_loader = generate_dataloader(
        train_data,
        valid_data,
        test_data,
        feature_name=feature_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pad_with_last_sample=pad_with_last_sample,
    )

    data_feature = dict(bundle["data_feature"])
    data_feature["scaler"] = deserialize_scaler(artifact_meta.get("scaler"))
    data_feature["ext_scaler"] = deserialize_scaler(artifact_meta.get("ext_scaler"))
    data_feature["num_batches"] = len(train_loader)

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    executor.train(train_loader, valid_loader)
    if saved_model:
        model_cache_file = os.path.join(
            get_run_subdir(exp_id, "model_cache"),
            "{}_{}.m".format(resolved_model, resolved_dataset),
        )
        executor.save_model(model_cache_file)
    test_result = executor.evaluate(test_loader)

    write_run_meta(
        exp_id,
        {
            "task": resolved_task,
            "model": resolved_model,
            "dataset": resolved_dataset,
            "artifact_id": str(artifact_meta.get("artifact_id", "") or ""),
            "artifact_dir": str(bundle["artifact_dir"]),
            "data_signature": str(artifact_meta.get("data_signature", "") or ""),
            "force_reuse": bool(force_reuse),
            "last_resumed_at": time.time(),
            "source": "run_resume_artifact",
        },
    )
    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="existing run directory name to resume")
    parser.add_argument("--task", type=str, default=None, help="task name (optional, fallback to run_meta)")
    parser.add_argument("--model", type=str, default=None, help="model name (optional, fallback to run_meta)")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name (optional, fallback to run_meta)")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=True, help="whether save the trained model")
    parser.add_argument("--artifact_id", type=str, default=None, help="data artifact id")
    parser.add_argument("--artifact_path", type=str, default=None, help="data artifact directory path")
    parser.add_argument("--force_reuse", type=str2bool, default=False, help="force reuse even if signature mismatch")
    parser.add_argument("--epoch", type=int, default=None, help="resume epoch")
    parser.add_argument("--exp_id", type=str, default=None, help="unused, run_id controls output directory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    add_general_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key
        not in [
            "run_id",
            "task",
            "model",
            "dataset",
            "config_file",
            "saved_model",
            "artifact_id",
            "artifact_path",
            "force_reuse",
        ]
        and val is not None
    }
    run_resume_artifact(
        run_id=str(args.run_id or "").strip(),
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        artifact_id=args.artifact_id,
        artifact_path=args.artifact_path,
        force_reuse=args.force_reuse,
        other_args=other_args,
    )
