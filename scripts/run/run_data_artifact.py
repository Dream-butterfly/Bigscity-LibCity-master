"""
独立数据工件构建脚本：执行完整数据处理并输出可复用的数据工件。
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data import get_dataset
from GNNTP.data.artifact_io import (
    build_data_artifact_id,
    build_data_signature_payload,
    compute_data_signature,
    extract_xy_arrays_from_loader,
    get_data_artifacts_root,
    serialize_scaler,
    write_data_artifact,
)
from GNNTP.utils import add_general_args, ensure_run_id, get_logger, set_random_seed, str2bool


def run_data_artifact(
    task=None,
    model_name=None,
    dataset_name=None,
    config_file=None,
    artifact_id=None,
    artifact_overwrite=False,
    other_args=None,
):
    config = ConfigParser(
        task,
        model_name,
        dataset_name,
        config_file,
        saved_model=False,
        train=False,
        other_args=other_args,
    )
    resolved_task = str(config.get("task", task))
    resolved_model = str(config.get("model", model_name))
    resolved_dataset = str(config.get("dataset", dataset_name))
    exp_id = ensure_run_id(config)
    logger = get_logger(config)
    logger.info(
        "Begin data artifact pipeline, task=%s, model_name=%s, dataset_name=%s, exp_id=%s",
        resolved_task,
        resolved_model,
        resolved_dataset,
        str(exp_id),
    )

    seed = config.get("seed", 0)
    set_random_seed(seed)

    # 数据工件阶段禁用补齐，避免引入 batch_size 维度耦合。
    config["pad_with_last_sample"] = False

    dataset = get_dataset(config)
    train_loader, valid_loader, test_loader = dataset.get_data()
    data_feature = dataset.get_data_feature()

    train_x, train_y = extract_xy_arrays_from_loader(train_loader, "train")
    valid_x, valid_y = extract_xy_arrays_from_loader(valid_loader, "valid")
    test_x, test_y = extract_xy_arrays_from_loader(test_loader, "test")

    dataset_class = str(config.get("dataset_class", "")).strip()
    signature_payload = build_data_signature_payload(config)
    data_signature = compute_data_signature(signature_payload)
    resolved_artifact_id = (
        str(artifact_id).strip()
        if str(artifact_id or "").strip()
        else build_data_artifact_id(resolved_dataset, dataset_class, data_signature)
    )

    artifact_dir = get_data_artifacts_root() / resolved_artifact_id
    scaler_payload = serialize_scaler(getattr(dataset, "scaler", None))
    ext_scaler_payload = serialize_scaler(getattr(dataset, "ext_scaler", None))
    meta = write_data_artifact(
        artifact_dir,
        artifact_id=resolved_artifact_id,
        task=resolved_task,
        model=resolved_model,
        dataset=resolved_dataset,
        dataset_class=dataset_class,
        feature_name=dict(getattr(dataset, "feature_name", {"X": "float", "y": "float"})),
        data_signature=data_signature,
        data_signature_payload=signature_payload,
        config_snapshot=dict(config.config),
        scaler_payload=scaler_payload,
        ext_scaler_payload=ext_scaler_payload,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
        data_feature=dict(data_feature),
        extra_meta={
            "cache_file_name": str(getattr(dataset, "cache_file_name", "") or ""),
            "train_batches": len(train_loader) if train_loader is not None else 0,
            "valid_batches": len(valid_loader) if valid_loader is not None else 0,
            "test_batches": len(test_loader) if test_loader is not None else 0,
            "data_feature_keys": (
                sorted([str(k) for k in data_feature.keys()]) if isinstance(data_feature, dict) else []
            ),
        },
        overwrite=bool(artifact_overwrite),
    )
    logger.info("Data artifact ready: %s", str(artifact_dir))

    return {
        "task": resolved_task,
        "model": resolved_model,
        "dataset": resolved_dataset,
        "exp_id": str(exp_id),
        "artifact_id": str(meta.get("artifact_id", resolved_artifact_id)),
        "artifact_dir": str(artifact_dir),
        "data_signature": str(meta.get("data_signature", data_signature)),
        "train_batches": int(meta.get("train_batches", 0) or 0),
        "valid_batches": int(meta.get("valid_batches", 0) or 0),
        "test_batches": int(meta.get("test_batches", 0) or 0),
        "cache_file_name": str(meta.get("cache_file_name", "") or ""),
        "data_feature_keys": meta.get("data_feature_keys", []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="the name of task")
    parser.add_argument("--model", type=str, default="STGCN", help="the name of model")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="the name of dataset")
    parser.add_argument("--config_file", type=str, default=None, help="the file name of config file")
    parser.add_argument("--saved_model", type=str2bool, default=False, help="unused for data artifact build")
    parser.add_argument("--train", type=str2bool, default=False, help="unused for data artifact build")
    parser.add_argument("--version_meta", type=str, default=None, help="path to write json metadata")
    parser.add_argument("--artifact_id", type=str, default=None, help="optional artifact id")
    parser.add_argument(
        "--artifact_overwrite",
        type=str2bool,
        default=False,
        help="whether overwrite artifact if exists",
    )
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
            "version_meta",
            "artifact_id",
            "artifact_overwrite",
        ]
        and val is not None
    }
    meta = run_data_artifact(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        artifact_id=args.artifact_id,
        artifact_overwrite=args.artifact_overwrite,
        other_args=other_args,
    )
    if args.version_meta:
        out = Path(args.version_meta)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
