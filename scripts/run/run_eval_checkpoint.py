"""
基于已保存的 checkpoint 进行模型评估（不训练）。

用法:
    uv run scripts/run/run_eval_checkpoint.py \
        --run_id 20260507_110413__traffic_state_pred__STGCN__METR_LA \
        --artifact_id da_20260507_110245__... \
        --epoch 10
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data.artifact_io import load_run_meta
from GNNTP.data import build_artifact_runtime
from GNNTP.utils import (
    add_general_args,
    get_executor,
    get_logger,
    get_model,
    set_random_seed,
    str2bool,
)


def run_eval_checkpoint(
    *,
    run_id: str,
    epoch: int,
    task: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    config_file: str | None = None,
    artifact_id: str | None = None,
    force_reuse: bool = False,
    other_args: dict | None = None,
):
    """从指定 run 的 checkpoint 加载模型并评估。

    Args:
        run_id: 已有运行目录名
        epoch: 要加载的 checkpoint epoch
        task: 任务名（可选，fallback 到 run_meta）
        model_name: 模型名（可选，fallback 到 run_meta）
        dataset_name: 数据集名（可选，fallback 到 run_meta）
        config_file: 额外配置文件路径
        artifact_id: 数据工件 ID（可选，fallback 到 run_meta）
        force_reuse: 强制复用（即使签名不匹配）
        other_args: 其他配置参数
    """
    text_run_id = str(run_id or "").strip()
    if not text_run_id:
        raise ValueError("run_id is required.")

    run_meta = load_run_meta(text_run_id)
    resolved_task = str(task or run_meta.get("task") or "").strip()
    resolved_model = str(model_name or run_meta.get("model") or "").strip()
    resolved_dataset = str(dataset_name or run_meta.get("dataset") or "").strip()

    if not resolved_task or not resolved_model or not resolved_dataset:
        raise ValueError(
            "Cannot determine task/model/dataset from run_meta.json; "
            "please provide --task, --model, --dataset explicitly."
        )

    # Resolve artifact_id
    bound_artifact_id = str(run_meta.get("artifact_id", "")).strip()
    effective_artifact_id = str(artifact_id or "").strip() or bound_artifact_id
    if not effective_artifact_id:
        raise ValueError(
            "No artifact_id provided and run_meta has no bound artifact_id. "
            "Please provide --artifact_id."
        )

    merged_other_args = dict(other_args or {})
    merged_other_args["exp_id"] = text_run_id
    # 将 epoch 注入 config，executor.init 中 _epoch_num > 0 时自动调用 load_model_with_epoch
    merged_other_args["epoch"] = int(epoch)

    config = ConfigParser(
        resolved_task,
        resolved_model,
        resolved_dataset,
        config_file,
        saved_model=False,  # 评估时不保存模型
        train=False,        # 评估时不训练
        other_args=merged_other_args,
    )

    logger = get_logger(config)
    logger.info(
        "Begin eval checkpoint pipeline, run_id=%s, epoch=%d, task=%s, model=%s, dataset=%s",
        text_run_id, epoch, resolved_task, resolved_model, resolved_dataset,
    )
    logger.info(str(config.config))

    seed = config.get("seed", 0)
    set_random_seed(seed)

    runtime = build_artifact_runtime(
        config,
        task=resolved_task,
        model_name=resolved_model,
        artifact_id=effective_artifact_id,
        force_reuse=force_reuse,
    )

    for msg in runtime.warnings:
        logger.warning("[FORCE_REUSE] %s", msg)

    model = get_model(config, runtime.data_feature)
    executor = get_executor(config, model, runtime.data_feature)
    # checkpoint 已在 executor.init 中通过 _epoch_num > 0 自动加载
    test_result = executor.evaluate(runtime.test_loader)

    logger.info("Eval checkpoint pipeline finished.")
    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model from saved checkpoint")
    parser.add_argument("--run_id", type=str, required=True, help="existing run directory name")
    parser.add_argument("--epoch", type=int, required=True, help="checkpoint epoch to evaluate")
    parser.add_argument("--task", type=str, default=None, help="task name (optional, fallback to run_meta)")
    parser.add_argument("--model", type=str, default=None, help="model name (optional, fallback to run_meta)")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name (optional, fallback to run_meta)")
    parser.add_argument("--config_file", type=str, default=None, help="extra config file")
    parser.add_argument("--artifact_id", type=str, default=None, help="data artifact id (optional, fallback to run_meta)")
    parser.add_argument("--force_reuse", type=str2bool, default=False)
    add_general_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key
        not in [
            "run_id", "epoch", "task", "model", "dataset",
            "config_file", "artifact_id", "force_reuse",
        ]
        and val is not None
    }

    run_eval_checkpoint(
        run_id=args.run_id,
        epoch=args.epoch,
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        artifact_id=args.artifact_id,
        force_reuse=args.force_reuse,
        other_args=other_args,
    )
