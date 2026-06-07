"""
仅消费数据工件进行训练与评估，不触发数据处理流程。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.common import ConfigParser
from GNNTP.data.artifact_io import (
    write_run_meta,
)
from GNNTP.data import build_artifact_runtime
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


def _maybe_wrap_ddp(config, model):
    """DDP 初始化 + 模型包装。非 DDP 时原样返回模型"""
    is_distributed = config.get('is_distributed', False)
    if not is_distributed:
        return model

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.get('dist_backend', 'nccl'),
            init_method='env://'
        )
    local_rank = config.get('local_rank', 0)
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    return model


def run_train_artifact(
    task=None,
    model_name=None,
    dataset_name=None,
    config_file=None,
    saved_model=True,
    train=True,
    artifact_id=None,
    artifact_path=None,
    force_reuse=False,
    other_args=None,
):
    config = ConfigParser(task, model_name, dataset_name, config_file, saved_model, train, other_args)
    resolved_task = str(config.get("task", task))
    resolved_model = str(config.get("model", model_name))
    resolved_dataset = str(config.get("dataset", dataset_name))
    exp_id = ensure_run_id(config)
    # 保存完整有效配置，供继续训练时直接读取
    _cfg_dir = Path(PROJECT_ROOT) / "outputs" / exp_id
    _cfg_dir.mkdir(parents=True, exist_ok=True)
    _cfg_save = dict(config.config)
    _cfg_save.pop("device", None)  # torch.device 不可 JSON 序列化
    with open(_cfg_dir / "effective_config.json", "w", encoding="utf-8") as _f:
        json.dump(_cfg_save, _f, ensure_ascii=False, indent=2, default=str)
    is_distributed = config.get("is_distributed", False)
    rank = config.get("rank", 0)
    logger = get_logger(config)
    if rank == 0:
        logger.info(
            "Begin artifact training pipeline, task=%s, model_name=%s, dataset_name=%s, exp_id=%s",
            resolved_task,
            resolved_model,
            resolved_dataset,
            str(exp_id),
        )
        logger.info(config.config)

    seed = config.get("seed", 0)
    set_random_seed(seed)

    runtime = build_artifact_runtime(
        config,
        task=resolved_task,
        model_name=resolved_model,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
        force_reuse=force_reuse,
    )
    for msg in runtime.warnings:
        logger.warning("[FORCE_REUSE] %s", msg)

    model = get_model(config, runtime.data_feature)
    model = _maybe_wrap_ddp(config, model)
    executor = get_executor(config, model, runtime.data_feature)
    model_cache_file = os.path.join(
        get_run_subdir(exp_id, "model_cache"),
        "{}_{}.m".format(resolved_model, resolved_dataset),
    )
    if train or not os.path.exists(model_cache_file):
        executor.train(runtime.train_loader, runtime.valid_loader)
        if saved_model and rank == 0:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)

    test_result = {}
    # DDP 下所有 rank 必须参与 evaluate()，因为内部有 all_gather 需要全 rank 同步
    if is_distributed or rank == 0:
        test_result = executor.evaluate(runtime.test_loader)

    if rank == 0 or not is_distributed:
        write_run_meta(
            exp_id,
            {
                "task": resolved_task,
                "model": resolved_model,
                "dataset": resolved_dataset,
                "artifact_id": str(runtime.artifact_meta.get("artifact_id", "") or ""),
                "artifact_dir": str(runtime.artifact_dir or ""),
                "data_signature": str(runtime.artifact_meta.get("data_signature", "") or ""),
                "force_reuse": bool(force_reuse),
                "source": "run_train_artifact",
            },
        )

    if is_distributed:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            dist.destroy_process_group()

    return test_result


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
            "artifact_id",
            "artifact_path",
            "force_reuse",
        ]
        and val is not None
    }
    run_train_artifact(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        artifact_id=args.artifact_id,
        artifact_path=args.artifact_path,
        force_reuse=args.force_reuse,
        other_args=other_args,
    )
