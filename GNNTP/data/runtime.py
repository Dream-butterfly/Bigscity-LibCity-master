from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data.distributed import DistributedSampler

from GNNTP.data.artifact_io import (
    deserialize_scaler,
    load_data_artifact,
    validate_artifact_for_config,
)
from GNNTP.data.core.list_dataset import ArrayTupleDataset, ListDataset
from GNNTP.data.dataloader import generate_dataloader
from GNNTP.data.factory import get_dataset
from GNNTP.models.locator import get_model_metadata


def _make_ddp_samplers(train_data, eval_data, test_data, world_size, rank):
    """为 DDP 创建 DistributedSampler 三元组。非 DDP 不应调用此函数。"""
    def _ds(data):
        return ArrayTupleDataset(*data) if isinstance(data, tuple) else ListDataset(data)

    return (
        DistributedSampler(_ds(train_data), num_replicas=world_size, rank=rank, shuffle=True),
        DistributedSampler(_ds(eval_data), num_replicas=world_size, rank=rank, shuffle=False),
        DistributedSampler(_ds(test_data), num_replicas=world_size, rank=rank, shuffle=False),
    )


@dataclass
class DataRuntime:
    train_loader: Any
    valid_loader: Any
    test_loader: Any
    data_feature: dict[str, Any]
    feature_name: dict[str, Any]
    source: str
    dataset: Any | None = None
    artifact_meta: dict[str, Any] = field(default_factory=dict)
    artifact_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def build_dataset_runtime(config: Any) -> DataRuntime:
    dataset = get_dataset(config)
    train_loader, valid_loader, test_loader = dataset.get_data()
    data_feature = dataset.get_data_feature()
    feature_name = dict(getattr(dataset, "feature_name", {"X": "float", "y": "float"}))
    return DataRuntime(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        data_feature=dict(data_feature),
        feature_name=feature_name,
        dataset=dataset,
        source="dataset",
    )


def build_artifact_runtime(
    config: Any,
    *,
    task: str,
    model_name: str,
    artifact_id: str | None = None,
    artifact_path: str | None = None,
    force_reuse: bool = False,
) -> DataRuntime:
    bundle = load_data_artifact(artifact_id=artifact_id, artifact_path=artifact_path)
    artifact_meta = dict(bundle["meta"])
    model_metadata = get_model_metadata(task, model_name)
    warnings = validate_artifact_for_config(
        artifact_meta=artifact_meta,
        config_like=config,
        model_dataset_class=str(model_metadata.get("dataset_class", "")),
        force_reuse=bool(force_reuse),
    )

    feature_name = artifact_meta.get("feature_name", {"X": "float", "y": "float"})
    if not isinstance(feature_name, dict):
        feature_name = {"X": "float", "y": "float"}

    is_distributed = bool(config.get("is_distributed", False))
    train_sampler = eval_sampler = test_sampler = None
    if is_distributed:
        train_sampler, eval_sampler, test_sampler = _make_ddp_samplers(
            bundle["train"], bundle["valid"], bundle["test"],
            config["world_size"], config["rank"],
        )

    train_loader, valid_loader, test_loader = generate_dataloader(
        bundle["train"],
        bundle["valid"],
        bundle["test"],
        feature_name=feature_name,
        batch_size=int(config.get("batch_size", 64)),
        num_workers=int(config.get("num_workers", 0)),
        shuffle=True,
        pad_with_last_sample=(
            False if is_distributed else bool(config.get("pad_with_last_sample", True))
        ),
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        test_sampler=test_sampler,
    )

    data_feature = dict(bundle["data_feature"])
    data_feature["scaler"] = deserialize_scaler(artifact_meta.get("scaler"))
    data_feature["ext_scaler"] = deserialize_scaler(artifact_meta.get("ext_scaler"))
    data_feature["num_batches"] = len(train_loader)

    return DataRuntime(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        data_feature=data_feature,
        feature_name=dict(feature_name),
        source="artifact",
        artifact_meta=artifact_meta,
        artifact_dir=Path(bundle["artifact_dir"]),
        warnings=warnings,
    )
