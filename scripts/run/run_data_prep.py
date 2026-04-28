"""
数据集预处理脚本：按 dataset_class 格式处理数据并保存为工件。

这是数据-训练解耦链路的数据准备入口。预处理不依赖模型，
仅需 --task --dataset --dataset_class。
产物可在训练时通过 --dataset 自动匹配。

用法：
    # 无模型模式（推荐）：纯数据集驱动
    uv run python scripts/run/run_data_prep.py \\
        --task traffic_state_pred \\
        --dataset METR_LA \\
        --dataset_class TrafficStatePointDataset

    # 有模型模式（兼容旧用法）：自动从 manifest 获取 dataset_class
    uv run python scripts/run/run_data_prep.py \\
        --task traffic_state_pred \\
        --model new_diffusion_fuzzy \\
        --dataset METR_LA
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run.run_data_artifact import run_data_artifact
from GNNTP.utils import add_general_args, str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="数据预处理：按 dataset_class 格式处理数据并保存为工件"
    )
    parser.add_argument("--task", type=str, default="traffic_state_pred", help="任务名")
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名（可选；不传则需显式指定 --dataset_class）"
    )
    parser.add_argument("--dataset", type=str, default="METR_LA", help="数据集名")
    parser.add_argument(
        "--dataset_class", type=str, default=None,
        help="数据集类名（如 TrafficStatePointDataset）；--model 不传时必须指定"
    )
    parser.add_argument("--config_file", type=str, default=None, help="外部配置文件")
    parser.add_argument("--artifact_id", type=str, default=None, help="自定义工件 ID")
    parser.add_argument(
        "--artifact_overwrite", type=str2bool, default=False, help="覆盖已有工件"
    )
    parser.add_argument("--exp_id", type=str, default=None, help="实验 ID")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--version_meta", type=str, default=None, help="元数据输出路径")
    add_general_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key
        not in [
            "task", "model", "dataset", "dataset_class",
            "config_file", "artifact_id", "artifact_overwrite",
            "exp_id", "seed", "version_meta",
        ]
        and val is not None
    }
    # 通过 other_args 或显式参数传递 dataset_class
    if args.dataset_class:
        other_args["dataset_class"] = args.dataset_class

    meta = run_data_artifact(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_class=args.dataset_class,
        config_file=args.config_file,
        artifact_id=args.artifact_id,
        artifact_overwrite=args.artifact_overwrite,
        other_args=other_args,
    )
    if args.version_meta:
        out = Path(args.version_meta)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
