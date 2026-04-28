"""
FuzDiff (new_diffusion_fuzzy) 独立实验入口。

使用数据-训练解耦链路：
1. 先 run_data_prep.py --dataset <DATASET> --dataset_class TrafficStatePointDataset
2. 再本脚本 --dataset <DATASET>

训练时按 dataset + dataset_class 自动匹配最新工件。

用法示例：
    # 步骤1: 预处理数据
    uv run python scripts/run/run_data_prep.py --task traffic_state_pred --dataset METR_LA --dataset_class TrafficStatePointDataset

    # 步骤2: 训练
    uv run python scripts/experiments/train_new_diffusion_fuzzy.py --dataset METR_LA

    # 步骤2 (调参): 覆盖部分参数
    uv run python scripts/experiments/train_new_diffusion_fuzzy.py --dataset METR_LA --learning_rate 0.001 --max_epoch 100

    # 列出可用工件
    uv run python scripts/experiments/train_new_diffusion_fuzzy.py --dataset METR_LA --artifact_list
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run.run_train_artifact import run_train_artifact
from GNNTP.utils import add_general_args, str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FuzDiff (new_diffusion_fuzzy) 实验训练入口"
    )
    parser.add_argument("--task", type=str, default="traffic_state_pred")
    parser.add_argument("--dataset", type=str, default="METR_LA", help="数据集名")
    parser.add_argument("--config_file", type=str, default=None, help="外部配置文件")
    parser.add_argument("--saved_model", type=str2bool, default=True)
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--artifact_id", type=str, default=None, help="显式指定工件 ID")
    parser.add_argument("--artifact_path", type=str, default=None, help="显式指定工件路径")
    parser.add_argument("--artifact_list", action="store_true", default=False, help="列出可用工件")
    parser.add_argument("--force_reuse", type=str2bool, default=False, help="强制复用签名不匹配的工件")
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_general_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {
        key: val
        for key, val in dict_args.items()
        if key
        not in [
            "task", "model", "dataset", "config_file",
            "saved_model", "train", "artifact_id", "artifact_path",
            "artifact_list", "force_reuse", "exp_id", "seed",
        ]
        and val is not None
    }
    result = run_train_artifact(
        task=args.task,
        model_name="new_diffusion_fuzzy",
        dataset_name=args.dataset,
        config_file=args.config_file,
        saved_model=args.saved_model,
        train=args.train,
        artifact_id=args.artifact_id,
        artifact_path=args.artifact_path,
        artifact_list=args.artifact_list,
        force_reuse=args.force_reuse,
        other_args=other_args,
    )
    if result is not None:
        print("Test result:", result)
    else:
        print("Done (artifact list mode).")
