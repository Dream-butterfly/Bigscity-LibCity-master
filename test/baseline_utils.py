import json
import os
from typing import List

import numpy as np
import pandas as pd


def get_project_root(file_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(file_path), ".."))


def load_dataset_3d(config: dict) -> np.ndarray:
    dataset = config.get("dataset", "")
    data_dir = os.path.join("resource_data", dataset)
    config_path = os.path.join(data_dir, "config.json")
    dyna_path = os.path.join(data_dir, f"{dataset}.dyna")
    geo_path = os.path.join(data_dir, f"{dataset}.geo")

    with open(config_path, "r", encoding="utf-8") as f:
        dataset_config = json.load(f)
    for key, value in dataset_config.items():
        if key not in config:
            config[key] = value

    geo_file = pd.read_csv(geo_path)
    geo_ids = list(geo_file["geo_id"])

    dyna_file = pd.read_csv(dyna_path)
    data_col = config.get("data_col", "")
    if data_col != "":
        feature_cols: List[str] = data_col.copy() if isinstance(data_col, list) else [data_col]
    else:
        exclude_cols = {"dyna_id", "type", "time", "entity_id"}
        feature_cols = [c for c in dyna_file.columns if c not in exclude_cols]

    selected_cols = ["time", "entity_id"] + feature_cols
    dyna_file = dyna_file[selected_cols]
    time_slots = dyna_file["time"].drop_duplicates().tolist()

    data = np.zeros((len(time_slots), len(geo_ids), len(feature_cols)), dtype=float)
    for i, feature_col in enumerate(feature_cols):
        pivot = dyna_file.pivot(index="time", columns="entity_id", values=feature_col)
        pivot = pivot.reindex(index=time_slots, columns=geo_ids)
        data[:, :, i] = pivot.to_numpy(dtype=float)
    return data
