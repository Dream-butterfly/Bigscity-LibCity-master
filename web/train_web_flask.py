"""
FastAPI web demo for LibCity model training and result visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
if __package__:
    from .pyecharts_views import (
        build_loss_line_option,
        build_model_param_bar_option,
        build_metrics_table_html,
        build_model_param_pie_option,
        build_prediction_line_option,
    )
else:
    from pyecharts_views import (
        build_loss_line_option,
        build_model_param_bar_option,
        build_metrics_table_html,
        build_model_param_pie_option,
        build_prediction_line_option,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_ROOT = PROJECT_ROOT / "libcity" / "models"
RESOURCE_DATA_ROOT = PROJECT_ROOT / "resource_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUN_SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "run"
RUN_MODEL_ENTRY = str(RUN_SCRIPTS_DIR / "run_model.py")
RUN_RESUME_ENTRY = str(RUN_SCRIPTS_DIR / "run_resume.py")
RUN_DATA_PREP_ENTRY = str(RUN_SCRIPTS_DIR / "run_data_prep.py")
DATA_VERSIONS_DIR = OUTPUTS_DIR / "data_versions"
ACTIVE_DATA_VERSION_FILE = DATA_VERSIONS_DIR / "active_version.json"
TRAIN_HISTORY_FILE = OUTPUTS_DIR / "web_train_history.json"
TRAIN_HISTORY_MAX_ITEMS = 2000
TRAIN_HISTORY_LOCK = threading.Lock()
EXP_ID_RE = re.compile(r"exp_id=([A-Za-z0-9_.:-]+)")
LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:,]+ - (?:INFO|WARNING|ERROR|DEBUG) - (.*)$")
MODEL_START_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\($")
PARAM_LINE_RE = re.compile(r"^([A-Za-z0-9_.]+)\s+torch\.Size\(\[([0-9,\s]+)\]\)")
TOTAL_PARAM_RE = re.compile(r"Total parameter numbers:\s*([0-9]+)")
EPOCH_LOSS_RE = re.compile(
    r"Epoch\s+\[(\d+)/(\d+)\]\s+train_loss:\s*([0-9.eE+-]+),\s*val_loss:\s*([0-9.eE+-]+)"
)
SAVED_EPOCH_RE = re.compile(r"Saved model at\s+([0-9]+)")
CHECKPOINT_EPOCH_RE = re.compile(r"_epoch(\d+)\.tar$")
CLI_OPTION_KEYS = [
    "config_file",
    "exp_id",
    "seed",
    "gpu",
    "gpu_id",
    "train_rate",
    "eval_rate",
    "batch_size",
    "learning_rate",
    "max_epoch",
    "dataset_class",
    "executor",
    "evaluator",
]
CLI_OPTION_TYPE = {
    "config_file": "str",
    "exp_id": "str",
    "seed": "int",
    "gpu": "bool",
    "gpu_id": "int",
    "train_rate": "float",
    "eval_rate": "float",
    "batch_size": "int",
    "learning_rate": "float",
    "max_epoch": "int",
    "dataset_class": "str",
    "executor": "str",
    "evaluator": "str",
}
DATA_LOCKED_CONFIG_KEYS = {
    "dataset",
    "seed",
    "dataset_class",
    "train_rate",
    "eval_rate",
    "input_window",
    "output_window",
    "scaler",
    "load_external",
    "normal_external",
    "ext_scaler",
    "add_time_in_day",
    "add_day_in_week",
    "cache_dataset",
    "cache_file_name",
}
DATA_LOCKED_CLI_KEYS = {"config_file", "seed", "dataset_class", "train_rate", "eval_rate"}

app = FastAPI(title="LibCity Web Trainer")
app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _strip_log_prefix(line: str) -> str:
    m = LOG_PREFIX_RE.match(line)
    return m.group(1) if m else line


def _bool_from_any(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in {"1", "true", "yes"}:
            return True
        if v.lower() in {"0", "false", "no"}:
            return False
    return default


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except Exception:
        return None


def _normalize_cli_options(raw: dict[str, Any], allow_config_file: bool = True) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in CLI_OPTION_KEYS:
        if key == "config_file" and not allow_config_file:
            continue
        if key not in raw:
            continue
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text == "":
            continue
        typ = CLI_OPTION_TYPE.get(key, "str")
        if typ == "bool":
            b = _bool_from_any(value, default=False)
            if isinstance(value, str) and value.strip().lower() not in {"1", "0", "true", "false", "yes", "no"}:
                raise ValueError(f"Invalid boolean value for --{key}: {value}")
            out[key] = str(b).lower()
        elif typ == "int":
            try:
                out[key] = str(int(text))
            except Exception as exc:
                raise ValueError(f"Invalid integer value for --{key}: {value}") from exc
        elif typ == "float":
            try:
                out[key] = str(float(text))
            except Exception as exc:
                raise ValueError(f"Invalid float value for --{key}: {value}") from exc
        else:
            out[key] = text
    return out


def _build_data_version_split_profile(cli_options: dict[str, Any], config_payload: dict[str, Any]) -> dict[str, Any]:
    seed = _int_or_none(cli_options.get("seed"))
    if seed is None:
        seed = _int_or_none(config_payload.get("seed"))

    train_rate = _float_or_none(cli_options.get("train_rate"))
    if train_rate is None:
        train_rate = _float_or_none(config_payload.get("train_rate"))
    eval_rate = _float_or_none(cli_options.get("eval_rate"))
    if eval_rate is None:
        eval_rate = _float_or_none(config_payload.get("eval_rate"))

    test_rate = None
    if train_rate is not None and eval_rate is not None:
        test_rate = round(1.0 - train_rate - eval_rate, 6)
    return {
        "seed": seed,
        "train_rate": train_rate,
        "eval_rate": eval_rate,
        "test_rate": test_rate,
    }


def _validate_split_rates(config_payload: dict[str, Any], cli_options: dict[str, Any]) -> str | None:
    train_rate = _float_or_none(cli_options.get("train_rate"))
    if train_rate is None:
        train_rate = _float_or_none(config_payload.get("train_rate"))
    eval_rate = _float_or_none(cli_options.get("eval_rate"))
    if eval_rate is None:
        eval_rate = _float_or_none(config_payload.get("eval_rate"))
    if train_rate is None and eval_rate is None:
        return None
    if train_rate is None or eval_rate is None:
        return "train_rate and eval_rate must both be set when overriding split."
    if not (0.0 <= train_rate < 1.0):
        return "train_rate must be in [0, 1)."
    if not (0.0 <= eval_rate < 1.0):
        return "eval_rate must be in [0, 1)."
    if train_rate + eval_rate >= 1.0:
        return "train_rate + eval_rate must be < 1.0 to keep non-empty test split."
    return None


def _validate_epoch_pair(config_payload: dict[str, Any]) -> str | None:
    if "epoch" not in config_payload or "max_epoch" not in config_payload:
        return None
    epoch = _int_or_none(config_payload.get("epoch"))
    max_epoch = _int_or_none(config_payload.get("max_epoch"))
    if epoch is None or max_epoch is None:
        return "config.epoch and config.max_epoch must be integers."
    if epoch < 0:
        return "config.epoch must be >= 0."
    if max_epoch <= epoch:
        return "config.max_epoch must be greater than config.epoch."
    return None


def _safe_version_name(raw: str) -> str:
    name = str(raw or "").strip()
    if not name:
        raise ValueError("version_id is required.")
    if any(sep in name for sep in ["/", "\\"]) or ".." in name:
        raise ValueError("Invalid version_id.")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", name):
        raise ValueError("version_id may only contain letters, digits, . _ : -")
    return name


def _merge_with_data_version_constraints(
    prep_config_payload: dict[str, Any],
    user_config_payload: dict[str, Any],
    prep_cli_options: dict[str, Any],
    user_cli_options: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_payload = dict(prep_config_payload)
    config_payload.update(user_config_payload)
    for key in DATA_LOCKED_CONFIG_KEYS:
        if key in prep_config_payload:
            config_payload[key] = prep_config_payload[key]
        elif key in config_payload:
            config_payload.pop(key, None)

    cli_options = dict(prep_cli_options)
    cli_options.update(user_cli_options)
    for key in DATA_LOCKED_CLI_KEYS:
        if key in prep_cli_options:
            cli_options[key] = prep_cli_options[key]
        elif key in cli_options:
            cli_options.pop(key, None)
    return config_payload, cli_options


@lru_cache(maxsize=1)
def _discover_models() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for manifest in MODELS_ROOT.rglob("manifest.json"):
        try:
            with manifest.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if "task" in meta and "model" in meta:
                items.append(
                    {
                        "task": meta.get("task"),
                        "model": meta.get("model"),
                        "dataset_class": meta.get("dataset_class"),
                        "executor": meta.get("executor"),
                        "evaluator": meta.get("evaluator"),
                        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
                    }
                )
        except Exception:
            continue
    items.sort(key=lambda x: (str(x["task"]), str(x["model"])))
    return items


def _discover_datasets() -> list[str]:
    if not RESOURCE_DATA_ROOT.exists():
        return []
    ds = [p.name for p in RESOURCE_DATA_ROOT.iterdir() if p.is_dir() and (p / "config.json").exists()]
    ds.sort()
    return ds


def _resolve_dataset_preview_file(dataset: str, file_type: str) -> Path:
    ds = str(dataset or "").strip()
    typ = str(file_type or "").strip().lower()
    if not ds:
        raise ValueError("dataset is required.")
    if any(sep in ds for sep in ["/", "\\"]) or ".." in ds:
        raise ValueError("Invalid dataset name.")
    if typ not in {"dyna", "geo", "rel"}:
        raise ValueError("file_type must be one of: dyna, geo, rel.")
    dataset_dir = RESOURCE_DATA_ROOT / ds
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ds}")

    config_path = dataset_dir / "config.json"
    config_data: dict[str, Any] = {}
    if config_path.exists():
        try:
            obj = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                config_data = obj
        except Exception:
            config_data = {}
    info = config_data.get("info", {})
    if not isinstance(info, dict):
        info = {}

    candidate_stems: list[str] = []
    if typ == "geo":
        stem = str(info.get("geo_file", "")).strip()
        if stem:
            candidate_stems.append(stem)
    elif typ == "rel":
        stem = str(info.get("rel_file", "")).strip()
        if stem:
            candidate_stems.append(stem)
    else:
        files = info.get("data_files", [])
        if isinstance(files, list):
            for item in files:
                stem = str(item).strip()
                if stem:
                    candidate_stems.append(stem)
    candidate_stems.append(ds)

    seen: set[str] = set()
    for stem in candidate_stems:
        if stem in seen:
            continue
        seen.add(stem)
        file_path = dataset_dir / f"{stem}.{typ}"
        if file_path.exists() and file_path.is_file():
            return file_path
    raise FileNotFoundError(f"Dataset preview file not found for dataset={ds}, file_type={typ}")


def _load_dataset_preview(
    dataset: str,
    file_type: str,
    rows: int,
    entity_id: str = "",
    time_start: str = "",
    time_end: str = "",
    columns: list[str] | None = None,
) -> dict[str, Any]:
    limit = max(1, min(int(rows), 100))
    file_path = _resolve_dataset_preview_file(dataset, file_type)
    entity = str(entity_id or "").strip()
    ts = str(time_start or "").strip()
    te = str(time_end or "").strip()
    wanted_cols = [str(c).strip() for c in (columns or []) if str(c).strip()]

    chunks: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(file_path, chunksize=50000):
        cur = chunk
        if entity and "entity_id" in cur.columns:
            cur = cur[cur["entity_id"].astype(str) == entity]
        if ts and "time" in cur.columns:
            cur = cur[cur["time"].astype(str) >= ts]
        if te and "time" in cur.columns:
            cur = cur[cur["time"].astype(str) <= te]
        if cur.empty:
            continue
        chunks.append(cur)
        total += len(cur.index)
        if total > limit:
            break
    if chunks:
        df_full = pd.concat(chunks, ignore_index=True)
    else:
        df_full = pd.read_csv(file_path, nrows=0)
    has_more = len(df_full.index) > limit
    df = df_full.iloc[:limit].copy() if has_more else df_full
    if wanted_cols:
        existing = [c for c in wanted_cols if c in df.columns]
        if existing:
            df = df[existing]

    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return {
        "dataset": str(dataset),
        "file_type": str(file_type).lower(),
        "file_name": file_path.name,
        "columns": [str(c) for c in df.columns],
        "rows": _to_jsonable(records),
        "has_more": bool(has_more),
    }


def _get_manifest_meta(task: str, model: str) -> tuple[dict[str, Any], Path]:
    for item in _discover_models():
        if item.get("task") == task and item.get("model") == model:
            manifest_path = PROJECT_ROOT / str(item["manifest"])
            with manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f), manifest_path
    raise FileNotFoundError(f"Model manifest not found for task={task}, model={model}")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {}
    return obj


def _build_default_config_without_config_parser(task: str, model: str, dataset: str) -> dict[str, Any]:
    manifest, manifest_path = _get_manifest_meta(task, model)
    model_dir = manifest_path.parent

    dataset_class = manifest.get("dataset_class")
    executor = manifest.get("executor")
    evaluator = manifest.get("evaluator")
    if not dataset_class or not executor or not evaluator:
        raise ValueError(f"Manifest missing dataset/executor/evaluator for task={task}, model={model}")

    merged: dict[str, Any] = {
        "task": task,
        "model": model,
        "dataset": dataset,
        "saved_model": True,
        "train": True,
        "dataset_class": dataset_class,
        "executor": executor,
        "evaluator": evaluator,
    }

    # Follow ConfigParser's default merge priority:
    # model config -> dataset config -> executor config -> evaluator config -> resource_data config
    merged.update(_load_json_if_exists(model_dir / "config.json"))

    dataset_cfg = _load_json_if_exists(PROJECT_ROOT / "libcity" / "data" / "dataset" / f"{dataset_class}.json")
    merged.update(dataset_cfg)

    executor_cfg = _load_json_if_exists(model_dir / "executor.json")
    if not executor_cfg:
        executor_cfg = _load_json_if_exists(PROJECT_ROOT / "libcity" / "common" / f"{executor}.json")
    merged.update(executor_cfg)

    evaluator_cfg = _load_json_if_exists(PROJECT_ROOT / "libcity" / "common" / f"{evaluator}.json")
    merged.update(evaluator_cfg)

    dataset_config_path = RESOURCE_DATA_ROOT / dataset / "config.json"
    if dataset_config_path.exists():
        with dataset_config_path.open("r", encoding="utf-8") as f:
            data_cfg = json.load(f)
        if isinstance(data_cfg, dict):
            for k, v in data_cfg.items():
                if k == "info" and isinstance(v, dict):
                    merged.update(v)
                else:
                    merged[k] = v

    # Keep compatibility with original parser behavior.
    if model.upper() in {"LSTM", "GRU", "RNN"}:
        merged["rnn_type"] = model
        merged["model"] = "RNN"
    merged["max_epoch"] = 10

    return merged


def _find_run_dir(task: str, model: str, dataset: str, exp_id: str | None) -> Path | None:
    if exp_id:
        p = OUTPUTS_DIR / exp_id
        if p.exists() and p.is_dir():
            return p
    if not OUTPUTS_DIR.exists():
        return None
    candidates = [p for p in OUTPUTS_DIR.glob(f"*__{task}__{model}__{dataset}") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_run_dir_name(name: str) -> dict[str, str]:
    parts = name.split("__", 3)
    if len(parts) == 4:
        return {"exp_id": parts[0], "task": parts[1], "model": parts[2], "dataset": parts[3]}
    return {"exp_id": name, "task": "", "model": "", "dataset": ""}


def _collect_completed_runs(limit: int = 200) -> list[dict[str, Any]]:
    if not OUTPUTS_DIR.exists():
        return []
    runs: list[dict[str, Any]] = []
    dirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in dirs:
        eval_dir = run_dir / "evaluate_cache"
        if not eval_dir.exists():
            continue
        csv_files = sorted(eval_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        npz_files = sorted(eval_dir.glob("*_predictions.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csv_files or not npz_files:
            continue
        meta = _parse_run_dir_name(run_dir.name)
        runs.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "task": meta["task"],
                "model": meta["model"],
                "dataset": meta["dataset"],
                "mtime": run_dir.stat().st_mtime,
                "metrics_csv": str(csv_files[0]),
                "predictions_npz": str(npz_files[0]),
            }
        )
        if len(runs) >= limit:
            break
    return runs


def _collect_resume_runs(limit: int = 300) -> list[dict[str, Any]]:
    if not OUTPUTS_DIR.exists():
        return []
    runs: list[dict[str, Any]] = []
    dirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in dirs:
        cache_dir = run_dir / "model_cache"
        if not cache_dir.exists() or not cache_dir.is_dir():
            continue
        epochs: list[int] = []
        for ckpt in cache_dir.glob("*_epoch*.tar"):
            m = CHECKPOINT_EPOCH_RE.search(ckpt.name)
            if not m:
                continue
            try:
                epochs.append(int(m.group(1)))
            except Exception:
                continue
        if not epochs:
            continue
        epochs = sorted(set(epochs))
        meta = _parse_run_dir_name(run_dir.name)
        runs.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "task": meta["task"],
                "model": meta["model"],
                "dataset": meta["dataset"],
                "mtime": run_dir.stat().st_mtime,
                "epochs": epochs,
                "latest_epoch": epochs[-1],
            }
        )
        if len(runs) >= limit:
            break
    return runs


def _read_train_history_items() -> list[dict[str, Any]]:
    if not TRAIN_HISTORY_FILE.exists():
        return []
    try:
        obj = json.loads(TRAIN_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[dict[str, Any]] = []
    for row in obj:
        if isinstance(row, dict):
            out.append(row)
    return out


def _write_train_history_items(items: list[dict[str, Any]]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_HISTORY_FILE.write_text(json.dumps(_to_jsonable(items), ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_train_history_item(record_id: str, payload: dict[str, Any]) -> None:
    rid = str(record_id).strip()
    if not rid:
        raise ValueError("record_id is required.")
    with TRAIN_HISTORY_LOCK:
        items = _read_train_history_items()
        idx = -1
        for i, row in enumerate(items):
            if str(row.get("record_id", "")).strip() == rid:
                idx = i
                break
        old = items[idx] if idx >= 0 else {}
        row = dict(old)
        row.update(payload)
        row["record_id"] = rid
        if "created_at" not in row:
            row["created_at"] = time.time()
        row["updated_at"] = time.time()
        if idx >= 0:
            items[idx] = row
        else:
            items.append(row)
        items.sort(
            key=lambda x: float(
                x.get("ended_at", None)
                or x.get("started_at", None)
                or x.get("updated_at", None)
                or x.get("created_at", None)
                or 0
            ),
            reverse=True,
        )
        _write_train_history_items(items[:TRAIN_HISTORY_MAX_ITEMS])


def _build_major_metrics_from_summary(summary: dict[str, Any], limit: int = 3) -> dict[str, float]:
    if not isinstance(summary, dict):
        return {}
    preferred = ["MAE", "RMSE", "MAPE", "masked_MAE", "masked_RMSE", "masked_MAPE"]
    keys: list[str] = []
    for key in preferred:
        if key in summary:
            keys.append(key)
    for key in summary.keys():
        sk = str(key)
        if sk not in keys:
            keys.append(sk)
    out: dict[str, float] = {}
    for key in keys[: max(1, limit)]:
        item = summary.get(key, {})
        if isinstance(item, dict):
            val = item.get("avg", None)
        else:
            val = None
        if val is None:
            continue
        try:
            out[str(key)] = float(val)
        except Exception:
            continue
    return out


def _version_meta_path(version_id: str) -> Path:
    return DATA_VERSIONS_DIR / version_id / "meta.json"


def _safe_version_id() -> str:
    return f"dv_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _write_data_version_meta(version_id: str, payload: dict[str, Any]) -> None:
    path = _version_meta_path(version_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    old: dict[str, Any] = {}
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                old = obj
        except Exception:
            old = {}
    data = dict(old)
    data.update(payload)
    data["version_id"] = version_id
    data["updated_at"] = time.time()
    path.write_text(json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_data_version_meta(version_id: str) -> dict[str, Any] | None:
    if any(sep in version_id for sep in ["/", "\\"]) or ".." in version_id:
        return None
    path = _version_meta_path(version_id)
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    obj["version_id"] = version_id
    return obj


def _list_data_versions(limit: int = 300, only_ready: bool = False) -> list[dict[str, Any]]:
    if not DATA_VERSIONS_DIR.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in DATA_VERSIONS_DIR.iterdir():
        if not p.is_dir():
            continue
        meta = _read_data_version_meta(p.name)
        if not meta:
            continue
        if only_ready and str(meta.get("status", "")).lower() != "ready":
            continue
        items.append(meta)
    items.sort(key=lambda x: float(x.get("updated_at", 0) or 0), reverse=True)
    return items[:limit]


def _set_active_data_version(version_id: str | None) -> None:
    DATA_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"active_version_id": version_id or "", "updated_at": time.time()}
    ACTIVE_DATA_VERSION_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_active_data_version() -> str:
    if not ACTIVE_DATA_VERSION_FILE.exists():
        return ""
    try:
        obj = json.loads(ACTIVE_DATA_VERSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("active_version_id", "")).strip()


def _extract_major_metrics(metrics_csv: Path, limit: int = 3) -> dict[str, float]:
    try:
        df = pd.read_csv(metrics_csv)
    except Exception:
        return {}
    preferred = ["MAE", "RMSE", "MAPE", "masked_MAE", "masked_RMSE", "masked_MAPE"]
    cols: list[str] = []
    for c in preferred:
        if c in df.columns and c not in cols:
            cols.append(c)
    for c in df.columns:
        if c not in cols:
            cols.append(str(c))
    out: dict[str, float] = {}
    for c in cols[: max(1, limit)]:
        try:
            out[str(c)] = float(df[c].mean())
        except Exception:
            continue
    return out


def _build_history_items(limit: int = 20) -> list[dict[str, Any]]:
    with TRAIN_HISTORY_LOCK:
        history_rows = _read_train_history_items()
    items: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for row in history_rows:
        status = str(row.get("status", "finished")).strip().lower() or "finished"
        if status == "error":
            status = "failed"
        run_id = str(row.get("run_id", "")).strip()
        output_dir = str(row.get("output_dir", "")).strip()
        try:
            started_at = float(row.get("started_at")) if row.get("started_at") is not None else None
        except Exception:
            started_at = None
        try:
            ended_at = float(row.get("ended_at")) if row.get("ended_at") is not None else None
        except Exception:
            ended_at = None
        try:
            duration_sec = float(row.get("duration_sec")) if row.get("duration_sec") is not None else None
        except Exception:
            duration_sec = None
        if duration_sec is None and started_at is not None and ended_at is not None:
            duration_sec = max(0.0, ended_at - started_at)
        raw_metrics = row.get("major_metrics", {})
        major_metrics: dict[str, float] = {}
        if isinstance(raw_metrics, dict):
            for k, v in raw_metrics.items():
                try:
                    major_metrics[str(k)] = float(v)
                except Exception:
                    continue
        item = {
            "run_id": run_id,
            "task": str(row.get("task", "")),
            "model": str(row.get("model", "")),
            "dataset": str(row.get("dataset", "")),
            "status": status,
            "duration_sec": duration_sec,
            "started_at": started_at,
            "ended_at": ended_at,
            "major_metrics": major_metrics,
            "output_dir": output_dir,
        }
        items.append(item)
        if run_id:
            seen_run_ids.add(run_id)

    # Backward compatibility: include legacy finished runs that were never recorded.
    runs = _collect_completed_runs(limit=max(limit * 3, 50))
    for run in runs:
        run_id = str(run.get("run_id", "")).strip()
        if run_id and run_id in seen_run_ids:
            continue
        run_dir = Path(run["run_dir"])
        try:
            st = run_dir.stat()
            started_at = float(st.st_ctime)
            ended_at = float(st.st_mtime)
            duration_sec = max(0.0, ended_at - started_at)
        except Exception:
            started_at = None
            ended_at = None
            duration_sec = None
        items.append(
            {
                "run_id": run_id,
                "task": run.get("task", ""),
                "model": run.get("model", ""),
                "dataset": run.get("dataset", ""),
                "status": "finished",
                "duration_sec": duration_sec,
                "started_at": started_at,
                "ended_at": ended_at,
                "major_metrics": _extract_major_metrics(Path(run["metrics_csv"]), limit=3),
                "output_dir": run["run_dir"],
            }
        )
    items.sort(key=lambda x: float(x.get("ended_at", None) or x.get("started_at", None) or 0), reverse=True)
    return items[:limit]


def _downsample_xy(y1: list[float], y2: list[float], max_points: int = 600) -> tuple[list[float], list[float]]:
    n = min(len(y1), len(y2))
    if n <= max_points:
        return y1[:n], y2[:n]
    step = max(1, n // max_points)
    return y1[:n:step], y2[:n:step]


def _extract_prediction_series(
    npz_path: Path, horizon: int | None = None, node: int | None = None, feature: int | None = None
) -> dict[str, Any]:
    with np.load(npz_path) as arr:
        pred = arr["prediction"]
        truth = arr["truth"]
    if pred.shape != truth.shape:
        raise ValueError(f"prediction/truth shape mismatch: {pred.shape} vs {truth.shape}")
    if pred.ndim != 4:
        raise ValueError(f"prediction array must be 4-D, got shape: {pred.shape}")

    _, out_steps, num_nodes, num_features = pred.shape

    def _pick(value: int | None, upper: int, name: str) -> int:
        if upper <= 0:
            raise ValueError(f"{name} upper bound must be positive, got {upper}.")
        if value is None:
            return 1
        v = int(value)
        if v < 1 or v > upper:
            raise ValueError(f"{name} must be in [1, {upper}], got {v}.")
        return v

    horizon_sel = _pick(horizon, out_steps, "horizon")
    node_sel = _pick(node, num_nodes, "node")
    feature_sel = _pick(feature, num_features, "feature")

    pred_series = pred[:, horizon_sel - 1, node_sel - 1, feature_sel - 1].astype(float).tolist()
    truth_series = truth[:, horizon_sel - 1, node_sel - 1, feature_sel - 1].astype(float).tolist()
    pred_series, truth_series = _downsample_xy(pred_series, truth_series, max_points=600)

    chart_payload = {
        "title": "",
        "labels": list(range(1, len(pred_series) + 1)),
        "prediction": pred_series,
        "truth": truth_series,
    }
    return {
        "chart": chart_payload,
        "chart_option": build_prediction_line_option(chart_payload),
        "selection": {"horizon": horizon_sel, "node": node_sel, "feature": feature_sel},
        "ranges": {"horizon": out_steps, "node": num_nodes, "feature": num_features},
        "shapes": {"prediction": list(pred.shape), "truth": list(truth.shape)},
    }


def _load_result_payload(run_dir: Path) -> dict[str, Any]:
    eval_dir = run_dir / "evaluate_cache"
    if not eval_dir.exists():
        raise FileNotFoundError(f"evaluate_cache not found: {eval_dir}")

    csv_files = sorted(eval_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    npz_files = sorted(eval_dir.glob("*_predictions.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No metric csv found in {eval_dir}")
    if not npz_files:
        raise FileNotFoundError(f"No prediction npz found in {eval_dir}")

    metrics_df = pd.read_csv(csv_files[0])
    metrics_rows = []
    metrics_columns = ["horizon"] + [str(c) for c in metrics_df.columns]
    for i, row in metrics_df.iterrows():
        row_dict = {"horizon": int(i + 1)}
        for col in metrics_df.columns:
            row_dict[col] = float(row[col])
        metrics_rows.append(row_dict)

    metrics_summary = {
        col: {
            "h1": float(metrics_df[col].iloc[0]),
            "avg": float(metrics_df[col].mean()),
            "best": float(metrics_df[col].min()),
        }
        for col in metrics_df.columns
    }

    series_payload = _extract_prediction_series(npz_files[0], horizon=1, node=1, feature=1)
    return {
        "run_dir": str(run_dir),
        "metrics_csv": str(csv_files[0]),
        "predictions_npz": str(npz_files[0]),
        "metrics_summary": metrics_summary,
        "metrics_rows": metrics_rows,
        "metrics_columns": metrics_columns,
        "metrics_table_html": build_metrics_table_html(metrics_columns, metrics_rows),
        "chart": series_payload["chart"],
        "chart_option": series_payload["chart_option"],
        "shapes": series_payload["shapes"],
        "prediction_selector": {
            "selection": series_payload["selection"],
            "ranges": series_payload["ranges"],
        },
    }


def _build_compare_payload(run_ids: list[str]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for rid in run_ids:
        if any(sep in rid for sep in ["/", "\\"]) or ".." in rid:
            continue
        run_dir = OUTPUTS_DIR / rid
        if not run_dir.exists() or not run_dir.is_dir():
            continue
        payload = _load_result_payload(run_dir)
        meta = _parse_run_dir_name(rid)
        pred = [float(v) for v in (payload.get("chart", {}).get("prediction") or [])]
        truth = [float(v) for v in (payload.get("chart", {}).get("truth") or [])]
        pred, truth = _downsample_xy(pred, truth, max_points=600)
        items.append(
            {
                "run_id": rid,
                "task": meta["task"],
                "model": meta["model"],
                "dataset": meta["dataset"],
                "metrics_summary": payload.get("metrics_summary", {}),
                "prediction": pred,
                "truth": truth,
            }
        )
    if not items:
        return {"items": [], "chart_option": {}}

    max_len = max(max(len(x["prediction"]), len(x["truth"])) for x in items)
    xaxis = [str(i + 1) for i in range(max_len)]
    series: list[dict[str, Any]] = []
    for it in items:
        base = f"{it['model']}@{it['run_id'][:8]}"
        series.append({"name": f"{base}-pred", "type": "line", "symbol": "none", "data": it["prediction"]})
        series.append(
            {
                "name": f"{base}-truth",
                "type": "line",
                "symbol": "none",
                "lineStyle": {"type": "dashed"},
                "data": it["truth"],
            }
        )
    chart_option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll", "top": "4%"},
        "xAxis": {"type": "category", "name": "step", "data": xaxis},
        "yAxis": {"type": "value"},
        "dataZoom": [{"type": "inside", "start": 0, "end": 100}, {"type": "slider", "start": 0, "end": 100}],
        "series": series,
    }
    return {"items": items, "chart_option": chart_option}


@dataclass
class TrainState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    command: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    model_logs: list[str] = field(default_factory=list)
    model_param_rows: list[dict[str, Any]] = field(default_factory=list)
    model_total_params: int | None = None
    loss_points: list[dict[str, Any]] = field(default_factory=list)
    saved_epochs: list[int] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    exp_id: str | None = None
    return_code: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    started_at: float | None = None
    ended_at: float | None = None
    _capturing_model_repr: bool = False
    stop_requested: bool = False
    model_plot_topk_pie: int = 8
    model_plot_topk_bar: int = 12

    def reset(self, command: list[str]) -> None:
        with self.lock:
            self.running = True
            self.command = command
            self.logs = []
            self.model_logs = []
            self.model_param_rows = []
            self.model_total_params = None
            self.loss_points = []
            self.saved_epochs = []
            self.process = None
            self.exp_id = None
            self.return_code = None
            self.error = None
            self.result = None
            self.started_at = time.time()
            self.ended_at = None
            self._capturing_model_repr = False
            self.stop_requested = False

    def append_log(self, line: str) -> None:
        line = line.rstrip("\n")
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 4000:
                self.logs = self.logs[-4000:]
        self._extract_model_log(line)

    def _append_model_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            if self.model_logs and self.model_logs[-1] == line:
                return
            self.model_logs.append(line)
            if len(self.model_logs) > 1500:
                self.model_logs = self.model_logs[-1500:]

    def _upsert_param_row(self, name: str, shape: list[int]) -> None:
        count = 1
        for d in shape:
            count *= d
        with self.lock:
            for row in self.model_param_rows:
                if row["name"] == name:
                    row["shape"] = shape
                    row["count"] = count
                    return
            self.model_param_rows.append({"name": name, "shape": shape, "count": count})

    def _upsert_loss_point(self, epoch: int, total_epoch: int, train_loss: float, val_loss: float) -> None:
        with self.lock:
            for row in self.loss_points:
                if int(row["epoch"]) == epoch:
                    row["total_epoch"] = total_epoch
                    row["train_loss"] = train_loss
                    row["val_loss"] = val_loss
                    return
            self.loss_points.append(
                {"epoch": epoch, "total_epoch": total_epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            self.loss_points.sort(key=lambda x: int(x["epoch"]))

    def _extract_model_log(self, raw_line: str) -> None:
        msg = _strip_log_prefix(raw_line).rstrip()
        if not msg:
            return
        if msg.startswith("{'task':") and "'model':" in msg:
            self._append_model_log("[Config] " + msg)
            return
        if MODEL_START_RE.match(msg.strip()):
            with self.lock:
                self._capturing_model_repr = True
            self._append_model_log(msg)
            return
        with self.lock:
            capturing = self._capturing_model_repr
        if capturing:
            self._append_model_log(msg)
            if msg.strip() == ")":
                with self.lock:
                    self._capturing_model_repr = False
            return
        if "torch.Size(" in msg or "Total parameter numbers" in msg:
            self._append_model_log(msg)
        pm = PARAM_LINE_RE.match(msg.strip())
        if pm:
            dims = [int(x.strip()) for x in pm.group(2).split(",") if x.strip()]
            if dims:
                self._upsert_param_row(pm.group(1), dims)
        tm = TOTAL_PARAM_RE.search(msg)
        if tm:
            with self.lock:
                self.model_total_params = int(tm.group(1))
        lm = EPOCH_LOSS_RE.search(msg)
        if lm:
            self._upsert_loss_point(
                epoch=int(lm.group(1)),
                total_epoch=int(lm.group(2)),
                train_loss=float(lm.group(3)),
                val_loss=float(lm.group(4)),
            )
        sm = SAVED_EPOCH_RE.search(msg)
        if sm:
            e = int(sm.group(1))
            with self.lock:
                if e not in self.saved_epochs:
                    self.saved_epochs.append(e)
                    self.saved_epochs.sort()

    def finish(self, return_code: int, error: str | None, result: dict[str, Any] | None) -> None:
        with self.lock:
            self.running = False
            self.return_code = return_code
            self.error = error
            self.result = result
            self.ended_at = time.time()
            self.process = None
            self._capturing_model_repr = False

    def clear(self) -> None:
        with self.lock:
            self.running = False
            self.command = []
            self.logs = []
            self.model_logs = []
            self.model_param_rows = []
            self.model_total_params = None
            self.loss_points = []
            self.saved_epochs = []
            self.process = None
            self.exp_id = None
            self.return_code = None
            self.error = None
            self.result = None
            self.started_at = None
            self.ended_at = None
            self._capturing_model_repr = False
            self.stop_requested = False


STATE = TrainState()


@dataclass
class DataPrepState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    command: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    version_id: str | None = None
    return_code: int | None = None
    error: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    stop_requested: bool = False

    def reset(self, command: list[str], version_id: str) -> None:
        with self.lock:
            self.running = True
            self.command = command
            self.logs = []
            self.process = None
            self.version_id = version_id
            self.return_code = None
            self.error = None
            self.started_at = time.time()
            self.ended_at = None
            self.stop_requested = False

    def append_log(self, line: str) -> None:
        line = line.rstrip("\n")
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 3000:
                self.logs = self.logs[-3000:]

    def finish(self, return_code: int, error: str | None) -> None:
        with self.lock:
            self.running = False
            self.return_code = return_code
            self.error = error
            self.ended_at = time.time()
            self.process = None


DATA_STATE = DataPrepState()


def _write_runtime_config(config_data: dict[str, Any]) -> tuple[str, Path]:
    fd, tmp_path = tempfile.mkstemp(prefix="webcfg_", suffix=".json", dir=str(PROJECT_ROOT))
    os.close(fd)
    path = Path(tmp_path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config_data), f, ensure_ascii=False, indent=2)
    # config_parser loads './<config_file>.json'; pass file stem only.
    return path.stem, path


def _remove_runtime_config(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _run_data_prep_background(
    version_id: str,
    task: str,
    model: str,
    dataset: str,
    config_payload: dict[str, Any],
    cli_options: dict[str, Any],
    extra_args: str,
) -> None:
    runtime_config_path: Path | None = None
    version_meta_out = _version_meta_path(version_id).parent / "script_meta.json"
    split_profile = _build_data_version_split_profile(cli_options=cli_options, config_payload=config_payload)
    try:
        effective_config = {}
        base_name = str(cli_options.get("config_file", "")).strip()
        if base_name:
            effective_config.update(_load_base_config(base_name))
        effective_config.update(config_payload)
    except Exception as exc:
        DATA_STATE.reset([], version_id=version_id)
        DATA_STATE.finish(return_code=-1, error=f"Config preparation failed: {exc}")
        _write_data_version_meta(version_id, {"status": "failed", "error": str(exc), "task": task, "model": model, "dataset": dataset})
        return

    try:
        config_file, runtime_config_path = _write_runtime_config(effective_config)
    except Exception as exc:
        DATA_STATE.reset([], version_id=version_id)
        DATA_STATE.finish(return_code=-1, error=f"Config write failed: {exc}")
        _write_data_version_meta(version_id, {"status": "failed", "error": str(exc), "task": task, "model": model, "dataset": dataset})
        return

    cmd = [
        "uv",
        "run",
        RUN_DATA_PREP_ENTRY,
        "--task",
        task,
        "--model",
        model,
        "--dataset",
        dataset,
        "--config_file",
        config_file,
        "--version_meta",
        str(version_meta_out),
    ]
    for key in CLI_OPTION_KEYS:
        if key == "config_file":
            continue
        value = cli_options.get(key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text == "":
            continue
        if key == "gpu":
            text = text.lower()
        cmd.extend([f"--{key}", text])
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args, posix=False))

    _write_data_version_meta(
        version_id,
        {
            "status": "processing",
            "task": task,
            "model": model,
            "dataset": dataset,
            "seed": split_profile.get("seed"),
            "train_rate": split_profile.get("train_rate"),
            "eval_rate": split_profile.get("eval_rate"),
            "test_rate": split_profile.get("test_rate"),
            "cli_options": cli_options,
            "config_payload": config_payload,
            "created_at": time.time(),
        },
    )
    DATA_STATE.reset(cmd, version_id=version_id)
    DATA_STATE.append_log("$ " + " ".join(cmd))
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except Exception as exc:
        _remove_runtime_config(runtime_config_path)
        DATA_STATE.finish(return_code=-1, error=f"Failed to start process: {exc}")
        _write_data_version_meta(version_id, {"status": "failed", "error": f"Failed to start process: {exc}"})
        return

    with DATA_STATE.lock:
        DATA_STATE.process = proc

    assert proc.stdout is not None
    for line in proc.stdout:
        DATA_STATE.append_log(line)
    code = proc.wait()
    try:
        if code != 0:
            with DATA_STATE.lock:
                stopped = DATA_STATE.stop_requested
            err = "Data processing stopped by user." if stopped else f"Data processing failed with return code {code}."
            DATA_STATE.finish(code, err)
            _write_data_version_meta(version_id, {"status": "failed", "error": err})
            return
        script_meta: dict[str, Any] = {}
        if version_meta_out.exists():
            try:
                obj = json.loads(version_meta_out.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    script_meta = obj
            except Exception:
                script_meta = {}
        DATA_STATE.finish(code, None)
        _write_data_version_meta(
            version_id,
            {
                "status": "ready",
                "error": None,
                "script_meta": script_meta,
            },
        )
        _set_active_data_version(version_id)
    except Exception as exc:
        DATA_STATE.finish(code, f"Data result parse error: {exc}")
        _write_data_version_meta(version_id, {"status": "failed", "error": f"Data result parse error: {exc}"})
    finally:
        _remove_runtime_config(runtime_config_path)


def _load_base_config(config_file_name: str) -> dict[str, Any]:
    name = config_file_name.strip()
    if not name:
        return {}
    path = PROJECT_ROOT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Base config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Base config must be an object: {path}")
    return obj


def _resolve_data_version(version_id: str) -> tuple[dict[str, Any], str]:
    vid = str(version_id or "").strip()
    if not vid:
        active = _get_active_data_version()
        vid = active
    meta = _read_data_version_meta(vid) if vid else None
    if not meta:
        raise FileNotFoundError("Data version not found.")
    if str(meta.get("status", "")).lower() != "ready":
        raise ValueError("Data version is not ready.")
    return meta, vid


def _run_training_background(
    task: str,
    model: str,
    dataset: str,
    config_payload: dict[str, Any],
    saved_model: bool,
    train: bool,
    cli_options: dict[str, Any],
    extra_args: str,
    entry_script: str = RUN_MODEL_ENTRY,
    history_record_id: str | None = None,
) -> None:
    history_id = str(history_record_id or "").strip()
    runtime_config_path: Path | None = None
    try:
        effective_config = {}
        base_name = str(cli_options.get("config_file", "")).strip()
        if base_name:
            effective_config.update(_load_base_config(base_name))
        effective_config.update(config_payload)
    except Exception as exc:
        STATE.reset([])
        STATE.finish(return_code=-1, error=f"Config preparation failed: {exc}", result=None)
        if history_id:
            _upsert_train_history_item(
                history_id,
                {
                    "task": task,
                    "model": model,
                    "dataset": dataset,
                    "status": "failed",
                    "run_id": str(cli_options.get("exp_id", "")).strip(),
                    "ended_at": time.time(),
                    "duration_sec": None,
                    "major_metrics": {},
                    "output_dir": "",
                    "error": f"Config preparation failed: {exc}",
                },
            )
        return

    try:
        config_file, runtime_config_path = _write_runtime_config(effective_config)
    except Exception as exc:
        STATE.reset([])
        STATE.finish(return_code=-1, error=f"Config write failed: {exc}", result=None)
        if history_id:
            _upsert_train_history_item(
                history_id,
                {
                    "task": task,
                    "model": model,
                    "dataset": dataset,
                    "status": "failed",
                    "run_id": str(cli_options.get("exp_id", "")).strip(),
                    "ended_at": time.time(),
                    "duration_sec": None,
                    "major_metrics": {},
                    "output_dir": "",
                    "error": f"Config write failed: {exc}",
                },
            )
        return
    cmd = [
        "uv",
        "run",
        entry_script,
        "--task",
        task,
        "--model",
        model,
        "--dataset",
        dataset,
        "--config_file",
        config_file,
        "--saved_model",
        str(saved_model).lower(),
        "--train",
        str(train).lower(),
    ]
    for key in CLI_OPTION_KEYS:
        if key == "config_file":
            continue
        value = cli_options.get(key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text == "":
            continue
        if key == "gpu":
            text = text.lower()
        cmd.extend([f"--{key}", text])
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args, posix=False))

    STATE.reset(cmd)
    STATE.append_log("$ " + " ".join(cmd))
    if history_id:
        _upsert_train_history_item(
            history_id,
            {
                "task": task,
                "model": model,
                "dataset": dataset,
                "status": "running",
                "run_id": str(cli_options.get("exp_id", "")).strip(),
                "started_at": STATE.started_at,
                "ended_at": None,
                "duration_sec": None,
                "major_metrics": {},
                "output_dir": "",
                "error": None,
            },
        )
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except Exception as exc:
        _remove_runtime_config(runtime_config_path)
        STATE.finish(return_code=-1, error=f"Failed to start process: {exc}", result=None)
        if history_id:
            with STATE.lock:
                started_at = STATE.started_at
            ended_at = time.time()
            duration_sec = max(0.0, ended_at - started_at) if started_at is not None else None
            _upsert_train_history_item(
                history_id,
                {
                    "status": "failed",
                    "ended_at": ended_at,
                    "duration_sec": duration_sec,
                    "error": f"Failed to start process: {exc}",
                },
            )
        return

    with STATE.lock:
        STATE.process = proc

    assert proc.stdout is not None
    for line in proc.stdout:
        STATE.append_log(line)
        m = EXP_ID_RE.search(line)
        if m:
            with STATE.lock:
                STATE.exp_id = m.group(1)
                current_started_at = STATE.started_at
            if history_id:
                _upsert_train_history_item(
                    history_id,
                    {
                        "run_id": m.group(1),
                        "started_at": current_started_at,
                    },
                )

    code = proc.wait()
    try:
        if code != 0:
            with STATE.lock:
                stopped = STATE.stop_requested
                started_at = STATE.started_at
            if stopped:
                STATE.finish(code, "Training stopped by user.", None)
                status = "stopped"
                error = "Training stopped by user."
            else:
                STATE.finish(code, f"Training failed with return code {code}.", None)
                status = "failed"
                error = f"Training failed with return code {code}."
            if history_id:
                ended_at = time.time()
                duration_sec = max(0.0, ended_at - started_at) if started_at is not None else None
                _upsert_train_history_item(
                    history_id,
                    {
                        "status": status,
                        "ended_at": ended_at,
                        "duration_sec": duration_sec,
                        "error": error,
                    },
                )
            return
        with STATE.lock:
            started_at = STATE.started_at
        run_dir = _find_run_dir(task, model, dataset, STATE.exp_id)
        if run_dir is None:
            STATE.finish(code, "Training finished but output run directory was not found.", None)
            if history_id:
                ended_at = time.time()
                duration_sec = max(0.0, ended_at - started_at) if started_at is not None else None
                _upsert_train_history_item(
                    history_id,
                    {
                        "status": "failed",
                        "ended_at": ended_at,
                        "duration_sec": duration_sec,
                        "error": "Training finished but output run directory was not found.",
                    },
                )
            return
        result = _load_result_payload(run_dir)
        STATE.finish(code, None, result)
        if history_id:
            ended_at = time.time()
            duration_sec = max(0.0, ended_at - started_at) if started_at is not None else None
            _upsert_train_history_item(
                history_id,
                {
                    "run_id": run_dir.name,
                    "status": "finished",
                    "ended_at": ended_at,
                    "duration_sec": duration_sec,
                    "major_metrics": _build_major_metrics_from_summary(result.get("metrics_summary", {}), limit=3),
                    "output_dir": str(run_dir),
                    "error": None,
                },
            )
    except Exception as exc:
        STATE.finish(code, f"Result parse error: {exc}", None)
        if history_id:
            with STATE.lock:
                started_at = STATE.started_at
            ended_at = time.time()
            duration_sec = max(0.0, ended_at - started_at) if started_at is not None else None
            _upsert_train_history_item(
                history_id,
                {
                    "status": "failed",
                    "ended_at": ended_at,
                    "duration_sec": duration_sec,
                    "error": f"Result parse error: {exc}",
                },
            )
    finally:
        _remove_runtime_config(runtime_config_path)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="train_web_flask.html")


@app.get("/api/meta")
def api_meta():
    return {
        "models": _discover_models(),
        "datasets": _discover_datasets(),
        "data_versions": _list_data_versions(limit=300, only_ready=True),
        "active_data_version_id": _get_active_data_version(),
    }


@app.post("/api/data/start")
async def api_data_start(request: Request):
    with DATA_STATE.lock:
        if DATA_STATE.running:
            return JSONResponse(status_code=409, content={"error": "A data processing task is already running."})
    body = await request.json()
    task = str(body.get("task", "traffic_state_pred")).strip() or "traffic_state_pred"
    model = str(body.get("model", "STGCN")).strip() or "STGCN"
    dataset = str(body.get("dataset", "PEMSD4")).strip() or "PEMSD4"
    extra_args = str(body.get("extra_args", ""))
    config_payload = body.get("config", {})
    cli_options = body.get("cli_options", {})
    if not isinstance(config_payload, dict):
        return JSONResponse(status_code=400, content={"error": "config must be an object."})
    if not isinstance(cli_options, dict):
        return JSONResponse(status_code=400, content={"error": "cli_options must be an object."})
    try:
        cli_options = _normalize_cli_options(cli_options, allow_config_file=True)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if "train_rate" not in cli_options:
        cli_options["train_rate"] = "0.6"
    if "eval_rate" not in cli_options:
        cli_options["eval_rate"] = "0.2"
    split_error = _validate_split_rates(config_payload, cli_options)
    if split_error:
        return JSONResponse(status_code=400, content={"error": split_error})
    version_id = _safe_version_id()
    t = threading.Thread(
        target=_run_data_prep_background,
        args=(version_id, task, model, dataset, config_payload, cli_options, extra_args),
        daemon=True,
    )
    t.start()
    return {"message": "Data processing started.", "version_id": version_id}


@app.post("/api/data/stop")
def api_data_stop():
    with DATA_STATE.lock:
        proc = DATA_STATE.process
        running = DATA_STATE.running
        DATA_STATE.stop_requested = True
    if not running or proc is None:
        return JSONResponse(status_code=409, content={"error": "No running data processing task."})
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
        return {"message": "Stop signal sent to data processing task."}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Failed to stop data process: {exc}"})


@app.get("/api/data/status")
def api_data_status():
    with DATA_STATE.lock:
        return {
            "running": DATA_STATE.running,
            "command": DATA_STATE.command,
            "version_id": DATA_STATE.version_id,
            "return_code": DATA_STATE.return_code,
            "error": DATA_STATE.error,
            "logs_tail": DATA_STATE.logs[-400:],
            "started_at": DATA_STATE.started_at,
            "ended_at": DATA_STATE.ended_at,
        }


@app.get("/api/data/preview")
def api_data_preview(
    dataset: str = Query(...),
    file_type: str = Query(default="dyna"),
    rows: int = Query(default=20, ge=1, le=100),
    entity_id: str = Query(default=""),
    time_start: str = Query(default=""),
    time_end: str = Query(default=""),
    columns: str = Query(default=""),
):
    try:
        cols = [c.strip() for c in str(columns).split(",") if c.strip()]
        return _load_dataset_preview(
            dataset=dataset,
            file_type=file_type,
            rows=rows,
            entity_id=entity_id,
            time_start=time_start,
            time_end=time_end,
            columns=cols,
        )
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Failed to load dataset preview: {exc}"})


@app.get("/api/data/versions")
def api_data_versions(ready_only: bool = Query(default=False), dataset: str = Query(default="")):
    ds = str(dataset or "").strip()
    versions = _list_data_versions(limit=300, only_ready=bool(ready_only))
    if ds:
        versions = [v for v in versions if str(v.get("dataset", "")).strip() == ds]
    return {
        "versions": versions,
        "active_data_version_id": _get_active_data_version(),
    }


@app.post("/api/data/version/note")
async def api_data_version_note(request: Request):
    body = await request.json()
    try:
        version_id = _safe_version_name(body.get("version_id", ""))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    note = str(body.get("note", "")).strip()
    meta = _read_data_version_meta(version_id)
    if not meta:
        return JSONResponse(status_code=404, content={"error": "Data version not found."})
    _write_data_version_meta(version_id, {"note": note})
    return {"message": "Note updated.", "version_id": version_id, "note": note}


@app.post("/api/data/version/delete")
async def api_data_version_delete(request: Request):
    body = await request.json()
    try:
        version_id = _safe_version_name(body.get("version_id", ""))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    version_dir = DATA_VERSIONS_DIR / version_id
    if not version_dir.exists() or not version_dir.is_dir():
        return JSONResponse(status_code=404, content={"error": "Data version not found."})
    active = _get_active_data_version()
    if active == version_id:
        _set_active_data_version("")
    try:
        shutil.rmtree(version_dir)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Failed to delete data version: {exc}"})
    return {"message": "Data version deleted.", "version_id": version_id}


@app.post("/api/data/version/rename")
async def api_data_version_rename(request: Request):
    body = await request.json()
    try:
        version_id = _safe_version_name(body.get("version_id", ""))
        new_version_id = _safe_version_name(body.get("new_version_id", ""))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if version_id == new_version_id:
        return JSONResponse(status_code=400, content={"error": "new_version_id must differ from version_id."})
    src_dir = DATA_VERSIONS_DIR / version_id
    dst_dir = DATA_VERSIONS_DIR / new_version_id
    if not src_dir.exists() or not src_dir.is_dir():
        return JSONResponse(status_code=404, content={"error": "Data version not found."})
    if dst_dir.exists():
        return JSONResponse(status_code=409, content={"error": "new_version_id already exists."})
    src_dir.rename(dst_dir)
    old_meta_path = dst_dir / "meta.json"
    if old_meta_path.exists():
        try:
            obj = json.loads(old_meta_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj["version_id"] = new_version_id
                old_meta_path.write_text(json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    active = _get_active_data_version()
    if active == version_id:
        _set_active_data_version(new_version_id)
    return {"message": "Data version renamed.", "version_id": new_version_id}


@app.post("/api/data/active")
async def api_data_active(request: Request):
    body = await request.json()
    version_id = str(body.get("version_id", "")).strip()
    if version_id:
        meta = _read_data_version_meta(version_id)
        if not meta:
            return JSONResponse(status_code=404, content={"error": "Data version not found."})
        if str(meta.get("status", "")).lower() != "ready":
            return JSONResponse(status_code=400, content={"error": "Only ready data versions can be activated."})
    _set_active_data_version(version_id)
    return {"active_data_version_id": version_id}


@app.post("/api/default_config")
async def api_default_config(request: Request):
    body = await request.json()
    task = str(body.get("task", "traffic_state_pred")).strip()
    model = str(body.get("model", "STGCN")).strip()
    dataset = str(body.get("dataset", "PEMSD4")).strip()
    try:
        cfg = _build_default_config_without_config_parser(task=task, model=model, dataset=dataset)
        manifest, manifest_path = _get_manifest_meta(task, model)
        model_dir = manifest_path.parent
        executor = manifest.get("executor")
        executor_cfg = _load_json_if_exists(model_dir / "executor.json")
        if not executor_cfg and executor:
            executor_cfg = _load_json_if_exists(PROJECT_ROOT / "libcity" / "common" / f"{executor}.json")
        executor_keys = sorted([str(k) for k in executor_cfg.keys()])
        return {
            "task": task,
            "model": model,
            "dataset": dataset,
            "config": _to_jsonable(cfg),
            "executor_keys": executor_keys,
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/start")
async def api_start(request: Request):
    with STATE.lock:
        if STATE.running:
            return JSONResponse(status_code=409, content={"error": "A training task is already running."})

    body = await request.json()
    data_version_id = str(body.get("data_version_id", "")).strip()
    try:
        data_meta, resolved_version_id = _resolve_data_version(data_version_id)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Invalid data_version_id: {exc}"})
    task = str(data_meta.get("task", body.get("task", "traffic_state_pred"))).strip() or "traffic_state_pred"
    model = str(data_meta.get("model", body.get("model", "STGCN"))).strip() or "STGCN"
    dataset = str(data_meta.get("dataset", body.get("dataset", "PEMSD4"))).strip() or "PEMSD4"
    saved_model = _bool_from_any(body.get("saved_model", True), True)
    train = _bool_from_any(body.get("train", True), True)
    extra_args = str(body.get("extra_args", ""))
    user_config_payload = body.get("config", {})
    cli_options = body.get("cli_options", {})
    if not isinstance(user_config_payload, dict):
        return JSONResponse(status_code=400, content={"error": "config must be an object."})
    if not isinstance(cli_options, dict):
        return JSONResponse(status_code=400, content={"error": "cli_options must be an object."})
    try:
        cli_options = _normalize_cli_options(cli_options, allow_config_file=True)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    split_error = _validate_split_rates(user_config_payload, cli_options)
    if split_error:
        return JSONResponse(status_code=400, content={"error": split_error})
    epoch_error = _validate_epoch_pair(user_config_payload)
    if epoch_error:
        return JSONResponse(status_code=400, content={"error": epoch_error})
    prep_config_payload = data_meta.get("config_payload", {})
    if not isinstance(prep_config_payload, dict):
        prep_config_payload = {}
    prep_cli_options = data_meta.get("cli_options", {})
    if not isinstance(prep_cli_options, dict):
        prep_cli_options = {}
    config_payload, cli_options = _merge_with_data_version_constraints(
        prep_config_payload=prep_config_payload,
        user_config_payload=user_config_payload,
        prep_cli_options=prep_cli_options,
        user_cli_options=cli_options,
    )
    config_payload["data_version_id"] = resolved_version_id
    history_record_id = f"tr_{int(time.time())}_{uuid.uuid4().hex[:10]}"
    _upsert_train_history_item(
        history_record_id,
        {
            "task": task,
            "model": model,
            "dataset": dataset,
            "status": "running",
            "run_id": str(cli_options.get("exp_id", "")).strip(),
            "started_at": time.time(),
            "ended_at": None,
            "duration_sec": None,
            "major_metrics": {},
            "output_dir": "",
            "error": None,
        },
    )

    t = threading.Thread(
        target=_run_training_background,
        args=(task, model, dataset, config_payload, saved_model, train, cli_options, extra_args, RUN_MODEL_ENTRY, history_record_id),
        daemon=True,
    )
    t.start()
    return {"message": "Training started."}


@app.post("/api/start_resume")
async def api_start_resume(request: Request):
    with STATE.lock:
        if STATE.running:
            return JSONResponse(status_code=409, content={"error": "A training task is already running."})

    body = await request.json()
    data_version_id = str(body.get("data_version_id", "")).strip()
    try:
        data_meta, resolved_version_id = _resolve_data_version(data_version_id)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Invalid data_version_id: {exc}"})
    task = str(data_meta.get("task", body.get("task", "traffic_state_pred"))).strip() or "traffic_state_pred"
    model = str(data_meta.get("model", body.get("model", "STGCN"))).strip() or "STGCN"
    dataset = str(data_meta.get("dataset", body.get("dataset", "PEMSD4"))).strip() or "PEMSD4"
    saved_model = _bool_from_any(body.get("saved_model", True), True)
    extra_args = str(body.get("extra_args", ""))
    user_config_payload = body.get("config", {})
    cli_options = body.get("cli_options", {})
    if not isinstance(user_config_payload, dict):
        return JSONResponse(status_code=400, content={"error": "config must be an object."})
    if not isinstance(cli_options, dict):
        return JSONResponse(status_code=400, content={"error": "cli_options must be an object."})
    try:
        cli_options = _normalize_cli_options(cli_options, allow_config_file=False)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    split_error = _validate_split_rates(user_config_payload, cli_options)
    if split_error:
        return JSONResponse(status_code=400, content={"error": split_error})

    exp_id = str(cli_options.get("exp_id", "")).strip()
    if not exp_id:
        return JSONResponse(status_code=400, content={"error": "Resume training requires cli_options.exp_id."})
    prep_config_payload = data_meta.get("config_payload", {})
    if not isinstance(prep_config_payload, dict):
        prep_config_payload = {}
    config_payload, _ = _merge_with_data_version_constraints(
        prep_config_payload=prep_config_payload,
        user_config_payload=user_config_payload,
        prep_cli_options={},
        user_cli_options={},
    )
    config_payload["data_version_id"] = resolved_version_id
    prep_cli_options = data_meta.get("cli_options", {})
    if not isinstance(prep_cli_options, dict):
        prep_cli_options = {}
    merged_cli_options = {k: v for k, v in prep_cli_options.items() if k != "config_file" and k not in DATA_LOCKED_CLI_KEYS}
    merged_cli_options["exp_id"] = exp_id
    cli_options = merged_cli_options

    epoch_raw = config_payload.get("epoch", None)
    max_epoch_raw = config_payload.get("max_epoch", None)
    try:
        epoch = int(epoch_raw)
        max_epoch = int(max_epoch_raw)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Resume training requires integer config.epoch and config.max_epoch."})
    if epoch < 0:
        return JSONResponse(status_code=400, content={"error": "config.epoch must be >= 0."})
    if max_epoch <= epoch:
        return JSONResponse(status_code=400, content={"error": "config.max_epoch must be greater than config.epoch."})
    history_record_id = f"tr_{int(time.time())}_{uuid.uuid4().hex[:10]}"
    _upsert_train_history_item(
        history_record_id,
        {
            "task": task,
            "model": model,
            "dataset": dataset,
            "status": "running",
            "run_id": str(cli_options.get("exp_id", "")).strip(),
            "started_at": time.time(),
            "ended_at": None,
            "duration_sec": None,
            "major_metrics": {},
            "output_dir": "",
            "error": None,
        },
    )

    t = threading.Thread(
        target=_run_training_background,
        args=(task, model, dataset, config_payload, saved_model, True, cli_options, extra_args, RUN_RESUME_ENTRY, history_record_id),
        daemon=True,
    )
    t.start()
    return {"message": "Resume training started."}


@app.post("/api/stop")
def api_stop():
    with STATE.lock:
        proc = STATE.process
        running = STATE.running
        STATE.stop_requested = True
    if not running or proc is None:
        return JSONResponse(status_code=409, content={"error": "No running training task."})
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
        return {"message": "Stop signal sent to process tree."}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Failed to stop process: {exc}"})


@app.post("/api/clear")
def api_clear():
    with STATE.lock:
        if STATE.running:
            return JSONResponse(status_code=409, content={"error": "Training is running. Stop it before clearing state."})
    STATE.clear()
    return {"message": "State cleared."}


@app.get("/api/status")
def api_status():
    with STATE.lock:
        pie_topk = max(1, min(100, int(STATE.model_plot_topk_pie)))
        bar_topk = max(1, min(100, int(STATE.model_plot_topk_bar)))
        top_rows = sorted(STATE.model_param_rows, key=lambda x: int(x["count"]), reverse=True)[:20]
        loss_epochs = [int(p["epoch"]) for p in STATE.loss_points]
        train_losses = [float(p["train_loss"]) for p in STATE.loss_points]
        val_losses = [float(p["val_loss"]) for p in STATE.loss_points]
        max_epoch = max([int(p["total_epoch"]) for p in STATE.loss_points], default=None)
        model_plot = {
            "labels": [row["name"] for row in top_rows],
            "counts": [int(row["count"]) for row in top_rows],
            "shapes": [row["shape"] for row in top_rows],
            "total_params": STATE.model_total_params,
        }
        loss_plot = {
            "epochs": loss_epochs,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "max_epoch": max_epoch,
            "saved_epochs": STATE.saved_epochs,
        }
        option_errors: list[str] = []
        try:
            model_plot_option_pie = build_model_param_pie_option(model_plot, topk=pie_topk)
        except Exception as exc:
            model_plot_option_pie = {}
            option_errors.append(f"model_plot_option_pie: {exc}")
        try:
            model_plot_option_bar = build_model_param_bar_option(model_plot, topk=bar_topk)
        except Exception as exc:
            model_plot_option_bar = {}
            option_errors.append(f"model_plot_option_bar: {exc}")
        try:
            loss_plot_option = build_loss_line_option(loss_plot)
        except Exception as exc:
            loss_plot_option = {}
            option_errors.append(f"loss_plot_option: {exc}")
        return {
            "running": STATE.running,
            "command": STATE.command,
            "exp_id": STATE.exp_id,
            "return_code": STATE.return_code,
            "error": STATE.error,
            "result_ready": STATE.result is not None,
            "logs_tail": STATE.logs[-400:],
            "model_logs_tail": STATE.model_logs[-300:],
            "model_plot": model_plot,
            "model_plot_option_pie": model_plot_option_pie,
            "model_plot_option_bar": model_plot_option_bar,
            "model_plot_topk": {"pie": pie_topk, "bar": bar_topk},
            "loss_plot": loss_plot,
            "loss_plot_option": loss_plot_option,
            "option_errors": option_errors,
            "started_at": STATE.started_at,
            "ended_at": STATE.ended_at,
        }


@app.post("/api/plot_settings")
async def api_plot_settings(request: Request):
    body = await request.json()

    def _parse_topk(value: Any, default: int) -> int:
        try:
            v = int(value)
        except Exception:
            return default
        return max(1, min(100, v))

    with STATE.lock:
        pie = _parse_topk(body.get("pie_topk", STATE.model_plot_topk_pie), STATE.model_plot_topk_pie)
        bar = _parse_topk(body.get("bar_topk", STATE.model_plot_topk_bar), STATE.model_plot_topk_bar)
        STATE.model_plot_topk_pie = pie
        STATE.model_plot_topk_bar = bar
    return {"model_plot_topk": {"pie": pie, "bar": bar}}


@app.get("/api/result")
def api_result():
    with STATE.lock:
        if STATE.result is None:
            return JSONResponse(status_code=404, content={"error": "Result not ready."})
        return STATE.result


@app.get("/api/result_series")
def api_result_series(
    horizon: int = Query(default=1, ge=1),
    node: int = Query(default=1, ge=1),
    feature: int = Query(default=1, ge=1),
):
    with STATE.lock:
        if STATE.result is None:
            return JSONResponse(status_code=404, content={"error": "Result not ready."})
        npz_path = STATE.result.get("predictions_npz")
    if not npz_path:
        return JSONResponse(status_code=404, content={"error": "Prediction npz path is missing."})
    try:
        return _extract_prediction_series(Path(str(npz_path)), horizon=horizon, node=node, feature=feature)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/api/runs")
def api_runs():
    runs = _collect_completed_runs(limit=300)
    return {"runs": runs}


@app.get("/api/resume_runs")
def api_resume_runs():
    runs = _collect_resume_runs(limit=300)
    return {"runs": runs}


@app.get("/api/history")
def api_history(limit: int = Query(default=20, ge=1, le=200)):
    limit = max(1, min(200, limit))
    return {"items": _build_history_items(limit)}


@app.post("/api/compare")
async def api_compare(request: Request):
    body = await request.json()
    run_ids = body.get("run_ids", [])
    if not isinstance(run_ids, list):
        return JSONResponse(status_code=400, content={"error": "run_ids must be an array."})
    run_ids = [str(x).strip() for x in run_ids if str(x).strip()]
    if not run_ids:
        return JSONResponse(status_code=400, content={"error": "No run_ids provided."})
    try:
        return _build_compare_payload(run_ids)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI web demo for LibCity training")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7817)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    print(f"Web server: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug" if args.debug else "info")


if __name__ == "__main__":
    main()

