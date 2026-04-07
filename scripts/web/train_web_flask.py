"""
Flask web demo for LibCity model training and result visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from pyecharts_views import (
    build_loss_line_option,
    build_model_param_bar_option,
    build_metrics_table_html,
    build_model_param_pie_option,
    build_prediction_line_option,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_ROOT = PROJECT_ROOT / "libcity" / "models"
RESOURCE_DATA_ROOT = PROJECT_ROOT / "resource_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXP_ID_RE = re.compile(r"exp_id=([A-Za-z0-9_.:-]+)")
LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:,]+ - (?:INFO|WARNING|ERROR|DEBUG) - (.*)$")
MODEL_START_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\($")
PARAM_LINE_RE = re.compile(r"^([A-Za-z0-9_.]+)\s+torch\.Size\(\[([0-9,\s]+)\]\)")
TOTAL_PARAM_RE = re.compile(r"Total parameter numbers:\s*([0-9]+)")
EPOCH_LOSS_RE = re.compile(
    r"Epoch\s+\[(\d+)/(\d+)\]\s+train_loss:\s*([0-9.eE+-]+),\s*val_loss:\s*([0-9.eE+-]+)"
)
SAVED_EPOCH_RE = re.compile(r"Saved model at\s+([0-9]+)")
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

app = Flask(__name__, template_folder=str(WEB_ROOT / "templates"))
app.json.sort_keys = False


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

    arr = np.load(npz_files[0])
    pred = arr["prediction"]
    truth = arr["truth"]
    pred_series = pred[:, 0, 0, 0].astype(float)
    truth_series = truth[:, 0, 0, 0].astype(float)

    chart_payload = {
        "title": "Prediction vs Truth (horizon=1, node=0, feature=0)",
        "labels": list(range(1, len(pred_series) + 1)),
        "prediction": pred_series.tolist(),
        "truth": truth_series.tolist(),
    }
    return {
        "run_dir": str(run_dir),
        "metrics_csv": str(csv_files[0]),
        "predictions_npz": str(npz_files[0]),
        "metrics_summary": metrics_summary,
        "metrics_rows": metrics_rows,
        "metrics_columns": metrics_columns,
        "metrics_table_html": build_metrics_table_html(metrics_columns, metrics_rows),
        "chart": chart_payload,
        "chart_option": build_prediction_line_option(chart_payload),
        "shapes": {"prediction": list(pred.shape), "truth": list(truth.shape)},
    }


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


def _run_training_background(
    task: str,
    model: str,
    dataset: str,
    config_payload: dict[str, Any],
    saved_model: bool,
    train: bool,
    cli_options: dict[str, Any],
    extra_args: str,
) -> None:
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
        return

    try:
        config_file, runtime_config_path = _write_runtime_config(effective_config)
    except Exception as exc:
        STATE.reset([])
        STATE.finish(return_code=-1, error=f"Config write failed: {exc}", result=None)
        return
    cmd = [
        "uv",
        "run",
        "run_model.py",
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

    code = proc.wait()
    try:
        if code != 0:
            with STATE.lock:
                stopped = STATE.stop_requested
            if stopped:
                STATE.finish(code, "Training stopped by user.", None)
            else:
                STATE.finish(code, f"Training failed with return code {code}.", None)
            return
        run_dir = _find_run_dir(task, model, dataset, STATE.exp_id)
        if run_dir is None:
            STATE.finish(code, "Training finished but output run directory was not found.", None)
            return
        STATE.finish(code, None, _load_result_payload(run_dir))
    except Exception as exc:
        STATE.finish(code, f"Result parse error: {exc}", None)
    finally:
        _remove_runtime_config(runtime_config_path)


@app.route("/")
def index():
    return render_template("train_web_flask.html")


@app.route("/api/meta", methods=["GET"])
def api_meta():
    return jsonify({"models": _discover_models(), "datasets": _discover_datasets()})


@app.route("/api/default_config", methods=["POST"])
def api_default_config():
    body = request.get_json(silent=True) or {}
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
        return jsonify(
            {
                "task": task,
                "model": model,
                "dataset": dataset,
                "config": _to_jsonable(cfg),
                "executor_keys": executor_keys,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/start", methods=["POST"])
def api_start():
    with STATE.lock:
        if STATE.running:
            return jsonify({"error": "A training task is already running."}), 409

    body = request.get_json(silent=True) or {}
    task = str(body.get("task", "traffic_state_pred")).strip() or "traffic_state_pred"
    model = str(body.get("model", "STGCN")).strip() or "STGCN"
    dataset = str(body.get("dataset", "PEMSD4")).strip() or "PEMSD4"
    saved_model = _bool_from_any(body.get("saved_model", True), True)
    train = _bool_from_any(body.get("train", True), True)
    extra_args = str(body.get("extra_args", ""))
    config_payload = body.get("config", {})
    cli_options = body.get("cli_options", {})
    if not isinstance(config_payload, dict):
        return jsonify({"error": "config must be an object."}), 400
    if not isinstance(cli_options, dict):
        return jsonify({"error": "cli_options must be an object."}), 400

    t = threading.Thread(
        target=_run_training_background,
        args=(task, model, dataset, config_payload, saved_model, train, cli_options, extra_args),
        daemon=True,
    )
    t.start()
    return jsonify({"message": "Training started."})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with STATE.lock:
        proc = STATE.process
        running = STATE.running
        STATE.stop_requested = True
    if not running or proc is None:
        return jsonify({"error": "No running training task."}), 409
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
        return jsonify({"message": "Stop signal sent to process tree."})
    except Exception as exc:
        return jsonify({"error": f"Failed to stop process: {exc}"}), 500


@app.route("/api/clear", methods=["POST"])
def api_clear():
    with STATE.lock:
        if STATE.running:
            return jsonify({"error": "Training is running. Stop it before clearing state."}), 409
    STATE.clear()
    return jsonify({"message": "State cleared."})


@app.route("/api/status", methods=["GET"])
def api_status():
    def _parse_topk(name: str, default: int) -> int:
        raw = request.args.get(name, default)
        try:
            v = int(raw)
        except Exception:
            return default
        return max(1, min(100, v))

    pie_topk = _parse_topk("pie_topk", 8)
    bar_topk = _parse_topk("bar_topk", 12)
    with STATE.lock:
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
        return jsonify(
            {
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
        )


@app.route("/api/result", methods=["GET"])
def api_result():
    with STATE.lock:
        if STATE.result is None:
            return jsonify({"error": "Result not ready."}), 404
        return jsonify(STATE.result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flask web demo for LibCity training")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7817)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    print(f"Web server: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

