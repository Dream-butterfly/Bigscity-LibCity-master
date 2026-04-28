import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from GNNTP.utils import get_cache_subdir, get_run_dir
from GNNTP.utils.normalization import (
    LogScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    NoneScaler,
    NormalScaler,
    StandardScaler,
)


SIGNATURE_FIELDS = (
    "task",
    "dataset",
    "dataset_class",
    "seed",
    "train_rate",
    "eval_rate",
    "input_window",
    "output_window",
    "scaler",
    "ext_scaler",
    "load_external",
    "normal_external",
    "add_time_in_day",
    "add_day_in_week",
    "output_dim",
    "data_col",
    "ext_col",
    "data_files",
)

SCALER_KIND_KEY = "kind"
SCALER_PARAMS_KEY = "params"
RUN_META_FILE = "run_meta.json"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _as_config_dict(config_like: Any) -> dict[str, Any]:
    if hasattr(config_like, "config") and isinstance(getattr(config_like, "config"), dict):
        return dict(config_like.config)
    if isinstance(config_like, dict):
        return dict(config_like)
    raise TypeError("config_like must be dict-like or ConfigParser with `config` dict.")


def build_data_signature_payload(config_like: Any) -> dict[str, Any]:
    cfg = _as_config_dict(config_like)
    payload = {}
    for key in SIGNATURE_FIELDS:
        if key in cfg:
            payload[key] = cfg[key]
    return _to_jsonable(payload)


def compute_data_signature(payload: dict[str, Any]) -> str:
    raw = json.dumps(_to_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def get_data_artifacts_root() -> Path:
    return Path(get_cache_subdir("data_artifacts"))


def build_data_artifact_id(dataset: str, dataset_class: str, data_signature: str, now_ts: float | None = None) -> str:
    t = time.localtime(now_ts if now_ts is not None else time.time())
    ts = time.strftime("%Y%m%d_%H%M%S", t)
    return "da_{}__{}__{}__{}".format(
        ts,
        _safe_token(dataset),
        _safe_token(dataset_class),
        str(data_signature)[:8],
    )


def _assert_safe_artifact_id(artifact_id: str) -> None:
    text = str(artifact_id or "").strip()
    if not text:
        raise ValueError("artifact_id is required.")
    if any(sep in text for sep in ["/", "\\"]) or ".." in text:
        raise ValueError("Invalid artifact_id.")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", text):
        raise ValueError("artifact_id may only contain letters, digits, . _ : -")


def _assert_safe_run_id(run_id: str) -> None:
    text = str(run_id or "").strip()
    if not text:
        raise ValueError("run_id is required.")
    if any(sep in text for sep in ["/", "\\"]) or ".." in text:
        raise ValueError("Invalid run_id.")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+(?:__[A-Za-z0-9_.:-]+){0,3}", text):
        raise ValueError("run_id may only contain letters, digits, . _ : - and `__` separators.")


def resolve_data_artifact_dir(artifact_id: str | None = None, artifact_path: str | None = None) -> Path:
    if artifact_id and artifact_path:
        raise ValueError("Use either artifact_id or artifact_path, not both.")
    if artifact_path:
        path = Path(artifact_path).expanduser().resolve()
    else:
        _assert_safe_artifact_id(str(artifact_id or ""))
        path = get_data_artifacts_root() / str(artifact_id).strip()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError("Data artifact directory not found: {}".format(path))
    return path


def serialize_scaler(scaler: Any) -> dict[str, Any]:
    if scaler is None or isinstance(scaler, NoneScaler):
        return {SCALER_KIND_KEY: "none", SCALER_PARAMS_KEY: {}}
    if isinstance(scaler, NormalScaler):
        return {SCALER_KIND_KEY: "normal", SCALER_PARAMS_KEY: {"max": _to_jsonable(scaler.max)}}
    if isinstance(scaler, StandardScaler):
        return {
            SCALER_KIND_KEY: "standard",
            SCALER_PARAMS_KEY: {"mean": _to_jsonable(scaler.mean), "std": _to_jsonable(scaler.std)},
        }
    if isinstance(scaler, MinMax01Scaler):
        return {
            SCALER_KIND_KEY: "minmax01",
            SCALER_PARAMS_KEY: {"min": _to_jsonable(scaler.min), "max": _to_jsonable(scaler.max)},
        }
    if isinstance(scaler, MinMax11Scaler):
        return {
            SCALER_KIND_KEY: "minmax11",
            SCALER_PARAMS_KEY: {"min": _to_jsonable(scaler.min), "max": _to_jsonable(scaler.max)},
        }
    if isinstance(scaler, LogScaler):
        return {SCALER_KIND_KEY: "log", SCALER_PARAMS_KEY: {"eps": _to_jsonable(scaler.eps)}}
    raise TypeError("Unsupported scaler type: {}".format(type(scaler).__name__))


def _as_scalar_or_array(value: Any) -> Any:
    if isinstance(value, list):
        return np.asarray(value)
    return value


def deserialize_scaler(payload: dict[str, Any] | None) -> Any:
    if not isinstance(payload, dict):
        return NoneScaler()
    kind = str(payload.get(SCALER_KIND_KEY, "none")).strip().lower()
    params = payload.get(SCALER_PARAMS_KEY, {})
    if not isinstance(params, dict):
        params = {}
    if kind == "none":
        return NoneScaler()
    if kind == "normal":
        return NormalScaler(maxx=_as_scalar_or_array(params.get("max", 1.0)))
    if kind == "standard":
        return StandardScaler(
            mean=_as_scalar_or_array(params.get("mean", 0.0)),
            std=_as_scalar_or_array(params.get("std", 1.0)),
        )
    if kind == "minmax01":
        return MinMax01Scaler(
            minn=_as_scalar_or_array(params.get("min", 0.0)),
            maxx=_as_scalar_or_array(params.get("max", 1.0)),
        )
    if kind == "minmax11":
        return MinMax11Scaler(
            minn=_as_scalar_or_array(params.get("min", 0.0)),
            maxx=_as_scalar_or_array(params.get("max", 1.0)),
        )
    if kind == "log":
        return LogScaler(eps=_as_scalar_or_array(params.get("eps", 0.999)))
    raise ValueError("Unsupported scaler kind in payload: {}".format(kind))


def extract_xy_arrays_from_loader(loader: Any, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise ValueError("Loader for split `{}` has no dataset.".format(split_name))
    if hasattr(dataset, "arrays"):
        arrays = getattr(dataset, "arrays")
        if isinstance(arrays, tuple) and len(arrays) >= 2:
            return np.asarray(arrays[0]), np.asarray(arrays[1])
    rows = [dataset[idx] for idx in range(len(dataset))]
    if not rows:
        raise ValueError("Loader split `{}` is empty.".format(split_name))
    x = np.asarray([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows])
    return x, y


def write_data_artifact(
    artifact_dir: Path,
    *,
    artifact_id: str,
    task: str,
    model: str,
    dataset: str,
    dataset_class: str,
    feature_name: dict[str, Any],
    data_signature: str,
    data_signature_payload: dict[str, Any],
    config_snapshot: dict[str, Any],
    scaler_payload: dict[str, Any],
    ext_scaler_payload: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    data_feature: dict[str, Any],
    extra_meta: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if artifact_dir.exists() and not overwrite:
        raise FileExistsError("Artifact directory already exists: {}".format(artifact_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = artifact_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        train_x=np.asarray(train_x),
        train_y=np.asarray(train_y),
        valid_x=np.asarray(valid_x),
        valid_y=np.asarray(valid_y),
        test_x=np.asarray(test_x),
        test_y=np.asarray(test_y),
    )

    payload_data_feature = dict(data_feature or {})
    payload_data_feature.pop("scaler", None)
    payload_data_feature.pop("ext_scaler", None)
    with (artifact_dir / "data_feature.pkl").open("wb") as f:
        pickle.dump(payload_data_feature, f, protocol=pickle.HIGHEST_PROTOCOL)

    now_ts = time.time()
    meta = {
        "artifact_id": artifact_id,
        "created_at": now_ts,
        "status": "ready",
        "task": task,
        "model": model,
        "dataset": dataset,
        "dataset_class": dataset_class,
        "data_signature": data_signature,
        "data_signature_payload": _to_jsonable(data_signature_payload),
        "feature_name": _to_jsonable(feature_name or {"X": "float", "y": "float"}),
        "scaler": _to_jsonable(scaler_payload),
        "ext_scaler": _to_jsonable(ext_scaler_payload),
        "config_snapshot": _to_jsonable(config_snapshot),
        "train_shape": list(np.asarray(train_x).shape),
        "valid_shape": list(np.asarray(valid_x).shape),
        "test_shape": list(np.asarray(test_x).shape),
        "files": {
            "arrays": arrays_path.name,
            "data_feature": "data_feature.pkl",
        },
    }
    if extra_meta:
        meta.update(_to_jsonable(extra_meta))

    (artifact_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_data_artifact(artifact_id: str | None = None, artifact_path: str | None = None) -> dict[str, Any]:
    artifact_dir = resolve_data_artifact_dir(artifact_id=artifact_id, artifact_path=artifact_path)
    meta_path = artifact_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found in artifact: {}".format(artifact_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Invalid artifact meta: {}".format(meta_path))
    arrays_file = str(meta.get("files", {}).get("arrays", "arrays.npz"))
    data_feature_file = str(meta.get("files", {}).get("data_feature", "data_feature.pkl"))
    arrays_path = artifact_dir / arrays_file
    data_feature_path = artifact_dir / data_feature_file
    if not arrays_path.exists():
        raise FileNotFoundError("arrays file not found: {}".format(arrays_path))
    if not data_feature_path.exists():
        raise FileNotFoundError("data_feature file not found: {}".format(data_feature_path))

    arrays = np.load(arrays_path)
    with data_feature_path.open("rb") as f:
        data_feature = pickle.load(f)
    if not isinstance(data_feature, dict):
        raise ValueError("data_feature payload is not dict: {}".format(data_feature_path))

    bundle = {
        "artifact_dir": artifact_dir,
        "meta": meta,
        "train": (np.asarray(arrays["train_x"]), np.asarray(arrays["train_y"])),
        "valid": (np.asarray(arrays["valid_x"]), np.asarray(arrays["valid_y"])),
        "test": (np.asarray(arrays["test_x"]), np.asarray(arrays["test_y"])),
        "data_feature": data_feature,
    }
    return bundle


def list_artifact_metas(
    dataset: str | None = None,
    dataset_class: str | None = None,
) -> list[dict[str, Any]]:
    """扫描 artifacts 根目录，返回按 created_at 降序排列的 meta 列表。

    Args:
        dataset: 可选，仅返回匹配此数据集名的 artifact
        dataset_class: 可选，仅返回匹配此 dataset_class 的 artifact
    """
    root = get_data_artifacts_root()
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(meta, dict):
            continue
        if str(meta.get("status", "")).lower() != "ready":
            continue
        if dataset is not None and str(meta.get("dataset", "")) != dataset:
            continue
        if dataset_class is not None and str(meta.get("dataset_class", "")) != dataset_class:
            continue
        results.append(meta)
    results.sort(key=lambda m: float(m.get("created_at", 0)), reverse=True)
    return results


def find_latest_artifact(
    dataset: str,
    dataset_class: str,
    config_like: Any | None = None,
) -> dict[str, Any]:
    """查找最新且签名匹配的 artifact。

    扫描 data_artifacts/，按 (dataset, dataset_class) 过滤，
    选 created_at 最新的，若提供 config_like 则校验签名。
    返回 (artifact_id, artifact_dir, meta) 的字典。
    """
    metas = list_artifact_metas(dataset=dataset, dataset_class=dataset_class)
    if not metas:
        raise FileNotFoundError(
            "No ready artifact found for dataset={}, dataset_class={}.".format(
                dataset, dataset_class
            )
        )
    if config_like is not None:
        # 尝试所有候选，直到找到签名匹配的
        for meta in metas:
            try:
                validate_artifact_for_config(
                    artifact_meta=meta,
                    config_like=config_like,
                    model_dataset_class=dataset_class,
                    force_reuse=False,
                )
                artifact_id = str(meta.get("artifact_id", ""))
                return {
                    "artifact_id": artifact_id,
                    "artifact_dir": get_data_artifacts_root() / artifact_id,
                    "meta": meta,
                }
            except ValueError:
                continue
        raise ValueError(
            "Found {} artifact(s) for dataset={}, dataset_class={}, "
            "but none match the current config signature. "
            "Re-run data artifact build or use --force_reuse.".format(
                len(metas), dataset, dataset_class
            )
        )
    # 无 config → 返回最新的
    meta = metas[0]
    artifact_id = str(meta.get("artifact_id", ""))
    return {
        "artifact_id": artifact_id,
        "artifact_dir": get_data_artifacts_root() / artifact_id,
        "meta": meta,
    }


def _mismatch_items(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    keys = sorted(set(expected.keys()) | set(actual.keys()))
    mismatch = []
    for key in keys:
        if _to_jsonable(expected.get(key)) != _to_jsonable(actual.get(key)):
            mismatch.append(key)
    return mismatch


def validate_artifact_for_config(
    *,
    artifact_meta: dict[str, Any],
    config_like: Any,
    model_dataset_class: str,
    force_reuse: bool = False,
) -> list[str]:
    if not isinstance(artifact_meta, dict):
        raise TypeError("artifact_meta must be dict.")
    warnings = []
    if str(artifact_meta.get("status", "")).lower() != "ready":
        raise ValueError("Artifact status is not ready.")

    config_dict = _as_config_dict(config_like)
    task = str(config_dict.get("task", ""))
    dataset = str(config_dict.get("dataset", ""))
    dataset_class = str(config_dict.get("dataset_class", ""))
    model_dataset_class = str(model_dataset_class or "").strip()

    hard_checks = {
        "task": (str(artifact_meta.get("task", "")), task),
        "dataset": (str(artifact_meta.get("dataset", "")), dataset),
        "dataset_class": (str(artifact_meta.get("dataset_class", "")), dataset_class),
        "model_dataset_class": (str(artifact_meta.get("dataset_class", "")), model_dataset_class),
    }
    for key, (artifact_val, expected_val) in hard_checks.items():
        if artifact_val != expected_val:
            msg = "Artifact {} mismatch: artifact=`{}`, expected=`{}`.".format(key, artifact_val, expected_val)
            if force_reuse:
                warnings.append(msg)
            else:
                raise ValueError(msg)

    expected_payload = build_data_signature_payload(config_dict)
    expected_signature = compute_data_signature(expected_payload)
    artifact_signature = str(artifact_meta.get("data_signature", "")).strip()
    if artifact_signature != expected_signature:
        mismatch_keys = _mismatch_items(
            artifact_meta.get("data_signature_payload", {}),
            expected_payload,
        )
        msg = (
            "Artifact data_signature mismatch: artifact=`{}`, expected=`{}`."
            " Different fields: {}.".format(
                artifact_signature,
                expected_signature,
                ", ".join(mismatch_keys) if mismatch_keys else "<unknown>",
            )
        )
        if force_reuse:
            warnings.append(msg)
        else:
            raise ValueError(msg)

    return warnings


def write_run_meta(run_id: str, payload: dict[str, Any]) -> Path:
    _assert_safe_run_id(run_id)
    run_dir = Path(get_run_dir(run_id))
    run_meta_path = run_dir / RUN_META_FILE
    data = dict(payload or {})
    data["run_id"] = str(run_id)
    data["updated_at"] = time.time()
    if run_meta_path.exists():
        try:
            old = json.loads(run_meta_path.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                merged = dict(old)
                merged.update(data)
                data = merged
        except Exception:
            pass
    run_meta_path.write_text(json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
    return run_meta_path


def load_run_meta(run_id: str) -> dict[str, Any]:
    _assert_safe_run_id(run_id)
    run_dir = Path(get_run_dir(run_id))
    run_meta_path = run_dir / RUN_META_FILE
    if not run_meta_path.exists():
        raise FileNotFoundError("run_meta.json not found for run_id={}".format(run_id))
    obj = json.loads(run_meta_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Invalid run_meta.json for run_id={}".format(run_id))
    return obj
