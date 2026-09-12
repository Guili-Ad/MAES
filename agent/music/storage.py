from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

from agent.common import data_root

from .models import MusicFailureCode, MusicRunResult


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def preflight_path() -> Path:
    return data_root() / "music_preflight.json"


def touch_path() -> Path:
    return data_root() / "music_touch.json"


def result_path() -> Path:
    return data_root() / "music_last_result.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * 0.95) + 0.999999) - 1))
    return float(ordered[index])


def metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "p50": round(float(statistics.median(values)), 3),
        "p95": round(percentile_95(values), 3),
        "max": round(float(max(values)), 3),
    }


def controller_signature(context: Any, image: Any) -> str:
    controller = context.tasker.controller
    try:
        info = controller.info
        if callable(info):
            info = info()
    except Exception:
        info = {}
    if not isinstance(info, dict):
        info = {}
    shape = getattr(image, "shape", ())
    width = int(shape[1]) if len(shape) >= 2 else 0
    height = int(shape[0]) if len(shape) >= 2 else 0
    try:
        uuid = controller.uuid
        if callable(uuid):
            uuid = uuid()
    except Exception:
        uuid = ""
    try:
        raw_resolution = list(controller.resolution)
    except Exception:
        raw_resolution = [0, 0]
    payload = {
        "schema": "music-v4-native-action",
        "framework": "5.12.2",
        "controller_type": info.get("type", ""),
        "input_methods": info.get("input_methods", ""),
        "screencap_methods": info.get("screencap_methods", info.get("screencap", "")),
        "uuid": str(uuid),
        "raw_resolution": raw_resolution,
        "screenshot_size": [width, height],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_touch_state(signature: str) -> dict[str, Any]:
    state = read_json(touch_path())
    if state.get("controller_signature") != signature:
        return {
            "schema_version": 1,
            "controller_signature": signature,
            "basic": "untested",
            "advanced": "untested",
            "fused": False,
        }
    reason = str(state.get("fuse_reason", ""))
    completed_tap = reason.startswith("tap up lane ")
    if bool(state.get("fused")) and completed_tap and " exceeded soft budget:" in reason:
        # Older builds treated a successfully completed but briefly slow action as a
        # permanent backend failure.  Restore the already-probed capability once so
        # that an existing false fuse does not block the next song.
        recovered = dict(state)
        recovered.update({
            "fused": False,
            "fuse_reason": "",
            "recovered_fuse_reason": reason,
        })
        try:
            advanced_p95_ms = float(state.get("advanced_p95_ms", 0.0))
        except (TypeError, ValueError):
            advanced_p95_ms = 0.0
        if bool(state.get("multi_touch")) and advanced_p95_ms > 0.0:
            recovered["advanced"] = "passed"
        elif recovered.get("advanced") == "failed":
            recovered["advanced"] = "untested"
        atomic_write_json(touch_path(), recovered)
        return recovered
    return state


def save_touch_state(state: dict[str, Any]) -> None:
    atomic_write_json(touch_path(), state)


def fuse_touch_backend(signature: str, reason: str) -> None:
    state = load_touch_state(signature)
    state.update({
        "controller_signature": signature,
        "advanced": "failed",
        "fused": True,
        "fuse_reason": reason,
    })
    save_touch_state(state)


def write_result(result: MusicRunResult) -> None:
    atomic_write_json(result_path(), result.to_dict())


def read_result() -> MusicRunResult:
    raw = read_json(result_path())
    try:
        code = MusicFailureCode(str(raw.get("failure_code", MusicFailureCode.INTERNAL_ERROR.value)))
    except ValueError:
        code = MusicFailureCode.INTERNAL_ERROR
    return MusicRunResult(
        status=str(raw.get("status", "failed")),
        failure_code=code,
        reason=str(raw.get("reason", "No music result is available")),
        provider=str(raw.get("provider", "")),
        input_mode=str(raw.get("input_mode", raw.get("mode", "compatibility"))),
        controller_signature=str(raw.get("controller_signature", "")),
        profile=str(raw.get("profile", "")),
        task_id=raw.get("task_id"),
        metrics_ms=dict(raw.get("metrics_ms", {})),
        schema_version=int(raw.get("schema_version", 2)),
        time=str(raw.get("time", "")),
    )
