from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("maes.agent")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    configured = os.environ.get("MAES_DATA_DIR")
    if configured:
        root = Path(configured).expanduser()
    elif os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "MAES"
    else:
        root = project_root() / "local"
    root.mkdir(parents=True, exist_ok=True)
    return root


def parse_custom_param(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("custom_action_param must decode to an object")
        return parsed
    raise TypeError(f"Unsupported custom_action_param type: {type(value)!r}")


def is_stopping(context: Any) -> bool:
    return bool(getattr(getattr(context, "tasker", None), "stopping", False))


def recognition_results(detail: Any) -> list[Any]:
    if detail is None:
        return []
    for attribute in ("filtered_results", "filterd_results", "all_results"):
        value = getattr(detail, attribute, None)
        if value is not None:
            return list(value)
    best = getattr(detail, "best_result", None)
    return [best] if best is not None else []


def recognition_hit(detail: Any) -> bool:
    if detail is None:
        return False
    hit = getattr(detail, "hit", None)
    if hit is not None:
        return bool(hit)
    return bool(recognition_results(detail) or getattr(detail, "best_result", None))


def task_succeeded(detail: Any, expected_nodes: Iterable[str] = ()) -> bool:
    if detail is None:
        return False
    nodes = list(getattr(detail, "nodes", None) or [])
    expected = set(expected_nodes)
    if expected:
        for node in nodes:
            if getattr(node, "name", None) in expected:
                return bool(getattr(node, "completed", False))
        return False
    status = getattr(detail, "status", None)
    if status is not None:
        succeeded = getattr(status, "succeeded", None)
        if succeeded is not None:
            return bool(succeeded() if callable(succeeded) else succeeded)
    success = getattr(detail, "success", None)
    if success is not None:
        return bool(success)
    if status is not None:
        return str(status).lower() in {"success", "succeeded", "completed", "2"}
    if nodes:
        return all(bool(getattr(node, "completed", True)) for node in nodes)
    return bool(detail)


def run_task_checked(
    context: Any,
    node_name: str,
    *,
    pipeline_override: dict[str, Any] | None = None,
    expected_nodes: Iterable[str] = (),
) -> bool:
    try:
        if pipeline_override:
            detail = context.run_task(node_name, pipeline_override=pipeline_override)
        else:
            detail = context.run_task(node_name)
    except TypeError:
        detail = context.run_task(node_name, pipeline_override or {})
    except Exception:
        LOGGER.exception("Pipeline task %s raised an exception", node_name)
        return False
    success = task_succeeded(detail, expected_nodes)
    if not success:
        LOGGER.error("Pipeline task %s did not complete successfully", node_name)
    return success


def capture_image(
    context: Any,
    *,
    timeout_ms: int | None = None,
    poll_interval_ms: int = 5,
) -> Any | None:
    try:
        job = context.tasker.controller.post_screencap()
        if timeout_ms is None or not hasattr(job, "done"):
            return job.wait().get()

        deadline = time.monotonic() + max(1, timeout_ms) / 1000.0
        interval = max(1, poll_interval_ms) / 1000.0
        while not job.done:
            if bool(getattr(getattr(context, "tasker", None), "stopping", False)):
                LOGGER.warning("Screenshot capture cancelled while waiting for controller job %s", getattr(job, "job_id", "?"))
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                LOGGER.error(
                    "Screenshot capture timed out after %s ms while waiting for controller job %s",
                    timeout_ms,
                    getattr(job, "job_id", "?"),
                )
                # A timed-out native controller job can otherwise remain at
                # the head of the shared screenshot queue and make both the
                # task and live-view recovery loop wait forever.  Request an
                # asynchronous Tasker stop; never wait for that stop from
                # inside the custom action that is being interrupted.
                stop = getattr(getattr(context, "tasker", None), "post_stop", None)
                if callable(stop):
                    try:
                        stop()
                        LOGGER.warning("Requested Tasker stop after screenshot timeout")
                    except Exception:
                        LOGGER.exception("Unable to request Tasker stop after screenshot timeout")
                return None
            time.sleep(min(interval, remaining))
        if hasattr(job, "succeeded") and not job.succeeded:
            LOGGER.error("Screenshot controller job %s completed unsuccessfully", getattr(job, "job_id", "?"))
            return None
        return job.get()
    except Exception:
        LOGGER.exception("Screenshot capture failed")
        return None
