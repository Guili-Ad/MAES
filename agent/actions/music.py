from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from agent.common import LOGGER, capture_image, parse_custom_param, recognition_hit, recognition_results
from agent.compat import AgentServer, Context, CustomAction, CustomRecognition
from agent.music.calibration import (
    CALIBRATION_VERSION,
    image_size,
    load_calibration,
    load_calibrations_for_resolution,
    save_calibration,
)
from agent.music.executor import (
    CompatibilityMusicInput,
    MusicActionExecutor,
    MusicTouchError,
    MusicTouchSession,
)
from agent.music.gestures import classify_note_gesture, patch_white_ratio, roi_white_ratio
from agent.music.models import (
    BASE_HEIGHT,
    BASE_WIDTH,
    FlickRequest,
    MusicCalibrationData,
    MusicConfig,
    MusicFailureCode,
    MusicRunResult,
    NoteGesture,
)
from agent.music.runtime import run_play_action, run_preflight_action, terminal_state
from agent.music.storage import (
    controller_signature,
    load_touch_state,
    percentile_95,
    read_result,
    save_touch_state,
    write_result,
)

try:
    from maa.pipeline import JActionType, JClick, JSwipe, JTouch, JTouchUp
except ModuleNotFoundError:  # pragma: no cover - MaaFramework is bundled in production
    JActionType = JClick = JSwipe = JTouch = JTouchUp = None  # type: ignore[assignment]


FAST_INPUT_METHOD_MASK = (1 << 1) | (1 << 2) | (1 << 3)
DEFAULT_CONTROLLER_CANDIDATE_MASK = (1 << 64) - 9
PROBE_POINTS = ((333, 320), (940, 320))


def _box_center(box: Any) -> tuple[int, int]:
    if box is None:
        raise ValueError("Recognition result has no box")
    values = list(box)
    if len(values) != 4:
        raise ValueError("Recognition result has an invalid box")
    return int(values[0]) + int(values[2]) // 2, int(values[1]) + int(values[3]) // 2


def _parse_input_methods(controller: Any) -> int:
    try:
        info = controller.info
        if callable(info):
            info = info()
        value = info.get("input_methods") if isinstance(info, dict) else 0
        if isinstance(value, str):
            value = int(value, 0)
        value = int(value)
        return 0 if value in (-9, DEFAULT_CONTROLLER_CANDIDATE_MASK) else value
    except Exception:
        return 0


def supports_fast_input(controller: Any) -> tuple[bool, int]:
    methods = _parse_input_methods(controller)
    return bool(methods & FAST_INPUT_METHOD_MASK), methods


def measure_capture_performance(
    context: Any,
    sample_count: int,
    *,
    capture_timeout_ms: int = 1200,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[float, float]:
    readings: list[float] = []
    for _ in range(sample_count):
        started = clock()
        image = capture_image(context, timeout_ms=capture_timeout_ms)
        readings.append((clock() - started) * 1000.0)
        if image is None:
            raise RuntimeError("Screenshot failed during music preflight")
    ordered = sorted(readings)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    return median, percentile_95(readings)


def _probe_point(point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    # Probe coordinates are always expressed in Framework screenshot space.
    return round(point[0] * width / BASE_WIDTH), round(point[1] * height / BASE_HEIGHT)


def _pause_visible(context: Any, image: Any) -> bool:
    detail = context.run_recognition("MusicPauseDialog", image)
    if detail is None:
        raise RuntimeError("Pause recognition did not start")
    return recognition_hit(detail)


def _action_succeeded(detail: Any) -> bool:
    if detail is None:
        return False
    success = getattr(detail, "success", None)
    return bool(success() if callable(success) else success)


def _run_probe_action(context: Any, action_type: Any, param: Any, budget_ms: float, description: str) -> float:
    started = time.perf_counter()
    detail = context.run_action_direct(action_type, param)
    elapsed = (time.perf_counter() - started) * 1000.0
    if not _action_succeeded(detail):
        raise MusicTouchError(f"{description} returned failure")
    if elapsed > budget_ms:
        raise MusicTouchError(f"{description} exceeded soft budget: {elapsed:.1f} ms")
    return elapsed


@AgentServer.custom_action("MusicTouchProbe")
class MusicTouchProbe(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        signature = ""
        active_contacts: set[int] = set()
        state: dict[str, Any] = {}
        probe_success = False
        try:
            config = MusicConfig.from_param(getattr(argv, "custom_action_param", None))
            image = capture_image(context, timeout_ms=config.capture_timeout_ms)
            if image is None or not _pause_visible(context, image):
                raise RuntimeError("Touch probe requires a confirmed song pause dialog")
            width, height = image_size(image)
            if (width, height) != (BASE_WIDTH, BASE_HEIGHT):
                raise RuntimeError("Touch probe requires a 1280x720 Framework screenshot")
            signature = controller_signature(context, image)
            state = load_touch_state(signature)
            left = _probe_point(PROBE_POINTS[0], width, height)
            right = _probe_point(PROBE_POINTS[1], width, height)
            if any(binding is None for binding in (JActionType, JClick, JSwipe, JTouch, JTouchUp)):
                raise RuntimeError("MaaFramework direct action bindings are unavailable")

            basic: list[float] = []
            basic.append(_run_probe_action(
                context,
                JActionType.TouchDown,
                JTouch(contact=0, target=(left[0], left[1], 1, 1), pressure=1),
                config.max_click_touch_ms,
                "safe-area TouchDown probe",
            ))
            basic.append(_run_probe_action(
                context,
                JActionType.TouchUp,
                JTouchUp(contact=0),
                config.max_click_touch_ms,
                "safe-area TouchUp probe",
            ))
            basic.append(_run_probe_action(
                context,
                JActionType.TouchDown,
                JTouch(contact=0, target=(left[0], left[1], 1, 1), pressure=1),
                config.max_click_touch_ms,
                "safe-area TouchMove probe",
            ))
            basic.append(_run_probe_action(
                context,
                JActionType.TouchMove,
                JTouch(contact=0, target=(left[0] + 4, left[1], 1, 1), pressure=1),
                config.max_click_touch_ms,
                "safe-area TouchMove probe",
            ))
            basic.append(_run_probe_action(
                context,
                JActionType.TouchUp,
                JTouchUp(contact=0),
                config.max_click_touch_ms,
                "safe-area TouchUp probe",
            ))
            checkpoint = capture_image(context, timeout_ms=config.capture_timeout_ms)
            if checkpoint is None or not _pause_visible(context, checkpoint):
                raise RuntimeError("Pause dialog disappeared during the basic touch probe")
            state.update({"basic": "passed", "basic_p95_ms": percentile_95(basic)})
            advanced: list[float] = []
            double_down_gaps: list[float] = []
            try:
                for sample in range(5):
                    first_down_started = time.perf_counter()
                    advanced.append(_run_probe_action(context, JActionType.TouchDown, JTouch(contact=0, target=(left[0], left[1], 1, 1), pressure=1), config.max_click_touch_ms, "TouchDown contact 0"))
                    active_contacts.add(0)
                    second_down_started = time.perf_counter()
                    double_down_gaps.append((second_down_started - first_down_started) * 1000.0)
                    advanced.append(_run_probe_action(context, JActionType.TouchDown, JTouch(contact=1, target=(right[0], right[1], 1, 1), pressure=1), config.max_click_touch_ms, "TouchDown contact 1"))
                    active_contacts.add(1)
                    if sample == 0:
                        advanced.append(_run_probe_action(context, JActionType.TouchMove, JTouch(contact=0, target=(left[0] + 4, left[1], 1, 1), pressure=1), config.max_click_touch_ms, "TouchMove contact 0"))
                        advanced.append(_run_probe_action(context, JActionType.TouchMove, JTouch(contact=1, target=(right[0] - 4, right[1], 1, 1), pressure=1), config.max_click_touch_ms, "TouchMove contact 1"))
                    advanced.append(_run_probe_action(context, JActionType.TouchUp, JTouchUp(contact=1), config.max_click_touch_ms, "TouchUp contact 1"))
                    active_contacts.discard(1)
                    advanced.append(_run_probe_action(context, JActionType.TouchUp, JTouchUp(contact=0), config.max_click_touch_ms, "TouchUp contact 0"))
                    active_contacts.discard(0)
                    checkpoint = capture_image(context, timeout_ms=config.capture_timeout_ms)
                    if checkpoint is None or not _pause_visible(context, checkpoint):
                        raise RuntimeError("Pause dialog disappeared during the advanced touch probe")
                down_gap_p95 = percentile_95(double_down_gaps)
                state.update({
                    "advanced": "passed",
                    "advanced_p95_ms": percentile_95(advanced),
                    "double_down_p95_ms": down_gap_p95,
                    "multi_touch": down_gap_p95 <= config.double_press_p95_ms,
                    "fused": False,
                    "fuse_reason": "",
                })
            except Exception as advanced_error:
                LOGGER.warning("Advanced music touch probe failed; basic Click/Swipe remain available: %s", advanced_error)
                state.update({
                    "advanced": "failed",
                    "multi_touch": False,
                    "fused": False,
                    "fuse_reason": str(advanced_error),
                })
            state.update({
                "schema_version": 1,
                "controller_signature": signature,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            save_touch_state(state)
            probe_success = True
        except Exception as error:
            LOGGER.error("Music touch probe failed: %s", error)
            if signature:
                state.update({
                    "schema_version": 1,
                    "controller_signature": signature,
                    "basic": "failed",
                    "advanced": "failed",
                    "fused": True,
                    "fuse_reason": str(error),
                    "time": datetime.now(timezone.utc).isoformat(),
                })
                save_touch_state(state)
        finally:
            if active_contacts and JActionType is not None and JTouchUp is not None:
                cleanup_failed = False
                for contact in sorted(active_contacts, reverse=True):
                    try:
                        _run_probe_action(context, JActionType.TouchUp, JTouchUp(contact=contact), 20.0, f"probe cleanup contact {contact}")
                    except Exception:
                        cleanup_failed = True
                        LOGGER.exception("Touch probe cleanup failed")
                if cleanup_failed and signature:
                    state.update({"fused": True, "advanced": "failed", "fuse_reason": "Probe contact cleanup failed"})
                    save_touch_state(state)
                    probe_success = False
        return probe_success


@AgentServer.custom_action("MusicCalibrate")
class MusicCalibrate(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            config = MusicConfig.from_param(getattr(argv, "custom_action_param", None))
            image = capture_image(context, timeout_ms=config.capture_timeout_ms)
            if image is None:
                raise RuntimeError("Screenshot unavailable")
            if _pause_visible(context, image):
                raise RuntimeError("Calibration must run on a live screen with visible lane targets, not the pause page")
            live = context.run_recognition("MusicLiveScreen", image)
            live_clear = context.run_recognition("MusicLiveClearScreen", image) if live is None or not recognition_hit(live) else None
            if (live is None or not recognition_hit(live)) and (live_clear is None or not recognition_hit(live_clear)):
                raise RuntimeError("Calibration requires a confirmed live concert screen")
            detail = context.run_recognition("MusicLaneTargets", image)
            centers = sorted({_box_center(getattr(item, "box", None)) for item in recognition_results(detail)}, key=lambda point: point[0])
            if len(centers) not in (7, 9):
                raise RuntimeError(f"Expected exactly 7 or 9 clear lane targets, recognized {len(centers)}")
            if config.lane_count and config.lane_count != len(centers):
                raise RuntimeError(f"Requested {config.lane_count} lanes but recognized {len(centers)}")
            data = save_calibration(image, centers)
            LOGGER.info("Saved V4 music calibration %s", data.profile_key)
            return True
        except Exception:
            LOGGER.exception("Music V4 calibration failed")
            return False


@AgentServer.custom_action("MusicPreflight")
class MusicPreflight(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            config = MusicConfig.from_param(getattr(argv, "custom_action_param", None))
            result = run_preflight_action(context, config, argv)
            if result.status != "succeeded":
                LOGGER.error("Music preflight failed [%s]: %s", result.failure_code.value, result.reason)
            return result.status == "succeeded"
        except Exception as error:
            result = MusicRunResult(status="failed", failure_code=MusicFailureCode.INTERNAL_ERROR, reason=f"Preflight action failed: {error}")
            write_result(result)
            LOGGER.exception("Music preflight action failed")
            return False


@AgentServer.custom_action("MusicPlay")
class MusicPlay(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            config = MusicConfig.from_param(getattr(argv, "custom_action_param", None))
            result = run_play_action(context, config, argv)
            if result.status != "succeeded":
                LOGGER.error("Music play failed [%s]: %s", result.failure_code.value, result.reason)
            return result.status == "succeeded"
        except Exception as error:
            result = MusicRunResult(status="failed", failure_code=MusicFailureCode.INTERNAL_ERROR, reason=f"MusicPlay action failed: {error}")
            write_result(result)
            LOGGER.exception("MusicPlay action failed")
            return False


@AgentServer.custom_recognition("MusicFailureCodeRecognition")
class MusicFailureCodeRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> Any:
        del context
        try:
            params = parse_custom_param(getattr(argv, "custom_recognition_param", None))
            expected = params.get("codes", params.get("code", []))
            if isinstance(expected, str):
                expected = [expected]
            result = read_result()
            current_task_id = getattr(getattr(argv, "task_detail", None), "task_id", None)
            if result.task_id is not None and current_task_id is not None and int(result.task_id) != int(current_task_id):
                return None
            if result.failure_code.value not in {str(value) for value in expected}:
                return None
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 1, 1),
                detail={"failure_code": result.failure_code.value, "reason": result.reason},
            )
        except Exception:
            LOGGER.exception("Music failure-code recognition failed")
            return None


@AgentServer.custom_action("ReportMusicFailure")
class ReportMusicFailure(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del context, argv
        result = read_result()
        LOGGER.error(
            "Music V4 failure [%s]: %s (provider=%s, input=%s, profile=%s)",
            result.failure_code.value,
            result.reason,
            result.provider or "unknown",
            result.input_mode,
            result.profile or "unknown",
        )
        return False


def _terminal_state(context: Any, image: Any) -> str:
    return terminal_state(context, image)


__all__ = [
    "CALIBRATION_VERSION",
    "CompatibilityMusicInput",
    "DEFAULT_CONTROLLER_CANDIDATE_MASK",
    "FAST_INPUT_METHOD_MASK",
    "FlickRequest",
    "MusicActionExecutor",
    "MusicCalibrationData",
    "MusicCalibrate",
    "MusicConfig",
    "MusicFailureCodeRecognition",
    "MusicPlay",
    "MusicPreflight",
    "MusicTouchError",
    "MusicTouchProbe",
    "MusicTouchSession",
    "NoteGesture",
    "ReportMusicFailure",
    "_probe_point",
    "_terminal_state",
    "classify_note_gesture",
    "controller_signature",
    "load_calibration",
    "load_calibrations_for_resolution",
    "measure_capture_performance",
    "patch_white_ratio",
    "roi_white_ratio",
    "save_calibration",
    "supports_fast_input",
]
