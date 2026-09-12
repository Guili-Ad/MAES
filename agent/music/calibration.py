from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agent.common import LOGGER, data_root

from .models import (
    BASE_HEIGHT,
    BASE_WIDTH,
    SUPPORTED_LANE_COUNTS,
    VISUAL_BASELINE_VERSION,
    MusicCalibrationData,
)
from .storage import atomic_write_json


CALIBRATION_VERSION = 4


def calibration_path() -> Path:
    return data_root() / "calibration" / "music.json"


def image_size(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("Screenshot does not expose a valid shape")
    return int(shape[1]), int(shape[0])


def _profile_key(lane_count: int, width: int, height: int) -> str:
    return f"{lane_count}@{width}x{height}"


def _read_profiles() -> dict[str, dict[str, Any]]:
    path = calibration_path()
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("version", 0)) != CALIBRATION_VERSION:
        raise ValueError("Music calibration format is outdated; V4 recalibration is required")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Music calibration profiles are missing")
    return dict(profiles)


def _clamped_roi(roi: Iterable[int], width: int, height: int) -> list[int]:
    x, y, roi_width, roi_height = (int(value) for value in roi)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return [x, y, max(1, min(roi_width, width - x)), max(1, min(roi_height, height - y))]


def _judgement_arc_origin(points: list[tuple[int, int]], height: int) -> tuple[float, float]:
    """Recover the common note origin from the circular judgement arc.

    The seven judgement points are samples of one circle and the notes travel
    radially from its centre.  The former V4 geometry started every lane on a
    horizontal band at 23% screen height.  That made an inward-moving hold cap
    cross several *model* lanes even though it remained on one physical ray,
    and could clamp right-to-left tail progress to 1.0 far above judgement.

    The arc is horizontally symmetric, so its median x is the circle x and the
    lowest (middle) judgement point supplies a stable radius reference.  Each
    remaining point independently estimates the circle y; the median limits
    the influence of near-centre samples whose small vertical denominator
    magnifies one-pixel calibration noise.
    """
    if len(points) < 3:
        raise ValueError("At least three judgement points are required")
    center_x = float(statistics.median(x for x, _y in points))
    reference_x, reference_y = max(points, key=lambda item: item[1])
    reference_dx = float(reference_x) - center_x
    estimates: list[float] = []
    for x, y in points:
        denominator = 2.0 * (float(reference_y) - float(y))
        if abs(denominator) < max(24.0, height * 0.05):
            continue
        dx = float(x) - center_x
        numerator = (
            reference_dx * reference_dx
            + float(reference_y) * float(reference_y)
            - dx * dx
            - float(y) * float(y)
        )
        estimates.append(numerator / denominator)
    center_y = float(statistics.median(estimates)) if estimates else height * 0.10
    return center_x, max(0.0, min(center_y, height * 0.25))


def _default_centerlines(points: list[tuple[int, int]], height: int) -> list[list[list[float]]]:
    origin_x, origin_y = _judgement_arc_origin(points, height)
    lines: list[list[list[float]]] = []
    for x, y in points:
        # The middle point is retained only to keep the persisted V4 shape
        # compatible.  All three points are collinear and share one origin, so
        # polyline progress is the direction-independent radial travel ratio.
        lines.append([
            [origin_x, origin_y],
            [origin_x + (float(x) - origin_x) * 0.55, origin_y + (float(y) - origin_y) * 0.55],
            [float(x), float(y)],
        ])
    return lines


def _from_dict(raw: dict[str, Any]) -> MusicCalibrationData:
    points = [[int(x), int(y)] for x, y in raw["points"]]
    height = int(raw["height"])
    return MusicCalibrationData(
        version=int(raw["version"]),
        lane_count=int(raw["lane_count"]),
        width=int(raw["width"]),
        height=height,
        points=points,
        # V4 profiles created before the radial fix contain the old horizontal
        # spawn-band approximation.  Rebuild this derived field from the
        # authoritative judgement points so existing users do not need to
        # recalibrate or mutate their saved profile.
        lane_centerlines=_default_centerlines([(x, y) for x, y in points], height),
        corridor_widths=[float(value) for value in raw["corridor_widths"]],
        trigger_progress=float(raw["trigger_progress"]),
        candidate_roi=[int(value) for value in raw["candidate_roi"]],
        exclusion_rois=[[int(value) for value in roi] for roi in raw.get("exclusion_rois", [])],
        baseline_version=str(raw["baseline_version"]),
        action_advance_ms=float(raw["action_advance_ms"]),
        color_lower=[[int(value) for value in color] for color in raw["color_lower"]],
        color_upper=[[int(value) for value in color] for color in raw["color_upper"]],
        candidate_min_pixels=int(raw["candidate_min_pixels"]),
        hold_min_length=float(raw["hold_min_length"]),
        created_at=str(raw.get("created_at", "")),
    )


def save_calibration(
    image: Any,
    points: list[tuple[int, int]],
    *,
    capture_median_ms: float = 0.0,
    capture_p95_ms: float = 0.0,
    input_methods: int = 0,
) -> MusicCalibrationData:
    del capture_median_ms, capture_p95_ms, input_methods
    width, height = image_size(image)
    if (width, height) != (BASE_WIDTH, BASE_HEIGHT):
        raise ValueError("V4 music calibration requires the Framework screenshot to be 1280x720")
    ordered = sorted(points, key=lambda point: point[0])
    if len(ordered) not in SUPPORTED_LANE_COUNTS:
        raise ValueError("Calibration must contain exactly 7 or 9 lanes")
    if len({point[0] for point in ordered}) != len(ordered):
        raise ValueError("Lane centers are not horizontally distinct")
    spacings = [ordered[index + 1][0] - ordered[index][0] for index in range(len(ordered) - 1)]
    spacing = float(statistics.median(spacings))
    top = max(100, min(y for _x, y in ordered) - 420)
    bottom = min(height, max(y for _x, y in ordered) + 36)
    candidate_roi = _clamped_roi([0, top, width, bottom - top], width, height)
    exclusion_rois = [
        _clamped_roi([x - int(spacing * 0.34), y - int(spacing * 0.34), int(spacing * 0.68), int(spacing * 0.68)], width, height)
        for x, y in ordered
    ]
    data = MusicCalibrationData(
        version=CALIBRATION_VERSION,
        lane_count=len(ordered),
        width=width,
        height=height,
        points=[[x, y] for x, y in ordered],
        lane_centerlines=_default_centerlines(ordered, height),
        corridor_widths=[max(24.0, spacing * 0.42) for _ in ordered],
        trigger_progress=1.0,
        candidate_roi=candidate_roi,
        exclusion_rois=exclusion_rois,
        baseline_version=VISUAL_BASELINE_VERSION,
        action_advance_ms=125.0,
        # Multiple calibrated HSV ranges can overlap; MAES de-duplicates their boxes.
        color_lower=[[0, 45, 110], [135, 35, 100], [20, 25, 145]],
        color_upper=[[25, 255, 255], [179, 255, 255], [105, 255, 255]],
        candidate_min_pixels=12,
        hold_min_length=max(70.0, spacing * 0.72),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        profiles = _read_profiles() if calibration_path().is_file() else {}
    except ValueError:
        LOGGER.info("Replacing outdated music calibration with V4 profiles")
        profiles = {}
    profiles[data.profile_key] = asdict(data)
    atomic_write_json(calibration_path(), {"version": CALIBRATION_VERSION, "profiles": profiles})
    return data


def _validate(data: MusicCalibrationData) -> None:
    if data.version != CALIBRATION_VERSION:
        raise ValueError("Music calibration format is outdated; V4 recalibration is required")
    if (data.width, data.height) != (BASE_WIDTH, BASE_HEIGHT):
        raise ValueError("Only 1280x720 Framework screenshots are accepted by V4")
    if data.lane_count not in SUPPORTED_LANE_COUNTS:
        raise ValueError("Music calibration lane count must be 7 or 9")
    if len(data.points) != data.lane_count or len(data.lane_centerlines) != data.lane_count:
        raise ValueError("Music calibration lane geometry is incomplete")
    if len(data.corridor_widths) != data.lane_count:
        raise ValueError("Music calibration corridor widths are incomplete")
    if data.baseline_version != VISUAL_BASELINE_VERSION:
        raise ValueError("Music visual baseline is outdated; V4 recalibration is required")
    if not 0.7 <= data.trigger_progress <= 1.2:
        raise ValueError("Music trigger progress is invalid")


def load_calibration(expected_lane_count: int, image: Any) -> MusicCalibrationData:
    width, height = image_size(image)
    profiles = _read_profiles()
    if expected_lane_count:
        raw = profiles.get(_profile_key(expected_lane_count, width, height))
        candidates = [raw] if raw is not None else []
    else:
        candidates = [value for key, value in profiles.items() if key.endswith(f"@{width}x{height}")]
    if not candidates:
        raise FileNotFoundError(f"No V4 music calibration for {expected_lane_count or 'auto'} lanes at {width}x{height}; run calibration first")
    if len(candidates) != 1:
        raise ValueError("Multiple music calibrations match this screen; detect or select the lane count first")
    data = _from_dict(candidates[0])
    _validate(data)
    return data


def load_calibrations_for_resolution(image: Any) -> list[MusicCalibrationData]:
    width, height = image_size(image)
    profiles = _read_profiles()
    result: list[MusicCalibrationData] = []
    for lane_count in SUPPORTED_LANE_COUNTS:
        raw = profiles.get(_profile_key(lane_count, width, height))
        if raw is None:
            continue
        data = _from_dict(raw)
        _validate(data)
        result.append(data)
    if not result:
        raise FileNotFoundError(f"No V4 music calibration for {width}x{height}; run calibration first")
    return result
