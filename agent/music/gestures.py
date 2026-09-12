from __future__ import annotations

from typing import Iterable

from .models import MusicCandidate, MusicConfig, NoteGesture
from .vision import classify_flick_family

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def classify_hold(candidate: MusicCandidate, lane_direction: tuple[float, float], min_length: float) -> bool:
    width, height = candidate.box[2], candidate.box[3]
    direction_x, direction_y = lane_direction
    projected = abs(direction_x) * width + abs(direction_y) * height
    transverse = abs(direction_y) * width + abs(direction_x) * height
    return projected >= min_length and projected >= transverse * 1.35 and candidate.fill_ratio >= 0.12


def classify_note_gesture(image: object, roi: Iterable[int], config: MusicConfig) -> NoteGesture:
    """Compatibility helper: colour-family flick classification on one ROI.

    Direction is decided by the sprite's colour family only; the sprite's
    shape and inner arrow are never consulted.
    """
    del config
    if np is None:
        return NoteGesture.UNKNOWN
    array = np.asarray(image)
    if array.ndim != 3:
        return NoteGesture.UNKNOWN
    x, y, width, height = (int(value) for value in roi)
    direction, _color = classify_flick_family(array, (x, y, width, height))
    return direction


def roi_white_ratio(image: object, roi: Iterable[int], threshold: int) -> float:
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    array = np.asarray(image)
    x, y, width, height = (int(value) for value in roi)
    crop = array[max(0, y) : max(0, y) + height, max(0, x) : max(0, x) + width]
    if not crop.size:
        return 1.0
    if crop.ndim == 2:
        return float((crop >= threshold).mean())
    return float(np.all(crop[..., :3] >= threshold, axis=2).mean())


def patch_white_ratio(image: object, x: int, y: int, radius: int, threshold: int) -> float:
    return roi_white_ratio(image, [x - radius, y - radius, radius * 2 + 1, radius * 2 + 1], threshold)
