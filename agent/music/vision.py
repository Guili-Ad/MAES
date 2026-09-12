from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agent.common import recognition_results

from .models import FLICK_COLOR_DIRECTIONS, MusicCalibrationData, MusicCandidate, MusicFrame, NoteGesture

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - NumPy is pinned in the bundled runtime
    np = None  # type: ignore[assignment]


def _as_box(value: Any) -> tuple[int, int, int, int]:
    if hasattr(value, "x"):
        return int(value.x), int(value.y), int(value.w), int(value.h)
    values = list(value)
    if len(values) != 4:
        raise ValueError("Candidate recognition returned an invalid box")
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def bgr_to_hsv(image: Any) -> Any:
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Music screenshot must be a BGR image")
    bgr = array[..., :3].astype(np.float32) / 255.0
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maximum = np.max(bgr, axis=2)
    minimum = np.min(bgr, axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue = np.mod(hue * 30.0, 180.0)
    saturation = np.zeros_like(maximum)
    np.divide(delta, maximum, out=saturation, where=maximum > 1e-6)
    saturation *= 255.0
    value = maximum * 255.0
    return np.stack((hue, saturation, value), axis=2).astype(np.uint8)


def build_color_mask(image: Any, lower: list[list[int]], upper: list[list[int]]) -> Any:
    """Integer BGR->HSV->mask without materializing a float HSV image."""
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    bgr = np.asarray(image)
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("Music screenshot must be a BGR image")
    b = bgr[..., 0].astype(np.int32)
    g = bgr[..., 1].astype(np.int32)
    r = bgr[..., 2].astype(np.int32)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    value = maxc.astype(np.uint8)
    saturation = ((delta * 255) // np.maximum(maxc, 1)).astype(np.uint8)
    nonzero = delta > 0
    divisor = np.where(nonzero, delta, 1)
    degree = np.where(nonzero & (maxc == r), 60 * (g - b) // divisor, 0)
    degree = np.where(nonzero & (maxc == g) & ~(maxc == r), 60 * (b - r) // divisor + 120, degree)
    degree = np.where(
        nonzero & (maxc == b) & ~(maxc == r) & ~(maxc == g),
        60 * (r - g) // divisor + 240,
        degree,
    )
    hue = (np.mod(degree, 360) // 2).astype(np.uint8)
    mask = np.zeros(hue.shape, dtype=bool)
    for low, high in zip(lower, upper):
        mask |= (
            (hue >= low[0]) & (hue <= high[0])
            & (saturation >= low[1]) & (saturation <= high[1])
            & (value >= low[2]) & (value <= high[2])
        )
    return mask


@dataclass(frozen=True)
class VisualMask:
    mask: Any
    hsv: Any = None
    roi_origin: tuple[int, int] = (0, 0)

    @classmethod
    def from_image(cls, image: Any, calibration: MusicCalibrationData) -> "VisualMask":
        x, y, width, height = calibration.candidate_roi
        crop = np.asarray(image)[y : y + height, x : x + width]
        small = crop[::2, ::2]
        # Flick colour families share the ordinary candidate path: one HSV
        # pass over the merged ranges, no second detection channel.
        flick_lower, flick_upper = flick_color_ranges()
        mask = build_color_mask(
            small,
            calibration.color_lower + flick_lower,
            calibration.color_upper + flick_upper,
        )
        mask = np.kron(mask, np.ones((2, 2), dtype=bool))[:height, :width]
        for ex, ey, ew, eh in calibration.exclusion_rois:
            crop_x = max(0, ex - x)
            crop_y = max(0, ey - y)
            mask[crop_y : crop_y + eh, crop_x : crop_x + ew] = False
        return cls(mask=mask, hsv=None, roi_origin=(x, y))

    def fill_ratio(self, box: tuple[int, int, int, int]) -> float:
        origin_x, origin_y = self.roi_origin
        x, y, width, height = box
        x0 = max(0, x - origin_x)
        y0 = max(0, y - origin_y)
        x1 = min(self.mask.shape[1], x - origin_x + width)
        y1 = min(self.mask.shape[0], y - origin_y + height)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        crop = self.mask[y0:y1, x0:x1]
        return float(crop.mean()) if getattr(crop, "size", 0) else 0.0


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x0, y0 = max(lx, rx), max(ly, ry)
    x1, y1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def deduplicate_candidates(candidates: Iterable[MusicCandidate], threshold: float) -> list[MusicCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.pixel_count, item.box[2] * item.box[3]), reverse=True)
    result: list[MusicCandidate] = []
    for candidate in ordered:
        if any(_iou(candidate.box, existing.box) >= threshold for existing in result):
            continue
        result.append(candidate)
    return sorted(result, key=lambda item: (item.center[1], item.center[0]))


def note_like_candidate(candidate: MusicCandidate, calibration: MusicCalibrationData, min_size: int = 0) -> bool:
    """Reject stage effects while retaining compact tap/flick note heads."""
    _x, _y, width, height = candidate.box
    shorter = min(width, height)
    longer = max(width, height)
    if shorter <= 0:
        return False
    minimum_extent = max(8, int(round(min_size * 0.65))) if min_size else 0
    maximum_extent = max(96, int(round(max(calibration.corridor_widths, default=60.0) * 1.6)))
    if minimum_extent and shorter < minimum_extent:
        return False
    if longer > maximum_extent or longer / shorter > 2.5:
        return False
    return candidate.fill_ratio >= 0.45


def detect_bonus_star_notes(image: Any, calibration: MusicCalibrationData) -> list[MusicCandidate]:
    """Return green score-bonus note heads without relaxing ordinary filters.

    Bonus notes are ordinary tap/hold heads with a lime double ring and a
    white five-point star.  Their outer ring has too little fill while their
    inner disc grows beyond the ordinary compact-note limit near judgement.
    A separate half-resolution channel therefore recognises the lime disc and
    the low-solidity white star.  Teal tap notes have a blue-dominant ring and
    a high-solidity circular white core, so neither can enter this channel.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return []
    roi_x, roi_y, roi_width, roi_height = calibration.candidate_roi
    x0 = max(0, roi_x)
    y0 = max(0, roi_y)
    x1 = min(array.shape[1], roi_x + roi_width)
    y1 = min(array.shape[0], roi_y + roi_height)
    if x1 <= x0 or y1 <= y0:
        return []
    # Most frames contain no yellow-green material of this volume.  A cheap
    # quarter-resolution gate keeps the ordinary-note path below 1 ms; only a
    # plausible star frame pays for the structural scan below.
    coarse = array[y0:y1:4, x0:x1:4, :3]
    coarse_blue, coarse_green, coarse_red = coarse[..., 0], coarse[..., 1], coarse[..., 2]
    coarse_lime = (
        (coarse_green >= 105)
        & (coarse_red >= 15)
        & (coarse_red <= coarse_green - 12)
        & (coarse_blue <= coarse_green - 12)
        & (coarse_blue <= coarse_red - 15)
    )
    if int(coarse_lime.sum()) < 130:
        return []

    sample = 2
    crop = array[y0:y1:sample, x0:x1:sample, :3]
    blue, green, red = crop[..., 0], crop[..., 1], crop[..., 2]
    # Real bonus lime is red-dominant over blue; the otherwise similar teal
    # ordinary note is blue-dominant over red by a wide margin.
    lime = (
        (green >= 105)
        & (red >= 15)
        & (red <= green - 12)
        & (blue <= green - 12)
        & (blue <= red - 15)
    )
    minimum = crop.min(axis=2)
    maximum = crop.max(axis=2)
    white = (minimum >= 185) & ((maximum - minimum) <= 48)
    minimum_pixels = max(4, int(round(calibration.candidate_min_pixels / (sample * sample))))
    candidates: list[MusicCandidate] = []
    for box, sampled_pixels in connected_components(lime, minimum_pixels):
        local_x, local_y, width, height = box
        shorter, longer = min(width, height), max(width, height)
        if shorter < 8 or longer > 80 or longer / max(shorter, 1) > 1.32:
            continue
        component_fill = sampled_pixels / max(width * height, 1)
        if not 0.42 <= component_fill <= 0.88:
            continue
        center_x = x0 + (local_x + width / 2.0) * sample
        center_y = y0 + (local_y + height / 2.0) * sample
        full_extent = longer * sample
        if center_y < 145.0 or not 0.07 <= full_extent / max(center_y, 1.0) <= 0.28:
            continue

        core = white[local_y : local_y + height, local_x : local_x + width]
        rows, columns = np.nonzero(core)
        if not columns.size:
            continue
        central = (
            (np.abs(columns - width / 2.0) <= width * 0.31)
            & (np.abs(rows - height / 2.0) <= height * 0.31)
        )
        columns = columns[central]
        rows = rows[central]
        if columns.size < 12:
            continue
        core_x0, core_x1 = int(columns.min()), int(columns.max()) + 1
        core_y0, core_y1 = int(rows.min()), int(rows.max()) + 1
        core_width = core_x1 - core_x0
        core_height = core_y1 - core_y0
        if not (
            0.36 <= core_width / width <= 0.62
            and 0.36 <= core_height / height <= 0.62
        ):
            continue
        solidity = columns.size / max(core_width * core_height, 1)
        if not 0.32 <= solidity <= 0.66:
            continue
        core_center_x = (core_x0 + core_x1) / 2.0
        core_center_y = (core_y0 + core_y1) / 2.0
        if (
            abs(core_center_x - width / 2.0) > width * 0.11 + 1.0
            or abs(core_center_y - height / 2.0) > height * 0.11 + 1.0
        ):
            continue
        candidates.append(MusicCandidate(
            box=(
                x0 + local_x * sample,
                y0 + local_y * sample,
                width * sample,
                height * sample,
            ),
            pixel_count=sampled_pixels * sample * sample,
            fill_ratio=component_fill,
            center=(center_x, center_y),
            variant="bonus_star",
        ))

    # The inner disc is normally the only dense lime component.  Keep a
    # centre-distance guard as protection against antialiasing splitting it.
    result: list[MusicCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.box[2] * item.box[3], reverse=True):
        radius = max(10.0, min(candidate.box[2:]) * 0.24)
        if any(
            (candidate.center[0] - existing.center[0]) ** 2
            + (candidate.center[1] - existing.center[1]) ** 2
            <= radius * radius
            for existing in result
        ):
            continue
        result.append(candidate)
    return sorted(result, key=lambda item: (item.center[1], item.center[0]))


# Flick notes are colour-coded by direction (blue=right, red=left,
# violet=up, pink=down).  The colour windows are seeded from the official
# sprite reference and tuned from trace statistics; the brightness and
# saturation floors separate the vivid flick sprites from the dimmer tap
# body, which is the only family whose hue band touches the blue window.
# These ranges are merged into the ordinary candidate detection so flicks
# flow through the single, established tracking path.
_FLICK_HUE_WINDOWS: dict[str, tuple[tuple[int, int], ...]] = {
    "blue": ((98, 122),),
    "violet": ((123, 147),),
    "pink": ((148, 165),),
    "red": ((166, 179), (0, 6)),
}
_FLICK_MIN_SATURATION = 120
_FLICK_MIN_VALUE = 185
_FLICK_MIN_PIXELS = 12
# Mask-side floors used when merging the flick ranges into the ordinary
# candidate colour mask (slightly looser than classification).
_FLICK_MASK_SATURATION = 130
_FLICK_MASK_VALUE = 180


def flick_color_ranges() -> tuple[list[list[int]], list[list[int]]]:
    """HSV ranges for the four flick colour families (build_color_mask / ColorMatch)."""
    lower: list[list[int]] = []
    upper: list[list[int]] = []
    for windows in _FLICK_HUE_WINDOWS.values():
        for low, high in windows:
            lower.append([low, _FLICK_MASK_SATURATION, _FLICK_MASK_VALUE])
            upper.append([high, 255, 255])
    return lower, upper


def _integer_hsv(crop: Any) -> tuple[Any, Any, Any]:
    """Integer HSV for a small crop, matching build_color_mask semantics."""
    bgr = crop.astype(np.int32)
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum
    value = maximum.astype(np.uint8)
    saturation = ((delta * 255) // np.maximum(maximum, 1)).astype(np.uint8)
    nonzero = delta > 0
    divisor = np.where(nonzero, delta, 1)
    degree = np.where(nonzero & (maximum == red), 60 * (green - blue) // divisor, 0)
    degree = np.where(nonzero & (maximum == green) & ~(maximum == red), 60 * (blue - red) // divisor + 120, degree)
    degree = np.where(nonzero & (maximum == blue) & ~(maximum == red) & ~(maximum == green), 60 * (red - green) // divisor + 240, degree)
    hue = (np.mod(degree, 360) // 2).astype(np.uint8)
    return hue, saturation, value


def classify_flick_family(image: Any, box: tuple[int, int, int, int]) -> tuple[Any, str]:
    """Colour-only flick classification: (direction, colour family).

    Direction comes from the colour table, never from the sprite's shape or
    arrow.  Brightness/saturation floors keep the teal tap body out of the
    blue window.
    """
    if np is None or image is None:
        return NoteGesture.UNKNOWN, ""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return NoteGesture.UNKNOWN, ""
    x, y, width, height = (int(value) for value in box)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(array.shape[1], x + width)
    y1 = min(array.shape[0], y + height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return NoteGesture.UNKNOWN, ""
    hue, saturation, value = _integer_hsv(array[y0:y1, x0:x1, :3])
    selected = (saturation >= _FLICK_MIN_SATURATION) & (value >= _FLICK_MIN_VALUE)
    best_name = ""
    best_count = 0
    for name, windows in _FLICK_HUE_WINDOWS.items():
        count = 0
        for low, high in windows:
            count += int((selected & (hue >= low) & (hue <= high)).sum())
        if count > best_count:
            best_name, best_count = name, count
    if best_count < _FLICK_MIN_PIXELS:
        return NoteGesture.UNKNOWN, ""
    return FLICK_COLOR_DIRECTIONS[best_name], best_name


def tag_flick_candidates(image: Any, candidates: Iterable[MusicCandidate]) -> list[MusicCandidate]:
    """Label candidates whose colour family is a flick; others pass through."""
    tagged: list[MusicCandidate] = []
    for candidate in candidates:
        direction, color = classify_flick_family(image, candidate.box)
        if direction == NoteGesture.UNKNOWN:
            tagged.append(candidate)
            continue
        tagged.append(MusicCandidate(
            box=candidate.box,
            pixel_count=candidate.pixel_count,
            fill_ratio=candidate.fill_ratio,
            center=candidate.center,
            variant="flick",
            flick_direction=direction,
            flick_color=color,
        ))
    return tagged


def detect_center_color_note(image: Any) -> list[MusicCandidate]:
    """Detect the perspective-scaled rainbow note on the center lane.

    The note's yellow outer ring is a compact component until it overlaps the
    judgement effect.  Its multicolour core separates it from ordinary
    orange/yellow notes.  Returned candidates use a negative pixel count as an
    internal marker so the tracker can keep this isolated from hold-head
    classification without changing the ordinary provider.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[0] < 600 or array.shape[1] < 760:
        return []
    origin_x, origin_y = 520, 120
    crop = array[origin_y:600, origin_x:760, :3].astype(np.int16)
    blue, green, red = crop[..., 0], crop[..., 1], crop[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    yellow = (red >= 165) & (green >= 145) & (red - blue >= 35) & (green - blue >= 25)
    warm = (maximum >= 150) & ((red - blue >= 35) | (red - green >= 40))
    cool = (maximum >= 150) & ((blue - red >= 28) | (green - red >= 28))
    result: list[MusicCandidate] = []
    for box, _count in connected_components(yellow, 20):
        x, y, width, height = box
        shorter, longer = min(width, height), max(width, height)
        center_x = origin_x + x + width / 2.0
        center_y = origin_y + y + height / 2.0
        if not (
            abs(center_x - 640.0) <= 12.0
            and 170.0 <= center_y <= 560.0
            and shorter >= 28
            and longer <= 170
            and longer / max(shorter, 1) <= 1.28
            and 0.22 <= longer / center_y <= 0.36
        ):
            continue
        # The corners of a yellow head's square box contain blue scenery.
        # Only colours *inside* the circular core establish a rainbow note.
        yy, xx = np.ogrid[:height, :width]
        core = (xx-width/2.)**2 + (yy-height/2.)**2 <= (min(width,height)*.38)**2
        area = max(int(core.sum()), 1)
        warm_pixels = int((warm[y : y + height, x : x + width] & core).sum())
        cool_pixels = int((cool[y : y + height, x : x + width] & core).sum())
        multicolour_ratio = min(warm_pixels, cool_pixels) / area
        if multicolour_ratio < 0.12:
            continue
        result.append(MusicCandidate(
            box=(origin_x + x, origin_y + y, width, height),
            pixel_count=-1,
            fill_ratio=multicolour_ratio,
            center=(center_x, center_y),
        ))
    return sorted(result, key=lambda item: item.center[1])


def giant_live_title_present(image: Any) -> bool:
    """Strictly confirm the short black transition with its large LIVE card."""
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[0] < 720 or array.shape[1] < 1280:
        return False
    # The sampled outer region must already be a black transition screen.  A
    # song-stage flash can contain a large pale area, but never this dark ratio.
    background = array[140:580:2, 80:1200:2, :3]
    if float((background.max(axis=2) <= 45).mean()) < 0.88:
        return False
    # In the captured transition the title is black text cut out of a broad
    # white card.  Row/column coverage rejects the preceding thin-line phase.
    panel = array[270:450, 350:930, :3]
    minimum = panel.min(axis=2)
    maximum = panel.max(axis=2)
    white = (minimum >= 210) & ((maximum - minimum) <= 35)
    ink = array[300:420, 450:830, :3].max(axis=2) <= 45
    return (
        int(white.sum()) >= 30000
        and int((white.sum(axis=1) >= 260).sum()) >= 80
        and int((white.sum(axis=0) >= 55).sum()) >= 350
        and int(ink.sum()) >= 8000
        and int((ink.sum(axis=0) >= 35).sum()) >= 100
        and int((ink.sum(axis=1) >= 80).sum()) >= 70
    )


def connected_components(mask: Any, min_pixels: int) -> list[tuple[tuple[int, int, int, int], int]]:
    """RLE connected components: Python iterates runs, never individual pixels."""
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    boolean = np.asarray(mask, dtype=bool)
    parent: list[int] = []
    runs: list[tuple[int, int, int, int]] = []  # row, start, end-exclusive, label
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in range(boolean.shape[0]):
        columns = np.flatnonzero(boolean[row])
        if not columns.size:
            previous = []
            continue
        split_at = np.flatnonzero(np.diff(columns) > 1) + 1
        groups = np.split(columns, split_at)
        current: list[tuple[int, int, int]] = []
        for group in groups:
            start, end = int(group[0]), int(group[-1]) + 1
            label = len(parent)
            parent.append(label)
            for previous_start, previous_end, previous_label in previous:
                if previous_end < start or previous_start > end:
                    continue
                union(label, previous_label)
            runs.append((row, start, end, label))
            current.append((start, end, label))
        previous = current

    aggregates: dict[int, list[int]] = {}
    for row, start, end, label in runs:
        root = find(label)
        if root not in aggregates:
            aggregates[root] = [start, row, end, row + 1, end - start]
        else:
            item = aggregates[root]
            item[0] = min(item[0], start)
            item[1] = min(item[1], row)
            item[2] = max(item[2], end)
            item[3] = max(item[3], row + 1)
            item[4] += end - start
    result: list[tuple[tuple[int, int, int, int], int]] = []
    for x0, y0, x1, y1, count in aggregates.values():
        if count >= min_pixels:
            result.append(((x0, y0, x1 - x0, y1 - y0), count))
    return result


def linked_tap_pair_present(image: Any, left: MusicCandidate, right: MusicCandidate, *, fast_channels: bool = False) -> bool:
    """Return whether two aligned heads are joined by the chart's white arc.

    The connector is deliberately detected as one long, thin neutral-white
    component between the two head centres.  Requiring both horizontal span
    and low fill rejects isolated highlights, judgement particles and lane
    decorations, so ordinary neighbouring taps keep their independent timing.
    """
    if np is None or image is None:
        return False
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return False
    left, right = sorted((left, right), key=lambda item: item.center[0])
    separation = right.center[0] - left.center[0]
    extent = max(18.0, float(max((*left.box[2:], *right.box[2:]))))
    if separation < max(60.0, extent * 1.6) or separation > array.shape[1] * 0.70:
        return False
    if abs(right.center[1] - left.center[1]) > max(20.0, extent * 0.42):
        return False

    x0 = int(round(left.center[0] + max(6.0, left.box[2] * 0.34)))
    x1 = int(round(right.center[0] - max(6.0, right.box[2] * 0.34)))
    y0 = int(round(min(left.center[1], right.center[1]) - extent * 0.18))
    # The observed link bows downward.  Its sag grows with head separation,
    # but remains much shallower than a lane-length ribbon.
    y1 = int(round(max(left.center[1], right.center[1]) + min(120.0, max(34.0, separation * 0.30))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(array.shape[1], x1), min(array.shape[0], y1)
    if x1 - x0 < 35 or y1 - y0 < 8:
        return False

    crop = array[y0:y1, x0:x1, :3]
    if fast_channels and crop.dtype == np.uint8:
        # Exactly the same min/max for three uint8 channels, without a strided
        # axis reduction or int16 copy. maximum >= minimum prevents underflow.
        minimum = np.minimum(np.minimum(crop[..., 0], crop[..., 1]), crop[..., 2])
        maximum = np.maximum(np.maximum(crop[..., 0], crop[..., 1]), crop[..., 2])
    else:
        crop = crop.astype(np.int16, copy=False)
        minimum = crop.min(axis=2)
        maximum = crop.max(axis=2)
    white = (minimum >= 210) & ((maximum - minimum) <= 40)
    inner_width = x1 - x0
    for (box_x, _box_y, width, height), count in connected_components(white, min_pixels=24):
        coverage = width / max(1.0, inner_width)
        fill = count / max(1.0, width * height)
        reaches_left = box_x <= inner_width * 0.20
        reaches_right = box_x + width >= inner_width * 0.80
        if (
            coverage >= 0.62
            and reaches_left
            and reaches_right
            and 3 <= height <= max(24.0, separation * 0.36)
            and width / max(1.0, height) >= 2.6
            and 0.012 <= fill <= 0.42
            and count >= max(24, int(width * 0.75))
        ):
            return True
    return False


class MaaCandidateProvider:
    name = "maa"

    def __init__(self, context: Any, calibration: MusicCalibrationData, iou_threshold: float = 0.55, min_size: int = 0) -> None:
        self.context = context
        self.calibration = calibration
        self.iou_threshold = iou_threshold
        self.min_size = min_size
        self.failures = 0
        self._configured = False

    def _configure(self) -> None:
        if self._configured:
            return
        flick_lower, flick_upper = flick_color_ranges()
        override = {
            "MusicNoteCandidates": {
                "rate_limit": 0,
                "pre_delay": 0,
                "post_delay": 0,
                "recognition": {
                    "type": "ColorMatch",
                    "param": {
                        "roi": self.calibration.candidate_roi,
                        "method": 40,
                        "lower": self.calibration.color_lower + flick_lower,
                        "upper": self.calibration.color_upper + flick_upper,
                        "count": self.calibration.candidate_min_pixels,
                        "order_by": "Area",
                        "connected": True,
                    },
                },
            }
        }
        override_pipeline = getattr(self.context, "override_pipeline", None)
        if callable(override_pipeline):
            if not override_pipeline(override):
                raise RuntimeError("MaaFramework rejected the MusicNoteCandidates V4 override")
        else:
            # Unit fakes and older shims can still inject the same one-time override here.
            self._legacy_override = override
        self._configured = True

    def detect(self, frame: MusicFrame, visual: VisualMask) -> list[MusicCandidate]:
        self._configure()
        legacy_override = getattr(self, "_legacy_override", None)
        detail = self.context.run_recognition("MusicNoteCandidates", frame.image, pipeline_override=legacy_override) if legacy_override else self.context.run_recognition("MusicNoteCandidates", frame.image)
        if detail is None:
            self.failures += 1
            raise RuntimeError("MaaFramework ColorMatch did not start")
        candidates: list[MusicCandidate] = []
        for item in recognition_results(detail):
            box = _as_box(getattr(item, "box", None))
            count = int(getattr(item, "count", 0))
            if count < self.calibration.candidate_min_pixels:
                continue
            x, y, width, height = box
            if self.min_size and max(width, height) < self.min_size:
                continue
            fill_ratio = visual.fill_ratio(box)
            if fill_ratio < 0.02:
                continue
            candidates.append(MusicCandidate(
                box=box,
                pixel_count=count,
                fill_ratio=fill_ratio,
                center=(x + width / 2.0, y + height / 2.0),
            ))
        self.failures = 0
        candidates = [
            candidate
            for candidate in candidates
            if note_like_candidate(candidate, self.calibration, self.min_size)
        ]
        return deduplicate_candidates(candidates, self.iou_threshold)


class NumpyCandidateProvider:
    name = "numpy"

    def __init__(self, calibration: MusicCalibrationData, iou_threshold: float = 0.55, min_size: int = 0) -> None:
        self.calibration = calibration
        self.iou_threshold = iou_threshold
        self.min_size = min_size
        self.failures = 0

    def detect(self, frame: MusicFrame, visual: VisualMask) -> list[MusicCandidate]:
        del frame
        origin_x, origin_y = visual.roi_origin
        candidates: list[MusicCandidate] = []
        for box, count in connected_components(visual.mask, self.calibration.candidate_min_pixels):
            x, y, width, height = box
            if self.min_size and max(width, height) < self.min_size:
                continue
            full_box = (x + origin_x, y + origin_y, width, height)
            candidates.append(MusicCandidate(
                box=full_box,
                pixel_count=count,
                fill_ratio=visual.fill_ratio(full_box),
                center=(full_box[0] + width / 2.0, full_box[1] + height / 2.0),
            ))
        self.failures = 0
        candidates = [
            candidate
            for candidate in candidates
            if note_like_candidate(candidate, self.calibration, self.min_size)
        ]
        return deduplicate_candidates(candidates, self.iou_threshold)
