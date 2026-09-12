from __future__ import annotations

import math
from dataclasses import dataclass

from .models import MusicCalibrationData, MusicCandidate, MusicConfig
from .vision import build_color_mask, connected_components
from .hold_topology import marker_evidence

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - NumPy is pinned in the bundled runtime
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class HoldTailDetection:
    progress: float
    score: float
    pixel_count: int
    lane: int
    center: tuple[float, float]
    distance: float
    # A terminal cap has one broad ribbon exit.  A sustain checkpoint is
    # embedded in the ribbon and therefore has two (or more) separated exits.
    # Keep the default terminal value for compatibility with synthetic callers;
    # live detections always populate this from local topology.
    ribbon_exit_count: int = 1
    topology: str = ""
    owner_lanes: tuple[int, ...] | None = None


def hold_head_color_ratio(image: object, candidate: MusicCandidate) -> float:
    """Return the orange hold-head ratio inside an already accepted note box.

    This classifier is intentionally downstream of the stable compact-note
    provider.  It can label an accepted note as a hold head, but it can never
    add, remove, resize, or reassociate an ordinary tap candidate.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the hold tracker")
    x, y, width, height = candidate.box
    crop = np.asarray(image)[max(0, y) : max(0, y) + height, max(0, x) : max(0, x) + width]
    if crop.size == 0:
        return 0.0
    mask = build_color_mask(crop, [[5, 70, 130]], [[32, 255, 255]])
    if min(width, height) >= 20 and max(width, height) <= min(width, height)*1.4:
        # Outer-ring boxes include white rings and blue square corners. Judge
        # the orange disc itself, not those unrelated background pixels.
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        core = (xx-width/2.)**2 + (yy-height/2.)**2 <= (min(width,height)*.38)**2
        if core.any():
            return float(mask[core].mean())
    return float(mask.mean())


def bonus_hold_ribbon_present(
    image: object,
    candidate: MusicCandidate,
    tangent: tuple[float, float],
) -> bool:
    """Confirm the bright ribbon immediately upstream of a bonus-star head.

    The green star is only a score modifier and cannot itself distinguish a
    tap from a hold.  Star holds expose a broad neutral ribbon behind the head;
    star taps expose only the stage.  Comparing the local ribbon strip with two
    side strips rejects pale backgrounds and the white judgement arc.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the hold tracker")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return False
    extent = float(max(candidate.box[2], candidate.box[3]))
    if extent < 18.0:
        return False
    radius = extent / 2.0
    center_x, center_y = candidate.center
    # The first implementation inspected only the star-sized halo.  On the
    # recorded curved holds that crop ended inside the judgement flash, so a
    # real ribbon could be positive for only one frame and never pass temporal
    # confirmation.  Retain enough upstream context to verify that the broad
    # strip continues well beyond the head.
    margin = int(math.ceil(radius * 4.5))
    x0 = max(0, int(round(center_x)) - margin)
    y0 = max(0, int(round(center_y)) - margin)
    x1 = min(array.shape[1], int(round(center_x)) + margin + 1)
    y1 = min(array.shape[0], int(round(center_y)) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = array[y0:y1, x0:x1, :3].astype(np.int16)
    minimum = crop.min(axis=2)
    maximum = crop.max(axis=2)
    rows, columns = np.indices(crop.shape[:2])
    relative_x = columns + x0 - center_x
    relative_y = rows + y0 - center_y
    tangent_x, tangent_y = tangent

    # High-confidence calibrated-upstream topology.  Both a near and a far
    # segment must contain a broad neutral strip.  Starting outside the star
    # ring prevents its white core from supplying the evidence; side strips
    # reject the static stage and judgement arc.  This path remains aligned to
    # the calibrated upstream direction, unlike a 360-degree fan which would
    # mistake a nearby PERFECT flash for a hold ribbon.
    along = relative_x * tangent_x + relative_y * tangent_y
    across = np.abs(-relative_x * tangent_y + relative_y * tangent_x)
    strict_neutral = (minimum >= 165) & ((maximum - minimum) <= 65)
    inner = (
        (along <= -radius * 1.05)
        & (along >= -radius * 2.40)
        & (across <= radius * 0.38)
    )
    outer = (
        (along <= -radius * 2.40)
        & (along >= -radius * 4.20)
        & (across <= radius * 0.50)
    )
    inner_sides = (
        (along <= -radius * 1.05)
        & (along >= -radius * 2.40)
        & (across >= radius * 0.65)
        & (across <= radius * 1.05)
    )
    outer_sides = (
        (along <= -radius * 2.40)
        & (along >= -radius * 4.20)
        & (across >= radius * 0.80)
        & (across <= radius * 1.25)
    )
    if all(int(region.sum()) >= 20 for region in (inner, outer, inner_sides, outer_sides)):
        inner_ratio = float(strict_neutral[inner].mean())
        outer_ratio = float(strict_neutral[outer].mean())
        inner_side_ratio = float(strict_neutral[inner_sides].mean())
        outer_side_ratio = float(strict_neutral[outer_sides].mean())
        inner_contrast = inner_ratio - inner_side_ratio
        outer_contrast = outer_ratio - outer_side_ratio
        if (
            inner_ratio >= 0.70
            and outer_ratio >= 0.62
            and outer_contrast >= 0.16
            and (inner_contrast >= 0.18 or outer_contrast >= 0.35)
        ):
            return True

    # Preserve the original high-confidence straight neutral-ribbon path.  It
    # is what made standard synthetic/neutral bonus holds reliable before the
    # tinted curved variant appeared in the new recordings.
    along = relative_x * tangent_x + relative_y * tangent_y
    across = np.abs(-relative_x * tangent_y + relative_y * tangent_x)
    straight = (
        (along <= -radius * 0.52)
        & (along >= -radius * 1.90)
        & (across <= radius * 0.42)
    )
    straight_sides = (
        (along <= -radius * 0.52)
        & (along >= -radius * 1.90)
        & (across >= radius * 0.72)
        & (across <= radius * 1.14)
    )
    if int(straight.sum()) >= 20 and int(straight_sides.sum()) >= 20:
        straight_ratio = float(strict_neutral[straight].mean())
        straight_side_ratio = float(strict_neutral[straight_sides].mean())
        if straight_ratio >= 0.42 and straight_ratio - straight_side_ratio >= 0.30:
            return True

    # Recorded ribbons inherit the blue stage tint (minimum channel around
    # 130) and can curve 20--40 degrees away from the calibrated lane tangent.
    # Test a small local direction fan instead of weakening the temporal rule:
    # the tracker still requires two consecutive positive frames before it can
    # turn a bonus tap into a persistent hold.
    bright_neutral = (minimum >= 130) & ((maximum - minimum) <= 95)
    for angle_degrees in (-40.0, -20.0, 0.0, 20.0, 40.0):
        angle = math.radians(angle_degrees)
        rotated_x = tangent_x * math.cos(angle) - tangent_y * math.sin(angle)
        rotated_y = tangent_x * math.sin(angle) + tangent_y * math.cos(angle)
        along = relative_x * rotated_x + relative_y * rotated_y
        across = np.abs(-relative_x * rotated_y + relative_y * rotated_x)
        upstream = (
            (along <= -radius * 0.52)
            & (along >= -radius * 2.15)
            & (across <= radius * 0.55)
        )
        sides = (
            (along <= -radius * 0.52)
            & (along >= -radius * 2.15)
            & (across >= radius * 0.78)
            & (across <= radius * 1.25)
        )
        if int(upstream.sum()) < 20 or int(sides.sum()) < 20:
            continue
        ribbon_ratio = float(bright_neutral[upstream].mean())
        side_ratio = float(bright_neutral[sides].mean())
        if ribbon_ratio >= 0.56 and ribbon_ratio - side_ratio >= 0.30:
            return True
    return False


def _project_to_line(point: tuple[float, float], line: list[list[float]]) -> tuple[float, float]:
    lengths = [math.dist(line[index], line[index + 1]) for index in range(len(line) - 1)]
    total = max(sum(lengths), 1e-6)
    consumed = 0.0
    best_progress = 0.0
    best_distance = float("inf")
    px, py = point
    for index, length in enumerate(lengths):
        if length <= 1e-6:
            continue
        x0, y0 = line[index]
        x1, y1 = line[index + 1]
        dx, dy = x1 - x0, y1 - y0
        ratio = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (length * length)))
        x = x0 + dx * ratio
        y = y0 + dy * ratio
        distance = math.hypot(px - x, py - y)
        if distance < best_distance:
            best_distance = distance
            best_progress = (consumed + length * ratio) / total
        consumed += length
    return best_progress, best_distance


def _point_on_line(line: list[list[float]], progress: float) -> tuple[float, float]:
    lengths = [math.dist(line[index], line[index + 1]) for index in range(len(line) - 1)]
    remaining = sum(lengths) * max(0.0, min(1.0, progress))
    for index, length in enumerate(lengths):
        if length <= 1e-6:
            continue
        if remaining <= length:
            ratio = remaining / length
            return (
                line[index][0] + (line[index + 1][0] - line[index][0]) * ratio,
                line[index][1] + (line[index + 1][1] - line[index][1]) * ratio,
            )
        remaining -= length
    return tuple(line[-1])  # type: ignore[return-value]


def hold_ribbon_present(image: object, calibration: MusicCalibrationData, lane: int) -> bool:
    """Detect a broad pale hold ribbon near the judgement half of one lane.

    This is deliberately a persistence signal, not a note detector.  It is only
    consulted after an orange head has already created a persistent contact.
    Three small crops keep the cost bounded and distinguish the broad ribbon
    from the narrow static white lane line.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the hold tracker")
    array = np.asarray(image)
    height, width = array.shape[:2]
    ratios: list[float] = []
    for progress in (0.66, 0.75, 0.84):
        center_x, center_y = _point_on_line(calibration.lane_centerlines[lane], progress)
        radius = 18
        x0, x1 = max(0, int(round(center_x)) - radius), min(width, int(round(center_x)) + radius + 1)
        y0, y1 = max(0, int(round(center_y)) - radius), min(height, int(round(center_y)) + radius + 1)
        crop = array[y0:y1, x0:x1]
        if crop.size == 0:
            ratios.append(0.0)
            continue
        pale = build_color_mask(crop, [[0, 0, 160]], [[179, 100, 255]])
        ratios.append(float(pale.mean()))
    return sorted(ratios)[1] >= 0.80


def _ribbon_exit_count(
    image: object,
    center: tuple[float, float],
    marker_radius: float,
) -> int:
    """Count separated broad ribbon connections outside a yellow marker.

    The final cap is an endpoint and has one connection back to the held head.
    The hourglass-like score marker shown in the new recordings can occur in
    the middle of a hold; its ribbon enters and leaves on opposite sides.  The
    two visuals share their gold ring, so the surrounding topology—not colour
    or arrival progress—must decide whether release prediction is permitted.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the hold tracker")
    array = np.asarray(image)
    height, width = array.shape[:2]
    radius = max(6.0, float(marker_radius))
    margin = int(math.ceil(radius * 3.3))
    center_x, center_y = center
    x0 = max(0, int(round(center_x)) - margin)
    y0 = max(0, int(round(center_y)) - margin)
    x1 = min(width, int(round(center_x)) + margin + 1)
    y1 = min(height, int(round(center_y)) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = array[y0:y1, x0:x1, :3].astype(np.int16)
    minimum = crop.min(axis=2)
    maximum = crop.max(axis=2)
    neutral = (minimum >= 160) & ((maximum - minimum) <= 75)
    rows, columns = np.indices(crop.shape[:2])
    relative_x = columns + x0 - center_x
    relative_y = rows + y0 - center_y
    radial = np.hypot(relative_x, relative_y)
    angle = (np.arctan2(relative_y, relative_x) + 2.0 * math.pi) % (2.0 * math.pi)
    sectors = 24
    active: list[bool] = []
    for index in range(sectors):
        sector = (
            (radial >= radius * 1.05)
            & (radial <= radius * 3.20)
            & (angle >= index * 2.0 * math.pi / sectors)
            & (angle < (index + 1) * 2.0 * math.pi / sectors)
        )
        active.append(int(sector.sum()) >= 8 and float(neutral[sector].mean()) >= 0.52)

    # Count circular runs, discarding isolated bright pixels and thin flashes.
    if all(active):
        return sectors
    first_gap = next((index for index, value in enumerate(active) if not value), 0)
    ordered = active[first_gap + 1 :] + active[: first_gap + 1]
    run = 0
    clusters = 0
    for value in ordered:
        if value:
            run += 1
        else:
            if run >= 2:
                clusters += 1
            run = 0
    if run >= 2:
        clusters += 1
    return clusters


def detect_hold_tails(
    image: object,
    calibration: MusicCalibrationData,
    config: MusicConfig,
) -> list[HoldTailDetection]:
    """Return pale-gold hold caps inside calibrated lane corridors.

    The tail cap becomes reliably visible only after (or shortly before) the
    orange head reaches the judgement line.  Detection is therefore global and
    independent of the compact head track.  Temporal association and release
    prediction are handled by ``MusicVisionEngine``.
    """
    if np is None:
        raise RuntimeError("NumPy is required by the hold tracker")
    array = np.asarray(image)
    image_height, image_width = array.shape[:2]
    roi_x, roi_y, roi_width, roi_height = calibration.candidate_roi
    x0 = max(0, roi_x)
    y0 = max(0, roi_y)
    x1 = min(image_width, roi_x + roi_width)
    y1 = min(image_height, roi_y + roi_height)
    if x1 <= x0 or y1 <= y0:
        return []
    # One-third resolution retains the 20--50 px cap while keeping this
    # hold-only channel below the ordinary-note provider's latency budget.  The
    # saturation ceiling excludes the orange head fill; an expanded orange-halo
    # check below rejects its pale inner ring too.
    sample = 3
    mask = build_color_mask(array[y0:y1:sample, x0:x1:sample], [[7, 5, 145]], [[45, 200, 255]])
    detections: list[HoldTailDetection] = []
    for box, sampled_pixels in connected_components(mask, max(3, config.hold_tail_min_pixels // (sample * sample))):
        sampled_x, sampled_y, sampled_width, sampled_height = box
        local_x, local_y = sampled_x * sample, sampled_y * sample
        width, height = sampled_width * sample, sampled_height * sample
        pixel_count = sampled_pixels * sample * sample
        shorter, longer = min(width, height), max(width, height)
        if shorter < 8 or longer > 90 or longer / max(shorter, 1) > 1.8:
            continue
        score = pixel_count / max(width * height, 1)
        if score < max(0.25, config.hold_tail_min_score) or pixel_count < max(60, config.hold_tail_min_pixels):
            continue
        center = (x0 + local_x + width / 2.0, y0 + local_y + height / 2.0)
        projections = [
            (*_project_to_line(center, line), lane)
            for lane, line in enumerate(calibration.lane_centerlines)
        ]
        item_progress, distance, target_lane = min(projections, key=lambda item: (item[1], item[2]))
        corridor = calibration.corridor_widths[target_lane]
        if distance > min(58.0, corridor * 0.75) or item_progress > 1.02:
            continue
        radius = max(12, longer)
        center_x, center_y = int(round(center[0])), int(round(center[1]))
        halo = array[
            max(0, center_y - radius) : min(image_height, center_y + radius + 1),
            max(0, center_x - radius) : min(image_width, center_x + radius + 1),
        ]
        if halo.size:
            orange = build_color_mask(halo, [[5, 70, 130]], [[32, 255, 255]])
            if float(orange.mean()) > 0.08:
                continue
            # The white star core also satisfies the pale-gold cap mask.  Its
            # surrounding lime ring is unique and must never become another
            # hold's tail (test05 otherwise followed the next star for seconds).
            blue, green, red = halo[..., 0], halo[..., 1], halo[..., 2]
            lime = (
                (green >= 105)
                & (red >= 15)
                & (red <= green - 12)
                & (blue <= green - 12)
                & (blue <= red - 15)
            )
            if float(lime.mean()) > 0.05:
                continue
        topology, owners, _, _ = marker_evidence(array, center, longer / 2.0, target_lane, calibration)
        # Directional/local-contrast evidence replaces the old 24-sector scan.
        # Keep the compatibility field, but do not pay for both classifiers.
        ribbon_exits = {'terminal': 1, 'checkpoint': 2, 'unknown': 0}[topology]
        detections.append(
            HoldTailDetection(
                item_progress,
                score,
                pixel_count,
                target_lane,
                center,
                distance,
                ribbon_exits,
                topology,
                owners,
            )
        )
    return sorted(detections, key=lambda item: (item.progress, item.center[1], item.center[0]))
