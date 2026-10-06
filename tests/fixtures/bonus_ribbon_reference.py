"""Unchanged ribbon classifier from checkpoint 7d7ba8e-before-performance.

Development-only executable reference. It is intentionally independent of the
runtime implementation and does not depend on .work or an installed package.
"""
from __future__ import annotations

import math
import numpy as np


def bonus_hold_ribbon_present(
    image: object,
    candidate,
    tangent: tuple[float, float],
    *,
    evidence: dict | None = None,
) -> bool:
    """Confirm the bright ribbon immediately upstream of a bonus-star head.

    The green star is only a score modifier and cannot itself distinguish a
    tap from a hold.  Star holds expose a broad neutral ribbon behind the head;
    star taps expose only the stage.  Comparing the local ribbon strip with two
    side strips rejects pale backgrounds and the white judgement arc.
    """
    # Optional point-mode metadata is stricter than the historical return.
    # Callers that classify/tune clicks continue to consume the exact old bool.
    # Reuse the existing near/far masks below instead of rescanning the image.
    if evidence is not None:
        evidence.clear()
        evidence['strict_bilateral'] = False
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
            if evidence is not None:
                signed_across = -relative_x * tangent_y + relative_y * tangent_x
                bilateral = True
                # Averaging left/right sides lets a bright stage wall on one
                # side and blue stage on the other masquerade as a ribbon.
                # Both edges must separately contrast with the near/far body.
                for region, body_ratio, contrast in (
                    (inner_sides, inner_ratio, .18),
                    (outer_sides, outer_ratio, .16),
                ):
                    for sign in (-1, 1):
                        side = region & (signed_across * sign > 0)
                        if (int(side.sum()) < 10
                                or body_ratio - float(strict_neutral[side].mean()) < contrast):
                            bilateral = False
                evidence['strict_bilateral'] = bilateral
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
