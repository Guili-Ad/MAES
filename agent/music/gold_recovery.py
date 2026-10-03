"""Local positive-pixel recovery for existing tap-mode gold identities.

A moving hourglass ring may join the fixed judgement arc in the global
connected-component mask. This helper searches only a bounded neighbourhood
of a healthy physical marker. It never creates an ID, an owner, a terminal
vote, a flick direction, or an input event, and it never updates the tracker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .holds import HoldTailDetection, _project_to_line, gold_ring_coverage, gold_ring_shape


@dataclass(frozen=True)
class _Prediction:
    marker_id: int
    owner: int
    last: object
    center: tuple[float, float]
    diameter: float
    progress: float
    span: int


@dataclass(frozen=True)
class _Proposal:
    prediction: _Prediction
    detection: HoldTailDetection
    confidence: float
    residual: float
    search_box: tuple[int, int, int, int]


def _prediction(marker, descriptor, frame, config):
    history = list(marker.observations)
    if (len(history) < 3 or marker.owner is None or descriptor is None
            or descriptor.box is None or descriptor.physical_ring is not True
            or descriptor.ring_coverage is None
            or not math.isfinite(descriptor.ring_coverage) or descriptor.ring_coverage < .5):
        return None
    last = history[-1]
    if frame.sequence <= last.frame_sequence or not math.isfinite(frame.midpoint):
        return None
    recent = history[-4:]
    times = np.asarray([item.timestamp for item in recent], float)
    centers = np.asarray([item.center for item in recent], float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(centers)):
        return None
    intervals = np.diff(times)
    if np.any(intervals <= 1e-6):
        return None
    age = frame.midpoint-last.timestamp
    # Reacquiring positive current pixels is not executing an old prediction.
    # Keep at most two captured-frame gaps (and an absolute .6 s ceiling),
    # even when host-stage jitter exceeded the old two-period coast budget.
    # The caller rechecks normal eligibility AFTER observing this evidence.
    if not 0 < age <= .6 or frame.sequence-last.frame_sequence > 2:
        return None
    if (last.progress < .55 or last.progress > 1.01
            or last.progress-recent[0].progress < .005
            or math.dist(last.center, recent[0].center) < 4.):
        return None
    speed = marker.speed()
    if not math.isfinite(speed) or speed <= config.hold_sustain_marker_min_speed:
        return None
    diameter = float(max(descriptor.box[2:]))
    if not 20. <= diameter <= 90.:
        return None
    # Fit x/y against relative time; long host uptime cannot corrupt motion.
    relative = times-times[-1]
    weights = np.arange(1, len(times)+1, dtype=float)
    mean = np.average(relative, weights=weights)
    denominator = float(np.sum(weights*(relative-mean)**2))
    if denominator <= 1e-10:
        return None
    mean_center = np.average(centers, axis=0, weights=weights)
    velocity = np.sum(weights[:, None]*(relative-mean)[:, None]
                      *(centers-mean_center), axis=0)/denominator
    expected = centers[-1]+velocity*age
    if not np.all(np.isfinite(expected)):
        return None
    return _Prediction(marker.marker_id, marker.owner, last,
        (float(expected[0]), float(expected[1])), diameter,
        last.progress+speed*age, min(48, max(16, int(math.ceil(diameter*.75)))))


@lru_cache(maxsize=1)
def _templates():
    angles = np.arange(24)*2*math.pi/24
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    return (np.array([(0., 0.), (.1, 0.), (-.1, 0.), (0., .1), (0., -.1)]),
            np.array([(-.65, -.1), (-.65, 0.), (-.65, .1),
                      (.65, -.1), (.65, 0.), (.65, .1)]),
            np.concatenate([unit*f for f in (.82, .92, 1.)]), unit[::3]*1.16)


def _pixels(array, centers, radii, offsets):
    xy = np.rint(centers[:, None, :]+radii[:, None, None]*offsets[None, :, :]).astype(np.int32)
    xy[:, :, 0] = np.clip(xy[:, :, 0], 0, array.shape[1]-1)
    xy[:, :, 1] = np.clip(xy[:, :, 1], 0, array.shape[0]-1)
    return array[xy[:, :, 1], xy[:, :, 0], :3].astype(np.int16)


def _gold(pixels):
    blue, green, red = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    return (red >= 170) & (green >= 110) & (red-blue >= 35) & (green-blue >= 15)


def _same_ring(left, right):
    if left.box is None or right.box is None:
        return False
    ld, rd = max(left.box[2:]), max(right.box[2:])
    return (.65 <= ld/max(rd, 1) <= 1.55
            and math.dist(left.center, right.center) <= min(ld, rd)*.4)


def _search(frame, calibration, prediction, global_detections):
    array = np.asarray(frame.image)
    span = prediction.span
    # Five-pixel sampling bounds centre uncertainty to a few pixels while
    # keeping all three size hypotheses below 1,200 lightweight candidates.
    offsets = np.arange(-span, span+1, 5, dtype=float)
    yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
    centers = np.column_stack((xx.ravel()+prediction.center[0],
                               yy.ravel()+prediction.center[1]))
    radii = np.clip(prediction.diameter*np.array([.45, .55, .65]), 10., 45.)
    centers = np.repeat(centers, len(radii), axis=0)
    radii = np.tile(radii, len(xx.ravel()))
    core_template, wing_template, rim_template, outer_template = _templates()
    # A repeat of the last physical sprite is not a velocity sample. A
    # coarse search must not turn a sub-box offset on those same pixels into
    # artificial progress towards its predicted centre.
    old_center = np.asarray([prediction.last.center], float)
    old_radius = np.asarray([prediction.diameter*.5], float)
    old_core = float(_pixels(array, old_center, old_radius, core_template).min(axis=2).mean())
    old_wings = _pixels(array, old_center, old_radius, wing_template)[0]
    old_brightness = old_wings.min(axis=1)
    old_white = (old_brightness >= 230) & ((old_wings.max(axis=1)-old_brightness) <= 50)
    old_contrast = min(float(old_brightness[:3].max()), float(old_brightness[3:].max()))-old_core
    if (old_core < 238 and old_white[:3].mean() >= .3 and old_white[3:].mean() >= .3
            and old_contrast >= 14
            and _gold(_pixels(array, old_center, old_radius, rim_template)).mean() >= .35):
        return []
    core = _pixels(array, centers, radii, core_template).min(axis=2).mean(axis=1)
    wings = _pixels(array, centers, radii, wing_template)
    brightness = wings.min(axis=2)
    white = (brightness >= 230) & ((wings.max(axis=2)-brightness) <= 50)
    left, right = white[:, :3].mean(axis=1), white[:, 3:].mean(axis=1)
    contrast = np.minimum(brightness[:, :3].max(axis=1),
                          brightness[:, 3:].max(axis=1))-core
    # The fixed target has a solid white core, not two neutral wings with
    # a darker hourglass middle. This guard is independent of screen position.
    selected = np.flatnonzero((core < 238) & (left >= .3) & (right >= .3) & (contrast >= 14))
    if not len(selected):
        return []
    centers, radii = centers[selected], radii[selected]
    rim = _gold(_pixels(array, centers, radii, rim_template)).mean(axis=1)
    outside = _gold(_pixels(array, centers, radii, outer_template)).mean(axis=1)
    residual = np.sqrt(((centers-np.asarray(prediction.center))**2).sum(axis=1))
    quality = (rim-.25*outside+.1*(left[selected]+right[selected])
               +np.minimum(.1, contrast[selected]/300.)-residual*.001)
    order = np.argsort(quality)[::-1]
    proposals = []
    for index in order[:24]:
        if quality[index] < .60:
            break
        if rim[index] < .35 or residual[index] > span+2:
            continue
        center = tuple(float(value) for value in centers[index])
        diameter = int(round(radii[index]*2))
        box = (int(round(center[0]-diameter/2)), int(round(center[1]-diameter/2)),
               diameter, diameter)
        x, y, width, height = box
        if x < 0 or y < 0 or x+width > array.shape[1] or y+height > array.shape[0]:
            continue
        # Many sampled sizes/centres describe the same physical ring. Reject
        # those before repeating its dense rim descriptor and geometry work.
        if any(other.box is not None
               and .65 <= diameter/max(other.box[2:]) <= 1.55
               and math.dist(center, other.center) <= min(diameter, max(other.box[2:]))*.4
               for other in [*global_detections, *(p.detection for p in proposals)]):
            continue
        if (math.dist(center, prediction.last.center) < 2.
                or not gold_ring_shape(array, box)):
            continue
        coverage = gold_ring_coverage(array, box)
        if coverage is None or coverage < .75:
            continue
        projections = [(*_project_to_line(center, line), lane)
                       for lane, line in enumerate(calibration.lane_centerlines)]
        progress, distance, lane = min(projections, key=lambda item: (item[1], item[2]))
        if (abs(lane-prediction.last.lane) > 1 or distance > min(58., calibration.corridor_widths[lane]*.75)
                or progress < prediction.last.progress+.001 or progress > 1.02
                or abs(progress-prediction.progress) > .12):
            continue
        # Unknown is deliberately not terminal or checkpoint. The caller may
        # retain its already-confirmed owner, but this cannot refresh proof.
        pixels = array[y:y+height, x:x+width, :3].astype(np.int16)
        blue, green, red = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        count = int(((red >= 170) & (green >= 110) & (red-blue >= 35) & (green-blue >= 15)).sum())
        detection = HoldTailDetection(progress, float(rim[index]), count, lane,
            center, distance, 0, 'unknown', (), box, float(coverage), None, True)
        if (any(_same_ring(detection, item) for item in global_detections)
                or any(_same_ring(detection, item.detection) for item in proposals)):
            continue
        search_box = (int(round(prediction.center[0]-span-45)),
                      int(round(prediction.center[1]-span-45)), 2*(span+45), 2*(span+45))
        proposals.append(_Proposal(prediction, detection, float(quality[index]),
                                   float(residual[index]), search_box))
        if len(proposals) >= 3:
            break
    return proposals


def _ordered(first, second):
    a, b = first.prediction, second.prediction
    if a.owner != b.owner or a.last.frame_sequence != b.last.frame_sequence:
        return True
    old = a.last.progress-b.last.progress
    new = first.detection.progress-second.detection.progress
    return abs(old) <= .01 or old*new >= -.0001


def _assign(proposals):
    """Bounded one-to-one assignment; equally plausible ID swaps stay unknown."""
    clusters, edges = [], {}
    for proposal in sorted(proposals, key=lambda p: -p.confidence):
        index = next((i for i, other in enumerate(clusters)
                      if _same_ring(proposal.detection, other.detection)), None)
        if index is None:
            index = len(clusters)
            clusters.append(proposal)
        key = proposal.prediction.marker_id, index
        if key not in edges or proposal.residual < edges[key].residual:
            edges[key] = proposal
    mids = sorted({mid for mid, _ in edges})
    if len(mids) > 4 or len(clusters) > 8:
        # Dense ambiguous clusters are not permission to rename rings.
        return [p for (mid, index), p in edges.items()
                if sum(m == mid for m, _ in edges) == 1
                and sum(i == index for _, i in edges) == 1]
    best = []
    visited = 0
    def walk(pos, chosen, used, cost):
        nonlocal visited
        visited += 1
        if visited > 2048:
            return
        if pos == len(mids):
            candidate = (-len(chosen), cost, tuple((p.prediction.marker_id, i) for i, p in chosen))
            best.append(candidate)
            best.sort()
            del best[2:]
            return
        walk(pos+1, chosen, used, cost)
        mid = mids[pos]
        for (owner, index), proposal in edges.items():
            if owner != mid or index in used or not all(_ordered(p, proposal) for _, p in chosen):
                continue
            walk(pos+1, chosen+[(index, proposal)], used | {index},
                 cost+proposal.residual+8*(1-proposal.confidence))
    walk(0, [], set(), 0.)
    if visited > 2048 or not best:
        return []
    assignment = dict(best[0][2])
    if len(best) > 1 and best[1][0] == best[0][0] and best[1][1]-best[0][1] <= 2.:
        alternate = dict(best[1][2])
        assignment = {mid: index for mid, index in assignment.items() if alternate.get(mid) == index}
    return [edges[mid, index] for mid, index in assignment.items()]


def recover_gold_markers(frame, calibration, config, markers, descriptors,
                         global_detections, *, excluded_ids=(), diagnostics=None):
    """Return {known physical ID: positive current-frame detection}.

    Run global association first. The caller additionally excludes closed,
    contradicted, started/completed, or owner-ineligible IDs. Observation and
    scheduling stay with the caller; current source timestamps are carried by
    ``frame`` as in the normal global observation path. Optional diagnostics
    is a caller-owned dict, not process-global mutable state.
    """
    if not config.hold_notes_as_taps or not config.enable_holds or frame.image is None:
        return {}
    array = np.asarray(frame.image)
    if array.ndim != 3 or array.shape[2] < 3:
        return {}
    excluded = set(excluded_ids)
    globals_ = [d for d in global_detections if d.physical_ring is True]
    proposals = []
    for marker in markers.values():
        if marker.marker_id in excluded:
            continue
        prediction = _prediction(marker, descriptors.get(marker.marker_id), frame, config)
        if prediction is not None:
            proposals.extend(_search(frame, calibration, prediction, globals_))
    recovered = {}
    for proposal in _assign(proposals):
        marker_id = proposal.prediction.marker_id
        recovered[marker_id] = proposal.detection
        if diagnostics is not None:
            diagnostics[marker_id] = dict(confidence=proposal.confidence,
                residual_px=proposal.residual, predicted_center=proposal.prediction.center,
                search_box=proposal.search_box, frame_sequence=frame.sequence,
                timestamp=frame.midpoint, capture_started=frame.capture_started,
                capture_finished=frame.capture_finished, topology='unknown', source='local-positive-ring')
    return recovered
