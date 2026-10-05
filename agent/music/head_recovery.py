"""Positive, bounded pixel recovery for already moving point-mode heads.

Global candidate colour/size rules are untouched. A local contour is extracted
from raw pixels *before its own size checks*, even when the global connected
component contains a much larger HUD/stage stroke. This never creates an ID,
event, owner relationship, or a motion sample; association owns those steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

from .models import MusicCandidate, TrackState
from .tap_identity import point_clickable
from .head_identity import stationary_from_birth


@dataclass(frozen=True)
class RecoveryProposal:
    track: object
    candidate: MusicCandidate
    projection: object
    residual: float
    confidence: float
    expected: tuple[float, float]


def _prediction(track, frame):
    if (not point_clickable(track) or track.visual_family == 'gold'
            or track.tap_input_started is not None
            or track.state not in {TrackState.APPROACHING, TrackState.TAP_PENDING}
            or stationary_from_birth(track)):
        return None
    prior = []
    for observation in track.observations:
        if observation.frame_sequence >= frame.sequence:
            continue
        if prior and (observation.center, observation.candidate.box) == (prior[-1].center, prior[-1].candidate.box):
            continue  # repeated capture is never forward evidence
        prior.append(observation)
    if len(prior) < 3:
        return None
    recent, last = prior[-3:], prior[-1]
    if not all(b.progress-a.progress > .002 and 0 < b.frame_sequence-a.frame_sequence <= 3
               and b.timestamp > a.timestamp for a, b in zip(recent, recent[1:])):
        return None
    age = frame.midpoint-last.timestamp
    if not 0 < age <= .18 or frame.sequence-last.frame_sequence > 3 or not .25 <= last.progress <= .96:
        return None
    span = recent[-1].timestamp-recent[0].timestamp
    expected_progress = last.progress+(last.progress-recent[0].progress)*age/span
    extent = float(max(last.candidate.box[2:]))
    if not .25 <= expected_progress <= .985 or not 16 <= extent <= 100:
        return None
    cx = last.center[0]+(last.center[0]-recent[0].center[0])*age/span
    cy = last.center[1]+(last.center[1]-recent[0].center[1])*age/span
    if not all(math.isfinite(v) for v in (cx, cy, expected_progress, extent)):
        return None
    return last, (cx, cy), expected_progress, extent


@lru_cache(maxsize=1)
def _templates():
    angles = np.arange(16)*2*math.pi/16
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    return (np.array([(0., 0.), (.12, 0.), (-.12, 0.), (0., .12), (0., -.12)]),
            np.concatenate([unit*r for r in (.40, .55, .70)]), unit*1.06)


@lru_cache(maxsize=17)
def _offset_grid(span):
    # Fixed sampling geometry only. No image, ownership or prediction cache.
    offsets = np.arange(-span, span+1, 4., dtype=float)
    yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    grid.flags.writeable = False
    return grid


def _bounds(pixels):
    # Equivalent three-channel min/max without a strided axis reduction.
    b, g, r = (pixels[..., i] for i in range(3))
    return np.minimum(np.minimum(b, g), r), np.maximum(np.maximum(b, g), r)


def _pixels(image, centers, radii, offsets):
    xs = np.rint(centers[:, 0, None]+radii[:, None]*offsets[None, :, 0]).astype(np.int32)
    ys = np.rint(centers[:, 1, None]+radii[:, None]*offsets[None, :, 1]).astype(np.int32)
    # Never bless a clipped disc by borrowing border pixels.
    return image[ys, xs, :3].astype(np.int16)


def _color(pixels, family):
    b, g, r = (pixels[..., i] for i in range(3))
    if family == 'ordinary':
        return (b >= 70) & (g >= 110) & (g > r+35) & (b > r+25)
    if family == 'bonus':
        return (g >= 105) & (r >= 15) & (r <= g-12) & (b <= g-12) & (b <= r-15)
    # Exact algebra for the accepted fixed HSV [5,70,130]..[32,255,255].
    # Preserve the original UInt8 cast of sampled arrays. R wins maximum ties;
    # degree [10,65] is 60*(g-b)>=10*delta in R, and
    # floor(-60*(r-b)/delta)+120<=65 <=> 60*(r-b)>54*delta in G.
    array = pixels.astype(np.uint8, copy=False)
    b, g, r = (array[..., i].astype(np.int16) for i in range(3))
    maximum = np.maximum(np.maximum(r, g), b)
    delta = maximum-np.minimum(np.minimum(r, g), b)
    saturation = np.multiply(delta, 255, dtype=np.int32) >= np.multiply(maximum, 70, dtype=np.int32)
    red = (r >= g) & (r >= b) & (60*(g-b) >= 10*delta)
    green = (g > r) & (g >= b) & (60*(r-b) > 54*delta)
    return (maximum >= 130) & saturation & (red | green)


def _quadrant_counts(disc):
    # Three radii times four angles in each of the four quadrants: twelve
    # Boolean samples. Exhaustive 4096-pattern test proves mean>=.65 iff sum>=8.
    return disc.reshape(-1, 3, 4, 4).sum(axis=(1, 3), dtype=np.uint8)


def _search(track, frame, prediction, project):
    last, expected, progress, extent = prediction
    image = np.asarray(frame.image)
    family = track.visual_family or 'ordinary'
    span = min(24, max(8, int(extent*.35)))
    centers = _offset_grid(span)+np.asarray(expected)
    # Every supported head has a neutral white nucleus. Eliminate empty/color
    # centres before expanding radius hypotheses, rather than repeatedly
    # fetching the same background for every possible disc size.
    center_xy = np.rint(centers).astype(np.int32)
    inside = ((center_xy[:, 0] >= 0) & (center_xy[:, 0] < image.shape[1])
              & (center_xy[:, 1] >= 0) & (center_xy[:, 1] < image.shape[0]))
    centers, center_xy = centers[inside], center_xy[inside]
    nucleus = image[center_xy[:, 1], center_xy[:, 0], :3]
    low, high = _bounds(nucleus)
    centers = centers[(low >= 190) & (high-low < 55)]
    if not len(centers):
        return []
    sizes = np.arange(extent*.45, extent*.70+.01, 2.)
    positions = len(centers)
    centers = np.repeat(centers, len(sizes), axis=0)
    radii = np.tile(sizes, positions)
    valid = ((centers[:, 0]-radii*1.1 >= 0) & (centers[:, 1]-radii*1.1 >= 0)
             & (centers[:, 0]+radii*1.1 < image.shape[1]) & (centers[:, 1]+radii*1.1 < image.shape[0]))
    centers, radii = centers[valid], radii[valid]
    if not len(centers):
        return []
    core_template, disc_template, outer_template = _templates()
    core_pixels = _pixels(image, centers, radii, core_template)
    minimum, maximum = _bounds(core_pixels)
    core = ((minimum >= 190) & (maximum-minimum < 55)).mean(axis=1)
    selected = core >= .60
    centers, radii, core = centers[selected], radii[selected], core[selected]
    if not len(centers):
        return []
    disc = _color(_pixels(image, centers, radii, disc_template), family)
    outside = _color(_pixels(image, centers, radii, outer_template), family).mean(axis=1)
    coverage = disc.mean(axis=1)
    # Each quadrant has to surround the white centre. A coloured glyph next
    # to a white stroke is not a physical circle, and solid particles fail core.
    quadrant_counts = _quadrant_counts(disc)
    selected = (coverage >= .80) & (quadrant_counts.min(axis=1) >= 8) & (outside <= .30)
    indices = np.flatnonzero(selected)
    residuals = np.linalg.norm(centers-np.asarray(expected), axis=1)
    confidence = coverage+core*.20-outside*.4
    indices = sorted(indices, key=lambda i: (-confidence[i]+residuals[i]*.006, residuals[i]))
    proposals = []
    for i in indices[:24]:
        center = tuple(float(v) for v in centers[i])
        diameter = int(round(radii[i]*2))
        if math.dist(center, last.center) < max(2., extent*.08):
            continue  # do not manufacture motion from an unchanged capture
        if any(math.dist(center, p.candidate.center) < min(diameter, p.candidate.box[2])*.4 for p in proposals):
            continue
        box = (int(round(center[0]-diameter/2)), int(round(center[1]-diameter/2)), diameter, diameter)
        x, y, w, h = box
        color = _color(image[y:y+h, x:x+w, :3].astype(np.int16), family)
        candidate = MusicCandidate(box, int(color.sum()), float(color.mean()), center,
                                   variant='bonus_star' if family == 'bonus' else '')
        # The extracted ROI is already a circle, not a compound component.
        # Prove its dense disc directly before entering the generic adapter's
        # more expensive seed search (which has nothing to extract here).
        patch = image[y:y+h, x:x+w, :3]
        minimum, maximum = _bounds(patch)
        white = (minimum >= (190 if family == 'ordinary' else 185)) & (maximum-minimum <= 55)
        py, px = np.ogrid[:h, :w]
        distance = ((px-(w-1)/2)/(w/2))**2+((py-(h-1)/2)/(h/2))**2
        annulus = (distance > .35**2) & (distance < .72**2)
        core_mask = distance < (.20 if family == 'ordinary' else .60)**2
        if (not core_mask.any() or not annulus.any()
                or white[core_mask].mean() < (.25 if family == 'ordinary' else .04)
                or color[annulus].mean() < (.65 if family == 'ordinary' else .55)):
            continue
        # Cache the same adapter's dense proof in this immutable frame. Sparse
        # templates only locate a proposal; they never override its physical
        # qualification or send it directly to the input queue.
        from .tap_physical_identity import contour_evidence
        evidence = contour_evidence(candidate, frame, family=family)
        if evidence.verdict != 'positive' or evidence.candidate is None:
            continue
        candidate = evidence.candidate
        center = candidate.center
        projection = project(candidate)
        if (projection is None or projection.lane != track.lane
                or projection.distance > min(16., diameter*.20)
                or abs(projection.progress-progress) > .045
                or not last.progress+.002 < projection.progress < .985):
            continue
        proposals.append(RecoveryProposal(track, candidate, projection, float(residuals[i]),
                                          float(confidence[i]), expected))
        if len(proposals) >= 3:
            break
    return proposals


def _same(left, right):
    return (left.projection.lane == right.projection.lane
            and math.dist(left.candidate.center, right.candidate.center)
            < min(left.candidate.box[2], right.candidate.box[2])*.4)


def _assign(proposals):
    """One-to-one, order-preserving ownership; ambiguous swaps stay unknown."""
    clusters, edges = [], {}
    for p in sorted(proposals, key=lambda p: (-p.confidence, p.residual)):
        index = next((i for i, other in enumerate(clusters) if _same(p, other)), None)
        if index is None:
            index = len(clusters); clusters.append(p)
        key = p.track.track_id, index
        if key not in edges or p.residual < edges[key].residual:
            edges[key] = p
    ids = sorted({tid for tid, _ in edges})
    def ordered(left, right):
        a, b = left.track.observations, right.track.observations
        if left.track.lane != right.track.lane:
            return True
        shared = {o.frame_sequence: o.progress for o in a}
        common = [(o.frame_sequence, shared[o.frame_sequence]-o.progress) for o in b if o.frame_sequence in shared]
        if not common:
            return False
        old = max(common)[1]
        return abs(old) > 1e-5 and old*(left.projection.progress-right.projection.progress) > 0
    if len(ids) > 4 or len(clusters) > 8:
        unique = [p for (tid, index), p in edges.items()
                  if sum(t == tid for t, _ in edges) == 1 and sum(i == index for _, i in edges) == 1]
        # Bounded fallback keeps the same ordering contract as the solver.
        # Unique spatial edges alone do not prove that a dense run did not swap.
        return [p for p in unique if all(p is other or ordered(p, other) for other in unique)]
    best, visited = [], 0
    def walk(pos, chosen, used, cost):
        nonlocal visited
        visited += 1
        if visited > 2048:
            return
        if pos == len(ids):
            best.append((-len(chosen), cost, tuple((p.track.track_id, i) for i, p in chosen)))
            best.sort(); del best[2:]
            return
        walk(pos+1, chosen, used, cost)
        for (tid, index), p in edges.items():
            if tid == ids[pos] and index not in used and all(ordered(other, p) for _, other in chosen):
                walk(pos+1, chosen+[(index, p)], used | {index}, cost+p.residual)
    walk(0, [], set(), 0.)
    if visited > 2048 or not best:
        return []
    assignment = dict(best[0][2])
    if len(best) > 1 and best[1][0] == best[0][0] and best[1][1]-best[0][1] <= 2.:
        alternate = dict(best[1][2])
        assignment = {tid: index for tid, index in assignment.items() if alternate.get(tid) == index}
    return [edges[tid, index] for tid, index in assignment.items()]


def recover_point_heads(tracks, frame, calibration, project, *, entries=None, trace=None):
    if not isinstance(frame.image, np.ndarray) or frame.image.ndim != 3 or frame.image.shape[2] < 3:
        return {}
    proposals = []
    for track in tracks.values():
        prediction = _prediction(track, frame)
        if prediction is None:
            continue
        last, expected, progress, extent = prediction
        # Do not rewrite an ordinary compact global observation merely because
        # its nucleus is partly covered/unknown. Existing normal association
        # remains authoritative; recovery only fills absent/compound entries.
        covered = False
        for candidate, projection in (entries or ()):
            if ((candidate.variant == 'bonus_star') != track.bonus_star or candidate.variant == 'flick'
                    or projection.lane != track.lane or abs(projection.progress-progress) > .045
                    or math.dist(candidate.center, expected) > extent*.5):
                continue
            width, height = candidate.box[2:]
            from .head_identity import identity_continuity_allowed
            if (.80 <= width/max(height, 1) <= 1.25
                    and .75*extent <= max(width, height) <= 1.5*extent
                    and identity_continuity_allowed(track, candidate, projection, frame)):
                covered = True; break
        if covered:
            continue
        found = _search(track, frame, prediction, project)
        # An independently observed neighbouring/sent head cannot be borrowed.
        found = [p for p in found if not any(other.track_id != track.track_id
            and other.lane == track.lane and other.state not in {TrackState.LOST, TrackState.RELEASED}
            and other.observations and other.observations[-1].frame_sequence == frame.sequence
            and math.dist(other.observations[-1].center, p.candidate.center) < p.candidate.box[2]*.4
            for other in tracks.values())]
        proposals.extend(found)
        if trace is not None:
            trace.add('head_recovery_search', time=frame.midpoint, frame=frame.sequence,
                      track=track.track_id, family=track.visual_family,
                      first_seen_time=track.first_seen_time,
                      latest_visual_time=last.timestamp, predicted_center=expected,
                      latest_capture_started=last.capture_started,
                      latest_capture_finished=last.capture_finished,
                      reason='positive-proposals' if found else 'no-owned-positive-contour', proposals=len(found))
    result = {}
    for p in _assign(proposals):
        result[p.track.track_id] = (p.candidate, p.projection)
    return result
