"""Local ribbon evidence, independent of head timing and tap scheduling.

Brightness alone is not a connection: compare the ribbon core with both
flanks along several samples. No song timing, screen-text ROI, or fixed route.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=16)
def _fractions(begin, stop):
    fraction = np.linspace(begin, stop, 12)[:, None]
    fraction.flags.writeable = False
    return fraction


@lru_cache(maxsize=8)
def _offsets(flank):
    offsets = np.array([-flank, -flank*.8, -.35, 0., .35, flank*.8, flank])
    offsets.flags.writeable = False
    return offsets


def _coordinates(image_shape, start, end, *, width=5., begin=.12, stop=.85,
                 flank=2.5, control=None):
    """Same ordered sample arithmetic for one route or a batch of routes."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    vector = end - start
    batch = vector.ndim == 2
    length = np.linalg.norm(vector, axis=-1) if batch else float(np.linalg.norm(vector))
    fraction = _fractions(begin, stop)
    if batch:
        fraction = fraction[None, :, :]
        start, end, vector = start[:, None, :], end[:, None, :], vector[:, None, :]
    if control is None:
        if batch:
            normal = np.concatenate((-vector[..., 1:], vector[..., :1]), axis=-1) / length[:, None, None]
            normal = np.broadcast_to(normal, (len(start), 12, 2))
        else:
            normal = np.broadcast_to(np.array([-vector[1], vector[0]]) / length, (12,2))
        samples = start + fraction * vector
    else:
        control = np.asarray(control,float)
        if batch:
            control = control[:, None, :]
        samples = (1-fraction)**2*start + 2*fraction*(1-fraction)*control + fraction**2*end
        tangent = (1-fraction)*(control-start) + fraction*(end-control)
        normal = np.stack((-tangent[...,1],tangent[...,0]),axis=-1)
        normal /= np.maximum(1.,np.linalg.norm(normal,axis=-1))[...,None]
    # Vectorized core/flank samples; no per-pixel Python loop or full HSV pass.
    offsets = _offsets(flank)
    widths = np.broadcast_to(np.asarray(width), (12,))
    xy = np.rint(samples[..., None, :] + (widths[:, None]*offsets)[..., None] * normal[...,None,:]).astype(int)
    xy[..., 0] = np.clip(xy[..., 0], 0, image_shape[1]-1)
    xy[..., 1] = np.clip(xy[..., 1], 0, image_shape[0]-1)
    return xy


def _score_samples(image, xy, brightness):
    # Only geometry can be cached. Every call samples the current frame.
    pixels = image[xy[..., 1], xy[..., 0], :3].astype(np.int16)
    low = pixels.min(axis=-1)
    high = pixels.max(axis=-1)
    core = low[..., 2:5].mean(axis=-1)
    flanks = (low[..., :2].mean(axis=-1) + low[..., 5:].mean(axis=-1)) / 2
    neutral = (high[..., 2:5] - low[..., 2:5]).mean(axis=-1) <= 70
    return ((core >= brightness) & neutral & (core - flanks >= 12)).mean(axis=-1)


def ribbon_score(image, start, end, *, width=5., begin=.12, stop=.85, flank=2.5, brightness=165, control=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    if float(np.linalg.norm(end-start)) < 4:
        return 0.
    xy = _coordinates(image.shape, start, end, width=width, begin=begin,
                      stop=stop, flank=flank, control=control)
    return float(_score_samples(image, xy, brightness))


def _owner_scores(image, center, unit, owner_width, points):
    if len(points) == 0:
        return ()
    ends = np.asarray(points, float)
    starts = np.broadcast_to(np.asarray(center, float), ends.shape)
    valid = np.linalg.norm(ends-starts, axis=1) >= 4
    scores = np.zeros(len(points), float)
    if valid.any():
        controls = np.asarray([
            (center[0]+unit[0]*math.dist(center,point)*.5,
             center[1]+unit[1]*math.dist(center,point)*.5)
            for point in points], float)
        xy = _coordinates(image.shape, starts[valid], ends[valid],
                          width=owner_width, control=controls[valid])
        scores[valid] = _score_samples(image, xy, 165)
    return tuple(float(score) for score in scores)


def marker_evidence(image, center, radius, lane, calibration):
    origin = calibration.lane_centerlines[lane][0]
    dx, dy = center[0]-origin[0], center[1]-origin[1]
    length = max(1., math.hypot(dx, dy))
    unit = (dx/length, dy/length)
    # Sample outside the gold ring, not its white star/inner core.
    r = max(6., radius)
    near = (center[0]-unit[0]*r*1.5, center[1]-unit[1]*r*1.5)
    far = (center[0]-unit[0]*r*4.5, center[1]-unit[1]*r*4.5)
    upstream = ribbon_score(image, near, far, width=max(3., r*.65), begin=0., stop=1., flank=1.6, brightness=190)
    # Perspective widens the ribbon towards the judgement arc. Fixed-width
    # flanks would lie *inside* a genuine wide ribbon and reject its owner.
    owner_width = np.linspace(max(4., r*.8), 44., 12)
    # Ribbon tangent leaves the marker along its destination ray, then bends
    # to the held judgement point. A straight chord misses genuine curved
    # ribbons by tens of pixels. This quadratic uses observed geometry only.
    owner_scores = _owner_scores(image, center, unit, owner_width, calibration.points)
    best = max(owner_scores, default=0.)
    owners = tuple(i for i, score in enumerate(owner_scores) if score >= .58 and score >= best-.12)
    # A short local downstream test also supports isolated caps in developer
    # fixtures whose ribbon ends before the judgement line.
    down = (center[0]+unit[0]*r*4.5, center[1]+unit[1]*r*4.5)
    downstream = ribbon_score(image, center, down, width=max(3., r*.65), begin=.35, stop=1.)
    if upstream >= .58:
        topology = 'checkpoint'
    elif upstream <= .17 and (best >= .58 or downstream >= .58):
        topology = 'terminal'
    else:
        topology = 'unknown'
    # Non-convergent historical calibrations cannot use the radial ownership
    # constraint. They retain the original association path.
    origins = calibration.lane_centerlines
    convergent = math.dist(origins[0][0], origins[-1][0]) < 20.
    return topology, owners if convergent else None, upstream, owner_scores


@lru_cache(maxsize=128)
def _judgement_geometry(height, width, origin, point):
    origin = np.asarray(origin, float)
    point = np.asarray(point, float)
    start = point + (origin-point)*.10
    if float(np.linalg.norm(point-start)) < 4:
        return None
    xy = _coordinates((height, width), start, point, width=18., begin=.1, stop=.85)
    xy.flags.writeable = False
    return xy


def ribbon_at_judgement(image, calibration, lane):
    xy = _judgement_geometry(image.shape[0], image.shape[1],
                            tuple(calibration.lane_centerlines[lane][0]),
                            tuple(calibration.points[lane]))
    return False if xy is None else bool(_score_samples(image, xy, 165) >= .58)
