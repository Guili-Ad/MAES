"""Local ribbon evidence, independent of head timing and tap scheduling.

Brightness alone is not a connection: compare the ribbon core with both
flanks along several samples. No song timing, screen-text ROI, or fixed route.
"""
from __future__ import annotations
import math
import numpy as np


def ribbon_score(image, start, end, *, width=5., begin=.12, stop=.85, flank=2.5, brightness=165, control=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 4:
        return 0.
    fraction = np.linspace(begin, stop, 12)[:, None]
    if control is None:
        normal = np.broadcast_to(np.array([-vector[1], vector[0]]) / length, (12,2))
        samples = start + fraction * vector
    else:
        control = np.asarray(control,float)
        samples = (1-fraction)**2*start + 2*fraction*(1-fraction)*control + fraction**2*end
        tangent = (1-fraction)*(control-start) + fraction*(end-control)
        normal = np.stack((-tangent[:,1],tangent[:,0]),axis=1)
        normal /= np.maximum(1.,np.linalg.norm(normal,axis=1))[:,None]
    # Vectorized core/flank samples; no per-pixel Python loop or full HSV pass.
    offsets = np.array([-flank, -flank*.8, -.35, 0., .35, flank*.8, flank])
    widths = np.broadcast_to(np.asarray(width), (12,))
    xy = np.rint(samples[:, None, :] + (widths[:, None]*offsets)[..., None] * normal[:,None,:]).astype(int)
    xy[..., 0] = np.clip(xy[..., 0], 0, image.shape[1]-1)
    xy[..., 1] = np.clip(xy[..., 1], 0, image.shape[0]-1)
    pixels = image[xy[..., 1], xy[..., 0], :3].astype(np.int16)
    low = pixels.min(axis=2)
    high = pixels.max(axis=2)
    core = low[:, 2:5].mean(axis=1)
    flanks = (low[:, :2].mean(axis=1) + low[:, 5:].mean(axis=1)) / 2
    neutral = (high[:, 2:5] - low[:, 2:5]).mean(axis=1) <= 70
    return float(((core >= brightness) & neutral & (core - flanks >= 12)).mean())


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
    owner_scores = tuple(ribbon_score(image, center, point, width=owner_width,
                         control=(center[0]+unit[0]*math.dist(center,point)*.5,
                                  center[1]+unit[1]*math.dist(center,point)*.5))
                         for point in calibration.points)
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


def ribbon_at_judgement(image, calibration, lane):
    origin = np.asarray(calibration.lane_centerlines[lane][0], float)
    point = np.asarray(calibration.points[lane], float)
    start = point + (origin-point)*.10
    return ribbon_score(image, start, point, width=18., begin=.1, stop=.85) >= .58
