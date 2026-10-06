"""Exact pre-optimization score oracle; no visual thresholds are relaxed."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.hold_topology import marker_evidence, ribbon_at_judgement, ribbon_score
from agent.music import hold_topology


def reference_score(image, start, end, *, width=5., begin=.12, stop=.85,
                    flank=2.5, brightness=165, control=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 4:
        return 0.
    fraction = np.linspace(begin, stop, 12)[:, None]
    if control is None:
        normal = np.broadcast_to(np.array([-vector[1], vector[0]]) / length, (12, 2))
        samples = start + fraction * vector
    else:
        control = np.asarray(control, float)
        samples = ((1-fraction)**2*start + 2*fraction*(1-fraction)*control
                   + fraction**2*end)
        tangent = (1-fraction)*(control-start) + fraction*(end-control)
        normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
        normal /= np.maximum(1., np.linalg.norm(normal, axis=1))[:, None]
    offsets = np.array([-flank, -flank*.8, -.35, 0., .35, flank*.8, flank])
    widths = np.broadcast_to(np.asarray(width), (12,))
    xy = np.rint(samples[:, None, :] + (widths[:, None]*offsets)[..., None]
                 * normal[:, None, :]).astype(int)
    xy[..., 0] = np.clip(xy[..., 0], 0, image.shape[1]-1)
    xy[..., 1] = np.clip(xy[..., 1], 0, image.shape[0]-1)
    pixels = image[xy[..., 1], xy[..., 0], :3].astype(np.int16)
    low, high = pixels.min(axis=2), pixels.max(axis=2)
    core = low[:, 2:5].mean(axis=1)
    flanks = (low[:, :2].mean(axis=1) + low[:, 5:].mean(axis=1))/2
    neutral = (high[:, 2:5]-low[:, 2:5]).mean(axis=1) <= 70
    return float(((core >= brightness) & neutral & (core-flanks >= 12)).mean())


def reference_marker(image, center, radius, lane, calibration):
    origin = calibration.lane_centerlines[lane][0]
    dx, dy = center[0]-origin[0], center[1]-origin[1]
    length = max(1., math.hypot(dx, dy))
    unit = dx/length, dy/length
    radius = max(6., radius)
    near = center[0]-unit[0]*radius*1.5, center[1]-unit[1]*radius*1.5
    far = center[0]-unit[0]*radius*4.5, center[1]-unit[1]*radius*4.5
    upstream = reference_score(image, near, far, width=max(3., radius*.65),
                               begin=0., stop=1., flank=1.6, brightness=190)
    owner_width = np.linspace(max(4., radius*.8), 44., 12)
    scores = tuple(reference_score(
        image, center, point, width=owner_width,
        control=(center[0]+unit[0]*math.dist(center, point)*.5,
                 center[1]+unit[1]*math.dist(center, point)*.5))
        for point in calibration.points)
    best = max(scores, default=0.)
    owners = tuple(i for i, score in enumerate(scores)
                   if score >= .58 and score >= best-.12)
    down = center[0]+unit[0]*radius*4.5, center[1]+unit[1]*radius*4.5
    downstream = reference_score(image, center, down, width=max(3., radius*.65),
                                 begin=.35, stop=1.)
    topology = ('checkpoint' if upstream >= .58 else
                'terminal' if upstream <= .17 and (best >= .58 or downstream >= .58)
                else 'unknown')
    convergent = math.dist(calibration.lane_centerlines[0][0],
                           calibration.lane_centerlines[-1][0]) < 20.
    return topology, owners if convergent else None, upstream, scores


def reference_judgement(image, calibration, lane):
    origin = np.asarray(calibration.lane_centerlines[lane][0], float)
    point = np.asarray(calibration.points[lane], float)
    start = point + (origin-point)*.10
    return reference_score(image, start, point, width=18., begin=.1, stop=.85) >= .58


def calibration(count=7, *, convergent=True, width=1280, height=720):
    points = [(float(i*(width-1)/max(1, count-1)), float(height*.95))
              for i in range(count)]
    origins = [(width*.5+(0 if convergent else i*20), height*.1) for i in range(count)]
    return SimpleNamespace(points=points,
                           lane_centerlines=[[origin, point] for origin, point in zip(origins, points)])


class HoldTopologyGeometryTests(unittest.TestCase):
    def test_scalar_and_vector_width_score_exact_equivalence(self):
        random = np.random.default_rng(20261003)
        image = random.integers(0, 256, (83, 127, 3), np.uint8)
        for _ in range(180):
            start, end, control = random.uniform((-30, -30), (157, 113), (3, 2))
            for width in (5., np.linspace(3., 44., 12)):
                for bend in (None, control):
                    kwargs = {'width': width, 'control': bend,
                              'begin': random.uniform(0, .3),
                              'stop': random.uniform(.5, 1.),
                              'flank': random.uniform(1., 3.),
                              'brightness': random.integers(120, 230)}
                    self.assertEqual(ribbon_score(image, start, end, **kwargs),
                                     reference_score(image, start, end, **kwargs))

    def test_owner_scores_exact_for_seven_and_nine_routes(self):
        random = np.random.default_rng(20261004)
        for shape in ((72, 128), (108, 192)):
            image = random.integers(0, 256, (*shape, 3), np.uint8)
            for count in (7, 9):
                for convergent in (False, True):
                    cal = calibration(count, convergent=convergent,
                                      width=shape[1], height=shape[0])
                    for _ in range(40):
                        center = tuple(random.uniform((-12, -12),
                                                      (shape[1]+12, shape[0]+12)))
                        lane = int(random.integers(0, count))
                        radius = float(random.uniform(1., 40.))
                        self.assertEqual(marker_evidence(image, center, radius, lane, cal),
                                         reference_marker(image, center, radius, lane, cal))

    def test_bright_neutral_and_coloured_pixels_remain_exact(self):
        random = np.random.default_rng(333)
        for level in (164, 165, 166, 189, 190, 191, 255):
            image = np.full((72, 128, 3), level, np.uint8)
            image[::3, ::5] = (0, 50, 255)
            image[::5, ::3] = random.integers(0, 256, (len(range(0, 72, 5)),
                                                    len(range(0, 128, 3)), 3), np.uint8)
            cal = calibration(width=128, height=72)
            for lane in range(7):
                self.assertEqual(marker_evidence(image, (64.5, 28.5), 8., lane, cal),
                                 reference_marker(image, (64.5, 28.5), 8., lane, cal))
                self.assertEqual(ribbon_at_judgement(image, cal, lane),
                                 reference_judgement(image, cal, lane))

    def test_short_and_zero_distance_owner_paths(self):
        image = np.full((72, 128, 3), 255, np.uint8)
        cal = calibration(width=128, height=72)
        for center in (cal.points[3], (cal.points[3][0]+3., cal.points[3][1])):
            self.assertEqual(marker_evidence(image, center, 10., 3, cal),
                             reference_marker(image, center, 10., 3, cal))
        self.assertEqual(ribbon_score(image, (10, 10), (11, 11)), 0.)

    def test_cache_never_reuses_pixels_across_frames(self):
        random = np.random.default_rng(12345)
        cal = calibration(width=128, height=72)
        for _ in range(24):
            image = random.integers(0, 256, (72, 128, 3), np.uint8)
            for lane in range(7):
                self.assertEqual(ribbon_at_judgement(image, cal, lane),
                                 reference_judgement(image, cal, lane))

    def test_changed_calibration_and_resolution_invalidate_geometry(self):
        random = np.random.default_rng(531)
        for height, width in ((72, 128), (108, 192), (1, 1), (33, 127)):
            image = random.integers(0, 256, (height, width, 3), np.uint8)
            for cal in (calibration(width=width, height=height),
                        calibration(convergent=False, width=width, height=height)):
                for lane in range(7):
                    self.assertEqual(ribbon_at_judgement(image, cal, lane),
                                     reference_judgement(image, cal, lane))

    def test_geometry_caches_are_bounded_and_immutable(self):
        for offset in range(200):
            hold_topology._fractions(offset/1000., .85)
            hold_topology._offsets(1.+offset/1000.)
            xy = hold_topology._judgement_geometry(
                720, 1280, (639.+offset/1000., 74.), (847., 625.))
            self.assertFalse(xy.flags.writeable)
        for function, limit in ((hold_topology._fractions, 16),
                                (hold_topology._offsets, 8),
                                (hold_topology._judgement_geometry, 128)):
            self.assertLessEqual(function.cache_info().currsize, limit)
            self.assertEqual(function.cache_info().maxsize, limit)
        self.assertFalse(hold_topology._fractions(.1, .85).flags.writeable)
        self.assertFalse(hold_topology._offsets(2.5).flags.writeable)


if __name__ == '__main__':
    unittest.main()
