"""Exact pixel/geometry contracts for the ribbon allocation-only change."""
import inspect
import math
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.music.holds import bonus_hold_ribbon_present
from agent.music.models import MusicCandidate
from test_point_ribbon_metadata import scene


REFERENCE = runpy.run_path(str(Path(__file__).parent / 'fixtures' /
    'bonus_ribbon_reference.py'))['bonus_hold_ribbon_present']


class BonusRibbonPixelTests(unittest.TestCase):
    def same(self, image, candidate, tangent):
        actual, expected = {'old': 1}, {'old': 2}
        result = bonus_hold_ribbon_present(image, candidate, tangent, evidence=actual)
        reference = REFERENCE(image, candidate, tangent, evidence=expected)
        self.assertIs(result, reference)
        self.assertEqual(actual, expected)
        self.assertIs(bonus_hold_ribbon_present(image, candidate, tangent), reference)
        return result, actual['strict_bilateral']

    def test_known_strict_bilateral_wall_straight_fan_and_negative_results(self):
        for kind, expected in (('ribbon', (True, True)),
                               ('one-sided-wall', (True, False))):
            image, candidate = scene(kind)
            self.assertEqual(self.same(image, candidate, (0., 1.)), expected)
        candidate = MusicCandidate((290, 210, 60, 60), 1800, .5,
                                   (320., 240.), 'bonus_star')
        rows, columns = np.indices((480, 640))
        rx, ry = columns-320., rows-240.
        straight = np.zeros((480, 640, 3), np.uint8)
        straight[(ry <= -30*.52) & (ry >= -30*1.90) & (np.abs(rx) <= 30*.42)] = 230
        self.assertEqual(self.same(straight, candidate, (0., 1.)), (True, False))
        tinted = np.zeros_like(straight)
        angle = math.radians(20.)
        tx, ty = -math.sin(angle), math.cos(angle)
        along, across = rx*tx+ry*ty, np.abs(-rx*ty+ry*tx)
        tinted[(along <= -30*.52) & (along >= -30*2.15) & (across <= 30*.55)] = [150, 195, 225]
        self.assertEqual(self.same(tinted, candidate, (0., 1.)), (True, False))
        self.assertEqual(self.same(np.full_like(straight, 210), candidate, (0., 1.)), (False, False))

    def test_fixed_random_all_tangents_threshold_pixels_and_border_crops(self):
        rng = np.random.default_rng(20261004)
        image = rng.integers(0, 256, (180, 240, 4), dtype=np.uint8)
        image[::3, ::4, :3] = [130, 165, 195]
        image[::4, ::3, :3] = [165, 230, 230]
        centers = ((120.25, 90.5), (0., 0.), (239.75, 179.5),
                   (-10.5, 30.), (100., 179.), (5000., 5000.))
        for i, center in enumerate(centers):
            for extent in (17, 18, 41, 60, 128, 320):
                for degrees in (-40., -20., 0., 20., 40., 90., 180., 270.):
                    tangent = (math.sin(math.radians(degrees)), math.cos(math.radians(degrees)))
                    candidate = MusicCandidate((0, 0, extent, extent+1), 100, .5, center, 'bonus_star')
                    with self.subTest(center=i, extent=extent, degrees=degrees):
                        self.same(image, candidate, tangent)

    def test_signed_float_and_noncontiguous_image_conversion_is_unchanged(self):
        rng = np.random.default_rng(974)
        images = (rng.integers(-300, 301, (100, 110, 3), dtype=np.int16),
                  rng.uniform(-20., 280., (100, 110, 3)),
                  rng.integers(0, 256, (100, 110, 3), dtype=np.uint8)[::-1, ::-1])
        candidate = MusicCandidate((24, 20, 52, 50), 500, .5, (50.25, 45.5), 'bonus_star')
        for image in images:
            for tangent in ((0., 1.), (.8, .6), (-.8, -.6)):
                self.same(image, candidate, tangent)
        for image in (np.zeros((10, 10)), np.zeros((10, 10, 2))):
            self.assertEqual(self.same(image, candidate, (0., 1.)), (False, False))

    def test_elementwise_channels_equal_full_reductions(self):
        rng = np.random.default_rng(365)
        crop = rng.integers(-32768, 32768, (720, 1280, 3), dtype=np.int16)
        b, g, r = (crop[..., i] for i in range(3))
        np.testing.assert_array_equal(crop.min(axis=2), np.minimum(np.minimum(b, g), r))
        np.testing.assert_array_equal(crop.max(axis=2), np.maximum(np.maximum(b, g), r))

    def test_sparse_projections_and_pixel_counts_equal_dense_maximum_crop(self):
        shape = (720, 1280)
        dense_y, dense_x = np.indices(shape)
        sparse_y, sparse_x = np.ogrid[:shape[0], :shape[1]]
        for tangent in ((0., 1.), (.8, .6), (-.8, -.6)):
            for angle in (-40., -20., 0., 20., 40.):
                rotation = math.radians(angle)
                tx = tangent[0]*math.cos(rotation)-tangent[1]*math.sin(rotation)
                ty = tangent[0]*math.sin(rotation)+tangent[1]*math.cos(rotation)
                dx, dy = dense_x+0-640.25, dense_y+0-360.5
                sx, sy = sparse_x+0-640.25, sparse_y+0-360.5
                da, sa = dx*tx+dy*ty, sx*tx+sy*ty
                dc, sc = np.abs(-dx*ty+dy*tx), np.abs(-sx*ty+sy*tx)
                np.testing.assert_array_equal(da, sa)
                np.testing.assert_array_equal(dc, sc)
                dense = (da <= -320*.52) & (da >= -320*4.2) & (dc <= 320*1.25)
                sparse = (sa <= -320*.52) & (sa >= -320*4.2) & (sc <= 320*1.25)
                np.testing.assert_array_equal(dense, sparse)
                self.assertEqual(int(dense.sum()), int(sparse.sum()))

    def test_new_implementation_uses_sparse_grid_and_pairwise_reductions(self):
        source = inspect.getsource(bonus_hold_ribbon_present)
        self.assertNotIn('np.indices(', source)
        self.assertNotIn('.min(axis=2)', source)
        self.assertNotIn('.max(axis=2)', source)
        self.assertIn('np.ogrid[', source)
        self.assertIn('np.minimum(np.minimum(', source)
        self.assertIn('np.maximum(np.maximum(', source)
        image = np.zeros((720, 1280, 3), np.uint8)
        candidate = MusicCandidate((0, 0, 640, 640), 100, .5, (640., 360.), 'bonus_star')
        shapes = []
        original = np.ogrid
        class RecordingGrid:
            def __getitem__(self, slices):
                shapes.append(tuple(item.stop for item in slices))
                return original[slices]
        with patch('agent.music.holds.np.ogrid', RecordingGrid()), \
                patch('agent.music.holds.np.indices', side_effect=AssertionError('dense grid forbidden')):
            actual = bonus_hold_ribbon_present(image, candidate, (0., 1.))
        self.assertEqual(shapes, [(720, 1280)])
        self.assertIs(actual, REFERENCE(image, candidate, (0., 1.)))


if __name__ == '__main__':
    unittest.main()
