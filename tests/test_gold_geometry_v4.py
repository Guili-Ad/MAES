"""Fail-first contract for pixel-independent gold descriptor geometry.

The reference functions retain the original independent formulas. Production
descriptor results must still be calculated from each new frame's pixels.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music import holds


def reference_shape_geometry(w, h):
    rows, cols = np.indices((h, w))
    nx, ny = (cols-(w-1)/2.)/(w/2.), (rows-(h-1)/2.)/(h/2.)
    radial = nx*nx+ny*ny
    rim = (radial >= .55**2) & (radial <= 1.05**2)
    bins = ((np.arctan2(ny, nx)+math.pi)*12/(2*math.pi)).astype(int).clip(0, 11)
    counts = np.bincount(bins[rim], minlength=12)
    return rim, bins, counts


def reference_coverage_geometry(w, h):
    yy, xx = np.ogrid[:h, :w]
    dx, dy = (xx-(w-1)/2.)/max(1., w/2.), (yy-(h-1)/2.)/max(1., h/2.)
    radial = dx*dx+dy*dy
    angle = (np.arctan2(dy, dx)+2*math.pi) % (2*math.pi)
    rim = (radial >= .45**2) & (radial <= 1.05**2)
    sectors = np.minimum(11, (angle[rim]*6/math.pi).astype(np.intp))
    totals = np.bincount(sectors, minlength=12)
    return rim, sectors, totals


def reference_shape(image, box):
    x, y, w, h = box
    if min(w, h) < 20 or not .75 <= w/max(h, 1) <= 1.35:
        return False
    crop = np.asarray(image)[y:y+h, x:x+w, :3].astype(np.int16)
    if crop.shape[:2] != (h, w):
        return False
    rim, bins, counts = reference_shape_geometry(w, h)
    b, g, r = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    gold = (r >= 170) & (g >= 110) & (r-b >= 35) & (g-b >= 15)
    filled = np.bincount(bins[rim & gold], minlength=12)
    return bool(np.count_nonzero(filled >= np.maximum(1, counts*.15)) >= 8)


def reference_coverage(image, box):
    x, y, w, h = box
    patch_pixels = image[y:y+h, x:x+w, :3]
    if patch_pixels.size == 0 or min(w, h) < 8:
        return None
    rim, sectors, totals = reference_coverage_geometry(w, h)
    b, g, r = (patch_pixels[..., i].astype(np.int16) for i in range(3))
    pale = (r >= 145) & (g >= 110) & (r >= b-20) & (r-g <= 100)
    counts = np.bincount(sectors, weights=pale[rim], minlength=12)
    return float(((totals > 0) & (counts >= totals*.25)).mean())


def outcome(function, *args):
    """Preserve even the old nonempty partially clipped patch error contract."""
    try:
        return ('value', function(*args))
    except Exception as error:
        return ('error', type(error))


class GoldGeometryCacheTests(unittest.TestCase):
    def setUp(self):
        for name in ('_gold_shape_geometry', '_gold_coverage_geometry'):
            helper = getattr(holds, name, None)
            if helper is not None:
                helper.cache_clear()

    def assert_geometry(self, actual, expected):
        self.assertEqual(len(actual), 3)
        for cached, independent in zip(actual, expected):
            self.assertIsInstance(cached, np.ndarray)
            self.assertEqual(cached.dtype, independent.dtype)
            self.assertEqual(cached.shape, independent.shape)
            np.testing.assert_array_equal(cached, independent)

    def test_shape_geometry_is_exact_original_formula(self):
        helper = holds._gold_shape_geometry
        rng = np.random.default_rng(6012)
        sizes = [(20, 20), (21, 20), (24, 33), (90, 90), (32, 31), (31, 32)]
        sizes += [tuple(map(int, rng.integers(20, 91, 2))) for _ in range(120)]
        for w, h in sizes:
            with self.subTest(w=w, h=h):
                self.assert_geometry(helper(w, h), reference_shape_geometry(w, h))

    def test_coverage_geometry_is_exact_original_formula(self):
        helper = holds._gold_coverage_geometry
        rng = np.random.default_rng(8031)
        sizes = [(8, 8), (9, 8), (32, 31), (31, 32), (90, 90), (20, 24)]
        sizes += [tuple(map(int, rng.integers(8, 91, 2))) for _ in range(120)]
        for w, h in sizes:
            with self.subTest(w=w, h=h):
                self.assert_geometry(helper(w, h), reference_coverage_geometry(w, h))

    def test_arrays_are_readonly_and_same_geometry_is_reused(self):
        for name in ('_gold_shape_geometry', '_gold_coverage_geometry'):
            helper = getattr(holds, name)
            arrays = helper(32, 31)
            self.assertIs(helper(32, 31), arrays)
            for array in arrays:
                self.assertFalse(array.flags.writeable, name)
                with self.assertRaises(ValueError):
                    array.flat[0] = 0
            self.assert_geometry(arrays, (
                reference_shape_geometry(32, 31) if name == '_gold_shape_geometry'
                else reference_coverage_geometry(32, 31)))

    def test_each_cache_is_bounded_to_at_most_128_entries(self):
        for name in ('_gold_shape_geometry', '_gold_coverage_geometry'):
            helper = getattr(holds, name)
            maximum = helper.cache_info().maxsize
            self.assertIsNotNone(maximum)
            self.assertGreater(maximum, 0)
            self.assertLessEqual(maximum, 128)
            for i in range(220):
                helper(20+i % 71, 20+i // 71)
            self.assertLessEqual(helper.cache_info().currsize, 128)
            self.assert_geometry(helper(32, 31), (
                reference_shape_geometry(32, 31) if name == '_gold_shape_geometry'
                else reference_coverage_geometry(32, 31)))

    def test_descriptors_use_geometry_but_never_cache_frame_pixels(self):
        image = np.zeros((90, 110, 3), np.uint8)
        box = (30, 20, 32, 32)
        rim, _, _ = reference_shape_geometry(32, 32)
        crop = image[20:52, 30:62]
        crop[rim] = (50, 180, 230)
        for helper_name, function, oracle in (
            ('_gold_shape_geometry', holds.gold_ring_shape, reference_shape),
            ('_gold_coverage_geometry', holds.gold_ring_coverage, reference_coverage),
        ):
            helper = getattr(holds, helper_name)
            with patch.object(holds, helper_name, wraps=helper) as observed:
                positive = function(image, box)
                self.assertEqual(positive, oracle(image, box))
                self.assertEqual(observed.call_count, 1)
                blank = np.zeros_like(image)
                negative = function(blank, box)
                self.assertEqual(negative, oracle(blank, box))
                self.assertNotEqual(positive, negative)
                self.assertEqual(function(image, box), positive)
                self.assertEqual(observed.call_count, 3)

    def test_random_noncontiguous_pixels_preserve_both_results(self):
        rng = np.random.default_rng(9058)
        for i in range(100):
            storage = rng.integers(0, 256, (220, 240, 6), np.uint8)
            image = storage[::2, ::2, ::2]
            self.assertFalse(image.flags.c_contiguous)
            w, h = map(int, rng.integers(8, 91, 2))
            x, y = int(rng.integers(0, 121-w)), int(rng.integers(0, 111-h))
            box = (x, y, w, h)
            with self.subTest(i=i, box=box):
                self.assertEqual(holds.gold_ring_shape(image, box), reference_shape(image, box))
                self.assertEqual(holds.gold_ring_coverage(image, box), reference_coverage(image, box))

    def test_color_boundary_and_full_rim_outputs_preserved(self):
        colors = [(0, 0, 0), (215, 225, 245), (135, 110, 170), (134, 110, 169),
                  (136, 110, 170), (135, 109, 170), (110, 110, 145),
                  (165, 110, 145), (166, 110, 145), (0, 110, 210), (0, 110, 211)]
        image = np.zeros((80, 90, 3), np.uint8)
        box = (20, 25, 31, 30)
        for color in colors:
            image[:] = color
            with self.subTest(color=color):
                self.assertEqual(holds.gold_ring_shape(image, box), reference_shape(image, box))
                self.assertEqual(holds.gold_ring_coverage(image, box), reference_coverage(image, box))

    def test_small_and_clipped_boxes_retain_old_early_return_or_error(self):
        image = np.full((60, 70, 3), (190, 210, 245), np.uint8)
        boxes = [(0, 0, 0, 0), (1, 1, 7, 7), (1, 1, 19, 19), (1, 1, 20, 20),
                 (70, 60, 30, 30), (65, 55, 30, 30), (-3, 0, 30, 30),
                 (0, -3, 30, 30), (10, 10, -4, 30), (0, 0, 20, 27)]
        for box in boxes:
            with self.subTest(box=box):
                self.assertEqual(outcome(holds.gold_ring_shape, image, box),
                                 outcome(reference_shape, image, box))
                self.assertEqual(outcome(holds.gold_ring_coverage, image, box),
                                 outcome(reference_coverage, image, box))
        self.assertFalse(holds.gold_ring_shape(image, (65, 55, 30, 30)))
        self.assertIsNone(holds.gold_ring_coverage(image, (70, 60, 30, 30)))


if __name__ == '__main__':
    unittest.main()
