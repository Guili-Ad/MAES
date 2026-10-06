"""Ordering/count contracts for online, weighted-root RLE aggregation.

Correctness itself is not a defect in the old labeler: the independent old
oracle must pass before and after optimization. The paired development
benchmark separately requires a real source change and measured improvement.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.music.components import connected_components
from test_components_v4 import reference_components, pixel_reference


class ComponentsOnlineUnionTests(unittest.TestCase):
    def assert_equivalent(self, mask, minimum=1, *, independent=False):
        expected = reference_components(mask, minimum)
        self.assertEqual(connected_components(mask, minimum), expected)
        if independent:
            self.assertEqual(expected, pixel_reference(mask, minimum))

    def test_long_single_run_component_does_not_change_geometry(self):
        for width in (1, 3, 31):
            mask = np.ones((1500, width), bool)
            self.assertEqual(connected_components(mask, 1),
                             [((0, 0, width, 1500), width*1500)])
            self.assert_equivalent(mask)

    def test_first_small_component_merged_into_later_large_root_keeps_order(self):
        mask = np.zeros((18, 52), bool)
        mask[:14, 3] = True
        mask[0, 10] = True  # independent second component, not latest bbox xmin
        mask[:14, 20:38] = True
        mask[13, 3:38] = True  # tiny early owner merges into much larger owner
        mask[16:18, 1:4] = True
        self.assert_equivalent(mask, independent=True)
        self.assertEqual(connected_components(mask, 1)[1], ((10, 0, 1, 1), 1))

    def test_output_order_is_first_pixel_not_late_bbox_minimum(self):
        mask = np.zeros((22, 32), bool)
        mask[0, 2] = True
        mask[np.arange(16), 15-np.arange(16)] = True
        mask[3:5, 26:28] = True
        self.assert_equivalent(mask, independent=True)
        self.assertEqual(connected_components(mask, 1),
                         [((2, 0, 1, 1), 1), ((0, 0, 16, 16), 16),
                          ((26, 3, 2, 2), 4)])

    def test_repeated_split_and_late_bridge_preserve_every_pixel(self):
        mask = np.zeros((70, 101), bool)
        mask[0:40:2, 2:99:4] = True
        mask[1:40:2, 1:100] = True
        mask[42:69, 10:16] = True
        mask[42:69, 39:88] = True
        mask[68, 10:88] = True
        for minimum in (0, 1, 100, 1800, 3000):
            self.assert_equivalent(mask, minimum, independent=True)

    def test_short_diagonal_bridge_cannot_be_filtered_before_aggregation(self):
        mask = np.zeros((10, 10), bool)
        mask[1:4, 1:4] = True
        mask[4, 4] = True
        mask[5:8, 5:8] = True
        self.assertEqual(connected_components(mask, 18), [((1, 1, 7, 7), 19)])
        self.assert_equivalent(mask, 18, independent=True)

    def test_weighted_root_ties_and_multiple_overlaps_are_exact(self):
        mask = np.zeros((30, 121), bool)
        mask[0:10, 1:15] = True
        mask[0:10, 31:45] = True
        mask[0:10, 61:75] = True
        mask[0:10, 91:105] = True
        mask[9, 1:45] = True
        mask[12:20, 43:64] = True
        mask[19, 43:104] = True
        mask[20:25, 55:58] = True
        self.assert_equivalent(mask, independent=True)

    def test_signed_numeric_negative_stride_and_row_gaps(self):
        rng = np.random.default_rng(7102)
        raw = rng.integers(-1, 2, (141, 99), np.int16)
        raw[17:22] = 0
        for mask in (raw[:, ::-2], raw[::-2, 1::3], raw.T[::3]):
            for minimum in (1, 4, 35, mask.size+1):
                self.assert_equivalent(mask, minimum)

    def test_random_saturated_and_fragmented_masks(self):
        rng = np.random.default_rng(9914)
        for size in ((22, 39), (110, 171), (150, 300)):
            for probability in (.001, .07, .32, .55, .92, 1.):
                mask = rng.random(size) < probability
                for minimum in (1, 3, 60):
                    self.assert_equivalent(mask, minimum)


if __name__ == '__main__':
    unittest.main()
