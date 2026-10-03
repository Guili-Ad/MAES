"""Tap-mode-only early filtering uses the existing physical-ring verdict."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.music import holds
from agent.music.models import MusicConfig
from test_sustain_chain import calibration


def mixed_frame():
    image = np.zeros((720, 1280, 3), np.uint8)
    yy, xx = np.ogrid[:720, :1280]
    radius = (xx-640)**2+(yy-400)**2
    image[(radius <= 16**2) & (radius >= 11**2)] = (190, 210, 245)
    # Two accepted pale-gold blobs have no chromatic circular rim.
    image[285:309, 148:172] = (215, 225, 245)
    image[490:514, 948:972] = (215, 225, 245)
    return image


class GoldPhysicalFastpathTests(unittest.TestCase):
    def setUp(self):
        self.cal = calibration()
        self.image = mixed_frame()
        self.config = MusicConfig(hold_notes_as_taps=True, lane_count=7, enable_holds=True)

    def test_public_default_is_exactly_unchanged(self):
        default = holds.detect_hold_tails(self.image, self.cal, self.config)
        self.assertTrue(any(d.physical_ring is False for d in default))
        self.assertTrue(any(d.physical_ring is True for d in default))
        explicit = holds.detect_hold_tails(self.image, self.cal, self.config, physical_only=False)
        self.assertEqual(explicit, default)

    def test_fastpath_is_exact_physical_subset_in_same_order(self):
        default = holds.detect_hold_tails(self.image, self.cal, self.config)
        expected = [d for d in default if d.physical_ring is True]
        actual = holds.detect_hold_tails(self.image, self.cal, self.config, physical_only=True)
        self.assertEqual(actual, expected)

    def test_false_candidates_do_not_compute_topology_or_coverage(self):
        expected = [d for d in holds.detect_hold_tails(self.image, self.cal, self.config)
                    if d.physical_ring is True]
        with patch.object(holds, 'marker_evidence', wraps=holds.marker_evidence) as topology, \
             patch.object(holds, 'gold_ring_coverage', wraps=holds.gold_ring_coverage) as coverage:
            actual = holds.detect_hold_tails(self.image, self.cal, self.config, physical_only=True)
        self.assertEqual(actual, expected)
        self.assertEqual(topology.call_count, len(expected))
        self.assertEqual(coverage.call_count, len(expected))

    def test_shape_is_computed_once_per_candidate_and_reused(self):
        default = holds.detect_hold_tails(self.image, self.cal, self.config)
        with patch.object(holds, 'gold_ring_shape', wraps=holds.gold_ring_shape) as shape:
            holds.detect_hold_tails(self.image, self.cal, self.config, physical_only=True)
        self.assertEqual(shape.call_count, len(default))

    def test_legacy_mode_ignores_optional_gate_and_keeps_all_outputs(self):
        legacy = MusicConfig(hold_notes_as_taps=False, lane_count=7, enable_holds=True)
        expected = holds.detect_hold_tails(self.image, self.cal, legacy)
        with patch.object(holds, 'gold_ring_shape', side_effect=AssertionError('legacy shape gate')):
            actual = holds.detect_hold_tails(self.image, self.cal, legacy, physical_only=True)
        self.assertEqual(actual, expected)
        self.assertTrue(all(d.physical_ring is None for d in actual))

    def test_empty_frame_and_clipped_candidate_roi(self):
        empty = np.zeros_like(self.image)
        self.assertEqual(holds.detect_hold_tails(empty, self.cal, self.config, physical_only=True), [])
        self.cal.candidate_roi = [1300, 740, 10, 10]
        self.assertEqual(holds.detect_hold_tails(self.image, self.cal, self.config, physical_only=True), [])


if __name__ == '__main__':
    unittest.main()
