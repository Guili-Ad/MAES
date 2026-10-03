"""Local gold recovery uses current pixels and preserves physical identities."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.gold_recovery import recover_gold_markers
from agent.music.holds import HoldTailDetection
from agent.music.models import MusicConfig, MusicFrame
from agent.music.sustain import SustainMarker, SustainObservation
from test_longtap_branch import calibration


def hourglass(image, center, diameter=48):
    yy, xx = np.indices(image.shape[:2])
    dx, dy = (xx-center[0])/(diameter/2), (yy-center[1])/(diameter/2)
    radial = dx*dx+dy*dy
    image[radial < 1.] = (130, 150, 170)
    image[(radial >= .80**2) & (radial <= 1.)] = (100, 205, 250)
    wings = (abs(dy) < .18) & (abs(dx) > .50) & (abs(dx) < .78)
    image[wings] = (255, 255, 255)


class GoldRecoveryTests(unittest.TestCase):
    def setup(self, *, owner=9, tid=7, x=640., old=500., diameter=48):
        cal = calibration()
        config = MusicConfig(enable_holds=True, hold_notes_as_taps=True)
        marker = SustainMarker(tid, owner=owner)
        for i in range(3):
            y = old+i*20
            marker.observe(SustainObservation(.8+i*.05, 10+i,
                (y-140)/480, 3, (x, y), .9, 500, 1, 'terminal', .79+i*.05, .81+i*.05), owner)
        last = marker.observations[-1]
        descriptor = HoldTailDetection(last.progress, .9, 500, 3, last.center, 0.,
            1, 'terminal', (3,), (int(x-diameter/2), int(last.center[1]-diameter/2), diameter, diameter),
            .9, None, True)
        image = np.zeros((720, 1280, 3), np.uint8)
        hourglass(image, (x, old+60), diameter)
        frame = MusicFrame(13, .94, .96, .95, image)
        return cal, config, marker, descriptor, frame

    def call(self, setup, **kwargs):
        cal, config, marker, descriptor, frame = setup
        return recover_gold_markers(frame, cal, config, {marker.marker_id: marker},
                                    {marker.marker_id: descriptor}, [], **kwargs)

    def test_recovers_only_the_existing_marker_without_mutation_or_terminal_evidence(self):
        setup = self.setup()
        before = tuple(setup[2].observations)
        diagnostics = {}
        result = self.call(setup, diagnostics=diagnostics)
        self.assertEqual(set(result), {7})
        detection = result[7]
        self.assertLess(np.linalg.norm(np.asarray(detection.center)-(640, 560)), 4.)
        self.assertTrue(detection.physical_ring)
        self.assertEqual((detection.topology, detection.ribbon_exit_count, detection.owner_lanes),
                         ('unknown', 0, ()))
        self.assertEqual(tuple(setup[2].observations), before)
        self.assertEqual(diagnostics[7]['capture_finished'], .96)

    def test_fixed_judgement_white_core_cannot_replace_a_moving_identity(self):
        setup = self.setup()
        image = setup[-1].image
        image[(np.indices(image.shape[:2])[0]-560)**2
              +(np.indices(image.shape[:2])[1]-640)**2 < 19**2] = (255, 255, 255)
        self.assertEqual(self.call(setup), {})

    def test_same_frame_and_short_freeze_do_not_add_an_observation(self):
        setup = self.setup()
        self.assertEqual(self.call((*setup[:-1], replace(setup[-1], sequence=12))), {})
        setup[-1].image[:] = 0
        hourglass(setup[-1].image, setup[2].observations[-1].center)
        self.assertEqual(self.call(setup), {})

    def test_stale_stationary_unowned_and_nonphysical_histories_are_rejected(self):
        for kind in ('stale', 'stationary', 'unowned', 'nonphysical', 'legacy'):
            setup = list(self.setup())
            if kind == 'stale':
                setup[-1] = replace(setup[-1], midpoint=1.3, sequence=20)
            elif kind == 'stationary':
                setup[2].observations = type(setup[2].observations)(
                    [replace(o, center=(640., 540.), progress=.8) for o in setup[2].observations], maxlen=12)
            elif kind == 'unowned':
                setup[2].owner = None
            elif kind == 'nonphysical':
                setup[3] = replace(setup[3], physical_ring=None)
            else:
                setup[1] = replace(setup[1], hold_notes_as_taps=False)
            with self.subTest(kind=kind):
                self.assertEqual(self.call(setup), {})

    def test_positive_reacquisition_after_two_periods_does_not_mean_unbounded_coast(self):
        setup = list(self.setup())
        setup[-1].image[:] = 0
        hourglass(setup[-1].image, (640., 585.))
        setup[-1] = replace(setup[-1], sequence=14, midpoint=1.012,
                            capture_started=1.002, capture_finished=1.022)
        result = self.call(setup)
        self.assertEqual(set(result), {7})
        self.assertEqual(result[7].topology, 'unknown')
        # Exactly the same history and age with no positive current pixels
        # must not manufacture a sample or an executable action.
        setup[-1].image[:] = 0
        self.assertEqual(self.call(setup), {})
        self.assertEqual(len(setup[2].observations), 3)

    def test_positive_pixels_cannot_bypass_absolute_age_or_frame_gap_limits(self):
        for midpoint, sequence in ((1.6, 14), (.95, 15)):
            setup = list(self.setup())
            setup[-1] = replace(setup[-1], midpoint=midpoint, sequence=sequence)
            with self.subTest(midpoint=midpoint, sequence=sequence):
                self.assertEqual(self.call(setup), {})

    def test_existing_global_candidate_and_excluded_input_identity_are_not_recovered_again(self):
        setup = self.setup()
        detection = self.call(setup)[7]
        cal, config, marker, descriptor, frame = setup
        self.assertEqual(recover_gold_markers(frame, cal, config, {7: marker}, {7: descriptor}, [detection]), {})
        self.assertEqual(self.call(setup, excluded_ids={7}), {})

    def test_two_indistinguishable_known_ids_do_not_both_take_the_same_circle(self):
        cal, config, marker, descriptor, frame = self.setup()
        second = SustainMarker(8, owner=10)
        second.observations.extend(marker.observations)
        result = recover_gold_markers(frame, cal, config, {7: marker, 8: second},
                                      {7: descriptor, 8: descriptor}, [])
        self.assertEqual(result, {})

    def test_two_distinct_known_rings_are_recovered_one_to_one(self):
        cal, config, first, d1, frame = self.setup(x=590.)
        _, _, second, d2, _ = self.setup(x=690., tid=8, owner=10)
        hourglass(frame.image, (690., 560.))
        # The fixture has parallel synthetic centre lines; use explicit lanes
        # matching each nearby ray so projection, not screen x, remains authority.
        cal.lane_centerlines[3] = [[590., 140.], [590., 400.], [590., 620.]]
        cal.lane_centerlines[4] = [[690., 140.], [690., 400.], [690., 620.]]
        second.observations = type(second.observations)([replace(o, lane=4) for o in second.observations], maxlen=12)
        result = recover_gold_markers(frame, cal, config, {7: first, 8: second}, {7: d1, 8: d2}, [])
        self.assertEqual(set(result), {7, 8})
        self.assertLess(result[7].center[0], result[8].center[0])


if __name__ == '__main__':
    unittest.main()
