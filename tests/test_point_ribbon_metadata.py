"""Strict ribbon metadata must never change the established click classifier."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from test_longtap_branch import calibration, candidate_at
from test_point_head_identity import paint_point
from agent.music.models import MusicFrame, MusicActionEvent, MusicConfig, NoteTrack, NoteGesture
from agent.music.tracking import MusicVisionEngine, LaneProjection
from agent.music.holds import bonus_hold_ribbon_present
from agent.music import association


def scene(kind):
    image = np.zeros((720, 1280, 3), np.uint8)
    candidate = replace(candidate_at(calibration(), 3, .50, 60), variant='bonus_star')
    cx, cy = map(int, candidate.center)
    if kind == 'ribbon':
        image[cy-126:cy-29, cx-11:cx+12] = 230
    elif kind == 'one-sided-wall':
        image[cy-140:cy-31, cx-100:cx+10] = 230
    paint_point(image, candidate, 'bonus')
    return image, candidate


def metadata(image, candidate):
    evidence = {}
    try:
        old = bonus_hold_ribbon_present(image, candidate, (0., 1.), evidence=evidence)
    except TypeError:
        old = bonus_hold_ribbon_present(image, candidate, (0., 1.))
    return old, evidence


class PointRibbonMetadataTests(unittest.TestCase):
    def test_one_sided_bright_wall_preserves_old_true_but_metadata_is_false(self):
        image, candidate = scene('one-sided-wall')
        old, evidence = metadata(image, candidate)
        self.assertTrue(old)
        self.assertFalse(evidence.get('strict_bilateral', False))
        self.assertIn('strict_bilateral', evidence)

    def test_true_near_far_ribbon_yields_strict_bilateral_evidence(self):
        image, candidate = scene('ribbon')
        old, evidence = metadata(image, candidate)
        self.assertTrue(old)
        self.assertTrue(evidence.get('strict_bilateral', False))

    def observe(self, track, sequence, progress, *, strict=True, center=None):
        candidate = replace(candidate_at(calibration(), 3, progress, 60), variant='bonus_star')
        if center is not None:
            candidate = replace(candidate, center=center)
        frame = MusicFrame(sequence, sequence*.05, sequence*.05, sequence*.05, None)
        projection = LaneProjection(3, progress, 0., (0., 1.))
        fn = getattr(association, 'observe_point_ribbon_metadata', lambda *_: None)
        fn(track, frame, candidate, projection, {'strict_bilateral': strict})

    def test_two_independent_forward_strict_frames_confirm_metadata(self):
        track = NoteTrack(1, 3, point_mode=True, bonus_star=True)
        self.observe(track, 1, .40)
        self.assertFalse(getattr(track, 'point_ribbon_confirmed', False))
        self.observe(track, 2, .45)
        self.assertTrue(getattr(track, 'point_ribbon_confirmed', False))

    def test_repeat_gap_and_unknown_do_not_manufacture_confirmation(self):
        for scenario in ('repeat', 'gap', 'unknown'):
            track = NoteTrack(1, 3, point_mode=True, bonus_star=True)
            self.observe(track, 1, .40)
            if scenario == 'repeat':
                self.observe(track, 2, .40)
            elif scenario == 'gap':
                self.observe(track, 4, .45)
            else:
                self.observe(track, 2, .45, strict=False)
                self.observe(track, 3, .50)
            self.assertFalse(getattr(track, 'point_ribbon_confirmed', False))

    def test_legacy_head_does_not_gain_new_metadata(self):
        track = NoteTrack(1, 3, point_mode=False, bonus_star=True)
        self.observe(track, 1, .40)
        self.observe(track, 2, .45)
        self.assertFalse(getattr(track, 'point_ribbon_confirmed', False))

    def head(self, confirmed):
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True, enable_holds=True))
        track = NoteTrack(1, 3, point_mode=True, bonus_star=True, hold_evidence_frames=2,
                          visual_family='bonus', timing_profile='yellow_head')
        track.point_ribbon_confirmed = confirmed
        engine.tracks[1] = track
        event = MusicActionEvent('point-star', 1, 3, NoteGesture.TAP, 1., (640, 620),
                                visual_family='bonus', timing_profile='yellow_head')
        receipt = SimpleNamespace(down_call_started=1., down_call_finished=1.001, up_call_finished=1.002, error='')
        engine.tap_hold_chain.acknowledge_head(event, receipt)
        return engine

    def test_bonus_false_whiteband_history_cannot_create_anchor(self):
        self.assertEqual(self.head(False).tap_hold_chain.anchors, {})

    def test_confirmed_bonus_ribbon_creates_metadata_anchor(self):
        self.assertIn(1, self.head(True).tap_hold_chain.anchors)


if __name__ == '__main__':
    unittest.main()
