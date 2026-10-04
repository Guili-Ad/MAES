"""Unified click identities: HUD rejection without lane or timing bans."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from test_tap_identity_v3 import history
from test_tap_pipeline import tap_event
from agent.music.head_identity import identity_continuity_allowed
from agent.music.models import MusicFrame, NoteGesture, TrackState
from agent.music.tap_identity import coastable_tap, tap_structure_ready, late_birth_ready
from agent.music.tap_physical_identity import reconcile_tap_identities
from agent.music.tap_trace import TapTrace
from agent.music.models import MusicConfig
from agent.music.pending_eligibility import valid_pending
from agent.music.tracking import LaneProjection, assign_lane


def point_track(values, tid=1, lane=3, family='ordinary'):
    track = history(values, tid, lane)
    track.point_mode, track.visual_family = True, family
    track.timing_profile = family
    track.physical_id = f'physical-{tid}'
    track.first_seen_time = values[0][0]
    if family == 'yellow_head':
        track.gesture, track.hold_evidence_frames = NoteGesture.HOLD_START, 2
    elif family == 'bonus':
        track.bonus_star = True
        track.observations = type(track.observations)([
            replace(o, candidate=replace(o.candidate, variant='bonus_star'))
            for o in track.observations], maxlen=12)
    if len(values) > 1:
        track.speed = (values[-1][1]-values[0][1])/(values[-1][0]-values[0][0])
    return track


def paint_point(image, candidate, family):
    x, y, w, h = candidate.box
    yy, xx = np.ogrid[:h, :w]
    d = ((xx-(w-1)/2)/(w/2))**2 + ((yy-(h-1)/2)/(h/2))**2
    color = {'ordinary': [190, 220, 20], 'yellow_head': [20, 170, 240],
             'bonus': [20, 220, 130]}[family]
    crop = image[y:y+h, x:x+w]
    crop[d < .85**2] = color
    crop[d < .20**2] = 255


class PointHeadIdentityTests(unittest.TestCase):
    def setUp(self):
        self.config, self.trace = MusicConfig(), TapTrace(MusicConfig())

    def test_static_birth_cannot_take_true_head_with_tiny_progress_jump(self):
        for family in ('ordinary', 'yellow_head', 'bonus'):
            track = point_track([(1., .6501), (1.05, .6501), (1.10, .6501)], family=family)
            note = candidate_at(calibration(), 3, .6518)
            if family == 'bonus':
                note = replace(note, variant='bonus_star')
            self.assertFalse(identity_continuity_allowed(track, note,
                             LaneProjection(3, .6518, 0., (0., 1.)),
                             MusicFrame(3, 1.15, 1.15, 1.15, None)))
            self.assertTrue(track.point_static_origin)

    def test_exact_stationary_observation_is_not_a_new_head(self):
        track = point_track([(1., .65), (1.05, .65), (1.10, .65)])
        note = track.observations[-1].candidate
        self.assertTrue(identity_continuity_allowed(track, note, LaneProjection(3, .65, 0., (0., 1.)),
                        MusicFrame(3, 1.15, 1.15, 1.15, None)))
        self.assertFalse(tap_structure_ready(track, MusicFrame(2, 1.10, 1.10, 1.10, None)))

    def test_previously_healthy_motion_can_freeze_then_resume(self):
        values = [(1., .40), (1.05, .45), (1.1, .50), (1.15, .50), (1.2, .50)]
        track = point_track(values)
        for progress in (.503, .530):
            note = candidate_at(calibration(), 3, progress)
            self.assertTrue(identity_continuity_allowed(track, note, LaneProjection(3, progress, 0., (0., 1.)),
                            MusicFrame(5, 1.25, 1.25, 1.25, None)))
        self.assertFalse(track.point_static_origin)

    def test_fast_captures_still_confirm_static_birth_prefix(self):
        track = point_track([(1.+i*.016, .6501) for i in range(7)])
        note = candidate_at(calibration(), 3, .6518)
        self.assertFalse(identity_continuity_allowed(track, note, LaneProjection(3, .6518, 0., (0., 1.)),
                         MusicFrame(7, 1.112, 1.112, 1.112, None)))
        self.assertTrue(track.point_static_origin)

    def test_static_origin_survives_observation_deque_rollover(self):
        track = point_track([(1., .65), (1.05, .65), (1.10, .65)])
        last = track.observations[-1]
        identity_continuity_allowed(track, last.candidate, LaneProjection(3, .65, 0., (0., 1.)),
                                    MusicFrame(3, 1.15, 1.15, 1.15, None))
        track.observations.clear()
        moving = point_track([(2., .66), (2.05, .70), (2.10, .74)])
        track.observations.extend(moving.observations)
        self.assertFalse(tap_structure_ready(track, MusicFrame(2, 2.10, 2.10, 2.10, None)))

    def test_bonus_and_yellow_current_colored_strokes_are_rejected(self):
        for family, color in [('yellow_head', [20, 170, 240]), ('bonus', [20, 220, 130])]:
            track = point_track([(1., .56), (1.05, .61), (1.10, .66)], family=family)
            image = np.zeros((720, 1280, 3), np.uint8)
            x, y, w, h = track.observations[-1].candidate.box
            image[y:y+h, x+w//2-2:x+w//2+2] = color
            self.assertFalse(tap_structure_ready(track, MusicFrame(2, 1.10, 1.10, 1.10, image)))

    def test_each_family_uses_own_disc_and_unknown_is_not_absence(self):
        for family in ('ordinary', 'yellow_head', 'bonus'):
            track = point_track([(1., .56), (1.05, .61), (1.10, .66)], family=family)
            image = np.zeros((720, 1280, 3), np.uint8)
            paint_point(image, track.observations[-1].candidate, family)
            self.assertTrue(tap_structure_ready(track, MusicFrame(2, 1.1, 1.1, 1.1, image)))
            self.assertTrue(tap_structure_ready(track, MusicFrame(2, 1.1, 1.1, 1.1, None)))

    def test_healthy_bonus_and_yellow_coast_without_changing_legacy_contract(self):
        for family in ('yellow_head', 'bonus'):
            track = point_track([(1., .50), (1.05, .60)], family=family)
            self.assertTrue(coastable_tap(track, self.config, now=1.15))
            track.point_mode = False
            self.assertFalse(coastable_tap(track, self.config, now=1.15))

    def test_pending_yellow_and_bonus_must_not_bypass_dropout_qualification(self):
        for family in ('yellow_head', 'bonus'):
            track = point_track([(1., .60), (1.05, .60), (1.10, .60)], family=family)
            track.state, track.missed_frames = TrackState.TAP_PENDING, 8
            self.assertFalse(valid_pending(tap_event(1, 3, 1.3), {1: track}, self.config,
                                          1.3, self.trace, sequence=10))

    def test_soft_dropout_cancellation_preserves_physical_event_until_positive_reobservation(self):
        track = point_track([(1., .50), (1.05, .55), (1.1, .60)])
        track.state, track.action_executed = TrackState.TAP_PENDING, True
        track.action_event_id = 'original-physical-event'
        track.missed_frames = 9
        event = replace(tap_event(1, 3, 2.9), event_id=track.action_event_id)
        self.assertFalse(valid_pending(event, {1: track}, self.config, 2.71, self.trace, sequence=12))
        self.assertEqual(track.state, TrackState.APPROACHING)
        self.assertFalse(track.action_executed)
        self.assertEqual(track.physical_id, 'physical-1')
        self.assertEqual(track.action_event_id, 'original-physical-event')
        self.assertEqual(track.point_requalification_sequence, 12)
        self.assertFalse(tap_structure_ready(track, MusicFrame(12, 2.71, 2.71, 2.71, None)))
        reappeared = point_track([(2.8, .80)], tid=1).observations[-1]
        track.observations.append(replace(reappeared, frame_sequence=13))
        track.missed_frames = 0
        empty = np.zeros((720, 1280, 3), np.uint8)
        self.assertFalse(tap_structure_ready(track, MusicFrame(13, 2.8, 2.8, 2.8, empty)))
        self.assertEqual(track.point_requalification_sequence, 12)
        positive = empty.copy()
        paint_point(positive, track.observations[-1].candidate, 'ordinary')
        frame = MusicFrame(13, 2.8, 2.8, 2.8, positive)
        self.assertTrue(tap_structure_ready(track, frame))
        self.assertIsNone(track.point_requalification_sequence)
        self.assertTrue(valid_pending(event, {1: track}, self.config, 2.8, self.trace, sequence=13, frame=frame))
        self.assertEqual(track.action_event_id, 'original-physical-event')

    def test_old_sequence_positive_pixels_cannot_revive_cancelled_point(self):
        track = point_track([(1., .50), (1.05, .55), (1.1, .60)])
        track.point_requalification_sequence = 2
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_point(image, track.observations[-1].candidate, 'ordinary')
        self.assertFalse(tap_structure_ready(track, MusicFrame(2, 1.1, 1.1, 1.1, image)))
        self.assertEqual(track.point_requalification_sequence, 2)

    def test_current_negative_after_soft_cancel_is_still_terminal(self):
        track = point_track([(1., .50), (1.05, .55), (1.1, .60)])
        track.point_requalification_sequence = 1
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = track.observations[-1].candidate.box
        image[y:y+h, x+w//2-2:x+w//2+2] = [190, 220, 20]
        self.assertFalse(valid_pending(tap_event(1, 3, 1.2), {1: track}, self.config,
                                      1.1, self.trace, sequence=2,
                                      frame=MusicFrame(2, 1.1, 1.1, 1.1, image)))
        self.assertEqual(track.state, TrackState.LOST)

    def test_legacy_dropout_cancellation_still_retires_track(self):
        track = point_track([(1., .50), (1.05, .55), (1.1, .60)])
        track.point_mode = False
        self.assertFalse(valid_pending(tap_event(1, 3, 2.9), {1: track}, self.config,
                                      2.71, self.trace, sequence=12))
        self.assertEqual(track.state, TrackState.LOST)

    def test_confirmed_static_origin_does_not_use_soft_recovery_path(self):
        track = point_track([(1., .65), (1.05, .65), (1.1, .65)])
        track.point_requalification_sequence = 1
        self.assertFalse(valid_pending(tap_event(1, 3, 1.2), {1: track}, self.config,
                                      1.1, self.trace, sequence=2,
                                      frame=MusicFrame(2, 1.1, 1.1, 1.1, None)))
        self.assertEqual(track.state, TrackState.LOST)

    def test_true_same_lane_pair_cannot_merge_by_time_or_close_progress(self):
        tracks = {i: point_track([(1., .60-i*.02), (1.05, .65-i*.02), (1.1, .70-i*.02)], tid=i)
                  for i in (1, 2)}
        image = np.zeros((720, 1280, 3), np.uint8)
        for track in tracks.values():
            paint_point(image, track.observations[-1].candidate, 'ordinary')
        self.assertEqual(reconcile_tap_identities(tracks, MusicFrame(2, 1.1, 1.1, 1.1, image),
                                                self.config, self.trace), {})

    def test_nearline_two_sample_birth_cannot_use_coast_as_maturity(self):
        for family in ('ordinary', 'yellow_head', 'bonus'):
            track = point_track([(1., .80), (1.05, .86)], family=family)
            self.assertFalse(late_birth_ready(track, lambda c: assign_lane(c, calibration())))
            matured = point_track([(1., .80), (1.05, .83), (1.10, .86)], family=family)
            self.assertTrue(late_birth_ready(matured, lambda c: assign_lane(c, calibration())))

    def test_earlier_two_sample_flight_keeps_existing_coast_rescue(self):
        track = point_track([(1., .45), (1.05, .55)])
        self.assertTrue(late_birth_ready(track, lambda c: assign_lane(c, calibration())))

    def test_coast_and_reborn_yellow_preserve_original_event(self):
        old = point_track([(.8, .30), (.85, .35), (.9, .40)], tid=84, family='yellow_head')
        new = point_track([(1.3, .80), (1.35, .85), (1.4, .90)], tid=83, family='yellow_head')
        old.state, old.action_event_id = TrackState.TAP_PENDING, 'retained-event'
        new.observations = type(new.observations)([
            replace(o, frame_sequence=10+i, candidate=replace(o.candidate,
                    box=(o.candidate.box[0]-13, o.candidate.box[1]-13, 56, 56)))
            for i, o in enumerate(new.observations)], maxlen=12)
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_point(image, new.observations[-1].candidate, 'yellow_head')
        self.assertEqual(reconcile_tap_identities({84: old, 83: new},
                         MusicFrame(12, 1.4, 1.4, 1.4, image), self.config, self.trace), {83: 84})
        self.assertEqual(old.action_event_id, 'retained-event')
        self.assertEqual(new.state, TrackState.LOST)


if __name__ == '__main__':
    unittest.main()
