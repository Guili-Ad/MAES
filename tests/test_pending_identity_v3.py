"""Queue eligibility regressions: identity evidence, never a timing retune."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from test_tap_pipeline import observed_track, tap_event
from agent.music.models import MusicConfig, MusicFrame, TrackObservation, NoteGesture, TrackState
from agent.music.tap_trace import TapTrace
from agent.music.tap_tracking import associate_taps, retire_converged_shadows
from agent.music.tracking import LaneProjection
from agent.music import tap_identity
from agent.music import head_identity


def eligibility(event, tracks, config, now, trace, **kwargs):
    # The old production path dispatched queued events without a qualification
    # check. This fallback captures its behaviour before the new API exists.
    try:
        from agent.music.pending_eligibility import valid_pending
    except ImportError:
        return True
    return valid_pending(event, tracks, config, now, trace, **kwargs)


def history(tid, values, size=30):
    track = observed_track(tid, 3, .8)
    track.observations.clear()
    for sequence, (stamp, progress) in enumerate(values):
        note = candidate_at(calibration(), 3, progress, size)
        track.observations.append(TrackObservation(sequence, stamp, note.center, progress, note))
    track.state, track.action_executed, track.action_event_id = TrackState.TAP_PENDING, True, str(tid)
    return track


class PendingIdentityV3Tests(unittest.TestCase):
    def check(self, track, now=1.1, **kwargs):
        return eligibility(tap_event(track.track_id, track.lane, now), {track.track_id: track},
                           MusicConfig(), now, TapTrace(MusicConfig()), **kwargs)

    def test_stale_unhealthy_queued_head_is_not_dispatched(self):
        track = history(487, [(0., .462), (.05, .475)])
        track.speed, track.missed_frames = .06, 8
        self.assertFalse(self.check(track, .689, sequence=10))

    def test_prediction_none_with_explicit_stationary_gap_is_cancelled(self):
        track = history(10, [(0., .61), (.52, .65), (.56, .65)])
        track.speed, track.predicted_hit_time = .01, None
        self.assertFalse(self.check(track, .59, sequence=3))

    def test_prediction_none_alone_does_not_cancel_short_dropout(self):
        track = history(11, [(1., .60), (1.05, .65), (1.10, .70)])
        track.predicted_hit_time, track.missed_frames = None, 2
        self.assertTrue(self.check(track, 1.15, sequence=4))

    def test_healthy_two_sample_coast_remains_valid(self):
        track = history(12, [(1., .50), (1.05, .60)])
        track.speed, track.missed_frames = 2., 8
        self.assertTrue(self.check(track, 1.45, sequence=9))

    def test_coast_expires_after_existing_age_budget(self):
        track = history(13, [(1., .50), (1.05, .60)])
        track.speed, track.missed_frames = 2., 8
        self.assertFalse(self.check(track, 2.651, sequence=9))

    def test_adaptive_coast_speed_is_honoured(self):
        track = history(14, [(1., .50), (1.05, .52)])
        track.speed, track.missed_frames = .13, 8
        self.assertTrue(self.check(track, 1.3, sequence=9, min_speed=.12))
        self.assertFalse(self.check(track, 1.3, sequence=9, min_speed=.25))

    def test_association_rejection_is_not_track_level_cancellation(self):
        track = history(15, [(1., .60), (1.05, .65), (1.10, .70)])
        trace = TapTrace(MusicConfig())
        trace.add('tap_association_rejected', track=15, reason='stationary-impostor', time=1.11)
        self.assertTrue(eligibility(tap_event(15, 3, 1.14), {15: track}, MusicConfig(),
                                    1.12, trace, sequence=3))

    def test_short_exact_contour_freeze_remains_valid(self):
        track = history(16, [(1., .84), (1.03, .88), (1.06, .92)])
        track.tap_contour_seen_time, track.missed_frames = 1.12, 0
        self.assertTrue(self.check(track, 1.14, sequence=5))

    def test_terminal_and_started_events_cannot_be_resent(self):
        track = history(17, [(1., .60), (1.05, .65)])
        for state in (TrackState.LOST, TrackState.RELEASED):
            track.state = state
            self.assertFalse(self.check(track))
        track.state, track.tap_input_started = TrackState.TAP_PENDING, 1.08
        self.assertFalse(self.check(track))

    def test_hold_markers_and_non_taps_keep_their_own_policy(self):
        trace = TapTrace(MusicConfig())
        event = tap_event(-1001, 3, 1.)
        self.assertTrue(eligibility(event, {}, MusicConfig(), 2., trace))
        event = replace(tap_event(1, 3, 1.), gesture=NoteGesture.HOLD_START)
        self.assertTrue(eligibility(event, {}, MusicConfig(), 2., trace))

    def test_pruned_observed_owner_cannot_revive_a_queued_event(self):
        trace = TapTrace(MusicConfig())
        observed = replace(tap_event(50, 3, 1.), source_capture_started=.8, source_capture_finished=.81)
        self.assertFalse(eligibility(observed, {}, MusicConfig(), 1., trace))
        # Legacy/synthetic inputs still have no owner or visual source.
        self.assertTrue(eligibility(tap_event(50, 3, 1.), {}, MusicConfig(), 1., trace))

    def test_same_colour_upper_static_flick_cannot_inherit_moving_note(self):
        static = history(18, [(1., .377), (1.05, .377), (1.10, .377)])
        static.flick, static.flick_direction, static.gesture = True, NoteGesture.FLICK_RIGHT, NoteGesture.FLICK_RIGHT
        note = replace(candidate_at(calibration(), 3, .402), variant='flick',
                       flick_direction=NoteGesture.FLICK_RIGHT, flick_color='blue')
        frame = MusicFrame(3, 1.15, 1.15, 1.15, None)
        entries = [(note, LaneProjection(3, .402, 0., (0., 1.)))]
        found = associate_taps([static], entries, frame, MusicConfig(), lambda *_: False,
                               TapTrace(MusicConfig()))
        self.assertEqual(found, {})

    def test_static_background_does_not_delete_a_moving_candidate(self):
        static = history(19, [(1., .377), (1.05, .377), (1.10, .377)])
        moving = history(20, [(1., .25), (1.05, .30), (1.10, .35)])
        for track in (static, moving):
            track.flick, track.flick_direction, track.gesture = True, NoteGesture.FLICK_RIGHT, NoteGesture.FLICK_RIGHT
        moving.speed = 1.
        note = replace(candidate_at(calibration(), 3, .402), variant='flick',
                       flick_direction=NoteGesture.FLICK_RIGHT, flick_color='blue')
        found = associate_taps([static, moving], [(note, LaneProjection(3, .402, 0., (0., 1.)))],
                               MusicFrame(3, 1.15, 1.15, 1.15, None), MusicConfig(),
                               lambda *_: False, TapTrace(MusicConfig()))
        self.assertIs(found[0], moving)

    def test_confirmed_round_hold_head_cannot_associate_off_corridor_flat_glyph(self):
        track = history(30, [(1., .42), (1.05, .46), (1.10, .49)], size=56)
        track.gesture, track.state, track.speed = NoteGesture.HOLD_START, TrackState.HOLD_PENDING, 1.
        previous = track.observations[-1].candidate
        glyph = replace(previous, box=(534,452,76,34), center=(572.,469.))
        frame = MusicFrame(3, 1.25, 1.25, 1.25, None)
        entries = [(glyph, LaneProjection(3, .64, 68., (0.,1.)))]
        found = associate_taps([track], entries, frame, MusicConfig(), lambda *_: False,
                               TapTrace(MusicConfig()))
        self.assertEqual(found, {})

    def test_round_head_and_legit_clipped_in_corridor_head_keep_identity(self):
        track = history(31, [(1., .42), (1.05, .46), (1.10, .49)], size=56)
        track.gesture, track.state, track.speed = NoteGesture.HOLD_START, TrackState.HOLD_PENDING, 1.
        previous = track.observations[-1].candidate
        for shape, distance in (((76,76), 0.), ((76,34), 0.)):
            w, h = shape
            note = replace(previous, box=(640-w//2,440,w,h), center=(640.,440+h/2))
            found = associate_taps([track], [(note, LaneProjection(3,.64,distance,(0.,1.)))],
                                   MusicFrame(3,1.25,1.25,1.25,None), MusicConfig(),
                                   lambda *_: False, TapTrace(MusicConfig()))
            self.assertIs(found[0],track)

    def test_queued_collapsed_shadow_can_be_retired_before_touchdown(self):
        shadow, real = observed_track(21, 3, .80), observed_track(22, 3, .82)
        for track in (shadow, real):
            track.state, track.action_executed, track.action_event_id = TrackState.TAP_PENDING, True, str(track.track_id)
            note = candidate_at(calibration(), 3, .84, 80)
            track.observations.append(TrackObservation(3, 1.03, note.center, .84, note))
        obs = list(shadow.observations)
        obs[1] = replace(obs[1], candidate=replace(obs[1].candidate, box=(630, 450, 6, 6)))
        shadow.observations.clear()
        shadow.observations.extend(obs)
        retire_converged_shadows({21: shadow, 22: real}, MusicFrame(3, 1.03, 1.03, 1.03, None),
                                 TapTrace(MusicConfig()))
        self.assertEqual(shadow.state, TrackState.LOST)
        self.assertFalse(self.check(shadow, 1.04))

    def test_started_shadow_is_never_cancelled_or_reissued(self):
        shadow, real = observed_track(23, 3, .80), observed_track(24, 3, .82)
        for track in (shadow, real):
            track.state = TrackState.TAP_PENDING
            note = candidate_at(calibration(), 3, .84, 80)
            track.observations.append(TrackObservation(3, 1.03, note.center, .84, note))
        obs = list(shadow.observations)
        obs[1] = replace(obs[1], candidate=replace(obs[1].candidate, box=(630, 450, 6, 6)))
        shadow.observations.clear()
        shadow.observations.extend(obs)
        shadow.tap_input_started = 1.02
        retire_converged_shadows({23: shadow, 24: real}, MusicFrame(3, 1.03, 1.03, 1.03, None),
                                 TapTrace(MusicConfig()))
        self.assertEqual(shadow.state, TrackState.TAP_PENDING)

    def test_hold_identity_helper_requires_same_contour_not_same_time(self):
        shadow, real = observed_track(28, 3, .70), observed_track(29, 3, .72)
        for track in (shadow, real):
            track.gesture, track.state = NoteGesture.HOLD_START, TrackState.HOLD_PENDING
            note = candidate_at(calibration(), 3, .74, 80)
            track.observations.append(TrackObservation(3, 1.03, note.center, .74, note))
        obs = list(shadow.observations)
        obs[1] = replace(obs[1], candidate=replace(obs[1].candidate, box=(630, 450, 6, 6)))
        shadow.observations.clear()
        shadow.observations.extend(obs)
        check = getattr(head_identity, 'duplicate_head_evidence', lambda *_: None)
        frame = MusicFrame(3, 1.03, 1.03, 1.03, None)
        self.assertEqual(check(shadow, real, frame), 'contained-contour-with-prior-size-collapse')
        # The recorded detached ribbon/HUD fragment is not concentric with
        # the big head: temporal closeness alone cannot prove duplication.
        latest = shadow.observations.pop()
        fragment = replace(latest.candidate, box=(534,452,76,34), center=(572.,469.))
        shadow.observations.append(replace(latest, candidate=fragment, center=fragment.center))
        latest = real.observations.pop()
        head = replace(latest.candidate, box=(582,486,116,114), center=(640.,543.))
        real.observations.append(replace(latest, candidate=head, center=head.center))
        self.assertIsNone(check(shadow, real, frame))

    def test_small_two_sample_hud_stroke_is_not_a_circular_head(self):
        track = history(25, [(1., .56), (1.05, .61)], size=20)
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = track.observations[-1].candidate.box
        image[y:y+h, x+w//2-2:x+w//2+2] = [190, 220, 20]
        frame = MusicFrame(1, 1.05, 1.05, 1.05, image)
        check = getattr(tap_identity, 'tap_structure_ready', lambda *_: True)
        self.assertFalse(check(track, frame))

    def test_small_real_white_core_teal_disc_still_allows_two_samples(self):
        track = history(26, [(1., .56), (1.05, .61)], size=24)
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = track.observations[-1].candidate.box
        yy, xx = np.ogrid[:h, :w]
        d = ((xx-(w-1)/2)/(w/2))**2 + ((yy-(h-1)/2)/(h/2))**2
        crop = image[y:y+h, x:x+w]
        crop[d<.85**2] = [190, 220, 20]
        crop[d<.20**2] = 255
        check = getattr(tap_identity, 'tap_structure_ready', lambda *_: True)
        self.assertTrue(check(track, MusicFrame(1, 1.05, 1.05, 1.05, image)))

    def test_unknown_pixels_and_coasted_old_position_are_not_negative_evidence(self):
        track = history(27, [(1., .56), (1.05, .61)], size=20)
        check = getattr(tap_identity, 'tap_structure_ready', lambda *_: True)
        for image, sequence in ((None, 1), (np.zeros((720, 1280, 3), np.uint8), 1),
                                (np.full((720, 1280, 3), [190, 220, 20], np.uint8), 4)):
            self.assertTrue(check(track, MusicFrame(sequence, 1.15, 1.15, 1.15, image)))


if __name__ == '__main__':
    unittest.main()
