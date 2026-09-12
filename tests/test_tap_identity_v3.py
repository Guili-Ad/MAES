"""Test2 SUPPORT regressions; timing/progress fixtures, not song action tables."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from test_tap_pipeline import observed_track, tap_event
from agent.music.models import NoteTrack, NoteGesture, TrackState, TrackObservation, MusicFrame, MusicConfig
from agent.music.tracking import MusicVisionEngine, LaneProjection
from agent.music.tap_tracking import associate_taps
from agent.music.tap_chords import TapChordManager
from agent.music.tap_policy import TapTimingPolicy
from agent.music.tap_trace import TapTrace
from agent.music.tap_identity import retire_bonus_fragments, late_birth_ready
from agent.music.tap_recovery import recover_masked_taps
from agent.music.tracking import assign_lane
from agent.music.vision import VisualMask


def history(values, tid=1, lane=3):
    track = NoteTrack(tid, lane)
    for seq, (time, progress) in enumerate(values):
        c = candidate_at(calibration(), lane, progress)
        track.observations.append(TrackObservation(seq, time, c.center, progress, c))
    return track


class TapIdentityV3Tests(unittest.TestCase):
    def match(self, tracks, time, progress, seq=10, safe=True):
        c = candidate_at(calibration(), 3, progress)
        return associate_taps(tracks, [(c, LaneProjection(3, progress, 0., (0., 1.)))],
                              MusicFrame(seq, time, time, time, None), MusicConfig(),
                              lambda *_: safe, TapTrace(MusicConfig()))

    def test_singleton_cannot_reappear_three_quarters_second_later(self):
        t = history([(0., .6212)])
        self.assertEqual(self.match([t], .7584, .6662), {})

    def test_old_low_progress_star_fragment_cannot_capture_next_head(self):
        t = history([(0., .2848)])
        self.assertEqual(self.match([t], .9679, .2810), {})

    def test_two_static_plateaus_cannot_become_a_new_moving_head(self):
        t = history([(0., .5839), (.5027, .6535), (.5552, .6535), (.6093, .6501)])
        self.assertEqual(self.match([t], 1.0167, .6790), {})

    def test_mature_identity_does_not_steal_a_fresh_adjacent_head(self):
        real = history([(0., .52), (.06, .56), (.12, .60)], 2)
        real.speed = 2/3
        particle = history([(.12, .65)], 1)
        found = self.match([particle, real], .21, .651, seq=4)
        self.assertIs(found[0], particle)

    def test_short_singleton_dropout_is_still_matchable(self):
        t = history([(0., .2848)])
        self.assertIs(self.match([t], .14, .33, seq=2)[0], t)

    def test_hold_and_uncertain_candidates_keep_original_path(self):
        for hold, safe in [(True, True), (False, False)]:
            t = history([(0., .6212)])
            if hold:
                t.gesture = NoteGesture.HOLD_START
            self.assertIs(self.match([t], .7584, .6662, safe=safe)[0], t)

    def test_prediction_outlier_cannot_delay_both_chord_members(self):
        tracks = {1: observed_track(1, 1, .8, hit=2.),
                  2: observed_track(2, 5, .8, hit=2.414)}
        tracks[1].linked_partner_id, tracks[2].linked_partner_id = 2, 1
        manager = TapChordManager(TapTimingPolicy(MusicConfig(), 7), TapTrace(MusicConfig()))
        events = manager.refine([tap_event(1, 1, 1.875), tap_event(2, 5, 2.289)], tracks, 1., 2)
        self.assertTrue(all(e.tap_group_id is None for e in events))
        self.assertAlmostEqual(events[0].deadline, 1.875)

    def test_bad_prediction_does_not_retime_previously_frozen_chord(self):
        tracks = {1: observed_track(1, 1, .8, hit=1.5), 2: observed_track(2, 5, .8, hit=1.5)}
        tracks[1].linked_partner_id, tracks[2].linked_partner_id = 2, 1
        manager = TapChordManager(TapTimingPolicy(MusicConfig(), 7), TapTrace(MusicConfig()))
        events = manager.refine([tap_event(1, 1, 1.375), tap_event(2, 5, 1.375)], tracks, 1.36, 2)
        tracks[2].predicted_hit_time = 2.0
        updated = manager.refine(events, tracks, 1.361, 2)
        self.assertEqual([e.deadline for e in updated], [1.375, 1.375])

    def test_bonus_transition_retires_fragment_not_neighbour_or_hold(self):
        fragment = history([(0., .28)])
        neighbour = history([(0., .20)], 2)
        hold = history([(0., .28)], 3)
        hold.gesture = NoteGesture.HOLD_START
        star = replace(candidate_at(calibration(), 3, .31, 40), variant='bonus_star')
        tracks = {t.track_id: t for t in (fragment, neighbour, hold)}
        retire_bonus_fragments(tracks, [star], MusicFrame(1, .06, .06, .06, None),
                               lambda c: assign_lane(c, calibration()), TapTrace(MusicConfig()))
        self.assertEqual(fragment.state, TrackState.LOST)
        self.assertEqual(neighbour.state, TrackState.APPROACHING)
        self.assertEqual(hold.state, TrackState.APPROACHING)

    def recover_scene(self, *, color=(180, 220, 20), white_core=True, gesture=NoteGesture.TAP,
                      sent=False, neighbour=False, empty=False):
        cal = replace(calibration(), exclusion_rois=[[579, 550, 122, 140]])
        engine = MusicVisionEngine(cal, MusicConfig())
        track = history([(0., .76), (.05, .80), (.10, .84)])
        track.observations.clear()
        for i, p in enumerate([.76, .80, .84]):
            c = candidate_at(cal, 3, p, 60)
            track.observations.append(TrackObservation(i, i*.05, c.center, p, c))
        track.gesture = gesture
        if sent:
            track.tap_input_started = .11
        engine.tracks[1] = track
        note = candidate_at(cal, 3, .88, 64)
        image = np.zeros((720, 1280, 3), np.uint8)
        yy, xx = np.ogrid[:720, :1280]
        d = (xx-note.center[0])**2+(yy-note.center[1])**2
        if not empty:
            image[d < 32**2] = color
            if white_core:
                image[d < 7**2] = 255
        if neighbour:
            other = NoteTrack(2, 3)
            other.observations.append(TrackObservation(3, .15, note.center, .88, note))
            engine.tracks[2] = other
        frame = MusicFrame(3,.15,.15,.15,image)
        recovered = recover_masked_taps(engine.tracks, frame, cal, lambda c: assign_lane(c, cal))
        engine._associate_lane(3, [], frame, VisualMask.from_image(image, cal), recovered)
        return track, engine

    def test_masked_confirmed_teal_head_gets_owner_only_observation(self):
        t, engine = self.recover_scene()
        self.assertEqual(t.observations[-1].frame_sequence, 3)
        self.assertAlmostEqual(t.observations[-1].progress, .88, delta=.005)
        self.assertEqual(len(engine.tracks), 1)
        self.assertEqual(engine.tap_trace.records[-1]['kind'], 'tap_mask_recovered')

    def test_recovery_does_not_touch_holds_sent_heads_or_foreign_contours(self):
        for args in [{'gesture': NoteGesture.HOLD_START}, {'sent': True}, {'neighbour': True}]:
            t, _ = self.recover_scene(**args)
            self.assertEqual(t.observations[-1].frame_sequence, 2)

    def test_judgement_ring_and_solid_particle_do_not_recover(self):
        for args in [{'color': (30, 210, 240)}, {'white_core': False}, {'empty': True}]:
            t, _ = self.recover_scene(**args)
            self.assertEqual(t.observations[-1].frame_sequence, 2)

    def test_late_particle_backward_jitter_is_not_an_urgent_head(self):
        t = history([(0., .8039), (.033, .7782), (.066, .8974)])
        self.assertFalse(late_birth_ready(t, lambda c: assign_lane(c, calibration())))

    def test_late_real_head_is_allowed_after_forward_in_lane_evidence(self):
        t = history([(0., .80), (.033, .84), (.066, .88)])
        self.assertTrue(late_birth_ready(t, lambda c: assign_lane(c, calibration())))

    def test_repeated_capture_alone_does_not_disqualify_late_head(self):
        t = history([(0., .80), (.033, .80), (.066, .84)])
        self.assertTrue(late_birth_ready(t, lambda c: assign_lane(c, calibration())))

    def test_late_off_lane_particle_does_not_qualify_on_progress_alone(self):
        t = history([(0., .80), (.033, .84), (.066, .88)])
        first = t.observations.popleft()
        bad = replace(first.candidate, center=(first.center[0]+40, first.center[1]))
        t.observations.appendleft(replace(first, center=bad.center, candidate=bad))
        self.assertFalse(late_birth_ready(t, lambda c: assign_lane(c, calibration())))

    def test_recovered_owner_and_clipped_box_do_not_create_second_identity(self):
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig())
        t = observed_track(1, 3, .84)
        t.state, t.action_executed, t.action_event_id = TrackState.TAP_PENDING, True, 'stable-event'
        engine.tracks[1], engine.next_track_id = t, 2
        image = np.zeros((720,1280,3), np.uint8)
        visual = VisualMask.from_image(image, cal)
        for seq, p in [(3,.90),(4,.95)]:
            c = candidate_at(cal,3,p,60)
            clipped = replace(c, box=(c.box[0],c.box[1],60,34), center=(c.center[0],c.center[1]-13))
            entries = [(clipped, assign_lane(clipped, cal))]
            frame = MusicFrame(seq,1.+seq*.03,1.+seq*.03,1.+seq*.03,image)
            engine._associate_lane(3,entries,frame,visual,{1:(c,assign_lane(c,cal))})
        self.assertEqual(len(engine.tracks),1)
        self.assertEqual(t.action_event_id,'stable-event')
        self.assertAlmostEqual(t.observations[-1].progress,.95)


if __name__ == '__main__':
    unittest.main()
