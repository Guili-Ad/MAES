"""Physical Tap evidence: no timing-only deduplication or HUD region bans."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from test_tap_pipeline import tap_event
from agent.music.models import MusicCandidate, MusicConfig, MusicFrame, NoteGesture, NoteTrack, TrackObservation, TrackState
from agent.music.tap_trace import TapTrace
from agent.music.tracking import LaneProjection
from agent.music.tap_identity import tap_structure_ready
from agent.music.pending_eligibility import valid_pending


def flight(tid, times, progress, *, lane=3, size=58, queued=True):
    track = NoteTrack(tid, lane, speed=(progress[-1]-progress[0])/(times[-1]-times[0]))
    for sequence, (stamp, p) in enumerate(zip(times, progress)):
        candidate = candidate_at(calibration(), lane, p, size)
        track.observations.append(TrackObservation(sequence, stamp, candidate.center, p, candidate))
    if queued:
        track.state, track.action_executed, track.action_event_id = TrackState.TAP_PENDING, True, f'original-{tid}'
    return track


def paint_head(image, candidate):
    x, y, w, h = candidate.box
    yy, xx = np.ogrid[:h, :w]
    d = ((xx-(w-1)/2)/(w/2))**2 + ((yy-(h-1)/2)/(h/2))**2
    patch = image[y:y+h, x:x+w]
    patch[d < .85**2] = [190, 220, 20]
    patch[d < .20**2] = 255


def api():
    # Missing old API means there was no physical-contour normalization/alias.
    try:
        from agent.music.tap_physical_identity import normalize_tap_entries, reconcile_tap_identities
    except ImportError:
        return lambda entries, *_args, **_kw: entries, lambda *_args, **_kw: {}
    return normalize_tap_entries, reconcile_tap_identities


class TapPhysicalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.config, self.trace = MusicConfig(), TapTrace(MusicConfig())

    def test_mature_large_hud_stroke_is_negative_not_only_two_samples(self):
        track = flight(1, [1., 1.05, 1.10], [.60, .65, .70], size=84)
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = track.observations[-1].candidate.box
        image[y:y+h, x+w//2-3:x+w//2+3] = [190, 220, 20]
        frame = MusicFrame(2, 1.1, 1.1, 1.1, image)
        self.assertFalse(tap_structure_ready(track, frame))

    def test_pending_large_hud_is_qualified_on_current_pixels(self):
        track = flight(2, [1., 1.05, 1.10], [.60, .65, .70], size=84)
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = track.observations[-1].candidate.box
        image[y:y+h, x+w//2-3:x+w//2+3] = [190, 220, 20]
        frame = MusicFrame(2, 1.1, 1.1, 1.1, image)
        try:
            ready = valid_pending(tap_event(2, 3, 1.11), {2: track}, self.config, 1.1, self.trace,
                                  sequence=2, frame=frame)
        except TypeError:
            ready = True  # pre-fix pending path did not inspect visible shape
        self.assertFalse(ready)

    def test_large_real_head_and_missing_core_occlusion_are_not_rejected(self):
        track = flight(3, [1., 1.05, 1.10], [.60, .65, .70], size=84)
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_head(image, track.observations[-1].candidate)
        self.assertTrue(tap_structure_ready(track, MusicFrame(2, 1.1, 1.1, 1.1, image)))
        x, y, w, h = track.observations[-1].candidate.box
        image[y+h//2-10:y+h//2+10, x+w//2-10:x+w//2+10] = 0
        self.assertTrue(tap_structure_ready(track, MusicFrame(2, 1.1, 1.1, 1.1, image)))

    def test_compound_hud_box_recovers_one_actual_circle_not_its_box_center(self):
        normalize, _ = api()
        image = np.zeros((720, 1280, 3), np.uint8)
        head = candidate_at(calibration(), 3, .875, 80)
        paint_head(image, head)
        x, y, w, h = head.box
        image[y+h//2-2:y+h//2+2, x-18:x+7] = [190, 220, 20]
        box = (x-21, y-2, 122, 84)
        compound = MusicCandidate(box, 5000, .60, (box[0]+61., box[1]+42.))
        frame = MusicFrame(10, 1.5, 1.5, 1.5, image)
        result = normalize([(compound, LaneProjection(3, .875, 0., (0., 1.)))], frame,
                           lambda note: LaneProjection(3, (note.center[1]-140.)/480., 0., (0., 1.)), self.trace)
        self.assertEqual(len(result), 1)
        self.assertLess(result[0][0].box[2], 100)
        self.assertLess(abs(result[0][0].center[0]-head.center[0]), 2.)
        self.assertLess(abs(result[0][0].center[1]-head.center[1]), 2.)

    def test_confirmed_hud_candidate_is_removed_but_unknown_is_kept(self):
        normalize, _ = api()
        note = candidate_at(calibration(), 3, .80, 84)
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = note.box
        image[y:y+h, x+w//2-3:x+w//2+3] = [190, 220, 20]
        entry = (note, LaneProjection(3, .80, 0., (0., 1.)))
        self.assertEqual(normalize([entry], MusicFrame(2, 1., 1., 1., image), lambda _: entry[1], self.trace), [])
        self.assertEqual(normalize([entry], MusicFrame(2, 1., 1., 1., None), lambda _: entry[1], self.trace), [entry])

    def test_coasted_owner_and_reborn_head_alias_only_with_pixel_and_motion_evidence(self):
        _, reconcile = api()
        old = flight(84, [.8, .85, .9], [.30, .35, .40], size=58)
        new = flight(83, [1.3, 1.35, 1.4], [.80, .85, .90], size=84)
        # These are distinct screenshot sequences, not simultaneous heads.
        new.observations = type(new.observations)([replace(o, frame_sequence=10+i) for i, o in enumerate(new.observations)], maxlen=12)
        old.missed_frames = 10
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_head(image, new.observations[-1].candidate)
        frame = MusicFrame(12, 1.4, 1.4, 1.4, image)
        aliases = reconcile({84: old, 83: new}, frame, self.config, self.trace)
        self.assertEqual(aliases, {83: 84})
        self.assertEqual(old.action_event_id, 'original-84')
        self.assertEqual(new.state, TrackState.LOST)
        self.assertEqual(old.observations[-1].frame_sequence, 12)
        self.assertEqual(old.missed_frames, 0)

    def test_time_proximity_without_pixels_never_merges_coasted_owner(self):
        _, reconcile = api()
        old = flight(84, [.8, .85, .9], [.30, .35, .40])
        new = flight(83, [1.3, 1.35, 1.4], [.80, .85, .90], size=84)
        new.observations = type(new.observations)([replace(o, frame_sequence=10+i) for i, o in enumerate(new.observations)], maxlen=12)
        self.assertEqual(reconcile({84: old, 83: new}, MusicFrame(12, 1.4, 1.4, 1.4, None), self.config, self.trace), {})

    def test_58x56_coast_and_122x84_compound_alias_keep_other_real_dense_head(self):
        _, reconcile = api()
        old = flight(84, [.8, .85, .9], [.30, .35, .40], size=58)
        old.observations = type(old.observations)([
            replace(o, candidate=replace(o.candidate, box=(o.candidate.box[0], o.candidate.box[1]+1, 58, 56)))
            for o in old.observations], maxlen=12)
        new = flight(83, [1.3, 1.35, 1.4], [.80, .85, .90], size=80)
        next_head = flight(85, [1.3, 1.35, 1.4], [.60, .65, .70], size=64)
        for track in (new, next_head):
            track.observations = type(track.observations)([
                replace(o, frame_sequence=10+i) for i, o in enumerate(track.observations)], maxlen=12)
        image = np.zeros((720, 1280, 3), np.uint8)
        head = new.observations[-1].candidate
        paint_head(image, head)
        paint_head(image, next_head.observations[-1].candidate)
        x, y, _, _ = head.box
        image[y+38:y+42, x-18:x+7] = [190, 220, 20]
        new.observations = type(new.observations)([
            replace(o, candidate=MusicCandidate((int(o.center[0])-61, int(o.center[1])-42, 122, 84),
                                                 5000, .60, o.center)) for o in new.observations], maxlen=12)
        self.assertEqual(reconcile({84: old, 83: new, 85: next_head},
                                   MusicFrame(12, 1.4, 1.4, 1.4, image), self.config, self.trace), {83: 84})
        self.assertEqual(next_head.state, TrackState.TAP_PENDING)
        self.assertEqual(next_head.action_event_id, 'original-85')

    def test_ambiguous_multiple_new_owners_are_not_force_aliased(self):
        _, reconcile = api()
        old = flight(84, [.8, .85, .9], [.30, .35, .40])
        image = np.zeros((720, 1280, 3), np.uint8)
        tracks = {84: old}
        for tid in (83, 85):
            new = flight(tid, [1.3, 1.35, 1.4], [.80, .85, .90], size=84)
            new.observations = type(new.observations)([
                replace(o, frame_sequence=10+i) for i, o in enumerate(new.observations)], maxlen=12)
            tracks[tid] = new
        paint_head(image, tracks[83].observations[-1].candidate)
        self.assertEqual(reconcile(tracks, MusicFrame(12, 1.4, 1.4, 1.4, image), self.config, self.trace), {})
        self.assertTrue(all(t.state != TrackState.LOST for t in tracks.values()))

    def test_two_and_three_close_same_lane_heads_are_not_aliased(self):
        normalize, reconcile = api()
        for count in (2, 3):
            image = np.zeros((720, 1280, 3), np.uint8)
            tracks = {i: flight(i, [1., 1.05, 1.1], [.69-i*.08, .74-i*.08, .79-i*.08], size=30) for i in range(count)}
            entries = []
            for track in tracks.values():
                note = track.observations[-1].candidate
                paint_head(image, note)
                entries.append((note, LaneProjection(3, track.observations[-1].progress, 0., (0., 1.))))
            frame = MusicFrame(2, 1.1, 1.1, 1.1, image)
            self.assertEqual(len(normalize(entries, frame, lambda note: LaneProjection(3, (note.center[1]-140)/480., 0., (0., 1.)), self.trace)), count)
            self.assertEqual(reconcile(tracks, frame, self.config, self.trace), {})
            self.assertTrue(all(track.state != TrackState.LOST for track in tracks.values()))

    def test_shared_frame_separation_blocks_later_convergence_alias(self):
        _, reconcile = api()
        old = flight(4, [1., 1.05, 1.1], [.70, .75, .80], size=50)
        new = flight(5, [1., 1.05, 1.1], [.62, .67, .72], size=50)
        last = old.observations[-1]
        new.observations[-1] = replace(new.observations[-1], center=last.center, progress=last.progress, candidate=last.candidate)
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_head(image, last.candidate)
        self.assertEqual(reconcile({4: old, 5: new}, MusicFrame(2, 1.1, 1.1, 1.1, image), self.config, self.trace), {})

    def test_sent_owner_can_retire_proven_unstarted_re_recognition_without_resend(self):
        _, reconcile = api()
        old = flight(4, [.8, .85, .9], [.30, .35, .40])
        old.tap_input_started = 1.39
        new = flight(5, [1.3, 1.35, 1.4], [.80, .85, .90], size=84)
        new.observations = type(new.observations)([replace(o, frame_sequence=10+i) for i, o in enumerate(new.observations)], maxlen=12)
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_head(image, new.observations[-1].candidate)
        self.assertEqual(reconcile({4: old, 5: new}, MusicFrame(12, 1.4, 1.4, 1.4, image), self.config, self.trace), {5: 4})
        self.assertEqual(new.state, TrackState.LOST)
        self.assertEqual(old.tap_input_started, 1.39)
        self.assertEqual(old.action_event_id, 'original-4')
        self.assertEqual(old.state, TrackState.TAP_PENDING)

    def test_two_already_sent_inputs_are_not_merged_to_hide_duplicate(self):
        _, reconcile = api()
        old = flight(4, [.8, .85, .9], [.30, .35, .40])
        new = flight(5, [1.3, 1.35, 1.4], [.80, .85, .90], size=84)
        old.tap_input_started, new.tap_input_started = 1.39, 1.4
        new.observations = type(new.observations)([replace(o, frame_sequence=10+i) for i, o in enumerate(new.observations)], maxlen=12)
        image = np.zeros((720, 1280, 3), np.uint8)
        paint_head(image, new.observations[-1].candidate)
        self.assertEqual(reconcile({4: old, 5: new}, MusicFrame(12, 1.4, 1.4, 1.4, image), self.config, self.trace), {})
        self.assertNotEqual(old.state, TrackState.LOST)
        self.assertNotEqual(new.state, TrackState.LOST)

    def test_hold_star_and_flick_do_not_enter_ordinary_normalization(self):
        normalize, reconcile = api()
        image = np.full((720, 1280, 3), [190, 220, 20], np.uint8)
        note = candidate_at(calibration(), 3, .8, 84)
        variants = [replace(note, variant='bonus_star'), replace(note, variant='flick', flick_direction=NoteGesture.FLICK_RIGHT), note]
        entries = [(n, LaneProjection(3, .8, 0., (0., 1.))) for n in variants]
        frame = MusicFrame(2, 1.1, 1.1, 1.1, image)
        self.assertEqual(normalize(entries, frame, lambda _: entries[0][1], self.trace, ordinary=lambda *_: False), entries)
        for gesture in (NoteGesture.HOLD_START, NoteGesture.FLICK_RIGHT):
            track = flight(1, [1., 1.05, 1.1], [.6, .65, .70])
            track.gesture = gesture
            self.assertEqual(reconcile({1: track}, frame, self.config, self.trace), {})

    def test_all_ordinary_lanes_use_same_pending_dropout_policy(self):
        for lane in (0, 3, 4, 6):
            track = flight(100+lane, [1., 1.05, 1.1], [.60, .60, .60], lane=lane)
            track.speed, track.missed_frames = .01, 8
            self.assertFalse(valid_pending(tap_event(track.track_id, lane, 1.6), {track.track_id: track},
                                           self.config, 1.6, self.trace, sequence=10))

    def test_pending_coast_cannot_bypass_birth_centre_or_single_owner_gate(self):
        track = flight(150, [1., 1.05, 1.1], [.50, .55, .60])
        track.missed_frames = 8
        self.assertTrue(valid_pending(tap_event(150, 3, 1.5), {150: track}, self.config,
                                      1.4, self.trace, sequence=10))
        self.assertFalse(valid_pending(tap_event(150, 3, 1.5), {150: track}, self.config,
                                       1.4, self.trace, sequence=10, coast_eligible=lambda _: False))


if __name__ == '__main__':
    unittest.main()
