"""Owner-only compound-contour recovery; no chart-specific actions."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from test_point_head_identity import point_track, paint_point
from agent.music.models import MusicConfig, MusicFrame, NoteGesture, TrackState
from agent.music.tap_recovery import recover_masked_taps
from agent.music.tracking import MusicVisionEngine, assign_lane
from agent.music.vision import VisualMask, NumpyCandidateProvider


def scene(family='ordinary', tracks=None, progresses=(.53,)):
    cal = calibration()
    # Deliberately no judgement exclusion: this regression is half-flight.
    tracks = tracks or {1: point_track([(1., .38), (1.05, .43), (1.10, .48)], family=family)}
    image = np.zeros((720, 1280, 3), np.uint8)
    for progress in progresses:
        c = candidate_at(cal, 3, progress)
        paint_point(image, c, family)
        # An attached stage/HUD stroke makes the global component oversized.
        y = int(c.center[1])+8
        image[y:y+3, 510:770] = {'ordinary': [190, 220, 20],
            'yellow_head': [20, 170, 240], 'bonus': [20, 220, 130]}[family]
    frame = MusicFrame(3, 1.145, 1.155, 1.15, image)
    return cal, tracks, frame


class OwnedHeadRecoveryTests(unittest.TestCase):
    def recover(self, cal, tracks, frame):
        return recover_masked_taps(tracks, frame, cal, lambda c: assign_lane(c, cal))

    def test_half_flight_compound_each_family_keeps_physical_owner(self):
        for family in ('ordinary', 'bonus', 'yellow_head'):
            with self.subTest(family=family):
                cal, tracks, frame = scene(family)
                recovered = self.recover(cal, tracks, frame)
                self.assertEqual(set(recovered), {1})
                candidate, projection = recovered[1]
                self.assertAlmostEqual(projection.progress, .53, delta=.006)
                self.assertEqual(candidate.variant, 'bonus_star' if family == 'bonus' else '')
                self.assertEqual(len(tracks[1].observations), 3)  # extraction never mutates

    def test_provider_size_filter_stays_strict_and_owner_recovery_sees_raw_pixels(self):
        cal, tracks, frame = scene()
        visual = VisualMask.from_image(frame.image, cal)
        provider = NumpyCandidateProvider(cal)
        self.assertFalse(provider.detect(frame, visual))
        self.assertEqual(set(self.recover(cal, tracks, frame)), {1})

    def test_recovered_head_updates_same_track_event_history_without_birth(self):
        for family in ('ordinary', 'bonus', 'yellow_head'):
            with self.subTest(family=family):
                cal, tracks, frame = scene(family)
                track = tracks[1]
                track.state, track.action_executed = TrackState.TAP_PENDING, True
                track.action_event_id = 'retained-event'
                engine = MusicVisionEngine(cal, MusicConfig(enable_holds=True, hold_notes_as_taps=True))
                engine.tracks = tracks
                recovered = self.recover(cal, tracks, frame)
                engine._associate_lane(3, [], frame, VisualMask.from_image(frame.image, cal), recovered)
                self.assertEqual(set(engine.tracks), {1})
                self.assertEqual(track.observations[-1].frame_sequence, 3)
                self.assertEqual(track.action_event_id, 'retained-event')
                self.assertEqual(track.physical_id, 'physical-1')
                self.assertEqual(len(track.observations), 4)

    def test_same_lane_two_heads_are_one_to_one_in_confirmed_order(self):
        tracks = {i: point_track([(1., .38-gap), (1.05, .43-gap), (1.1, .48-gap)], tid=i)
                  for i, gap in ((1, 0.), (2, .09))}
        cal, tracks, frame = scene(tracks=tracks, progresses=(.53, .44))
        recovered = self.recover(cal, tracks, frame)
        self.assertEqual(set(recovered), {1, 2})
        self.assertGreater(recovered[1][1].progress, recovered[2][1].progress)
        self.assertGreater(np.linalg.norm(np.subtract(recovered[1][0].center, recovered[2][0].center)), 30.)

    def test_single_remaining_head_cannot_refresh_both_owners(self):
        tracks = {i: point_track([(1., .38-gap), (1.05, .43-gap), (1.1, .48-gap)], tid=i)
                  for i, gap in ((1, 0.), (2, .09))}
        cal, tracks, frame = scene(tracks=tracks, progresses=(.44,))
        recovered = self.recover(cal, tracks, frame)
        self.assertEqual(set(recovered), {2})

    def test_static_hud_solid_particle_empty_ring_and_capture_repeat_do_not_recover(self):
        for kind in ('hud', 'solid', 'ring', 'repeat', 'stale', 'sent', 'legacy-hold', 'flick'):
            with self.subTest(kind=kind):
                cal, tracks, frame = scene()
                track = tracks[1]
                if kind in ('hud', 'solid', 'ring', 'repeat'):
                    image = np.zeros_like(frame.image)
                    c = candidate_at(cal, 3, .48 if kind == 'repeat' else .53)
                    paint_point(image, c, 'ordinary')
                    x, y, w, h = c.box
                    if kind == 'hud':
                        image.fill(0); image[y:y+h, x+w//2-2:x+w//2+2] = [190, 220, 20]
                    elif kind == 'solid':
                        image[y:y+h, x:x+w] = [190, 220, 20]
                    elif kind == 'ring':
                        image[y+8:y+h-8, x+8:x+w-8] = 0
                    frame = replace(frame, image=image)
                elif kind == 'stale':
                    frame = replace(frame, sequence=10, midpoint=1.8)
                elif kind == 'sent':
                    track.tap_input_started = 1.12
                elif kind == 'legacy-hold':
                    track.point_mode = False; track.gesture = NoteGesture.HOLD_START
                elif kind == 'flick':
                    track.flick = True; track.gesture = NoteGesture.FLICK_RIGHT
                self.assertEqual(self.recover(cal, tracks, frame), {})


if __name__ == '__main__':
    unittest.main()
