from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BRANCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_longtap_branch import calibration, candidate_at
from test_tap_pipeline import observed_track, tap_event
from agent.music.models import (
    MusicCandidate,
    MusicConfig,
    MusicFrame,
    NoteTrack,
    NoteGesture,
    TrackObservation,
    TrackState,
)
from agent.music.tap_chords import TapChordManager
from agent.music.tap_identity import coastable_tap
from agent.music.tap_policy import TapTimingPolicy
from agent.music.tap_trace import TapTrace
from agent.music.tap_tracking import associate_taps
from agent.music.tracking import LaneProjection, MusicVisionEngine
from agent.music.vision import NumpyCandidateProvider, VisualMask


def note_box(center, radius):
    x, y = center
    return (int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))


class StackedNoteSplitTests(unittest.TestCase):
    def _detect(self, centers):
        cal = calibration()
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        ys, xs = np.ogrid[0:720, 0:1280]
        for cy in centers:
            image[(xs - 640) ** 2 + (ys - cy) ** 2 <= 25 ** 2] = [0, 165, 255]
        visual = VisualMask.from_image(image, cal)
        frame = MusicFrame(1, 0.0, 0.0, 0.0, image)
        provider = NumpyCandidateProvider(cal, 0.55, 18, True)
        return provider.detect(frame, visual)

    def test_two_touching_notes_are_split(self):
        candidates = self._detect((300, 344))
        self.assertEqual(len(candidates), 2)
        centers = sorted(c.center[1] for c in candidates)
        self.assertAlmostEqual(centers[0], 300.0, delta=6.0)
        self.assertAlmostEqual(centers[1], 344.0, delta=6.0)

    def test_single_note_is_not_split(self):
        candidates = self._detect((320,))
        self.assertEqual(len(candidates), 1)


class TextZoneTests(unittest.TestCase):
    def _engine(self) -> MusicVisionEngine:
        config = MusicConfig(
            lane_count=7,
            enable_holds=False,
            stationary_zone_enabled=True,
            stationary_zone_min_tracks=3,
            stationary_zone_freeze_ms=300.0,
        )
        return MusicVisionEngine(calibration(), config)

    def _frozen_track(self, engine, track_id, dx):
        track = NoteTrack(track_id=track_id, lane=3)
        track.state = TrackState.APPROACHING
        note = MusicCandidate(note_box((640 + dx, 459), 7), 200, 0.8, (640.0 + dx, 459.0))
        for index in range(3):
            track.observations.append(TrackObservation(8 + index, 0.6 + index * 0.16, note.center, 0.652, note))
        track.speed = 0.02
        engine.tracks[track_id] = track
        return track

    def test_recurring_frozen_tracks_form_zone_and_retire(self):
        engine = self._engine()
        tracks = [self._frozen_track(engine, tid, dx) for tid, dx in ((1, 0), (2, 2), (3, 4))]
        engine._update_text_zones(MusicFrame(10, 1.0, 1.0, 1.0, None))
        self.assertEqual(len(engine.text_zones), 1)
        for track in tracks:
            self.assertEqual(track.state, TrackState.LOST)
        self.assertTrue(engine._in_text_zone((644.0, 459.0)))
        self.assertFalse(engine._in_text_zone((844.0, 459.0)))
        self.assertTrue(any(record.get('kind') == 'text_zone' for record in engine.tap_trace.records))

    def test_candidate_inside_zone_is_not_deleted_by_position(self):
        engine = self._engine()
        engine.text_zones.append((640.0, 459.0, 48.0))
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask.from_image(image, calibration())
        candidate = MusicCandidate(note_box((640, 459), 7), 200, 0.8, (640.0, 459.0))
        engine.update(MusicFrame(1, 0.0, 0.0, 0.0, image), [candidate], visual)
        self.assertEqual(len(engine.tracks), 1)

    def test_moving_note_crosses_learned_zone_without_loss(self):
        engine = self._engine()
        engine.text_zones.append((640., 459., 48.))
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask.from_image(image, calibration())
        for sequence in range(18):
            note = candidate_at(calibration(), 3, .45+sequence*.03)
            engine.update(MusicFrame(sequence, sequence*.03, sequence*.03, sequence*.03, image), [note], visual)
        self.assertEqual(engine.next_track_id, 2)
        self.assertTrue(any(t.action_executed for t in engine.tracks.values()))

    def test_static_hud_and_repeated_frames_never_schedule_input(self):
        engine = self._engine()
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask.from_image(image, calibration())
        note = candidate_at(calibration(), 3, .65)
        events = []
        for sequence in range(40):
            events.extend(engine.update(MusicFrame(sequence, sequence*.04, sequence*.04, sequence*.04, image), [note], visual))
        self.assertFalse(events)

    def test_star_and_flick_are_not_filtered_by_a_learned_region(self):
        from dataclasses import replace
        for variant in ('bonus_star', 'flick'):
            engine = self._engine()
            engine.text_zones.append((640., 459., 48.))
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            visual = VisualMask.from_image(image, calibration())
            note = replace(candidate_at(calibration(), 3, .665), variant=variant,
                           flick_direction=NoteGesture.FLICK_LEFT)
            engine.update(MusicFrame(1, 0., 0., 0., image), [note], visual)
            self.assertEqual(len(engine.tracks), 1)

    def test_repeated_screenshot_does_not_create_stationary_region(self):
        engine = self._engine()
        tracks = [self._frozen_track(engine, tid, dx) for tid, dx in ((1,0),(2,2),(3,4))]
        image = np.zeros((720,1280,3), dtype=np.uint8)
        engine._stationary_fingerprint = image[::32,::32,:3].copy()
        engine._update_text_zones(MusicFrame(10,1.,1.,1.,image))
        self.assertFalse(engine.text_zones)
        self.assertTrue(all(t.state == TrackState.APPROACHING for t in tracks))


class ImpostorGuardTests(unittest.TestCase):
    def _match(self, guard: bool):
        config = MusicConfig(lane_count=7, stationary_impostor_guard=guard)
        track = observed_track(1, 3, 0.652, speed=0.5)
        entries = [(candidate_at(calibration(), 3, 0.652), LaneProjection(3, 0.652, 0.0, (0.0, 1.0)))]
        trace = TapTrace(config)
        frame = MusicFrame(3, 1.07, 1.07, 1.07, None)
        matches = associate_taps([track], entries, frame, config, lambda *_: True, trace)
        return matches, trace

    def test_stationary_candidate_rejected_by_guard(self):
        matches, trace = self._match(guard=True)
        self.assertEqual(matches, {})
        self.assertTrue(any(record.get('reason') == 'stationary-impostor' for record in trace.records))

    def test_legacy_path_still_matches_without_guard(self):
        matches, _trace = self._match(guard=False)
        self.assertEqual(matches[0].track_id, 1)


class AdaptiveCoastTests(unittest.TestCase):
    def test_median_based_threshold(self):
        config = MusicConfig(
            lane_count=7,
            coast_adaptive_speed=True,
            coast_speed_ratio=0.4,
            coast_speed_floor=0.10,
            coast_min_speed=0.25,
        )
        engine = MusicVisionEngine(calibration(), config)
        for track_id in range(1, 6):
            engine.tracks[track_id] = observed_track(track_id, 3, 0.6, speed=0.30)
        engine._update_coast_threshold()
        self.assertAlmostEqual(engine.coast_speed_threshold, 0.12, places=6)

    def test_coastable_honours_explicit_min_speed(self):
        config = MusicConfig(lane_count=7)
        track = observed_track(1, 3, 0.6, speed=0.13)
        self.assertFalse(coastable_tap(track, config, min_speed=0.25))
        self.assertTrue(coastable_tap(track, config, min_speed=0.12))


class ChordSkewTests(unittest.TestCase):
    def _manager(self):
        config = MusicConfig(lane_count=7, tap_chord_max_skew_ms=90.0)
        manager = TapChordManager(TapTimingPolicy(config, 7), TapTrace(config))
        tracks = {1: observed_track(1, 1, 0.8), 2: observed_track(2, 5, 0.8)}
        tracks[1].linked_partner_id, tracks[2].linked_partner_id = 2, 1
        return manager, tracks

    def test_large_skew_keeps_independent_deadlines(self):
        manager, tracks = self._manager()
        tracks[1].predicted_hit_time = 1.30
        tracks[2].predicted_hit_time = 1.41
        events = [tap_event(1, 1, 1.30), tap_event(2, 5, 1.41)]
        refined = manager.refine(events, tracks, 1.0, 2)
        self.assertTrue(all(event.tap_group_id is None for event in refined))
        self.assertFalse(manager.groups)

    def test_small_skew_still_groups(self):
        manager, tracks = self._manager()
        tracks[1].predicted_hit_time = 1.30
        tracks[2].predicted_hit_time = 1.32
        events = [tap_event(1, 1, 1.30), tap_event(2, 5, 1.32)]
        refined = manager.refine(events, tracks, 1.0, 2)
        self.assertTrue(all(event.tap_group_id is not None for event in refined))


if __name__ == "__main__":
    unittest.main()
