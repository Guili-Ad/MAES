"""Coast-through-occlusion and bonus-star hold-bias regression tests."""
from __future__ import annotations

import unittest

import numpy as np

from test_longtap_branch import calibration, candidate_at
from test_tap_identity_v3 import history
from agent.music.models import (
    MusicActionEvent,
    MusicConfig,
    MusicFrame,
    NoteGesture,
    NoteTrack,
    TrackObservation,
    TrackState,
)
from agent.music.tap_dispatch import due_tap_batches
from agent.music.tracking import MusicVisionEngine
from agent.music.vision import VisualMask


def coast_track(config: MusicConfig, progresses=(0.50, 0.60)):
    engine = MusicVisionEngine(calibration(), config)
    track = history([(1.0 + index * 0.05, progress) for index, progress in enumerate(progresses)])
    engine._update_motion(track)
    return engine, track


class CoastBonusTests(unittest.TestCase):
    def test_coast_schedules_occluded_tap_from_prediction(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), hsv=None, roi_origin=(0, 0))
        engine, track = coast_track(MusicConfig(lane_count=7, coast_enabled=True))
        engine.tracks[track.track_id] = track
        frame = MusicFrame(2, 1.10, 1.11, 1.105, image)
        events = engine.update(frame, [], visual)
        self.assertEqual(track.state, TrackState.TAP_PENDING)
        self.assertTrue(any(event.track_id == track.track_id for event in events))

    def test_coast_eligibility_rejects_slow_bonus_flick_and_hold(self) -> None:
        config = MusicConfig(lane_count=7)
        engine, healthy = coast_track(config, (0.50, 0.60))
        frame = MusicFrame(5, 1.15, 1.16, 1.155, None)
        self.assertTrue(engine._coast_eligible(healthy, frame))

        _, slow = coast_track(config, (0.50, 0.505))
        self.assertFalse(engine._coast_eligible(slow, frame))

        _, bonus = coast_track(config, (0.50, 0.60))
        bonus.bonus_star = True
        self.assertFalse(engine._coast_eligible(bonus, frame))

        _, flick = coast_track(config, (0.50, 0.60))
        flick.direction_evidence.append(NoteGesture.FLICK_LEFT)
        self.assertFalse(engine._coast_eligible(flick, frame))

        _, held = coast_track(config, (0.50, 0.60))
        held.hold_evidence_frames = 1
        self.assertFalse(engine._coast_eligible(held, frame))

    def test_coast_eligibility_expires_with_age(self) -> None:
        engine, track = coast_track(MusicConfig(lane_count=7, coast_max_age_ms=500.0))
        fresh = MusicFrame(5, 1.20, 1.21, 1.205, None)
        self.assertTrue(engine._coast_eligible(track, fresh))
        stale = MusicFrame(9, 1.70, 1.71, 1.705, None)
        self.assertFalse(engine._coast_eligible(track, stale))

    def test_coast_requires_center_lane_and_single_owner(self) -> None:
        config = MusicConfig(lane_count=7)
        engine, healthy = coast_track(config)
        frame = MusicFrame(5, 1.15, 1.16, 1.155, None)
        engine.tracks[healthy.track_id] = healthy
        self.assertTrue(engine._coast_eligible(healthy, frame))
        edge = history([(1.0, 0.50), (1.05, 0.60)], tid=90, lane=0)
        engine._update_motion(edge)
        engine.tracks[90] = edge
        self.assertFalse(engine._coast_eligible(edge, frame))
        sibling = history([(1.00, 0.50), (1.05, 0.575), (1.10, 0.65)], tid=91, lane=3)
        engine._update_motion(sibling)
        engine.tracks[91] = sibling
        self.assertFalse(engine._coast_eligible(healthy, frame))
        self.assertTrue(engine._coast_eligible(sibling, frame))

    def test_coast_owner_must_pass_gate(self) -> None:
        config = MusicConfig(lane_count=7)
        engine, healthy = coast_track(config)
        frame = MusicFrame(5, 1.15, 1.16, 1.155, None)
        engine.tracks[healthy.track_id] = healthy
        # The sibling has more observations but fails the speed gate; it must
        # not veto the eligible sibling that can still rescue the note.
        sibling = history([(1.00, 0.600), (1.05, 0.610), (1.10, 0.618), (1.15, 0.624)], tid=91, lane=3)
        engine._update_motion(sibling)
        engine.tracks[91] = sibling
        self.assertFalse(engine._coast_eligible(sibling, frame))
        self.assertTrue(engine._coast_eligible(healthy, frame))

    def test_stale_fragment_does_not_block_confirmed_tap(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        real = NoteTrack(track_id=11, lane=3)
        for sequence, progress in ((97, 0.28), (98, 0.30), (99, 0.32)):
            note = candidate_at(calibration(), 3, progress)
            real.observations.append(TrackObservation(sequence, 99.0 + sequence * 0.03, note.center, progress, note))
        fragment = NoteTrack(track_id=10, lane=3)
        for sequence, progress in ((98, 0.63), (99, 0.64), (100, 0.6535)):
            note = candidate_at(calibration(), 3, progress)
            fragment.observations.append(TrackObservation(sequence, 99.0 + sequence * 0.03, note.center, progress, note))
        engine.tracks[10] = fragment
        engine.tracks[11] = real
        engine.last_frame_sequence = 110
        pending = [
            MusicActionEvent("track-10@200000", 10, 3, NoteGesture.TAP, 200.0, (640, 620)),
            MusicActionEvent("track-11@100500", 11, 3, NoteGesture.TAP, 100.5, (640, 620)),
        ]
        batches = due_tap_batches(pending, 101.0, engine)
        flat = [event.event_id for batch in batches for event in batch]
        self.assertIn("track-11@100500", flat)

    def test_hold_committed_prefers_single_bonus_evidence(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        engine = MusicVisionEngine(calibration(), config)
        normal = NoteTrack(track_id=1, lane=0)
        normal.hold_evidence_frames = 1
        self.assertFalse(engine._hold_committed(normal))
        normal.hold_evidence_frames = 2
        self.assertTrue(engine._hold_committed(normal))
        bonus = NoteTrack(track_id=2, lane=0)
        bonus.bonus_star = True
        bonus.hold_evidence_frames = 1
        self.assertTrue(engine._hold_committed(bonus))


if __name__ == "__main__":
    unittest.main()
