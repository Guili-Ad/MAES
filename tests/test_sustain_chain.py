from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BRANCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH_ROOT))

from agent.music.holds import HoldTailDetection
from agent.music.models import (
    MusicActionEvent,
    MusicCalibrationData,
    MusicConfig,
    MusicFrame,
    NoteGesture,
    NoteTrack,
    TrackState,
)
from agent.music.runtime import MusicRuntime, RuntimeMetrics
from agent.music.sustain import SustainMarkerTracker
from agent.music.tracking import MusicVisionEngine


def calibration() -> MusicCalibrationData:
    points = [[160 + lane * 160, 620] for lane in range(7)]
    return MusicCalibrationData(
        version=4,
        lane_count=7,
        width=1280,
        height=720,
        points=points,
        lane_centerlines=[
            [[float(x), 140.0], [float(x), 380.0], [float(x), float(y)]]
            for x, y in points
        ],
        corridor_widths=[52.0] * 7,
        trigger_progress=1.0,
        candidate_roi=[0, 100, 1280, 590],
        exclusion_rois=[],
        baseline_version="maes-music-v4-2026-08",
        action_advance_ms=125.0,
        color_lower=[[0, 45, 110]],
        color_upper=[[179, 255, 255]],
        candidate_min_pixels=12,
        hold_min_length=100.0,
        created_at="",
    )


def detection(progress: float, lane: int, exits: int = 2, topology: str = "checkpoint") -> HoldTailDetection:
    return HoldTailDetection(progress, 0.8, 100, lane, (640.0, 300.0), 0.0, exits, topology, None)


def make_frame(moment: float, sequence: int) -> MusicFrame:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    return MusicFrame(sequence, moment, moment, moment, image)


def sustain_engine(**kwargs) -> tuple[MusicVisionEngine, NoteTrack]:
    config = MusicConfig(
        hold_sustain_enabled=True,
        lane_count=7,
        enable_holds=True,
        **kwargs,
    )
    engine = MusicVisionEngine(calibration(), config)
    track = NoteTrack(track_id=5, lane=2)
    track.gesture = NoteGesture.HOLD_START
    track.state = TrackState.HOLDING
    track.predicted_hit_time = 0.0
    engine.tracks[track.track_id] = track
    return engine, track


def add_marker(
    engine: MusicVisionEngine,
    marker_id: int,
    owner: int,
    lane: int,
    hit: float,
    now: float,
    *,
    samples: int = 3,
    exits: int = 2,
    speed: float = 0.5,
) -> None:
    for index in range(samples):
        moment = now - (samples - 1 - index) * 0.1
        progress = 1.0 - max(0.0, hit - moment) * speed
        engine.sustain_tracker.observe(
            marker_id,
            detection(progress, lane, exits=exits),
            make_frame(moment, index + 1),
            owner=owner,
        )


class SustainTrackerTests(unittest.TestCase):
    def test_marker_requires_motion_and_minimum_samples(self) -> None:
        tracker = SustainMarkerTracker(trigger_progress=1.0)
        config = MusicConfig(lane_count=7, enable_holds=True)
        tracker.observe(1, detection(0.30, 3), make_frame(0.0, 1))
        tracker.observe(1, detection(0.30, 3), make_frame(0.1, 2))
        self.assertFalse(tracker.markers[1].stable(config, 0.1))
        tracker.observe(1, detection(0.35, 3), make_frame(0.2, 3))
        self.assertTrue(tracker.markers[1].stable(config, 0.2))

    def test_marker_lane_consensus_and_terminal_votes(self) -> None:
        tracker = SustainMarkerTracker(trigger_progress=1.0)
        for index, progress in enumerate((0.30, 0.35, 0.40)):
            tracker.observe(9, detection(progress, 3, exits=2), make_frame(index * 0.1, index + 1))
        marker = tracker.markers[9]
        self.assertEqual(marker.lane(), 3)
        self.assertFalse(marker.is_terminal())
        self.assertAlmostEqual(marker.predicted_hit(), 1.4)

    def test_marker_prune_removes_stale_markers(self) -> None:
        tracker = SustainMarkerTracker(trigger_progress=1.0)
        tracker.observe(4, detection(0.30, 3), make_frame(0.0, 1))
        tracker.prune(5.0)
        self.assertNotIn(4, tracker.markers)

    def test_tracker_heals_broken_association(self) -> None:
        tracker = SustainMarkerTracker(trigger_progress=1.0)
        tracker.observe(3, detection(0.30, 3), make_frame(0.0, 1), owner=5)
        tracker.observe(3, detection(0.35, 3), make_frame(0.1, 2), owner=5)
        healed = tracker.match(detection(0.42, 3), make_frame(0.2, 3), 5)
        self.assertIsNotNone(healed)
        self.assertEqual(healed.marker_id, 3)
        self.assertIsNone(tracker.match(detection(0.42, 5), make_frame(0.2, 3), 5))


class SustainPlannerTests(unittest.TestCase):
    def test_single_marker_schedules_press_and_release(self) -> None:
        engine, track = sustain_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.0)
        events = engine.release_events(1.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        releases = [event for event in events if event.gesture == NoteGesture.SUSTAIN_RELEASE]
        self.assertEqual(len(presses), 1)
        self.assertEqual(presses[0].lane, 3)
        self.assertAlmostEqual(
            presses[0].deadline,
            2.0 - engine.config.hold_sustain_press_advance_ms / 1000.0,
        )
        self.assertEqual(len(releases), 1)
        self.assertAlmostEqual(
            releases[0].deadline,
            2.0 + engine.config.hold_sustain_press_hold_ms / 1000.0,
        )
        self.assertEqual(releases[0].contact_policy, "sustain")
        self.assertIn(7, track.hold_sustain_planned_ids)

    def test_close_markers_merge_into_one_window(self) -> None:
        engine, track = sustain_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.0)
        add_marker(engine, 8, track.track_id, 3, hit=2.2, now=1.0)
        events = engine.release_events(1.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        releases = [event for event in events if event.gesture == NoteGesture.SUSTAIN_RELEASE]
        moves = [event for event in events if event.gesture == NoteGesture.SUSTAIN_MOVE]
        self.assertEqual(len(presses), 1)
        self.assertEqual(len(moves), 0)
        self.assertEqual(len(releases), 1)
        self.assertAlmostEqual(
            releases[0].deadline,
            2.2 + engine.config.hold_sustain_press_hold_ms / 1000.0,
        )
        self.assertEqual(track.hold_sustain_planned_ids, {7, 8})

    def test_merged_lane_change_emits_move(self) -> None:
        engine, track = sustain_engine()
        add_marker(engine, 7, track.track_id, 2, hit=2.0, now=1.0)
        add_marker(engine, 8, track.track_id, 4, hit=2.2, now=1.0)
        events = engine.release_events(1.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        moves = [event for event in events if event.gesture == NoteGesture.SUSTAIN_MOVE]
        self.assertEqual(presses[0].lane, 2)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].lane, 4)
        self.assertAlmostEqual(
            moves[0].deadline,
            2.2 - engine.config.hold_sustain_move_advance_ms / 1000.0,
        )

    def test_distant_markers_start_separate_windows(self) -> None:
        engine, track = sustain_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.0)
        add_marker(engine, 8, track.track_id, 3, hit=3.0, now=1.0)
        events = engine.release_events(1.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        self.assertEqual(len(presses), 1)
        self.assertIn(7, track.hold_sustain_planned_ids)
        self.assertNotIn(8, track.hold_sustain_planned_ids)

    def test_terminal_marker_does_not_stop_planning(self) -> None:
        engine, track = sustain_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.0, exits=1)
        events = engine.release_events(1.0)
        releases = [event for event in events if event.gesture == NoteGesture.SUSTAIN_RELEASE]
        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0].contact_policy, "sustain")
        self.assertFalse(track.hold_sustain_final_emitted)
        add_marker(engine, 8, track.track_id, 3, hit=4.0, now=3.0)
        events = engine.release_events(3.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        self.assertEqual(len(presses), 1)
        self.assertEqual(presses[0].lane, 3)

    def test_locked_cap_adds_tail_safety_window(self) -> None:
        engine, track = sustain_engine()
        track.hold_release_locked = True
        track.hold_release_time = 2.5
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.0)
        events = engine.release_events(1.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        self.assertEqual(len(presses), 1)
        self.assertIn(7, track.hold_sustain_planned_ids)
        self.assertNotIn("tail", track.hold_sustain_planned_ids)
        events = engine.release_events(2.2)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        releases = [event for event in events if event.gesture == NoteGesture.SUSTAIN_RELEASE]
        self.assertEqual(len(presses), 1)
        self.assertIn("tail", track.hold_sustain_planned_ids)
        self.assertAlmostEqual(
            releases[-1].deadline,
            2.5 + engine.config.hold_sustain_press_hold_ms / 1000.0
            + engine.config.hold_sustain_release_delay_ms / 1000.0,
        )

    def test_flick_tail_presses_lane_and_emits_held_flick(self) -> None:
        engine, track = sustain_engine()
        flick = NoteTrack(track_id=9, lane=4)
        flick.flick = True
        flick.flick_direction = NoteGesture.FLICK_UP
        flick.gesture = NoteGesture.FLICK_UP
        engine.tracks[flick.track_id] = flick
        track.hold_end_flick_track = flick.track_id
        track.hold_end_flick_direction = NoteGesture.FLICK_UP
        track.hold_end_flick_arrival = 3.0
        events = engine.release_events(1.0)
        presses = [event for event in events if event.gesture == NoteGesture.SUSTAIN_PRESS]
        flicks = [event for event in events if event.gesture == NoteGesture.FLICK_UP]
        self.assertEqual(len(presses), 1)
        self.assertEqual(presses[0].lane, 4)
        self.assertAlmostEqual(presses[0].deadline, 3.0 - engine.config.hold_sustain_move_advance_ms / 1000.0)
        self.assertEqual(len(flicks), 1)
        self.assertEqual(flicks[0].contact_policy, "held_flick")
        self.assertEqual(flicks[0].lane, 4)
        self.assertTrue(track.hold_sustain_final_emitted)

    def test_finalize_fallback_releases_track(self) -> None:
        engine, track = sustain_engine()
        track.hold_release_time = 1.0
        events = engine.release_events(1.0)
        releases = [event for event in events if event.gesture == NoteGesture.SUSTAIN_RELEASE]
        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0].contact_policy, "sustain_final")
        self.assertTrue(track.hold_sustain_final_emitted)

    def test_end_flick_binds_on_route_lane(self) -> None:
        engine, track = sustain_engine()
        track.hold_route_lanes = [2, 3]
        track.hold_target_lane = 3
        track.hold_release_time = 2.5
        flick = NoteTrack(track_id=9, lane=3)
        flick.flick = True
        flick.flick_direction = NoteGesture.FLICK_UP
        flick.predicted_hit_time = 2.0
        flick.speed = 0.2
        from agent.music.models import MusicCandidate, TrackObservation

        candidate = MusicCandidate((0, 0, 30, 30), 900, 1.0, (640.0, 500.0))
        flick.observations.append(TrackObservation(1, 1.0, candidate.center, 0.72, candidate))
        engine.tracks[flick.track_id] = flick
        engine._bind_hold_end_flicks(make_frame(1.0, 1))
        self.assertEqual(track.hold_end_flick_track, 9)

    def test_end_flick_binding_requires_release_and_motion(self) -> None:
        from agent.music.models import MusicCandidate, TrackObservation

        def build(flick_speed: float):
            engine, track = sustain_engine()
            flick = NoteTrack(track_id=9, lane=2)
            flick.flick = True
            flick.flick_direction = NoteGesture.FLICK_UP
            flick.predicted_hit_time = 2.0
            flick.speed = flick_speed
            candidate = MusicCandidate((0, 0, 30, 30), 900, 1.0, (640.0, 500.0))
            flick.observations.append(TrackObservation(1, 1.0, candidate.center, 0.72, candidate))
            engine.tracks[flick.track_id] = flick
            return engine, track, flick

        engine, track, flick = build(0.2)
        engine._bind_hold_end_flicks(make_frame(1.0, 1))
        self.assertIsNone(track.hold_end_flick_track)

        engine, track, flick = build(0.0)
        track.hold_release_time = 2.5
        engine._bind_hold_end_flicks(make_frame(1.0, 1))
        self.assertIsNone(track.hold_end_flick_track)

        engine, track, flick = build(0.2)
        track.hold_release_time = 2.5
        engine._bind_hold_end_flicks(make_frame(1.0, 1))
        self.assertEqual(track.hold_end_flick_track, 9)

    def test_end_flick_unbinds_after_release_drift(self) -> None:
        from agent.music.models import MusicCandidate, TrackObservation

        engine, track = sustain_engine()
        flick = NoteTrack(track_id=9, lane=2)
        flick.flick = True
        flick.flick_direction = NoteGesture.FLICK_UP
        flick.predicted_hit_time = 2.0
        flick.speed = 0.2
        candidate = MusicCandidate((0, 0, 30, 30), 900, 1.0, (640.0, 500.0))
        flick.observations.append(TrackObservation(1, 1.0, candidate.center, 0.72, candidate))
        engine.tracks[flick.track_id] = flick
        track.hold_release_time = 2.5
        engine._bind_hold_end_flicks(make_frame(1.0, 1))
        self.assertEqual(track.hold_end_flick_track, 9)
        track.hold_release_time = 1.0
        engine._bind_hold_end_flicks(make_frame(1.1, 2))
        self.assertIsNone(track.hold_end_flick_track)
        self.assertIsNone(flick.hold_end_owner)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.owners: dict[int, int] = {}
        self.supports_holds = True
        self.async_input = False

    @property
    def active_contacts(self) -> dict[int, int]:
        return dict(self.owners)

    def hold_owner(self, lane: int):
        return self.owners.get(lane)

    def touch_down(self, lane, x, y, *, track_id=None):
        self.calls.append(("down", lane, x, y, track_id))
        self.owners[lane] = track_id

    def touch_move(self, lane, x, y, *, track_id=None):
        self.calls.append(("move", lane, x, y, track_id))

    def touch_up(self, lane, *, track_id=None):
        self.calls.append(("up", lane, track_id))
        self.owners.pop(lane, None)

    def release_track(self, track_id):
        for lane, owner in list(self.owners.items()):
            if owner == track_id:
                self.touch_up(lane, track_id=track_id)

    def tap_many(self, requests, *, event_ids=None):
        self.calls.append(("tap_many", list(requests)))
        return []

    def swipe(self, request, *, event_id="", track_id=None, tick=None):
        self.calls.append(("swipe", request.already_down, track_id))


class SustainRuntimeTests(unittest.TestCase):
    def _runtime(self) -> MusicRuntime:
        return MusicRuntime(
            SimpleNamespace(),
            MusicConfig(hold_sustain_enabled=True, lane_count=7, enable_holds=True),
            clock=lambda: 1.0,
        )

    def test_sustain_press_moves_owned_contact_only(self) -> None:
        runtime = self._runtime()
        engine = MusicVisionEngine(calibration(), runtime.config)
        executor = FakeExecutor()
        press = MusicActionEvent(
            "sustain-press-5-1", 5, 3, NoteGesture.SUSTAIN_PRESS, 1.0, (640, 620), contact_policy="sustain",
        )
        runtime._dispatch_due_event(press, executor, engine, RuntimeMetrics(), [], set())
        self.assertEqual(executor.calls, [("down", 3, 640, 620, 5)])

    def test_sustain_press_skips_occupied_lane(self) -> None:
        runtime = self._runtime()
        engine = MusicVisionEngine(calibration(), runtime.config)
        executor = FakeExecutor()
        executor.owners[3] = 99
        press = MusicActionEvent(
            "sustain-press-5-1", 5, 3, NoteGesture.SUSTAIN_PRESS, 1.0, (640, 620), contact_policy="sustain",
        )
        runtime._dispatch_due_event(press, executor, engine, RuntimeMetrics(), [], set())
        self.assertEqual(executor.calls, [])

    def test_sustain_final_release_finishes_track(self) -> None:
        runtime = self._runtime()
        engine = MusicVisionEngine(calibration(), runtime.config)
        track = NoteTrack(track_id=5, lane=3)
        track.gesture = NoteGesture.HOLD_START
        track.state = TrackState.HOLDING
        engine.tracks[5] = track
        executor = FakeExecutor()
        executor.owners[3] = 5
        release = MusicActionEvent(
            "sustain-release-5", 5, 3, NoteGesture.SUSTAIN_RELEASE, 1.0, (640, 620),
            contact_policy="sustain_final",
        )
        runtime._dispatch_due_event(release, executor, engine, RuntimeMetrics(), [], set())
        self.assertIn(("up", 3, 5), executor.calls)
        self.assertEqual(track.state, TrackState.RELEASED)

    def test_held_flick_without_contact_falls_back_to_standalone(self) -> None:
        runtime = self._runtime()
        engine = MusicVisionEngine(calibration(), runtime.config)
        executor = FakeExecutor()
        flick = MusicActionEvent(
            "sustain-flick-5", 5, 3, NoteGesture.FLICK_UP, 1.0, (640, 620),
            direction=NoteGesture.FLICK_UP, contact_policy="held_flick",
        )
        runtime._dispatch_due_event(flick, executor, engine, RuntimeMetrics(), [], set())
        swipes = [call for call in executor.calls if call[0] == "swipe"]
        self.assertEqual(len(swipes), 1)
        self.assertFalse(swipes[0][1])

    def test_head_tap_uses_tap_path_and_keeps_track_live(self) -> None:
        runtime = self._runtime()
        engine = MusicVisionEngine(calibration(), runtime.config)
        track = NoteTrack(track_id=1, lane=2)
        track.gesture = NoteGesture.HOLD_START
        track.state = TrackState.HOLD_PENDING
        track.action_executed = True
        engine.tracks[1] = track
        executor = FakeExecutor()
        event = MusicActionEvent(
            "track-1@1000", 1, 2, NoteGesture.HOLD_START, 0.5, (320, 620), contact_policy="persistent",
        )
        runtime._execute_due(executor, [event], 1.0, RuntimeMetrics(), engine)
        self.assertIn(("tap_many", [(2, 320, 620)]), executor.calls)
        self.assertEqual(track.state, TrackState.HOLDING)


if __name__ == "__main__":
    unittest.main()
