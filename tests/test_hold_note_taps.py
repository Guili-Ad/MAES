from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BRANCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH_ROOT))

from agent.music.executor import MusicActionExecutor
from agent.music.holds import HoldTailDetection
from agent.music.models import (
    LaneInputState,
    MusicActionEvent,
    MusicCalibrationData,
    MusicConfig,
    MusicFrame,
    NoteGesture,
    NoteTrack,
    TrackState,
)
from agent.music.runtime import MusicRuntime, RuntimeMetrics
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


def detection(progress: float, lane: int, exits: int = 2, topology: str | None = None) -> HoldTailDetection:
    topology = topology or ('terminal' if exits == 1 else 'checkpoint')
    return HoldTailDetection(progress, 0.8, 100, lane,
        (float(calibration().points[lane][0]), 140.+480.*progress), 0.0, exits, topology, None,
        physical_ring=True)


def make_frame(moment: float, sequence: int) -> MusicFrame:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    return MusicFrame(sequence, moment, moment, moment, image)


def tap_engine(**kwargs) -> tuple[MusicVisionEngine, NoteTrack]:
    config = MusicConfig(
        hold_notes_as_taps=True,
        hold_sustain_enabled=False,
        enable_holds=True,
        lane_count=7,
        **kwargs,
    )
    engine = MusicVisionEngine(calibration(), config)
    track = NoteTrack(track_id=5, lane=2)
    track.gesture = NoteGesture.HOLD_START
    track.state = TrackState.TAP_PENDING
    track.point_mode = True
    track.visual_family = 'yellow_head'
    track.timing_profile = 'yellow_head'
    track.predicted_hit_time = 0.0
    engine.tracks[track.track_id] = track
    register_head(engine, track)
    return engine, track


def register_head(engine, track, now=0.):
    """Fixtures must acknowledge actual head input, not inject HOLDING."""
    track.point_mode = True
    track.visual_family = 'yellow_head'
    track.timing_profile = 'yellow_head'
    event = MusicActionEvent(f'fixture-head-{track.track_id}', track.track_id, track.lane,
                            NoteGesture.TAP, now, tuple(engine.calibration.points[track.lane]),
                            visual_family='yellow_head', timing_profile='yellow_head')
    engine.tap_hold_chain.acknowledge_head(event, SimpleNamespace(
        down_call_started=now, down_call_finished=now+.001,
        up_call_finished=now+.002, error=''))


def linked_owner(engine, owner):
    other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START,
                      state=TrackState.TAP_PENDING, predicted_hit_time=0.)
    owner.linked_partner_id, other.linked_partner_id = 6, owner.track_id
    engine.tracks[6] = other
    register_head(engine, other)
    return other


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


class HoldNoteTapTests(unittest.TestCase):
    def test_marker_emits_standard_tap_within_horizon(self) -> None:
        engine, track = tap_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.0)
        self.assertEqual(engine.release_events(1.0), [])
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.7)
        events = engine.release_events(1.7)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.gesture, NoteGesture.TAP)
        self.assertEqual(event.lane, 3)
        self.assertEqual(event.track_id, -7)
        self.assertAlmostEqual(event.deadline, 2.0 - engine.config.tap_action_advance_ms / 1000.0)
        self.assertAlmostEqual(event.tap_reference_hit_time, 2.0)
        self.assertIsNone(event.tap_group_id)
        self.assertEqual(event.visual_family, 'gold_ring')
        self.assertIsNotNone(event.physical_id)

    def test_simultaneous_markers_form_one_chord_group(self) -> None:
        engine, track = tap_engine()
        other = linked_owner(engine, track)
        add_marker(engine, 7, track.track_id, 2, hit=2.0, now=1.8)
        add_marker(engine, 8, other.track_id, 4, hit=2.01, now=1.8)
        events = engine.release_events(1.8)
        self.assertEqual(len(events), 2)
        groups = {event.tap_group_id for event in events}
        self.assertEqual(len(groups), 1)
        self.assertIsNotNone(next(iter(groups)))
        self.assertEqual({event.lane for event in events}, {2, 4})
        self.assertTrue(all(event.gesture == NoteGesture.TAP for event in events))

    def test_same_lane_duplicate_marker_is_deduped(self) -> None:
        engine, track = tap_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.8)
        add_marker(engine, 8, track.track_id, 3, hit=2.05, now=1.8)
        # This case is an actual duplicate contour, not a nearby real note.
        engine.sustain_tracker.markers[8].observations = type(engine.sustain_tracker.markers[7].observations)(
            engine.sustain_tracker.markers[7].observations, maxlen=12)
        events = engine.release_events(1.8)
        self.assertEqual(len(events), 1)
        from agent.music.point_events import point_registry
        self.assertIs(point_registry(engine).state_for('gold', 7),
                      point_registry(engine).state_for('gold', 8))

    def test_marker_never_emits_sustain_events(self) -> None:
        engine, track = tap_engine()
        add_marker(engine, 7, track.track_id, 3, hit=2.0, now=1.8)
        events = engine.release_events(1.8)
        self.assertFalse(any(
            event.gesture.startswith("Sustain") if isinstance(event.gesture, str) else False
            for event in events
        ))
        self.assertTrue(all(event.gesture == NoteGesture.TAP for event in events))

    def test_old_release_time_does_not_finalize_tap_anchor(self) -> None:
        engine, track = tap_engine()
        track.hold_release_time = 1.0
        events = engine.release_events(1.0)
        self.assertEqual(events, [])
        self.assertFalse(track.hold_sustain_final_emitted)
        self.assertEqual(track.state, TrackState.TAP_PENDING)

    def test_flick_binding_disabled_in_tap_mode(self) -> None:
        engine, track = tap_engine()
        from agent.music.models import MusicCandidate, TrackObservation

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

    def tap_many(self, requests, *, event_ids=None):
        self.calls.append(("tap_many", list(requests)))
        return []

    def touch_move(self, lane, x, y, *, track_id=None):
        self.calls.append(("move", lane, x, y, track_id))

    def release_track(self, track_id):
        pass


class HeadTapRuntimeTests(unittest.TestCase):
    def test_head_event_uses_tap_path_and_keeps_anchor(self) -> None:
        config = MusicConfig(
            hold_notes_as_taps=True,
            hold_sustain_enabled=False,
            enable_holds=True,
            lane_count=7,
        )
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.0)
        engine = MusicVisionEngine(calibration(), config)
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


class FakeContext:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def run_action_direct(self, kind, param):
        self.actions.append(kind.value)
        return SimpleNamespace(success=True)


class StaleContactWatchdogTests(unittest.TestCase):
    def test_stale_contact_is_released_in_tap_mode(self) -> None:
        config = MusicConfig(
            hold_notes_as_taps=True,
            hold_sustain_enabled=False,
            enable_holds=True,
            lane_count=7,
        )
        context = FakeContext()
        executor = MusicActionExecutor(context, 1280, 720, config, advanced=True)
        executor.lanes[3] = LaneInputState(lane=3, contact=0, hold_track_id=5, contact_started=100.0)
        executor.enforce_contact_limits(now=101.5)
        self.assertNotIn(3, executor.active_contacts)
        self.assertIn("TouchUp", context.actions)

    def test_legacy_mode_keeps_long_contacts(self) -> None:
        config = MusicConfig(
            hold_notes_as_taps=False,
            hold_sustain_enabled=False,
            enable_holds=True,
            lane_count=7,
        )
        context = FakeContext()
        executor = MusicActionExecutor(context, 1280, 720, config, advanced=True)
        executor.lanes[3] = LaneInputState(lane=3, contact=0, hold_track_id=5, contact_started=100.0)
        executor.enforce_contact_limits(now=101.5)
        self.assertIn(3, executor.active_contacts)
        self.assertEqual(context.actions, [])


if __name__ == "__main__":
    unittest.main()
