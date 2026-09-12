"""Four-direction flick support: colour families, hold-end binding, execution."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from test_longtap_branch import calibration, candidate_at
from agent.music.executor import MusicActionExecutor, MusicTouchError
from agent.music.models import (
    FlickRequest,
    HoldTailObservation,
    MusicActionEvent,
    MusicCandidate,
    MusicConfig,
    MusicFrame,
    NoteGesture,
    NoteTrack,
    TrackObservation,
    TrackState,
)
from agent.music.tracking import MusicVisionEngine
from agent.music.vision import (
    VisualMask,
    classify_flick_family,
    detect_center_color_note,
    tag_flick_candidates,
)

# Seed family sprites (BGR) matching the colour windows in vision.py.
BLUE_SPRITE = (230, 140, 40)
RED_SPRITE = (60, 60, 235)
VIOLET_SPRITE = (200, 50, 150)
PINK_SPRITE = (180, 60, 230)
TAP_SPRITE = (200, 230, 115)


def draw_sprite(image: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = box
    crop = image[y : y + height, x : x + width]
    radius_x = (width - 1) / 2.0
    radius_y = (height - 1) / 2.0
    yy, xx = np.ogrid[:height, :width]
    radius = ((xx - radius_x) / radius_x) ** 2 + ((yy - radius_y) / radius_y) ** 2
    crop[radius <= 1.0] = color


class FlickColorFamilyTests(unittest.TestCase):
    def test_four_colours_resolve_their_directions(self) -> None:
        cases = (
            (BLUE_SPRITE, NoteGesture.FLICK_RIGHT, "blue"),
            (RED_SPRITE, NoteGesture.FLICK_LEFT, "red"),
            (VIOLET_SPRITE, NoteGesture.FLICK_UP, "violet"),
            (PINK_SPRITE, NoteGesture.FLICK_DOWN, "pink"),
        )
        for color, direction, name in cases:
            with self.subTest(color=color):
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                box = (240, 280, 80, 60)
                draw_sprite(image, box, color)
                got_direction, got_name = classify_flick_family(image, box)
                self.assertEqual(got_direction, direction)
                self.assertEqual(got_name, name)

    def test_tap_green_is_not_a_flick(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        box = (240, 280, 80, 60)
        draw_sprite(image, box, TAP_SPRITE)
        self.assertEqual(classify_flick_family(image, box), (NoteGesture.UNKNOWN, ""))

    def test_dim_blue_is_not_a_flick(self) -> None:
        # The dimmer tap body must never enter the vivid flick family.
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        box = (240, 280, 80, 60)
        draw_sprite(image, box, (120, 90, 60))
        self.assertEqual(classify_flick_family(image, box), (NoteGesture.UNKNOWN, ""))

    def test_tag_candidates_labels_only_flick_families(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        flick_box = (240, 280, 80, 60)
        tap_box = (500, 280, 80, 60)
        draw_sprite(image, flick_box, BLUE_SPRITE)
        draw_sprite(image, tap_box, TAP_SPRITE)
        flick = MusicCandidate(flick_box, 4800, 1.0, (280, 310))
        tap = MusicCandidate(tap_box, 4800, 1.0, (540, 310))
        tagged = tag_flick_candidates(image, [flick, tap])
        self.assertEqual(tagged[0].variant, "flick")
        self.assertEqual(tagged[0].flick_direction, NoteGesture.FLICK_RIGHT)
        self.assertEqual(tagged[0].flick_color, "blue")
        self.assertEqual(tagged[1].variant, "")
        self.assertEqual(tagged[1].flick_direction, NoteGesture.UNKNOWN)

    def test_center_color_note_helper_still_works(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertEqual(detect_center_color_note(image), [])


class HoldEndFlickTests(unittest.TestCase):
    def _hold(self, release: float = 12.0, hit: float = 10.0) -> NoteTrack:
        track = NoteTrack(track_id=1, lane=3)
        track.gesture = NoteGesture.HOLD_START
        track.state = TrackState.HOLDING
        track.hold_release_time = release
        track.predicted_hit_time = hit
        return track

    def _flick(self, tid: int, lane: int, progress: float, hit: float, direction: NoteGesture) -> NoteTrack:
        track = NoteTrack(track_id=tid, lane=lane)
        track.flick = True
        track.flick_direction = direction
        track.gesture = direction
        track.predicted_hit_time = hit
        base = candidate_at(calibration(), lane, progress)
        track.observations.append(TrackObservation(1, hit - 1.0, base.center, progress, base))
        return track

    def _frame(self, moment: float) -> MusicFrame:
        return MusicFrame(1, moment, moment, moment, np.zeros((720, 1280, 3), dtype=np.uint8))

    def test_flick_track_binds_to_active_hold(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold()
        flick = self._flick(20, 3, 0.72, 10.4, NoteGesture.FLICK_UP)
        engine.tracks[hold.track_id] = hold
        engine.tracks[flick.track_id] = flick
        engine._bind_hold_end_flicks(self._frame(10.2))
        self.assertEqual(hold.hold_end_flick_track, 20)
        self.assertEqual(hold.hold_end_flick_direction, NoteGesture.FLICK_UP)
        self.assertAlmostEqual(hold.hold_end_flick_arrival, 10.4)
        self.assertEqual(flick.hold_end_owner, 1)

    def test_binding_picks_earliest_upcoming_flick(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold(release=12.0)
        late = self._flick(21, 3, 0.70, 11.2, NoteGesture.FLICK_UP)
        early = self._flick(22, 3, 0.70, 10.4, NoteGesture.FLICK_DOWN)
        engine.tracks[hold.track_id] = hold
        engine.tracks[late.track_id] = late
        engine.tracks[early.track_id] = early
        engine._bind_hold_end_flicks(self._frame(10.2))
        self.assertEqual(hold.hold_end_flick_track, 22)
        self.assertIsNone(late.hold_end_owner)

    def test_binding_rejects_flick_later_than_release_estimate(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold(release=10.6)
        flick = self._flick(23, 3, 0.70, 11.4, NoteGesture.FLICK_UP)
        engine.tracks[hold.track_id] = hold
        engine.tracks[flick.track_id] = flick
        engine._bind_hold_end_flicks(self._frame(10.2))
        self.assertIsNone(hold.hold_end_flick_track)
        self.assertIsNone(flick.hold_end_owner)

    def test_binding_ignores_other_lane_and_existing_owner(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold()
        other_lane = self._flick(24, 5, 0.72, 10.4, NoteGesture.FLICK_UP)
        owned = self._flick(25, 3, 0.72, 10.3, NoteGesture.FLICK_DOWN)
        owned.hold_end_owner = 99
        engine.tracks[hold.track_id] = hold
        engine.tracks[other_lane.track_id] = other_lane
        engine.tracks[owned.track_id] = owned
        engine._bind_hold_end_flicks(self._frame(10.2))
        self.assertIsNone(hold.hold_end_flick_track)

    def test_binding_skips_already_scheduled_flick(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold()
        flick = self._flick(28, 3, 0.72, 10.4, NoteGesture.FLICK_UP)
        flick.action_event_id = "track-28@10400"
        engine.tracks[hold.track_id] = hold
        engine.tracks[flick.track_id] = flick
        engine._bind_hold_end_flicks(self._frame(10.2))
        self.assertIsNone(hold.hold_end_flick_track)
        self.assertIsNone(flick.hold_end_owner)

    def test_binding_refreshes_arrival_from_bound_track(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold()
        flick = self._flick(26, 3, 0.72, 10.4, NoteGesture.FLICK_UP)
        engine.tracks[hold.track_id] = hold
        engine.tracks[flick.track_id] = flick
        engine._bind_hold_end_flicks(self._frame(10.2))
        flick.predicted_hit_time = 10.55
        engine._bind_hold_end_flicks(self._frame(10.3))
        self.assertAlmostEqual(hold.hold_end_flick_arrival, 10.55)
        self.assertAlmostEqual(engine._hold_release_deadline(hold, hold.hold_release_time), 10.55)

    def test_retire_releases_bound_track(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        hold = self._hold()
        flick = self._flick(27, 3, 0.72, 10.4, NoteGesture.FLICK_UP)
        engine.tracks[hold.track_id] = hold
        engine.tracks[flick.track_id] = flick
        engine._bind_hold_end_flicks(self._frame(10.2))
        engine._retire_hold_end_flick(hold)
        self.assertEqual(flick.state, TrackState.RELEASED)

    def test_release_direction_prefers_bound_direction(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        track = NoteTrack(track_id=8, lane=2)
        track.hold_end_flick_direction = NoteGesture.FLICK_DOWN
        self.assertEqual(engine._hold_release_direction(track, 10.0), NoteGesture.FLICK_DOWN)

    def test_release_direction_unknown_without_binding(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        track = NoteTrack(track_id=9, lane=2)
        self.assertEqual(engine._hold_release_direction(track, 10.0), NoteGesture.UNKNOWN)


class FlickReleaseRefineTests(unittest.TestCase):
    def test_refine_pending_upgrades_hold_release_to_held_flick(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=7, lane=3)
        track.gesture = NoteGesture.HOLD_START
        track.state = TrackState.HOLDING
        track.hold_release_time = 10.0
        track.predicted_hit_time = 9.5
        track.hold_end_flick_direction = NoteGesture.FLICK_UP
        track.hold_end_flick_arrival = 9.8
        engine.tracks[7] = track
        release = MusicActionEvent(
            "release-7", 7, 3, NoteGesture.HOLD_END, 10.0, (640, 620), contact_policy="persistent",
        )
        refined = engine.refine_pending([release], 9.5)
        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].gesture, NoteGesture.FLICK_UP)
        self.assertEqual(refined[0].direction, NoteGesture.FLICK_UP)
        self.assertEqual(refined[0].contact_policy, "held_flick")
        self.assertAlmostEqual(refined[0].deadline, 9.8)
        again = engine.refine_pending(refined, 9.7)
        self.assertEqual(again[0].gesture, NoteGesture.FLICK_UP)
        self.assertEqual(again[0].contact_policy, "held_flick")
        self.assertAlmostEqual(again[0].deadline, 9.8)


class FlickTrackingTests(unittest.TestCase):
    def test_flick_candidate_direction_reaches_track_gesture(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), hsv=None, roi_origin=(0, 0))
        for sequence, progress in ((1, 0.30), (2, 0.33), (3, 0.36)):
            base = candidate_at(calibration(), 3, progress)
            note = MusicCandidate(
                base.box,
                base.pixel_count,
                base.fill_ratio,
                base.center,
                variant="flick",
                flick_direction=NoteGesture.FLICK_UP,
            )
            engine.update(
                MusicFrame(sequence, 1.0 + sequence * 0.03, 1.0 + sequence * 0.03, 1.0 + sequence * 0.03, image),
                [note],
                visual,
            )
        gestures = {track.gesture for track in engine.tracks.values()}
        self.assertIn(NoteGesture.FLICK_UP, gestures)

    def test_ordinary_candidate_never_gets_flick_gesture(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), hsv=None, roi_origin=(0, 0))
        for sequence, progress in ((1, 0.30), (2, 0.33), (3, 0.36), (4, 0.40)):
            note = candidate_at(calibration(), 3, progress)
            engine.update(
                MusicFrame(sequence, 1.0 + sequence * 0.03, 1.0 + sequence * 0.03, 1.0 + sequence * 0.03, image),
                [note],
                visual,
            )
        gestures = {track.gesture for track in engine.tracks.values()}
        self.assertNotIn(NoteGesture.FLICK_LEFT, gestures)
        self.assertNotIn(NoteGesture.FLICK_UP, gestures)

    def test_stationary_flick_candidate_never_becomes_flick(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), hsv=None, roi_origin=(0, 0))
        for sequence in (1, 2, 3, 4):
            base = candidate_at(calibration(), 3, 0.33)
            note = MusicCandidate(
                base.box,
                base.pixel_count,
                base.fill_ratio,
                base.center,
                variant="flick",
                flick_direction=NoteGesture.FLICK_UP,
            )
            engine.update(
                MusicFrame(sequence, 1.0 + sequence * 0.03, 1.0 + sequence * 0.03, 1.0 + sequence * 0.03, image),
                [note],
                visual,
            )
        gestures = {track.gesture for track in engine.tracks.values()}
        self.assertNotIn(NoteGesture.FLICK_UP, gestures)


class FlickExecutorTests(unittest.TestCase):
    def _executor(self, sleeps: list[float]) -> MusicActionExecutor:
        return MusicActionExecutor(
            SimpleNamespace(),
            1280,
            720,
            MusicConfig(lane_count=7, flick_distance_px=56, flick_duration_ms=60, flick_steps=3, flick_end_hold_ms=16.0),
            sleeper=sleeps.append,
        )

    def test_flick_targets_cover_four_screen_directions(self) -> None:
        executor = self._executor([])
        self.assertEqual(executor._flick_target(640, 400, NoteGesture.FLICK_LEFT), (584, 400))
        self.assertEqual(executor._flick_target(640, 400, NoteGesture.FLICK_RIGHT), (696, 400))
        self.assertEqual(executor._flick_target(640, 400, NoteGesture.FLICK_UP), (640, 344))
        self.assertEqual(executor._flick_target(640, 400, NoteGesture.FLICK_DOWN), (640, 456))
        self.assertEqual(executor._flick_target(10, 400, NoteGesture.FLICK_LEFT), (0, 400))
        self.assertEqual(executor._flick_target(640, 10, NoteGesture.FLICK_UP), (640, 0))

    def test_swipe_moves_in_steps_with_real_duration(self) -> None:
        sleeps: list[float] = []
        executor = self._executor(sleeps)
        with patch.object(executor, "_run") as run:
            executor.swipe(FlickRequest(5, 1030, 516, NoteGesture.FLICK_RIGHT), event_id="flick-1")
        descriptions = [call.args[2] for call in run.call_args_list]
        self.assertIn("down", descriptions[0])
        self.assertIn("up", descriptions[-1])
        moves = [call.args[1] for call in run.call_args_list if "move" in call.args[2]]
        self.assertEqual(len(moves), 3)
        xs = [touch.target[0] for touch in moves]
        self.assertEqual(xs, sorted(xs))
        self.assertEqual(xs[-1], 1086)
        self.assertGreaterEqual(sum(sleeps), 0.076)

    def test_swipe_ticks_between_steps_for_due_input(self) -> None:
        sleeps: list[float] = []
        ticks: list[int] = []
        executor = self._executor(sleeps)
        with patch.object(executor, "_run"):
            executor.swipe(
                FlickRequest(5, 1030, 516, NoteGesture.FLICK_RIGHT),
                event_id="flick-2",
                tick=lambda: ticks.append(1),
            )
        self.assertEqual(len(ticks), 4)

    def test_hold_flick_interpolates_existing_contact_then_releases(self) -> None:
        sleeps: list[float] = []
        executor = self._executor(sleeps)
        with patch.object(executor, "touch_move") as move, patch.object(executor, "touch_up") as up:
            executor.hold_flick(5, 1030, 516, NoteGesture.FLICK_LEFT, track_id=9)
        xs = [call.args[1] for call in move.call_args_list]
        self.assertEqual(len(xs), 3)
        self.assertEqual(xs, sorted(xs, reverse=True))
        self.assertEqual(xs[-1], 974)
        up.assert_called_once_with(5, track_id=9)
        self.assertGreaterEqual(sum(sleeps), 0.076)


class _FakeJob:
    def __init__(self, *, done: bool = True, failed: bool = False, job_id: int = 1) -> None:
        self.job_id = job_id
        self.done = done
        self.failed = failed


class _FakeController:
    def __init__(self, job_id: int = 1) -> None:
        self.calls: list[tuple] = []
        self.job_id = job_id

    def _job(self) -> _FakeJob:
        return _FakeJob(job_id=self.job_id)

    def post_touch_down(self, x, y, contact=0, pressure=1):
        self.calls.append(("down", x, y, contact))
        return self._job()

    def post_touch_up(self, contact=0):
        self.calls.append(("up", contact))
        return self._job()

    def post_touch_move(self, x, y, contact=0, pressure=1):
        self.calls.append(("move", x, y, contact))
        return self._job()

    def post_swipe(self, x1, y1, x2, y2, duration, contact=0, pressure=1):
        self.calls.append(("swipe", x1, y1, x2, y2, duration, contact))
        return self._job()


class _FakeTasker:
    def __init__(self, controller: _FakeController) -> None:
        self.controller = controller
        self.actions: list[tuple] = []

    def post_action(self, action_type, action_param):
        self.actions.append((action_type, action_param))
        return _FakeJob()


class FlickAsyncInputTests(unittest.TestCase):
    def _executor(self, *, async_input: bool = True, async_flicks: bool = True):
        controller = _FakeController()
        tasker = _FakeTasker(controller)
        executor = MusicActionExecutor(
            SimpleNamespace(tasker=tasker),
            1280,
            720,
            MusicConfig(lane_count=7, flick_distance_px=56, flick_duration_ms=60),
            advanced=True,
            multi_touch=True,
            async_input=async_input,
            async_flicks=async_flicks,
            sleeper=lambda _d: None,
        )
        return executor, controller, tasker

    def test_async_tap_posts_without_blocking_and_releases_on_poll(self) -> None:
        executor, controller, _tasker = self._executor()
        receipts = executor.tap_many([(5, 1030, 516)], event_ids=["tap-1"])
        self.assertEqual(len(receipts), 1)
        self.assertIsNone(receipts[0].up_call_finished)
        self.assertEqual([call[0] for call in controller.calls], ["down", "up"])
        self.assertIn(receipts[0].contact, executor._temporary_contacts)
        completed = executor.poll_inputs(10.0)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].event_id, "tap-1")
        self.assertIsNotNone(completed[0].down_call_finished)
        self.assertIsNotNone(completed[0].up_call_finished)
        self.assertEqual(executor._temporary_contacts, set())

    def test_async_chord_posts_all_downs_before_ups(self) -> None:
        executor, controller, _tasker = self._executor()
        executor.tap_many([(1, 320, 620), (5, 960, 620)], event_ids=["a", "b"])
        kinds = [call[0] for call in controller.calls]
        self.assertEqual(kinds, ["down", "down", "up", "up"])
        executor.poll_inputs(10.0)
        self.assertEqual(executor.active_contacts, {})

    def test_async_standalone_flick_posts_native_swipe(self) -> None:
        executor, controller, _tasker = self._executor()
        executor.swipe(FlickRequest(5, 1030, 516, NoteGesture.FLICK_RIGHT), event_id="flick-1")
        swipes = [call for call in controller.calls if call[0] == "swipe"]
        self.assertEqual(len(swipes), 1)
        self.assertEqual(swipes[0][2], 516)
        self.assertGreater(swipes[0][5], 0)
        self.assertEqual(executor.poll_inputs(10.0), [])
        self.assertEqual(executor._temporary_contacts, set())

    def test_async_held_flick_hovers_existing_contact_then_releases(self) -> None:
        executor, controller, tasker = self._executor()
        state = executor._lane_state(5)
        state.contact = 3
        state.hold_track_id = 11
        executor.swipe(FlickRequest(5, 1030, 516, NoteGesture.FLICK_LEFT, already_down=True), event_id="held-1")
        self.assertEqual(len(tasker.actions), 1)
        action_type, param = tasker.actions[0]
        self.assertEqual(action_type.value, "Swipe")
        self.assertTrue(param.only_hover)
        self.assertEqual(param.contact, 3)
        ups = [call for call in controller.calls if call[0] == "up" and call[1] == 3]
        self.assertGreaterEqual(len(ups), 1)
        executor.poll_inputs(10.0)
        self.assertNotIn(5, executor.active_contacts)

    def test_invalid_tap_job_is_rejected_before_any_poll(self) -> None:
        executor, controller, _tasker = self._executor()
        controller.job_id = -1
        with self.assertRaises(MusicTouchError):
            executor.tap_many([(5, 1030, 516)], event_ids=["tap-bad"])
        self.assertFalse(executor.healthy)
        self.assertEqual(executor._pending_inputs, [])
        self.assertEqual(executor.poll_inputs(10.0), [])

    def test_invalid_flick_job_is_rejected_before_any_poll(self) -> None:
        executor, controller, _tasker = self._executor()
        controller.job_id = -1
        with self.assertRaises(MusicTouchError):
            executor.swipe(FlickRequest(5, 1030, 516, NoteGesture.FLICK_RIGHT), event_id="flick-bad")
        self.assertFalse(executor.healthy)
        self.assertEqual(executor._pending_inputs, [])
        self.assertEqual(executor.poll_inputs(10.0), [])

    def test_swipe_falls_back_to_sync_when_async_flicks_disabled(self) -> None:
        executor, controller, _tasker = self._executor(async_flicks=False)
        with patch.object(executor, "_run"):
            executor.swipe(FlickRequest(5, 1030, 516, NoteGesture.FLICK_RIGHT), event_id="flick-sync")
        self.assertEqual([call for call in controller.calls if call[0] == "swipe"], [])
        self.assertEqual(executor._pending_inputs, [])


class HoldRouteHopTests(unittest.TestCase):
    def _track(self, **overrides) -> NoteTrack:
        track = NoteTrack(track_id=5, lane=4, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 10.0
        track.hold_release_time = 13.0
        track.hold_target_lane = 3
        track.hold_route_lanes = [4, 3]
        for key, value in overrides.items():
            setattr(track, key, value)
        return track

    def test_second_hop_requires_fold_confirmation(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        track = self._track()
        engine._register_hold_target(track, 2)
        self.assertEqual(track.hold_route_lanes, [4, 3])
        track.hold_fold_route_confirmed = True
        engine._register_hold_target(track, 2)
        self.assertEqual(track.hold_target_lane, 2)
        self.assertEqual(track.hold_route_lanes, [4, 3, 2])
        self.assertEqual(track.hold_segment_index, 1)
        self.assertEqual(track.hold_route_steps_completed, 0)
        self.assertFalse(track.hold_fold_route_confirmed)

    def test_hop_limit_and_return_hop(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True, hold_max_route_hops=2))
        track = self._track(hold_fold_route_confirmed=True)
        engine._register_hold_target(track, 2)
        self.assertEqual(track.hold_route_lanes, [4, 3, 2])
        track.hold_fold_route_confirmed = True
        engine._register_hold_target(track, 0)
        self.assertEqual(track.hold_route_lanes, [4, 3, 2])
        engine2 = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True, hold_max_route_hops=3))
        track2 = self._track(hold_fold_route_confirmed=True)
        engine2._register_hold_target(track2, 4)
        self.assertEqual(track2.hold_route_lanes, [4, 3, 4])

    def test_route_steps_use_segment_source_lane(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        cal = calibration()
        track = NoteTrack(track_id=6, lane=4, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 10.0
        track.hold_release_time = 12.0
        track.hold_target_lane = 2
        track.hold_route_lanes = [4, 3, 2]
        track.hold_segment_index = 1
        track.hold_tail_observations.append(HoldTailObservation(1, 10.5, 0.9, 0.8, 100, 3, (700.0, 300.0)))
        engine.tracks[track.track_id] = track
        events = engine.release_events(10.5)
        routes = [event for event in events if event.event_id.startswith("route-")]
        self.assertTrue(routes)
        start, target = cal.points[3], cal.points[2]
        expected = (
            int(round(start[0] + (target[0] - start[0]) * 0.35)),
            int(round(start[1] + (target[1] - start[1]) * 0.35)),
        )
        self.assertEqual(routes[0].coordinate, expected)


if __name__ == "__main__":
    unittest.main()
