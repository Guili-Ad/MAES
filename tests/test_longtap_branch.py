from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

BRANCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH_ROOT))

from agent.common import capture_image
from agent.music.models import (
    HoldTailObservation,
    LaneInputState,
    MusicActionEvent,
    MusicCalibrationData,
    MusicCandidate,
    MusicConfig,
    MusicFrame,
    NoteTrack,
    NoteGesture,
    TrackObservation,
    TrackState,
)
from agent.music.executor import MusicActionExecutor
from agent.music.holds import HoldTailDetection, bonus_hold_ribbon_present, detect_hold_tails
from agent.music.runtime import MusicRuntime, RuntimeMetrics, terminal_state
from agent.music.tracking import MusicVisionEngine
from agent.music.vision import (
    VisualMask,
    detect_bonus_star_notes,
    giant_live_title_present,
    linked_tap_pair_present,
)


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


def candidate_at(cal: MusicCalibrationData, lane: int, progress: float, size: int = 30) -> MusicCandidate:
    x = cal.lane_centerlines[lane][0][0]
    y = 140.0 + (620.0 - 140.0) * progress
    box = (int(x - size / 2), int(y - size / 2), size, size)
    return MusicCandidate(box, size * size, 1.0, (x, y))


def frame_image(cal: MusicCalibrationData, head: MusicCandidate, tail_progress: float | None) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    x, y, width, height = head.box
    image[y : y + height, x : x + width] = [0, 165, 255]
    if tail_progress is not None:
        tail = candidate_at(cal, 3, tail_progress, 24)
        tx, ty, tw, th = tail.box
        image[ty : ty + th, tx : tx + tw] = [200, 225, 245]
    return image


def tail_frame(cal: MusicCalibrationData, lane: int, progress: float) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    tail = candidate_at(cal, lane, progress, 24)
    x, y, width, height = tail.box
    center_x = int(round(tail.center[0]))
    ribbon_top = min(719, y + height)
    ribbon_bottom = min(720, ribbon_top + 82)
    image[ribbon_top:ribbon_bottom, center_x - 7 : center_x + 8] = [220, 220, 220]
    image[y : y + height, x : x + width] = [200, 225, 245]
    return image


def sustain_checkpoint_frame(cal: MusicCalibrationData, lane: int, progress: float) -> np.ndarray:
    image = tail_frame(cal, lane, progress)
    marker = candidate_at(cal, lane, progress, 24)
    center_x = int(round(marker.center[0]))
    upper_bottom = max(0, marker.box[1])
    upper_top = max(0, upper_bottom - 82)
    image[upper_top:upper_bottom, center_x - 7 : center_x + 8] = [220, 220, 220]
    return image


def linked_pair_frame(left: MusicCandidate, right: MusicCandidate, *, connected: bool = True) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    if not connected:
        return image
    left, right = sorted((left, right), key=lambda item: item.center[0])
    start_x, start_y = left.center
    end_x, end_y = right.center
    for parameter in np.linspace(0.0, 1.0, 260):
        x = (1.0 - parameter) * start_x + parameter * end_x
        baseline = (1.0 - parameter) * start_y + parameter * end_y
        y = baseline + 62.0 * 4.0 * parameter * (1.0 - parameter)
        ix, iy = int(round(x)), int(round(y))
        image[max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 245
    return image


def center_color_frame(center_y: int) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    extent = int(round(center_y * 0.28))
    radius = extent // 2
    y0, y1 = center_y - radius, center_y + radius + 1
    x0, x1 = 640 - radius, 640 + radius + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance = xx * xx + yy * yy
    crop = image[y0:y1, x0:x1]
    ring = (distance <= radius * radius) & (distance >= (radius - 5) * (radius - 5))
    inner = distance < (radius - 6) * (radius - 6)
    crop[ring] = [0, 210, 255]
    crop[inner & (xx < 0)] = [40, 80, 230]
    crop[inner & (xx >= 0)] = [230, 200, 40]
    return image


def bonus_star_frame(
    cal: MusicCalibrationData,
    lane: int,
    progress: float,
    *,
    hold: bool = False,
) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    center_x = int(round(cal.lane_centerlines[lane][0][0]))
    center_y = int(round(140.0 + (620.0 - 140.0) * progress))
    extent = int(round(36.0 + progress * 90.0))
    radius = max(10, extent // 2)
    if hold:
        ribbon_half = max(4, int(round(radius * 0.35)))
        ribbon_top = max(0, center_y - int(round(radius * 4.20)))
        ribbon_bottom = max(ribbon_top + 1, center_y - int(round(radius * 0.52)))
        image[
            ribbon_top:ribbon_bottom,
            center_x - ribbon_half : center_x + ribbon_half + 1,
        ] = [220, 220, 220]
    y0, y1 = center_y - radius, center_y + radius + 1
    x0, x1 = center_x - radius, center_x + radius + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    crop = image[y0:y1, x0:x1]
    disc = xx * xx + yy * yy <= radius * radius
    crop[disc] = [45, 210, 170]
    core_half = max(5, int(round(radius * 0.48)))
    arm = max(2, int(round(radius * 0.14)))
    star = (
        ((np.abs(xx) <= arm) & (np.abs(yy) <= core_half))
        | ((np.abs(yy) <= arm) & (np.abs(xx) <= core_half))
    )
    crop[star] = [245, 245, 245]
    return image


def acknowledge_hold_starts(engine: MusicVisionEngine, events: list[MusicActionEvent]) -> None:
    """Model the runtime's successful TouchDown acknowledgement in engine tests."""
    for event in events:
        if event.gesture == NoteGesture.HOLD_START:
            engine.tracks[event.track_id].state = TrackState.HOLDING


class StrictTerminalContext:
    def __init__(self, loading: bool, live: bool) -> None:
        self.loading = loading
        self.live = live
        self.called: list[str] = []

    def run_recognition(self, node: str, image: object) -> object:
        del image
        self.called.append(node)
        hits = {
            "MusicResultLoading": self.loading,
            "MusicResultLive": self.live,
            # These legacy signals must not be consulted by terminal_state.
            "MusicStopDialog": True,
            "MusicResultSuccess": True,
        }
        return SimpleNamespace(hit=hits.get(node, False), filtered_results=[])


class PauseSequenceContext:
    def __init__(self) -> None:
        self.called: list[tuple[str, object]] = []

    def run_recognition(self, node: str, image: object) -> object:
        self.called.append((node, image))
        hits = {
            "MusicPauseDialog": image == "paused",
            "MusicLiveScreen": image == "live",
            "MusicLiveClearScreen": False,
        }
        return SimpleNamespace(hit=hits.get(node, False), filtered_results=[])


class ReleaseProbe:
    def __init__(self) -> None:
        self.releases = 0

    def release_all(self) -> None:
        self.releases += 1


class LongTapBranchTests(unittest.TestCase):
    def test_teal_tap_events_are_identical_with_hold_channel_on_or_off(self) -> None:
        cal = calibration()
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        engines = [
            MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False)),
            MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True)),
        ]
        outputs: list[list[dict[str, object]]] = [[], []]
        for sequence, progress in enumerate([0.20, 0.30, 0.42, 0.56, 0.72]):
            note = candidate_at(cal, 2, progress)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            x, y, width, height = note.box
            image[y : y + height, x : x + width] = [210, 200, 0]
            frame = MusicFrame(sequence, sequence * 0.1, sequence * 0.1, sequence * 0.1, image)
            for index, engine in enumerate(engines):
                outputs[index].extend(asdict(event) for event in engine.update(frame, [note], visual))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual([item["gesture"] for item in outputs[1]], [NoteGesture.TAP])

    def test_white_arc_links_two_taps_and_forces_one_shared_deadline(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events: list[MusicActionEvent] = []
        for sequence, left_progress in enumerate([0.20, 0.29, 0.39, 0.51, 0.65]):
            right_progress = left_progress - (0.018 + sequence * 0.001)
            left = candidate_at(cal, 2, left_progress, 46)
            right = candidate_at(cal, 4, right_progress, 46)
            image = linked_pair_frame(left, right)
            frame = MusicFrame(sequence, sequence * 0.1, sequence * 0.1, sequence * 0.1, image)
            events.extend(engine.update(frame, [left, right], visual))
        taps = [event for event in events if event.gesture == NoteGesture.TAP]
        self.assertEqual(len(taps), 2)
        self.assertEqual(taps[0].deadline, taps[1].deadline)
        self.assertEqual(engine.tracks[taps[0].track_id].linked_partner_id, taps[1].track_id)
        self.assertEqual(engine.tracks[taps[1].track_id].linked_partner_id, taps[0].track_id)
        self.assertNotEqual(
            engine.tracks[taps[0].track_id].predicted_hit_time,
            engine.tracks[taps[1].track_id].predicted_hit_time,
        )
        refined = engine.refine_pending(taps, 0.4)
        self.assertEqual(refined[0].deadline, refined[1].deadline)

    def test_neighbouring_taps_without_white_arc_are_not_linked(self) -> None:
        cal = calibration()
        left = candidate_at(cal, 2, 0.45, 46)
        right = candidate_at(cal, 4, 0.45, 46)
        self.assertTrue(linked_tap_pair_present(linked_pair_frame(left, right), left, right))
        self.assertFalse(linked_tap_pair_present(linked_pair_frame(left, right, connected=False), left, right))

    def test_three_sample_deadline_rescue_applies_to_fresh_taps_on_every_lane(self) -> None:
        cal = calibration()
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        def run(*, lane: int, active_hold: bool) -> list[MusicActionEvent]:
            engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
            if active_hold:
                owner = NoteTrack(track_id=90, lane=0, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
                owner.action_executed = True
                owner.predicted_hit_time = 0.0
                owner.hold_release_time = 1.8
                engine.tracks[owner.track_id] = owner
                engine.next_track_id = 91
            events: list[MusicActionEvent] = []
            for sequence, progress in enumerate([0.55, 0.68, 0.81]):
                note = candidate_at(cal, lane, progress)
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                x, y, width, height = note.box
                image[y : y + height, x : x + width] = [210, 200, 0]
                timestamp = sequence * 0.1
                events.extend(engine.update(MusicFrame(sequence, timestamp, timestamp, timestamp, image), [note], visual))
            return [event for event in events if event.gesture == NoteGesture.TAP]

        ordinary_rescued = run(lane=1, active_hold=False)
        self.assertEqual(len(ordinary_rescued), 1)
        center_rescued = run(lane=3, active_hold=False)
        self.assertEqual(len(center_rescued), 1)
        rescued = run(lane=1, active_hold=True)
        self.assertEqual(len(rescued), 1)
        self.assertAlmostEqual(rescued[0].deadline, 0.221153846, places=6)

        linked_engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        linked_events: list[MusicActionEvent] = []
        for sequence, progress in enumerate([0.55, 0.68, 0.81]):
            left = candidate_at(cal, 2, progress, 46)
            right = candidate_at(cal, 4, progress - 0.01, 46)
            image = linked_pair_frame(left, right)
            timestamp = sequence * 0.1
            linked_events.extend(
                linked_engine.update(
                    MusicFrame(sequence, timestamp, timestamp, timestamp, image),
                    [left, right],
                    visual,
                )
            )
        linked_taps = [event for event in linked_events if event.gesture == NoteGesture.TAP]
        self.assertEqual(len(linked_taps), 2)
        self.assertEqual(linked_taps[0].deadline, linked_taps[1].deadline)

    def test_three_sample_deadline_rescue_applies_to_confirmed_hold_head(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events: list[MusicActionEvent] = []
        for sequence, progress in enumerate([0.55, 0.68, 0.81]):
            note = candidate_at(cal, 2, progress)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            x, y, width, height = note.box
            image[y : y + height, x : x + width] = [0, 165, 255]
            timestamp = sequence * 0.1
            events.extend(
                engine.update(
                    MusicFrame(sequence, timestamp, timestamp, timestamp, image),
                    [note],
                    visual,
                )
            )
        starts = [event for event in events if event.gesture == NoteGesture.HOLD_START]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].track_id, 1)

    def test_far_three_sample_head_waits_for_reliable_fourth_observation(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events: list[MusicActionEvent] = []
        # Each inter-frame displacement remains within the production
        # association gate.  After three samples the fitted deadline remains
        # 220 ms away, so it must not commit the early-biased rescue observed
        # in the third live-test round.
        for sequence, progress in enumerate([0.18, 0.31, 0.44]):
            timestamp = sequence * 0.08
            note = candidate_at(cal, 3, progress)
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
            events.extend(engine.update(frame, [note], visual))

        taps = [event for event in events if event.gesture == NoteGesture.TAP]
        self.assertEqual(taps, [])

        note = candidate_at(cal, 3, 0.57)
        frame = MusicFrame(3, 0.24, 0.24, 0.24, np.zeros((720, 1280, 3), dtype=np.uint8))
        taps = [event for event in engine.update(frame, [note], visual) if event.gesture == NoteGesture.TAP]
        self.assertEqual(len(taps), 1)
        self.assertGreater(taps[0].deadline, 0.24)

    def test_center_lane_keeps_a_bounded_lower_progress_rescue(self) -> None:
        cal = calibration()
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        def run(lane: int) -> list[MusicActionEvent]:
            engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
            events: list[MusicActionEvent] = []
            for sequence, progress in enumerate([0.16, 0.30, 0.44]):
                timestamp = sequence * 0.07
                note = candidate_at(cal, lane, progress)
                frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
                events.extend(engine.update(frame, [note], visual))
            return [event for event in events if event.gesture == NoteGesture.TAP]

        self.assertEqual(len(run(3)), 1)
        self.assertEqual(run(2), [])

    def test_dense_cross_lane_cluster_delays_only_unlinked_ordinary_taps(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        for track_id, lane, progress, hit_time in (
            (1, 2, 0.70, 1.20),
            (2, 3, 0.60, 1.42),
            (3, 4, 0.50, 1.61),
            (4, 5, 0.30, 2.40),
        ):
            track = NoteTrack(track_id=track_id, lane=lane, speed=1.0, predicted_hit_time=hit_time)
            for sequence in range(3):
                note = candidate_at(cal, lane, progress - (2 - sequence) * 0.05)
                track.observations.append(
                    TrackObservation(sequence, 0.9 + sequence * 0.05, note.center, progress - (2 - sequence) * 0.05, note)
                )
            engine.tracks[track_id] = track

        frame = MusicFrame(2, 1.0, 1.0, 1.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        engine._stabilize_dense_tap_timing(frame)

        self.assertEqual([engine.tracks[track_id].dense_tap for track_id in (1, 2, 3)], [True, True, True])
        self.assertFalse(engine.tracks[4].dense_tap)
        self.assertEqual(engine._action_advance_ms(engine.tracks[1]), 95.0)
        self.assertEqual(engine._action_advance_ms(engine.tracks[4]), 125.0)
        engine.tracks[1].linked_partner_id = engine.tracks[2].track_id
        self.assertEqual(engine._action_advance_ms(engine.tracks[1]), 125.0)

    def test_dense_timing_does_not_fabricate_fixed_visual_hit_gaps(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        for track_id, progress, hit_time in ((1, 0.80, 1.00), (2, 0.70, 0.98), (3, 0.60, 1.01)):
            track = NoteTrack(track_id=track_id, lane=2, speed=0.5 + track_id, predicted_hit_time=hit_time)
            note = candidate_at(cal, 2, progress)
            track.observations.append(TrackObservation(2, 0.9, note.center, progress, note))
            engine.tracks[track_id] = track

        frame = MusicFrame(2, 1.0, 1.0, 1.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        engine._stabilize_dense_tap_timing(frame)
        hit_times = [engine.tracks[track_id].predicted_hit_time for track_id in (1, 2, 3)]

        self.assertEqual(hit_times, [1.0, 0.98, 1.01])
        self.assertEqual([engine.tracks[track_id].speed for track_id in (1, 2, 3)], [1.5, 2.5, 3.5])

    def test_two_note_same_or_adjacent_lane_burst_uses_tap_only_dense_timing(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        for track_id, lane, hit_time in ((1, 2, 1.00), (2, 3, 1.27), (3, 5, 2.00)):
            track = NoteTrack(track_id=track_id, lane=lane, predicted_hit_time=hit_time)
            note = candidate_at(cal, lane, 0.70)
            track.observations.append(TrackObservation(2, 0.9, note.center, 0.70, note))
            engine.tracks[track_id] = track

        frame = MusicFrame(2, 0.9, 0.9, 0.9, np.zeros((720, 1280, 3), dtype=np.uint8))
        engine._stabilize_dense_tap_timing(frame)

        self.assertTrue(engine.tracks[1].dense_tap)
        self.assertTrue(engine.tracks[2].dense_tap)
        self.assertFalse(engine.tracks[3].dense_tap)
        hold = NoteTrack(track_id=4, lane=2, gesture=NoteGesture.HOLD_START)
        hold.dense_tap = True
        self.assertEqual(engine._action_advance_ms(hold), engine.config.hold_start_action_advance_ms)

    def test_exactly_two_same_lane_taps_reach_the_pair_policy(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        for track_id, progress, hit_time in ((1, 0.72, 1.00), (2, 0.54, 1.55)):
            track = NoteTrack(track_id=track_id, lane=2, predicted_hit_time=hit_time)
            note = candidate_at(cal, 2, progress)
            track.observations.append(TrackObservation(2, 0.9, note.center, progress, note))
            engine.tracks[track_id] = track

        engine._stabilize_dense_tap_timing(
            MusicFrame(2, 0.9, 0.9, 0.9, np.zeros((720, 1280, 3), dtype=np.uint8))
        )

        self.assertTrue(engine.tracks[1].dense_tap)
        self.assertTrue(engine.tracks[2].dense_tap)

    def test_equal_first_seen_cadence_does_not_rewrite_visual_hit_times(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        for track_id, first_seen, progress, raw_hit in (
            (1, 0.0, 0.82, 2.00),
            (2, 0.5, 0.62, 2.80),
            (3, 1.0, 0.42, 3.05),
        ):
            track = NoteTrack(
                track_id=track_id,
                lane=3,
                first_seen_time=first_seen,
                predicted_hit_time=raw_hit,
            )
            previous = candidate_at(cal, 3, progress - 0.04)
            current = candidate_at(cal, 3, progress)
            track.observations.extend((
                TrackObservation(9, 1.35, previous.center, progress - 0.04, previous),
                TrackObservation(10, 1.40, current.center, progress, current),
            ))
            engine.tracks[track_id] = track

        engine._stabilize_dense_tap_timing(
            MusicFrame(10, 1.4, 1.4, 1.4, np.zeros((720, 1280, 3), dtype=np.uint8))
        )
        locked = [engine.tracks[index].tap_cadence_hit_time for index in (1, 2, 3)]

        self.assertEqual(locked, [None, None, None])
        self.assertEqual([engine.tracks[index].predicted_hit_time for index in (1, 2, 3)], [2.0, 2.8, 3.05])
        self.assertTrue(all(engine.tracks[index].dense_tap for index in (2, 3)))

    def test_terminal_target_does_not_drift_across_intermediate_lane(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=7, lane=4, gesture=NoteGesture.HOLD_START)
        track.predicted_hit_time = 0.0
        engine.tracks[track.track_id] = track

        for sequence, progress in enumerate([0.40, 0.50, 0.60]):
            timestamp = sequence * 0.1
            engine._record_active_hold_tail(
                track,
                MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8)),
                HoldTailDetection(progress, 0.8, 100, 2, (480.0, 140.0 + 480.0 * progress), 0.0),
            )
        self.assertEqual(track.hold_target_lane, 2)

        # A curved right-to-left ribbon can visually pass through lane 3 near
        # judgement.  That later corridor consensus must not replace the
        # already proven terminal ray.
        for offset, progress in enumerate([0.70, 0.80, 0.89], start=3):
            timestamp = offset * 0.1
            engine._record_active_hold_tail(
                track,
                MusicFrame(offset, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8)),
                HoldTailDetection(progress, 0.8, 100, 3, (600.0, 140.0 + 480.0 * progress), 0.0),
            )
        self.assertEqual(track.hold_target_lane, 2)

    def test_folded_hold_confirms_target_only_ribbon_temporally(self) -> None:
        engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=7, lane=1, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.hold_target_lane = 4

        self.assertFalse(engine.hold_policy.observe_route_ribbons(
            track, source_lane_visible=True, target_lane_visible=True
        ))
        self.assertFalse(engine.hold_policy.observe_route_ribbons(
            track, source_lane_visible=False, target_lane_visible=True
        ))
        self.assertTrue(engine.hold_policy.observe_route_ribbons(
            track, source_lane_visible=False, target_lane_visible=True
        ))
        self.assertTrue(track.hold_fold_route_confirmed)

    def test_folded_hold_moves_to_target_then_keeps_straight_segment(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(
            track_id=7,
            lane=1,
            gesture=NoteGesture.HOLD_START,
            state=TrackState.HOLDING,
            predicted_hit_time=0.0,
        )
        track.hold_target_lane = 4
        track.hold_release_time = 5.0
        track.hold_fold_route_confirmed = True
        tail = candidate_at(cal, 4, 0.60)
        track.hold_tail_observations.append(
            HoldTailObservation(5, 1.0, 0.60, 0.8, 100, 4, tail.center)
        )
        engine.tracks[track.track_id] = track

        events = engine.release_events(1.0)
        folded = next(event for event in events if event.event_id == "route-fold-7")

        self.assertEqual(folded.deadline, 1.0)
        self.assertEqual(folded.coordinate, tuple(cal.points[4]))
        self.assertTrue(track.hold_fold_move_scheduled)
        self.assertTrue(track.hold_move_scheduled)
        self.assertEqual(track.hold_route_steps_completed, 3)

    def test_white_arc_links_two_holds_and_shares_reliable_release(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events: list[MusicActionEvent] = []
        for sequence, progress in enumerate([0.20, 0.29, 0.39, 0.51, 0.65]):
            left = candidate_at(cal, 2, progress, 46)
            right = candidate_at(cal, 4, progress, 46)
            image = linked_pair_frame(left, right)
            for note in (left, right):
                x, y, width, height = note.box
                image[y : y + height, x : x + width] = [0, 165, 255]
            frame = MusicFrame(sequence, sequence * 0.1, sequence * 0.1, sequence * 0.1, image)
            events.extend(engine.update(frame, [left, right], visual))
        starts = [event for event in events if event.gesture == NoteGesture.HOLD_START]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0].deadline, starts[1].deadline)
        acknowledge_hold_starts(engine, starts)
        left_track = engine.tracks[starts[0].track_id]
        right_track = engine.tracks[starts[1].track_id]
        self.assertEqual(left_track.linked_partner_id, right_track.track_id)
        self.assertEqual(right_track.linked_partner_id, left_track.track_id)

        left_track.predicted_hit_time = 1.0
        right_track.predicted_hit_time = 1.08
        refined_starts = engine.refine_pending(starts, 0.4)
        self.assertEqual(refined_starts[0].deadline, refined_starts[1].deadline)

        left_track.hold_release_time = 2.2
        right_track.hold_release_time = 2.2
        engine._synchronize_linked_hold_releases(MusicFrame(8, 1.8, 1.8, 1.8, np.zeros((720, 1280, 3), dtype=np.uint8)))
        self.assertGreater(left_track.hold_release_time, 2.2)
        self.assertEqual(left_track.hold_release_time, right_track.hold_release_time)

        left_track.hold_release_time = 3.4
        left_track.hold_release_locked = True
        right_track.hold_release_locked = False
        right_before = right_track.hold_release_time
        engine._synchronize_linked_hold_releases(MusicFrame(9, 2.5, 2.5, 2.5, np.zeros((720, 1280, 3), dtype=np.uint8)))
        self.assertFalse(right_track.hold_release_locked)
        self.assertIsNone(right_track.linked_release_source_id)
        self.assertEqual(right_track.hold_release_time, right_before)

        right_track.hold_release_time = 5.0
        right_track.hold_tail_observations.extend(
            HoldTailObservation(index, 2.5 + index * 0.1, 0.45 + index * 0.1, 0.8, 100, 4, (700.0, 350.0 + index * 20.0))
            for index in range(3)
        )
        engine._synchronize_linked_hold_releases(MusicFrame(10, 2.6, 2.6, 2.6, np.zeros((720, 1280, 3), dtype=np.uint8)))
        self.assertIsNone(right_track.linked_release_source_id)
        self.assertEqual(right_track.hold_release_time, 5.0)

        right_track.hold_tail_observations.clear()
        right_track.hold_tail_observations.extend(
            HoldTailObservation(index, 2.7 + index * 0.1, 0.84 + index * 0.04, 0.8, 100, 4, (700.0, 500.0 + index * 12.0))
            for index in range(3)
        )
        right_track.hold_terminal_confirmed = True
        right_track.hold_release_time = 3.9
        engine._synchronize_linked_hold_releases(MusicFrame(11, 2.9, 2.9, 2.9, np.zeros((720, 1280, 3), dtype=np.uint8)))
        self.assertTrue(right_track.hold_release_locked)
        self.assertEqual(left_track.hold_release_time, right_track.hold_release_time)
        self.assertEqual(right_track.hold_release_time, 3.4)

    def test_visible_held_head_can_link_to_later_bonus_hold_detection(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        first_events: list[MusicActionEvent] = []
        for sequence, progress in enumerate([0.20, 0.29, 0.39, 0.51, 0.65]):
            left = candidate_at(cal, 2, progress, 46)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            x, y, width, height = left.box
            image[y : y + height, x : x + width] = [0, 165, 255]
            frame = MusicFrame(sequence, sequence * 0.1, sequence * 0.1, sequence * 0.1, image)
            first_events.extend(engine.update(frame, [left], visual))
        first_start = next(event for event in first_events if event.gesture == NoteGesture.HOLD_START)
        first_track = engine.tracks[first_start.track_id]
        self.assertTrue(first_track.action_executed)

        left = candidate_at(cal, 2, 0.77, 46)
        right = candidate_at(cal, 4, 0.75, 46)
        image = linked_pair_frame(left, right)
        for note in (left, right):
            x, y, width, height = note.box
            image[y : y + height, x : x + width] = [0, 165, 255]
        engine.update(MusicFrame(5, 0.5, 0.5, 0.5, image), [left, right], visual)

        later_track = next(
            track
            for track in engine.tracks.values()
            if track.track_id != first_track.track_id
            and track.observations
            and track.observations[-1].frame_sequence == 5
        )
        self.assertEqual(first_track.linked_partner_id, later_track.track_id)
        self.assertEqual(later_track.linked_partner_id, first_track.track_id)

    def test_arrived_active_hold_cannot_consume_a_later_same_lane_tap(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        owner = NoteTrack(
            track_id=20,
            lane=3,
            gesture=NoteGesture.HOLD_START,
            state=TrackState.HOLDING,
            predicted_hit_time=1.0,
        )
        owner.action_executed = True
        prior = candidate_at(cal, 3, 0.78)
        owner.observations.append(TrackObservation(1, 0.9, prior.center, 0.78, prior))
        owner.speed = 0.4
        engine.tracks[owner.track_id] = owner
        engine.next_track_id = 21

        incoming = candidate_at(cal, 3, 0.90)
        frame = MusicFrame(2, 1.25, 1.25, 1.25, np.zeros((720, 1280, 3), dtype=np.uint8))
        engine.update(frame, [incoming], visual)

        self.assertEqual(owner.observations[-1].frame_sequence, 1)
        self.assertIn(21, engine.tracks)
        self.assertEqual(engine.tracks[21].observations[-1].frame_sequence, 2)
        self.assertEqual(engine.isolated_same_lane_head_count, 1)

    def test_tap_and_hold_start_rescue_settings_are_independent(self) -> None:
        config = MusicConfig(
            lane_count=7,
            tap_action_advance_ms=91.0,
            hold_start_action_advance_ms=127.0,
            urgent_tap_min_progress=0.61,
            urgent_hold_start_min_progress=0.43,
        )
        engine = MusicVisionEngine(calibration(), config)
        tap = NoteTrack(track_id=1, lane=2, gesture=NoteGesture.TAP)
        hold = NoteTrack(track_id=2, lane=2, gesture=NoteGesture.HOLD_START)

        self.assertEqual(engine._action_advance_ms(tap), 91.0)
        self.assertEqual(engine._action_advance_ms(hold), 127.0)

    def test_bonus_star_detector_rejects_teal_round_core(self) -> None:
        cal = calibration()
        star_image = bonus_star_frame(cal, 4, 0.55)
        stars = detect_bonus_star_notes(star_image, cal)
        self.assertEqual(len(stars), 1)
        self.assertEqual(stars[0].variant, "bonus_star")

        teal = np.zeros((720, 1280, 3), dtype=np.uint8)
        note = candidate_at(cal, 4, 0.55, 64)
        x, y, width, height = note.box
        yy, xx = np.ogrid[-height // 2 : height - height // 2, -width // 2 : width - width // 2]
        crop = teal[y : y + height, x : x + width]
        crop[xx * xx + yy * yy <= 30 * 30] = [180, 210, 30]
        crop[xx * xx + yy * yy <= 10 * 10] = [245, 245, 245]
        self.assertEqual(detect_bonus_star_notes(teal, cal), [])

    def test_bonus_star_tap_uses_one_ordinary_lane_event(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        progress_samples = [0.22, 0.27, 0.32, 0.36, 0.44, 0.52, 0.60, 0.70]
        for sequence, progress in enumerate(progress_samples):
            timestamp = sequence * 0.1
            image = bonus_star_frame(cal, 4, progress)
            ordinary_duplicate = candidate_at(cal, 4, progress, 30)
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
            current = engine.update(frame, [ordinary_duplicate], visual)
            if progress < 0.58:
                self.assertEqual(current, [])
            events.extend(current)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.gesture, NoteGesture.TAP)
        self.assertEqual(event.lane, 4)
        self.assertGreater(event.track_id, 0)
        self.assertTrue(engine.tracks[event.track_id].bonus_star)

    def test_center_lane_bonus_star_does_not_enter_rainbow_singleton(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        for sequence, progress in enumerate([0.25, 0.35, 0.46, 0.58, 0.72]):
            timestamp = sequence * 0.1
            frame = MusicFrame(
                sequence,
                timestamp,
                timestamp,
                timestamp,
                bonus_star_frame(cal, 3, progress),
            )
            events.extend(engine.update(frame, [], visual))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].gesture, NoteGesture.TAP)
        self.assertGreater(events[0].track_id, 0)
        self.assertTrue(engine.tracks[events[0].track_id].bonus_star)

    def test_bonus_star_with_upstream_ribbon_reuses_hold_start(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        progress_samples = [0.22, 0.27, 0.32, 0.36, 0.44, 0.52, 0.60, 0.70]
        for sequence, progress in enumerate(progress_samples):
            timestamp = sequence * 0.1
            frame = MusicFrame(
                sequence,
                timestamp,
                timestamp,
                timestamp,
                # One confirmed ribbon frame now commits a bonus hold (the star
                # head is shared by taps and holds); the next frame deliberately
                # hides it like the real judgement flash.
                bonus_star_frame(cal, 4, progress, hold=sequence in {4, 5}),
            )
            current = engine.update(frame, [], visual)
            if progress <= 0.36:
                self.assertEqual(current, [])
            events.extend(current)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.gesture, NoteGesture.HOLD_START)
        self.assertEqual(event.contact_policy, "persistent")
        self.assertEqual(engine.tracks[event.track_id].state, TrackState.HOLD_PENDING)

    def test_tinted_curved_bonus_ribbon_is_detected_without_relaxing_temporal_confirmation(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        candidate = MusicCandidate((610, 350, 60, 60), 1800, 0.5, (640.0, 380.0), "bonus_star")
        radius = 30.0
        angle = np.deg2rad(20.0)
        tangent = (0.0, 1.0)
        direction = (
            tangent[0] * np.cos(angle) - tangent[1] * np.sin(angle),
            tangent[0] * np.sin(angle) + tangent[1] * np.cos(angle),
        )
        normal = (-direction[1], direction[0])
        for along in np.linspace(radius * 0.55, radius * 2.10, 100):
            for across in np.linspace(-radius * 0.42, radius * 0.42, 28):
                x = int(round(candidate.center[0] - direction[0] * along + normal[0] * across))
                y = int(round(candidate.center[1] - direction[1] * along + normal[1] * across))
                image[y, x] = [150, 195, 225]
        self.assertTrue(bonus_hold_ribbon_present(image, candidate, tangent))

    def test_pale_stage_without_local_ribbon_contrast_is_not_a_bonus_hold(self) -> None:
        image = np.full((720, 1280, 3), [145, 180, 205], dtype=np.uint8)
        candidate = MusicCandidate((610, 350, 60, 60), 1800, 0.5, (640.0, 380.0), "bonus_star")
        self.assertFalse(bonus_hold_ribbon_present(image, candidate, (0.0, 1.0)))

    def test_bonus_star_white_core_is_not_reused_as_a_hold_tail(self) -> None:
        cal = calibration()
        config = MusicConfig(lane_count=7, enable_holds=True)
        image = bonus_star_frame(cal, 4, 0.62)
        tails = detect_hold_tails(image, cal, config)
        star_center = candidate_at(cal, 4, 0.62).center
        self.assertFalse(any(np.hypot(tail.center[0] - star_center[0], tail.center[1] - star_center[1]) < 55 for tail in tails))

    def test_hold_marker_topology_separates_terminal_from_mid_hold_checkpoint(self) -> None:
        cal = calibration()
        config = MusicConfig(lane_count=7, enable_holds=True)
        terminal = detect_hold_tails(tail_frame(cal, 3, 0.45), cal, config)
        checkpoint = detect_hold_tails(sustain_checkpoint_frame(cal, 3, 0.45), cal, config)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(len(checkpoint), 1)
        self.assertEqual(terminal[0].ribbon_exit_count, 1)
        self.assertGreaterEqual(checkpoint[0].ribbon_exit_count, 2)

    def test_mid_hold_checkpoint_refreshes_watchdog_but_never_starts_release_fit(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=95, lane=3, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 0.0
        track.hold_release_time = 1.8
        engine.tracks[track.track_id] = track
        previous_image = sustain_checkpoint_frame(cal, 3, 0.35)
        current_image = sustain_checkpoint_frame(cal, 3, 0.42)
        previous = detect_hold_tails(previous_image, cal, engine.config)
        engine.previous_hold_tails = previous
        engine.previous_hold_tail_streaks = [1] * len(previous)
        frame = MusicFrame(2, 1.0, 1.0, 1.0, current_image)
        engine._update_active_hold_tails(frame)
        self.assertEqual(list(track.hold_tail_observations), [])
        self.assertFalse(track.hold_terminal_confirmed)
        self.assertFalse(track.hold_release_locked)
        self.assertEqual(track.hold_last_continuity_time, 1.0)
        self.assertGreater(track.hold_release_time, 1.8)

    def test_checkpoint_identity_survives_one_sided_judgement_line_occlusion(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=97, lane=3, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 0.0
        track.hold_release_time = 1.8
        engine.tracks[track.track_id] = track

        first_image = sustain_checkpoint_frame(cal, 3, 0.35)
        first = detect_hold_tails(first_image, cal, engine.config)
        engine.previous_hold_tails = first
        engine.previous_hold_tail_streaks = [1] * len(first)
        engine.previous_hold_tail_checkpoint_flags = [False] * len(first)
        engine._update_active_hold_tails(MusicFrame(2, 1.0, 1.0, 1.0, sustain_checkpoint_frame(cal, 3, 0.42)))

        # The lower ribbon is now hidden by the judgement effect, so the same
        # marker looks terminal in this frame.  Its established trajectory
        # identity must still prevent release acquisition.
        engine._update_active_hold_tails(MusicFrame(3, 1.1, 1.1, 1.1, tail_frame(cal, 3, 0.50)))
        self.assertEqual(list(track.hold_tail_observations), [])
        self.assertFalse(track.hold_terminal_confirmed)
        self.assertFalse(track.hold_release_locked)
        self.assertEqual(track.hold_last_continuity_time, 1.1)

    def test_moving_same_lane_tail_schedules_persistent_hold_and_predicted_release(self) -> None:
        cal = calibration()
        config = MusicConfig(lane_count=7, enable_holds=True)
        engine = MusicVisionEngine(cal, config)
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        samples = [0.20, 0.30, 0.40, 0.55, 0.72]
        for sequence, head_progress in enumerate(samples):
            head = candidate_at(cal, 3, head_progress)
            image = frame_image(cal, head, None)
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
            current = engine.update(frame, [head], visual)
            events.extend(current)
            acknowledge_hold_starts(engine, current)
            events.extend(engine.release_events(timestamp))
        for sequence, progress in enumerate([0.03, 0.10, 0.22, 0.38, 0.58, 0.78, 0.92], start=8):
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, tail_frame(cal, 3, progress))
            events.extend(engine.update(frame, [], visual))
            events.extend(engine.release_events(timestamp))
        self.assertIn(NoteGesture.HOLD_START, [event.gesture for event in events])
        self.assertIn(NoteGesture.HOLD_END, [event.gesture for event in events])
        start = next(event for event in events if event.gesture == NoteGesture.HOLD_START)
        track = engine.tracks[start.track_id]
        self.assertEqual(start.contact_policy, "persistent")
        self.assertEqual(track.hold_target_lane, 3)
        self.assertTrue(track.hold_release_locked)
        self.assertIsNotNone(track.hold_release_time)
        self.assertGreater(track.hold_release_time, start.deadline + 0.7)

    def test_confirmed_orange_head_without_visible_tail_uses_bounded_fallback_hold(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        for sequence, head_progress in enumerate([0.30, 0.40, 0.55, 0.70, 0.88]):
            head = candidate_at(cal, 3, head_progress)
            image = frame_image(cal, head, None)
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
            current = engine.update(frame, [head], visual)
            events.extend(current)
            acknowledge_hold_starts(engine, current)
            events.extend(engine.release_events(timestamp))
        start = next(event for event in events if event.gesture == NoteGesture.HOLD_START)
        end = next(event for event in events if event.gesture == NoteGesture.HOLD_END)
        self.assertGreater(end.deadline, start.deadline)
        self.assertLess(end.deadline - start.deadline, 2.1)

    def test_transient_head_ribbon_disappearance_cannot_release_standard_hold_early(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        for sequence, head_progress in enumerate([0.20, 0.30, 0.40, 0.55, 0.72]):
            head = candidate_at(cal, 3, head_progress)
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, frame_image(cal, head, None))
            current = engine.update(frame, [head], visual)
            events.extend(current)
            acknowledge_hold_starts(engine, current)
            events.extend(engine.release_events(timestamp))
        start = next(event for event in events if event.gesture == NoteGesture.HOLD_START)
        track = engine.tracks[start.track_id]

        ribbon = np.zeros((720, 1280, 3), dtype=np.uint8)
        ribbon[430:570, 620:660] = [220, 220, 220]
        frame = MusicFrame(5, 0.5, 0.5, 0.5, ribbon)
        engine.update(frame, [], visual)
        engine.release_events(0.5)
        for sequence in range(6, 10):
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
            engine.update(frame, [], visual)
            engine.release_events(timestamp)

        self.assertEqual(track.state, TrackState.HOLDING)
        self.assertFalse(track.hold_release_locked)
        self.assertGreater(track.hold_release_time, 2.0)

    def test_unverified_ribbon_cannot_keep_a_hold_alive_past_the_safe_limit(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        for sequence, head_progress in enumerate([0.20, 0.30, 0.40, 0.55, 0.72]):
            head = candidate_at(cal, 3, head_progress)
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, frame_image(cal, head, None))
            current = engine.update(frame, [head], visual)
            events.extend(current)
            acknowledge_hold_starts(engine, current)
            engine.release_events(timestamp)
        start = next(event for event in events if event.gesture == NoteGesture.HOLD_START)
        track = engine.tracks[start.track_id]
        # A static pale patch models the false-positive that kept live track
        # 282 pressed for roughly fifty seconds.  Ribbon alone must not unlock
        # the extended-duration path.
        for sequence in range(5, 206):
            timestamp = sequence * 0.1
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            image[430:570, 620:660] = [220, 220, 220]
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
            engine.update(frame, [], visual)
            engine.release_events(timestamp)
        self.assertEqual(track.state, TrackState.RELEASED)
        self.assertLessEqual(
            track.hold_release_time,
            track.predicted_hit_time + engine.config.hold_unverified_max_tail_ms / 1000.0,
        )
        self.assertGreater(engine.config.max_contact_ms, engine.config.hold_max_tail_ms)

    def test_stable_slow_tail_motion_unlocks_extended_release_prediction(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=90, lane=3)
        track.state = TrackState.HOLDING
        track.predicted_hit_time = 0.1
        engine.tracks[track.track_id] = track
        for sequence, progress in enumerate([0.020, 0.035, 0.050, 0.065, 0.080, 0.095], start=1):
            timestamp = 0.5 + sequence * 0.5
            center = (640.0, 140.0 + 480.0 * progress)
            tail = HoldTailDetection(progress, 0.8, 100, 3, center, 0.0)
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
            engine._record_active_hold_tail(track, frame, tail)
        self.assertIsNotNone(track.hold_release_time)
        self.assertGreater(track.hold_release_time, track.predicted_hit_time + 14.0)
        self.assertFalse(track.hold_release_locked)
        self.assertTrue(track.hold_long_verified)

    def test_sub_twenty_two_second_slow_fit_cannot_unlock_ultra_long_mode(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=92, lane=3)
        track.state = TrackState.HOLDING
        track.predicted_hit_time = 0.1
        engine.tracks[track.track_id] = track
        for sequence, progress in enumerate([0.020, 0.042, 0.064, 0.086, 0.108, 0.130], start=1):
            timestamp = 0.5 + sequence * 0.4
            tail = HoldTailDetection(progress, 0.8, 100, 3, (640.0, 140.0 + 480.0 * progress), 0.0)
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
            engine._record_active_hold_tail(track, frame, tail)
        self.assertFalse(track.hold_long_verified)
        if track.hold_release_time is not None:
            self.assertLessEqual(
                track.hold_release_time,
                track.predicted_hit_time + engine.config.hold_unverified_max_tail_ms / 1000.0,
            )

    def test_static_mature_cap_and_ribbon_loss_infers_short_release(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=93, lane=3, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 0.0
        track.hold_release_time = engine.config.hold_fallback_duration_ms / 1000.0
        track.hold_terminal_confirmed = True
        for sequence, timestamp in enumerate([0.40, 0.48], start=1):
            track.hold_tail_observations.append(
                HoldTailObservation(sequence, timestamp, 0.60, 0.8, 100, 3, (640.0, 428.0))
            )
        engine.tracks[track.track_id] = track
        with patch("agent.music.tracking.detect_hold_tails", return_value=[]), patch(
            "agent.music.tracking.hold_ribbon_present", return_value=False
        ):
            for sequence, timestamp in enumerate([0.60, 0.68, 0.76], start=3):
                frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
                engine._update_active_hold_tails(frame)
        self.assertTrue(track.hold_release_locked)
        self.assertAlmostEqual(track.hold_release_time, 0.84)
        self.assertLess(track.hold_release_time, engine.config.hold_fallback_duration_ms / 1000.0)

    def test_disappearing_mid_hold_checkpoint_cannot_infer_release(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=96, lane=3, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 0.0
        track.hold_release_time = 1.8
        track.hold_tail_observations.extend(
            HoldTailObservation(sequence, timestamp, 0.60, 0.8, 100, 3, (640.0, 428.0), 2)
            for sequence, timestamp in enumerate([0.40, 0.48], start=1)
        )
        engine.tracks[track.track_id] = track
        with patch("agent.music.tracking.detect_hold_tails", return_value=[]), patch(
            "agent.music.tracking.hold_ribbon_present", return_value=False
        ):
            for sequence, timestamp in enumerate([0.60, 0.68, 0.76], start=3):
                frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
                engine._update_active_hold_tails(frame)
        self.assertFalse(track.hold_release_locked)
        self.assertEqual(track.hold_release_time, 1.8)

    def test_early_tail_lane_wobble_cannot_schedule_a_route(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=94, lane=5, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 0.0
        track.hold_release_time = 1.8
        engine.tracks[track.track_id] = track
        for sequence, progress in enumerate([0.20, 0.27, 0.331], start=1):
            timestamp = sequence * 0.1
            tail = HoldTailDetection(progress, 0.8, 100, 6, (1120.0, 140.0 + 480.0 * progress), 0.0)
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, np.zeros((720, 1280, 3), dtype=np.uint8))
            engine._record_active_hold_tail(track, frame, tail)
        self.assertIsNone(track.hold_target_lane)
        self.assertFalse(any(event.gesture == NoteGesture.HOLD_CONTINUE for event in engine.release_events(0.3)))

    def test_linked_hold_caps_are_jointly_assigned_without_crossing_sides(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        left = NoteTrack(track_id=30, lane=1, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        right = NoteTrack(track_id=31, lane=5, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        left.predicted_hit_time = right.predicted_hit_time = 0.0
        left.linked_partner_id = right.track_id
        right.linked_partner_id = left.track_id
        engine.tracks = {left.track_id: left, right.track_id: right}
        engine.previous_hold_tails = [
            HoldTailDetection(0.24, 0.8, 100, 5, (890.0, 250.0), 0.0),
            HoldTailDetection(0.23, 0.8, 100, 3, (630.0, 250.0), 0.0),
        ]
        engine.previous_hold_tail_streaks = [1, 1]
        current = [
            HoldTailDetection(0.29, 0.8, 100, 5, (895.0, 258.0), 0.0),
            HoldTailDetection(0.30, 0.8, 100, 3, (625.0, 258.0), 0.0),
        ]
        frame = MusicFrame(10, 0.5, 0.5, 0.5, np.zeros((720, 1280, 3), dtype=np.uint8))
        with patch("agent.music.tracking.detect_hold_tails", return_value=current):
            engine._update_active_hold_tails(frame)
        self.assertEqual(left.hold_tail_observations[-1].center, (625.0, 258.0))
        self.assertEqual(right.hold_tail_observations[-1].center, (895.0, 258.0))

    def test_solitary_hold_prefers_continuous_cap_over_closer_lane_impostor(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=32, lane=2, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
        track.predicted_hit_time = 0.0
        track.hold_release_time = 1.8
        engine.tracks[track.track_id] = track
        engine.previous_hold_tails = [HoldTailDetection(0.40, 0.8, 100, 1, (400.0, 300.0), 0.0)]
        engine.previous_hold_tail_streaks = [1]
        real_cap = HoldTailDetection(0.50, 0.8, 100, 1, (380.0, 330.0), 0.0)
        lane_impostor = HoldTailDetection(0.40, 0.8, 100, 2, (520.0, 320.0), 0.0)
        frame = MusicFrame(10, 0.5, 0.5, 0.5, np.zeros((720, 1280, 3), dtype=np.uint8))
        with patch("agent.music.tracking.detect_hold_tails", return_value=[real_cap, lane_impostor]):
            engine._update_active_hold_tails(frame)
        self.assertEqual(track.hold_tail_observations[-1].center, real_cap.center)

    def test_verified_very_long_hold_reacquires_a_distant_final_cap(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=91, lane=3)
        track.state = TrackState.HOLDING
        track.gesture = NoteGesture.HOLD_START
        track.predicted_hit_time = 0.0
        track.hold_release_time = 50.0
        track.hold_long_verified = True
        track.hold_target_lane = 3
        track.hold_tail_observations.append(HoldTailObservation(1, 1.0, 0.20, 0.8, 100, 3, (640.0, 236.0)))
        engine.tracks[track.track_id] = track
        for sequence, progress in enumerate([0.72, 0.82, 0.92], start=20):
            timestamp = 14.0 + (sequence - 20) * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, tail_frame(cal, 3, progress))
            engine._update_active_hold_tails(frame)
        self.assertTrue(track.hold_tail_reacquired)
        self.assertTrue(track.hold_release_locked)
        self.assertLess(track.hold_release_time, 15.0)


    def test_contact_watchdog_does_not_cut_off_a_twenty_second_hold(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        executor.lanes[3] = LaneInputState(
            lane=3,
            contact=0,
            hold_track_id=8,
            contact_started=1.0,
        )
        executor.enforce_contact_limits(now=21.0)
        self.assertEqual(executor.hold_owner(3), 8)
        self.assertIn(3, executor.active_contacts)

    def test_single_direction_tail_schedules_move_without_changing_owner_contact(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        samples = [0.20, 0.30, 0.40, 0.55, 0.72]
        for sequence, head_progress in enumerate(samples):
            head = candidate_at(cal, 3, head_progress)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            x, y, width, height = head.box
            image[y : y + height, x : x + width] = [0, 165, 255]
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
            current = engine.update(frame, [head], visual)
            events.extend(current)
            acknowledge_hold_starts(engine, current)
            events.extend(engine.release_events(timestamp))
        for sequence, progress in enumerate([0.03, 0.10, 0.22, 0.38, 0.58, 0.78, 0.92], start=8):
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, tail_frame(cal, 4, progress))
            events.extend(engine.update(frame, [], visual))
            events.extend(engine.release_events(timestamp))
        moves = [event for event in events if event.gesture == NoteGesture.HOLD_CONTINUE]
        self.assertEqual(len(moves), 4)
        self.assertTrue(all(move.lane == 3 for move in moves))
        final = next(move for move in moves if move.event_id.startswith("move-"))
        route_moves = [move for move in moves if move.event_id.startswith("route-")]
        self.assertEqual(final.coordinate, tuple(cal.points[4]))
        self.assertEqual([move.event_id for move in route_moves], [
            f"route-{final.track_id}-1",
            f"route-{final.track_id}-2",
            f"route-{final.track_id}-3",
        ])
        self.assertTrue(all(cal.points[3][0] < move.coordinate[0] < cal.points[4][0] for move in route_moves))

    def test_center_color_note_uses_isolated_center_tap_track(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        for sequence, center_y in enumerate([180, 204, 228, 258, 290, 324]):
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, center_color_frame(center_y))
            events.extend(engine.update(frame, [], visual))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.gesture, NoteGesture.TAP)
        self.assertEqual(event.lane, 3)
        self.assertEqual(event.coordinate, tuple(cal.points[3]))
        self.assertLess(event.track_id, 0)
        self.assertIsNotNone(engine.center_color_track)
        event = engine.refine_pending([event], 0.5)[0]
        self.assertAlmostEqual(
            engine.center_color_track.predicted_hit_time - event.deadline,
            engine.config.center_tap_action_advance_ms / 1000.0,
        )

        # Once the large singleton has left, it must relinquish the centre
        # lane.  Otherwise later ordinary centre taps can be swallowed by this
        # already-executed special track.
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        for sequence in range(6, 10):
            timestamp = sequence * 0.1
            engine.update(MusicFrame(sequence, timestamp, timestamp, timestamp, blank), [], visual)
        self.assertIsNone(engine.center_color_track)

    def test_center_color_timing_adapts_to_fast_and_slow_fall_rates(self) -> None:
        cal = calibration()
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        def deadline(interval: float) -> float:
            engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
            pending = []
            for sequence, center_y in enumerate([180, 220, 265, 315, 365]):
                timestamp = sequence * interval
                frame = MusicFrame(sequence, timestamp, timestamp, timestamp, center_color_frame(center_y))
                pending.extend(engine.update(frame, [], visual))
                pending = engine.refine_pending(pending, timestamp)
            return pending[0].deadline

        self.assertLess(deadline(0.05), deadline(0.10))

    def test_route_steps_keep_observed_deadlines_during_refinement(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        track = NoteTrack(track_id=7, lane=3)
        track.hold_release_time = 5.0
        track.hold_target_lane = 4
        engine.tracks[7] = track
        route = MusicActionEvent("route-7-1", 7, 3, NoteGesture.HOLD_CONTINUE, 2.0, (700, 620), contact_policy="persistent")
        final = MusicActionEvent("move-7", 7, 3, NoteGesture.HOLD_CONTINUE, 2.0, (800, 620), contact_policy="persistent")
        refined = engine.refine_pending([route, final], 1.0)
        self.assertEqual(refined[0].deadline, 2.0)
        self.assertEqual(refined[1].deadline, 5.0 - engine.config.hold_move_advance_ms / 1000.0)
        self.assertEqual(refined[1].coordinate, tuple(cal.points[4]))
        track.hold_target_lane = 3
        corrected = engine.refine_pending(refined, 1.1)
        self.assertEqual(corrected[1].coordinate, tuple(cal.points[3]))

    def test_split_same_lane_hold_does_not_schedule_a_second_contact(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = []
        for sequence, progress in enumerate([0.20, 0.30, 0.42, 0.56, 0.72]):
            notes = [candidate_at(cal, 3, progress), candidate_at(cal, 3, progress - 0.10)]
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            for note in notes:
                x, y, width, height = note.box
                image[y : y + height, x : x + width] = [0, 165, 255]
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
            events.extend(engine.update(frame, notes, visual))
        starts = [event for event in events if event.gesture == NoteGesture.HOLD_START]
        self.assertEqual(len(starts), 1)

    def test_stale_same_lane_hold_is_suppressed_when_contact_intervals_overlap(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        current = NoteTrack(track_id=20, lane=2)
        current.state = TrackState.HOLDING
        current.action_executed = True
        current.predicted_hit_time = 10.0
        current.hold_release_time = 12.0
        stale = NoteTrack(track_id=1, lane=2)
        stale.gesture = NoteGesture.HOLD_START
        stale.speed = 0.5
        stale.predicted_hit_time = 8.0
        for sequence, progress in enumerate([0.40, 0.50, 0.60, 0.70]):
            note = candidate_at(cal, 2, progress)
            stale.observations.append(TrackObservation(sequence, 6.0 + sequence * 0.1, note.center, progress, note))
        engine.tracks = {20: current, 1: stale}
        frame = MusicFrame(4, 7.0, 7.0, 7.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        self.assertEqual(engine.update(frame, [], visual), [])
        self.assertEqual(stale.state, TrackState.LOST)

    def test_overage_approaching_track_cannot_schedule_after_stage_flash(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        stale = NoteTrack(track_id=1, lane=2, first_seen_time=1.0)
        stale.speed = 0.5
        stale.predicted_hit_time = 10.1
        for sequence, progress in enumerate([0.40, 0.52, 0.65, 0.78], start=96):
            note = candidate_at(cal, 2, progress)
            stale.observations.append(TrackObservation(sequence, 9.6 + (sequence - 96) * 0.1, note.center, progress, note))
        engine.tracks[stale.track_id] = stale
        frame = MusicFrame(100, 10.0, 10.0, 10.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        self.assertEqual(engine.update(frame, [], visual), [])
        self.assertEqual(stale.state, TrackState.LOST)
        self.assertFalse(stale.action_executed)

    def test_lost_linked_pair_cannot_resurface_seconds_late(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        tracks: list[NoteTrack] = []
        for track_id, lane, predicted in ((245, 2, 8.0), (249, 4, 32.0)):
            track = NoteTrack(track_id=track_id, lane=lane, state=TrackState.LOST)
            track.speed = 0.5
            track.predicted_hit_time = predicted
            for sequence, progress in enumerate([0.40, 0.52, 0.65, 0.78], start=97):
                note = candidate_at(cal, lane, progress)
                track.observations.append(
                    TrackObservation(sequence, 29.7 + (sequence - 97) * 0.1, note.center, progress, note)
                )
            tracks.append(track)
        tracks[0].linked_partner_id = tracks[1].track_id
        tracks[1].linked_partner_id = tracks[0].track_id
        engine.tracks = {track.track_id: track for track in tracks}
        frame = MusicFrame(100, 30.0, 30.0, 30.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        self.assertEqual(engine.update(frame, [], visual), [])
        self.assertTrue(all(not track.action_executed for track in tracks))

    def test_current_note_replaces_an_overage_static_track(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        stale = NoteTrack(track_id=1, lane=2, first_seen_time=1.0)
        old_note = candidate_at(cal, 2, 0.70)
        stale.observations.append(TrackObservation(9, 9.9, old_note.center, 0.70, old_note))
        stale.speed = 0.2
        stale.predicted_hit_time = 11.0
        engine.tracks[stale.track_id] = stale
        engine.next_track_id = 2
        current = candidate_at(cal, 2, 0.78)
        frame = MusicFrame(10, 10.0, 10.0, 10.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        self.assertEqual(engine.update(frame, [current], visual), [])
        self.assertEqual(stale.state, TrackState.LOST)
        replacement = engine.tracks[2]
        self.assertEqual(replacement.state, TrackState.APPROACHING)
        self.assertEqual(len(replacement.observations), 1)
        self.assertEqual(replacement.first_seen_time, 10.0)

    def test_old_terminal_tracks_are_pruned_from_per_frame_work(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=False))
        stale = NoteTrack(track_id=1, lane=2, state=TrackState.LOST)
        note = candidate_at(cal, 2, 0.70)
        stale.observations.append(TrackObservation(0, 0.0, note.center, 0.70, note))
        engine.tracks[stale.track_id] = stale
        frame = MusicFrame(30, 3.0, 3.0, 3.0, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask(mask=np.zeros((720, 1280), dtype=bool), roi_origin=(0, 0))

        engine.update(frame, [], visual)
        self.assertNotIn(stale.track_id, engine.tracks)

    def test_distinct_later_same_lane_hold_is_scheduled_for_runtime_handoff(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        owner = NoteTrack(track_id=20, lane=2, state=TrackState.HOLDING)
        owner.action_executed = True
        owner.predicted_hit_time = 8.0
        owner.hold_release_time = 20.0
        later = NoteTrack(track_id=21, lane=2, gesture=NoteGesture.HOLD_START)
        later.speed = 0.5
        later.predicted_hit_time = 10.0
        for sequence, progress in enumerate([0.40, 0.50, 0.60, 0.70]):
            note = candidate_at(cal, 2, progress)
            later.observations.append(TrackObservation(sequence, 6.7 + sequence * 0.1, note.center, progress, note))
        engine.tracks = {owner.track_id: owner, later.track_id: later}
        frame = MusicFrame(4, 7.1, 7.1, 7.1, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        events = engine.update(frame, [], visual)
        self.assertEqual([event.track_id for event in events if event.gesture == NoteGesture.HOLD_START], [21])
        self.assertEqual(later.state, TrackState.HOLD_PENDING)

    def test_runtime_discards_duplicate_touch_down_without_fusing_or_releasing_owner(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.0, sleeper=lambda _seconds: None)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        executor.lanes[2] = LaneInputState(lane=2, contact=0, hold_track_id=20, contact_started=0.5)
        engine = MusicVisionEngine(calibration(), config)
        stale = NoteTrack(track_id=1, lane=2)
        stale.state = TrackState.HOLDING
        stale.action_executed = True
        engine.tracks[1] = stale
        pending = [
            MusicActionEvent("stale-start", 1, 2, NoteGesture.HOLD_START, 1.0, (431, 624), contact_policy="persistent"),
            MusicActionEvent("stale-release", 1, 2, NoteGesture.HOLD_END, 2.0, (431, 624), contact_policy="persistent"),
        ]
        runtime._execute_due(executor, pending, 1.0, RuntimeMetrics(), engine)
        self.assertEqual(pending, [])
        self.assertEqual(stale.state, TrackState.LOST)
        self.assertEqual(executor.hold_owner(2), 20)
        self.assertTrue(executor.healthy)
        executor.touch_move(2, 500, 620, track_id=1)
        executor.touch_up(2, track_id=1)
        self.assertEqual(executor.hold_owner(2), 20)
        self.assertTrue(executor.healthy)

    def test_runtime_atomically_hands_a_lane_to_a_distinct_later_hold(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.0, sleeper=lambda _seconds: None)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        executor.lanes[2] = LaneInputState(lane=2, contact=0, hold_track_id=20, contact_started=0.5)
        engine = MusicVisionEngine(calibration(), config)
        owner = NoteTrack(track_id=20, lane=2, state=TrackState.HOLDING, predicted_hit_time=8.0)
        later = NoteTrack(track_id=21, lane=2, state=TrackState.HOLDING, predicted_hit_time=10.0)
        engine.tracks = {owner.track_id: owner, later.track_id: later}
        pending = [
            MusicActionEvent("later-start", 21, 2, NoteGesture.HOLD_START, 1.0, (480, 620), contact_policy="persistent"),
            MusicActionEvent("owner-release", 20, 2, NoteGesture.HOLD_END, 2.0, (480, 620), contact_policy="persistent"),
        ]

        def release_lane(lane: int, *, track_id: int | None = None) -> None:
            self.assertEqual((lane, track_id), (2, 20))
            executor.lanes[2].contact = None
            executor.lanes[2].hold_track_id = None

        def press_lane(lane: int, x: int, y: int, *, track_id: int | None = None) -> None:
            self.assertEqual((lane, x, y, track_id), (2, 480, 620, 21))
            executor.lanes[2].contact = 0
            executor.lanes[2].hold_track_id = track_id

        with patch.object(executor, "touch_up", side_effect=release_lane) as touch_up, patch.object(
            executor, "touch_down", side_effect=press_lane
        ) as touch_down:
            runtime._execute_due(executor, pending, 1.0, RuntimeMetrics(), engine)
        touch_up.assert_called_once_with(2, track_id=20)
        touch_down.assert_called_once_with(2, 480, 620, track_id=21)
        self.assertEqual(owner.state, TrackState.RELEASED)
        self.assertEqual(executor.hold_owner(2), 21)
        self.assertEqual(pending, [])

    def test_hold_start_uses_precision_window_without_changing_its_deadline(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.04, sleeper=lambda _seconds: None)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        engine = MusicVisionEngine(calibration(), config)
        engine.tracks[7] = NoteTrack(track_id=7, lane=3)
        pending = [
            MusicActionEvent("hold-start", 7, 3, NoteGesture.HOLD_START, 1.04, (640, 620), contact_policy="persistent"),
        ]
        with patch.object(executor, "touch_down") as touch_down:
            runtime._execute_due(executor, pending, 1.0, RuntimeMetrics(), engine)
        touch_down.assert_called_once_with(3, 640, 620, track_id=7)
        self.assertEqual(engine.tracks[7].state, TrackState.HOLDING)
        self.assertEqual(pending, [])

    def test_queued_hold_cannot_track_or_release_tail_before_touchdown_ack(self) -> None:
        cal = calibration()
        engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
        visual = VisualMask(mask=np.ones((720, 1280), dtype=bool), roi_origin=(0, 0))
        starts: list[MusicActionEvent] = []
        for sequence, progress in enumerate([0.30, 0.42, 0.56, 0.72]):
            head = candidate_at(cal, 5, progress)
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, frame_image(cal, head, None))
            starts.extend(engine.update(frame, [head], visual))
        start = next(event for event in starts if event.gesture == NoteGesture.HOLD_START)
        track = engine.tracks[start.track_id]
        self.assertEqual(track.state, TrackState.HOLD_PENDING)

        followups: list[MusicActionEvent] = []
        for sequence, progress in enumerate([0.40, 0.62, 0.84, 0.94], start=5):
            timestamp = sequence * 0.1
            frame = MusicFrame(sequence, timestamp, timestamp, timestamp, tail_frame(cal, 4, progress))
            followups.extend(engine.update(frame, [], visual))
            followups.extend(engine.release_events(timestamp))
        self.assertEqual(track.state, TrackState.HOLD_PENDING)
        self.assertEqual(len(track.hold_tail_observations), 0)
        self.assertFalse(any(event.gesture in {NoteGesture.HOLD_CONTINUE, NoteGesture.HOLD_END} for event in followups))

    def test_hold_move_uses_precision_window(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.11, sleeper=lambda _seconds: None)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        executor.lanes[5] = LaneInputState(lane=5, contact=0, hold_track_id=7, contact_started=0.5)
        engine = MusicVisionEngine(calibration(), config)
        engine.tracks[7] = NoteTrack(track_id=7, lane=5, state=TrackState.HOLDING)
        pending = [
            MusicActionEvent("move-7", 7, 5, NoteGesture.HOLD_CONTINUE, 1.11, (800, 620), contact_policy="persistent"),
        ]
        with patch.object(executor, "touch_move") as touch_move:
            runtime._execute_due(executor, pending, 1.0, RuntimeMetrics(), engine)
        touch_move.assert_called_once_with(5, 800, 620, track_id=7)
        self.assertEqual(pending, [])

    def test_compatibility_hold_fallback_does_not_leave_phantom_active_contact(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.0, sleeper=lambda _seconds: None)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=False)
        engine = MusicVisionEngine(calibration(), config)
        engine.tracks[7] = NoteTrack(track_id=7, lane=5, state=TrackState.HOLD_PENDING)
        pending = [
            MusicActionEvent("hold-start", 7, 5, NoteGesture.HOLD_START, 1.0, (960, 620), contact_policy="persistent"),
        ]
        with patch.object(executor, "tap_many") as tap_many:
            runtime._execute_due(executor, pending, 1.0, RuntimeMetrics(), engine)
        tap_many.assert_called_once()
        self.assertEqual(engine.tracks[7].state, TrackState.RELEASED)
        self.assertEqual(pending, [])

    def test_terminal_ocr_requires_prior_activity_and_full_quiet_window(self) -> None:
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7, enable_holds=True))
        base = dict(
            now=20.0,
            schedule_started=0.0,
            next_terminal_check=2.0,
            last_chart_activity=18.0,
        )
        self.assertFalse(runtime.terminal_ocr_ready(**base, note_activity_seen=False, chart_active=False))
        self.assertFalse(runtime.terminal_ocr_ready(**base, note_activity_seen=True, chart_active=True))
        self.assertTrue(runtime.terminal_ocr_ready(**base, note_activity_seen=True, chart_active=False))
        self.assertFalse(runtime.terminal_ocr_ready(
            **{**base, "last_chart_activity": 19.0},
            note_activity_seen=True,
            chart_active=False,
        ))

    def test_pending_hold_counts_as_chart_activity_before_touchdown(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        engine = MusicVisionEngine(calibration(), config)
        engine.tracks[7] = NoteTrack(track_id=7, lane=5, state=TrackState.HOLD_PENDING)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        self.assertTrue(MusicRuntime.chart_activity_present(engine, executor, [], 10))

    def test_screenshot_capture_timeout_and_stop_are_non_blocking(self) -> None:
        class NeverDoneJob:
            job_id = 77
            done = False
            succeeded = False

            def wait(self) -> object:
                raise AssertionError("bounded capture must not call wait()")

            def get(self) -> object:
                raise AssertionError("unfinished capture has no result")

        job = NeverDoneJob()
        controller = SimpleNamespace(post_screencap=lambda: job)
        stop_requests: list[bool] = []
        context = SimpleNamespace(tasker=SimpleNamespace(
            controller=controller,
            stopping=False,
            post_stop=lambda: stop_requests.append(True),
        ))
        with patch("agent.common.time.monotonic", side_effect=[0.0, 0.0, 0.002]), patch("agent.common.time.sleep"):
            self.assertIsNone(capture_image(context, timeout_ms=1, poll_interval_ms=1))
        self.assertEqual(stop_requests, [True])

        context.tasker.stopping = True
        with patch("agent.common.time.monotonic", return_value=0.0):
            self.assertIsNone(capture_image(context, timeout_ms=1000, poll_interval_ms=1))

    def test_special_and_dense_taps_use_isolated_precision_windows(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        engine = MusicVisionEngine(calibration(), config)
        bonus = NoteTrack(track_id=40, lane=2, bonus_star=True)
        dense_a = NoteTrack(track_id=41, lane=4)
        dense_b = NoteTrack(track_id=42, lane=4)
        engine.tracks = {40: bonus, 41: dense_a, 42: dense_b}

        def run(pending: list[MusicActionEvent]) -> None:
            runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.08, sleeper=lambda _seconds: None)
            executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=False)
            with patch.object(executor, "tap_many") as tap_many:
                runtime._execute_due(executor, pending, 1.0, RuntimeMetrics(), engine)
            tap_many.assert_called_once()

        run([MusicActionEvent("bonus", 40, 2, NoteGesture.TAP, 1.08, (480, 620))])
        run([MusicActionEvent("center-color-1", -1, 3, NoteGesture.TAP, 1.08, (640, 620))])
        dense_pending = [
            MusicActionEvent("dense-a", 41, 4, NoteGesture.TAP, 1.08, (800, 620)),
            MusicActionEvent("dense-b", 42, 4, NoteGesture.TAP, 1.50, (800, 620)),
        ]
        run(dense_pending)
        self.assertEqual([event.event_id for event in dense_pending], ["dense-b"])

    def test_isolated_ordinary_tap_uses_precision_window_without_moving_deadline(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.08, sleeper=lambda _seconds: None)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=False)
        pending = [MusicActionEvent("ordinary", 50, 3, NoteGesture.TAP, 1.08, (640, 620))]
        with patch.object(executor, "tap_many") as tap_many:
            runtime._execute_due(executor, pending, 1.0, RuntimeMetrics())
        tap_many.assert_called_once()
        self.assertEqual(pending, [])

    def test_stale_dispatch_timestamp_does_not_add_a_late_precision_sleep(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        sleeps: list[float] = []
        runtime = MusicRuntime(
            SimpleNamespace(),
            config,
            clock=lambda: 1.06,
            sleeper=sleeps.append,
        )
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=False)
        pending = [MusicActionEvent("ordinary", 51, 4, NoteGesture.TAP, 1.05, (800, 620))]

        with patch.object(executor, "tap_many") as tap_many:
            runtime._execute_due(executor, pending, 1.0, RuntimeMetrics())

        tap_many.assert_called_once()
        self.assertEqual(sleeps, [])
        self.assertEqual(pending, [])

    def test_cap_locked_and_fallback_releases_use_bounded_precision_windows(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        engine = MusicVisionEngine(calibration(), config)

        long_track = NoteTrack(track_id=8, lane=3)
        long_track.state = TrackState.HOLDING
        long_track.predicted_hit_time = 0.1
        long_track.hold_release_time = 3.04
        long_track.hold_release_locked = True
        engine.tracks[8] = long_track
        long_executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        long_executor.lanes[3] = LaneInputState(lane=3, contact=0, hold_track_id=8, contact_started=0.5)
        long_pending = [MusicActionEvent("long-end", 8, 3, NoteGesture.HOLD_END, 3.04, (640, 620), contact_policy="persistent")]
        long_runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 3.04, sleeper=lambda _seconds: None)
        with patch.object(long_executor, "touch_up") as touch_up:
            long_runtime._execute_due(long_executor, long_pending, 3.0, RuntimeMetrics(), engine)
        touch_up.assert_called_once_with(3, track_id=8)
        self.assertEqual(long_pending, [])

        verified_track = NoteTrack(track_id=10, lane=4)
        verified_track.state = TrackState.HOLDING
        verified_track.predicted_hit_time = 0.1
        verified_track.hold_release_time = 3.09
        verified_track.hold_long_verified = True
        engine.tracks[10] = verified_track
        verified_executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        verified_executor.lanes[4] = LaneInputState(lane=4, contact=0, hold_track_id=10, contact_started=0.5)
        verified_pending = [MusicActionEvent("verified-end", 10, 4, NoteGesture.HOLD_END, 3.09, (800, 620), contact_policy="persistent")]
        verified_runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 3.09, sleeper=lambda _seconds: None)
        with patch.object(verified_executor, "touch_up") as touch_up:
            verified_runtime._execute_due(verified_executor, verified_pending, 3.0, RuntimeMetrics(), engine)
        touch_up.assert_called_once_with(4, track_id=10)
        self.assertEqual(verified_pending, [])

        short_track = NoteTrack(track_id=9, lane=2)
        short_track.state = TrackState.HOLDING
        short_track.predicted_hit_time = 0.5
        short_track.hold_release_time = 1.04
        short_track.hold_release_locked = True
        engine.tracks[9] = short_track
        short_executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        short_executor.lanes[2] = LaneInputState(lane=2, contact=0, hold_track_id=9, contact_started=0.5)
        short_pending = [MusicActionEvent("short-end", 9, 2, NoteGesture.HOLD_END, 1.04, (480, 620), contact_policy="persistent")]
        short_runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.04, sleeper=lambda _seconds: None)
        with patch.object(short_executor, "touch_up") as touch_up:
            short_runtime._execute_due(short_executor, short_pending, 1.0, RuntimeMetrics(), engine)
        touch_up.assert_called_once_with(2, track_id=9)
        self.assertEqual(short_pending, [])

        fallback_track = NoteTrack(track_id=11, lane=1)
        fallback_track.state = TrackState.HOLDING
        fallback_track.predicted_hit_time = 0.5
        fallback_track.hold_release_time = 1.04
        engine.tracks[11] = fallback_track
        fallback_executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        fallback_executor.lanes[1] = LaneInputState(lane=1, contact=0, hold_track_id=11, contact_started=0.5)
        fallback_pending = [MusicActionEvent("fallback-end", 11, 1, NoteGesture.HOLD_END, 1.04, (320, 620), contact_policy="persistent")]
        fallback_now = [1.0]
        fallback_runtime = MusicRuntime(
            SimpleNamespace(),
            config,
            clock=lambda: fallback_now[0],
            sleeper=lambda seconds: fallback_now.__setitem__(0, fallback_now[0] + seconds + 0.006),
        )
        with patch.object(fallback_executor, "touch_up") as touch_up:
            fallback_runtime._execute_due(fallback_executor, fallback_pending, 1.0, RuntimeMetrics(), engine)
        touch_up.assert_called_once_with(1, track_id=11)
        self.assertEqual(fallback_pending, [])

    def test_imminent_event_is_serviced_before_starting_another_capture(self) -> None:
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.0)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True)
        engine = MusicVisionEngine(calibration(), config)
        imminent = [MusicActionEvent("tap", 1, 2, NoteGesture.TAP, 1.04, (431, 624))]

        with patch.object(runtime, "_execute_due") as execute_due:
            runtime._service_imminent_before_capture(executor, imminent, RuntimeMetrics(), engine)

        execute_due.assert_called_once()

        later = [MusicActionEvent("tap-later", 2, 2, NoteGesture.TAP, 1.06, (431, 624))]
        with patch.object(runtime, "_execute_due") as execute_due:
            runtime._service_imminent_before_capture(executor, later, RuntimeMetrics(), engine)

        execute_due.assert_not_called()

    def test_mid_song_pause_waits_for_two_live_frames_and_releases_contacts(self) -> None:
        context = PauseSequenceContext()
        runtime = MusicRuntime(context, MusicConfig(lane_count=7), sleeper=lambda _seconds: None)
        executor = ReleaseProbe()
        labels = ["paused", "transition", "live", "live"]
        frames = [MusicFrame(index, 0.0, 0.0, 0.0, label) for index, label in enumerate(labels)]
        with patch("agent.music.runtime._capture_frame", side_effect=[(frame, 1.0) for frame in frames]):
            frame, sequence, _paused_seconds, failure = runtime._wait_until_resumed(executor, 0, 7)  # type: ignore[arg-type]
        self.assertIsNone(failure)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.image, "live")
        self.assertEqual(sequence, 4)
        self.assertEqual(executor.releases, 1)
        self.assertFalse(any(node.startswith("MusicResult") for node, _image in context.called))

    def test_terminal_state_has_exactly_two_normal_finish_signals(self) -> None:
        context = StrictTerminalContext(loading=True, live=False)
        self.assertEqual(terminal_state(context, object()), "result")
        self.assertEqual(context.called, ["MusicResultLoading"])
        context = StrictTerminalContext(loading=False, live=True)
        self.assertEqual(terminal_state(context, object()), "result")
        self.assertEqual(context.called, ["MusicResultLoading", "MusicResultLive"])

    def test_giant_live_visual_requires_black_screen_white_card_and_black_title(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[305:415, 389:891] = 255
        # Four broad glyph groups are enough to exercise the strict ink extent;
        # OCR remains the independent semantic fallback in the real pipeline.
        for x0, x1 in ((510, 545), (570, 588), (620, 675), (710, 752)):
            image[318:402, x0:x1] = 0
        self.assertTrue(giant_live_title_present(image))
        bright_stage = image.copy()
        bright_stage[140:580, 80:1200] = np.maximum(bright_stage[140:580, 80:1200], 80)
        self.assertFalse(giant_live_title_present(bright_stage))
        thin_line = np.zeros_like(image)
        thin_line[355:365, 389:891] = 255
        self.assertFalse(giant_live_title_present(thin_line))
        context = StrictTerminalContext(loading=False, live=False)
        self.assertEqual(terminal_state(context, object()), "unknown")
        self.assertEqual(context.called, ["MusicResultLoading", "MusicResultLive"])

    def test_pipeline_enables_holds_only_for_play_and_uses_strict_finish_rois(self) -> None:
        nodes = json.loads((BRANCH_ROOT / "resource" / "base" / "pipeline" / "my_task.json").read_text(encoding="utf-8"))
        play = nodes["MusicPlayRun7"]["action"]["param"]["custom_action_param"]
        preflight = nodes["MusicPlayPreflight7"]["action"]["param"]["custom_action_param"]
        self.assertIs(play["enable_holds"], True)
        self.assertNotIn("enable_holds", preflight)
        self.assertEqual(nodes["MusicPlayRun7"]["timeout"], 600000)
        # MusicPlay already performs strict loading/LIVE confirmation.  A
        # successful return must terminate the user task immediately instead
        # of entering post-result nodes that can wait forever on the LIVE card.
        self.assertEqual(nodes["MusicPlayRun7"]["next"], [])
        loading = nodes["MusicResultLoading"]["recognition"]["param"]
        self.assertEqual(loading["roi"], [960, 540, 320, 180])
        self.assertEqual(loading["expected"], "^载入中.*$")
        live = nodes["MusicResultLive"]["recognition"]["param"]
        self.assertEqual(live["roi"], [360, 280, 560, 170])
        self.assertEqual(live["threshold"], 0.7)
        self.assertEqual(
            nodes["MusicTerminalState"]["recognition"]["param"]["any_of"],
            ["MusicResultLoading", "MusicResultLive"],
        )


if __name__ == "__main__":
    unittest.main()
