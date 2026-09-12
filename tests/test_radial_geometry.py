from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BRANCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH_ROOT))

from agent.music.calibration import _default_centerlines, _judgement_arc_origin
from agent.music.holds import HoldTailDetection
from agent.music.models import MusicCalibrationData, MusicConfig, MusicFrame, NoteGesture, NoteTrack
from agent.music.tracking import MusicVisionEngine, project_to_polyline


JUDGEMENT_POINTS = [
    (119, 349),
    (250, 515),
    (431, 624),
    (640, 663),
    (848, 625),
    (1030, 516),
    (1161, 350),
]


class RadialLaneGeometryTests(unittest.TestCase):
    def calibration(self) -> MusicCalibrationData:
        return MusicCalibrationData(
            version=4,
            lane_count=7,
            width=1280,
            height=720,
            points=[[x, y] for x, y in JUDGEMENT_POINTS],
            lane_centerlines=_default_centerlines(JUDGEMENT_POINTS, 720),
            corridor_widths=[76.0] * 7,
            trigger_progress=1.0,
            candidate_roi=[0, 100, 1280, 599],
            exclusion_rois=[],
            baseline_version="maes-music-v4-2026-08",
            action_advance_ms=125.0,
            color_lower=[[0, 45, 110]],
            color_upper=[[179, 255, 255]],
            candidate_min_pixels=12,
            hold_min_length=130.0,
            created_at="",
        )

    def test_calibrated_arc_recovers_one_common_upper_origin(self) -> None:
        origin = _judgement_arc_origin(JUDGEMENT_POINTS, 720)

        self.assertAlmostEqual(origin[0], 640.0, places=3)
        self.assertTrue(70.0 <= origin[1] <= 80.0)
        lines = _default_centerlines(JUDGEMENT_POINTS, 720)
        self.assertEqual({tuple(line[0]) for line in lines}, {origin})

    def test_equal_radial_fraction_has_equal_progress_on_every_lane(self) -> None:
        lines = _default_centerlines(JUDGEMENT_POINTS, 720)
        origin_x, origin_y = lines[0][0]

        for line, endpoint in zip(lines, JUDGEMENT_POINTS):
            fraction = 0.43
            point = (
                origin_x + (endpoint[0] - origin_x) * fraction,
                origin_y + (endpoint[1] - origin_y) * fraction,
            )
            progress, distance, _tangent = project_to_polyline(point, line)
            self.assertAlmostEqual(progress, fraction, places=6)
            self.assertAlmostEqual(distance, 0.0, places=6)

    def test_mirrored_lane_change_tails_lock_the_same_release_time(self) -> None:
        cal = self.calibration()

        def release(head_lane: int, target_lane: int) -> tuple[float, int | None]:
            engine = MusicVisionEngine(cal, MusicConfig(lane_count=7, enable_holds=True))
            track = NoteTrack(track_id=1, lane=head_lane, gesture=NoteGesture.HOLD_START)
            track.predicted_hit_time = 0.0
            engine.tracks[track.track_id] = track
            origin_x, origin_y = cal.lane_centerlines[target_lane][0]
            endpoint_x, endpoint_y = cal.points[target_lane]
            for sequence, progress in enumerate([0.40, 0.52, 0.65, 0.78, 0.91]):
                timestamp = sequence * 0.1
                center = (
                    origin_x + (endpoint_x - origin_x) * progress,
                    origin_y + (endpoint_y - origin_y) * progress,
                )
                engine._record_active_hold_tail(
                    track,
                    MusicFrame(
                        sequence,
                        timestamp,
                        timestamp,
                        timestamp,
                        np.zeros((720, 1280, 3), dtype=np.uint8),
                    ),
                    HoldTailDetection(progress, 0.8, 100, target_lane, center, 0.0),
                )
            self.assertTrue(track.hold_release_locked)
            self.assertIsNotNone(track.hold_release_time)
            return float(track.hold_release_time), track.hold_target_lane

        left_to_right, right_target = release(2, 4)
        right_to_left, left_target = release(4, 2)
        self.assertEqual(right_target, 4)
        self.assertEqual(left_target, 2)
        self.assertAlmostEqual(left_to_right, right_to_left, places=6)


if __name__ == "__main__":
    unittest.main()
