"""Production motion-gate and marker identity regressions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_sustain_chain import calibration, make_frame
from agent.music.holds import HoldTailDetection
from agent.music.models import MusicConfig, NoteGesture, NoteTrack, TrackState
from agent.music.tracking import MusicVisionEngine
from agent.music.hold_marker_identity import HoldMarkerMotion, MarkerAssociationRequest


def tail(y: float, progress: float = .5, lane: int = 3, x: float = 640.) -> HoldTailDetection:
    return HoldTailDetection(progress, .8, 100, lane, (x, y), 0., 2, 'checkpoint', None)


class MarkerIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MusicVisionEngine(calibration(), MusicConfig(lane_count=7, enable_holds=True))
        self.owner = NoteTrack(5, 3, gesture=NoteGesture.HOLD_START,
                               state=TrackState.HOLDING, predicted_hit_time=0.)
        self.engine.tracks[5] = self.owner

    def feed(self, current: list[HoldTailDetection], streak: int) -> list[int]:
        self.engine.previous_hold_tails = [tail(t.center[1] - 1., t.progress - .01, t.lane, t.center[0])
                                           for t in current]
        self.engine.previous_hold_tail_streaks = [streak] * len(current)
        moving, _, _, ids = self.engine._moving_hold_tails(current)
        self.engine._track_sustain_markers(make_frame(.3, 3), current, moving, ids,
                                           [self.owner], lambda *_: True)
        return ids

    def test_slow_mature_marker_uses_frame_count_not_pixel_distance(self) -> None:
        self.feed([tail(301.)], 4)
        self.assertEqual(len(self.engine.sustain_tracker.markers), 1)

    def test_fast_first_step_is_not_multiple_confirmed_frames(self) -> None:
        self.engine.previous_hold_tails = [tail(290., .48)]
        self.engine.previous_hold_tail_streaks = [0]
        current = [tail(300.)]
        moving, _, _, ids = self.engine._moving_hold_tails(current)
        self.engine._track_sustain_markers(make_frame(.3, 3), current, moving, ids,
                                           [self.owner], lambda *_: True)
        self.assertEqual(self.engine.sustain_tracker.markers, {})

    def established(self, marker: int = 20) -> None:
        self.engine.sustain_tracker.observe(marker, tail(290., .45), make_frame(.1, 1), owner=5)
        self.engine.sustain_tracker.observe(marker, tail(300., .5), make_frame(.2, 2), owner=5)

    def test_two_rings_cannot_heal_to_same_marker_in_one_frame(self) -> None:
        self.established()
        current = [tail(310., .55), tail(320., .60)]
        self.engine.previous_hold_tails = [tail(300., .5), tail(310., .55)]
        self.engine.previous_hold_tail_streaks = [1, 1]
        self.engine.previous_hold_tail_ids = [101, 102]
        moving, _, _, ids = self.engine._moving_hold_tails(current)
        self.engine._track_sustain_markers(make_frame(.3, 3), current, moving, ids,
                                           [self.owner], lambda *_: True)
        markers = self.engine.sustain_tracker.markers
        self.assertEqual(len(markers[20].observations), 3)
        self.assertEqual(len(markers), 2)
        self.assertEqual(len(set(ids)), 2)

    def test_healed_identity_is_written_back_for_next_frame(self) -> None:
        self.established()
        self.engine.previous_hold_tails = [tail(300.)]
        self.engine.previous_hold_tail_streaks = [1]
        self.engine.previous_hold_tail_ids = [101]
        current = [tail(310., .55)]
        moving, _, _, ids = self.engine._moving_hold_tails(current)
        self.engine._track_sustain_markers(make_frame(.3, 3), current, moving, ids,
                                           [self.owner], lambda *_: True)
        self.assertEqual(ids, [20])
        self.assertEqual(self.engine.previous_hold_tail_ids, [20])
        self.assertEqual(self.engine.sustain_tracker.resolve_id(101), 20)

    def test_motion_evidence_names_fields_without_changing_values(self) -> None:
        self.engine.previous_hold_tails = [tail(300.)]
        self.engine.previous_hold_tail_streaks = [4]
        moving, streaks, _, _ = self.engine._moving_hold_tails([tail(301., .51)])
        self.assertEqual(moving[0], HoldMarkerMotion(1., 1., 5))
        self.assertEqual(streaks, [5])

    def test_repeated_frame_cannot_add_second_observation(self) -> None:
        self.established()
        tracker = self.engine.sustain_tracker
        tracker.observe(20, tail(310., .55), make_frame(.3, 3), owner=5)
        tracker.observe(20, tail(320., .60), make_frame(.3, 3), owner=5)
        self.assertEqual(len(tracker.markers[20].observations), 3)
        self.assertEqual(tracker.markers[20].observations[-1].progress, .55)

    def test_raw_candidate_order_does_not_invert_two_markers(self) -> None:
        self.established(20)
        tracker = self.engine.sustain_tracker
        tracker.observe(21, tail(305., .52), make_frame(.1, 1), owner=5)
        tracker.observe(21, tail(315., .57), make_frame(.2, 2), owner=5)
        requests = [MarkerAssociationRequest(0, 102, tail(325., .62), 5),
                    MarkerAssociationRequest(1, 101, tail(310., .55), 5)]
        matches = tracker.associate_frame(requests, make_frame(.3, 3), eligible_owners={5})
        self.assertEqual(matches, {1: 20, 0: 21})
        self.assertEqual(tracker.resolve_id(101), 20)
        self.assertEqual(tracker.resolve_id(102), 21)

    def test_two_missed_frames_recover_curve_and_original_raw_alias(self) -> None:
        self.established()
        tracker = self.engine.sustain_tracker
        request = MarkerAssociationRequest(0, 101, tail(320., .60, lane=4, x=680.), 5)
        matches = tracker.associate_frame([request], make_frame(.45, 5), eligible_owners={5})
        self.assertEqual(matches[0], 20)
        tracker.observe(matches[0], request.detection, make_frame(.45, 5), owner=5)
        tracker.observe(101, tail(330., .63, lane=4, x=690.), make_frame(.5, 6), owner=5)
        self.assertEqual(set(tracker.markers), {20})
        self.assertEqual(len(tracker.markers[20].observations), 4)

    def test_completed_owner_cannot_heal_or_redirect_its_marker(self) -> None:
        self.established()
        tracker = self.engine.sustain_tracker
        request = MarkerAssociationRequest(0, 20, tail(310., .55), 6)
        matches = tracker.associate_frame([request], make_frame(.3, 3), eligible_owners={6})
        self.assertNotEqual(matches[0], 20)
        self.assertEqual(tracker.resolve_id(20), 20)
        self.assertEqual(tracker.markers[20].owner, 5)

    def test_unowned_moving_marker_is_kept_for_later_owner_evidence(self) -> None:
        tracker = self.engine.sustain_tracker
        request = MarkerAssociationRequest(0, 101, tail(310., .55), None)
        matches = tracker.associate_frame([request], make_frame(.3, 3), eligible_owners={5})
        marker = tracker.observe(matches[0], request.detection, make_frame(.3, 3))
        self.assertIsNone(marker.owner)
        follow = MarkerAssociationRequest(0, 101, tail(320., .6), 5)
        next_matches = tracker.associate_frame([follow], make_frame(.4, 4), eligible_owners={5})
        tracker.observe(next_matches[0], follow.detection, make_frame(.4, 4), owner=5)
        self.assertEqual(len(tracker.markers), 1)
        self.assertEqual(marker.owner, 5)

    def test_far_or_old_marker_does_not_capture_new_ring(self) -> None:
        self.established()
        tracker = self.engine.sustain_tracker
        for moment, current in [(.3, tail(400., .65)), (.9, tail(310., .55))]:
            request = MarkerAssociationRequest(0, 101, current, 5)
            matches = tracker.associate_frame([request], make_frame(moment, 3), eligible_owners={5})
            self.assertNotEqual(matches[0], 20)

    def test_aliases_are_bounded_by_marker_lifetime(self) -> None:
        self.established()
        tracker = self.engine.sustain_tracker
        tracker.associate_frame([MarkerAssociationRequest(0, 101, tail(310., .55), 5)],
                                 make_frame(.3, 3), eligible_owners={5})
        self.assertEqual(tracker.resolve_id(101), 20)
        tracker.prune(3.)
        self.assertEqual(tracker.marker_aliases, {})

    def update(self, sequence: int, moment: float, detections: list[HoldTailDetection]) -> None:
        with patch('agent.music.tracking.detect_hold_tails', return_value=detections):
            self.engine._update_active_hold_tails(make_frame(moment, sequence))

    def test_production_entry_recovers_after_two_empty_frames(self) -> None:
        for seq in range(1, 6):
            self.update(seq, seq * .05, [tail(290. + seq * 5., .3 + seq * .025)])
        marker_id = next(iter(self.engine.sustain_tracker.markers))
        self.update(6, .30, [])
        self.update(7, .35, [])
        for seq in range(8, 11):
            self.update(seq, seq * .05, [tail(290. + seq * 5., .3 + seq * .025)])
        self.assertEqual(set(self.engine.sustain_tracker.markers), {marker_id})
        self.assertEqual(self.engine.previous_hold_tail_ids, [marker_id])
        self.assertEqual(self.engine.sustain_tracker.markers[marker_id].last_seen_frame, 10)

    def test_production_entry_nearby_active_owners_cannot_steal_existing_marker(self) -> None:
        second = NoteTrack(6, 3, gesture=NoteGesture.HOLD_START,
                           state=TrackState.HOLDING, predicted_hit_time=0.)
        self.engine.tracks[6] = second
        tracker = self.engine.sustain_tracker
        for seq, y in [(1, 290.), (2, 300.)]:
            tracker.observe(20, tail(y, .4 + seq * .025, x=630.), make_frame(seq * .05, seq), owner=5)
            tracker.observe(21, tail(y + 5., .41 + seq * .025, x=650.), make_frame(seq * .05, seq), owner=6)
        self.engine.previous_hold_tails = [tail(300., .45, x=630.), tail(305., .46, x=650.)]
        self.engine.previous_hold_tail_streaks = [3, 3]
        self.engine.previous_hold_tail_ids = [20, 21]
        self.update(3, .15, [tail(310., .5, x=635.), tail(315., .51, x=645.)])
        self.assertEqual(tracker.markers[20].owner, 5)
        self.assertEqual(tracker.markers[21].owner, 6)
        self.assertEqual(tracker.markers[21].last_seen_frame, 3)
        self.assertEqual(set(tracker.markers), {20, 21})

    def test_production_entry_folded_route_keeps_id_on_destination_ray(self) -> None:
        self.owner.hold_target_lane = 4
        self.owner.hold_route_lanes = [3, 4]
        path = [(640., 290., 3), (648., 300., 3), (656., 310., 3),
                (664., 320., 4), (664., 330., 4), (664., 340., 4)]
        for seq, (x, y, lane) in enumerate(path, 1):
            self.update(seq, seq * .05, [tail(y, .3 + seq * .025, lane=lane, x=x)])
        self.assertEqual(len(self.engine.sustain_tracker.markers), 1)
        state = next(iter(self.engine.sustain_tracker.markers.values()))
        self.assertEqual(state.owner, 5)
        self.assertEqual(state.lane(), 4)
        self.assertEqual(state.last_seen_frame, 6)


if __name__ == '__main__':
    unittest.main()
