"""Unified point scheduling keeps visual/timing adapters, never hold state."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_hold_note_taps import calibration, make_frame
from test_tap_pipeline import observed_track
from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture, TrackState
from agent.music.tracking import MusicVisionEngine
from agent.music.vision import VisualMask
from agent.music.tap_policy import TapTimingPolicy
from agent.music.runtime import MusicRuntime, MusicCandidateError


class PointIntegrationTests(unittest.TestCase):
    def head_engine(self):
        engine = MusicVisionEngine(calibration(), MusicConfig(
            lane_count=7, enable_holds=True, hold_notes_as_taps=True,
            hold_start_action_advance_ms=175))
        track = observed_track(1, 3, .8, hit=1.2)
        track.point_mode = True
        track.visual_family = track.timing_profile = 'yellow_head'
        track.gesture = NoteGesture.HOLD_START
        engine.tracks[1] = track
        engine.next_track_id = 2
        return engine, track

    def schedule(self, engine):
        frame = make_frame(1., 2)
        with patch.object(engine, '_associate_lane'), patch.object(engine, '_ready_to_schedule', return_value=True), \
                patch.object(engine, '_update_active_hold_tails'), \
                patch('agent.music.tracking.detect_bonus_star_notes', return_value=[]), \
                patch.object(engine, '_update_center_color_note', return_value=(None, [])):
            return engine.update(frame, [], VisualMask.from_image(frame.image, engine.calibration))

    def test_yellow_head_is_tap_without_persistent_state(self):
        engine, track = self.head_engine()
        events = self.schedule(engine)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].gesture, NoteGesture.TAP)
        self.assertEqual(events[0].contact_policy, 'auto')
        self.assertEqual(track.state, TrackState.TAP_PENDING)
        self.assertAlmostEqual(events[0].deadline, 1.025)
        self.assertEqual(events[0].timing_profile, 'yellow_head')
        self.assertIsNotNone(events[0].physical_id)

    def test_yellow_profile_preserves_advance_and_wait_window(self):
        engine, track = self.head_engine()
        track.dense_tap = True
        event = MusicActionEvent('head', 1, 3, NoteGesture.TAP, 1., (640, 620),
                                 visual_family='yellow_head', timing_profile='yellow_head')
        self.assertEqual(engine.tap_policy.action_advance_ms(track), 175.)
        self.assertEqual(engine.tap_policy.execution_window_ms(event, track, [event]), 100.)

    def test_new_track_opts_into_point_mode_but_legacy_does_not(self):
        engine, _ = self.head_engine()
        self.assertTrue(engine._new_track(0, 1.).point_mode)
        legacy = MusicVisionEngine(calibration(), MusicConfig(lane_count=7))
        self.assertFalse(legacy._new_track(0, 1.).point_mode)

    def test_healthy_outer_point_has_same_coast_right_as_middle(self):
        engine, _ = self.head_engine()
        track = observed_track(10, 0, .7, hit=1.2)
        track.point_mode = True
        track.visual_family = track.timing_profile = 'ordinary'
        engine.tracks = {10: track}
        self.assertTrue(engine._coast_eligible(track, make_frame(1.05, 4)))

    def test_yellow_queue_is_not_retired_before_actual_down(self):
        from agent.music.association import associate_lane
        engine, track = self.head_engine()
        track.state, track.action_executed = TrackState.TAP_PENDING, True
        frame = make_frame(1.5, 7)
        associate_lane(engine, track.lane, [], frame,
                       VisualMask.from_image(frame.image, engine.calibration))
        self.assertEqual(track.state, TrackState.TAP_PENDING)

    def test_head_wait_does_not_outlive_existing_age_budget(self):
        engine, track = self.head_engine()
        runtime = MusicRuntime(SimpleNamespace(), engine.config)
        latest = track.observations[-1].timestamp
        deadline = latest + engine.config.coast_max_age_ms/1000. + .03
        event = MusicActionEvent('head', 1, 3, NoteGesture.TAP, deadline, (640, 620),
                                 physical_id='point-1', timing_profile='yellow_head')
        self.assertTrue(runtime._point_wait_needs_refresh([event], engine, deadline, deadline-.05))
        self.assertEqual(event.deadline, deadline)
        self.assertFalse(runtime._point_wait_needs_refresh([event], engine, deadline, deadline))

    def test_provider_fault_is_distinct_from_tracking_fault(self):
        engine, _ = self.head_engine()
        runtime = MusicRuntime(SimpleNamespace(), engine.config)
        runtime.calibration = engine.calibration
        runtime.provider = SimpleNamespace(detect=lambda *args: [])
        frame = make_frame(1., 5)
        with patch.object(engine, 'update', side_effect=ValueError('tracking fault')):
            with self.assertRaisesRegex(ValueError, 'tracking fault'):
                runtime._observe_frame(engine, frame)
        with patch.object(runtime.provider, 'detect', side_effect=ValueError('provider fault')):
            with self.assertRaisesRegex(MusicCandidateError, 'provider fault'):
                runtime._observe_frame(engine, frame)

    def test_real_loop_updates_new_capture_before_point_dispatch_only_once(self):
        from test_optimization import ProductionLoopReplayTests
        calls = []
        original = MusicRuntime._qualify_pending
        def qualify(runtime, pending, engine, now):
            if engine is not None and any(e.physical_id is not None for e in pending):
                frame = getattr(runtime, '_dispatch_frame', None)
                calls.append((frame.sequence if frame else None, engine.last_frame_sequence))
            return original(runtime, pending, engine, now)
        with patch.object(MusicRuntime, '_qualify_pending', qualify):
            result = ProductionLoopReplayTests().replay()
        self.assertTrue(calls)
        self.assertTrue(all(captured == observed for captured, observed in calls))
        seen = [(row['segment'], row['sequence']) for row in result['observations']]
        self.assertEqual(len(seen), len(set(seen)))

    def test_unqueued_point_rejections_have_identity_and_bounded_sampling(self):
        engine, track = self.head_engine()
        frame = make_frame(1., 5)
        for _ in range(30):
            self.assertFalse(engine._point_rejected(track, frame, 'head-birth-motion-unqualified'))
        records = [r for r in engine.tap_trace.records if r['kind'] == 'point_qualification']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['physical_id'], track.physical_id)
        self.assertEqual(records[0]['family'], 'yellow_head')
        self.assertNotIn('event', records[0])


if __name__ == '__main__':
    unittest.main()
