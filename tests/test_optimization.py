"""Regressions for the staged optimization candidate, never a live controller."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.music.build_identity import create_manifest, verify_manifest
from agent.music.executor import MusicActionExecutor, MusicTouchError, TapInputReceipt
from agent.music.models import MusicConfig, LaneInputState


class BuildIdentityTests(unittest.TestCase):
    def test_source_manifest_is_deterministic_and_detects_tampering(self):
        root = Path(__file__).resolve().parents[1]
        first = create_manifest(root)
        self.assertEqual(first, create_manifest(root))
        verify_manifest(root, first)
        first['files']['interface.json'] = 'bad'
        with self.assertRaises(ValueError):
            verify_manifest(root, first)


class InputLifecycleTests(unittest.TestCase):
    def executor(self):
        return MusicActionExecutor(SimpleNamespace(), 1280, 720,
                                   MusicConfig(lane_count=7, enable_holds=True),
                                   advanced=True, multi_touch=True)

    def test_pause_segment_allows_new_special_note_but_not_duplicate(self):
        for name in ('center-color-1', 'holdnote-1-1'):
            executor = self.executor()
            with patch.object(executor, '_run'):
                executor.begin_segment('run', 0)
                self.assertEqual(len(executor.tap(3, 640, 620, event_id=name)), 1)
                self.assertEqual(executor.tap(3, 640, 620, event_id=name), [])
                executor.release_all()
                executor.begin_segment('run', 1)
                receipt = executor.tap(3, 640, 620, event_id=name)[0]
                self.assertEqual((receipt.run_id, receipt.segment_id), ('run', 1))

    def test_segment_cannot_advance_with_unreleased_contact(self):
        executor = self.executor()
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=1)
        with self.assertRaises(MusicTouchError):
            executor.begin_segment('run', 1)

    def test_failed_lane_release_keeps_contact_and_reports_when_fused(self):
        executor = self.executor()
        executor.healthy = False
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=1)
        with patch.object(executor, '_run', side_effect=MusicTouchError('native failed')) as run:
            with self.assertRaises(MusicTouchError):
                executor.release_all()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(executor.active_contacts, {3: 0})
        with patch.object(executor, '_run'):
            executor.release_all()
        self.assertEqual(executor.active_contacts, {})

    def test_cleanup_retries_up_only(self):
        executor = self.executor()
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=1)
        with patch.object(executor, '_run', side_effect=[MusicTouchError('first'), 0]) as run:
            executor.release_all()
        self.assertEqual([call.args[0].value for call in run.call_args_list], ['TouchUp', 'TouchUp'])
        self.assertFalse(executor.active_contacts)

    def test_old_receipt_cannot_ack_new_segment(self):
        from agent.music.runtime import MusicRuntime, RuntimeMetrics
        from test_tap_pipeline import tap_event
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7))
        runtime.tap_trace.segment_id = 1
        old = TapInputReceipt('1', 3, 0, 1., 1.001, 1.002, run_id=runtime.tap_trace.run_id, segment_id=0)
        runtime._acknowledge_taps([old], [tap_event(1, 3, 1.)], None, RuntimeMetrics())
        self.assertFalse(runtime.head_action_trace)

    def test_paused_strict_result_needs_two_confirmations(self):
        from agent.music.runtime import MusicRuntime
        from agent.music.models import MusicFrame
        from test_longtap_branch import ReleaseProbe
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7), sleeper=lambda _: None)
        frame = MusicFrame(0, 0., 0., 0., None)
        with patch('agent.music.runtime._capture_frame', return_value=(frame, 0.)), \
             patch('agent.music.runtime.terminal_state', return_value='result'):
            resumed, sequence, _, result = runtime._wait_until_resumed(ReleaseProbe(), 0, 7)
        self.assertIsNone(resumed)
        self.assertEqual(sequence, 2)
        self.assertEqual(result.status, 'succeeded')

    def test_wait_slices_observe_cancel(self):
        from agent.music.runtime import MusicRuntime, MusicCancelled
        sleeps = []
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(), sleeper=sleeps.append)
        with patch('agent.music.runtime.is_stopping', side_effect=[False, False, True]):
            with self.assertRaises(MusicCancelled):
                runtime._sleep_interruptibly(.1)
        self.assertEqual(sleeps, [.01, .01])
