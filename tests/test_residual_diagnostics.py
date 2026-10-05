"""Host/source diagnostics do not change action timing or infer game grades."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.executor import TapInputReceipt
from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture, TrackObservation
from agent.music.runtime import RuntimeMetrics
from agent.music.sustain import SustainMarker, SustainObservation
from agent.music.tap_trace import TapTrace
import test_flick_runtime_v4 as flick_tests
from test_point_head_identity import point_track


class ResidualDiagnosticsTests(unittest.TestCase):
    def setup_runtime(self):
        return flick_tests.FlickRuntimeV4Tests().setup_runtime(tap_mode=True)

    def test_input_distinguishes_birth_latest_source_deadline_and_native_call(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        track = point_track([(9., .50), (9.90, .62), (9.95, .70)])
        track.observations[-1] = replace(track.observations[-1], capture_started=9.94, capture_finished=9.96)
        engine = SimpleNamespace(tracks={track.track_id: track}, config=runtime.config, tap_hold_chain=None)
        event = MusicActionEvent('diagnostic', track.track_id, track.lane, NoteGesture.TAP, 9.98,
            (640, 600), source_capture_started=9.50, source_capture_finished=9.52)
        receipt = TapInputReceipt('diagnostic', track.lane, 0, down_call_started=10.,
            down_call_finished=10.008, up_call_finished=10.020)
        runtime._acknowledge_taps([receipt], [event], engine, metrics)
        row = next(r for r in runtime.tap_trace.records if r['kind'] == 'input')
        self.assertEqual(row['first_seen_time'], 9.)
        self.assertEqual(row['latest_center'], track.observations[-1].center)
        self.assertEqual(row['latest_capture_started'], 9.94)
        self.assertEqual(row['latest_visual_time'], 9.95)
        self.assertAlmostEqual(row['visual_age_ms'], 50.)
        self.assertAlmostEqual(row['deadline_lateness_ms'], 20.)
        self.assertAlmostEqual(metrics.latest_observation_to_action[0], 50.)
        self.assertAlmostEqual(metrics.latest_capture_to_action[0], 40.)
        # Backwards-compatible event-source metric stays distinct, not renamed.
        self.assertAlmostEqual(metrics.perception_to_action[0], 500.)
        self.assertEqual(event.deadline, 9.98)

    def test_unknown_latest_source_never_becomes_zero_latency(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        receipt = TapInputReceipt('unknown', 2, 0, down_call_started=10., down_call_finished=10.01)
        event = MusicActionEvent('unknown', 1, 2, NoteGesture.TAP, 10., (640, 600))
        runtime._acknowledge_taps([receipt], [event], None, metrics)
        row = next(r for r in runtime.tap_trace.records if r['kind'] == 'input')
        self.assertIsNone(row['latest_visual_time'])
        self.assertIsNone(row['latest_capture_finished'])
        self.assertIsNone(row['visual_age_ms'])
        self.assertEqual(len(metrics.latest_capture_to_action), 0)
        self.assertEqual(metrics.missing_latest_source_times, 1)

    def test_gold_first_seen_survives_twelve_sample_window(self):
        marker = SustainMarker(1)
        for i in range(20):
            marker.observe(SustainObservation(5.+i*.04, i, .2+i*.03, 3,
                (640., 300.+i*10), 1., 100, 2, 'checkpoint'))
        self.assertEqual(marker.first_seen_time, 5.)
        self.assertEqual(marker.observations[0].timestamp, 5.32)

    def test_recovery_search_cannot_evict_input_receipts(self):
        trace = TapTrace(MusicConfig(), capacity=4)
        trace.add('input', event='protected')
        for i in range(20):
            trace.add('head_recovery_search', track=1, time=i*.02, reason='no-owned-positive-contour')
        self.assertTrue(any(r['kind'] == 'input' for r in trace.records))
        self.assertEqual(trace.critical_dropped, 0)
        self.assertGreater(trace.visual_sampled_out, 0)

    def test_flick_lane_block_logs_reason_without_dispatch_or_retiming(self):
        clock, context, runtime, _, metrics = self.setup_runtime()
        executor = SimpleNamespace(active_flick_lanes={3}, available_flick_contacts=2,
            supports_multi_touch=True, supports_holds=True, may_open_contact_during_gesture=lambda: False)
        event = MusicActionEvent('blocked', 1, 3, NoteGesture.TAP, 9.98, (640, 600))
        pending = [event]
        for _ in range(40):
            runtime._flush_due_taps(executor, pending, None, metrics)
            clock.sleep(.002)
        rows = [r for r in runtime.tap_trace.records if r['kind'] == 'tap_dispatch_blocked']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['reason'], 'active-flick-lane')
        self.assertEqual(rows[0]['occupied_lanes'], [3])
        self.assertEqual(pending, [event])
        self.assertFalse(context.calls)
        self.assertEqual(event.deadline, 9.98)

    def test_recent_and_whole_statistics_name_source_clock_semantics(self):
        metrics = RuntimeMetrics()
        for i in range(80):
            metrics.latest_capture_to_action.append(float(i))
            metrics.head_recovery.append(.2)
        result = metrics.summaries([])
        self.assertEqual(result['whole_run']['latest_capture_to_action']['count'], 80)
        self.assertEqual(len(metrics.latest_capture_to_action), 60)
        self.assertIn('host', result['diagnostics']['time_semantics'])


if __name__ == '__main__':
    unittest.main()
