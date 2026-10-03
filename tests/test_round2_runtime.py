"""Production budgets and bounded diagnostics, independent of game grades."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture
from agent.music.runtime import MusicRuntime, RuntimeMetrics
from agent.music.tap_trace import TapTrace
from test_tap_pipeline import tap_event
from test_hold_note_taps import tap_engine, add_marker
from agent.music.executor import MusicActionExecutor, TapInputReceipt
from agent.music.models import TrackState


class Clock:
    now = 1.
    def __call__(self):
        self.now += .00001
        return self.now
    def sleep(self, duration):
        self.now += duration


class Round2RuntimeTests(unittest.TestCase):
    def test_mixed_queue_cannot_extend_tap_blind_wait(self):
        for gesture in (NoteGesture.HOLD_START, NoteGesture.HOLD_CONTINUE, NoteGesture.HOLD_END):
            with self.subTest(gesture=gesture):
                clock = Clock()
                runtime = MusicRuntime(SimpleNamespace(), MusicConfig(hold_notes_as_taps=True),
                                       clock=clock, sleeper=clock.sleep)
                pending = [tap_event(1, 3, 1.08),
                           MusicActionEvent('hold', 2, 4, gesture, 1.10, (800, 620))]
                executor = SimpleNamespace(tap_many=lambda *a, **k: [], supports_holds=True, async_input=False)
                with patch.object(runtime, '_acknowledge_taps'), patch.object(runtime, '_dispatch_due_event'):
                    runtime._execute_due(executor, pending, 1., RuntimeMetrics(), tap_wait_ms=45.)
                self.assertEqual(len(pending), 2)
                self.assertLess(clock.now, 1.001)

    def test_due_tap_runs_even_with_future_hold(self):
        clock = Clock()
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(), clock=clock, sleeper=clock.sleep)
        pending = [tap_event(1, 3, .99),
                   MusicActionEvent('hold', 2, 4, NoteGesture.HOLD_START, 1.10, (800, 620))]
        executor = SimpleNamespace(tap_many=lambda *a, **k: [], supports_holds=True, async_input=False)
        with patch.object(runtime, '_acknowledge_taps'):
            runtime._execute_due(executor, pending, 1., RuntimeMetrics(), tap_wait_ms=45.)
        self.assertEqual([event.event_id for event in pending], ['hold'])

    def test_visual_noise_cannot_evict_critical_input(self):
        trace = TapTrace(MusicConfig(), capacity=4)
        trace.add('input', time=1., event='first')
        for i in range(100):
            trace.add('flick_detected', time=2. + i, frame=i, box=(10, 10, 20, 20))
        self.assertTrue(any(r.get('event') == 'first' for r in trace.records))
        self.assertEqual(trace.critical_dropped, 0)
        self.assertLessEqual(len(trace.records), 8)

    def test_critical_overflow_is_reported_separately(self):
        trace = TapTrace(MusicConfig(), capacity=2)
        for i in range(3):
            trace.add('input', time=float(i), event=str(i))
        self.assertEqual(trace.critical_dropped, 1)
        self.assertEqual(trace.visual_dropped, 0)
        self.assertEqual(trace.dropped, 1)

    def marker_setup(self, *, exits=2):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 3, hit=2., now=1.8, exits=exits)
        event = engine.release_events(1.8)[0]
        clock = Clock()
        clock.now = 1.82
        runtime = MusicRuntime(SimpleNamespace(), engine.config, clock=clock, sleeper=clock.sleep)
        engine.tap_trace = runtime.tap_trace
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, engine.config,
            advanced=True, multi_touch=True, clock=clock, sleeper=clock.sleep)
        executor.begin_segment(runtime.tap_trace.run_id, runtime.tap_trace.segment_id)
        return engine, owner, event, clock, runtime, executor

    def test_dispatch_updates_marker_prediction_before_any_input(self):
        engine, owner, event, clock, runtime, executor = self.marker_setup()
        marker = engine.sustain_tracker.markers[7]
        marker.observations[-1] = replace(marker.observations[-1], progress=.88)
        pending = [event]
        with patch.object(executor, '_run') as run:
            runtime._execute_due(executor, pending, clock.now, RuntimeMetrics(), engine, wait=False)
        run.assert_not_called()
        self.assertEqual(len(pending), 1)
        self.assertGreater(pending[0].deadline, event.deadline)
        self.assertEqual(pending[0].origin, 'hold_note')
        self.assertEqual(owner.state, TrackState.HOLDING)

    def test_terminal_marker_runtime_receipt_retires_only_after_touchup(self):
        engine, owner, event, clock, runtime, executor = self.marker_setup(exits=1)
        owner.hold_terminal_confirmed = True
        clock.now = 1.9
        pending = [event]
        seen = []
        def action(kind, *args, **kwargs):
            seen.append((kind.value, owner.state))
        with patch.object(executor, '_run', side_effect=action):
            runtime._execute_due(executor, pending, clock.now, RuntimeMetrics(), engine, wait=False)
        self.assertEqual([kind for kind, _ in seen], ['TouchDown', 'TouchUp'])
        self.assertTrue(all(state == TrackState.HOLDING for _, state in seen))
        self.assertEqual(owner.state, TrackState.RELEASED)
        self.assertFalse(pending)
        inputs = [r for r in runtime.tap_trace.records if r['kind'] == 'input']
        self.assertEqual(inputs[0]['origin'], 'hold_note')
        self.assertEqual(inputs[0]['latest_visual_time'], 1.8)

    def test_old_segment_receipt_cannot_acknowledge_current_marker(self):
        engine, owner, event, clock, runtime, executor = self.marker_setup()
        receipt = TapInputReceipt(event.event_id, event.lane, 0, 1.9, 1.901, 1.902,
            run_id=runtime.tap_trace.run_id, segment_id=runtime.tap_trace.segment_id - 1)
        runtime._acknowledge_taps([receipt], [event], engine, RuntimeMetrics())
        state = engine.hold_note_event_registry.states[event.event_id]
        self.assertIsNone(state.started)
        self.assertIsNone(state.completed)
        self.assertTrue(any(r['kind'] == 'stale_receipt' for r in runtime.tap_trace.records))

    def test_lost_ordinary_head_cancelled_at_main_and_flick_input_entries(self):
        for during_flick in (False, True):
            with self.subTest(during_flick=during_flick):
                engine, owner, event, clock, runtime, executor = self.marker_setup()
                owner.state, owner.gesture = TrackState.LOST, NoteGesture.TAP
                pending = [tap_event(owner.track_id, owner.lane, 1.)]
                with patch.object(executor, '_run') as run:
                    if during_flick:
                        runtime._flush_due_taps(executor, pending, engine, RuntimeMetrics())
                    else:
                        runtime._execute_due(executor, pending, clock.now, RuntimeMetrics(), engine, wait=False)
                run.assert_not_called()
                self.assertFalse(pending)

    def test_marker_and_track_refinement_are_independent(self):
        engine, owner, event, clock, runtime, executor = self.marker_setup()
        owner.predicted_hit_time = 500.
        # A negative marker ID must never inherit the head's timing policy.
        pending = engine.refine_pending([event], 1.82)
        self.assertEqual(len(pending), 1)
        self.assertAlmostEqual(pending[0].deadline, 1.875)
        self.assertEqual(pending[0].origin, 'hold_note')

    def test_unsupported_hold_due_during_flick_uses_flat_fallback_batch(self):
        engine, owner, event, clock, runtime, executor = self.marker_setup()
        executor.supports_holds = False
        pending = [MusicActionEvent('fallback-hold', owner.track_id, owner.lane,
            NoteGesture.HOLD_START, 1.8, engine._lane_point(owner.lane))]
        with patch.object(executor, '_run') as run:
            runtime._flush_due_taps(executor, pending, engine, RuntimeMetrics())
        self.assertEqual([call.args[0].value for call in run.call_args_list], ['TouchDown', 'TouchUp'])
        self.assertFalse(pending)
        self.assertEqual(owner.state, TrackState.RELEASED)

    def test_old_async_receipt_does_not_pop_new_segment_inflight_event(self):
        engine, owner, event, clock, runtime, executor = self.marker_setup()
        stale = TapInputReceipt(event.event_id, event.lane, 0, 1.9, 1.901, 1.902,
            error='previous segment error', run_id=runtime.tap_trace.run_id,
            segment_id=runtime.tap_trace.segment_id - 1)
        runtime._in_flight_taps = {event.event_id: event}
        executor.async_input = True
        with patch.object(executor, 'poll_inputs', return_value=[stale]):
            runtime._poll_async_inputs(executor, engine, RuntimeMetrics())
        self.assertEqual(runtime._in_flight_taps[event.event_id], event)
        state = engine.hold_note_event_registry.states[event.event_id]
        self.assertIsNone(state.started)


if __name__ == '__main__':
    unittest.main()
