"""Production budgets and bounded diagnostics, independent of game grades."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture
from agent.music.runtime import MusicRuntime, RuntimeMetrics
from agent.music.tap_trace import TapTrace
from test_tap_pipeline import tap_event


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


if __name__ == '__main__':
    unittest.main()
