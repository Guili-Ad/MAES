"""Staged flick integration with the real runtime, but simulated native input.

No emulator, screenshots, or game judgment claims are involved. The event
deadlines and the native-call clock are kept distinct in these regressions.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.executor import FlickInputReceipt, MusicActionExecutor, MusicTouchError
from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture
from agent.music.runtime import MusicCancelled, MusicRuntime, RuntimeMetrics
from test_flick_session import Clock, Context


class FlickRuntimeV4Tests(unittest.TestCase):
    def setup_runtime(self, *, capacity=10, tap_mode=False, multi=True):
        clock = Clock()
        context = Context(clock, cost=.008)
        config = MusicConfig(lane_count=7, enable_holds=True, hold_notes_as_taps=tap_mode,
            flick_duration_ms=60,
            flick_steps=3, flick_end_hold_ms=16., max_contacts=capacity)
        runtime = MusicRuntime(context, config, clock=clock, sleeper=clock.sleep)
        executor = MusicActionExecutor(context, 1280, 720, config,
            advanced=True, multi_touch=multi, clock=clock, sleeper=clock.sleep)
        executor.begin_segment(runtime.tap_trace.run_id, runtime.tap_trace.segment_id)
        return clock, context, runtime, executor, RuntimeMetrics()

    def event(self, name, lane, *, deadline=10., gesture=NoteGesture.FLICK_RIGHT):
        return MusicActionEvent(name, 100 + lane, lane, gesture, deadline,
            (250 if lane < 3 else 640 if lane == 3 else 1030, 516),
            source_capture_started=9.98, source_capture_finished=9.99)

    def execute(self, runtime, executor, pending, metrics):
        runtime._execute_due(executor, pending, runtime.clock(), metrics, wait=False)

    def inputs(self, runtime, event_id=None):
        return [row for row in runtime.tap_trace.records
                if row['kind'] == 'input' and (event_id is None or row.get('event') == event_id)]

    def test_two_due_independent_flicks_start_down_down_without_a_shared_deadline(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('left', 1, deadline=9.98, gesture=NoteGesture.FLICK_LEFT),
                   self.event('right', 5, deadline=10.)]
        self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual([row['kind'] for row in context.calls[:2]], ['TouchDown', 'TouchDown'])
        self.assertLessEqual(downs[1]['time'] - downs[0]['time'], .008001)
        inputs = {row['event']: row for row in self.inputs(runtime)}
        self.assertEqual(inputs['left']['deadline'], 9.98)
        self.assertEqual(inputs['right']['deadline'], 10.)
        self.assertFalse(pending)

    def test_twenty_ms_later_due_flick_is_collected_during_the_active_session(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('left', 1, gesture=NoteGesture.FLICK_LEFT),
                   self.event('right', 5, deadline=10.02)]
        self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        self.assertLess(downs[1]['time'] - 10.02, .025)
        first_up = next(row['time'] for row in context.calls if row['kind'] == 'TouchUp')
        self.assertLess(downs[1]['time'], first_up)
        self.assertFalse(pending)

    def test_tap_mode_hold_tail_without_owned_contact_uses_staged_standalone_flicks(self):
        clock, context, runtime, executor, metrics = self.setup_runtime(tap_mode=True)
        pending = [replace(self.event('tail-a', 1), contact_policy='held_flick'),
                   replace(self.event('tail-b', 5), contact_policy='held_flick')]
        self.execute(runtime, executor, pending, metrics)
        self.assertEqual([row['kind'] for row in context.calls[:2]], ['TouchDown', 'TouchDown'])
        self.assertFalse(pending)

    def test_actual_owned_hold_tail_stays_on_legacy_held_flick_path(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        event = replace(self.event('owned-tail', 1), contact_policy='held_flick')
        executor.touch_down(event.lane, *event.coordinate, track_id=event.track_id)
        context.calls.clear()
        pending = [event]
        with patch.object(executor, 'swipe', wraps=executor.swipe) as legacy, \
                patch.object(executor, 'swipe_many', wraps=executor.swipe_many) as staged:
            self.execute(runtime, executor, pending, metrics)
        self.assertEqual(legacy.call_count, 1)
        self.assertEqual(staged.call_count, 0)
        self.assertEqual([row['kind'] for row in context.calls],
                         ['TouchMove', 'TouchMove', 'TouchMove', 'TouchUp'])
        self.assertFalse(executor.active_contacts)

    def test_flick_receipt_contains_real_per_call_host_timestamps_and_source_latency(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        self.execute(runtime, executor, [self.event('one', 5)], metrics)
        rows = self.inputs(runtime, 'one')
        self.assertEqual(len(rows), 1)
        receipt = rows[0]['receipt']
        self.assertEqual(receipt['down_call_started'], context.calls[0]['time'])
        self.assertAlmostEqual(receipt['down_call_finished'], context.calls[0]['time'] + .008)
        self.assertEqual(receipt['move_call_started'],
                         [row['time'] for row in context.calls if row['kind'] == 'TouchMove'])
        self.assertAlmostEqual(receipt['up_call_finished'], context.calls[-1]['time'] + .008)
        self.assertEqual((receipt['run_id'], receipt['segment_id']),
                         (runtime.tap_trace.run_id, runtime.tap_trace.segment_id))
        self.assertEqual(len(metrics.perception_to_action), 1)
        self.assertAlmostEqual(metrics.perception_to_action[0], 20.)

    def test_stale_segment_flick_receipt_never_acknowledges_current_event(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        event = self.event('same-id', 5)
        stale = FlickInputReceipt(event.event_id, event.lane, 0,
            down_call_started=10., down_call_finished=10.008,
            up_call_started=10.108, up_call_finished=10.116,
            run_id=runtime.tap_trace.run_id, segment_id=runtime.tap_trace.segment_id - 1)
        runtime._acknowledge_flicks([stale], [event], None, metrics)
        self.assertFalse(self.inputs(runtime, event.event_id))
        self.assertFalse(runtime.head_action_trace)
        self.assertFalse(metrics.perception_to_action)
        self.assertTrue(any(row['kind'] == 'stale_receipt' for row in runtime.tap_trace.records))

    def test_same_lane_tap_waits_for_flick_up_instead_of_overlapping(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('flick', 1),
                   self.event('tap', 1, deadline=10.02, gesture=NoteGesture.TAP)]
        self.execute(runtime, executor, pending, metrics)
        if pending:
            self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        first_up = next(row for row in context.calls if row['kind'] == 'TouchUp')
        self.assertGreaterEqual(downs[1]['time'], first_up['time'] + .008 - 1e-9)
        self.assertFalse(pending)

    def test_other_lane_tap_is_serviced_before_flick_finishes(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('flick', 1),
                   self.event('tap', 3, deadline=10.02, gesture=NoteGesture.TAP)]
        self.execute(runtime, executor, pending, metrics)
        flick_down = context.calls[0]
        tap_down = next(row for row in context.calls
                        if row['kind'] == 'TouchDown' and row['target'][:2] == (640, 516))
        flick_up = next(row for row in context.calls
                        if row['kind'] == 'TouchUp' and row['contact'] == flick_down['contact'])
        self.assertLess(tap_down['time'], flick_up['time'])
        self.assertFalse(pending)

    def test_tap_mode_hold_head_during_flick_is_a_complete_tap_not_a_persistent_down(self):
        clock, context, runtime, executor, metrics = self.setup_runtime(tap_mode=True)
        pending = [self.event('flick', 1),
                   self.event('head', 3, deadline=10.02, gesture=NoteGesture.HOLD_START)]
        self.execute(runtime, executor, pending, metrics)
        head_down = next(row for row in context.calls
                         if row['kind'] == 'TouchDown' and row['target'][:2] == (640, 516))
        head_calls = [row['kind'] for row in context.calls if row['contact'] == head_down['contact']]
        self.assertEqual(head_calls, ['TouchDown', 'TouchUp'])
        self.assertFalse(executor.active_contacts)
        self.assertTrue(self.inputs(runtime, 'head'))
        self.assertFalse(pending)

    def test_tap_chord_with_one_locked_lane_is_not_split_or_silently_consumed(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('flick', 1),
                   replace(self.event('tap-a', 1, deadline=10.02, gesture=NoteGesture.TAP),
                           tap_group_id='pair'),
                   replace(self.event('tap-b', 3, deadline=10.02, gesture=NoteGesture.TAP),
                           tap_group_id='pair')]
        self.execute(runtime, executor, pending, metrics)
        if pending:
            self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 3)
        flick_up = next(row for row in context.calls if row['kind'] == 'TouchUp')
        self.assertGreaterEqual(downs[1]['time'], flick_up['time'] + .008 - 1e-9)
        self.assertAlmostEqual(downs[2]['time'] - downs[1]['time'], .008)
        self.assertEqual({row['event'] for row in self.inputs(runtime)}, {'flick', 'tap-a', 'tap-b'})

    def test_unready_tap_chord_during_flick_never_falls_through_hold_dispatch(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('flick', 1),
                   replace(self.event('tap-a', 3, deadline=10.02, gesture=NoteGesture.TAP),
                           tap_group_id='pair'),
                   replace(self.event('tap-b', 5, deadline=10.05, gesture=NoteGesture.TAP),
                           tap_group_id='pair')]
        self.execute(runtime, executor, pending, metrics)
        if pending:
            self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 3)
        self.assertGreaterEqual(downs[1]['time'], 10.05 - 1e-9)
        self.assertAlmostEqual(downs[2]['time'] - downs[1]['time'], .008)
        self.assertFalse(pending)

    def test_other_lane_tap_waits_when_flick_owns_the_last_contact(self):
        clock, context, runtime, executor, metrics = self.setup_runtime(capacity=1)
        pending = [self.event('flick', 1),
                   self.event('tap', 3, deadline=10.02, gesture=NoteGesture.TAP)]
        self.execute(runtime, executor, pending, metrics)
        if pending:
            self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        first_up = next(row for row in context.calls if row['kind'] == 'TouchUp')
        self.assertGreaterEqual(downs[1]['time'], first_up['time'] + .008 - 1e-9)
        self.assertTrue(executor.healthy)
        self.assertFalse(pending)

    def test_single_touch_backend_never_taps_during_an_airborne_flick(self):
        clock, context, runtime, executor, metrics = self.setup_runtime(multi=False)
        pending = [self.event('flick', 1),
                   self.event('tap', 3, deadline=10.02, gesture=NoteGesture.TAP)]
        self.execute(runtime, executor, pending, metrics)
        if pending:
            self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        first_up = next(row for row in context.calls if row['kind'] == 'TouchUp')
        self.assertGreaterEqual(downs[1]['time'], first_up['time'] + .008 - 1e-9)
        self.assertFalse(pending)

    def test_persistent_press_waits_if_flick_owns_the_last_contact(self):
        clock, context, runtime, executor, metrics = self.setup_runtime(capacity=1)
        pending = [self.event('flick', 1),
                   self.event('press', 3, deadline=10.02, gesture=NoteGesture.SUSTAIN_PRESS)]
        self.execute(runtime, executor, pending, metrics)
        if pending:
            self.execute(runtime, executor, pending, metrics)
        downs = [row for row in context.calls if row['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        first_up = next(row for row in context.calls if row['kind'] == 'TouchUp')
        self.assertGreaterEqual(downs[1]['time'], first_up['time'] + .008 - 1e-9)
        self.assertEqual(executor.hold_owner(3), 103)
        executor.release_all()

    def test_first_down_failure_keeps_unstarted_partner_in_pending(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        context.fail = lambda row: row['kind'] == 'TouchDown'
        pending = [self.event('a', 1), self.event('b', 5)]
        with self.assertRaises(MusicTouchError):
            self.execute(runtime, executor, pending, metrics)
        self.assertEqual([event.event_id for event in pending], ['b'])
        self.assertNotIn(executor._event_key('b'), executor._used_event_ids)
        self.assertEqual(sum(row['kind'] == 'TouchDown' for row in context.calls), 1)
        attempted = self.inputs(runtime, 'a')
        self.assertEqual(len(attempted), 1)
        self.assertIsNone(attempted[0]['receipt']['down_call_finished'])
        self.assertIsNotNone(attempted[0]['receipt']['up_call_finished'])
        self.assertFalse(executor._temporary_contacts)

    def test_second_down_failure_keeps_third_prepared_but_unstarted_event(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        context.fail = lambda row: row['kind'] == 'TouchDown' and row['contact'] == 1
        pending = [self.event('a', 1), self.event('b', 3), self.event('c', 5)]
        with self.assertRaises(MusicTouchError):
            self.execute(runtime, executor, pending, metrics)
        self.assertEqual([event.event_id for event in pending], ['c'])
        self.assertNotIn(executor._event_key('c'), executor._used_event_ids)
        self.assertEqual(sum(row['kind'] == 'TouchDown' for row in context.calls), 2)
        self.assertEqual({row['event'] for row in self.inputs(runtime)
                          if row['receipt']['down_call_started'] is not None}, {'a', 'b'})
        self.assertFalse(executor._temporary_contacts)

    def test_no_contact_capacity_preserves_pending_without_waiting_or_consuming_id(self):
        clock, context, runtime, executor, metrics = self.setup_runtime(capacity=1)
        executor.release_unconfirmed.add(0)
        pending = [self.event('a', 1)]
        self.execute(runtime, executor, pending, metrics)
        self.assertEqual([event.event_id for event in pending], ['a'])
        self.assertFalse(context.calls)
        self.assertFalse(clock.sleeps)
        self.assertNotIn(executor._event_key('a'), executor._used_event_ids)

    def test_cancel_during_flick_cleans_and_does_not_consume_future_partner(self):
        clock, context, runtime, executor, metrics = self.setup_runtime()
        pending = [self.event('a', 1), self.event('b', 5, deadline=10.05)]
        def check():
            if any(row['kind'] == 'TouchMove' for row in context.calls):
                raise MusicCancelled('simulated cancellation')
        with patch.object(runtime, '_check_cancelled', side_effect=check):
            with self.assertRaises(MusicCancelled):
                self.execute(runtime, executor, pending, metrics)
        self.assertEqual([event.event_id for event in pending], ['b'])
        self.assertNotIn(executor._event_key('b'), executor._used_event_ids)
        self.assertFalse(executor._temporary_contacts)
        self.assertFalse(executor.active_flick_lanes)
        self.assertTrue(self.inputs(runtime, 'a')[0]['receipt']['error'])


if __name__ == '__main__':
    unittest.main()
