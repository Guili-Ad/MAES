"""Synchronous interleaved flicks; clocks and native calls are simulated."""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.executor import FlickSubmission, MusicActionExecutor, MusicTouchError
from agent.music.models import FlickRequest, LaneInputState, MusicConfig, NoteGesture


@dataclass(frozen=True)
class Submission:
    request: FlickRequest
    event_id: str
    track_id: int | None = None
    deadline: float | None = None


class Clock:
    def __init__(self):
        self.now = 10.
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class Context:
    def __init__(self, clock, cost=.008):
        self.clock, self.cost = clock, cost
        self.calls = []
        self.fail = None

    def run_action_direct(self, kind, param):
        call = dict(kind=kind.value, contact=param.contact, time=self.clock(),
                    target=getattr(param, 'target', None))
        self.calls.append(call)
        self.clock.now += self.cost
        if self.fail is not None and self.fail(call):
            raise RuntimeError('simulated native failure')
        return SimpleNamespace(success=True)


class FlickSessionTests(unittest.TestCase):
    def setup_executor(self, *, cost=.008, capacity=10, multi=True):
        clock = Clock()
        context = Context(clock, cost)
        config = MusicConfig(flick_duration_ms=60, flick_steps=3,
                             flick_end_hold_ms=16., max_contacts=capacity)
        executor = MusicActionExecutor(context, 1280, 720, config,
            advanced=True, multi_touch=multi, clock=clock, sleeper=clock.sleep)
        executor.begin_segment('run', 4)
        return clock, context, executor

    def submission(self, event_id, lane, direction, deadline=10.):
        return Submission(FlickRequest(lane, 250 if lane < 3 else 1030,
                                       516, direction), event_id, lane + 100, deadline)

    def dispatch(self, executor, items, **kwargs):
        # These first two regressions also exercise the old implementation,
        # reproducing its whole-swipe blocking instead of just missing APIs.
        if hasattr(executor, 'swipe_many'):
            return executor.swipe_many(items, **kwargs)
        tick = kwargs.get('tick')
        for item in items:
            executor.swipe(item.request, event_id=item.event_id, tick=tick)
        collector = kwargs.get('collect_due')
        if collector is not None:
            for item in collector(frozenset(), executor.config.max_contacts):
                executor.swipe(item.request, event_id=item.event_id, tick=tick)
        return []

    def test_due_double_flick_starts_before_either_finishes(self):
        clock, context, executor = self.setup_executor()
        items = [self.submission('left', 1, NoteGesture.FLICK_LEFT),
                 self.submission('right', 5, NoteGesture.FLICK_RIGHT)]
        self.dispatch(executor, items)
        downs = [call for call in context.calls if call['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        self.assertLessEqual(downs[1]['time'] - downs[0]['time'], .008001)
        self.assertEqual([c['kind'] for c in context.calls[:2]], ['TouchDown', 'TouchDown'])

    def test_newly_due_flick_joins_without_waiting_for_first_up(self):
        clock, context, executor = self.setup_executor()
        second = self.submission('right', 5, NoteGesture.FLICK_RIGHT, 10.02)
        offered = []

        def collect(lanes, available):
            if clock() >= second.deadline and not offered:
                offered.append(True)
                return [second]
            return []

        self.dispatch(executor, [self.submission('left', 1, NoteGesture.FLICK_LEFT)], collect_due=collect)
        downs = [c for c in context.calls if c['kind'] == 'TouchDown']
        self.assertEqual(len(downs), 2)
        self.assertLess(downs[1]['time'] - second.deadline, .025)
        self.assertLess(downs[1]['time'], next(c['time'] for c in context.calls if c['kind'] == 'TouchUp'))

    def test_single_staged_swipe_matches_legacy_calls_coordinates_and_times(self):
        snapshots = []
        for staged in (False, True):
            clock, context, executor = self.setup_executor()
            item = self.submission('one', 5, NoteGesture.FLICK_RIGHT)
            if staged:
                executor.swipe_many([item])
            else:
                executor.swipe(item.request, event_id=item.event_id)
            snapshots.append(context.calls)
        self.assertEqual(snapshots[0], snapshots[1])

    def test_distinct_due_deadlines_are_not_replaced_by_a_group_mean(self):
        clock, context, executor = self.setup_executor()
        items = [self.submission('left', 1, NoteGesture.FLICK_LEFT, 9.95),
                 self.submission('right', 5, NoteGesture.FLICK_RIGHT, 9.98)]
        seen = []
        executor.swipe_many(items, on_started=lambda item, receipt: seen.append(item.deadline))
        self.assertEqual(seen, [9.95, 9.98])

    def test_future_submission_is_deferred_without_down_or_sleep(self):
        clock, context, executor = self.setup_executor()
        item = self.submission('future', 5, NoteGesture.FLICK_RIGHT, 10.02)
        self.assertEqual(executor.swipe_many([item]), [])
        self.assertFalse(context.calls)
        self.assertFalse(clock.sleeps)
        self.assertEqual(executor.last_flick_deferred, [(item, 'not-due')])
        self.assertNotIn(executor._event_key(item.event_id), executor._used_event_ids)

    def test_same_lane_requests_never_overlap(self):
        clock, context, executor = self.setup_executor()
        items = [self.submission('first', 1, NoteGesture.FLICK_LEFT),
                 self.submission('second', 1, NoteGesture.FLICK_RIGHT)]
        receipts = executor.swipe_many(items)
        calls = [c['kind'] for c in context.calls]
        self.assertEqual(calls, ['TouchDown', 'TouchMove', 'TouchMove', 'TouchMove', 'TouchUp'] * 2)
        self.assertEqual([r.event_id for r in receipts], ['first', 'second'])
        self.assertEqual(receipts[1].deferred_reasons, ['occupied-lane'])
        self.assertFalse(executor.active_flick_lanes)

    def test_duplicate_submission_does_not_repeat_down(self):
        clock, context, executor = self.setup_executor()
        item = self.submission('one', 1, NoteGesture.FLICK_LEFT)
        self.assertEqual(len(executor.swipe_many([item, item])), 1)
        self.assertEqual(executor.swipe_many([item]), [])
        self.assertEqual(sum(c['kind'] == 'TouchDown' for c in context.calls), 1)
        self.assertEqual(executor.last_flick_deferred, [(item, 'duplicate-event')])

    def test_capacity_one_runs_without_overlapping_contacts(self):
        clock, context, executor = self.setup_executor(capacity=1)
        receipts = executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                                       self.submission('b', 5, NoteGesture.FLICK_RIGHT)])
        self.assertEqual(len(receipts), 2)
        kinds = [c['kind'] for c in context.calls]
        self.assertEqual(kinds[:5], ['TouchDown', 'TouchMove', 'TouchMove', 'TouchMove', 'TouchUp'])
        self.assertEqual(kinds[5], 'TouchDown')
        self.assertEqual(receipts[0].contact, receipts[1].contact)
        self.assertEqual(receipts[1].deferred_reasons, ['contact-capacity'])

    def test_unconfirmed_contacts_leave_unstarted_queue_entries_deferred(self):
        clock, context, executor = self.setup_executor(capacity=1)
        executor.release_unconfirmed.add(0)
        item = self.submission('a', 1, NoteGesture.FLICK_LEFT)
        self.assertEqual(executor.swipe_many([item]), [])
        self.assertFalse(context.calls)
        self.assertFalse(clock.sleeps)
        self.assertEqual(executor.last_flick_deferred, [(item, 'contact-capacity')])

    def test_persistent_hold_lane_is_not_stolen(self):
        clock, context, executor = self.setup_executor()
        executor.lanes[1] = LaneInputState(1, contact=0, hold_track_id=40)
        blocked = self.submission('blocked', 1, NoteGesture.FLICK_LEFT)
        free = self.submission('free', 5, NoteGesture.FLICK_RIGHT)
        receipts = executor.swipe_many([blocked, free])
        self.assertEqual([r.event_id for r in receipts], ['free'])
        self.assertEqual(executor.active_contacts, {1: 0})
        self.assertEqual(executor.last_flick_deferred, [(blocked, 'occupied-lane')])
        self.assertTrue(all(c['contact'] != 0 for c in context.calls))

    def test_single_touch_backend_preserves_serial_compatibility(self):
        clock, context, executor = self.setup_executor(multi=False)
        executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                             self.submission('b', 5, NoteGesture.FLICK_RIGHT)])
        self.assertEqual([c['kind'] for c in context.calls],
                         ['TouchDown', 'TouchMove', 'TouchMove', 'TouchMove', 'TouchUp'] * 2)

    def test_input_receipts_are_per_member_host_call_times(self):
        clock, context, executor = self.setup_executor()
        receipts = executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                                       self.submission('b', 5, NoteGesture.FLICK_RIGHT)])
        for receipt in receipts:
            owned = [c for c in context.calls if c['contact'] == receipt.contact]
            self.assertEqual(receipt.down_call_started, owned[0]['time'])
            self.assertAlmostEqual(receipt.down_call_finished, owned[0]['time'] + .008)
            self.assertEqual(receipt.move_call_started, [c['time'] for c in owned if c['kind'] == 'TouchMove'])
            self.assertEqual(receipt.up_call_started, owned[-1]['time'])
            self.assertAlmostEqual(receipt.up_call_finished, owned[-1]['time'] + .008)
            self.assertEqual((receipt.run_id, receipt.segment_id), ('run', 4))
            self.assertFalse(receipt.error)

    def test_on_started_sees_registered_attempt_before_native_call(self):
        clock, context, executor = self.setup_executor()
        observed = []

        def started(item, receipt):
            observed.append((len(context.calls), receipt.down_call_started,
                             executor._event_key(item.event_id) in executor._used_event_ids))

        executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)], on_started=started)
        self.assertEqual(observed, [(0, 10., True)])

    def test_tick_can_tap_another_lane_but_not_an_active_flick_lane(self):
        clock, context, executor = self.setup_executor()
        tapped = []

        def tick():
            if tapped or not executor.active_flick_lanes:
                return
            with self.assertRaisesRegex(MusicTouchError, 'active flick lane'):
                executor.tap(1, 250, 516, event_id='overlap')
            self.assertNotIn(executor._event_key('overlap'), executor._used_event_ids)
            tapped.extend(executor.tap(3, 640, 620, event_id='other-lane'))

        executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)], tick=tick)
        self.assertEqual(len(tapped), 1)
        self.assertFalse(executor._temporary_contacts)

    def test_second_down_failure_cleans_both_and_does_not_resend(self):
        clock, context, executor = self.setup_executor()
        context.fail = lambda call: call['kind'] == 'TouchDown' and call['contact'] == 1
        items = [self.submission('a', 1, NoteGesture.FLICK_LEFT),
                 self.submission('b', 5, NoteGesture.FLICK_RIGHT)]
        with self.assertRaises(MusicTouchError) as raised:
            executor.swipe_many(items)
        self.assertEqual(len(raised.exception.receipts), 2)
        self.assertTrue(all(r.down_call_started is not None for r in raised.exception.receipts))
        self.assertIsNone(raised.exception.receipts[1].down_call_finished)
        self.assertTrue(all(r.up_call_finished is not None for r in raised.exception.receipts))
        self.assertEqual(sum(c['kind'] == 'TouchDown' for c in context.calls), 2)
        self.assertFalse(executor._temporary_contacts)
        self.assertFalse(executor.active_flick_lanes)

    def test_first_down_failure_keeps_prepared_second_member_unstarted(self):
        clock, context, executor = self.setup_executor()
        context.fail = lambda call: call['kind'] == 'TouchDown'
        started = []
        with self.assertRaises(MusicTouchError) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                                 self.submission('b', 5, NoteGesture.FLICK_RIGHT)],
                                on_started=lambda item, receipt: started.append(item.event_id))
        receipts = raised.exception.receipts
        self.assertEqual(started, ['a'])
        self.assertIsNone(receipts[1].down_call_started)
        self.assertNotIn(executor._event_key('b'), executor._used_event_ids)
        self.assertEqual(sum(c['kind'] == 'TouchUp' for c in context.calls), 1)
        self.assertFalse(executor._temporary_contacts)

    def test_move_failure_retains_completed_and_failed_call_timestamps(self):
        clock, context, executor = self.setup_executor()
        context.fail = lambda call: call['kind'] == 'TouchMove'
        with self.assertRaises(MusicTouchError) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)])
        receipt = raised.exception.receipts[0]
        self.assertEqual(len(receipt.move_call_started), 1)
        self.assertFalse(receipt.move_call_finished)
        self.assertIsNotNone(receipt.up_call_finished)
        self.assertIn('failed', receipt.error)

    def test_up_failure_retries_up_once_and_preserves_original_error(self):
        clock, context, executor = self.setup_executor()
        failed = []

        def fail(call):
            if call['kind'] == 'TouchUp' and not failed:
                failed.append(True)
                return True
            return False

        context.fail = fail
        with self.assertRaises(MusicTouchError) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)])
        receipt = raised.exception.receipts[0]
        self.assertIsNotNone(receipt.up_call_finished)
        self.assertIn('failed', receipt.error)
        self.assertEqual(sum(c['kind'] == 'TouchUp' for c in context.calls), 2)
        self.assertFalse(executor.release_unconfirmed)

    def test_up_double_failure_keeps_touch_and_lane_reserved_until_cleanup(self):
        clock, context, executor = self.setup_executor(capacity=2)
        context.fail = lambda call: call['kind'] == 'TouchUp'
        with self.assertRaisesRegex(MusicTouchError, 'cleanup unconfirmed') as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)])
        receipt = raised.exception.receipts[0]
        self.assertIsNone(receipt.up_call_finished)
        self.assertEqual(executor.release_unconfirmed, {0})
        self.assertEqual(executor.active_flick_lanes, frozenset({1}))
        self.assertEqual(executor._allocate_temporary_contact(), 1)
        context.fail = None
        executor.release_all()
        self.assertFalse(executor.release_unconfirmed)
        self.assertFalse(executor.active_flick_lanes)
        self.assertFalse(executor._temporary_contacts)

    def test_cancel_cleans_contacts_and_preserves_cancellation_type(self):
        clock, context, executor = self.setup_executor()
        class Cancelled(RuntimeError):
            pass

        def check():
            if any(c['kind'] == 'TouchMove' for c in context.calls):
                raise Cancelled('user cancelled')

        with self.assertRaises(Cancelled) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)], check_cancelled=check)
        self.assertIsNotNone(raised.exception.receipts[0].up_call_finished)
        self.assertFalse(executor._temporary_contacts)
        self.assertIsNone(executor._flick_session)

    def test_release_all_ends_session_without_reusing_released_contacts(self):
        clock, context, executor = self.setup_executor()
        released = []

        def tick():
            if not released:
                released.append(True)
                executor.release_all()

        with self.assertRaisesRegex(MusicTouchError, 'released during execution'):
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)], tick=tick)
        self.assertEqual([c['kind'] for c in context.calls], ['TouchDown', 'TouchUp'])
        self.assertFalse(executor.active_flick_lanes)
        self.assertFalse(executor._temporary_contacts)

    def test_release_all_records_each_actual_up_before_batch_cleanup_returns(self):
        clock, context, executor = self.setup_executor()
        released = []

        def tick():
            if not released:
                released.append(True)
                executor.release_all()

        with self.assertRaises(MusicTouchError) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                                 self.submission('b', 5, NoteGesture.FLICK_RIGHT)], tick=tick)
        for receipt in raised.exception.receipts:
            up = next(c for c in context.calls
                      if c['kind'] == 'TouchUp' and c['contact'] == receipt.contact)
            self.assertEqual(receipt.up_call_started, up['time'])
            self.assertAlmostEqual(receipt.up_call_finished, up['time'] + .008)

    def test_cancel_before_down_does_not_consume_events_or_contacts(self):
        clock, context, executor = self.setup_executor()
        def check():
            raise RuntimeError('cancel before starting')
        with self.assertRaisesRegex(RuntimeError, 'cancel before starting') as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)],
                                check_cancelled=check)
        self.assertEqual(raised.exception.receipts, [])
        self.assertFalse(context.calls)
        self.assertFalse(executor._temporary_contacts)
        self.assertFalse(executor._used_event_ids)

    def test_cancel_after_first_down_cleans_only_started_member(self):
        clock, context, executor = self.setup_executor()
        def check():
            if context.calls:
                raise RuntimeError('cancel after first native call')
        with self.assertRaises(RuntimeError) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                                 self.submission('b', 5, NoteGesture.FLICK_RIGHT)],
                                check_cancelled=check)
        self.assertEqual([c['kind'] for c in context.calls], ['TouchDown', 'TouchUp'])
        self.assertIsNotNone(raised.exception.receipts[0].up_call_finished)
        self.assertIsNone(raised.exception.receipts[1].down_call_started)
        self.assertNotIn(executor._event_key('b'), executor._used_event_ids)
        self.assertFalse(executor._temporary_contacts)

    def test_release_in_started_callback_cannot_press_after_cleanup(self):
        clock, context, executor = self.setup_executor()
        with self.assertRaisesRegex(MusicTouchError, 'released during execution'):
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)],
                                on_started=lambda item, receipt: executor.release_all())
        self.assertFalse(any(call['kind'] == 'TouchDown' for call in context.calls))
        self.assertFalse(executor.active_flick_lanes)
        self.assertFalse(executor._temporary_contacts)

    def test_completed_member_remains_success_when_other_member_later_fails(self):
        clock, context, executor = self.setup_executor()
        context.fail = lambda call: call['kind'] == 'TouchUp' and call['contact'] == 1
        with self.assertRaises(MusicTouchError) as raised:
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT),
                                 self.submission('b', 5, NoteGesture.FLICK_RIGHT)])
        first, second = raised.exception.receipts
        self.assertFalse(first.error)
        self.assertIsNotNone(first.up_call_finished)
        self.assertTrue(second.error)
        self.assertIsNone(second.up_call_finished)

    def test_repeated_collector_offers_and_same_lane_new_due_keep_one_cycle_each(self):
        clock, context, executor = self.setup_executor()
        later = self.submission('b', 1, NoteGesture.FLICK_RIGHT, 10.02)
        receipts = executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)],
            collect_due=lambda lanes, capacity: [later] if clock() >= later.deadline else [])
        self.assertEqual([r.event_id for r in receipts], ['a', 'b'])
        self.assertEqual([c['kind'] for c in context.calls],
                         ['TouchDown', 'TouchMove', 'TouchMove', 'TouchMove', 'TouchUp'] * 2)

    def test_production_submission_type_is_accepted(self):
        clock, context, executor = self.setup_executor()
        item = FlickSubmission(FlickRequest(1, 250, 516, NoteGesture.FLICK_LEFT),
                               'a', 11, 10.)
        receipts = executor.swipe_many([item])
        self.assertEqual([r.event_id for r in receipts], ['a'])

    def test_cancel_waits_are_at_most_ten_milliseconds(self):
        clock, context, executor = self.setup_executor(cost=0.)
        executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)])
        self.assertTrue(clock.sleeps)
        self.assertLessEqual(max(clock.sleeps), .010)
        self.assertGreaterEqual(sum(clock.sleeps), .076 - 1e-9)

    def test_all_four_directions_keep_waypoints_and_clipped_endpoints(self):
        for direction, endpoint in ((NoteGesture.FLICK_LEFT, (0, 10)),
                                    (NoteGesture.FLICK_RIGHT, (66, 10)),
                                    (NoteGesture.FLICK_UP, (10, 0)),
                                    (NoteGesture.FLICK_DOWN, (10, 66))):
            with self.subTest(direction=direction):
                clock, context, executor = self.setup_executor()
                executor.swipe_many([Submission(FlickRequest(1, 10, 10, direction), 'a', deadline=10.)])
                moves = [c for c in context.calls if c['kind'] == 'TouchMove']
                self.assertEqual(len(moves), 3)
                self.assertEqual(moves[-1]['target'][:2], endpoint)

    def test_held_or_async_flick_is_not_silently_migrated(self):
        clock, context, executor = self.setup_executor()
        held = Submission(FlickRequest(1, 250, 516, NoteGesture.FLICK_LEFT, already_down=True), 'held')
        self.assertEqual(executor.swipe_many([held]), [])
        self.assertEqual(executor.last_flick_deferred, [(held, 'legacy-held-flick')])
        self.assertFalse(context.calls)
        executor.async_flicks = True
        with self.assertRaisesRegex(MusicTouchError, 'synchronous'):
            executor.swipe_many([self.submission('a', 1, NoteGesture.FLICK_LEFT)])

    def test_segment_reset_uses_the_same_id_without_stale_active_state(self):
        clock, context, executor = self.setup_executor()
        item = self.submission('one', 1, NoteGesture.FLICK_LEFT)
        executor.swipe_many([item])
        executor.release_all()
        executor.begin_segment('run', 5)
        receipts = executor.swipe_many([item])
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].segment_id, 5)


if __name__ == '__main__':
    unittest.main()
