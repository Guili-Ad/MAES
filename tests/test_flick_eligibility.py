"""Queued arrows need the same fresh motion authority at queue and Down."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from test_point_head_identity import point_track
import test_flick_runtime_v4 as flick_tests
from agent.music.models import MusicFrame, TrackState, NoteGesture, MusicConfig, MusicActionEvent
from agent.music.tracking import MusicVisionEngine
from agent.music.vision import VisualMask


def flick_track(values, tid=1, lane=4):
    track = point_track(values, tid, lane)
    track.flick = True
    track.gesture = track.flick_direction = NoteGesture.FLICK_RIGHT
    track.flick_color = 'blue'
    track.direction_evidence.extend([NoteGesture.FLICK_RIGHT]*2)
    track.observations = type(track.observations)([replace(o, candidate=replace(o.candidate,
        variant='flick', flick_direction=NoteGesture.FLICK_RIGHT, flick_color='blue'))
        for o in track.observations], maxlen=12)
    return track


def engine_for(track, frame):
    engine = MusicVisionEngine(calibration(), MusicConfig(enable_holds=True, hold_notes_as_taps=True))
    engine.tracks = {track.track_id: track}
    engine.last_frame = frame
    engine.last_frame_sequence = frame.sequence
    return engine


class FlickEligibilityTests(unittest.TestCase):
    def setup_case(self, values, *, now=10., sequence=4):
        clock, context, runtime, executor, metrics = flick_tests.FlickRuntimeV4Tests().setup_runtime(tap_mode=True)
        clock.now = now
        track = flick_track(values)
        track.state, track.action_executed = TrackState.FLICK_PENDING, True
        track.predicted_hit_time = now+.094
        track.action_event_id = 'same-flick'
        frame = MusicFrame(sequence, values[-1][0], values[-1][0], values[-1][0], None)
        engine = engine_for(track, frame)
        event = MusicActionEvent('same-flick', track.track_id, track.lane,
            track.gesture, now, (800, 620), direction=track.gesture,
            source_capture_started=values[-1][0], source_capture_finished=values[-1][0])
        return clock, context, runtime, executor, metrics, track, engine, event

    def test_966_ms_stale_arrow_cannot_start_native_input(self):
        case = self.setup_case([(9., .60), (9.04, .64), (9.08, .68), (9.12, .72)], now=10.086, sequence=20)
        clock, context, runtime, executor, metrics, track, engine, event = case
        pending = [event]
        runtime._execute_due(executor, pending, clock(), metrics, engine, wait=False)
        self.assertFalse(context.calls)
        self.assertFalse(pending)
        self.assertNotIn(executor._event_key(event.event_id), executor._used_event_ids)
        self.assertEqual(track.state, TrackState.APPROACHING)
        self.assertEqual(track.action_event_id, event.event_id)

    def test_static_birth_small_later_jump_cannot_be_scheduled(self):
        values = [(1.+i*.04, .600341) for i in range(5)]+[(1.24, .630432)]
        track = flick_track(values)
        frame = MusicFrame(5, 1.24, 1.24, 1.24, None)
        engine = engine_for(track, frame)
        engine._update_motion(track)
        self.assertFalse(engine._ready_to_schedule(track, frame))

    def test_static_birth_is_remembered_before_motion_fit_and_deque_rollover(self):
        from agent.music.flick_eligibility import observe_arrow_origin, failure_reason
        track = flick_track([(1., .60), (1.05, .60), (1.1, .60)])
        self.assertTrue(observe_arrow_origin(track))
        moving = flick_track([(2.+i*.04, .60+i*.025) for i in range(12)])
        track.observations = moving.observations
        self.assertEqual(failure_reason(track, 2.44, sequence=11), 'stationary-arrow-origin')

    def test_previously_moving_arrow_short_freeze_does_not_become_static_birth(self):
        from agent.music.flick_eligibility import failure_reason
        track = flick_track([(9.84, .60), (9.88, .64), (9.92, .68), (9.96, .72), (10., .72)])
        self.assertIsNone(failure_reason(track, 10.01, sequence=4))
        self.assertFalse(track.flick_static_origin)
        self.assertEqual(failure_reason(track, 10.2, sequence=5), 'arrow-observation-age-exceeded')

    def test_legal_positive_reappearance_preserves_unattempted_event(self):
        from agent.music.flick_eligibility import valid_flick_pending
        case = self.setup_case([(9., .60), (9.04, .64), (9.08, .68), (9.12, .72)], now=10.086, sequence=20)
        clock, context, runtime, executor, metrics, track, engine, event = case
        self.assertFalse(valid_flick_pending(event, engine, clock(), runtime.tap_trace))
        self.assertEqual(track.action_event_id, 'same-flick')

        # Three later observed moving arrow pixels, not a new physical birth.
        newer = flick_track([(10.05, .80), (10.09, .84), (10.13, .88)])
        track.observations.extend(replace(o, frame_sequence=21+i) for i, o in enumerate(newer.observations))
        track.missed_frames = 0
        image = np.zeros((720, 1280, 3), np.uint8)
        x, y, w, h = track.observations[-1].candidate.box
        image[y:y+h, x:x+w] = [230, 140, 40]
        frame = MusicFrame(23, 10.13, 10.13, 10.13, image)
        engine.last_frame, engine.last_frame_sequence = frame, 23
        self.assertTrue(valid_flick_pending(event, engine, 10.13, runtime.tap_trace))
        self.assertIsNone(track.flick_requalification_sequence)
        self.assertEqual(track.action_event_id, 'same-flick')

    def test_prediction_age_alone_cannot_retire_a_queued_unattempted_arrow(self):
        case = self.setup_case([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)
        _, _, _, _, _, track, engine, event = case
        frame = MusicFrame(5, 10.30, 10.30, 10.30, np.zeros((720, 1280, 3), np.uint8))
        engine._associate_lane(track.lane, [], frame, VisualMask.from_image(frame.image, engine.calibration))
        self.assertEqual(track.state, TrackState.FLICK_PENDING)
        self.assertEqual(track.action_event_id, event.event_id)
        self.assertIsNone(track.flick_input_started)

    def test_attempted_arrow_keeps_original_post_hit_retirement(self):
        case = self.setup_case([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)
        _, _, _, _, _, track, engine, _ = case
        track.flick_input_started = 10.
        frame = MusicFrame(5, 10.30, 10.30, 10.30, np.zeros((720, 1280, 3), np.uint8))
        engine._associate_lane(track.lane, [], frame, VisualMask.from_image(frame.image, engine.calibration))
        self.assertEqual(track.state, TrackState.RELEASED)

    def test_unknown_pixels_cannot_revive_soft_cancelled_arrow(self):
        from agent.music.flick_eligibility import valid_flick_pending
        case = self.setup_case([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)
        clock, context, runtime, executor, metrics, track, engine, event = case
        track.flick_requalification_sequence = 1
        engine.last_frame = MusicFrame(3, 9.98, 9.98, 9.98, np.zeros((720, 1280, 3), np.uint8))
        engine.last_frame_sequence = 3
        self.assertFalse(valid_flick_pending(event, engine, 10., runtime.tap_trace))

    def test_an_attempted_arrow_is_never_rearmed_even_after_failed_down(self):
        from agent.music.flick_eligibility import valid_flick_pending
        case = self.setup_case([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)
        clock, context, runtime, executor, metrics, track, engine, event = case
        track.flick_input_started = 10.
        self.assertFalse(valid_flick_pending(event, engine, 10.001, runtime.tap_trace))
        self.assertTrue(track.action_executed)

    def test_stale_arrow_cannot_invalidate_fresh_double_gold_queue(self):
        from test_tap_hold_chain_v4 import feed, gold
        case = self.setup_case([(9., .60), (9.04, .64), (9.08, .68), (9.12, .72)], now=10., sequence=20)
        clock, context, runtime, executor, metrics, track, engine, event = case
        for sequence, stamp in enumerate((9.90, 9.94, 9.98), 1):
            feed(engine, stamp, sequence, [gold(.76+(stamp-9.90), lane=0, owner_lanes=()),
                                          gold(.76+(stamp-9.90), lane=6, owner_lanes=())])
        pending = [event]+engine.release_events(9.98)
        self.assertEqual(len(pending), 3)
        runtime._execute_due(executor, pending, clock(), metrics, engine, wait=False)
        self.assertFalse(context.calls)
        self.assertEqual(len(pending), 2)
        self.assertTrue(all(e.origin == 'hold_note' for e in pending))
        self.assertFalse(any(r['kind']=='hold_note_cancelled' for r in runtime.tap_trace.records))

    def test_birth_refusal_sampling_is_bounded(self):
        from agent.music.flick_eligibility import record_rejection
        track = flick_track([(1., .60), (1.05, .60), (1.1, .60)])
        engine = engine_for(track, MusicFrame(3, 1.1, 1.1, 1.1, None))
        for _ in range(100):
            record_rejection(track, 'stationary-arrow-origin', 1.1, engine.tap_trace, stage='birth')
        self.assertEqual(len([r for r in engine.tap_trace.records if r['kind']=='flick_qualification']), 1)

    def test_healthy_flick_still_uses_original_waypoints_and_deadline(self):
        case = self.setup_case([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)
        clock, context, runtime, executor, metrics, track, engine, event = case
        runtime._execute_due(executor, [event], clock(), metrics, engine, wait=False)
        self.assertEqual([r['kind'] for r in context.calls], ['TouchDown', 'TouchMove', 'TouchMove', 'TouchMove', 'TouchUp'])
        row = next(r for r in runtime.tap_trace.records if r['kind']=='input')
        self.assertEqual(row['deadline'], event.deadline)

    def test_two_observed_healthy_arrows_preserve_native_calls_at_zero_and_twenty_ms_skew(self):
        for skew in (0., .020):
            with self.subTest(skew=skew):
                case = self.setup_case([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)
                clock, context, runtime, executor, metrics, track, engine, first = case
                other = flick_track([(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], tid=2, lane=1)
                other.gesture = other.flick_direction = NoteGesture.FLICK_LEFT
                other.observations = type(other.observations)([
                    replace(o, candidate=replace(o.candidate, flick_direction=NoteGesture.FLICK_LEFT))
                    for o in other.observations], maxlen=12)
                other.state, other.action_executed = TrackState.FLICK_PENDING, True
                second = replace(first, event_id='other-flick', track_id=2, lane=1,
                    gesture=NoteGesture.FLICK_LEFT, direction=NoteGesture.FLICK_LEFT,
                    deadline=10.+skew, coordinate=(260, 620))
                other.action_event_id, other.predicted_hit_time = second.event_id, 10.094+skew
                engine.tracks[2] = other
                runtime._execute_due(executor, [first, second], clock(), metrics, engine, wait=False)
                guarded = context.calls
                _, plain_context, plain_runtime, plain_executor, plain_metrics = self.setup_case(
                    [(9.86, .60), (9.90, .64), (9.94, .68), (9.98, .72)], sequence=4)[:5]
                plain_runtime._execute_due(plain_executor, [first, second], plain_runtime.clock(),
                    plain_metrics, None, wait=False)
                self.assertEqual(guarded, plain_context.calls)
                rows = [r for r in runtime.tap_trace.records if r['kind'] == 'input']
                self.assertEqual({r['event']: r['deadline'] for r in rows},
                    {first.event_id: first.deadline, second.event_id: second.deadline})

    def test_precise_wait_does_not_outwait_arrow_observation_budget(self):
        case = self.setup_case([(9.80, .60), (9.84, .64), (9.88, .68), (9.92, .72)], now=9.98, sequence=4)
        clock, context, runtime, executor, metrics, track, engine, event = case
        pending = [replace(event, deadline=10.005)]
        def progressing_clock():
            clock.now += .000001
            return clock.now
        runtime.clock = progressing_clock
        runtime._execute_due(executor, pending, clock(), metrics, engine, wait=True)
        self.assertFalse(context.calls)
        self.assertEqual(len(pending), 1)
        self.assertLess(clock(), 9.99)

    def test_down_recheck_after_first_native_call_releases_only_unstarted_reservation(self):
        from agent.music.executor import FlickSubmission
        from agent.music.models import FlickRequest
        clock, context, runtime, executor, metrics = flick_tests.FlickRuntimeV4Tests().setup_runtime(tap_mode=True)
        items = [FlickSubmission(FlickRequest(1, 250, 516, NoteGesture.FLICK_LEFT), 'a', 1, 10.),
                 FlickSubmission(FlickRequest(5, 1030, 516, NoteGesture.FLICK_RIGHT), 'b', 2, 10.)]
        def guard(item):
            return None if item.event_id == 'a' or clock() == 10. else 'visual-age-exceeded'
        receipts = executor.swipe_many(items, qualify=guard)
        self.assertEqual([r.event_id for r in receipts if r.down_call_started is not None], ['a'])
        self.assertFalse(executor._temporary_contacts)
        self.assertNotIn(executor._event_key('b'), executor._used_event_ids)
        self.assertTrue(any(s.event_id=='b' and reason=='visual-age-exceeded' for s, reason in executor.last_flick_deferred))


if __name__ == '__main__':
    unittest.main()
