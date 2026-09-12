from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at
from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture, NoteTrack, TrackObservation, TrackState
from agent.music.tap_policy import TapTimingPolicy
from agent.music.tracking import MusicVisionEngine
from agent.music.runtime import MusicRuntime, RuntimeMetrics
from agent.music.executor import MusicActionExecutor
from agent.music.executor import MusicTouchError
from agent.music.models import LaneInputState, MusicFrame
from agent.music.tap_chords import TapChordManager
from agent.music.tap_dispatch import due_tap_batches
from agent.music.tap_tracking import associate_taps, retire_converged_shadows
from agent.music.tap_trace import TapTrace
from agent.music.tracking import LaneProjection
import numpy as np


def observed_track(tid, lane, progress, *, speed=1., hit=1.5):
    cal = calibration()
    track = NoteTrack(tid, lane, speed=speed, predicted_hit_time=hit)
    for sequence in range(3):
        p = progress - (2 - sequence) * .03
        note = candidate_at(cal, lane, p)
        track.observations.append(TrackObservation(sequence, .94 + sequence * .03, note.center, p, note))
    return track


def tap_event(tid, lane, deadline, group=None):
    return MusicActionEvent(str(tid), tid, lane, NoteGesture.TAP, deadline,
                            (160 + lane * 160, 620), tap_group_id=group)


class TapTransportTests(unittest.TestCase):
    def make_executor(self, *, multi=True, capacity=10):
        return MusicActionExecutor(SimpleNamespace(), 1280, 720,
                                   MusicConfig(lane_count=7, enable_holds=True, max_contacts=capacity),
                                   advanced=True, multi_touch=multi)

    def test_duplicate_ids_within_and_across_batches_are_not_resent(self):
        executor = self.make_executor()
        with patch.object(executor, '_run') as run:
            receipts = executor.tap_many([(1, 320, 620), (1, 320, 620), (5, 960, 620)], event_ids=['a', 'a', 'b'])
            executor.tap_many([(1, 320, 620), (5, 960, 620)], event_ids=['a', 'b'])
        self.assertEqual([r.event_id for r in receipts], ['a', 'b'])
        self.assertEqual(run.call_count, 4)
        self.assertTrue(all(r.down_call_finished is not None and r.up_call_finished is not None for r in receipts))

    def test_mismatched_event_ids_fail_before_any_input(self):
        executor = self.make_executor()
        with patch.object(executor, '_run') as run, self.assertRaises(ValueError):
            executor.tap_many([(1, 320, 620)], event_ids=[])
        run.assert_not_called()

    def test_failed_second_down_cleans_both_and_does_not_retry_either(self):
        executor = self.make_executor()
        def action(kind, param, *args, **kwargs):
            if kind.value == 'TouchDown' and param.contact == 1:
                raise RuntimeError('injected second down failure')
        with patch.object(executor, '_run', side_effect=action) as run:
            with self.assertRaises(MusicTouchError) as captured:
                executor.tap_many([(1, 320, 620), (5, 960, 620)], event_ids=['a', 'b'])
            self.assertEqual(len(captured.exception.receipts), 2)
            self.assertIsNotNone(captured.exception.receipts[0].down_call_finished)
            self.assertIsNone(captured.exception.receipts[1].down_call_finished)
            calls = run.call_count
            self.assertEqual(executor.tap_many([(1, 320, 620), (5, 960, 620)], event_ids=['a', 'b']), [])
            self.assertEqual(run.call_count, calls)
        self.assertFalse(executor._temporary_contacts)

    def test_failed_up_keeps_contact_reserved_until_cleanup(self):
        executor = self.make_executor()
        def action(kind, *args, **kwargs):
            if kind.value == 'TouchUp':
                raise RuntimeError('injected up failure')
        with patch.object(executor, '_run', side_effect=action), self.assertRaises(MusicTouchError):
            executor.tap(1, 320, 620, event_id='a')
        self.assertEqual(executor._temporary_contacts, {0})
        with patch.object(executor, '_run'):
            executor.release_all()
        self.assertFalse(executor._temporary_contacts)

    def test_capacity_fallback_preserves_existing_hold(self):
        executor = self.make_executor(capacity=2)
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=99, contact_started=.5)
        with patch.object(executor, '_run') as run:
            executor.tap_many([(1, 320, 620), (5, 960, 620)], event_ids=['a', 'b'])
        self.assertEqual([c.args[0].value for c in run.call_args_list], ['TouchDown', 'TouchUp'] * 2)
        self.assertEqual({c.args[1].contact for c in run.call_args_list}, {1})
        self.assertEqual(executor.active_contacts, {3: 0})
        self.assertEqual(executor.tap_fallbacks, ['contact-capacity'])

    def test_no_multitouch_uses_serial_compatibility(self):
        executor = self.make_executor(multi=False)
        with patch.object(executor, '_run') as run:
            executor.tap_many([(1, 320, 620), (5, 960, 620)], event_ids=['a', 'b'])
        self.assertEqual([c.args[0].value for c in run.call_args_list], ['TouchDown', 'TouchUp'] * 2)
        self.assertEqual(executor.tap_fallbacks, ['no-multitouch'])


class TapOrderingTests(unittest.TestCase):
    def test_same_lane_reversed_deadlines_follow_observed_order(self):
        engine = SimpleNamespace(tracks={1: observed_track(1, 3, .80), 2: observed_track(2, 3, .70)})
        events = [tap_event(2, 3, .98), tap_event(1, 3, 1.0)]
        self.assertEqual(due_tap_batches(events, .99, engine), [])
        batches = due_tap_batches(events, 1., engine)
        self.assertEqual([[e.track_id for e in batch] for batch in batches], [[1], [2]])
        self.assertEqual([e.deadline for e in events], [.98, 1.])

    def test_only_declared_chords_batch_and_successive_chords_stay_ordered(self):
        events = [tap_event(1, 1, 1., 'pair-a'), tap_event(2, 5, 1., 'pair-a'),
                  tap_event(3, 1, 1.1, 'pair-b'), tap_event(4, 5, 1.1, 'pair-b')]
        self.assertEqual([[e.track_id for e in b] for b in due_tap_batches(events, 1.2)], [[1, 2], [3, 4]])
        singles = [tap_event(5, 2, 1.), tap_event(6, 4, 1.)]
        self.assertEqual(len(due_tap_batches(singles, 1.)), 2)

    def test_unready_chord_never_dispatches_one_side(self):
        events = [tap_event(1, 1, 1., 'pair'), tap_event(2, 5, 1.1, 'pair')]
        self.assertEqual(due_tap_batches(events, 1.05), [])

    def test_conflicting_chord_orders_do_not_deadlock_or_split_groups(self):
        events = [tap_event(1, 1, 1., 'a'), tap_event(2, 5, 1., 'a'),
                  tap_event(3, 1, 1.01, 'b'), tap_event(4, 5, 1.01, 'b')]
        engine = SimpleNamespace(tracks={1: observed_track(1, 1, .8), 2: observed_track(2, 5, .7),
                                        3: observed_track(3, 1, .7), 4: observed_track(4, 5, .8)})
        self.assertEqual(due_tap_batches(events, 1.005, engine), [])
        self.assertEqual([[e.track_id for e in b] for b in due_tap_batches(events, 1.02, engine)], [[1, 2], [3, 4]])


class TapChordTests(unittest.TestCase):
    def setUp(self):
        self.config = MusicConfig(lane_count=7)
        self.trace = TapTrace(self.config)
        self.manager = TapChordManager(TapTimingPolicy(self.config, 7), self.trace)
        self.tracks = {1: observed_track(1, 1, .8), 2: observed_track(2, 5, .8)}
        self.tracks[1].linked_partner_id, self.tracks[2].linked_partner_id = 2, 1

    def test_shared_prediction_updates_until_whole_group_freezes(self):
        events = [tap_event(1, 1, 1.3), tap_event(2, 5, 1.4)]
        events = self.manager.refine(events, self.tracks, 1., 2)
        self.assertEqual([e.deadline for e in events], [1.375, 1.375])
        self.tracks[1].predicted_hit_time = 1.6
        events = self.manager.refine(events, self.tracks, 1.1, 2)
        self.assertEqual([e.deadline for e in events], [1.425, 1.425])
        events = self.manager.refine(events, self.tracks, 1.410, 2)
        self.tracks[2].predicted_hit_time = 2.
        events = self.manager.refine(events, self.tracks, 1.411, 2)
        self.assertTrue(all(e.tap_frozen and e.deadline == 1.425 for e in events))

    def test_stale_or_one_way_partner_dissolves_without_losing_survivor(self):
        events = self.manager.refine([tap_event(1, 1, 1.3), tap_event(2, 5, 1.4)], self.tracks, 1., 2)
        self.tracks[2].state = TrackState.LOST
        events = self.manager.refine(events, self.tracks, 1.1, 2)
        self.assertEqual([e.track_id for e in events], [1])
        self.assertTrue(all(e.tap_group_id is None for e in events))
        self.assertFalse(self.manager.groups)

    def test_already_sent_member_is_never_replayed(self):
        events = self.manager.refine([tap_event(1, 1, 1.3), tap_event(2, 5, 1.4)], self.tracks, 1., 2)
        self.tracks[1].tap_input_started = 1.2
        remaining = self.manager.refine(events, self.tracks, 1.3, 2)
        self.assertEqual([e.track_id for e in remaining], [2])
        self.assertIsNone(remaining[0].tap_group_id)

    def test_hold_link_does_not_enter_tap_manager(self):
        for track in self.tracks.values():
            track.gesture = NoteGesture.HOLD_START
        events = [MusicActionEvent('h1', 1, 1, NoteGesture.HOLD_START, 1.3, (320, 620)),
                  MusicActionEvent('h2', 2, 5, NoteGesture.HOLD_START, 1.4, (960, 620))]
        self.assertEqual(self.manager.refine(events, self.tracks, 1., 2), events)
        self.assertFalse(self.manager.groups)

    def test_pending_head_can_pair_with_late_visible_partner(self):
        from test_longtap_branch import linked_pair_frame
        engine = MusicVisionEngine(calibration(), self.config)
        engine.tracks = self.tracks
        for track in self.tracks.values():
            track.linked_partner_id = None
        self.tracks[1].state = TrackState.TAP_PENDING
        self.tracks[1].action_executed = True
        left, right = (self.tracks[i].observations[-1].candidate for i in (1, 2))
        frame = MusicFrame(2, 1., 1., 1., linked_pair_frame(left, right))
        engine._update_linked_tap_pairs(frame)
        self.assertEqual(self.tracks[1].linked_partner_id, 2)
        self.assertEqual(self.tracks[2].linked_partner_id, 1)

    def test_one_way_pair_never_groups(self):
        self.tracks[2].linked_partner_id = None
        events = self.manager.refine([tap_event(1, 1, 1.3), tap_event(2, 5, 1.4)], self.tracks, 1., 2)
        self.assertTrue(all(e.tap_group_id is None for e in events))

    def test_frozen_survivor_keeps_deadline_on_partner_loss(self):
        events = self.manager.refine([tap_event(1, 1, 1.01), tap_event(2, 5, 1.02)], self.tracks, 1., 2)
        self.tracks[2].state = TrackState.LOST
        self.tracks[1].predicted_hit_time = 2.
        events = self.manager.refine(events, self.tracks, 1.001, 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].deadline, 1.01)
        self.assertTrue(events[0].tap_frozen)


class TapAssociationTests(unittest.TestCase):
    def test_contained_corrupted_shadow_is_cancelled_but_dense_real_heads_are_not(self):
        from dataclasses import replace
        shadow = observed_track(1, 3, .80)
        real = observed_track(2, 3, .82)
        # Reproduce a normal head trajectory falling onto a tiny score particle,
        # then converging onto the already-tracked outer note contour.
        for track in (shadow, real):
            track.state = TrackState.TAP_PENDING
            note = candidate_at(calibration(), 3, .81, 80)
            track.observations.append(TrackObservation(3, 1.03, note.center, .84, note))
        obs = list(shadow.observations)
        obs[1] = replace(obs[1], candidate=replace(obs[1].candidate, box=(630, 450, 6, 6)))
        shadow.observations.clear()
        shadow.observations.extend(obs)
        frame = MusicFrame(3, 1.03, 1.03, 1.03, None)
        retire_converged_shadows({1: shadow, 2: real}, frame, TapTrace(MusicConfig()))
        self.assertEqual(shadow.state, TrackState.LOST)
        self.assertEqual(real.state, TrackState.TAP_PENDING)
        # With two regular contours, overlap or close timestamps alone is not evidence.
        shadow.state = TrackState.TAP_PENDING
        shadow.observations.clear()
        shadow.observations.extend(real.observations)
        retire_converged_shadows({1: shadow, 2: real}, frame, TapTrace(MusicConfig()))
        self.assertEqual(shadow.state, TrackState.TAP_PENDING)

    def test_conflicting_greedy_choice_preserves_two_heads(self):
        tracks = [observed_track(1, 3, .7, speed=.5), observed_track(2, 3, .6, speed=1.6)]
        entries = [(candidate_at(calibration(), 3, p), LaneProjection(3, p, 0., (0., 1.))) for p in (.81, .73)]
        frame = MusicFrame(3, 1.1, 1.1, 1.1, None)
        trace = TapTrace(MusicConfig())
        matches = associate_taps(tracks, entries, frame, MusicConfig(), lambda *_: True, trace)
        self.assertEqual([matches[i].track_id for i in (0, 1)], [1, 2])
        self.assertEqual(trace.records[-1]['kind'], 'association')

    def test_mixed_hold_component_keeps_legacy_assignment(self):
        tracks = [observed_track(1, 3, .7, speed=.5), observed_track(2, 3, .6, speed=1.6)]
        tracks[1].gesture = NoteGesture.HOLD_START
        entries = [(candidate_at(calibration(), 3, p), LaneProjection(3, p, 0., (0., 1.))) for p in (.81, .73)]
        matches = associate_taps(tracks, entries, MusicFrame(3, 1.1, 1.1, 1.1, None), MusicConfig(), lambda *_: True, TapTrace(MusicConfig()))
        self.assertEqual(matches[0].track_id, 2)
        self.assertEqual(matches[1].track_id, 1)

    def test_dropout_and_reappearance_preserve_identity_without_forcing_match(self):
        tracks = [observed_track(1, 3, .7, speed=.8), observed_track(2, 3, .6, speed=.8)]
        trace, config = TapTrace(MusicConfig()), MusicConfig()
        for seq, timestamp, positions, expected in [(3, 1.03, [.724], [1]), (4, 1.06, [.748], [1]),
                                                     (5, 1.09, [.772, .672], [1, 2])]:
            entries = [(candidate_at(calibration(), 3, p), LaneProjection(3, p, 0., (0., 1.))) for p in positions]
            matches = associate_taps(tracks, entries, MusicFrame(seq, timestamp, timestamp, timestamp, None), config, lambda *_: True, trace)
            self.assertEqual([matches[i].track_id for i in range(len(entries))], expected)
            for i, t in matches.items():
                note, projection = entries[i]
                t.observations.append(TrackObservation(seq, timestamp, note.center, projection.progress, note))


class TapLifecycleTests(unittest.TestCase):
    def test_ack_uses_frozen_hit_anchor_after_dense_flag_changes(self):
        from dataclasses import replace
        from agent.music.executor import TapInputReceipt
        config = MusicConfig(lane_count=7)
        runtime = MusicRuntime(SimpleNamespace(), config)
        engine = MusicVisionEngine(calibration(), config)
        track = observed_track(1, 3, .8)
        track.dense_tap = True
        engine.tracks[1] = track
        event = replace(tap_event(1, 3, 1.), tap_reference_hit_time=1.125, tap_frozen=True)
        receipt = TapInputReceipt('1', 3, 0, 1., 1.005, 1.010)
        runtime._acknowledge_taps([receipt], [event], engine, RuntimeMetrics())
        self.assertEqual(track.tap_executed_hit_time, 1.125)
        self.assertEqual(track.tap_input_completed, 1.010)

    def test_queued_head_is_not_retired_before_input_but_ack_uses_fixed_hit_anchor(self):
        config, cal = MusicConfig(lane_count=7), calibration()
        engine = MusicVisionEngine(cal, config)
        track = observed_track(1, 3, .8)
        track.state, track.action_executed, track.predicted_hit_time = TrackState.TAP_PENDING, True, .1
        engine.tracks[1] = track
        from agent.music.vision import VisualMask
        frame = MusicFrame(3, 1.1, 1.1, 1.1, np.zeros((720, 1280, 3), dtype=np.uint8))
        visual = VisualMask.from_image(frame.image, cal)
        engine._associate_lane(3, [], frame, visual)
        self.assertEqual(track.state, TrackState.TAP_PENDING)
        track.tap_input_started, track.tap_executed_hit_time, track.predicted_hit_time = .8, .925, 9.
        engine._associate_lane(3, [], frame, visual)
        self.assertEqual(track.state, TrackState.RELEASED)

    def test_diagnostics_are_bounded_and_reset_engine_has_no_groups(self):
        config = MusicConfig(lane_count=7)
        trace = TapTrace(config, capacity=2)
        for i in range(5):
            trace.add('test', value=i)
        self.assertEqual(len(trace.records), 2)
        self.assertEqual(trace.dropped, 3)
        self.assertFalse(MusicVisionEngine(calibration(), config).tap_chords.groups)

    def test_fast_tap_connector_is_pixel_equivalent(self):
        from test_longtap_branch import linked_pair_frame
        from agent.music.vision import linked_tap_pair_present
        left, right = candidate_at(calibration(), 1, .7), candidate_at(calibration(), 5, .7)
        linked = linked_pair_frame(left, right)
        noise = np.random.default_rng(7).integers(0, 256, linked.shape, dtype=np.uint8)
        for image in (linked, np.zeros_like(linked), noise):
            self.assertEqual(linked_tap_pair_present(image, left, right),
                             linked_tap_pair_present(image, left, right, fast_channels=True))


class TapPipelineRegressionTests(unittest.TestCase):
    def test_new_visual_prediction_is_not_overridden_by_old_cadence(self):
        track = NoteTrack(1, 3, predicted_hit_time=2.0, tap_cadence_hit_time=2.2)
        track.predicted_hit_time = 2.05
        self.assertEqual(TapTimingPolicy.hit_time(track), 2.05)

    def test_overdue_same_lane_notes_have_independent_down_up_cycles(self):
        config = MusicConfig(lane_count=7, enable_holds=True)
        runtime = MusicRuntime(SimpleNamespace(), config, clock=lambda: 1.1)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config, advanced=True, multi_touch=True)
        pending = [MusicActionEvent('a', 1, 3, NoteGesture.TAP, 1.02, (640, 620)),
                   MusicActionEvent('b', 2, 3, NoteGesture.TAP, 1.08, (640, 620))]
        with patch.object(executor, '_run') as run:
            runtime._execute_due(executor, pending, 1.1, RuntimeMetrics())
        actions = [call.args[0].value for call in run.call_args_list]
        self.assertEqual(actions, ['TouchDown', 'TouchUp', 'TouchDown', 'TouchUp'])

    def test_linked_pair_freezes_as_one_unit(self):
        config = MusicConfig(lane_count=7)
        engine = MusicVisionEngine(calibration(), config)
        engine.tracks = {1: NoteTrack(1, 1, predicted_hit_time=1.185, linked_partner_id=2),
                         2: NoteTrack(2, 5, predicted_hit_time=1.185, linked_partner_id=1)}
        for track in engine.tracks.values():
            candidate = candidate_at(engine.calibration, track.lane, 0.8)
            track.observations.append(TrackObservation(0, 1., candidate.center, 0.8, candidate))
        pending = [MusicActionEvent('a', 1, 1, NoteGesture.TAP, 1.019, (320, 620)),
                   MusicActionEvent('b', 2, 5, NoteGesture.TAP, 1.021, (960, 620))]
        refined = engine.refine_pending(pending, 1.0)
        self.assertEqual(refined[0].deadline, refined[1].deadline)


if __name__ == '__main__':
    unittest.main()
