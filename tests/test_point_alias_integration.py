"""A physical alias preserves input identity, not its obsolete visual source."""
from dataclasses import replace
import unittest

from test_hold_note_taps import make_frame
from test_point_events import self_receipt
from test_tap_pipeline import observed_track
from test_tap_hold_chain_v4 import tap_engine

from agent.music.models import MusicActionEvent, NoteGesture
from agent.music.point_events import point_registry


class PointAliasIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.engine, _ = tap_engine()
        self.engine.tracks = {}
        for tid in (1, 2):
            track = observed_track(tid, 2, .7, hit=1.5)
            track.point_mode = True
            track.visual_family = track.timing_profile = 'ordinary'
            self.engine.tracks[tid] = track
        self.engine.last_frame = make_frame(1., 2)
        self.engine.last_frame_sequence = 2
        self.registry = point_registry(self.engine)

    def adopt(self, tid, *, deadline=1.375):
        event = MusicActionEvent(f'first-{tid}', tid, 2, NoteGesture.TAP,
            deadline, (5, 6), tap_reference_hit_time=1.5)
        return self.registry.adopt_track(event, self.engine.tracks[tid])

    def bind(self):
        self.engine.tap_physical_aliases[2] = 1
        return self.registry.bind_track_source(2, 1, 1.)

    def test_first_shadow_event_transfers_to_existing_canonical_visual_track(self):
        first = self.adopt(2)
        self.bind()
        authority = self.registry.state_for('track', 1).event
        self.assertEqual(authority.event_id, first.event_id)
        self.assertEqual(authority.physical_id, first.physical_id)
        self.assertEqual(authority.track_id, 1)
        self.assertEqual(authority.coordinate, self.engine._lane_point(2))
        pending = self.engine.refine_pending([first], 1.)
        self.assertEqual([event.event_id for event in pending], [first.event_id])
        self.assertFalse(self.registry.states[first.event_id].cancelled)

    def test_old_and_readopted_same_origin_pending_representatives_become_one(self):
        first = self.adopt(2)
        self.bind()
        current = self.adopt(1)
        self.assertEqual(current.event_id, first.event_id)
        self.assertEqual(self.registry.canonical_event(first).track_id, 1)
        pending = self.engine.refine_pending([first, current], 1.)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].track_id, 1)
        self.assertFalse(self.registry.states[first.event_id].cancelled)

    def test_unfrozen_transfer_uses_current_profile_prediction_and_capture(self):
        first = self.adopt(2)
        canonical = self.engine.tracks[1]
        canonical.predicted_hit_time = 1.6
        canonical.visual_family = canonical.timing_profile = 'yellow_head'
        canonical.gesture = NoteGesture.HOLD_START
        self.bind()
        event = self.registry.state_for('track', 1).event
        self.assertAlmostEqual(event.deadline,
            1.6-self.engine.config.hold_start_action_advance_ms/1000.)
        self.assertEqual(event.event_id, first.event_id)
        self.assertEqual(event.timing_profile, 'yellow_head')
        self.assertEqual(event.gesture, NoteGesture.TAP)
        self.assertEqual(event.source_capture_finished,
                         self.engine.last_frame.capture_finished)

    def test_frozen_deadline_is_not_rewritten_by_visual_source_transfer(self):
        first = self.adopt(2, deadline=1.015)
        self.engine.tracks[1].predicted_hit_time = 1.9
        self.bind()
        event = self.registry.state_for('track', 1).event
        self.assertEqual(event.deadline, first.deadline)
        self.assertTrue(event.tap_frozen)
        self.assertEqual(event.tap_reference_hit_time, first.tap_reference_hit_time)

    def test_attempted_down_is_inherited_without_rewriting_actual_input_event(self):
        first = self.adopt(2)
        self.registry.acknowledge(self.engine, first, self_receipt(.9))
        before = self.registry.states[first.event_id].event
        self.bind()
        canonical = self.engine.tracks[1]
        self.assertEqual(canonical.tap_input_started, .9)
        self.assertEqual(canonical.tap_input_completed, .902)
        self.assertEqual(canonical.action_event_id, first.event_id)
        self.assertTrue(canonical.action_executed)
        self.assertEqual(canonical.tap_executed_hit_time, first.tap_reference_hit_time)
        self.assertEqual(self.registry.states[first.event_id].event, before)
        self.assertIsNone(self.adopt(1))

    def test_two_actual_down_facts_remain_distinct_after_conflicting_alias(self):
        a, b = self.adopt(1), self.adopt(2)
        self.registry.acknowledge(self.engine, a, self_receipt(.8))
        self.registry.acknowledge(self.engine, b, self_receipt(.9))
        self.bind()
        self.assertNotEqual(self.registry.source_ids[('track', 1)],
                            self.registry.source_ids[('track', 2)])
        self.assertEqual(self.registry.states[a.event_id].started, .8)
        self.assertEqual(self.registry.states[b.event_id].started, .9)
        self.assertEqual(self.registry.states[a.event_id].event, a)
        self.assertEqual(self.registry.states[b.event_id].event, b)

    def test_duplicate_representatives_are_removed_without_aliasing_distinct_downs(self):
        first = self.adopt(1)
        result = self.registry.refine(self.engine, [first, replace(first)], 1.)
        self.assertEqual(result, [first])
        self.assertEqual(len(self.registry.states), 1)


if __name__ == '__main__':
    unittest.main()
