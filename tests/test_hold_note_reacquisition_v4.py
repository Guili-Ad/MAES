"""Revoked unsent gold events refit before freezing; sent identities never retry."""
import unittest
from types import SimpleNamespace

from test_hold_note_taps import tap_engine
from test_tap_hold_chain_v4 import gold
from agent.music.models import MusicFrame
from agent.music.hold_note_events import hold_note_registry


class HoldNoteReacquisitionTests(unittest.TestCase):
    def make_old_event(self):
        engine, owner = tap_engine()
        chain = engine.tap_hold_chain
        # Existing physically confirmed motion; source times are observations,
        # not event fields patched to force an expected output.
        for sequence, (stamp, progress) in enumerate(((1.7, .80), (1.8, .85), (1.9, .90)), 1):
            frame = MusicFrame(sequence, stamp - .008, stamp + .008, stamp, None)
            engine.sustain_tracker.observe(7, gold(progress), frame, owner=owner.track_id)
        chain.last_frame_sequence = 3
        chain.periods.extend([.1, .1])
        registry = hold_note_registry(engine)
        event = registry.plan(engine, 1.9)[0]
        self.assertAlmostEqual(event.deadline, 1.975)
        return engine, registry, event

    def revoke_old_event(self, engine, registry, event):
        frozen = registry.refine(engine, [event], 1.96)[0]
        self.assertTrue(frozen.tap_frozen)
        engine.tap_hold_chain.last_frame_sequence = 4
        self.assertEqual(registry.refine(engine, [frozen], 2.105), [])
        state = registry.states[event.event_id]
        self.assertTrue(state.cancelled)
        self.assertIsNone(state.started)
        return state

    def positive_reappearance(self, engine, progress):
        # Same physical ID, fresh captured-frame evidence supplied by tracking.
        # The registry must not substitute the old cutoff for this prediction.
        frame = MusicFrame(5, 2.142, 2.158, 2.15, None)
        marker = engine.sustain_tracker.observe(7, gold(progress), frame, owner=5)
        engine.tap_hold_chain.last_frame_sequence = 5
        self.assertTrue(engine.tap_hold_chain.eligible(marker, frame.midpoint))
        return frame, marker

    def test_revoked_expired_deadline_uses_new_prediction_before_freeze(self):
        engine, registry, original = self.make_old_event()
        state = self.revoke_old_event(engine, registry, original)
        frame, marker = self.positive_reappearance(engine, .93)
        expected_hit = marker.predicted_hit(engine.calibration.trigger_progress)
        expected_deadline = expected_hit - engine.config.tap_action_advance_ms / 1000.
        self.assertGreater(expected_deadline, frame.midpoint + .02)
        recovered = registry.plan(engine, frame.midpoint)
        self.assertEqual(len(recovered), 1)
        event = recovered[0]
        self.assertEqual(event.event_id, original.event_id)
        self.assertEqual((event.origin, event.owner_id, event.marker_id, event.track_id),
                         (original.origin, original.owner_id, original.marker_id, original.track_id))
        self.assertAlmostEqual(event.tap_reference_hit_time, expected_hit)
        self.assertAlmostEqual(event.deadline, expected_deadline)
        self.assertFalse(event.tap_frozen)
        self.assertIsNone(event.tap_group_id)
        self.assertEqual((event.source_capture_started, event.source_capture_finished),
                         (frame.capture_started, frame.capture_finished))
        self.assertIs(registry.states[event.event_id], state)
        self.assertFalse(state.cancelled)
        self.assertIsNone(state.started)
        self.assertEqual(len(registry.states), 1)

    def test_reappearance_inside_twenty_ms_freezes_new_not_old_cutoff(self):
        engine, registry, original = self.make_old_event()
        self.revoke_old_event(engine, registry, original)
        frame, marker = self.positive_reappearance(engine, .956)
        expected_deadline = (marker.predicted_hit(engine.calibration.trigger_progress)
                             - engine.config.tap_action_advance_ms / 1000.)
        self.assertGreater(expected_deadline, frame.midpoint)
        self.assertLessEqual(expected_deadline, frame.midpoint + .02)
        recovered = registry.plan(engine, frame.midpoint)[0]
        self.assertEqual(recovered.event_id, original.event_id)
        self.assertTrue(recovered.tap_frozen)
        self.assertAlmostEqual(recovered.deadline, expected_deadline)
        self.assertGreater(recovered.deadline, original.deadline)
        self.assertEqual(recovered.source_capture_finished, frame.capture_finished)

    def test_revoked_reappearance_outside_normal_horizon_stays_cancelled(self):
        engine, registry, original = self.make_old_event()
        state = self.revoke_old_event(engine, registry, original)
        frame, marker = self.positive_reappearance(engine, .92)
        self.assertGreater(marker.predicted_hit(engine.calibration.trigger_progress) - frame.midpoint,
                           engine.config.hold_note_tap_horizon_ms / 1000.)
        self.assertEqual(registry.plan(engine, frame.midpoint), [])
        self.assertTrue(state.cancelled)
        self.assertIsNone(state.started)

    def test_started_partial_or_successful_input_tombstone_never_revives(self):
        for completed, error in ((None, 'forced-up-failure'), (1.967, '')):
            with self.subTest(completed=completed, error=error):
                engine, registry, original = self.make_old_event()
                receipt = SimpleNamespace(down_call_started=1.965, down_call_finished=1.966,
                                          up_call_finished=completed, error=error)
                self.assertTrue(registry.acknowledge(engine, original, receipt))
                state = registry.states[original.event_id]
                engine.tap_hold_chain.last_frame_sequence = 4
                self.assertEqual(registry.refine(engine, [original], 2.105), [])
                frame, _ = self.positive_reappearance(engine, .93)
                self.assertEqual(registry.plan(engine, frame.midpoint), [])
                self.assertIs(registry.states[original.event_id], state)
                self.assertEqual(state.started, 1.965)
                self.assertEqual(state.completed, completed)
                self.assertEqual(len(registry.states), 1)


if __name__ == '__main__':
    unittest.main()
