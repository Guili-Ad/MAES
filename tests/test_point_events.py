"""Independent physical points, including positively observed unowned rings."""
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.music.models import MusicActionEvent, MusicConfig, NoteGesture, NoteTrack, TrackState
from agent.music.tracking import MusicVisionEngine
from test_hold_note_taps import calibration, register_head, tap_engine
from test_tap_hold_chain_v4 import feed, gold
from agent.music.point_events import point_registry
from agent.music.hold_note_events import hold_note_registry


class PointQualificationRegressionTests(unittest.TestCase):
    def test_positive_moving_ring_without_head_is_independently_clickable(self):
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True))
        for i in range(3):
            feed(engine, 1. + i * .1, i, [gold(.80 + i * .05, owner_lanes=())])
        events = engine.release_events(1.2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].gesture, NoteGesture.TAP)
        self.assertIsNone(events[0].owner_id)

    def test_new_head_does_not_veto_preobserved_physical_ring(self):
        engine, old = tap_engine()
        for i in range(3):
            feed(engine, .8 + i * .1, i, [gold(.60 + i * .05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        new = NoteTrack(6, 2, gesture=NoteGesture.HOLD_START, state=TrackState.HOLD_PENDING)
        engine.tracks[6] = new
        register_head(engine, new, now=1.05)
        feed(engine, 1.1, 3, [gold(.85)])
        self.assertTrue(engine.tap_hold_chain.eligible(physical, 1.1))
        self.assertEqual(len(engine.release_events(1.1)), 1)

    def test_cross_lane_ring_without_original_head_lane_is_clickable(self):
        engine, _ = tap_engine()
        for i in range(3):
            feed(engine, 1. + i * .1, i, [gold(.80 + i * .05, 6, owner_lanes=(5,))])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        self.assertIsNone(physical.owner)
        events = engine.release_events(1.2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].lane, 6)

    def test_stationary_ring_cannot_become_a_moving_ring_identity(self):
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True))
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.65, owner_lanes=())])
        old_id = next(iter(engine.sustain_tracker.markers))
        self.assertEqual(engine.release_events(1.2), [])
        for i in range(3):
            feed(engine, 1.3+i*.1, 3+i, [gold(.75+i*.05, owner_lanes=())])
        events = engine.release_events(1.5)
        self.assertEqual(len(events), 1)
        self.assertNotEqual(events[0].marker_id, old_id)

    def test_unowned_gold_local_recovery_preserves_identity(self):
        from unittest.mock import patch
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True))
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.80+i*.05, owner_lanes=())])
        event = engine.release_events(1.2)[0]
        recovered = gold(.95, topology='unknown', owner_lanes=())
        with patch('agent.music.gold_recovery.recover_gold_markers',
                   return_value={event.marker_id: recovered}) as recover:
            feed(engine, 1.3, 3, [])
        recover.assert_called_once()
        marker = engine.sustain_tracker.markers[event.marker_id]
        self.assertIsNone(marker.owner)
        self.assertEqual(marker.last_seen_time, 1.3)

    def test_local_search_prediction_needs_physical_ring_not_owner(self):
        from agent.music.gold_recovery import _prediction
        from test_hold_note_taps import make_frame
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True))
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.70+i*.05, owner_lanes=())])
        marker = next(iter(engine.sustain_tracker.markers.values()))
        descriptor = engine.sustain_tracker.descriptors[marker.marker_id]
        self.assertIsNotNone(_prediction(marker, descriptor, make_frame(1.3, 3), engine.config))

    def test_plain_bonus_input_does_not_create_ribbon_anchor(self):
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True))
        track = NoteTrack(7, 2, gesture=NoteGesture.HOLD_START,
                          visual_family='bonus', timing_profile='bonus', point_mode=True)
        engine.tracks[7] = track
        event = MusicActionEvent('bonus-7', 7, 2, NoteGesture.TAP, 1., (480, 620),
                                 visual_family='bonus', timing_profile='bonus')
        engine.tap_hold_chain.acknowledge_head(event, self_receipt(1.))
        self.assertEqual(engine.tap_hold_chain.anchors, {})


def self_receipt(start, *, run_id='', segment_id=None, error='', up=None):
    return SimpleNamespace(down_call_started=start, down_call_finished=start+.001,
        up_call_finished=start+.002 if up is None else up, error=error,
        run_id=run_id, segment_id=segment_id)


class PointRegistryTests(unittest.TestCase):
    def setUp(self):
        self.engine, _ = tap_engine()
        self.registry = point_registry(self.engine)

    def adopt(self, key, *, family='ordinary', source='track', event_id=None):
        event = MusicActionEvent(event_id or f'event-{source}-{key}', key, 2,
                                 NoteGesture.TAP, 1., (480, 620))
        return self.registry.adopt(event, source=source, key=key,
                                   family=family, timing_profile=family)

    def test_facade_and_all_families_share_one_registry(self):
        self.assertIs(hold_note_registry(self.engine), self.registry)
        events = [self.adopt(i, family=family) for i, family in enumerate(
            ('ordinary', 'bonus', 'yellow_head', 'gold_ring'), 1)]
        self.assertEqual(len({e.physical_id for e in events}), 4)
        self.assertEqual(len(self.registry.states), 4)
        self.assertTrue(all(e.gesture == NoteGesture.TAP for e in events))

    def test_family_or_prediction_change_preserves_original_event_id(self):
        original = self.adopt(1)
        changed = self.adopt(1, family='bonus', event_id='new-name')
        self.assertEqual(changed.event_id, original.event_id)
        self.assertEqual(changed.physical_id, original.physical_id)
        revised = self.registry.revise(replace(changed, deadline=1.7))
        self.assertEqual(revised.event_id, original.event_id)
        self.assertEqual(len(self.registry.states), 1)

    def test_optional_metadata_loss_cannot_reset_physical_lifecycle(self):
        event = self.adopt(1, family='yellow_head')
        missing = replace(event, physical_id=None, visual_family='', timing_profile='')
        updated = self.registry.refine(self.engine, [missing], .8)[0]
        self.assertEqual(updated.physical_id, event.physical_id)
        self.assertEqual(updated.visual_family, 'yellow_head')
        self.assertEqual(updated.timing_profile, 'yellow_head')
        self.registry.cancel(updated.event_id, 'alias-duplicate')
        self.assertEqual(self.registry.refine(self.engine, [updated], .8), [])

    def test_head_queue_is_not_ribbon_metadata_pending_gold(self):
        self.adopt(1, family='yellow_head')
        self.assertFalse(self.registry.has_pending(None))

    def test_down_attempted_is_a_tombstone_even_when_input_fails(self):
        event = self.adopt(1)
        receipt = self_receipt(1., error='down-call-failed')
        receipt.down_call_finished = None
        receipt.up_call_finished = None
        self.registry.acknowledge(self.engine, event, receipt)
        self.assertFalse(self.registry.cancel(event.event_id, 'retry'))
        self.assertIsNone(self.adopt(1))
        self.assertIsNone(self.registry.revise(event))

    def test_unstarted_cancelled_identity_can_legally_reappear(self):
        original = self.adopt(1)
        self.assertTrue(self.registry.cancel(original.event_id, 'no-fresh-evidence'))
        recovered = self.adopt(1, family='yellow_head', event_id='different-name')
        self.assertEqual(recovered.event_id, original.event_id)
        self.assertEqual(recovered.physical_id, original.physical_id)
        self.assertFalse(self.registry.states[original.event_id].cancelled)

    def test_cross_source_alias_preserves_canonical_event_and_cancels_duplicate(self):
        ordinary = self.adopt(1)
        gold_event = self.adopt(5, family='gold_ring', source='gold')
        physical = self.registry.alias('gold', 5, 'track', 1)
        self.assertEqual(physical, ordinary.physical_id)
        self.assertTrue(self.registry.states[gold_event.event_id].cancelled)
        aliased = self.adopt(5, family='gold_ring', source='gold')
        self.assertEqual(aliased.event_id, ordinary.event_id)
        self.assertEqual(aliased.physical_id, ordinary.physical_id)

    def test_alias_prefers_started_identity_and_never_retries_it(self):
        ordinary = self.adopt(1)
        gold_event = self.adopt(5, family='gold_ring', source='gold')
        self.registry.acknowledge(self.engine, ordinary, self_receipt(1.))
        self.registry.alias('track', 1, 'gold', 5)
        self.assertTrue(self.registry.states[gold_event.event_id].cancelled)
        self.assertIsNone(self.adopt(5, family='gold_ring', source='gold'))

    def test_two_sent_aliases_keep_both_input_facts(self):
        a, b = self.adopt(1), self.adopt(2)
        self.registry.acknowledge(self.engine, a, self_receipt(1.))
        self.registry.acknowledge(self.engine, b, self_receipt(1.1))
        self.registry.alias('track', 1, 'track', 2)
        self.assertEqual(self.registry.states[a.event_id].started, 1.)
        self.assertEqual(self.registry.states[b.event_id].started, 1.1)
        self.assertTrue(any(r['kind'] == 'point_identity_sent_conflict'
                            for r in self.engine.tap_trace.records))

    def test_resume_segment_has_fresh_namespace_and_rejects_old_receipt(self):
        old = self.adopt(1)
        old_segment = self.engine.tap_trace.segment_id
        self.engine.tap_trace.segment_id += 1
        self.registry = point_registry(self.engine)
        new = self.adopt(1)
        self.assertEqual(old.event_id, new.event_id)
        self.assertFalse(self.registry.acknowledge(self.engine, new,
            self_receipt(1., run_id=self.engine.tap_trace.run_id, segment_id=old_segment)))
        self.assertIsNone(self.registry.states[new.event_id].started)


if __name__ == '__main__':
    unittest.main()
