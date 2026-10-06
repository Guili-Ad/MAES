from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.music.models import NoteGesture, NoteTrack, TrackState
from test_hold_note_taps import add_marker, detection, make_frame, tap_engine, register_head, linked_owner


def shift_marker(engine, marker_id, dx):
    marker = engine.sustain_tracker.markers[marker_id]
    marker.observations = type(marker.observations)(
        [replace(item, center=(item.center[0] + dx, item.center[1]))
         for item in marker.observations], maxlen=12)


class HoldNoteEventLifecycleTests(unittest.TestCase):
    def refine(self, engine, pending, now):
        from agent.music.hold_note_events import refine_hold_note_events
        return refine_hold_note_events(engine, pending, now)

    def acknowledge(self, engine, event, *, start=1.9, up=1.91, error=None):
        from agent.music.hold_note_events import acknowledge_hold_note_event
        return acknowledge_hold_note_event(engine, event, SimpleNamespace(
            down_call_started=start, down_call_finished=start + .001,
            up_call_finished=up, error=error))

    def test_close_real_same_lane_markers_are_not_time_deduped(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 3, hit=2.0, now=1.8)
        add_marker(engine, 8, owner.track_id, 3, hit=2.05, now=1.8)
        shift_marker(engine, 8, 20)
        self.assertEqual(len(engine.release_events(1.8)), 2)

    def test_connected_double_hold_markers_share_one_deadline(self):
        engine, owner = tap_engine()
        other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START,
                          state=TrackState.HOLDING, predicted_hit_time=0.)
        owner.linked_partner_id = other.track_id
        other.linked_partner_id = owner.track_id
        engine.tracks[other.track_id] = other
        register_head(engine, other)
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        add_marker(engine, 8, other.track_id, 4, hit=2.02, now=1.8)
        events = engine.release_events(1.8)
        self.assertIsNotNone(events[0].tap_group_id)
        self.assertEqual(events[0].tap_group_id, events[1].tap_group_id)
        self.assertAlmostEqual(events[0].deadline, 1.885)
        self.assertEqual(events[0].deadline, events[1].deadline)

    def test_unconnected_double_hold_markers_do_not_group(self):
        engine, owner = tap_engine()
        other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START,
                          state=TrackState.HOLDING, predicted_hit_time=0.)
        engine.tracks[other.track_id] = other
        register_head(engine, other)
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        add_marker(engine, 8, other.track_id, 4, hit=2.02, now=1.8)
        self.assertTrue(all(event.tap_group_id is None
                            for event in engine.release_events(1.8)))

    def test_event_has_explicit_origin_and_latest_prediction(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 3, hit=2.0, now=1.8)
        event = engine.release_events(1.8)[0]
        self.assertEqual(getattr(event, 'origin', None), 'hold_note')
        self.assertEqual(event.owner_id, owner.track_id)
        self.assertEqual(event.marker_id, 7)
        marker = engine.sustain_tracker.markers[7]
        marker.observations[-1] = replace(marker.observations[-1], progress=.88, lane=4)
        marker.last_seen_time = 1.8
        expected = marker.predicted_hit(1.) - .125
        updated = self.refine(engine, [event], 1.82)[0]
        self.assertEqual(updated.event_id, event.event_id)
        self.assertAlmostEqual(updated.deadline, expected)
        self.assertEqual(updated.lane, marker.lane())
        self.assertEqual(updated.coordinate, engine._lane_point(updated.lane))
        self.assertEqual(updated.source_capture_finished, 1.8)

    def test_group_updates_and_freezes_as_one(self):
        engine, owner = tap_engine()
        other = linked_owner(engine, owner)
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        add_marker(engine, 8, other.track_id, 4, hit=2.02, now=1.8)
        events = engine.release_events(1.8)
        events = self.refine(engine, events, events[0].deadline - .019)
        self.assertTrue(all(event.tap_frozen for event in events))
        deadline = events[0].deadline
        marker = engine.sustain_tracker.markers[7]
        marker.observations[-1] = replace(marker.observations[-1], progress=.86)
        newer = self.refine(engine, events, deadline - .01)
        self.assertTrue(all(event.deadline == deadline for event in newer))

    def test_invalid_group_member_dissolves_without_losing_valid_member(self):
        engine, owner = tap_engine()
        other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START,
                          state=TrackState.HOLDING, predicted_hit_time=0.)
        owner.linked_partner_id, other.linked_partner_id = 6, 5
        engine.tracks[6] = other
        register_head(engine, other)
        add_marker(engine, 7, 5, 2, hit=2.0, now=1.8)
        add_marker(engine, 8, 6, 4, hit=2.02, now=1.8)
        events = engine.release_events(1.8)
        engine.sustain_tracker.markers.pop(8)
        events = self.refine(engine, events, 1.82)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].owner_id, 5)
        self.assertIsNone(events[0].tap_group_id)

    def test_completed_or_partially_started_marker_never_requeues(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        event = engine.release_events(1.8)[0]
        self.acknowledge(engine, event, up=None, error='touch up failed')
        self.assertEqual(self.refine(engine, [event], 1.91), [])
        self.assertEqual(engine.release_events(1.91), [])
        self.assertEqual(owner.state, TrackState.TAP_PENDING)

    def test_mid_checkpoint_completion_does_not_release_owner(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8, exits=2)
        event = engine.release_events(1.8)[0]
        self.acknowledge(engine, event)
        self.assertEqual(owner.state, TrackState.TAP_PENDING)
        self.assertFalse(owner.hold_sustain_final_emitted)

    def test_confirmed_terminal_completion_closes_metadata_without_releasing_point(self):
        engine, owner = tap_engine()
        owner.hold_terminal_confirmed = True
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8, exits=1)
        event = engine.release_events(1.8)[0]
        self.acknowledge(engine, event)
        self.assertEqual(owner.state, TrackState.TAP_PENDING)
        self.assertFalse(owner.hold_sustain_final_emitted)
        from agent.music.tap_hold_chain import AnchorState
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].state, AnchorState.CLOSED)

    def test_weak_single_exit_does_not_release_owner(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8, exits=1)
        # Three positive terminal observations are strong in the new mode;
        # make this genuinely weak/unknown evidence instead of an old flag.
        marker = engine.sustain_tracker.markers[7]
        marker.terminal_votes = marker.sustain_votes = 0
        event = engine.release_events(1.8)[0]
        self.acknowledge(engine, event)
        self.assertEqual(owner.state, TrackState.TAP_PENDING)

    def test_later_real_checkpoint_blocks_terminal_owner_retirement(self):
        engine, owner = tap_engine()
        owner.hold_terminal_confirmed = True
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8, exits=1)
        add_marker(engine, 8, owner.track_id, 2, hit=2.5, now=1.8, exits=2)
        shift_marker(engine, 8, 20)
        event = engine.release_events(1.8)[0]
        self.acknowledge(engine, event)
        self.assertEqual(owner.state, TrackState.TAP_PENDING)

    def test_alias_keeps_event_id_without_replanning_a_marker(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        event = engine.release_events(1.8)[0]
        marker = engine.sustain_tracker.markers.pop(7)
        marker.marker_id = 9
        engine.sustain_tracker.markers[9] = marker
        engine.sustain_tracker.marker_aliases[7] = 9
        revised = self.refine(engine, [event], 1.81)[0]
        self.assertEqual(revised.event_id, event.event_id)
        self.assertEqual(revised.marker_id, 9)
        self.assertEqual(engine.release_events(1.81), [])

    def test_started_uncompleted_marker_blocks_anchor_retirement(self):
        engine, owner = tap_engine()
        owner.hold_release_time = 1.
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        event = engine.release_events(1.8)[0]
        self.acknowledge(engine, event, up=None, error='touch up failed')
        engine.release_events(2.2)
        self.assertEqual(owner.state, TrackState.TAP_PENDING)

    def test_pending_marker_blocks_fallback_anchor_retirement(self):
        engine, owner = tap_engine()
        owner.hold_release_time = 1.0
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        self.assertEqual(len(engine.release_events(1.8)), 1)
        engine.release_events(2.2)
        self.assertEqual(owner.state, TrackState.TAP_PENDING)

    def test_new_segment_new_engine_has_empty_marker_registry(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2.0, now=1.8)
        engine.release_events(1.8)
        other_engine, other_owner = tap_engine()
        add_marker(other_engine, 7, other_owner.track_id, 2, hit=2.0, now=1.8)
        self.assertEqual(len(other_engine.release_events(1.8)), 1)

    def test_late_partner_joins_frozen_side_without_moving_its_deadline(self):
        engine, owner = tap_engine()
        other = linked_owner(engine, owner)
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        left = engine.release_events(1.8)[0]
        left = self.refine(engine, [left], left.deadline - .019)[0]
        self.assertTrue(left.tap_frozen)
        add_marker(engine, 8, other.track_id, 4, hit=2.02, now=1.856)
        new_events = engine.release_events(1.856)
        self.assertEqual(len(new_events), 1)
        combined = self.refine(engine, [left, new_events[0]], 1.856)
        self.assertEqual(combined[0].tap_group_id, combined[1].tap_group_id)
        self.assertEqual(combined[0].deadline, left.deadline)
        self.assertEqual(combined[1].deadline, left.deadline)
        self.assertTrue(all(event.tap_frozen for event in combined))

    def test_frozen_events_with_distinct_deadlines_cannot_be_merged(self):
        engine, owner = tap_engine()
        other = linked_owner(engine, owner)
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        left = engine.release_events(1.8)[0]
        left = self.refine(engine, [left], left.deadline - .019)[0]
        add_marker(engine, 8, other.track_id, 4, hit=2.02, now=1.88)
        right = engine.release_events(1.88)[0]
        # The new right side is already inside its own 20 ms freeze window.
        self.assertNotEqual(left.deadline, right.deadline)
        combined = self.refine(engine, [left, right], 1.88)
        self.assertTrue(all(event.tap_group_id is None for event in combined))

    def test_confirmed_double_hold_with_large_skew_keeps_solo_events(self):
        engine, owner = tap_engine()
        other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START,
                          state=TrackState.HOLDING, predicted_hit_time=0.)
        owner.linked_partner_id, other.linked_partner_id = 6, 5
        engine.tracks[6] = other
        register_head(engine, other)
        add_marker(engine, 7, 5, 2, hit=2., now=1.8)
        add_marker(engine, 8, 6, 4, hit=2.05, now=1.8)
        events = engine.release_events(1.8)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.tap_group_id is None for event in events))

    def test_successful_first_side_of_double_terminal_does_not_resend_on_failure(self):
        engine, owner = tap_engine()
        other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START,
                          state=TrackState.HOLDING, predicted_hit_time=0.,
                          hold_terminal_confirmed=True)
        owner.hold_terminal_confirmed = True
        owner.linked_partner_id, other.linked_partner_id = 6, 5
        engine.tracks[6] = other
        register_head(engine, other)
        add_marker(engine, 7, 5, 2, hit=2., now=1.8, exits=1)
        add_marker(engine, 8, 6, 4, hit=2.02, now=1.8, exits=1)
        events = engine.release_events(1.8)
        self.acknowledge(engine, events[0])
        self.acknowledge(engine, events[1], up=None, error='touch-up failed')
        self.assertEqual(owner.state, TrackState.TAP_PENDING)
        self.assertEqual(other.state, TrackState.HOLDING)
        self.assertEqual(self.refine(engine, events, 1.91), [])
        self.assertEqual(engine.release_events(1.91), [])

    def test_nonfinite_prediction_cannot_execute_invalid_visual_fit(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        event = engine.release_events(1.8)[0]
        marker = engine.sustain_tracker.markers[7]
        marker.observations[-1] = replace(marker.observations[-1], progress=float('nan'))
        self.assertEqual(self.refine(engine, [event], 1.81), [])

    def test_late_visual_correction_preserves_real_deadline_and_diagnostic(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        event = engine.release_events(1.8)[0]
        marker = engine.sustain_tracker.markers[7]
        marker.observations[-1] = replace(marker.observations[-1], progress=.99)
        expected = marker.predicted_hit(1.) - .125
        revised = self.refine(engine, [event], 1.82)[0]
        self.assertLess(expected, 1.82)
        self.assertAlmostEqual(revised.deadline, expected)
        records = [row for row in engine.tap_trace.records if row.get('kind') == 'hold_note_refine']
        self.assertAlmostEqual(records[-1]['correction_late_ms'], (1.82 - expected) * 1000.)

    def test_group_late_correction_freezes_updated_mean_not_old_deadline(self):
        engine, owner = tap_engine()
        other = linked_owner(engine, owner)
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        add_marker(engine, 8, other.track_id, 4, hit=2.02, now=1.8)
        events = engine.release_events(1.8)
        for mid in (7, 8):
            marker = engine.sustain_tracker.markers[mid]
            marker.observations[-1] = replace(marker.observations[-1], progress=.99)
        expected = sum(engine.sustain_tracker.markers[mid].predicted_hit(1.)
                       for mid in (7, 8)) / 2. - .125
        revised = self.refine(engine, events, 1.82)
        self.assertLess(expected, 1.82)
        self.assertTrue(all(event.tap_frozen for event in revised))
        self.assertTrue(all(abs(event.deadline - expected) < 1e-9 for event in revised))

    def test_checkpoint_completion_does_not_requeue_or_change_tap_fields(self):
        from agent.music.models import MusicActionEvent
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        event = engine.release_events(1.8)[0]
        ordinary = MusicActionEvent('ordinary-1', 1, 1, NoteGesture.TAP, 1.9, (320, 620))
        self.assertEqual(self.refine(engine, [ordinary, event], 1.81)[0], ordinary)
        self.acknowledge(engine, event)
        self.assertEqual(engine.release_events(1.91), [])
        self.assertEqual(owner.state, TrackState.TAP_PENDING)


if __name__ == '__main__':
    unittest.main()
