"""Cross-source aliases require three captured-frame contour witnesses."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.music.models import MusicActionEvent, MusicCandidate, MusicConfig, NoteGesture, NoteTrack, TrackObservation
from agent.music.tracking import MusicVisionEngine
from agent.music.point_sources import reconcile_point_sources
from agent.music.point_events import point_registry
from agent.music.hold_note_events import refine_hold_note_events
from test_hold_note_taps import calibration, make_frame
from test_tap_hold_chain_v4 import feed, gold, success_receipt


class PointSourceTests(unittest.TestCase):
    def setup(self, *, dx=0., observations=3):
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True))
        track = NoteTrack(9, 2, point_mode=True, visual_family='yellow_head', timing_profile='yellow_head')
        engine.tracks[9] = track
        for i in range(observations):
            stamp, progress = 1.+i*.1, .80+i*.05
            ring = gold(progress, owner_lanes=())
            feed(engine, stamp, i, [ring])
            center = (ring.center[0]+dx, ring.center[1])
            candidate = MusicCandidate((int(center[0])-16, int(center[1])-16, 32, 32), 700, .8, center)
            track.observations.append(TrackObservation(i, stamp, center, progress, candidate))
        frame = make_frame(1.+(observations-1)*.1, observations-1)
        return engine, track, next(iter(engine.sustain_tracker.markers.values())), frame

    def head_event(self, engine, track, *, deadline=1.275):
        event = MusicActionEvent('first-head-event', track.track_id, track.lane,
                                 NoteGesture.TAP, deadline, (480, 620))
        return point_registry(engine).adopt_track(event, track)

    def test_same_contour_moves_unstarted_head_to_gold_without_new_event(self):
        engine, track, marker, frame = self.setup()
        event = self.head_event(engine, track)
        self.assertEqual(reconcile_point_sources(engine, frame), {9})
        registry = point_registry(engine)
        state = registry.state_for('gold', marker.marker_id)
        self.assertEqual(state.event.event_id, event.event_id)
        self.assertEqual(state.event.physical_id, event.physical_id)
        self.assertEqual((state.event.origin, state.event.visual_family, state.event.timing_profile),
                         ('hold_note', 'gold_ring', 'gold_ring'))
        self.assertEqual(state.event.marker_id, marker.marker_id)
        revised = refine_hold_note_events(engine, [event], frame.midpoint)
        self.assertEqual(len(revised), 1)
        self.assertEqual(revised[0], state.event)
        self.assertEqual(registry.revise(event).origin, 'hold_note')
        self.assertEqual(registry.canonical_event(event).origin, 'hold_note')
        self.assertIsNone(registry.adopt_track(event, track))
        self.assertEqual(engine.release_events(frame.midpoint), [])

    def test_unqueued_head_shadows_and_gold_plans_one_point(self):
        engine, _, marker, frame = self.setup()
        self.assertEqual(reconcile_point_sources(engine, frame), {9})
        events = engine.release_events(frame.midpoint)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].marker_id, marker.marker_id)
        self.assertEqual(point_registry(engine).source_ids[('track', 9)], events[0].physical_id)

    def test_both_queued_adapters_keep_first_head_event_id_and_cancel_later_gold(self):
        engine, track, marker, frame = self.setup()
        first = self.head_event(engine, track)
        later_gold = engine.release_events(frame.midpoint)[0]
        self.assertNotEqual(first.event_id, later_gold.event_id)
        self.assertEqual(reconcile_point_sources(engine, frame), {9})
        registry = point_registry(engine)
        winner = registry.state_for('gold', marker.marker_id)
        self.assertEqual(winner.event.event_id, first.event_id)
        self.assertEqual(winner.event.origin, 'hold_note')
        self.assertTrue(registry.states[later_gold.event_id].cancelled)
        self.assertEqual([event.event_id for event in registry.refine(
            engine, [first, later_gold], frame.midpoint)], [first.event_id])

    def test_already_attempted_head_prevents_gold_resend(self):
        engine, track, marker, frame = self.setup()
        event = self.head_event(engine, track)
        registry = point_registry(engine)
        registry.acknowledge(engine, event, success_receipt(1.1))
        self.assertEqual(reconcile_point_sources(engine, frame), {9})
        self.assertEqual(engine.release_events(frame.midpoint), [])
        self.assertEqual(registry.state_for('gold', marker.marker_id).started, 1.1)

    def test_frozen_head_event_keeps_cutoff_on_source_transfer(self):
        engine, track, _, frame = self.setup()
        event = self.head_event(engine, track, deadline=1.205)
        self.assertEqual(reconcile_point_sources(engine, frame), {9})
        state = point_registry(engine).states[event.event_id]
        self.assertEqual(state.event.deadline, event.deadline)
        self.assertTrue(state.event.tap_frozen)

    def test_nearby_actual_head_and_gold_are_not_merged(self):
        engine, _, _, frame = self.setup(dx=8.)
        self.assertEqual(reconcile_point_sources(engine, frame), set())

    def test_two_samples_never_prove_same_physical_source(self):
        engine, _, _, frame = self.setup(observations=2)
        self.assertEqual(reconcile_point_sources(engine, frame), set())

    def test_timestamp_mismatch_does_not_merge_even_with_same_centres(self):
        engine, track, _, frame = self.setup()
        track.observations[0] = replace(track.observations[0], timestamp=.99)
        self.assertEqual(reconcile_point_sources(engine, frame), set())

    def test_unknown_descriptor_box_or_false_gold_glyph_does_not_merge(self):
        for changes in ({'box': None}, {'physical_ring': False}, {'ring_coverage': .2}):
            with self.subTest(changes=changes):
                engine, _, marker, frame = self.setup()
                tracker = engine.sustain_tracker
                tracker.descriptors[marker.marker_id] = replace(tracker.descriptors[marker.marker_id], **changes)
                self.assertEqual(reconcile_point_sources(engine, frame), set())

    def test_gold_box_outside_head_contour_is_not_same_physical_source(self):
        engine, _, marker, frame = self.setup()
        tracker = engine.sustain_tracker
        old = tracker.descriptors[marker.marker_id]
        tracker.descriptors[marker.marker_id] = replace(old, box=(old.box[0]-10, old.box[1], 50, 24))
        self.assertEqual(reconcile_point_sources(engine, frame), set())

    def test_ambiguous_two_heads_cannot_claim_one_gold(self):
        engine, track, _, frame = self.setup()
        other = replace(track, track_id=10, observations=type(track.observations)(track.observations, maxlen=12))
        engine.tracks[10] = other
        self.assertEqual(reconcile_point_sources(engine, frame), set())

    def test_proved_shadow_survives_short_source_occlusion(self):
        engine, _, _, frame = self.setup()
        self.assertEqual(reconcile_point_sources(engine, frame), {9})
        self.assertEqual(reconcile_point_sources(engine, make_frame(1.3, 3)), {9})


if __name__ == '__main__':
    unittest.main()
