"""Test3 contracts: physical gold identity, not per-song actions."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_hold_note_taps import tap_engine, add_marker, make_frame, calibration, register_head, linked_owner
from test_tap_pipeline import observed_track
from agent.music.models import MusicActionEvent, MusicConfig, NoteTrack, NoteGesture, TrackState
from agent.music.tracking import MusicVisionEngine
from agent.music.hold_note_events import refine_hold_note_events, acknowledge_hold_note_event, hold_note_registry
from agent.music.holds import HoldTailDetection
from agent.music.tap_hold_chain import AnchorState
from agent.music.runtime import MusicRuntime
from agent.music.runtime import RuntimeMetrics
from agent.music.executor import MusicActionExecutor


def gold(progress, lane=2, *, topology='checkpoint', owner_lanes=None, ring=.90,
         physical=True, center=None):
    """Inject classifier evidence, not infer physical_ring from ring coverage.

    owner_lanes=() means no seven-lane score reached the connection threshold:
    this can be an occluded real ribbon, and is UNKNOWN, not proof of absence.
    """
    center = center or (float(calibration().points[lane][0]), 140.+480.*progress)
    lanes = (lane,) if owner_lanes is None else owner_lanes
    scores = tuple(.90 if i in lanes else .0 for i in range(7))
    return HoldTailDetection(progress, .90, 200, lane, center, 0.,
        1 if topology == 'terminal' else (0 if topology == 'unknown' else 2),
        topology, lanes, (int(center[0])-12, int(center[1])-12, 24, 24), ring,
        owner_scores=scores, physical_ring=physical)


def feed(engine, stamp, sequence, detections, *, ribbon=False):
    # Only the image classifier is mocked; ownership, identity, warm-up,
    # source timestamps, scheduling, freeze and receipt contracts are real.
    with patch('agent.music.tap_hold_chain.ribbon_at_judgement', return_value=ribbon):
        engine.tap_hold_chain.refresh(make_frame(stamp, sequence), detections)


def success_receipt(start, *, up=None, error=''):
    return SimpleNamespace(down_call_started=start, down_call_finished=start+.001,
        up_call_finished=start+.002 if up is None else up, error=error)


def marker_event(engine, event, start=1.9):
    return acknowledge_hold_note_event(engine, event, success_receipt(start))


class TapHoldFoundationTests(unittest.TestCase):
    def test_stale_marker_cannot_execute_even_when_time_frozen(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 3, hit=2.7, now=1.8)
        pending = engine.release_events(2.4)
        # Test3 had >1s-old input; dictionary retention is not eligibility.
        self.assertEqual(refine_hold_note_events(engine, pending, 2.6), [])

    def test_one_owner_two_lanes_is_not_a_confirmed_double_hold(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        add_marker(engine, 8, owner.track_id, 4, hit=2.01, now=1.8)
        self.assertTrue(all(e.tap_group_id is None for e in engine.release_events(1.8)))

    def test_old_release_time_does_not_close_virtual_anchor(self):
        engine, owner = tap_engine()
        owner.hold_release_time = .1
        engine.release_events(1.)
        self.assertEqual(owner.state, TrackState.HOLDING)

    def test_holding_label_without_input_receipt_cannot_own_gold(self):
        engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True,
                                   hold_sustain_enabled=False, enable_holds=True))
        ghost = NoteTrack(5, 2, gesture=NoteGesture.HOLD_START,
                          state=TrackState.HOLDING, predicted_hit_time=0.)
        engine.tracks[5] = ghost
        add_marker(engine, 7, 5, 3, hit=2., now=1.8)
        self.assertEqual(engine.release_events(1.8), [])

    def test_tap_mode_never_updates_persistent_cap_fields(self):
        engine, owner = tap_engine()
        owner.hold_release_time = .1
        with patch('agent.music.tracking.detect_hold_tails', return_value=[]), \
             patch('agent.music.tracking.hold_ribbon_present', return_value=True):
            engine._update_active_hold_tails(make_frame(2., 20))
        self.assertEqual(owner.hold_release_time, .1)

    def test_first_observation_survives_and_three_frames_schedule_two_internal_rings(self):
        engine, owner = tap_engine()
        identities = None
        for i in range(3):
            rings = [gold(.80+i*.05), gold(.75+i*.05)]
            # Candidate order is not physical identity.
            feed(engine, 1.+i*.1, i, rings if i != 1 else rings[::-1])
            if i == 0:
                identities = set(engine.sustain_tracker.markers)
                self.assertEqual(len(identities), 2)
                self.assertTrue(all(len(m.observations) == 1 for m in engine.sustain_tracker.markers.values()))
            self.assertEqual(set(engine.sustain_tracker.markers), identities)
            self.assertTrue(all(len(m.observations) == i+1 for m in engine.sustain_tracker.markers.values()))
            if i < 2:
                self.assertEqual(engine.release_events(1.+i*.1), [])
        events = engine.release_events(1.2)
        self.assertEqual(len(events), 2)
        self.assertEqual({e.owner_id for e in events}, {owner.track_id})
        self.assertTrue(all(not e.marker_terminal and e.tap_group_id is None for e in events))
        self.assertEqual({round(m.observations[0].progress, 2) for m in engine.sustain_tracker.markers.values()}, {.75, .80})

    def test_repeated_frame_is_not_a_second_warmup_sample(self):
        engine, _ = tap_engine()
        feed(engine, 1., 10, [gold(.70)])
        feed(engine, 1., 10, [gold(.70)])
        self.assertEqual(len(engine.sustain_tracker.markers), 1)
        self.assertEqual(len(next(iter(engine.sustain_tracker.markers.values())).observations), 1)

    def test_legacy_raw_id_and_streak_noise_do_not_warm_or_rename_gold(self):
        engine, _ = tap_engine()
        engine.previous_hold_tail_ids = [-999, 999]
        engine.previous_hold_tail_streaks = [999, 999]
        engine.previous_hold_tails = [gold(.95, ring=.10)]
        for i in range(3):
            with patch('agent.music.holds.detect_hold_tails', return_value=[gold(.80+i*.05)]), \
                 patch('agent.music.tap_hold_chain.ribbon_at_judgement', return_value=False):
                engine._update_active_hold_tails(make_frame(1.+i*.1, i))
        self.assertEqual(set(engine.sustain_tracker.markers), {1})
        self.assertEqual(len(engine.sustain_tracker.markers[1].observations), 3)
        self.assertEqual(len(engine.release_events(1.2)), 1)

    def test_low_ring_coverage_and_hud_jump_cannot_replace_physical_history(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        before = tuple(physical.observations)
        evidence = engine.tap_hold_chain.anchors[owner.track_id].last_evidence
        bad = replace(gold(.76, ring=.15), center=(630., 458.), box=(619,447,22,22))
        feed(engine, 1.3, 3, [bad])
        self.assertEqual(tuple(physical.observations), before)
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].last_evidence, evidence)
        self.assertEqual(len(engine.sustain_tracker.markers), 1)
        self.assertEqual(engine.release_events(1.3), [])

    def test_positive_circle_with_unknown_topology_updates_motion_not_anchor_evidence(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        anchor = engine.tap_hold_chain.anchors[owner.track_id]
        before_evidence, before_frame = anchor.last_evidence, anchor.evidence_frame
        feed(engine, 1.3, 3, [gold(.75, topology='unknown', owner_lanes=(), ring=.9)])
        self.assertEqual(len(physical.observations), 4)
        self.assertEqual(physical.observations[-1].topology, 'unknown')
        self.assertEqual(physical.owner, owner.track_id)
        self.assertEqual(anchor.last_evidence, before_evidence)
        self.assertEqual(anchor.evidence_frame, before_frame)

    def test_unknown_contour_without_positive_ring_does_not_pollute_history(self):
        engine, _ = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        before = tuple(physical.observations)
        feed(engine, 1.3, 3, [gold(.75, topology='unknown', owner_lanes=(), ring=None, physical=None)])
        self.assertEqual(tuple(physical.observations), before)

    def test_explicit_owner_lane_contradiction_does_not_update_known_ring(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        before = tuple(physical.observations)
        evidence = engine.tap_hold_chain.anchors[owner.track_id].last_evidence
        feed(engine, 1.3, 3, [gold(.75, owner_lanes=(4,))])
        self.assertEqual(tuple(physical.observations), before)
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].last_evidence, evidence)
        self.assertFalse(engine.tap_hold_chain.eligible(physical, 1.3))

    def test_checkpoint_empty_connection_is_unknown_preserves_physical_motion_not_owner_proof(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        before = tuple(physical.observations)
        evidence = engine.tap_hold_chain.anchors[owner.track_id].last_evidence
        feed(engine, 1.3, 3, [gold(.75, topology='checkpoint', owner_lanes=())])
        self.assertEqual(len(physical.observations), len(before)+1)
        self.assertEqual(physical.owner, owner.track_id)
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].last_evidence, evidence)

    def test_ring_coverage_alone_is_not_positive_physical_evidence_when_topology_unknown(self):
        engine, _ = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        before = tuple(physical.observations)
        feed(engine, 1.3, 3, [gold(.75, topology='unknown', owner_lanes=(), ring=.90, physical=None)])
        self.assertEqual(tuple(physical.observations), before)

    def test_high_ring_coverage_but_explicit_nonphysical_hud_does_not_update_gold(self):
        engine, _ = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        before = tuple(physical.observations)
        feed(engine, 1.3, 3, [gold(.75, topology='checkpoint', ring=.95, physical=False)])
        self.assertEqual(tuple(physical.observations), before)

    def test_one_two_frame_coast_is_bounded_by_measured_capture_period(self):
        for period in (1./120., 1./30., .20):
            engine, _ = tap_engine()
            for i in range(3):
                feed(engine, 1.+i*period, i, [gold(.60+i*.05)])
            chain = engine.tap_hold_chain
            marker = next(iter(engine.sustain_tracker.markers.values()))
            last = marker.last_seen_time
            self.assertAlmostEqual(chain.coast_budget, period*2.)
            for gap in (1, 2):
                chain.last_frame_sequence = marker.last_seen_frame+gap
                self.assertTrue(chain.eligible(marker, last+period*gap-1e-6))
            chain.last_frame_sequence = marker.last_seen_frame+3
            self.assertFalse(chain.eligible(marker, last+period*2.-1e-6))
            chain.last_frame_sequence = marker.last_seen_frame+2
            self.assertFalse(chain.eligible(marker, last+period*2.+.001))

    def test_freeze_does_not_keep_stale_event_and_reappearance_keeps_event_id(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        original = engine.release_events(1.8)[0]
        frozen = refine_hold_note_events(engine, [original], 1.86)[0]
        self.assertTrue(frozen.tap_frozen)
        engine.tap_hold_chain.last_frame_sequence = 6
        self.assertEqual(refine_hold_note_events(engine, [frozen], 1.95), [])
        state = hold_note_registry(engine).states[original.event_id]
        self.assertTrue(state.cancelled)
        engine.sustain_tracker.observe(7, gold(.97), make_frame(2., 7), owner=owner.track_id)
        engine.tap_hold_chain.last_frame_sequence = 7
        recovered = engine.release_events(2.)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].event_id, original.event_id)
        self.assertFalse(state.cancelled)
        self.assertIsNone(state.started)

    def test_stale_marker_reacquisition_cannot_resend_already_started_event(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        event = engine.release_events(1.8)[0]
        acknowledge_hold_note_event(engine, event, SimpleNamespace(
            down_call_started=1.9, down_call_finished=1.901, up_call_finished=None, error='up-failed'))
        engine.sustain_tracker.observe(7, gold(.98), make_frame(2., 7), owner=owner.track_id)
        self.assertEqual(engine.release_events(2.), [])

    def test_completed_physical_event_tombstone_survives_owner_pruning(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8)
        event = engine.release_events(1.8)[0]
        marker_event(engine, event)
        registry = hold_note_registry(engine)
        completed = registry.states[event.event_id]
        del engine.tracks[owner.track_id]
        registry.refine(engine, [], 2.)
        self.assertIs(registry.states[event.event_id], completed)
        # A corrected owner is metadata, not a new physical event or permission
        # to resend a circle whose Down has already been attempted.
        new = NoteTrack(6, 2, gesture=NoteGesture.HOLD_START, state=TrackState.HOLD_PENDING)
        engine.tracks[6] = new
        register_head(engine, new, now=2.)
        engine.sustain_tracker.observe(7, gold(.97), make_frame(2.01, 7), owner=6)
        self.assertEqual(engine.release_events(2.01), [])
        self.assertIs(registry.states[event.event_id], completed)

    def test_dormant_virtual_anchor_does_not_permanently_block_end_ocr(self):
        engine, owner = tap_engine()
        executor = SimpleNamespace(active_contacts={})
        self.assertTrue(MusicRuntime.chart_activity_present(engine, executor, [], 10))
        engine.tap_hold_chain.anchors[owner.track_id].state = AnchorState.QUIESCENT
        self.assertFalse(MusicRuntime.chart_activity_present(engine, executor, [], 10))
        pending = [MusicActionEvent('real-pending', 8, 3, NoteGesture.TAP, 1., (640,620))]
        self.assertTrue(MusicRuntime.chart_activity_present(engine, executor, pending, 10))
        executor.active_contacts = {0: 3}
        self.assertTrue(MusicRuntime.chart_activity_present(engine, executor, [], 10))

    def test_unregistered_compatibility_holding_label_does_not_block_end_ocr(self):
        engine, owner = tap_engine()
        engine.tap_hold_chain.anchors.clear()
        self.assertFalse(MusicRuntime.chart_activity_present(
            engine, SimpleNamespace(active_contacts={}), [], 10))

    def test_precision_wait_rechecks_visual_qualification_before_actual_down(self):
        from test_round2_runtime import Clock
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=1.975, now=1.82)
        chain = engine.tap_hold_chain
        chain.periods.extend([.010]*8)
        event = engine.release_events(1.82)[0]
        self.assertAlmostEqual(event.deadline, 1.85)
        clock = Clock()
        clock.now = 1.82
        runtime = MusicRuntime(SimpleNamespace(), engine.config, clock=clock, sleeper=clock.sleep)
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, engine.config,
            advanced=True, multi_touch=True, clock=clock, sleeper=clock.sleep)
        pending = [event]
        with patch.object(executor, '_run') as run:
            runtime._execute_due(executor, pending, clock.now, RuntimeMetrics(), engine)
        run.assert_not_called()
        self.assertFalse(pending)
        self.assertGreaterEqual(clock.now, event.deadline)
        self.assertTrue(hold_note_registry(engine).states[event.event_id].cancelled)

    def test_local_recovery_keeps_physical_event_and_source_without_fabricating_owner_proof(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.80+i*.05)])
        marker = next(iter(engine.sustain_tracker.markers.values()))
        event = engine.release_events(1.2)[0]
        anchor = engine.tap_hold_chain.anchors[owner.track_id]
        proof, votes = anchor.last_evidence, (marker.terminal_votes, marker.sustain_votes)
        recovered = gold(.95, topology='unknown', owner_lanes=())
        with patch('agent.music.gold_recovery.recover_gold_markers', return_value={marker.marker_id: recovered}):
            feed(engine, 1.3, 3, [])
            feed(engine, 1.3, 3, [])
        self.assertEqual(len(marker.observations), 4)
        self.assertEqual(set(engine.sustain_tracker.markers), {marker.marker_id})
        self.assertEqual(marker.owner, owner.track_id)
        self.assertEqual(marker.observations[-1].capture_finished, 1.3)
        self.assertEqual(anchor.last_evidence, proof)
        self.assertEqual((marker.terminal_votes, marker.sustain_votes), votes)
        pending = refine_hold_note_events(engine, [event], 1.3)
        self.assertEqual([e.event_id for e in pending], [event.event_id])

    def test_actual_down_identity_is_never_locally_recovered_or_resent(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.80+i*.05)])
        event = engine.release_events(1.2)[0]
        marker_event(engine, event, 1.25)
        marker = engine.sustain_tracker.markers[event.marker_id]
        with patch('agent.music.gold_recovery.recover_gold_markers') as recover:
            feed(engine, 1.3, 3, [])
        recover.assert_not_called()
        self.assertEqual(len(marker.observations), 3)
        self.assertEqual(engine.release_events(1.3), [])

    def test_fresh_explicit_contradiction_cannot_be_bypassed_by_local_recovery(self):
        engine, _ = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.1, i, [gold(.60+i*.05)])
        marker = next(iter(engine.sustain_tracker.markers.values()))
        with patch('agent.music.gold_recovery.recover_gold_markers') as recover:
            feed(engine, 1.3, 3, [gold(.75, owner_lanes=(4,))])
        recover.assert_not_called()
        self.assertEqual(len(marker.observations), 3)
        self.assertFalse(engine.tap_hold_chain.eligible(marker, 1.3))

    def test_positive_second_frame_reappearance_can_recover_after_coast_jitter(self):
        engine, owner = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.05, i, [gold(.80+i*.025)])
        marker = next(iter(engine.sustain_tracker.markers.values()))
        self.assertAlmostEqual(engine.tap_hold_chain.coast_budget, .1)
        event = engine.release_events(1.1)[0]
        expired = 1.200010
        self.assertFalse(engine.tap_hold_chain.eligible(marker, expired))
        self.assertEqual(refine_hold_note_events(engine, [event], expired), [])
        reappearance = gold(.90, topology='unknown', owner_lanes=())
        with patch('agent.music.gold_recovery.recover_gold_markers',
                   return_value={marker.marker_id: reappearance}) as recover:
            feed(engine, expired, 4, [])
        recover.assert_called_once()
        self.assertEqual(marker.last_seen_time, expired)
        restored = engine.release_events(expired)
        self.assertEqual([e.event_id for e in restored], [event.event_id])

    def test_second_frame_without_positive_pixels_cannot_refresh_expired_prediction(self):
        engine, _ = tap_engine()
        for i in range(3):
            feed(engine, 1.+i*.05, i, [gold(.80+i*.025)])
        marker = next(iter(engine.sustain_tracker.markers.values()))
        event = engine.release_events(1.1)[0]
        with patch('agent.music.gold_recovery.recover_gold_markers', return_value={}):
            feed(engine, 1.200010, 4, [])
        self.assertEqual(marker.last_seen_time, 1.1)
        self.assertEqual(refine_hold_note_events(engine, [event], 1.200010), [])

    def test_old_quiet_same_lane_anchor_cannot_claim_new_head_gold(self):
        engine, old = tap_engine()
        feed(engine, 1., 1, [])
        self.assertEqual(engine.tap_hold_chain.anchors[old.track_id].state, AnchorState.QUIESCENT)
        new = NoteTrack(6, 2, gesture=NoteGesture.HOLD_START, state=TrackState.HOLD_PENDING)
        engine.tracks[6] = new
        register_head(engine, new, now=1.1)
        for i in range(3):
            feed(engine, 1.2+i*.1, 2+i, [gold(.80+i*.05)])
        markers = list(engine.sustain_tracker.markers.values())
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].owner, new.track_id)
        events = engine.release_events(1.4)
        self.assertEqual({e.owner_id for e in events}, {new.track_id})

    def test_new_head_epoch_revokes_old_known_marker_claim_without_reassigning_history(self):
        engine, old = tap_engine()
        for i in range(3):
            feed(engine, .8+i*.1, i, [gold(.60+i*.05)])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        new = NoteTrack(6, 2, gesture=NoteGesture.HOLD_START, state=TrackState.HOLD_PENDING)
        engine.tracks[6] = new
        register_head(engine, new, now=1.05)
        feed(engine, 1.1, 3, [gold(.75)])
        self.assertFalse(engine.tap_hold_chain.eligible(physical, 1.1))
        self.assertEqual(physical.owner, old.track_id)

    def test_route_exit_does_not_steal_another_anchor_connection_epoch(self):
        engine, old = tap_engine()
        other = NoteTrack(6, 4, gesture=NoteGesture.HOLD_START, state=TrackState.HOLD_PENDING)
        engine.tracks[6] = other
        register_head(engine, other, now=1.)
        chain = engine.tap_hold_chain
        old_anchor, other_anchor = chain.anchors[5], chain.anchors[6]
        original_epoch = other_anchor.connection_epoch
        event = MusicActionEvent('route-checkpoint', -7, 4, NoteGesture.TAP, 1.2, (800,620),
                                 origin='hold_note', owner_id=5, marker_id=7)
        chain.completed_marker(event, 1.201, hold_note_registry(engine))
        self.assertEqual(old_anchor.exit_lane, 4)
        self.assertEqual(old_anchor.connection_lane, 2)
        self.assertEqual(chain.connection_epochs[4], original_epoch)
        self.assertTrue(chain.current_connection(other_anchor))
        chosen, _ = chain.choose_owner(gold(.75, 4), None, make_frame(1.25, 10))
        self.assertNotEqual(chosen, old.track_id)

    def feed_curve(self, engine, *, endpoint_head=False):
        for i in range(3):
            p = .60+i*.05
            feed(engine, 1.+i*.1, i, [gold(p, 2, center=(480.+25.*i, 140.+480.*p))])
        physical = next(iter(engine.sustain_tracker.markers.values()))
        if endpoint_head:
            other = NoteTrack(6, 3, gesture=NoteGesture.HOLD_START, state=TrackState.HOLD_PENDING)
            engine.tracks[6] = other
            register_head(engine, other, now=1.25)
        for i in range(3, 6):
            p = .60+i*.05
            feed(engine, 1.+i*.1, i, [gold(p, 3, owner_lanes=(3,),
                                         center=(480.+25.*i, 140.+480.*p))])
        return physical

    def test_confirmed_curved_ring_changes_exit_lane_without_changing_birth_epoch(self):
        engine, owner = tap_engine()
        original_epoch = engine.tap_hold_chain.anchors[owner.track_id].connection_epoch
        physical = self.feed_curve(engine)
        self.assertEqual(len(engine.sustain_tracker.markers), 1)
        self.assertEqual(len(physical.observations), 6)
        self.assertEqual(physical.owner, owner.track_id)
        anchor = engine.tap_hold_chain.anchors[owner.track_id]
        self.assertEqual(anchor.exit_lane, 3)
        self.assertEqual(anchor.connection_lane, 2)
        self.assertEqual(anchor.connection_epoch, original_epoch)
        self.assertEqual(engine.tap_hold_chain.connection_epochs.get(3, 0), 0)
        events = engine.release_events(1.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].owner_id, owner.track_id)
        self.assertEqual(events[0].lane, 3)

    def test_curve_into_lane_with_new_head_does_not_steal_new_head_connection_epoch(self):
        engine, owner = tap_engine()
        physical = self.feed_curve(engine, endpoint_head=True)
        chain = engine.tap_hold_chain
        self.assertEqual(physical.owner, owner.track_id)
        self.assertEqual(len(physical.observations), 6)
        self.assertEqual(chain.anchors[5].exit_lane, 3)
        self.assertEqual(chain.anchors[5].connection_lane, 2)
        self.assertEqual(chain.connection_epochs[3], 1)
        self.assertEqual(chain.anchors[6].connection_epoch, 1)
        self.assertEqual(chain.anchors[6].connection_lane, 3)
        self.assertTrue(chain.current_connection(chain.anchors[6]))
        self.assertEqual(engine.tracks[6].state, TrackState.HOLDING)

    def test_head_requires_successful_down_and_up_receipt_not_compatibility_label(self):
        for down, up, error in ((None, .1, ''), (.05, None, ''), (.05, .1, 'input-failed')):
            engine = MusicVisionEngine(calibration(), MusicConfig(hold_notes_as_taps=True,
                                       hold_sustain_enabled=False, enable_holds=True))
            track = NoteTrack(5, 2, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING)
            engine.tracks[5] = track
            event = MusicActionEvent('physical-head', 5, 2, NoteGesture.HOLD_START, .0, (480,620))
            engine.tap_hold_chain.acknowledge_head(event, SimpleNamespace(
                down_call_started=.0, down_call_finished=down, up_call_finished=up, error=error))
            self.assertEqual(engine.tap_hold_chain.anchors, {})
            self.assertEqual(engine.tap_hold_chain.connection_epochs, {})

    def test_duplicate_success_head_receipt_does_not_create_second_anchor_epoch(self):
        engine, owner = tap_engine()
        original = engine.tap_hold_chain.anchors[owner.track_id]
        register_head(engine, owner, now=1.)
        self.assertIs(engine.tap_hold_chain.anchors[owner.track_id], original)
        self.assertEqual(engine.tap_hold_chain.connection_epochs[owner.lane], 1)

    def test_ultra_long_visible_ribbon_ignores_legacy_release_and_watchdog(self):
        engine, owner = tap_engine()
        owner.hold_release_time, owner.hold_started_time = .1, 0.
        for i in range(1, 181):
            feed(engine, i*.1, i, [], ribbon=True)
        engine.release_events(18.)
        anchor = engine.tap_hold_chain.anchors[owner.track_id]
        self.assertEqual(anchor.state, AnchorState.ACTIVE)
        self.assertEqual(owner.state, TrackState.HOLDING)
        self.assertAlmostEqual(anchor.last_evidence, 18.)
        self.assertEqual(owner.hold_release_time, .1)
        self.assertFalse(owner.hold_sustain_final_emitted)

    def test_checkpoint_input_never_closes_anchor(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8, exits=2)
        event = engine.release_events(1.8)[0]
        self.assertFalse(event.marker_terminal)
        self.assertTrue(marker_event(engine, event))
        self.assertEqual(owner.state, TrackState.HOLDING)
        self.assertIsNone(engine.tap_hold_chain.anchors[owner.track_id].terminal_completed)

    def test_two_confirmed_terminal_votes_close_only_after_successful_input(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8, exits=1)
        event = engine.release_events(1.8)[0]
        self.assertTrue(event.marker_terminal)
        self.assertEqual(owner.state, TrackState.HOLDING)
        self.assertTrue(marker_event(engine, event))
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].state, AnchorState.CLOSED)
        self.assertEqual(owner.state, TrackState.RELEASED)

    def test_checkpoint_history_is_not_overwritten_by_two_terminal_flash_votes(self):
        engine, _ = tap_engine()
        for i, topology in enumerate(('checkpoint', 'terminal', 'terminal')):
            feed(engine, 1.+i*.1, i, [gold(.8+i*.05, topology=topology)])
        marker = next(iter(engine.sustain_tracker.markers.values()))
        self.assertFalse(engine.tap_hold_chain.terminal(marker))
        self.assertTrue(all(not e.marker_terminal for e in engine.release_events(1.2)))

    def test_future_valid_circle_blocks_terminal_close(self):
        engine, owner = tap_engine()
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8, exits=1)
        add_marker(engine, 8, owner.track_id, 2, hit=2.4, now=1.8, exits=2)
        event = next(e for e in engine.release_events(1.8) if e.marker_id == 7)
        marker_event(engine, event)
        self.assertEqual(owner.state, TrackState.HOLDING)
        self.assertNotEqual(engine.tap_hold_chain.anchors[owner.track_id].state, AnchorState.CLOSED)

    def test_stale_phantom_future_circle_does_not_block_true_terminal_close(self):
        engine, owner = tap_engine()
        add_marker(engine, 8, owner.track_id, 2, hit=2.6, now=1., exits=2)
        add_marker(engine, 7, owner.track_id, 2, hit=2., now=1.8, exits=1)
        event = next(e for e in engine.release_events(1.8) if e.marker_id == 7)
        marker_event(engine, event)
        self.assertEqual(owner.state, TrackState.RELEASED)

    def bind_one_flick(self):
        engine, owner = tap_engine()
        flick = observed_track(9, 2, .9)
        flick.gesture, flick.state, flick.flick = NoteGesture.FLICK_LEFT, TrackState.FLICK_PENDING, True
        engine.tracks[9] = flick
        frame = make_frame(1., 3)
        engine.tap_hold_chain.note_evidence(owner.track_id, frame)
        with patch('agent.music.holds.bonus_hold_ribbon_present', return_value=True), \
             patch('agent.music.hold_topology.marker_evidence', return_value=('terminal', (2,), .9, (1.,))):
            engine.tap_hold_chain.bind_flicks(frame)
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].end_flick_id, flick.track_id)
        event = MusicActionEvent('real-flick-9', 9, 2, NoteGesture.FLICK_LEFT, 1.1, (480,620))
        return engine, owner, event

    def test_bound_flick_success_closes_anchor_without_persistent_release(self):
        engine, owner, event = self.bind_one_flick()
        engine.tap_hold_chain.completed_flick(event, success_receipt(1.1), hold_note_registry(engine))
        self.assertEqual(engine.tap_hold_chain.anchors[owner.track_id].state, AnchorState.CLOSED)
        self.assertEqual(owner.state, TrackState.RELEASED)

    def test_bound_flick_failure_does_not_close_or_fake_completion(self):
        engine, owner, event = self.bind_one_flick()
        engine.tap_hold_chain.completed_flick(event, success_receipt(1.1, error='swipe-failed'), hold_note_registry(engine))
        self.assertEqual(owner.state, TrackState.HOLDING)
        self.assertIsNone(engine.tap_hold_chain.anchors[owner.track_id].terminal_completed)

    def test_dual_tail_identity_survives_candidate_order_and_owners_never_cross(self):
        engine, left = tap_engine()
        right = linked_owner(engine, left)
        by_lane = None
        for i in range(3):
            rings = [gold(.80+i*.05, 2, topology='terminal'), gold(.80+i*.05, 4, topology='terminal')]
            feed(engine, 1.+i*.1, i, rings if i % 2 else rings[::-1])
            current = {m.lane(): m.marker_id for m in engine.sustain_tracker.markers.values()}
            if by_lane is None:
                by_lane = current
            self.assertEqual(current, by_lane)
            self.assertEqual({m.lane():m.owner for m in engine.sustain_tracker.markers.values()}, {2:left.track_id,4:right.track_id})
        events = engine.release_events(1.2)
        self.assertEqual(len(events), 2)
        self.assertEqual(len({e.tap_group_id for e in events}), 1)
        self.assertIsNotNone(events[0].tap_group_id)
        self.assertEqual(events[0].deadline, events[1].deadline)
        for event in events:
            marker_event(engine, event, start=1.3)
        self.assertEqual(left.state, TrackState.RELEASED)
        self.assertEqual(right.state, TrackState.RELEASED)

    def test_dual_tail_partial_input_failure_closes_only_completed_owner(self):
        engine, left = tap_engine()
        right = linked_owner(engine, left)
        add_marker(engine, 7, left.track_id, 2, hit=2., now=1.8, exits=1)
        add_marker(engine, 8, right.track_id, 4, hit=2., now=1.8, exits=1)
        events = engine.release_events(1.8)
        left_event = next(e for e in events if e.owner_id == left.track_id)
        right_event = next(e for e in events if e.owner_id == right.track_id)
        marker_event(engine, left_event)
        acknowledge_hold_note_event(engine, right_event, SimpleNamespace(
            down_call_started=1.9, down_call_finished=1.901, up_call_finished=None, error='right-up-failed'))
        self.assertEqual(left.state, TrackState.RELEASED)
        self.assertEqual(right.state, TrackState.HOLDING)
        self.assertEqual(engine.tap_hold_chain.anchors[left.track_id].state, AnchorState.CLOSED)
        self.assertIsNone(engine.tap_hold_chain.anchors[right.track_id].terminal_completed)
        engine.sustain_tracker.observe(8, gold(.97, 4, topology='terminal'), make_frame(2., 7), owner=right.track_id)
        self.assertEqual(engine.release_events(2.), [])


if __name__ == '__main__':
    unittest.main()
