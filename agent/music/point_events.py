"""One physical-point lifecycle for disks, stars, yellow heads and gold rings.

Visual trackers remain the owners of observations and predictions. This module
owns only identity aliases, the queued event and input-attempt tombstones. Gold
rings do not need a ribbon owner in order to become executable points.
"""
from __future__ import annotations

from dataclasses import replace

from .hold_note_events import (
    HoldNoteEventRegistry, HoldNoteEventState, _finite_prediction, _same_pixels,
)
from .models import MusicActionEvent, NoteGesture

PointEventState = HoldNoteEventState


class PointEventRegistry(HoldNoteEventRegistry):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.run_id = engine.tap_trace.run_id
        self.segment_id = engine.tap_trace.segment_id
        self.source_ids: dict[tuple[str, int], str] = {}
        self.physical_events: dict[str, str] = {}
        self.next_identity = 1
        self._qualification_seen = {}
        self.gold_shadows: dict[int, int] = {}

    def _identity(self, source, key):
        ref = (source, key)
        if ref not in self.source_ids:
            self.source_ids[ref] = f'point-{self.next_identity}'
            self.next_identity += 1
        return self.source_ids[ref]

    def state_for(self, source, key):
        physical = self.source_ids.get((source, key))
        event_id = self.physical_events.get(physical)
        return self.states.get(event_id)

    def has_pending(self, owner_id):
        """Ribbon closure is metadata about gold, never the head queue."""
        return any(state.event.origin == 'hold_note'
            and state.event.owner_id == owner_id and not state.cancelled
            and state.completed is None for state in self.states.values())

    def adopt(self, event, *, source, key, family, timing_profile):
        physical = self._identity(source, key)
        previous = self.state_for(source, key)
        if previous is not None and previous.started is not None:
            return None
        if (previous is not None and source == 'track'
                and previous.event.origin == 'hold_note'):
            return None  # proven gold authority cannot birth a head adapter again
        if previous is not None:
            event = replace(event, event_id=previous.event.event_id)
        event = replace(event, physical_id=physical, visual_family=family,
                        timing_profile=timing_profile, gesture=NoteGesture.TAP)
        if previous is None:
            self.states[event.event_id] = PointEventState(event)
            self.physical_events[physical] = event.event_id
            self.engine.tap_trace.add('point_registered', event=event.event_id,
                physical_id=physical, family=family, timing_profile=timing_profile,
                source=source, source_id=key)
        else:
            previous.event = event
            previous.cancelled = False
        return event

    def adopt_track(self, event, track):
        adopted = self.adopt(event, source='track', key=track.track_id,
            family=track.visual_family or ('bonus' if track.bonus_star else 'ordinary'),
            timing_profile=track.timing_profile or track.visual_family or 'ordinary')
        track.physical_id = self.source_ids[('track', track.track_id)]
        return adopted

    def revise(self, event):
        state = self.states.get(event.event_id)
        if state is None or state.started is not None or state.cancelled:
            return None
        if state.event.origin == 'hold_note' and event.origin != 'hold_note':
            return state.event  # a stale head representation cannot reclaim authority
        state.event = replace(event, physical_id=state.event.physical_id,
            visual_family=event.visual_family or state.event.visual_family,
            timing_profile=event.timing_profile or state.event.timing_profile,
            gesture=NoteGesture.TAP)
        return state.event

    def canonical_event(self, event):
        """Return the authoritative representation of a still-unsent event."""
        state = self.states.get(event.event_id)
        if state is None:
            return event
        if state.cancelled or state.started is not None:
            return None
        return state.event if state.event.origin != event.origin else event

    def cancel(self, event_id, reason=''):
        state = self.states.get(event_id)
        if state is None or state.started is not None:
            return False
        if not state.cancelled:
            state.cancelled = True
            self.engine.tap_trace.add('point_cancelled', event=event_id,
                physical_id=state.event.physical_id, reason=reason)
        return True

    def alias(self, source, key, canonical_source, canonical_key):
        a = self._identity(source, key)
        b = self._identity(canonical_source, canonical_key)
        if a == b:
            return a
        left = self.state_for(source, key)
        right = self.state_for(canonical_source, canonical_key)
        if left is not None and right is not None and left.started is not None and right.started is not None:
            self.engine.tap_trace.add('point_identity_sent_conflict',
                physical_ids=(a, b), events=(left.event.event_id, right.event.event_id))
            return a  # retain both actual input facts; no silent history rewrite
        # A sent identity wins. Otherwise preserve the first registered event.
        first_registered = next((state for state in self.states.values()
            if state is left or state is right), None)
        if left is not None and (right is None or left.started is not None
                or (right.started is None and first_registered is left)):
            winner, loser, kept, discarded = a, b, left, right
        else:
            winner, loser, kept, discarded = b, a, right, left
        if discarded is not None:
            self.cancel(discarded.event.event_id, 'confirmed-physical-alias')
        for ref, physical in list(self.source_ids.items()):
            if physical == loser:
                self.source_ids[ref] = winner
        if kept is not None:
            self.physical_events[winner] = kept.event.event_id
        self.physical_events.pop(loser, None)
        self.engine.tap_trace.add('point_identity_alias', physical_id=winner,
            previous=loser, event=kept.event.event_id if kept else None)
        return winner

    def _canonicalize(self, engine, now):
        resolver = engine.sustain_tracker.resolve_id
        for state in list(self.states.values()):
            if state.event.origin != 'hold_note':
                continue
            old = state.event.marker_id
            canonical = resolver(old)
            if old == canonical:
                continue
            # A tracker-confirmed identity alias carries its visual evidence,
            # not only the event's integer marker field. Do not re-warm the
            # same physical circle or require a fabricated new descriptor.
            tracker = engine.sustain_tracker
            descriptor = tracker.descriptors.get(old)
            if descriptor is not None:
                tracker.descriptors.setdefault(canonical, descriptor)
            for names in (tracker.stationary_origins, tracker.moving_origins):
                if old in names:
                    names.add(canonical)
            self.alias('gold', old, 'gold', canonical)
            if not state.cancelled:
                state.event = replace(state.event, marker_id=canonical)
                engine.tap_trace.add('hold_note_identity', time=now,
                    event=state.event.event_id, marker=canonical, reason='tracker-alias')

    def bind_gold_source(self, track_id, marker_id, now):
        """Transfer a proved shared contour to gold without a second event."""
        physical = self.alias('track', track_id, 'gold', marker_id)
        state = self.state_for('gold', marker_id)
        self.gold_shadows[track_id] = marker_id
        if state is None or state.started is not None or state.event.origin == 'hold_note':
            return physical
        marker = self.engine.sustain_tracker.markers.get(marker_id)
        prediction = _finite_prediction(self.engine, marker) if marker is not None else None
        if marker is None or prediction is None:
            return physical
        hit, lane = prediction
        latest = marker.observations[-1]
        old = state.event
        frozen = not state.cancelled and (old.tap_frozen or old.deadline <= now+.02)
        deadline = old.deadline if frozen else hit-self.engine.config.tap_action_advance_ms/1000.
        state.event = replace(old, track_id=-marker_id, origin='hold_note',
            marker_id=marker_id, owner_id=marker.owner, lane=lane,
            coordinate=self.engine._lane_point(lane), visual_family='gold_ring',
            timing_profile='gold_ring', tap_reference_hit_time=hit,
            deadline=deadline, tap_frozen=frozen, tap_group_id=None,
            source_capture_started=latest.capture_started,
            source_capture_finished=latest.capture_finished,
            marker_terminal=self.engine.tap_hold_chain.terminal(marker))
        state.cancelled = False
        self.engine.tap_trace.add('point_source_authority', time=now,
            event=old.event_id, physical_id=physical, previous_source='track',
            source='gold', track=track_id, marker=marker_id,
            before=old.deadline, deadline=deadline, frozen=frozen)
        return physical

    def _qualification(self, engine, marker, now):
        return engine.tap_hold_chain.eligibility_reason(marker, now)

    def _record_qualification(self, engine, marker, now, reason):
        key = (marker.marker_id, reason)
        last = self._qualification_seen.get(key)
        if last is None or now-last >= .25:
            self._qualification_seen[key] = now
            engine.tap_trace.add('point_qualification', time=now,
                source='gold', source_id=marker.marker_id,
                physical_id=self._identity('gold', marker.marker_id),
                family='gold_ring', timing_profile='gold_ring', reason=reason,
                owner=marker.owner, latest_visual_time=marker.last_seen_time,
                visual_frame=marker.last_seen_frame,
                raw_hit=marker.predicted_hit(engine.calibration.trigger_progress))
        # At most the live marker/reason inventory, not a full-song frame log.
        if len(self._qualification_seen) > 256:
            self._qualification_seen.pop(next(iter(self._qualification_seen)))

    def plan(self, engine, now):
        self._canonicalize(engine, now)
        events = []
        horizon = engine.config.hold_note_tap_horizon_ms / 1000.
        advance = engine.config.tap_action_advance_ms / 1000.
        for marker in engine.sustain_tracker.markers.values():
            prior = self.state_for('gold', marker.marker_id)
            if prior is not None and prior.started is not None:
                continue
            reason = self._qualification(engine, marker, now)
            if reason is not None:
                self._record_qualification(engine, marker, now, reason)
                continue
            prediction = _finite_prediction(engine, marker)
            if prediction is None:
                self._record_qualification(engine, marker, now, 'nonfinite-or-missing-prediction')
                continue
            hit, lane = prediction
            if prior is not None and not prior.cancelled:
                continue
            if prior is None and (hit < now-.05 or hit-now > horizon):
                continue
            duplicate = next((state for state in self.states.values()
                if state.event.origin == 'hold_note' and not state.cancelled
                and state.event.marker_id != marker.marker_id
                and state.event.marker_id in engine.sustain_tracker.markers
                and _same_pixels(engine.sustain_tracker.markers[state.event.marker_id],
                                 marker, ignore_owner=True)), None)
            if duplicate is not None:
                self.alias('gold', marker.marker_id, 'gold', duplicate.event.marker_id)
                engine.tap_trace.add('hold_note_duplicate', time=now,
                    marker=marker.marker_id, canonical_marker=duplicate.event.marker_id,
                    reason='repeated-same-contour')
                continue
            latest = marker.observations[-1]
            event = MusicActionEvent(
                event_id=prior.event.event_id if prior else f'holdnote-{marker.marker_id}',
                track_id=-marker.marker_id, lane=lane, gesture=NoteGesture.TAP,
                deadline=hit-advance, coordinate=engine._lane_point(lane),
                tap_reference_hit_time=hit,
                source_capture_started=latest.capture_started,
                source_capture_finished=latest.capture_finished,
                origin='hold_note', owner_id=marker.owner, marker_id=marker.marker_id,
                marker_terminal=engine.tap_hold_chain.terminal(marker))
            old_deadline = prior.event.deadline if prior else None
            event = self.adopt(event, source='gold', key=marker.marker_id,
                family='gold_ring', timing_profile='gold_ring')
            if event is None:
                continue
            events.append(event)
            if old_deadline is None:
                engine.tap_trace.add('hold_note_tap', time=now, event=event.event_id,
                    physical_id=event.physical_id, track=marker.owner,
                    marker=marker.marker_id, lane=lane, hit=hit,
                    deadline=event.deadline, terminal=event.marker_terminal)
            else:
                engine.tap_trace.add('hold_note_reacquired', time=now, event=event.event_id,
                    owner=marker.owner, marker=marker.marker_id, before=old_deadline,
                    deadline=event.deadline, hit=hit,
                    source_capture_finished=latest.capture_finished,
                    correction_late_ms=max(0., (now-event.deadline)*1000.))
        candidates = [state.event for state in self.states.values()
            if state.event.origin == 'hold_note' and not state.cancelled and state.started is None]
        revised = {event.event_id: event for event in self.refine(engine, candidates, now)}
        return [revised[event.event_id] for event in events if event.event_id in revised]

    def _valid(self, engine, state, now):
        marker = engine.sustain_tracker.markers.get(state.event.marker_id)
        return bool(not state.cancelled and state.started is None and marker is not None
                    and self._qualification(engine, marker, now) is None)

    def refine(self, engine, pending, now):
        # Source authority can change without an event ID change. Replace the
        # queued old head representation before gold qualification/refinement.
        canonical = []
        for event in pending:
            updated = self.canonical_event(event)
            if updated is not None:
                canonical.append(updated)
        result = super().refine(engine, canonical, now)
        # Old-compatible callers may replace an event without copying optional
        # point metadata. Registered lifecycle authority restores those fields.
        # This also removes aliased/cancelled heads before their final dispatch.
        refined = []
        for event in result:
            if event.event_id not in self.states:
                refined.append(event)
                continue
            revised = self.revise(event)
            if revised is not None:
                refined.append(revised)
        return refined

    def acknowledge(self, engine, event, receipt):
        receipt_run = getattr(receipt, 'run_id', '')
        receipt_segment = getattr(receipt, 'segment_id', None)
        if ((receipt_run and receipt_run != self.run_id)
                or (receipt_segment is not None and receipt_segment != self.segment_id)):
            engine.tap_trace.add('stale_receipt', event=event.event_id)
            return False
        state = self.states.get(event.event_id)
        if state is None:
            return False
        started = getattr(receipt, 'down_call_started', None)
        completed = getattr(receipt, 'up_call_finished', None)
        error = getattr(receipt, 'error', None)
        before = (state.started, state.completed, state.error)
        if started is not None and state.started is None:
            state.started = started
        if completed is not None:
            state.completed = completed
        if error:
            state.error = str(error)
        if before == (state.started, state.completed, state.error):
            return True
        engine.tap_trace.add('point_input', event=event.event_id,
            physical_id=state.event.physical_id, family=state.event.visual_family,
            timing_profile=state.event.timing_profile, started=state.started,
            completed=state.completed, error=state.error)
        if state.event.origin == 'hold_note':
            engine.tap_trace.add('hold_note_input', event=event.event_id,
                owner=state.event.owner_id, marker=state.event.marker_id,
                started=state.started, completed=state.completed, error=state.error)
            if state.started is not None and state.completed is not None and not state.error:
                engine.tap_hold_chain.completed_marker(state.event, state.completed, self)
        return True


def point_registry(engine):
    registry = getattr(engine, 'point_event_registry', None)
    trace = engine.tap_trace
    if (registry is None or registry.run_id != trace.run_id
            or registry.segment_id != trace.segment_id):
        registry = PointEventRegistry(engine)
        engine.point_event_registry = registry
    return registry
