"""Independent scheduling and input lifecycle for ribbon gold rings.

No input is sent here; the synchronous executor alone owns input resources.
The negative diagnostic track id is retained, but never selects a tap track.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .models import MusicActionEvent, NoteGesture, TrackState


@dataclass
class HoldNoteEventState:
    event: MusicActionEvent
    started: float | None = None
    completed: float | None = None
    error: str | None = None
    cancelled: bool = False


@dataclass
class HoldNoteGroup:
    group_id: str
    members: tuple[str, str]
    deadline: float
    frozen: bool = False


def _finite_prediction(engine, marker):
    hit = marker.predicted_hit(engine.calibration.trigger_progress)
    if hit is None or not math.isfinite(hit):
        return None
    lane = marker.lane()
    if lane is None:
        return None
    if not 0 <= lane < len(engine.calibration.points):
        return None
    return hit, lane


def _same_pixels(left, right):
    """Repeated structural identity is required, not merely close timing."""
    if left.owner != right.owner or left.lane() != right.lane():
        return False
    a, b = list(left.observations)[-3:], list(right.observations)[-3:]
    if len(a) < 3 or len(b) < 3:
        return False
    return all(
        x.frame_sequence == y.frame_sequence
        and abs(x.timestamp - y.timestamp) <= 1e-6
        and x.exits == y.exits and x.topology == y.topology
        and math.hypot(x.center[0] - y.center[0], x.center[1] - y.center[1]) <= 3.
        and min(x.pixel_count, y.pixel_count) > 0
        and max(x.pixel_count, y.pixel_count) <= 2 * min(x.pixel_count, y.pixel_count)
        for x, y in zip(a, b)
    )


class HoldNoteEventRegistry:
    def __init__(self):
        self.states: dict[str, HoldNoteEventState] = {}
        self.groups: dict[str, HoldNoteGroup] = {}

    def has_pending(self, owner_id):
        return any(state.event.owner_id == owner_id and not state.cancelled
                   and state.completed is None for state in self.states.values())

    def _canonicalize(self, engine, now):
        resolver = getattr(engine.sustain_tracker, 'resolve_id', lambda marker_id: marker_id)
        by_marker = {}
        for state in self.states.values():
            canonical = resolver(state.event.marker_id)
            if canonical != state.event.marker_id:
                state.event = replace(state.event, marker_id=canonical)
                engine.tap_trace.add('hold_note_identity', time=now,
                    event=state.event.event_id, marker=canonical, reason='tracker-alias')
            owner = engine.tracks.get(state.event.owner_id)
            if owner is not None:
                owner.hold_sustain_planned_ids.add(canonical)
            if state.cancelled:
                continue
            key = canonical if engine.tap_hold_chain is not None else (state.event.owner_id, canonical)
            prior = by_marker.get(key)
            if prior is None:
                by_marker[key] = state
                continue
            # Canonical identity is strong evidence. Never cancel an already
            # sent input; only revoke an unstarted duplicate queue entry.
            loser = state if prior.started is not None or state.started is None else prior
            if loser.started is None:
                loser.cancelled = True
                engine.tap_trace.add('hold_note_cancelled', time=now,
                    event=loser.event.event_id, reason='canonical-marker-duplicate')
            if loser is prior:
                by_marker[key] = state

    def plan(self, engine, now):
        self._canonicalize(engine, now)
        events = []
        horizon = engine.config.hold_note_tap_horizon_ms / 1000.
        advance = engine.config.tap_action_advance_ms / 1000.
        chain = engine.tap_hold_chain
        prior_by_marker = {s.event.marker_id: s for s in self.states.values()}
        for owner in engine.tracks.values():
            if owner.state != TrackState.HOLDING:
                continue
            for marker in engine.sustain_tracker.targets(owner.track_id, engine.config, now):
                if chain is not None and not chain.eligible(marker, now):
                    continue
                prior = prior_by_marker.get(marker.marker_id)
                if prior is not None:
                    # Revoked, unstarted physical events can legally reacquire.
                    if prior.cancelled and prior.started is None and chain is not None:
                        prior.cancelled = False
                        prior.event = replace(prior.event, owner_id=owner.track_id,
                                              tap_frozen=False, tap_group_id=None)
                        events.append(prior.event)
                    continue
                if chain is None and marker.marker_id in owner.hold_sustain_planned_ids:
                    continue
                prediction = _finite_prediction(engine, marker)
                if prediction is None:
                    continue
                hit, lane = prediction
                if hit < now - .05 or hit - now > horizon:
                    continue
                duplicate = next((state for state in self.states.values()
                    if not state.cancelled and state.event.owner_id == owner.track_id
                    and state.event.marker_id in engine.sustain_tracker.markers
                    and _same_pixels(engine.sustain_tracker.markers[state.event.marker_id], marker)), None)
                if duplicate is not None:
                    owner.hold_sustain_planned_ids.add(marker.marker_id)
                    engine.tap_trace.add('hold_note_duplicate', time=now,
                        owner=owner.track_id, marker=marker.marker_id,
                        canonical_marker=duplicate.event.marker_id, reason='repeated-same-contour')
                    continue
                latest = marker.observations[-1]
                event = MusicActionEvent(
                    event_id=(f'holdnote-{marker.marker_id}' if chain is not None
                              else f'holdnote-{owner.track_id}-{marker.marker_id}'),
                    track_id=-marker.marker_id, lane=lane, gesture=NoteGesture.TAP,
                    deadline=max(now, hit - advance), coordinate=engine._lane_point(lane),
                    tap_reference_hit_time=hit,
                    source_capture_started=latest.capture_started,
                    source_capture_finished=latest.capture_finished,
                    origin='hold_note', owner_id=owner.track_id, marker_id=marker.marker_id,
                    marker_terminal=(chain.terminal(marker) if chain is not None
                                     else marker.is_terminal() and owner.hold_terminal_confirmed),
                )
                self.states[event.event_id] = HoldNoteEventState(event)
                prior_by_marker[marker.marker_id] = self.states[event.event_id]
                owner.hold_sustain_planned_ids.add(marker.marker_id)
                events.append(event)
                engine.tap_trace.add('hold_note_tap', time=now, event=event.event_id,
                    track=owner.track_id, marker=marker.marker_id, lane=lane,
                    hit=hit, deadline=event.deadline, terminal=event.marker_terminal)
        candidates = [state.event for state in self.states.values()
                      if not state.cancelled and state.started is None]
        revised = {event.event_id: event for event in self.refine(engine, candidates, now)}
        return [revised[event.event_id] for event in events if event.event_id in revised]

    def _valid(self, engine, state, now):
        owner = engine.tracks.get(state.event.owner_id)
        marker = engine.sustain_tracker.markers.get(state.event.marker_id)
        if engine.tap_hold_chain is not None:
            return bool(not state.cancelled and state.started is None and marker is not None
                        and engine.tap_hold_chain.eligible(marker, now))
        return bool(not state.cancelled and state.started is None
                    and owner is not None and owner.state == TrackState.HOLDING
                    and marker is not None and marker.owner == owner.track_id)

    def _pair_allowed(self, engine, left, right):
        if left.lane == right.lane:
            return False
        if engine.tap_hold_chain is not None:
            return engine.tap_hold_chain.linked(left.owner_id, right.owner_id)
        if left.owner_id == right.owner_id:
            return True
        a, b = engine.tracks.get(left.owner_id), engine.tracks.get(right.owner_id)
        return bool(a is not None and b is not None
                    and a.linked_partner_id == b.track_id
                    and b.linked_partner_id == a.track_id)

    def refine(self, engine, pending, now):
        self._canonicalize(engine, now)
        live = {}
        frozen_before_update = set()
        advance = engine.config.tap_action_advance_ms / 1000.
        for event in pending:
            if event.origin != 'hold_note':
                continue
            state = self.states.get(event.event_id)
            if state is None:
                state = HoldNoteEventState(event)
                self.states[event.event_id] = state
            event = state.event
            if not self._valid(engine, state, now):
                if state.started is None and not state.cancelled:
                    state.cancelled = True
                    engine.tap_trace.add('hold_note_cancelled', time=now,
                                         event=event.event_id, reason='invalid-owner-or-marker')
                continue
            marker = engine.sustain_tracker.markers[event.marker_id]
            if engine.tap_hold_chain is not None and marker.owner != state.event.owner_id:
                state.event = replace(state.event, owner_id=marker.owner)
                event = state.event
            prediction = _finite_prediction(engine, marker)
            frozen = state.event.tap_frozen or event.tap_frozen or state.event.deadline <= now + .02
            if frozen:
                frozen_before_update.add(event.event_id)
            updated = replace(state.event, tap_frozen=frozen)
            owner = engine.tracks[event.owner_id]
            updated = replace(updated, marker_terminal=(engine.tap_hold_chain.terminal(marker)
                if engine.tap_hold_chain is not None else marker.is_terminal() and owner.hold_terminal_confirmed))
            if not frozen and prediction is not None:
                hit, lane = prediction
                latest = marker.observations[-1]
                updated = replace(updated, deadline=hit - advance, lane=lane,
                    coordinate=engine._lane_point(lane), tap_reference_hit_time=hit,
                    source_capture_started=latest.capture_started,
                    source_capture_finished=latest.capture_finished)
                updated = replace(updated, tap_frozen=updated.deadline <= now + .02)
            live[event.event_id] = updated

        grouped = {}
        live_groups = set()
        for gid, group in self.groups.items():
            if all(member in live for member in group.members):
                left, right = [live[member] for member in group.members]
                if self._pair_allowed(engine, left, right):
                    live_groups.add(gid)
                    if group.deadline <= now + .02:
                        group.frozen = True
                    if not group.frozen:
                        hits = [member.tap_reference_hit_time for member in (left, right)]
                        if None in hits or abs(hits[0] - hits[1]) > engine.config.hold_note_chord_window_ms / 1000.:
                            live_groups.discard(gid)
                            continue
                        group.deadline = sum(hits) / 2. - advance
                        group.frozen = group.deadline <= now + .02
                    for member in (left, right):
                        grouped[member.event_id] = replace(member, deadline=group.deadline,
                            tap_group_id=gid, tap_frozen=group.frozen)
        for gid in set(self.groups) - live_groups:
            engine.tap_trace.add('hold_note_group_dissolved', time=now, group=gid)
            del self.groups[gid]

        remaining = sorted((event for eid, event in live.items() if eid not in grouped),
                           key=lambda event: (event.tap_reference_hit_time or event.deadline, event.event_id))
        window = engine.config.hold_note_chord_window_ms / 1000.
        while remaining:
            left = remaining.pop(0)
            eligible = [right for right in remaining if self._pair_allowed(engine, left, right)
                and left.tap_reference_hit_time is not None and right.tap_reference_hit_time is not None
                and abs(left.tap_reference_hit_time - right.tap_reference_hit_time) <= window]
            if not eligible:
                continue
            right = min(eligible, key=lambda event: (abs(event.tap_reference_hit_time - left.tap_reference_hit_time), event.event_id))
            remaining.remove(right)
            members = tuple(sorted((left.event_id, right.event_id)))
            gid = 'holdnote-pair-' + '|'.join(members)
            frozen_sides = [event for event in (left, right)
                            if event.event_id in frozen_before_update]
            if len(frozen_sides) == 2 and abs(left.deadline - right.deadline) > 1e-9:
                continue
            deadline = (frozen_sides[0].deadline if frozen_sides else
                        (left.tap_reference_hit_time + right.tap_reference_hit_time) / 2. - advance)
            frozen = bool(frozen_sides) or deadline <= now + .02
            self.groups[gid] = HoldNoteGroup(gid, members, deadline, frozen)
            for member in (left, right):
                grouped[member.event_id] = replace(member, deadline=deadline,
                    tap_group_id=gid, tap_frozen=frozen)
            engine.tap_trace.add('hold_note_group', time=now, group=gid,
                                 members=members, deadline=deadline, frozen=frozen)

        result = []
        for event in pending:
            if event.origin != 'hold_note':
                result.append(event)
                continue
            if event.event_id not in live:
                continue
            updated = grouped.get(event.event_id, replace(live[event.event_id], tap_group_id=None))
            old = self.states[event.event_id].event
            if old != updated:
                engine.tap_trace.add('hold_note_refine', time=now, event=event.event_id,
                    owner=event.owner_id, marker=event.marker_id,
                    before=old.deadline, deadline=updated.deadline,
                    raw_hit=updated.tap_reference_hit_time, lane=updated.lane,
                    group=updated.tap_group_id, frozen=updated.tap_frozen,
                    source_capture_finished=updated.source_capture_finished,
                    correction_late_ms=max(0., (now - updated.deadline) * 1000.))
            self.states[event.event_id].event = updated
            result.append(updated)
        for eid in [eid for eid, state in self.states.items()
                    if state.event.owner_id not in engine.tracks]:
            del self.states[eid]
        return result

    def acknowledge(self, engine, event, receipt):
        state = self.states.get(event.event_id)
        if state is None:
            return False
        started = getattr(receipt, 'down_call_started', None)
        if started is not None and state.started is None:
            state.started = started
        completed = getattr(receipt, 'up_call_finished', None)
        error = getattr(receipt, 'error', None)
        if completed is not None:
            state.completed = completed
        if error:
            state.error = str(error)
        engine.tap_trace.add('hold_note_input', event=event.event_id,
            owner=event.owner_id, marker=event.marker_id,
            started=state.started, completed=state.completed, error=state.error)
        if state.completed is None or error or state.started is None:
            return True
        if engine.tap_hold_chain is not None:
            engine.tap_hold_chain.completed_marker(state.event, state.completed, self)
            return True
        owner = engine.tracks.get(event.owner_id)
        resolver = getattr(engine.sustain_tracker, 'resolve_id', lambda marker_id: marker_id)
        marker = engine.sustain_tracker.markers.get(resolver(event.marker_id))
        future_marker = any(
            item.owner == event.owner_id and item.marker_id != resolver(event.marker_id)
            and item.stable(engine.config, state.completed, engine.calibration.trigger_progress)
            and (item.predicted_hit(engine.calibration.trigger_progress) or -math.inf)
                > (event.tap_reference_hit_time or event.deadline) + .02
            for item in engine.sustain_tracker.markers.values())
        if (owner is not None and owner.state == TrackState.HOLDING
                and event.marker_terminal and owner.hold_terminal_confirmed
                and marker is not None and marker.is_terminal()
                and not self.has_pending(owner.track_id) and not future_marker):
            owner.hold_sustain_final_emitted = True
            owner.state = TrackState.RELEASED
            engine._retire_hold_end_flick(owner)
            engine.tap_trace.add('hold_note_anchor_done', time=state.completed,
                track=owner.track_id, marker=event.marker_id, reason='terminal-input-completed',
                deadline=event.deadline, last_activity=marker.last_seen_time)
        return True


def hold_note_registry(engine):
    registry = getattr(engine, 'hold_note_event_registry', None)
    if registry is None:
        registry = HoldNoteEventRegistry()
        engine.hold_note_event_registry = registry
    return registry


def refine_hold_note_events(engine, pending, now):
    if not any(event.origin == 'hold_note' for event in pending):
        return pending
    return hold_note_registry(engine).refine(engine, pending, now)


def acknowledge_hold_note_event(engine, event, receipt):
    if engine is None or event.origin != 'hold_note':
        return False
    return hold_note_registry(engine).acknowledge(engine, event, receipt)
