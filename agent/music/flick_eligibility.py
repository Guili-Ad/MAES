"""Standalone-arrow authority, independent of tap/hold deadlines and owners.

Qualifying a queue entry is not executing it. Unknown/aged evidence leaves
the same unstarted identity dormant; only a later positive arrow can re-arm.
Legacy held-contact releases never enter this branch.
"""
from __future__ import annotations
import math
from statistics import median
from .models import FLICK_GESTURES, TrackState


def _static_birth(track):
    if track.flick_static_origin:
        return True
    history = list(track.observations)
    if len(history) < 3 or (track.first_seen_time is not None
            and history[0].timestamp > track.first_seen_time+1e-6):
        return False
    ending = next((i for i in range(2, len(history))
                   if history[i].timestamp-history[0].timestamp >= .085), None)
    if ending is None:
        return False
    initial = history[:ending+1]
    extent = min(min(o.candidate.box[2:]) for o in initial)
    if (min(o.progress for o in initial) >= .35
            and max(o.progress for o in initial)-min(o.progress for o in initial) < .007
            and max(math.dist(initial[0].center, o.center) for o in initial) <= max(2., extent*.04)):
        track.flick_static_origin = True
    return track.flick_static_origin


def observe_arrow_origin(track):
    # Check while birth samples still exist, even when a zero speed means
    # there is not yet a prediction or a schedulable event.
    return _static_birth(track)


def motion_observations(track):
    """Unchanged sprite repeats retain ownership, never refresh motion age."""
    unique = []
    for o in track.observations:
        if unique and (o.candidate.box, o.center) == (unique[-1].candidate.box, unique[-1].center):
            continue
        unique.append(o)
    return unique


def observation_budget(track):
    history = motion_observations(track)[-4:]
    intervals = [(b.timestamp-a.timestamp)/max(1, b.frame_sequence-a.frame_sequence)
                 for a, b in zip(history, history[1:]) if b.timestamp > a.timestamp]
    return min(.6, 2*median(intervals)) if intervals else 0.


def failure_reason(track, now, *, sequence=None, frame=None):
    if track.state in {TrackState.LOST, TrackState.RELEASED}:
        return 'terminal-track'
    if track.flick_input_started is not None:
        return 'input-already-started'
    if _static_birth(track):
        return 'stationary-arrow-origin'
    history = motion_observations(track)
    if len(history) < 3:
        return 'arrow-motion-unconfirmed'
    recent, last = history[-4:], history[-1]
    if (track.gesture not in FLICK_GESTURES or not track.flick
            or any(o.candidate.variant != 'flick'
                   or o.candidate.flick_direction != track.gesture for o in recent)):
        return 'arrow-direction-unconfirmed'
    if (last.progress-recent[0].progress < .01
            or math.dist(last.center, recent[0].center) <= 2.
            or any(b.timestamp <= a.timestamp or b.progress < a.progress-.002
                   for a, b in zip(recent, recent[1:]))):
        return 'arrow-motion-unconfirmed'
    if not math.isfinite(now) or now-last.timestamp > observation_budget(track)+1e-9:
        return 'arrow-observation-age-exceeded'
    if sequence is not None and sequence-last.frame_sequence > 2:
        return 'arrow-observation-frame-budget-exceeded'
    if track.flick_requalification_sequence is not None:
        required = track.flick_requalification_sequence
        if frame is None or last.frame_sequence != frame.sequence or last.frame_sequence <= required:
            return 'awaiting-positive-arrow-reobservation'
        from .vision import classify_flick_family
        direction, _ = classify_flick_family(frame.image, last.candidate.box)
        if direction != track.gesture:
            return 'awaiting-positive-arrow-reobservation'
        track.flick_requalification_sequence = None
    return None


def record_rejection(track, reason, now, trace, *, stage, sequence=None, event=None):
    if stage == 'birth':
        previous = getattr(track, '_flick_birth_rejection', None)
        if previous is not None and previous[0] == reason and 0 <= now-previous[1] < .25:
            return
        track._flick_birth_rejection = (reason, now)
    last = track.observations[-1] if track.observations else None
    trace.add('flick_qualification', time=now, event=event, track=track.track_id,
        lane=track.lane, stage=stage, reason=reason,
        first_seen_time=track.first_seen_time, latest_visual_time=last.timestamp if last else None,
        latest_box=last.candidate.box if last else None,
        observation_budget=observation_budget(track), frame=sequence)


def valid_flick_pending(event, engine, now, trace, *, stage='pending', frame=None):
    if (engine is None or not engine.config.hold_notes_as_taps
            or event.gesture not in FLICK_GESTURES or event.origin != 'track'
            or event.contact_policy == 'held_flick'):
        return True
    track = engine.tracks.get(event.track_id)
    if track is None:
        # Synthetic/legacy actions with no observation source keep their API;
        # an observed owner disappearing is not authority for a blind swipe.
        if event.source_capture_finished is None:
            return True
        trace.add('flick_qualification', time=now, event=event.event_id, stage=stage,
                  reason='observed-arrow-owner-pruned', track=event.track_id)
        return False
    reason = failure_reason(track, now, sequence=engine.last_frame_sequence, frame=frame or engine.last_frame)
    if reason is None:
        return True
    recoverable = reason not in {'terminal-track', 'input-already-started', 'stationary-arrow-origin'}
    if recoverable:
        track.state, track.action_executed = TrackState.APPROACHING, False
        if track.flick_requalification_sequence is None:
            track.flick_requalification_sequence = engine.last_frame_sequence
    elif reason == 'stationary-arrow-origin':
        track.state = TrackState.LOST
    record_rejection(track, reason, now, trace, stage=stage,
                     sequence=engine.last_frame_sequence, event=event.event_id)
    return False
