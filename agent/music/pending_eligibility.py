"""Qualification of queued track-backed taps, shared by every input entry.

A queue record is not evidence that a head still exists. Conversely, a
rejected candidate or one missing prediction is not proof of a lost head.
Only the track's own observations can revoke its not-yet-started input.
"""
from __future__ import annotations

from .models import NoteGesture, TrackState
from .tap_identity import (coastable_tap, discontinuity, ordinary_tap, point_clickable,
                           point_reobservation_ready, tap_structure_ready)


_RECOVERABLE_POINT_FAILURES = frozenset({
    'visual-age-exceeds-coast-budget',
    'visual-dropout-without-coast-evidence',
    'visual-dropout-without-coast-ownership',
    'awaiting-valid-reobservation',
})


def _failure_reason(track, config, now, *, sequence=None, min_speed=None, frame=None,
                    coast_eligible=None):
    if track.state in {TrackState.LOST, TrackState.RELEASED}:
        return 'terminal-track'
    if track.tap_input_started is not None:
        return 'input-already-started'
    observations = list(track.observations)
    if not observations:
        # Legacy/synthetic inputs without visual history retain compatibility.
        return None
    last = observations[-1]
    # Promotion to a hold or bonus classification remains that subsystem's
    # decision; do not consult or mutate its timing/lifecycle here.
    if not (ordinary_tap(track) or point_clickable(track)):
        return None
    if point_clickable(track):
        from .head_identity import stationary_from_birth
        if stationary_from_birth(track):
            return 'stationary-origin'
        reason = discontinuity(track, last.progress, last.timestamp)
        if reason is not None:
            return reason
    if point_clickable(track) and not point_reobservation_ready(track, frame):
        # A currently proven non-head remains a hard contradiction. An
        # unknown/absent observation only keeps the same identity dormant.
        if frame is not None and last.frame_sequence == frame.sequence:
            from .tap_physical_identity import contour_evidence
            if contour_evidence(last.candidate, frame, family=track.visual_family).verdict == 'negative':
                return 'current-pixels-prove-non-head-contour'
        return 'awaiting-valid-reobservation'
    if not tap_structure_ready(track, frame):
        return 'current-pixels-prove-non-head-contour'
    reason = discontinuity(track, last.progress, last.timestamp)
    if reason is not None:
        return reason
    missing = track.missed_frames > 2 or (
        sequence is not None and sequence - last.frame_sequence > 2)
    contour_seen = track.tap_contour_seen_time
    if (contour_seen is not None and 0 <= now - contour_seen <= .12
            and now - last.timestamp <= .12):
        # Repeated, owned pixels are a short screenshot freeze, not a dropout.
        missing = False
    if now - last.timestamp > config.coast_max_age_ms / 1000.:
        return 'visual-age-exceeds-coast-budget'
    if missing:
        if not coastable_tap(track, config, now=now, min_speed=min_speed):
            return 'visual-dropout-without-coast-evidence'
        # Birth and queue qualification must consult the same centre-lane and
        # physical owner decision. Queued is not evidence of independent head
        # ownership. Callback omitted retains synthetic/legacy compatibility.
        if coast_eligible is not None and not coast_eligible(track):
            return 'visual-dropout-without-coast-ownership'
    return None


def valid_pending(event, tracks, config, now, trace, *, sequence=None, min_speed=None, frame=None,
                  coast_eligible=None):
    """Return whether a not-yet-sent event still owns valid visual evidence.

    ``sequence`` and ``min_speed`` should be the engine's current frame sequence
    and adaptive coast speed. ``frame`` is the latest fully processed image;
    it is optional for legacy/synthetic owners. ``coast_eligible(track)`` is the
    engine's same centre/owner qualification used before initial queueing.
    Hold marker/centre-special events have independent
    owners and are deliberately not reinterpreted by this ordinary tap policy.
    This function does not read rejection log records or choose new deadlines.
    """
    if (event.gesture != NoteGesture.TAP or event.track_id < 0
            or event.event_id.startswith('center-color-')
            or getattr(event, 'origin', 'track') != 'track'):
        return True
    track = tracks.get(event.track_id)
    if track is None:
        if event.source_capture_finished is not None:
            trace.add('cancelled', time=now, event=event.event_id, track=event.track_id,
                      reason='observed-owner-pruned', source='pending-eligibility')
            return False
        return True  # unknown legacy owner is not a negative visual finding
    reason = _failure_reason(track, config, now, sequence=sequence, min_speed=min_speed, frame=frame,
                             coast_eligible=coast_eligible)
    if reason is None:
        return True
    recoverable = point_clickable(track) and reason in _RECOVERABLE_POINT_FAILURES
    if recoverable:
        # Cancelling a queue record is not deleting a physical object. Re-arm
        # only after a new positive observation; never continue coasting on the
        # old prediction. The original event and physical ids stay intact.
        track.state = TrackState.APPROACHING
        track.action_executed = False
        if track.point_requalification_sequence is None:
            track.point_requalification_sequence = (
                sequence if sequence is not None else track.observations[-1].frame_sequence)
    elif reason not in {'terminal-track', 'input-already-started'}:
        track.state = TrackState.LOST
    trace.add('cancelled', time=now, event=event.event_id, track=track.track_id,
              reason=reason, source='pending-eligibility', recoverable=recoverable,
              last_visual_time=track.observations[-1].timestamp if track.observations else None)
    return False
