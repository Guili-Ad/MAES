"""Qualification of queued track-backed taps, shared by every input entry.

A queue record is not evidence that a head still exists. Conversely, a
rejected candidate or one missing prediction is not proof of a lost head.
Only the track's own observations can revoke its not-yet-started input.
"""
from __future__ import annotations

from .models import NoteGesture, TrackState
from .tap_identity import coastable_tap, discontinuity, ordinary_tap


def _failure_reason(track, config, now, *, sequence=None, min_speed=None):
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
    if not ordinary_tap(track):
        return None
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
    if missing and not coastable_tap(track, config, now=now, min_speed=min_speed):
        return 'visual-dropout-without-coast-evidence'
    return None


def valid_pending(event, tracks, config, now, trace, *, sequence=None, min_speed=None):
    """Return whether a not-yet-sent event still owns valid visual evidence.

    ``sequence`` and ``min_speed`` should be the engine's current frame sequence
    and adaptive coast speed. Hold marker/centre-special events have independent
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
    reason = _failure_reason(track, config, now, sequence=sequence, min_speed=min_speed)
    if reason is None:
        return True
    if reason not in {'terminal-track', 'input-already-started'}:
        track.state = TrackState.LOST
    trace.add('cancelled', time=now, event=event.event_id, track=track.track_id,
              reason=reason, source='pending-eligibility',
              last_visual_time=track.observations[-1].timestamp if track.observations else None)
    return False
