"""Long-hold small-note planning is independent of ordinary tap scheduling."""
from __future__ import annotations

from .hold_note_events import hold_note_registry
from .models import MusicActionEvent, TrackState


def plan_hold_note_taps(engine, now: float) -> list[MusicActionEvent]:
    registry = hold_note_registry(engine)
    events = registry.plan(engine, now)
    for track in engine.tracks.values():
        if track.state == TrackState.HOLDING and not registry.has_pending(track.track_id):
            # Compatibility retirement must not discard a queued real marker.
            engine._finalize_hold_note_anchor(track, now)
    return events
