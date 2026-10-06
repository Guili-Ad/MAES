"""Strong cross-detector physical aliases, never deadline/owner heuristics."""
from __future__ import annotations

import math

from .models import FLICK_GESTURES, NoteGesture
from .point_events import point_registry


def _same_contour(track, marker, descriptor, frame):
    if (descriptor is None or descriptor.physical_ring is not True
            or descriptor.box is None or track.lane != marker.lane()
            or track.flick or track.gesture in FLICK_GESTURES
            or track.gesture not in {NoteGesture.TAP, NoteGesture.HOLD_START}
            or not track.observations or not marker.observations):
        return False
    coverage = descriptor.ring_coverage
    if coverage is not None and (not math.isfinite(coverage) or coverage < .5):
        return False
    last = track.observations[-1]
    if last.frame_sequence != frame.sequence or marker.last_seen_frame != frame.sequence:
        return False
    head = {item.frame_sequence: item for item in track.observations}
    common = [(head[item.frame_sequence], item) for item in marker.observations
              if item.frame_sequence in head]
    if len(common) < 3:
        return False
    for left, right in common[-3:]:
        if (right.lane != track.lane or abs(left.timestamp-right.timestamp) > 1e-6
                or not math.isfinite(left.timestamp) or not math.isfinite(right.timestamp)
                or any(not math.isfinite(value) for value in (*left.center, *right.center))
                or math.dist(left.center, right.center) > 3.):
            return False
    hx, hy, hw, hh = last.candidate.box
    gx, gy, gw, gh = descriptor.box
    if min(hw, hh, gw, gh) <= 0:
        return False
    # Same centre alone is insufficient: the dedicated hourglass ring must be
    # physically contained by the other detector's current captured contour.
    return hx <= gx and hy <= gy and gx+gw <= hx+hw and gy+gh <= hy+hh


def reconcile_point_sources(engine, frame):
    """Return head IDs whose previously proved physical source is now gold.

    Call after both detectors observed this frame, before head scheduling.
    Already queued events retain their ID and are canonicalized by the common
    registry before dispatch. Unknown/ambiguous pairs are never merged.
    """
    if engine.tap_hold_chain is None:
        return set()
    registry = point_registry(engine)
    registry.gold_shadows = {tid: mid for tid, mid in registry.gold_shadows.items()
                             if tid in engine.tracks}
    possible = []
    for marker in engine.sustain_tracker.markers.values():
        if not engine.tap_hold_chain.eligible(marker, frame.midpoint):
            continue
        descriptor = engine.sustain_tracker.descriptors.get(marker.marker_id)
        for track in engine.tracks.values():
            if _same_contour(track, marker, descriptor, frame):
                possible.append((track.track_id, marker.marker_id))
    heads, rings = {}, {}
    for tid, mid in possible:
        heads.setdefault(tid, set()).add(mid)
        rings.setdefault(mid, set()).add(tid)
    for tid, mid in possible:
        if len(heads[tid]) != 1 or len(rings[mid]) != 1:
            continue
        registry.bind_gold_source(tid, mid, frame.midpoint)
        engine.tracks[tid].physical_id = registry.source_ids[('track', tid)]
    return set(registry.gold_shadows)
