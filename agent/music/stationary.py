"""Behavior-preserving stationary boundary; orchestration remains in the engine."""
from __future__ import annotations

from .models import MusicFrame, TrackState
from .tap_identity import ordinary_tap


def update_stationary_evidence(engine, frame: MusicFrame) -> None:
    """Learn stationary HUD text locations from recurring frozen tracks.

    Judgement text and similar banners draw small chromatic glyphs over the
    note field.  They freeze centre-lane tracks exactly where they appear.
    Regions are diagnostic only. Retire individually stationary tracks,
    never real notes merely passing through the same screen position.
    """
    if not engine.config.stationary_zone_enabled:
        return
    import numpy as np
    # Identical screenshots are not independent stationary-glyph evidence.
    if isinstance(frame.image, np.ndarray):
        fingerprint = frame.image[::32, ::32, :3]
        previous = getattr(engine, '_stationary_fingerprint', None)
        engine._stationary_fingerprint = fingerprint.copy()
        if previous is not None and np.array_equal(previous, fingerprint):
            return
    freeze_seconds = engine.config.stationary_zone_freeze_ms / 1000.0
    for track in list(engine.tracks.values()):
        if track.state not in {TrackState.APPROACHING, TrackState.TAP_PENDING}:
            continue
        if not ordinary_tap(track):
            continue
        observations = list(track.observations)
        if len(observations) < 3:
            continue
        recent = observations[-3:]
        if frame.sequence - recent[-1].frame_sequence > 1:
            continue
        if frame.midpoint - recent[0].timestamp < freeze_seconds:
            continue
        span = max(o.progress for o in recent) - min(o.progress for o in recent)
        if span > 0.008 or track.speed > 0.06:
            continue
        center = recent[-1].center
        if engine._in_text_zone(center):
            track.state = TrackState.LOST
            continue
        cell = (int(center[0] // 32), int(center[1] // 32))
        site = engine.frozen_sites.setdefault(cell, set())
        if len(site) < engine.config.stationary_zone_min_tracks:
            site.add(track.track_id)
        if (
            len(site) >= engine.config.stationary_zone_min_tracks
            and len(engine.text_zones) < engine.config.stationary_zone_max_zones
        ):
            engine.text_zones.append((center[0], center[1], engine.config.stationary_zone_radius_px))
            engine.tap_trace.add(
                'text_zone', time=frame.midpoint, frame=frame.sequence,
                center=(round(center[0]), round(center[1])),
                tracks=sorted(site), lane=track.lane,
            )
            for site_track_id in site:
                site_track = engine.tracks.get(site_track_id)
                if site_track is not None:
                    site_track.state = TrackState.LOST
