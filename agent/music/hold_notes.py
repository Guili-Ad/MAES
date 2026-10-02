"""Behavior-preserving hold_notes boundary; orchestration remains in the engine."""
from __future__ import annotations

from .models import MusicActionEvent, NoteGesture, TrackState
from .sustain import SustainMarker


def plan_hold_note_taps(engine, now: float) -> list[MusicActionEvent]:
    events: list[MusicActionEvent] = []
    advance = engine.config.tap_action_advance_ms / 1000.0
    horizon = engine.config.hold_note_tap_horizon_ms / 1000.0
    dedupe = engine.config.hold_note_tap_dedupe_ms / 1000.0
    chord_window = engine.config.hold_note_chord_window_ms / 1000.0
    scheduled: list[tuple[int, float]] = []
    for track in engine.tracks.values():
        if track.state != TrackState.HOLDING:
            continue
        candidates: list[tuple[SustainMarker, float, int]] = []
        for marker in engine.sustain_tracker.targets(track.track_id, engine.config, now):
            if marker.marker_id in track.hold_sustain_planned_ids:
                continue
            hit = marker.predicted_hit(engine.calibration.trigger_progress)
            lane = marker.lane()
            if hit is None or lane is None:
                continue
            if hit < now - 0.05 or hit - now > horizon:
                continue
            if any(
                existing_lane == lane and abs(existing_hit - hit) <= dedupe
                for existing_lane, existing_hit in scheduled
            ):
                track.hold_sustain_planned_ids.add(marker.marker_id)
                continue
            candidates.append((marker, hit, lane))
        candidates.sort(key=lambda item: item[1])
        clusters: list[list[tuple[SustainMarker, float, int]]] = []
        for item in candidates:
            if clusters and item[1] - clusters[-1][-1][1] <= chord_window:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        for cluster in clusters:
            lanes = {lane for _marker, _hit, lane in cluster}
            group_id = (
                f"holdnote-chord-{track.track_id}-{int(round(cluster[0][1] * 1000.0))}"
                if len(cluster) > 1 and len(lanes) > 1
                else None
            )
            for marker, hit, lane in cluster:
                if any(
                    existing_lane == lane and abs(existing_hit - hit) <= dedupe
                    for existing_lane, existing_hit in scheduled
                ):
                    track.hold_sustain_planned_ids.add(marker.marker_id)
                    continue
                deadline = hit - advance
                if deadline < now:
                    deadline = now
                events.append(MusicActionEvent(
                    event_id=f"holdnote-{track.track_id}-{marker.marker_id}",
                    track_id=-marker.marker_id,
                    lane=lane,
                    gesture=NoteGesture.TAP,
                    deadline=deadline,
                    coordinate=engine._lane_point(lane),
                    tap_group_id=group_id,
                    tap_reference_hit_time=hit,
                    source_capture_started=marker.observations[-1].capture_started,
                    source_capture_finished=marker.observations[-1].capture_finished,
                ))
                track.hold_sustain_planned_ids.add(marker.marker_id)
                scheduled.append((lane, hit))
                engine.tap_trace.add(
                    'hold_note_tap', time=now, track=track.track_id, marker=marker.marker_id,
                    lane=lane, hit=round(hit, 3), deadline=round(deadline, 3), group=group_id,
                )
    for track in engine.tracks.values():
        if track.state == TrackState.HOLDING:
            engine._finalize_hold_note_anchor(track, now)
    return events
