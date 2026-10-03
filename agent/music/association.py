"""Behavior-preserving association boundary; orchestration remains in the engine."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
from agent.common import LOGGER
from .models import FLICK_GESTURES, MusicCandidate, MusicFrame, NoteGesture, TrackObservation, TrackState
from .tap_tracking import associate_taps, repeated_head_pixels
from .vision import VisualMask
from .holds import bonus_hold_ribbon_present, hold_head_color_ratio
if TYPE_CHECKING:
    from .tracking import LaneProjection


def associate_lane(
    engine,
    lane: int,
    entries: list[tuple[MusicCandidate, LaneProjection]],
    frame: MusicFrame,
    visual: VisualMask,
    recovered=None,
) -> None:
    for track in engine.tracks.values():
        if (
            track.lane == lane
            and track.state in {TrackState.TAP_PENDING, TrackState.FLICK_PENDING}
            and (
                (track.gesture == NoteGesture.TAP
                 and track.tap_input_started is not None
                 and track.tap_executed_hit_time is not None
                 and frame.midpoint > track.tap_executed_hit_time + 0.12)
                or (track.gesture != NoteGesture.TAP and track.action_executed
                    and track.predicted_hit_time is not None
                    and frame.midpoint > track.predicted_hit_time + 0.12)
            )
        ):
            track.state = TrackState.RELEASED
    active_holds_outside_head_window = [
        track
        for track in engine.tracks.values()
        if track.lane == lane
        and track.state in {TrackState.HOLD_PENDING, TrackState.HOLDING}
        and not engine.hold_policy.head_association_open(track, frame.midpoint)
    ]
    tracks = [
        track
        for track in engine.tracks.values()
        if track.lane == lane
        and track.state not in {TrackState.RELEASED, TrackState.LOST}
        and engine.hold_policy.head_association_open(track, frame.midpoint)
    ]
    unmatched_tracks = set(track.track_id for track in tracks)
    classification_cache = {}

    def cached_flick(candidate):
        # Directions are colour-classified at detection time.  Ordinary
        # candidates never carry flick evidence, which restores the v0.1
        # tap identity semantics exactly.
        return candidate.flick_direction if candidate.variant == "flick" else NoteGesture.UNKNOWN

    def cached_head_ratio(candidate):
        key = ("head", candidate.box)
        value = classification_cache.get(key)
        if value is None:
            value = hold_head_color_ratio(frame.image, candidate)
            classification_cache[key] = value
        return value

    def safe_tap_candidate(candidate, projection):
        return (candidate.variant != "bonus_star"
                and (not engine.config.enable_holds or cached_head_ratio(candidate) < engine.config.hold_head_color_ratio)
                and cached_flick(candidate) not in FLICK_GESTURES)

    recovery_owners = {}
    for tid, (candidate, projection) in (recovered or {}).items():
        if projection.lane != lane or tid not in unmatched_tracks:
            continue
        x, y, w, h = candidate.box
        covered = []
        for entry in entries:
            c, p = entry
            cx, cy, cw, ch = c.box
            overlap = max(0, min(x+w, cx+cw)-max(x, cx))*max(0, min(y+h, cy+ch)-max(y, cy))
            if overlap >= .8*cw*ch and math.dist(c.center, candidate.center) < min(w,h)*.4:
                covered.append(entry)
        if any(not safe_tap_candidate(*entry) for entry in covered):
            continue
        if len(covered) == 1 and math.dist(covered[0][0].center, candidate.center) < 4.:
            # Full healthy contour: neither change its centre nor claim it.
            continue
        entries = [entry for entry in entries if entry not in covered]
        entries.append((candidate, projection))
        recovery_owners[id(candidate)] = tid
    entries.sort(key=lambda item: (item[1].progress, item[0].center[0]), reverse=True)
    owned = {i: recovery_owners[id(c)] for i, (c, _) in enumerate(entries) if id(c) in recovery_owners}
    matches = associate_taps(tracks, entries, frame, engine.config, safe_tap_candidate, engine.tap_trace, owned)
    for index, (candidate, projection) in enumerate(entries):
        if index in matches:
            track = matches[index]
            unmatched_tracks.discard(track.track_id)
            if safe_tap_candidate(candidate, projection) and repeated_head_pixels(track, candidate, frame):
                # Do not feed an unchanged contour into the velocity fit or
                # manufacture a late duplicate. New/different contours still
                # follow the original one-to-one matching/classification.
                track.missed_frames = 0
                track.tap_contour_seen_time = frame.midpoint
                engine.tap_trace.add('tap_contour_repeat', time=frame.midpoint, frame=frame.sequence,
                                     track=track.track_id, box=candidate.box)
                continue
        else:
            track = engine._new_track(lane, frame.midpoint)
            if active_holds_outside_head_window:
                engine.isolated_same_lane_head_count += 1
                LOGGER.info(
                    "Music isolated incoming head from active hold lane=%s new_track=%s hold_tracks=%s progress=%.3f",
                    lane,
                    track.track_id,
                    [item.track_id for item in active_holds_outside_head_window],
                    projection.progress,
                )
        observation = TrackObservation(
            frame_sequence=frame.sequence,
            timestamp=frame.midpoint,
            center=candidate.center,
            progress=projection.progress,
            candidate=candidate,
        )
        track.observations.append(observation)
        track.tap_contour_seen_time = None
        if track.first_seen_time is None:
            track.first_seen_time = frame.midpoint
        if candidate.variant == "bonus_star":
            track.bonus_star = True
        if candidate.variant == "flick":
            track.flick = True
            track.flick_direction = candidate.flick_direction
            track.flick_color = candidate.flick_color
        track.missed_frames = 0
        track.tail_missing_frames = 0
        engine._update_motion(track)
        if index in owned:
            engine.tap_trace.add('tap_mask_recovered', time=frame.midpoint, frame=frame.sequence,
                               track=track.track_id, box=candidate.box, progress=projection.progress,
                               raw_hit=track.predicted_hit_time)
        flick = NoteGesture.UNKNOWN if track.bonus_star else cached_flick(candidate)
        track.direction_evidence.append(flick)
        if engine.config.enable_holds and track.state not in {TrackState.RELEASED, TrackState.LOST}:
            if track.bonus_star:
                hold_evidence = bonus_hold_ribbon_present(frame.image, candidate, projection.tangent)
            else:
                head_ratio = cached_head_ratio(candidate)
                hold_evidence = head_ratio >= engine.config.hold_head_color_ratio
            if hold_evidence:
                track.hold_evidence_frames += 1
            elif track.bonus_star:
                # The star head is shared by bonus taps and holds: a single
                # confirmed ribbon frame must not be erased by a later
                # occluded frame, or real bonus holds stay classified as
                # taps and lose their tails.
                pass
            else:
                track.hold_evidence_frames = max(0, track.hold_evidence_frames - 1)
        if (track.state == TrackState.TAP_PENDING and track.tap_input_started is None
                and engine._hold_committed(track)):
            # A queued event is not a physical press. Late structural
            # evidence may promote this same head without a second input.
            track.gesture = NoteGesture.HOLD_START
            track.state = TrackState.HOLD_PENDING
            engine.tap_trace.add('head_promoted_to_hold', time=frame.midpoint,
                               track=track.track_id, event=track.action_event_id)
        if not track.action_executed:
            if engine._hold_committed(track):
                # Two head observations are sufficient to press.  Tail tracking
                # refines release and route, but may no longer turn a real hold
                # into a 6-ms ordinary tap merely because its ribbon is diagonal
                # or its cap is still close to the spawn point.
                track.gesture = NoteGesture.HOLD_START
            elif (len(track.direction_evidence) >= 2
                    and len(set(track.direction_evidence)) == 1
                    and flick in FLICK_GESTURES
                    and track.speed > engine.config.min_downward_progress):
                # A colour-tagged blob that never moved is a stage
                # decoration, not a flick note.
                track.gesture = flick
            else:
                track.gesture = NoteGesture.TAP

    for track_id in unmatched_tracks:
        track = engine.tracks[track_id]
        track.missed_frames += 1
        if track.state == TrackState.HOLDING:
            track.tail_missing_frames += 1
        elif track.state in {TrackState.TAP_PENDING, TrackState.HOLD_PENDING, TrackState.FLICK_PENDING}:
            continue
        elif engine._coast_eligible(track, frame):
            # The judgement text may hide the head for many frames; keep the
            # track so its bounded prediction can still schedule the tap.
            continue
        elif track.missed_frames > engine.config.track_lost_frames:
            engine._record_unscheduled_head_loss(track, frame, "visual-loss")
            track.state = TrackState.LOST
