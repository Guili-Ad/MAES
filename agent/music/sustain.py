"""Ribbon small-note tracking for the sustained B chain.

A hold head is judged as a tap (ABA); every gold ring drawn along the ribbon
(middle checkpoints and the terminal cap) is a small note that only requires a
held contact on its lane when it reaches the judgement point.  This module
turns motion-gated marker detections into per-marker trajectories with a lane
consensus and an arrival prediction, so the engine can schedule one
conservative B window per marker.  Static stage decorations never enter
because callers only feed detections that already passed the motion gate.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .holds import HoldTailDetection
from .models import MusicConfig, MusicFrame
from .hold_marker_identity import MarkerAssociationRequest, MarkerSnapshot, associate_marker_batch


@dataclass(frozen=True)
class SustainObservation:
    timestamp: float
    frame_sequence: int
    progress: float
    lane: int
    center: tuple[float, float]
    score: float
    pixel_count: int
    exits: int
    topology: str
    capture_started: float | None = None
    capture_finished: float | None = None


@dataclass
class SustainMarker:
    marker_id: int
    observations: deque = field(default_factory=lambda: deque(maxlen=12))
    owner: int | None = None
    last_seen_time: float = 0.0
    last_seen_frame: int = 0
    terminal_votes: int = 0
    sustain_votes: int = 0
    stable_logged: bool = False
    # Diagnostics only: bounded observations must not redefine physical birth.
    first_seen_time: float | None = None

    def observe(self, observation: SustainObservation, owner: int | None = None) -> None:
        if self.first_seen_time is None:
            self.first_seen_time = observation.timestamp
        self.observations.append(observation)
        self.last_seen_time = observation.timestamp
        self.last_seen_frame = observation.frame_sequence
        if owner is not None:
            self.owner = owner
        if observation.exits == 1:
            self.terminal_votes += 1
        elif observation.exits >= 2:
            self.sustain_votes += 1

    def lane(self) -> int | None:
        if not self.observations:
            return None
        observations = list(self.observations)
        highest = max(observation.progress for observation in observations)
        tail = [observation for observation in observations if observation.progress >= highest - 0.12]
        votes: dict[int, int] = {}
        for observation in tail:
            votes[observation.lane] = votes.get(observation.lane, 0) + 1
        return max(votes, key=lambda lane: votes[lane])

    def speed(self) -> float:
        return _recency_slope(self.observations)

    def predicted_hit(self, trigger_progress: float = 1.0) -> float | None:
        if not self.observations:
            return None
        slope = self.speed()
        if slope <= 0.0:
            return None
        last = self.observations[-1]
        return last.timestamp + max(0.0, trigger_progress - last.progress) / slope

    def stable(self, config: MusicConfig, now: float, trigger_progress: float = 1.0) -> bool:
        observations = list(self.observations)
        if len(observations) < config.hold_sustain_marker_min_samples:
            return False
        if observations[-1].progress <= observations[0].progress + 0.005:
            return False
        if self.speed() <= config.hold_sustain_marker_min_speed:
            return False
        hit = self.predicted_hit(trigger_progress)
        if hit is None or hit < now - 0.30:
            return False
        return hit <= now + config.hold_sustain_marker_max_horizon_ms / 1000.0

    def is_terminal(self) -> bool:
        return self.terminal_votes > self.sustain_votes

    def fresh(self, now: float, linger: float = 1.5) -> bool:
        return now - self.last_seen_time <= linger


def _recency_slope(observations) -> float:
    from .motion import weighted_slope
    return weighted_slope(observations)


class SustainMarkerTracker:
    def __init__(self, trigger_progress: float = 1.0) -> None:
        self.trigger_progress = trigger_progress
        self.markers: dict[int, SustainMarker] = {}
        self.next_marker_id = 1
        self.marker_aliases: dict[int, int] = {}

    def resolve_id(self, marker_id: int) -> int:
        """Raw channel ids continue to name the healed physical marker."""
        while marker_id in self.marker_aliases:
            marker_id = self.marker_aliases[marker_id]
        return marker_id

    def associate_frame(
        self, requests: list[MarkerAssociationRequest], frame: MusicFrame,
        *, eligible_owners: set[int] | None = None,
    ) -> dict[int, int]:
        snapshots = [
            MarkerSnapshot(state.marker_id, state.owner, state.last_seen_time,
                           state.last_seen_frame, state.observations[-1].progress,
                           state.observations[-1].lane, state.observations[-1].center,
                           len(state.observations))
            for state in self.markers.values()
            if state.observations and (state.owner is None or eligible_owners is None
                                       or state.owner in eligible_owners)
        ]
        normalized = [MarkerAssociationRequest(item.index, self.resolve_id(item.marker_id),
                                                item.detection, item.owner) for item in requests]
        matches = associate_marker_batch(normalized, snapshots, frame.midpoint, frame.sequence)
        recovered = set(matches)
        used = set(matches.values())
        for item in normalized:
            if item.index in matches:
                continue
            marker_id = item.marker_id
            # An unmatched raw id may already refer to an old/ineligible ring,
            # or a duplicated raw contour. It cannot overwrite that trajectory.
            if marker_id in used or marker_id in self.markers:
                marker_id = self.allocate_id()
                while marker_id in used or marker_id in self.markers:
                    marker_id = self.allocate_id()
            matches[item.index] = marker_id
            used.add(marker_id)
        for original in requests:
            if original.index not in recovered:
                continue
            marker_id = matches[original.index]
            prior_id = self.resolve_id(original.marker_id)
            if marker_id != prior_id and prior_id not in used:
                # Alias only a genuinely replaced id. A second physical ring
                # sharing a raw id gets its own id without redirecting the first.
                self.marker_aliases[prior_id] = marker_id
                self.markers.pop(prior_id, None)
            if original.marker_id != marker_id and original.marker_id not in used:
                self.marker_aliases[original.marker_id] = marker_id
        return matches

    def allocate_id(self) -> int:
        marker_id = self.next_marker_id
        self.next_marker_id += 1
        return marker_id

    def observe(
        self,
        marker_id: int,
        detection: HoldTailDetection,
        frame: MusicFrame,
        owner: int | None = None,
    ) -> SustainMarker:
        marker_id = self.resolve_id(marker_id)
        state = self.markers.get(marker_id)
        if state is None:
            state = SustainMarker(marker_id)
            self.markers[marker_id] = state
        # One frame contributes at most one observation per physical marker.
        if state.observations and state.last_seen_frame == frame.sequence:
            return state
        state.observe(
            SustainObservation(
                timestamp=frame.midpoint,
                frame_sequence=frame.sequence,
                progress=detection.progress,
                lane=detection.lane,
                center=detection.center,
                score=detection.score,
                pixel_count=detection.pixel_count,
                exits=detection.ribbon_exit_count,
                topology=detection.topology,
                capture_started=frame.capture_started,
                capture_finished=frame.capture_finished,
            ),
            owner=owner,
        )
        return state

    def prune(self, now: float, linger: float = 1.5) -> None:
        stale = [marker_id for marker_id, state in self.markers.items() if not state.fresh(now, linger)]
        for marker_id in stale:
            self.markers.pop(marker_id, None)
        self.marker_aliases = {raw: canonical for raw, canonical in self.marker_aliases.items()
                               if self.resolve_id(canonical) in self.markers}

    def targets(self, owner: int, config: MusicConfig, now: float) -> list[SustainMarker]:
        result = [
            state
            for state in self.markers.values()
            if state.owner == owner and state.stable(config, now, self.trigger_progress)
        ]
        result.sort(key=lambda state: state.predicted_hit(self.trigger_progress) or 0.0)
        return result

    def match(
        self,
        detection: HoldTailDetection,
        frame: MusicFrame,
        owner: int | None,
        *,
        max_distance: float = 80.0,
        max_age: float = 0.6,
    ) -> SustainMarker | None:
        """Find a recent marker that continues into this detection.

        Association in the raw streak channel breaks whenever several markers
        cross, which used to split one descending ring into many short-lived
        ids.  Reusing the established trajectory keeps the arrival prediction
        alive through those crossings.
        """
        best: tuple[float, SustainMarker] | None = None
        for state in self.markers.values():
            if owner is not None and state.owner is not None and state.owner != owner:
                continue
            if not state.observations:
                continue
            if state.last_seen_frame == frame.sequence:
                continue
            if frame.midpoint - state.last_seen_time > max_age:
                continue
            last = state.observations[-1]
            if abs(last.lane - detection.lane) > 1:
                continue
            if detection.progress < last.progress - 0.06:
                continue
            distance = math.hypot(last.center[0] - detection.center[0], last.center[1] - detection.center[1])
            if distance > max_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, state)
        return None if best is None else best[1]
