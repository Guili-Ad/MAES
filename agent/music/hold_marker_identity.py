"""Frame-local, one-to-one identity recovery for moving ribbon markers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .holds import HoldTailDetection


@dataclass(frozen=True)
class HoldMarkerMotion:
    distance_px: float
    delta_y_px: float
    consecutive_frames: int


@dataclass(frozen=True)
class MarkerAssociationRequest:
    index: int
    marker_id: int
    detection: HoldTailDetection
    owner: int | None


@dataclass(frozen=True)
class MarkerSnapshot:
    marker_id: int
    owner: int | None
    timestamp: float
    frame_sequence: int
    progress: float
    lane: int
    center: tuple[float, float]
    samples: int


def associate_marker_batch(
    requests: list[MarkerAssociationRequest],
    snapshots: list[MarkerSnapshot],
    now: float,
    frame_sequence: int,
    *,
    max_distance: float = 80.,
    max_age: float = .6,
) -> dict[int, int]:
    """Recover identities without mutating any trajectory while pairing.

    Each ribbon's longitudinal order is retained. Gaps are allowed on either
    side, so a briefly missing ring cannot force its successor onto its id.
    Known-owner requests claim eligible trajectories before ownerless ones.
    The eligibility bounds are the former healing distance, age and lane gate.
    """
    used: set[int] = set()
    result: dict[int, int] = {}
    owners = sorted({item.owner for item in requests if item.owner is not None})
    if any(item.owner is None for item in requests):
        owners.append(None)
    for owner in owners:
        current = sorted((item for item in requests if item.owner == owner),
                         key=lambda item: (item.detection.progress, item.detection.center[0], item.index))
        previous = sorted((item for item in snapshots if item.marker_id not in used
                           and (owner is None or item.owner is None or item.owner == owner)),
                          key=lambda item: (item.progress, item.center[0], item.marker_id))

        def cost(request: MarkerAssociationRequest, state: MarkerSnapshot) -> float | None:
            if state.frame_sequence == frame_sequence:
                return None
            if now < state.timestamp or now - state.timestamp > max_age:
                return None
            tail = request.detection
            if abs(state.lane - tail.lane) > 1 or tail.progress < state.progress - .06:
                return None
            distance = math.dist(state.center, tail.center)
            # A mature raw trajectory already passed the frame-continuity gate.
            # Preserve that established path rather than adding a healing gate
            # to fast notes that never needed identity recovery.
            if request.marker_id == state.marker_id and state.samples >= 2:
                return 0.
            if distance > max_distance:
                return None
            return distance + (0. if request.marker_id == state.marker_id else .001)

        @lru_cache(maxsize=None)
        def solve(i: int, j: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
            if i >= len(current) or j >= len(previous):
                return 0, 0., ()
            choices = [solve(i + 1, j), solve(i, j + 1)]
            residual = cost(current[i], previous[j])
            if residual is not None:
                count, total, pairs = solve(i + 1, j + 1)
                choices.append((count + 1, total + residual,
                                ((current[i].index, previous[j].marker_id),) + pairs))
            return min(choices, key=lambda item: (-item[0], item[1], item[2]))

        for index, marker_id in solve(0, 0)[2]:
            result[index] = marker_id
            used.add(marker_id)
    return result
