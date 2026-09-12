from __future__ import annotations

from agent.common import LOGGER

from .models import MusicActionEvent, MusicConfig, MusicFrame, NoteGesture, NoteTrack, TrackState


class TapTimingPolicy:
    """Timing and dense-passage rules owned exclusively by ordinary taps.

    Hold heads deliberately do not enter this policy.  Keeping the two timing
    domains separate prevents a tap rescue or dense-note experiment from
    silently changing HoldStart scheduling.
    """

    def __init__(self, config: MusicConfig, lane_count: int) -> None:
        self.config = config
        self.lane_count = lane_count

    def action_advance_ms(self, track: NoteTrack) -> float:
        if (
            track.dense_tap
            and not track.bonus_star
            and track.linked_partner_id is None
        ):
            return self.config.dense_tap_action_advance_ms
        return self.config.tap_action_advance_ms

    @staticmethod
    def hit_time(track: NoteTrack) -> float | None:
        return track.predicted_hit_time

    def urgent_ready(
        self,
        track: NoteTrack,
        frame: MusicFrame,
        action_deadline: float,
    ) -> bool:
        observations = list(track.observations)
        if len(observations) < 3 or track.gesture != NoteGesture.TAP or track.bonus_star:
            return False
        minimum_progress = (
            self.config.center_urgent_tap_min_progress
            if track.lane == self.lane_count // 2
            else self.config.urgent_tap_min_progress
        )
        deadline_delta = action_deadline - frame.midpoint
        return (
            observations[-1].progress >= minimum_progress
            and -max(self.config.tap_execution_window_ms, self.config.urgent_tap_late_horizon_ms) / 1000.0
            <= deadline_delta
            <= self.config.urgent_tap_deadline_horizon_ms / 1000.0
        )

    def execution_window_ms(
        self,
        event: MusicActionEvent,
        track: NoteTrack | None,
        pending: list[MusicActionEvent],
    ) -> float:
        window = self.config.tap_execution_window_ms
        if event.event_id.startswith("center-color-") or (track is not None and track.bonus_star):
            return max(window, self.config.special_tap_execution_window_ms)
        if any(
            other.event_id != event.event_id
            and other.gesture == NoteGesture.TAP
            and other.lane == event.lane
            and abs(other.deadline - event.deadline) <= self.config.dense_tap_neighbour_ms / 1000.0
            for other in pending
        ):
            return max(window, self.config.dense_tap_execution_window_ms)
        return window

    def stabilize_dense_timing(self, tracks: dict[int, NoteTrack], frame: MusicFrame) -> None:
        """Mark tap clusters without rewriting visual arrival predictions."""
        current = [
            track
            for track in tracks.values()
            if track.state in {TrackState.APPROACHING, TrackState.TAP_PENDING}
            and track.gesture == NoteGesture.TAP
            and not track.bonus_star
            and track.hold_evidence_frames == 0
            and track.linked_partner_id is None
            and track.predicted_hit_time is not None
            and track.observations
            and track.observations[-1].frame_sequence == frame.sequence
        ]
        # The previous implementation returned here for exactly two tracks,
        # making its two-note branch unreachable in the live failure case.
        if len(current) < 2:
            return
        horizon = self.config.dense_tap_cluster_horizon_ms / 1000.0
        near_pair_horizon = self.config.dense_tap_pair_horizon_ms / 1000.0
        same_pair_horizon = self.config.dense_same_lane_pair_horizon_ms / 1000.0
        dense_ids: set[int] = set()
        # A same/adjacent-lane two-note burst is already a dense input problem:
        # both fits are susceptible to the early perspective estimate, but the
        # old minimum-of-three rule never selected them.
        for index, track in enumerate(current):
            for other in current[index + 1 :]:
                lane_gap = abs(track.lane - other.lane)
                if lane_gap > 1:
                    continue
                pair_horizon = same_pair_horizon if lane_gap == 0 else near_pair_horizon
                if abs((self.hit_time(other) or 0.0) - (self.hit_time(track) or 0.0)) <= pair_horizon:
                    dense_ids.update((track.track_id, other.track_id))
        if len(current) >= self.config.dense_tap_min_tracks:
            for track in current:
                cluster = [
                    other
                    for other in current
                    if abs((self.hit_time(other) or 0.0) - (self.hit_time(track) or 0.0)) <= horizon
                ]
                if len(cluster) >= self.config.dense_tap_min_tracks:
                    dense_ids.update(item.track_id for item in cluster)
        if not dense_ids:
            return

        newly_dense = [track for track in current if track.track_id in dense_ids and not track.dense_tap]
        for track in current:
            if track.track_id in dense_ids:
                track.dense_tap = True
        if newly_dense:
            new_hit_times = [self.hit_time(track) for track in newly_dense if self.hit_time(track) is not None]
            LOGGER.info(
                "Music dense tap cluster tracks=%s lanes=%s hit_span_ms=%.1f advance_ms=%.1f",
                [track.track_id for track in newly_dense],
                [track.lane for track in newly_dense],
                (max(new_hit_times) - min(new_hit_times)) * 1000.0 if new_hit_times else 0.0,
                self.config.dense_tap_action_advance_ms,
            )
