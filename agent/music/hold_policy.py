from __future__ import annotations

from .models import MusicActionEvent, MusicConfig, MusicFrame, NoteGesture, NoteTrack, TrackState


class HoldTimingPolicy:
    """Head-association and scheduling rules owned exclusively by holds."""

    def __init__(self, config: MusicConfig) -> None:
        self.config = config

    def action_advance_ms(self, _track: NoteTrack) -> float:
        return self.config.hold_start_action_advance_ms

    def urgent_ready(
        self,
        track: NoteTrack,
        frame: MusicFrame,
        action_deadline: float,
    ) -> bool:
        observations = list(track.observations)
        if len(observations) < 3 or track.gesture != NoteGesture.HOLD_START:
            return False
        deadline_delta = action_deadline - frame.midpoint
        return (
            observations[-1].progress >= self.config.urgent_hold_start_min_progress
            and -max(
                self.config.hold_start_execution_window_ms,
                self.config.urgent_hold_start_late_horizon_ms,
            ) / 1000.0
            <= deadline_delta
            <= self.config.urgent_hold_start_deadline_horizon_ms / 1000.0
        )

    def head_association_open(self, track: NoteTrack, now: float) -> bool:
        """Allow the visible pressed head to refine briefly, never indefinitely.

        A HoldStart is dispatched before the visual head reaches the judgement
        line.  It remains useful for connector recognition during that small
        interval.  Once the head has arrived, however, a later same-lane visual
        component must become a new point/hold track instead of being consumed
        by the already-owned hold contact.
        """
        if track.state not in {TrackState.HOLD_PENDING, TrackState.HOLDING}:
            return True
        return (
            track.predicted_hit_time is not None
            and now
            <= track.predicted_hit_time + self.config.hold_head_association_grace_ms / 1000.0
        )

    def observe_route_ribbons(
        self,
        track: NoteTrack,
        *,
        source_lane_visible: bool,
        target_lane_visible: bool,
    ) -> bool:
        """Confirm the second, straight segment of a folded lane-change hold."""
        if track.hold_target_lane is None or track.hold_target_lane == track.lane:
            track.hold_fold_target_frames = 0
            return False
        if target_lane_visible and not source_lane_visible:
            track.hold_fold_target_frames += 1
        else:
            track.hold_fold_target_frames = max(0, track.hold_fold_target_frames - 1)
        if track.hold_fold_target_frames < self.config.hold_fold_target_confirm_frames:
            return False
        newly_confirmed = not track.hold_fold_route_confirmed
        track.hold_fold_route_confirmed = True
        return newly_confirmed

    def execution_window_ms(self, event: MusicActionEvent, track: NoteTrack | None) -> float:
        if event.gesture == NoteGesture.HOLD_START:
            return self.config.hold_start_execution_window_ms
        if event.gesture == NoteGesture.HOLD_CONTINUE:
            return self.config.hold_move_execution_window_ms
        if event.gesture == NoteGesture.HOLD_END:
            precise_release = (
                track is not None
                and track.predicted_hit_time is not None
                and (
                    track.hold_release_locked
                    or track.linked_release_source_id is not None
                    or (
                        track.hold_long_verified
                        and event.deadline - track.predicted_hit_time
                        >= self.config.hold_long_duration_ms / 1000.0
                    )
                )
            )
            return (
                self.config.hold_release_execution_window_ms
                if precise_release
                else self.config.hold_fallback_release_execution_window_ms
            )
        return self.config.deadline_execution_window_ms
