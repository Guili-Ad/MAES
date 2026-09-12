from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

from agent.common import LOGGER

from .holds import (
    HoldTailDetection,
    bonus_hold_ribbon_present,
    detect_hold_tails,
    hold_head_color_ratio,
    hold_ribbon_present,
)
from .hold_policy import HoldTimingPolicy
from .models import (
    FLICK_GESTURES,
    HoldTailObservation,
    MusicActionEvent,
    MusicCalibrationData,
    MusicCandidate,
    MusicConfig,
    MusicFrame,
    NoteGesture,
    NoteTrack,
    TrackObservation,
    TrackState,
)
from .tap_policy import TapTimingPolicy
from .head_identity import unique_head_candidates
from .hold_topology import ribbon_at_judgement
from .tap_tracking import associate_taps, retire_converged_shadows
from .tap_chords import TapChordManager, valid_tap_pair, coherent_tap_predictions
from .tap_identity import retire_bonus_fragments, late_birth_ready, coastable_tap
from .tap_recovery import recover_masked_taps
from .tap_trace import TapTrace
from .vision import (
    VisualMask,
    detect_bonus_star_notes,
    detect_center_color_note,
    linked_tap_pair_present,
    tag_flick_candidates,
)


@dataclass(frozen=True)
class LaneProjection:
    lane: int
    progress: float
    distance: float
    tangent: tuple[float, float]


def project_to_polyline(point: tuple[float, float], line: list[list[float]]) -> tuple[float, float, tuple[float, float]]:
    if len(line) < 2:
        raise ValueError("Lane centerline must contain at least two points")
    segment_lengths = [math.dist(line[index], line[index + 1]) for index in range(len(line) - 1)]
    total_length = sum(segment_lengths)
    best_distance = float("inf")
    best_progress = 0.0
    best_tangent = (0.0, 1.0)
    consumed = 0.0
    px, py = point
    for index, segment_length in enumerate(segment_lengths):
        start_x, start_y = line[index]
        end_x, end_y = line[index + 1]
        dx, dy = end_x - start_x, end_y - start_y
        if segment_length <= 1e-6:
            continue
        parameter = max(0.0, min(1.0, ((px - start_x) * dx + (py - start_y) * dy) / (segment_length * segment_length)))
        projected_x = start_x + dx * parameter
        projected_y = start_y + dy * parameter
        distance = math.hypot(px - projected_x, py - projected_y)
        if distance < best_distance:
            best_distance = distance
            best_progress = (consumed + segment_length * parameter) / max(total_length, 1e-6)
            best_tangent = (dx / segment_length, dy / segment_length)
        consumed += segment_length
    return best_progress, best_distance, best_tangent


def assign_lane(candidate: MusicCandidate, calibration: MusicCalibrationData) -> LaneProjection | None:
    projections: list[LaneProjection] = []
    for lane, line in enumerate(calibration.lane_centerlines):
        progress, distance, tangent = project_to_polyline(candidate.center, line)
        if distance <= calibration.corridor_widths[lane]:
            projections.append(LaneProjection(lane, progress, distance, tangent))
    if not projections:
        return None
    projections.sort(key=lambda item: (item.distance, item.lane))
    if len(projections) > 1 and projections[1].distance <= projections[0].distance * 1.15 + 2.0:
        return None
    return projections[0]


def regression_slope(observations: Iterable[TrackObservation], *, recency: bool = True) -> float:
    """Least-squares slope of progress over time; recency-weighted by default.

    Real notes accelerate as they approach the line (perspective projection), so
    recent observations must dominate the estimate or the terminal speed is
    underestimated and taps land late.
    """
    values = list(observations)
    count = len(values)
    if count < 2:
        return 0.0
    w_sum = 0.0
    wx_sum = 0.0
    wy_sum = 0.0
    wxx_sum = 0.0
    wxy_sum = 0.0
    for index, observation in enumerate(values):
        weight = float(index + 1) if recency else 1.0
        x = observation.timestamp
        y = observation.progress
        w_sum += weight
        wx_sum += weight * x
        wy_sum += weight * y
        wxx_sum += weight * x * x
        wxy_sum += weight * x * y
    denominator = w_sum * wxx_sum - wx_sum * wx_sum
    if denominator <= 1e-9:
        return 0.0
    return (w_sum * wxy_sum - wx_sum * wy_sum) / denominator


class MusicVisionEngine:
    def __init__(self, calibration: MusicCalibrationData, config: MusicConfig, *, tap_trace=None) -> None:
        self.calibration = calibration
        self.config = config
        self.tracks: dict[int, NoteTrack] = {}
        self.next_track_id = 1
        self.previous_hold_tails: list[HoldTailDetection] = []
        self.previous_hold_tail_streaks: list[int] = []
        # A sustain checkpoint can lose its downstream ribbon inside the
        # judgement flash and look one-sided for its last frame.  Preserve the
        # marker's established topology across its moving trajectory so that
        # this occlusion cannot turn it into a terminal cap near TouchUp.
        self.previous_hold_tail_checkpoint_flags: list[bool] = []
        self.previous_hold_tail_terminal_streaks: list[int] = []
        self.center_color_track: NoteTrack | None = None
        self.center_color_serial = 0
        self.expired_track_count = 0
        self.pruned_track_count = 0
        self.urgent_rescue_count = 0
        self.isolated_same_lane_head_count = 0
        self.unscheduled_head_loss_count = 0
        self.tap_policy = TapTimingPolicy(config, calibration.lane_count)
        self.tap_trace = tap_trace if tap_trace is not None else TapTrace(config)
        self.tap_chords = TapChordManager(self.tap_policy, self.tap_trace)
        self.last_frame_sequence = 0
        self.hold_policy = HoldTimingPolicy(config)

    def _new_track(self, lane: int, first_seen_time: float | None = None) -> NoteTrack:
        track = NoteTrack(
            track_id=self.next_track_id,
            lane=lane,
            first_seen_time=first_seen_time,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def _detach_link(self, track: NoteTrack) -> None:
        partner_id = track.linked_partner_id
        track.linked_partner_id = None
        if partner_id is None:
            return
        partner = self.tracks.get(partner_id)
        if partner is not None and partner.linked_partner_id == track.track_id:
            partner.linked_partner_id = None

    def _record_unscheduled_head_loss(self, track: NoteTrack, frame: MusicFrame, reason: str) -> None:
        if track.action_executed or not track.observations:
            return
        last = track.observations[-1]
        # Static stage components at progress ~0.58--0.61 and one-frame colour
        # flashes are not actionable heads.  Logging each of them produced
        # hundreds of synchronous warning writes in a song and added avoidable
        # pressure precisely during dense passages.
        if (
            last.progress < 0.35
            or len(track.observations) < 2
            or track.speed < self.config.min_downward_progress
        ):
            return
        self.unscheduled_head_loss_count += 1
        LOGGER.warning(
            "Music unscheduled head lost track=%s lane=%s gesture=%s reason=%s samples=%s progress=%.3f speed=%.3f age_ms=%.0f",
            track.track_id,
            track.lane,
            track.gesture.value,
            reason,
            len(track.observations),
            last.progress,
            track.speed,
            (frame.midpoint - (track.first_seen_time or last.timestamp)) * 1000.0,
        )

    def _expire_approaching_tracks(self, frame: MusicFrame) -> None:
        """Retire visual heads that outlive one legitimate approach flight.

        The bounded observation deque previously hid a track's real age.  A
        static stage component could remain APPROACHING for 15--35 seconds,
        eventually acquire a positive speed during a flash, and schedule an
        ancient Tap/HoldStart after combo 80.  A real head must be actionable
        within the existing schedule horizon plus the pending-event age guard.
        Active holds are deliberately excluded; their duration is governed by
        tail continuity and may span most of a song.
        """
        maximum_age = self.config.max_schedule_horizon_ms / 1000.0 + 0.5
        for track in self.tracks.values():
            if (
                track.state == TrackState.APPROACHING
                and not track.action_executed
                and track.first_seen_time is not None
                and frame.midpoint - track.first_seen_time > maximum_age
            ):
                self._record_unscheduled_head_loss(track, frame, "maximum-age")
                track.state = TrackState.LOST
                self._detach_link(track)
                self.expired_track_count += 1

    def _prune_terminal_tracks(self, frame: MusicFrame) -> None:
        """Keep per-frame work bounded without touching active/pending input."""
        retention_frames = max(12, self.config.track_lost_frames * 3)
        removable = [
            track_id
            for track_id, track in self.tracks.items()
            if track.state in {TrackState.LOST, TrackState.RELEASED}
            and track.observations
            and frame.sequence - track.observations[-1].frame_sequence > retention_frames
        ]
        for track_id in removable:
            track = self.tracks.get(track_id)
            if track is not None:
                self._detach_link(track)
                self.tracks.pop(track_id, None)
                self.pruned_track_count += 1

    def _update_center_color_note(
        self,
        frame: MusicFrame,
        bonus_candidates: Iterable[MusicCandidate] = (),
    ) -> tuple[MusicCandidate | None, list[MusicActionEvent]]:
        """Track the one large rainbow center note outside ordinary candidates."""
        detections = detect_center_color_note(frame.image)
        bonus = list(bonus_candidates)
        if bonus:
            detections = [
                item
                for item in detections
                if not any(
                    math.dist(item.center, star.center)
                    <= max(30.0, max(item.box[2:]) * 0.55)
                    for star in bonus
                )
            ]
        track = self.center_color_track
        selected: MusicCandidate | None = None
        if track is None:
            # A real large note first separates from the common spawn point in
            # this upper band.  Requiring that entry prevents result effects or
            # the centre judgement target from creating a special track.
            entering = [item for item in detections if item.center[1] <= 280.0]
            if entering:
                selected = min(entering, key=lambda item: item.center[1])
                self.center_color_serial += 1
                track = NoteTrack(track_id=-self.center_color_serial, lane=self.calibration.lane_count // 2)
                track.gesture = NoteGesture.TAP
                self.center_color_track = track
        else:
            observations = list(track.observations)
            last_y = observations[-1].center[1] if observations else 170.0
            delta_y = last_y - observations[-2].center[1] if len(observations) >= 2 else 12.0
            expected_y = last_y + max(6.0, delta_y)
            maximum_step = max(90.0, delta_y * 2.5 + 24.0)
            choices = [
                item
                for item in detections
                if -4.0 <= item.center[1] - last_y <= maximum_step
            ]
            if choices:
                selected = min(choices, key=lambda item: abs(item.center[1] - expected_y))

        if track is None:
            return selected, []
        if selected is None:
            track.missed_frames += 1
            # An executed singleton used to live forever.  A later ordinary
            # yellow/teal centre tap could then be consumed as the old special
            # note and excluded from the normal tracker, explaining the dense
            # centre-lane Miss/Bad cluster after the mid-song rainbow note.
            if track.missed_frames > 3:
                self.center_color_track = None
            return None, []

        projection = project_to_polyline(selected.center, self.calibration.lane_centerlines[track.lane])
        track.observations.append(TrackObservation(
            frame_sequence=frame.sequence,
            timestamp=frame.midpoint,
            center=selected.center,
            progress=projection[0],
            candidate=selected,
        ))
        track.missed_frames = 0
        observations = list(track.observations)
        if len(observations) >= 2 and observations[-1].progress < observations[-2].progress - 0.02:
            return selected, []
        last = observations[-1]
        # Perspective makes raw lane progress accelerate sharply.  In the
        # autoplay samples, (1-progress)^1.8 is close to linear in time across
        # both the fast and slow songs, so its measured slope adapts the hit
        # deadline without imposing one song-specific fall duration.
        recent = observations[-6:]
        warped = [(item.timestamp, max(0.0, 1.0 - item.progress) ** 1.8) for item in recent]
        mean_time = sum(item[0] for item in warped) / len(warped)
        mean_value = sum(item[1] for item in warped) / len(warped)
        denominator = sum((item[0] - mean_time) ** 2 for item in warped)
        slope = (
            sum((item[0] - mean_time) * (item[1] - mean_value) for item in warped) / denominator
            if denominator > 1e-9
            else 0.0
        )
        if slope < -0.05:
            remaining = warped[-1][1] / -slope
        else:
            remaining = 1.05 * max(0.0, 1.0 - last.progress) ** 1.6
        track.predicted_hit_time = last.timestamp + max(0.08, min(2.0, remaining))
        if track.action_executed or len(observations) < 4 or last.progress < 0.20:
            return selected, []
        point = self.calibration.points[track.lane]
        event = MusicActionEvent(
            event_id=f"center-color-{self.center_color_serial}",
            track_id=track.track_id,
            lane=track.lane,
            gesture=NoteGesture.TAP,
            deadline=track.predicted_hit_time - self.config.center_tap_action_advance_ms / 1000.0,
            coordinate=(int(point[0]), int(point[1])),
            source_capture_started=frame.capture_started,
            source_capture_finished=frame.capture_finished,
        )
        track.action_executed = True
        track.state = TrackState.TAP_PENDING
        LOGGER.info(
            "Music center color note scheduled track=%s progress=%.3f deadline=%.3f samples=%s",
            track.track_id,
            last.progress,
            event.deadline,
            len(observations),
        )
        return selected, [event]

    def _update_motion(self, track: NoteTrack) -> None:
        observations = list(track.observations)
        distinct = []
        for item in observations:
            if (track.gesture == NoteGesture.HOLD_START and distinct and item.progress >= .35 and item.candidate.box == distinct[-1].candidate.box
                    and item.center == distinct[-1].center):
                continue  # repeated captured pixels are not a new motion measurement
            distinct.append(item)
        recent = distinct[-6:]
        track.speed = regression_slope(recent)
        if track.speed > self.config.static_speed_threshold:
            remaining = self.calibration.trigger_progress - recent[-1].progress
            track.predicted_hit_time = recent[-1].timestamp + max(0.0, remaining / track.speed)
        else:
            track.predicted_hit_time = None

    def _best_hold_tail_lane(self, observations: list[HoldTailObservation]) -> int:
        recent = observations[-8:]
        return min(
            range(self.calibration.lane_count),
            key=lambda lane: sum(
                (index + 1) * project_to_polyline(item.center, self.calibration.lane_centerlines[lane])[1] ** 2
                for index, item in enumerate(recent)
            ),
        )

    def _bind_hold_end_flicks(self, frame: MusicFrame) -> None:
        """Bind a ribbon-tip flick track to the hold it terminates.

        The sprite is tracked like any other flick (association, speed fit,
        ``predicted_hit_time``), but once bound to a hold it never emits a
        standalone swipe.  The hold's release uses the bound track's live
        prediction so the held flick fires when the tip reaches judgement.
        Binding only accepts the earliest upcoming flick on the same lane and
        only when it is not later than the hold's own (possibly over-extended)
        release estimate, which keeps later standalone flicks standalone.
        """
        for hold in self.tracks.values():
            if hold.gesture != NoteGesture.HOLD_START:
                continue
            if hold.state not in {TrackState.HOLD_PENDING, TrackState.HOLDING}:
                continue
            if hold.hold_end_flick_track is not None:
                bound = self.tracks.get(hold.hold_end_flick_track)
                if bound is not None and bound.predicted_hit_time is not None:
                    hold.hold_end_flick_arrival = bound.predicted_hit_time
                    direction = bound.flick_direction
                    if direction in FLICK_GESTURES:
                        hold.hold_end_flick_direction = direction
                continue
            candidates = [
                track
                for track in self.tracks.values()
                if track.flick
                and track.hold_end_owner is None
                and track.lane == hold.lane
                and track.state not in {TrackState.RELEASED, TrackState.LOST}
                and not track.action_event_id
                and track.predicted_hit_time is not None
                and track.observations
                # Bind before the track is schedulable (four observations,
                # progress 0.35) so no duplicate standalone swipe escapes.
                and track.observations[-1].progress >= min(self.config.hold_end_flick_min_progress, 0.30)
                and (
                    hold.predicted_hit_time is None
                    or track.predicted_hit_time >= hold.predicted_hit_time - 0.05
                )
                and (
                    hold.hold_release_time is None
                    or track.predicted_hit_time <= hold.hold_release_time + 0.15
                )
                and track.flick_direction in FLICK_GESTURES
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda track: track.predicted_hit_time)
            best.hold_end_owner = hold.track_id
            hold.hold_end_flick_track = best.track_id
            hold.hold_end_flick_direction = best.flick_direction
            hold.hold_end_flick_arrival = best.predicted_hit_time
            LOGGER.info(
                "Music hold end flick bound hold=%s lane=%s flick_track=%s direction=%s color=%s hit=%.3f",
                hold.track_id,
                hold.lane,
                best.track_id,
                best.flick_direction.value,
                best.flick_color,
                best.predicted_hit_time,
            )
            self.tap_trace.add(
                'hold_end_flick', time=frame.midpoint, frame=frame.sequence, track=hold.track_id,
                lane=hold.lane, direction=best.flick_direction.value, color=best.flick_color, source="bound",
                flick_track=best.track_id, hit=round(best.predicted_hit_time, 3),
            )

    def _register_hold_target(self, track: NoteTrack, lane: int) -> None:
        """Record a route target; later targets need a confirmed fold.

        The first consensus fixes the destination ray.  A different, later
        consensus opens the next segment only when the current ribbon has
        visibly folded onto its target (``hold_fold_route_confirmed``), which
        keeps ordinary curves single-hop and supports A~B~A chains up to
        ``hold_max_route_hops``.
        """
        if track.hold_target_lane is None:
            track.hold_target_lane = lane
            track.hold_route_lanes = [track.lane, lane]
            track.hold_segment_index = 0
            return
        if lane == track.hold_target_lane:
            return
        if not track.hold_fold_route_confirmed:
            return
        if len(track.hold_route_lanes) - 1 >= self.config.hold_max_route_hops:
            return
        track.hold_route_lanes.append(lane)
        track.hold_target_lane = lane
        track.hold_segment_index += 1
        track.hold_route_steps_completed = 0
        track.hold_fold_target_frames = 0
        track.hold_fold_route_confirmed = False
        track.hold_fold_move_scheduled = False
        track.hold_move_scheduled = False
        LOGGER.info(
            "Music hold route hop track=%s lane=%s->%s segment=%s route=%s",
            track.track_id,
            track.hold_route_lanes[-2],
            lane,
            track.hold_segment_index,
            track.hold_route_lanes,
        )

    @staticmethod
    def _hold_segment_source_lane(track: NoteTrack) -> int:
        if track.hold_route_lanes and 0 <= track.hold_segment_index < len(track.hold_route_lanes) - 1:
            return track.hold_route_lanes[track.hold_segment_index]
        return track.lane

    def _hold_release_deadline(self, track: NoteTrack, release_time: float | None) -> float | None:
        """Clamp the release estimate with the ribbon-tip arrival when known."""
        deadline = release_time
        if track.hold_end_flick_arrival is not None:
            if deadline is None or track.hold_end_flick_arrival < deadline:
                deadline = track.hold_end_flick_arrival
        return deadline

    def _hold_release_direction(self, track: NoteTrack, release_time: float, *, log: bool = True) -> NoteGesture:
        """Direction for a hold release: the bound colour-classified tip flick."""
        if track.hold_end_flick_direction in FLICK_GESTURES:
            return track.hold_end_flick_direction
        directions = list(track.direction_evidence)
        return directions[-1] if len(directions) >= 2 and len(set(directions[-2:])) == 1 else NoteGesture.UNKNOWN

    def _revoke_terminal(self, track: NoteTrack, frame: MusicFrame) -> None:
        self.tap_trace.add('hold_terminal_revoked', time=frame.midpoint, track=track.track_id,
                           old_deadline=track.hold_release_time, locked=track.hold_release_locked)
        track.hold_terminal_confirmed = track.hold_release_locked = False
        track.hold_tail_observations.clear()
        track.hold_long_verified = track.hold_tail_reacquired = False
        track.linked_release_source_id = None
        track.hold_terminal_conflicts = 0
        track.hold_release_time = max(track.hold_release_time or 0.,
                                     frame.midpoint + self.config.hold_checkpoint_extension_ms / 1000.)

    def _record_active_hold_tail(self, track: NoteTrack, frame: MusicFrame, tail: HoldTailDetection) -> None:
        checkpoint = tail.topology == 'checkpoint' or (not tail.topology and tail.ribbon_exit_count >= 2)
        # Near the judgement flash a second exit alone is ambiguous. Require
        # upstream ribbon evidence (or two mature pre-line legacy samples).
        if checkpoint and (tail.topology or tail.progress < .85):
            track.hold_terminal_conflicts += 1
            if track.hold_terminal_conflicts >= 2:
                self._revoke_terminal(track, frame)
            return
        track.hold_terminal_conflicts = 0
        # A two-sided marker is embedded in the hold ribbon.  It may be used by
        # an already-established terminal trajectory when the judgement arc
        # temporarily adds a second visual exit, but it can never originate a
        # release trajectory of its own.
        terminal = tail.topology == 'terminal' or (not tail.topology and tail.ribbon_exit_count == 1)
        if not track.hold_terminal_confirmed and not terminal:
            return
        if terminal:
            track.hold_terminal_confirmed = True
        previous = track.hold_tail_observations[-1] if track.hold_tail_observations else None
        if previous is not None:
            if frame.sequence == previous.frame_sequence:
                return
            delta_y = tail.center[1] - previous.center[1]
            if delta_y < -10.0 or math.dist(tail.center, previous.center) > 150.0:
                return
        first_sample = not track.hold_tail_observations
        track.hold_tail_observations.append(HoldTailObservation(
            frame_sequence=frame.sequence,
            timestamp=frame.midpoint,
            progress=tail.progress,
            score=tail.score,
            pixel_count=tail.pixel_count,
            lane=tail.lane,
            center=tail.center,
            ribbon_exit_count=tail.ribbon_exit_count,
        ))
        observations = list(track.hold_tail_observations)
        lane = self._best_hold_tail_lane(observations)
        projected = [
            replace(
                item,
                progress=project_to_polyline(item.center, self.calibration.lane_centerlines[lane])[0],
                lane=lane,
            )
            for item in observations
        ]
        # Ignore a short backwards wobble from judgement effects while keeping
        # the accelerating cap samples that determine the release deadline.
        monotonic: list[HoldTailObservation] = []
        for item in projected:
            if not monotonic or item.progress >= monotonic[-1].progress - 0.04:
                monotonic.append(item)
        recent = monotonic[-6:]
        if first_sample:
            LOGGER.info(
                "Music hold tail acquired track=%s head_lane=%s center=(%.0f,%.0f)",
                track.track_id,
                track.lane,
                tail.center[0],
                tail.center[1],
            )
        if (
            track.hold_terminal_confirmed
            and track.hold_tail_reacquired
            and not track.hold_release_locked
            and tail.progress >= self.config.hold_cap_lock_progress
        ):
            track.hold_release_time = frame.midpoint + self.config.hold_release_delay_ms / 1000.0
            track.hold_release_locked = True
            track.linked_release_source_id = None
            LOGGER.info(
                "Music very-long hold tail re-acquired at release track=%s lane=%s progress=%.3f deadline=%.3f",
                track.track_id,
                tail.lane,
                tail.progress,
                track.hold_release_time,
            )
            return
        if len(recent) < 3:
            return
        last = recent[-1]
        target_samples = self.config.hold_target_confirm_samples
        nearest_lanes = [
            min(
                range(self.calibration.lane_count),
                key=lambda item: project_to_polyline(obs.center, self.calibration.lane_centerlines[item])[1],
            )
            for obs in recent[-target_samples:]
        ]
        if (
            len(nearest_lanes) >= target_samples
            and last.progress >= self.config.hold_target_min_progress
            and len(set(nearest_lanes)) == 1
        ):
            # A terminal cap moves on one destination ray.  Once three mature
            # samples agree, crossing an intermediate visual corridor must not
            # retarget an already scheduled finger route.  A later consensus
            # only starts a new fold segment after the current one has been
            # visibly confirmed.
            self._register_hold_target(track, lane)
        progress_span = max(0.0, recent[-1].progress - recent[0].progress)
        speed = regression_slope(recent)
        stable_slow_motion = (
            len(recent) >= self.config.hold_long_verify_samples
            and progress_span >= self.config.hold_long_verify_progress_span
            and speed > max(self.config.hold_tail_min_motion, self.config.static_speed_threshold / 4.0)
        )
        minimum_speed = (
            max(self.config.hold_tail_min_motion, self.config.static_speed_threshold / 4.0)
            if stable_slow_motion or track.hold_long_verified
            else max(0.10, self.config.static_speed_threshold)
        )
        if speed <= minimum_speed:
            return
        predicted = last.timestamp + max(0.0, self.calibration.trigger_progress - last.progress) / speed
        predicted += self.config.hold_release_delay_ms / 1000.0
        hit_time = track.predicted_hit_time or frame.midpoint
        if (
            stable_slow_motion
            and predicted - hit_time >= self.config.hold_long_min_predicted_duration_ms / 1000.0
            and not track.hold_long_verified
        ):
            track.hold_long_verified = True
            LOGGER.info(
                "Music very-long hold verified track=%s samples=%s span=%.4f speed=%.4f predicted_duration=%.2fs",
                track.track_id,
                len(recent),
                progress_span,
                speed,
                predicted - hit_time,
            )
        maximum_ms = self.config.hold_max_tail_ms if track.hold_long_verified else self.config.hold_unverified_max_tail_ms
        maximum = (track.predicted_hit_time or frame.midpoint) + maximum_ms / 1000.0
        if predicted < frame.midpoint - 0.03 or predicted > maximum:
            return
        if not track.hold_release_locked:
            track.hold_release_time = predicted
        if track.hold_terminal_confirmed and last.progress >= self.config.hold_cap_lock_progress and not track.hold_release_locked:
            track.hold_release_locked = True
            track.linked_release_source_id = None
            LOGGER.info(
                "Music hold release locked track=%s route=%s->%s deadline=%.3f samples=%s",
                track.track_id,
                track.lane,
                track.hold_target_lane if track.hold_target_lane is not None else lane,
                track.hold_release_time,
                len(observations),
            )

    def _moving_hold_tails(
        self,
        tails: list[HoldTailDetection],
    ) -> tuple[dict[int, tuple[float, float, int]], list[int], list[bool]]:
        moving: dict[int, tuple[float, float, int]] = {}
        current_streaks = [0] * len(tails)
        current_checkpoint_flags = [False] * len(tails)
        terminal_streaks = [0] * len(tails)
        used_previous: set[int] = set()
        for index, tail in enumerate(tails):
            matches: list[tuple[float, float, int, int]] = []
            for previous_index, previous in enumerate(self.previous_hold_tails):
                if previous_index in used_previous:
                    continue
                delta_y = tail.center[1] - previous.center[1]
                distance = math.dist(tail.center, previous.center)
                if self.config.hold_tail_frame_min_motion_px <= delta_y <= 110.0 and distance <= 130.0:
                    prior_streak = self.previous_hold_tail_streaks[previous_index] if previous_index < len(self.previous_hold_tail_streaks) else 0
                    matches.append((distance, delta_y, prior_streak + 1, previous_index))
            if matches:
                best = min(matches, key=lambda item: (item[0], -item[2]))
                previous_index = best[3]
                used_previous.add(previous_index)
                moving[index] = best[:3]
                current_streaks[index] = best[2]
                previous_checkpoint = (
                    self.previous_hold_tail_checkpoint_flags[previous_index]
                    if previous_index < len(self.previous_hold_tail_checkpoint_flags)
                    else False
                )
                previous_tail = self.previous_hold_tails[previous_index]
                previous_multi_exit = previous_tail.topology == 'checkpoint' or (not previous_tail.topology and previous_tail.ribbon_exit_count >= 2)
                multi_exit = tail.topology == 'checkpoint' or (not tail.topology and tail.ribbon_exit_count >= 2)
                terminal = tail.topology == 'terminal' or (not tail.topology and tail.ribbon_exit_count == 1)
                prior_terminal = (self.previous_hold_tail_terminal_streaks[previous_index]
                                  if previous_index < len(self.previous_hold_tail_terminal_streaks) else 0)
                terminal_streaks[index] = prior_terminal + 1 if terminal else 0
                # Require two temporally associated multi-exit observations
                # before establishing checkpoint identity.  Once established,
                # retain it through one-sided judgement-line occlusion.
                current_checkpoint_flags[index] = (previous_checkpoint and terminal_streaks[index] < 2) or (multi_exit and previous_multi_exit)
        self._current_terminal_streaks = terminal_streaks
        return moving, current_streaks, current_checkpoint_flags

    def _update_active_hold_tails(self, frame: MusicFrame) -> None:
        active = [
            track
            for track in self.tracks.values()
            if track.state == TrackState.HOLDING
            and track.predicted_hit_time is not None
        ]
        if not active:
            self.previous_hold_tails = []
            self.previous_hold_tail_streaks = []
            self.previous_hold_tail_checkpoint_flags = []
            self.previous_hold_tail_terminal_streaks = []
            return
        # Gold rings are only visual *markers*.  Their surrounding ribbon
        # topology decides whether they are terminal caps (one exit) or sustain
        # checkpoints (two exits).  Keep all markers for continuity matching,
        # but only a one-exit marker may originate a release trajectory.
        tails = detect_hold_tails(frame.image, self.calibration, self.config)
        self.tap_trace.add('hold_markers', time=frame.midpoint, frame=frame.sequence,
                           markers=[(t.center, t.lane, t.topology, t.owner_lanes) for t in tails],
                           holds=[(t.track_id, t.lane, t.hold_target_lane, t.hold_release_time,
                                   t.hold_release_locked) for t in active])
        moving, current_streaks, current_checkpoint_flags = self._moving_hold_tails(tails)
        used: set[int] = set()

        def belongs(track, tail):
            # Explicit empty ownership is negative evidence; None is the
            # compatibility path for old synthetic/non-radial callers.  A
            # folded hold is owned by both its current segment source and its
            # active target.
            if tail.owner_lanes is None:
                return True
            lanes = {track.lane, self._hold_segment_source_lane(track)}
            if track.hold_target_lane is not None:
                lanes.add(track.hold_target_lane)
            return any(lane in tail.owner_lanes for lane in lanes)

        def usable_cap(tail):
            return tail.topology == 'terminal' or (not tail.topology and tail.ribbon_exit_count == 1)

        # A moving two-sided checkpoint proves that the contact is still live.
        # Refresh only a bounded watchdog window; never extrapolate a release
        # deadline from its progress and never mark TouchUp as locked.
        checkpoint_indices = [
            index
            for index, tail in enumerate(tails)
            if current_checkpoint_flags[index]
            and index in moving
            and moving[index][2] >= self.config.hold_tail_min_frames
        ]
        checkpoint_used: set[int] = set()
        for track in sorted(active, key=lambda item: item.predicted_hit_time or 0.0):
            hit_time = track.predicted_hit_time
            if hit_time is None or frame.midpoint < hit_time:
                continue
            target_lane = track.hold_target_lane if track.hold_target_lane is not None else track.lane
            choices = [
                (
                    min(abs(tails[index].lane - track.lane), abs(tails[index].lane - target_lane)),
                    tails[index].distance,
                    index,
                )
                for index in checkpoint_indices
                if index not in checkpoint_used
                and belongs(track, tails[index])
                and min(abs(tails[index].lane - track.lane), abs(tails[index].lane - target_lane)) <= 1
            ]
            if not choices:
                continue
            _lane_gap, _distance, index = min(choices)
            if track.hold_release_locked and not track.hold_tail_observations:
                self._revoke_terminal(track, frame)
            if track.hold_release_locked:
                continue
            checkpoint_used.add(index)
            maximum_ms = self.config.hold_max_tail_ms if track.hold_long_verified else self.config.hold_unverified_max_tail_ms
            maximum = hit_time + maximum_ms / 1000.0
            extended = min(maximum, frame.midpoint + self.config.hold_checkpoint_extension_ms / 1000.0)
            if track.hold_release_time is None or track.hold_release_time < extended:
                track.hold_release_time = extended
            previous_continuity = track.hold_last_continuity_time
            track.hold_last_continuity_time = frame.midpoint
            if previous_continuity is None or frame.midpoint - previous_continuity >= 0.5:
                LOGGER.info(
                    "Music hold sustain checkpoint track=%s lane=%s progress=%.3f exits=%s watchdog=%.3f",
                    track.track_id,
                    tails[index].lane,
                    tails[index].progress,
                    tails[index].ribbon_exit_count,
                    track.hold_release_time,
                )

        # Preserve the proven fallback path for a confirmed hold whose cap is
        # not yet usable.  The enlarged hard limit below permits very long
        # holds without changing ordinary yellow-hold release behaviour.
        for track in active:
            hit_time = track.predicted_hit_time
            if (
                hit_time is not None
                and frame.midpoint >= hit_time
                and len(track.hold_tail_observations) < 3
                and hold_ribbon_present(frame.image, self.calibration, track.lane)
            ):
                maximum = hit_time + self.config.hold_unverified_max_tail_ms / 1000.0
                extended = min(maximum, frame.midpoint + self.config.hold_ribbon_extension_ms / 1000.0)
                if track.hold_release_time is None or track.hold_release_time < extended:
                    track.hold_release_time = extended

        # Once a cap is acquired, position continuity is stronger evidence than
        # its temporarily ambiguous lane near the common spawn point.
        acquired = sorted(
            (track for track in active if track.hold_tail_observations),
            key=lambda item: item.predicted_hit_time or 0.0,
        )
        for track in acquired:
            observations = list(track.hold_tail_observations)
            distinct = []
            for observation in observations:
                # All supported routes descend. A repeated capture can move
                # its component centroid sideways without advancing the cap;
                # that is not a new velocity sample (especially on curves).
                if not distinct or observation.center[1] - distinct[-1].center[1] >= 2.:
                    distinct.append(observation)
            last = distinct[-1]
            expected = last.center
            allowed_distance = 150.  # two samples are needed for a velocity gate
            maximum_dy = 120.
            if len(distinct) >= 2:
                before = distinct[-2]
                base_elapsed = max(1e-4, last.timestamp - before.timestamp)
                scale = min(.30, max(0.0, frame.midpoint-last.timestamp)) / base_elapsed
                expected = (
                    last.center[0] + (last.center[0] - before.center[0]) * scale,
                    last.center[1] + (last.center[1] - before.center[1]) * scale,
                )
                projected_step = math.dist(expected,last.center)
                allowed_distance = max(24., min(90., projected_step*.5+12.))
                maximum_dy = max(120., (expected[1]-last.center[1])*1.8+20.)
            choices: list[tuple[float, int, HoldTailDetection]] = []
            for index, tail in enumerate(tails):
                if index in used:
                    continue
                delta_y = tail.center[1] - last.center[1]
                distance = math.dist(tail.center, expected)
                if -10.0 <= delta_y <= maximum_dy and distance <= allowed_distance:
                    choices.append((distance, index, tail))
            if choices:
                _distance, index, tail = min(choices)
                used.add(index)
                self._record_active_hold_tail(track, frame, tail)

        # A genuine very-long cap may leave the 150-px continuity window for
        # many seconds and later re-enter near the judgement line.  Reacquire
        # only a moving, mature cap on the established target lane, and only
        # after the hold has already exceeded the former safe-duration bound.
        # This cannot extend a static ribbon false positive or an ordinary
        # short hold.
        long_lost = sorted(
            (
                track
                for track in active
                if track.hold_long_verified
                and track.hold_tail_observations
                and track.predicted_hit_time is not None
                and frame.midpoint - track.predicted_hit_time >= self.config.hold_long_reacquire_after_ms / 1000.0
                and frame.midpoint - track.hold_tail_observations[-1].timestamp >= self.config.hold_long_reacquire_gap_ms / 1000.0
            ),
            key=lambda item: item.predicted_hit_time or 0.0,
        )
        for track in long_lost:
            target_lane = track.hold_target_lane if track.hold_target_lane is not None else track.lane
            choices: list[tuple[float, int, HoldTailDetection]] = []
            for index, tail in enumerate(tails):
                if (
                    index in used
                    or index not in moving
                    or moving[index][2] < self.config.hold_tail_min_frames
                    or current_checkpoint_flags[index]
                    or not usable_cap(tail)
                    or not belongs(track, tail)
                    or tail.progress < self.config.hold_tail_min_gap_progress
                    or tail.progress < self.config.hold_long_reacquire_min_progress
                    or abs(tail.lane - target_lane) > 1
                ):
                    continue
                lane_gap = abs(tail.lane - target_lane)
                choices.append((lane_gap * 0.30 + (1.0 - tail.progress) - moving[index][1] / 500.0, index, tail))
            if not choices:
                continue
            _score, index, tail = min(choices)
            previous_center = track.hold_tail_observations[-1].center
            track.hold_tail_observations.clear()
            track.hold_tail_reacquired = True
            used.add(index)
            LOGGER.info(
                "Music very-long hold tail re-acquisition track=%s old=(%.0f,%.0f) new=(%.0f,%.0f) progress=%.3f",
                track.track_id,
                previous_center[0],
                previous_center[1],
                tail.center[0],
                tail.center[1],
                tail.progress,
            )
            self._record_active_hold_tail(track, frame, tail)

        # Resolve linked simultaneous holds as a pair before either head can
        # consume an ambiguous cap on the opposite side.  The 2026-08-24 test
        # footage exposed two symmetric heads whose independently greedy
        # assignments crossed lane 1 all the way to lane 5.  Two fingers cannot
        # physically cross order without the ribbons intersecting, so preserve
        # left/right order and require two distinct moving caps in the frame.
        paired_waiting: set[int] = set()
        visited_pairs: set[int] = set()
        active_ids = {item.track_id for item in active}
        for track in sorted(active, key=lambda item: item.track_id):
            if (
                track.track_id in visited_pairs
                or track.linked_partner_id is None
                or track.hold_tail_observations
            ):
                continue
            partner = self.tracks.get(track.linked_partner_id)
            if (
                partner is None
                or partner.track_id not in active_ids
                or partner.hold_tail_observations
                or partner.linked_partner_id != track.track_id
            ):
                continue
            visited_pairs.update({track.track_id, partner.track_id})
            paired_waiting.update({track.track_id, partner.track_id})
            if (
                track.predicted_hit_time is None
                or partner.predicted_hit_time is None
                or frame.midpoint
                < max(track.predicted_hit_time, partner.predicted_hit_time)
                + self.config.hold_cap_acquire_delay_ms / 1000.0
            ):
                continue
            usable = [
                index
                for index, tail in enumerate(tails)
                if index not in used
                and index in moving
                and moving[index][2] >= self.config.hold_tail_min_frames
                and not current_checkpoint_flags[index]
                and usable_cap(tail)
                and tail.progress >= self.config.hold_tail_min_gap_progress
                and tail.progress <= 0.88
            ]
            if len(usable) < 2:
                continue
            left_track, right_track = sorted(
                (track, partner),
                key=lambda item: self.calibration.points[item.lane][0],
            )

            def pair_score(item: NoteTrack, tail_index: int) -> float:
                tail = tails[tail_index]
                hit_time = item.predicted_hit_time or frame.midpoint
                expected_progress = min(0.75, max(0.0, (frame.midpoint - hit_time) / 1.4))
                _motion_distance, motion_y, _motion_streak = moving[tail_index]
                return (
                    abs(tail.progress - expected_progress)
                    + tail.distance / 250.0
                    - motion_y / 500.0
                    + abs(tail.lane - item.lane) * 0.18
                )

            assignments: list[tuple[float, int, int]] = []
            for left_index in usable:
                for right_index in usable:
                    if left_index == right_index:
                        continue
                    left_tail, right_tail = tails[left_index], tails[right_index]
                    if not belongs(left_track, left_tail) or not belongs(right_track, right_tail):
                        continue
                    if left_tail.center[0] > right_tail.center[0] - 12.0:
                        continue
                    score = (
                        pair_score(left_track, left_index)
                        + pair_score(right_track, right_index)
                        + abs(left_tail.progress - right_tail.progress) * 0.25
                    )
                    assignments.append((score, left_index, right_index))
            if not assignments:
                continue
            _score, left_index, right_index = min(assignments)
            used.update({left_index, right_index})
            self._record_active_hold_tail(left_track, frame, tails[left_index])
            self._record_active_hold_tail(right_track, frame, tails[right_index])
            LOGGER.info(
                "Music linked hold tails jointly acquired tracks=%s/%s centers=(%.0f,%.0f)/(%.0f,%.0f)",
                left_track.track_id,
                right_track.track_id,
                tails[left_index].center[0],
                tails[left_index].center[1],
                tails[right_index].center[0],
                tails[right_index].center[1],
            )

        # A new solitary cap must first demonstrate downward motion.  Delay
        # acquisition past the head judgement flash, then score against a
        # conservative 1.4s duration prior; subsequent frames use position
        # continuity exclusively.
        waiting = sorted(
            (
                track
                for track in active
                if not track.hold_tail_observations and track.track_id not in paired_waiting
            ),
            key=lambda item: item.predicted_hit_time or 0.0,
        )
        for track in waiting:
            hit_time = track.predicted_hit_time
            if hit_time is None:
                continue
            expected_progress = min(0.75, max(0.0, (frame.midpoint - hit_time) / 1.4))
            choices: list[tuple[float, int, HoldTailDetection]] = []
            for index, tail in enumerate(tails):
                if (
                    index in used
                    or index not in moving
                    or moving[index][2] < 2
                    or current_checkpoint_flags[index]
                    or not usable_cap(tail)
                    or not belongs(track, tail)
                    or (frame.midpoint < hit_time + self.config.hold_cap_acquire_delay_ms / 1000.0
                        and not (tail.topology == 'terminal' and tail.owner_lanes and track.lane in tail.owner_lanes))
                    or tail.progress < self.config.hold_tail_min_gap_progress
                    or tail.progress > 0.88
                ):
                    continue
                motion_distance, motion_y, _motion_streak = moving[index]
                # Simultaneous symmetric holds expose two visually identical
                # caps.  Prefer the cap whose provisional lane remains closest
                # to this head; continuity takes over after the first match and
                # still permits the common one-lane inward/outward routes.
                lane_gap = abs(tail.lane - track.lane)
                score = (
                    abs(tail.progress - expected_progress)
                    + tail.distance / 250.0
                    + motion_distance / 160.0
                    - motion_y / 500.0
                    + lane_gap * 0.18
                )
                choices.append((score, index, tail))
            if choices:
                _score, index, tail = min(choices)
                used.add(index)
                self._record_active_hold_tail(track, frame, tail)

        # A folded hold changes lanes first and then continues along one target
        # ray.  Detect that second segment directly in the lower judgement
        # region: two frames of target ribbon with no remaining source ribbon
        # are stronger evidence than extrapolating one terminal cap.
        for track in active:
            target_lane = track.hold_target_lane
            source_lane = self._hold_segment_source_lane(track)
            if target_lane is None or target_lane == source_lane:
                continue
            source_visible = hold_ribbon_present(frame.image, self.calibration, source_lane)
            target_visible = (hold_ribbon_present(frame.image, self.calibration, target_lane)
                              and ribbon_at_judgement(frame.image, self.calibration, target_lane))
            if self.hold_policy.observe_route_ribbons(
                track,
                source_lane_visible=source_visible,
                target_lane_visible=target_visible,
            ):
                LOGGER.info(
                    "Music folded hold straight segment confirmed track=%s lane=%s->%s tail_progress=%.3f",
                    track.track_id,
                    track.lane,
                    target_lane,
                    track.hold_tail_observations[-1].progress if track.hold_tail_observations else -1.0,
                )

        # A short curved cap can merge into its ribbon before the 90% lock.
        # Infer a bounded release only after both signals have disappeared for
        # several frames.  Moving predictions and verified long holds retain
        # their existing paths unchanged.
        for track in active:
            if (
                track.hold_release_locked
                or track.hold_long_verified
                or not track.hold_terminal_confirmed
                or track.predicted_hit_time is None
                or not track.hold_tail_observations
            ):
                continue
            observations = list(track.hold_tail_observations)
            last = observations[-1]
            if last.frame_sequence == frame.sequence:
                track.hold_tail_loss_frames = 0
                continue
            age = frame.midpoint - track.predicted_hit_time
            speed = regression_slope(observations[-6:])
            target_lane = track.hold_target_lane if track.hold_target_lane is not None else track.lane
            ribbon_visible = hold_ribbon_present(frame.image, self.calibration, track.lane)
            if target_lane != track.lane:
                ribbon_visible = ribbon_visible or hold_ribbon_present(frame.image, self.calibration, target_lane)
            if (
                age > self.config.hold_tail_loss_max_age_ms / 1000.0
                or last.progress < self.config.hold_tail_loss_min_progress
                or speed > max(0.10, self.config.static_speed_threshold)
                or ribbon_visible
            ):
                track.hold_tail_loss_frames = 0
                continue
            track.hold_tail_loss_frames += 1
            if track.hold_tail_loss_frames < self.config.hold_tail_loss_confirm_frames:
                continue
            inferred = frame.midpoint + self.config.hold_tail_loss_release_delay_ms / 1000.0
            minimum = track.predicted_hit_time + self.config.hold_min_duration_ms / 1000.0
            track.hold_release_time = max(minimum, inferred)
            track.hold_release_locked = True
            track.linked_release_source_id = None
            LOGGER.info(
                "Music hold release inferred after terminal-cap/ribbon loss track=%s route=%s->%s progress=%.3f deadline=%.3f",
                track.track_id,
                track.lane,
                target_lane,
                last.progress,
                track.hold_release_time,
            )
        self.previous_hold_tails = tails
        self.previous_hold_tail_streaks = current_streaks
        self.previous_hold_tail_checkpoint_flags = current_checkpoint_flags
        self.previous_hold_tail_terminal_streaks = self._current_terminal_streaks

    def _associate_lane(
        self,
        lane: int,
        entries: list[tuple[MusicCandidate, LaneProjection]],
        frame: MusicFrame,
        visual: VisualMask,
        recovered=None,
    ) -> None:
        for track in self.tracks.values():
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
            for track in self.tracks.values()
            if track.lane == lane
            and track.state in {TrackState.HOLD_PENDING, TrackState.HOLDING}
            and not self.hold_policy.head_association_open(track, frame.midpoint)
        ]
        tracks = [
            track
            for track in self.tracks.values()
            if track.lane == lane
            and track.state not in {TrackState.RELEASED, TrackState.LOST}
            and self.hold_policy.head_association_open(track, frame.midpoint)
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
                    and (not self.config.enable_holds or cached_head_ratio(candidate) < self.config.hold_head_color_ratio)
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
        matches = associate_taps(tracks, entries, frame, self.config, safe_tap_candidate, self.tap_trace, owned)
        for index, (candidate, projection) in enumerate(entries):
            if index in matches:
                track = matches[index]
                unmatched_tracks.discard(track.track_id)
            else:
                track = self._new_track(lane, frame.midpoint)
                if active_holds_outside_head_window:
                    self.isolated_same_lane_head_count += 1
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
            self._update_motion(track)
            if index in owned:
                self.tap_trace.add('tap_mask_recovered', time=frame.midpoint, frame=frame.sequence,
                                   track=track.track_id, box=candidate.box, progress=projection.progress,
                                   raw_hit=track.predicted_hit_time)
            flick = NoteGesture.UNKNOWN if track.bonus_star else cached_flick(candidate)
            track.direction_evidence.append(flick)
            if self.config.enable_holds and track.state not in {TrackState.RELEASED, TrackState.LOST}:
                if track.bonus_star:
                    hold_evidence = bonus_hold_ribbon_present(frame.image, candidate, projection.tangent)
                else:
                    head_ratio = cached_head_ratio(candidate)
                    hold_evidence = head_ratio >= self.config.hold_head_color_ratio
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
                    and self._hold_committed(track)):
                # A queued event is not a physical press. Late structural
                # evidence may promote this same head without a second input.
                track.gesture = NoteGesture.HOLD_START
                track.state = TrackState.HOLD_PENDING
                self.tap_trace.add('head_promoted_to_hold', time=frame.midpoint,
                                   track=track.track_id, event=track.action_event_id)
            if not track.action_executed:
                if self._hold_committed(track):
                    # Two head observations are sufficient to press.  Tail tracking
                    # refines release and route, but may no longer turn a real hold
                    # into a 6-ms ordinary tap merely because its ribbon is diagonal
                    # or its cap is still close to the spawn point.
                    track.gesture = NoteGesture.HOLD_START
                elif (len(track.direction_evidence) >= 2
                        and len(set(track.direction_evidence)) == 1
                        and flick in FLICK_GESTURES
                        and track.speed > self.config.min_downward_progress):
                    # A colour-tagged blob that never moved is a stage
                    # decoration, not a flick note.
                    track.gesture = flick
                else:
                    track.gesture = NoteGesture.TAP

        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.state == TrackState.HOLDING:
                track.tail_missing_frames += 1
            elif track.state in {TrackState.TAP_PENDING, TrackState.HOLD_PENDING, TrackState.FLICK_PENDING}:
                continue
            elif self._coast_eligible(track, frame):
                # The judgement text may hide the head for many frames; keep the
                # track so its bounded prediction can still schedule the tap.
                continue
            elif track.missed_frames > self.config.track_lost_frames:
                self._record_unscheduled_head_loss(track, frame, "visual-loss")
                track.state = TrackState.LOST

    def _stabilize_dense_tap_timing(self, frame: MusicFrame) -> None:
        self.tap_policy.stabilize_dense_timing(self.tracks, frame)

    def _action_advance_ms(self, track: NoteTrack) -> float:
        if track.gesture == NoteGesture.TAP:
            return self.tap_policy.action_advance_ms(track)
        if track.gesture == NoteGesture.HOLD_START:
            return self.hold_policy.action_advance_ms(track)
        return self.config.action_advance_ms

    def _update_linked_tap_pairs(self, frame: MusicFrame) -> None:
        """Bind tap or hold heads only when the chart's white arc is visible."""
        for track in self.tracks.values():
            if track.gesture != NoteGesture.TAP or track.linked_partner_id is None:
                continue
            partner = self.tracks.get(track.linked_partner_id)
            if (partner is None or partner.gesture == NoteGesture.TAP) and not valid_tap_pair(track, partner, frame.sequence):
                self._detach_link(track)
        current = [
            track
            for track in self.tracks.values()
            if track.gesture in {NoteGesture.TAP, NoteGesture.HOLD_START}
            and (
                not track.action_executed
                or (track.gesture == NoteGesture.TAP
                    and track.state == TrackState.TAP_PENDING
                    and track.tap_input_started is None)
                # A bonus-star head becomes structurally recognisable a few
                # frames after its ordinary partner.  Keep the still-visible
                # held head eligible so the connector can bind the pair in
                # time to share tail/release evidence, even if that first
                # finger is already down.
                or (
                    track.gesture == NoteGesture.HOLD_START
                    and track.state in {TrackState.HOLD_PENDING, TrackState.HOLDING}
                )
            )
            and track.observations
            and track.observations[-1].frame_sequence == frame.sequence
        ]
        current.sort(key=lambda item: item.observations[-1].center[0])
        for index, left in enumerate(current):
            for right in current[index + 1 :]:
                if left.lane == right.lane:
                    continue
                if left.linked_partner_id not in {None, right.track_id}:
                    continue
                if right.linked_partner_id not in {None, left.track_id}:
                    continue
                left_observation = left.observations[-1]
                right_observation = right.observations[-1]
                if abs(left_observation.progress - right_observation.progress) > 0.065:
                    continue
                if not linked_tap_pair_present(
                    frame.image,
                    left_observation.candidate,
                    right_observation.candidate,
                    fast_channels=left.gesture == right.gesture == NoteGesture.TAP,
                ):
                    continue
                first_confirmation = left.linked_partner_id is None
                left.linked_partner_id = right.track_id
                right.linked_partner_id = left.track_id
                left.linked_evidence_frames += 1
                right.linked_evidence_frames += 1
                if first_confirmation:
                    LOGGER.info(
                        "Music linked note pair detected tracks=%s/%s lanes=%s/%s gestures=%s/%s progress=%.3f/%.3f",
                        left.track_id,
                        right.track_id,
                        left.lane,
                        right.lane,
                        left.gesture.value,
                        right.gesture.value,
                        left_observation.progress,
                        right_observation.progress,
                    )
                break

    def _synchronize_linked_hold_releases(self, frame: MusicFrame) -> None:
        visited: set[int] = set()
        for track in self.tracks.values():
            if track.track_id in visited or track.linked_partner_id is None:
                continue
            partner = self.tracks.get(track.linked_partner_id)
            if (
                partner is None
                or track.gesture != NoteGesture.HOLD_START
                or partner.gesture != NoteGesture.HOLD_START
                or track.state != TrackState.HOLDING
                or partner.state != TrackState.HOLDING
            ):
                continue
            visited.update({track.track_id, partner.track_id})
            locked = [item for item in (track, partner) if item.hold_release_locked and item.hold_release_time is not None]
            if len(locked) == 1:
                reliable = locked[0]
                waiting = partner if reliable is track else track
                reliable_deadline = reliable.hold_release_time
                waiting_deadline = waiting.hold_release_time
                waiting_observations = list(waiting.hold_tail_observations)
                # A connector synchronizes the heads, not arbitrary future
                # markers.  Never copy one side's TouchUp into an evidence-free
                # partner: that was the source of the observed asymmetric
                # double-hold release.  Permit a shared release only when the
                # second side independently owns a mature terminal trajectory.
                if (
                    reliable_deadline is None
                    or waiting_deadline is None
                    or not waiting.hold_terminal_confirmed
                    or len(waiting_observations) < 3
                    or waiting_observations[-1].progress < self.config.hold_cap_lock_progress - 0.08
                    or abs(waiting_deadline - reliable_deadline)
                    > self.config.hold_linked_sync_tolerance_ms / 1000.0
                ):
                    waiting.linked_release_source_id = None
                    continue
                # The locked trajectory owns the final timing estimate.  The
                # waiting side may still carry the deliberately late fallback;
                # taking max() here reproduced the observed 0.62-s asymmetric
                # release even after both visual tails had been acquired.
                shared = reliable_deadline
                waiting.hold_release_time = shared
                waiting.hold_release_locked = True
                reliable.linked_release_source_id = waiting.track_id
                waiting.linked_release_source_id = reliable.track_id
                LOGGER.info(
                    "Music linked hold terminal releases synchronized tracks=%s/%s deadline=%.3f samples=%s/%s",
                    track.track_id,
                    partner.track_id,
                    shared,
                    len(track.hold_tail_observations),
                    len(partner.hold_tail_observations),
                )
                continue
            if len(locked) > 1:
                first = track.hold_release_time
                second = partner.hold_release_time
                if first is None or second is None:
                    continue
                if abs(first - second) > self.config.hold_linked_sync_tolerance_ms / 1000.0:
                    # Independently confirmed unequal tails are allowed; the
                    # white start connector alone does not prove equal length.
                    continue
                shared = max(first, second)
                changed = abs(first - shared) > 1e-6 or abs(second - shared) > 1e-6
                track.hold_release_time = shared
                partner.hold_release_time = shared
                track.linked_release_source_id = partner.track_id
                partner.linked_release_source_id = track.track_id
                if changed:
                    LOGGER.info(
                        "Music linked hold locked releases aligned tracks=%s/%s deadline=%.3f",
                        track.track_id,
                        partner.track_id,
                        shared,
                    )
                continue
            if track.predicted_hit_time is None or partner.predicted_hit_time is None:
                continue
            pair_hit_time = (track.predicted_hit_time + partner.predicted_hit_time) / 2.0
            for item in (track, partner):
                if item.predicted_hit_time is None or len(item.hold_tail_observations) >= 3:
                    continue
                # Buy one bounded observation window beyond the 1.8-s
                # evidence-free fallback.  A rolling `now + extension` would
                # keep an unresolved chord down for the full 14-s corruption
                # guard and is therefore intentionally avoided.
                maximum_ms = min(
                    self.config.hold_unverified_max_tail_ms,
                    self.config.hold_fallback_duration_ms + self.config.hold_linked_release_extension_ms,
                )
                extended = pair_hit_time + maximum_ms / 1000.0
                if item.hold_release_time is None or item.hold_release_time < extended:
                    item.hold_release_time = extended

    def _coast_eligible(self, track: NoteTrack, frame: MusicFrame) -> bool:
        """Ordinary tap may survive the judgement-text occlusion band briefly.

        Restricted to the centre lanes that the text overlays, and resolved to
        a single owner per position: fragment tracks of one note must not each
        schedule an input.  The strongest live sibling (already dispatched,
        then most observations, then lowest id) is the only one allowed.
        """
        if not coastable_tap(track, self.config, now=frame.midpoint):
            return False
        center = self.calibration.lane_count // 2
        if abs(track.lane - center) > 1:
            return False
        last_progress = track.observations[-1].progress
        siblings = [
            other
            for other in self.tracks.values()
            if other.track_id != track.track_id
            and other.lane == track.lane
            and other.state not in {TrackState.RELEASED, TrackState.LOST}
            and other.observations
            and abs(other.observations[-1].progress - last_progress) <= 0.06
        ]
        if siblings:
            # Only siblings that themselves pass the coast gates (or already
            # dispatched their input) may claim ownership.  An ineligible
            # fragment must not veto a better sibling that could still rescue
            # the note.
            eligible = [
                other
                for other in siblings
                if other.action_executed or coastable_tap(other, self.config, now=frame.midpoint)
            ]
            owner = max([track, *eligible], key=lambda item: (item.action_executed, len(item.observations), -item.track_id))
            if owner.track_id != track.track_id:
                return False
        return True

    def _hold_committed(self, track: NoteTrack) -> bool:
        """Bonus stars commit on a single ribbon frame; others need two."""
        if not self.config.enable_holds:
            return False
        threshold = self.config.bonus_hold_min_evidence if track.bonus_star else 2
        return track.hold_evidence_frames >= threshold

    def _ready_to_schedule(self, track: NoteTrack, frame: MusicFrame) -> bool:
        if (
            track.action_executed
            or track.predicted_hit_time is None
            or track.state != TrackState.APPROACHING
        ):
            return False
        observations = list(track.observations)
        last = observations[-1] if observations else None
        coast_ready = last is not None and self._coast_eligible(track, frame)
        # A prediction is useful for a short visual dropout, but never after a
        # track has been absent for the full lost-frame window.  This is the
        # direct guard against the recorded 3.9--11.8 second late Tap events.
        # Coast-eligible tracks keep their bounded prediction while the
        # judgement text tints/fragments the centre-lane pixels.
        if last is None or (frame.sequence - last.frame_sequence > 2 and not coast_ready):
            return False
        if not coast_ready and not late_birth_ready(track, lambda c: assign_lane(c, self.calibration)):
            return False
        standard_ready = len(observations) >= 4 or coast_ready
        scheduled_hit_time = (
            self.tap_policy.hit_time(track)
            if track.gesture == NoteGesture.TAP
            else track.predicted_hit_time
        )
        if scheduled_hit_time is None:
            return False
        action_deadline = scheduled_hit_time - self._action_advance_ms(track) / 1000.0
        if track.gesture == NoteGesture.TAP:
            urgent_ready = self.tap_policy.urgent_ready(track, frame, action_deadline)
        elif track.gesture == NoteGesture.HOLD_START:
            urgent_ready = self.hold_policy.urgent_ready(track, frame, action_deadline)
        else:
            urgent_ready = False
        if not standard_ready and not urgent_ready:
            return False
        if track.speed < self.config.min_downward_progress:
            return False
        if scheduled_hit_time - frame.midpoint > self.config.max_schedule_horizon_ms / 1000.0:
            return False
        if not 0.35 <= last.progress < 0.99:
            return False
        if track.bonus_star and track.gesture != NoteGesture.HOLD_START:
            tap_commit_progress = 0.70 if track.hold_evidence_frames else 0.58
            if last.progress < tap_commit_progress:
                return False
        return True

    def update(self, frame: MusicFrame, candidates: Iterable[MusicCandidate], visual: VisualMask) -> list[MusicActionEvent]:
        self.last_frame_sequence = frame.sequence
        self._expire_approaching_tracks(frame)
        ordinary_candidates = list(candidates)
        bonus_candidates = detect_bonus_star_notes(frame.image, self.calibration) if self.config.enable_holds else []
        # A bonus note can still be returned by the ordinary colour provider
        # while it is small.  Replace that partial inner-disc box with the
        # explicitly labelled bonus candidate so it creates only one track.
        if bonus_candidates:
            retire_bonus_fragments(self.tracks, bonus_candidates, frame,
                                   lambda c: assign_lane(c, self.calibration), self.tap_trace)
            filtered: list[MusicCandidate] = []
            for candidate in ordinary_candidates:
                if any(
                    math.dist(candidate.center, star.center)
                    <= max(24.0, max(star.box[2:]) * 0.55)
                    for star in bonus_candidates
                ):
                    continue
                filtered.append(candidate)
            ordinary_candidates = filtered
        # Flick colour families are merged into the ordinary candidate mask;
        # tag the vivid sprites here so they flow through the single tracking
        # path.  No parallel channel, no candidate replacement.
        if self.config.enable_flicks:
            ordinary_candidates = tag_flick_candidates(frame.image, ordinary_candidates)
            detected = [item for item in ordinary_candidates if item.variant == "flick"]
            if detected:
                self.tap_trace.add(
                    'flick_detected', time=frame.midpoint, frame=frame.sequence,
                    candidates=[[list(item.center), item.flick_direction.value, item.flick_color, list(item.box)]
                                for item in detected],
                )
        special_candidate: MusicCandidate | None = None
        special_events: list[MusicActionEvent] = []
        if self.config.enable_holds:
            special_candidate, special_events = self._update_center_color_note(frame, bonus_candidates)
        by_lane: dict[int, list[tuple[MusicCandidate, LaneProjection]]] = {lane: [] for lane in range(self.calibration.lane_count)}
        for candidate in [*ordinary_candidates, *bonus_candidates]:
            if special_candidate is not None:
                exclusion_radius = max(42.0, max(special_candidate.box[2:]) * 0.75)
                if math.dist(candidate.center, special_candidate.center) <= exclusion_radius:
                    continue
            projection = assign_lane(candidate, self.calibration)
            if projection is not None:
                by_lane[projection.lane].append((candidate, projection))
        recovered = recover_masked_taps(self.tracks, frame, self.calibration,
                                        lambda c: assign_lane(c, self.calibration))
        for lane, entries in by_lane.items():
            entries = unique_head_candidates(entries, self.tap_trace, frame)
            self._associate_lane(lane, entries, frame, visual, recovered)
        self._update_linked_tap_pairs(frame)
        retire_converged_shadows(self.tracks, frame, self.tap_trace)
        self._stabilize_dense_tap_timing(frame)
        self._bind_hold_end_flicks(frame)
        events: list[MusicActionEvent] = list(special_events)
        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            if track.hold_end_owner is not None:
                # A ribbon-tip flick bound to a hold never swipes on its own;
                # its prediction drives the hold's held-flick release instead.
                continue
            if not self._ready_to_schedule(track, frame):
                continue
            last = track.observations[-1]
            urgent_rescue = len(track.observations) == 3
            event_hit_time = (
                self.tap_policy.hit_time(track)
                if track.gesture == NoteGesture.TAP
                else track.predicted_hit_time
            )
            partner = self.tracks.get(track.linked_partner_id) if track.linked_partner_id is not None else None
            linked_note = (
                track.gesture in {NoteGesture.TAP, NoteGesture.HOLD_START}
                and partner is not None
                and partner.gesture == track.gesture
                and partner.predicted_hit_time is not None
                and (track.gesture == NoteGesture.HOLD_START
                     or (valid_tap_pair(track, partner, frame.sequence)
                         and coherent_tap_predictions(track, partner)))
            )
            if linked_note:
                # Do not let the faster independent fit fire before the second
                # head has accumulated the same scheduling-quality evidence.
                if not partner.action_executed and not self._ready_to_schedule(partner, frame):
                    continue
                event_hit_time = (track.predicted_hit_time + partner.predicted_hit_time) / 2.0
            event_id = f"track-{track.track_id}@{int(event_hit_time * 1000)}"
            track.action_event_id = event_id
            if track.gesture == NoteGesture.HOLD_START:
                start_deadline = track.predicted_hit_time - self.hold_policy.action_advance_ms(track) / 1000.0
                duplicate = next(
                    (
                        other
                        for other in self.tracks.values()
                        if other.track_id != track.track_id
                        and other.lane == track.lane
                        and other.state in {TrackState.HOLD_PENDING, TrackState.HOLDING}
                        and other.action_executed
                        and other.predicted_hit_time is not None
                        and (
                            abs(other.predicted_hit_time - track.predicted_hit_time)
                            <= self.config.hold_same_lane_guard_ms / 1000.0
                            # A newly created trajectory that predicts an
                            # earlier hit than the active owner is stale.  A
                            # later head beyond the duplicate guard is a real
                            # same-lane handoff and is resolved atomically by
                            # the runtime instead of being discarded here.
                            or track.predicted_hit_time < other.predicted_hit_time
                        )
                    ),
                    None,
                )
                if duplicate is not None:
                    track.state = TrackState.LOST
                    track.action_executed = True
                    LOGGER.warning(
                        "Music suppressed overlapping hold track=%s lane=%s prior_track=%s hit_gap_ms=%.0f prior_release=%s new_start=%.3f",
                        track.track_id,
                        track.lane,
                        duplicate.track_id,
                        abs((duplicate.predicted_hit_time or 0.0) - track.predicted_hit_time) * 1000.0,
                        f"{duplicate.hold_release_time:.3f}" if duplicate.hold_release_time is not None else "pending",
                        start_deadline,
                    )
                    continue
                # Scheduling is not execution.  Runtime promotes this track to
                # HOLDING only after executor.touch_down returns successfully.
                # Until then, tail acquisition and route/release generation are
                # deliberately disabled.
                track.state = TrackState.HOLD_PENDING
            elif track.gesture in FLICK_GESTURES:
                track.state = TrackState.FLICK_PENDING
            else:
                track.state = TrackState.TAP_PENDING
            point = self.calibration.points[track.lane]
            advance_ms = self._action_advance_ms(track)
            events.append(MusicActionEvent(
                event_id=event_id,
                track_id=track.track_id,
                lane=track.lane,
                gesture=track.gesture,
                deadline=event_hit_time - advance_ms / 1000.0,
                coordinate=(int(point[0]), int(point[1])),
                direction=track.gesture if track.gesture in FLICK_GESTURES else NoteGesture.UNKNOWN,
                contact_policy="persistent" if track.gesture == NoteGesture.HOLD_START else "auto",
                source_capture_started=frame.capture_started,
                source_capture_finished=frame.capture_finished,
                tap_reference_hit_time=event_hit_time if track.gesture == NoteGesture.TAP else None,
            ))
            if linked_note and partner is not None and track.track_id < partner.track_id:
                LOGGER.info(
                    "Music linked %s pair scheduled tracks=%s/%s shared_deadline=%.3f raw_skew_ms=%.1f",
                    "hold" if track.gesture == NoteGesture.HOLD_START else "tap",
                    track.track_id,
                    partner.track_id,
                    event_hit_time - advance_ms / 1000.0,
                    abs(track.predicted_hit_time - partner.predicted_hit_time) * 1000.0,
                )
            if track.bonus_star:
                LOGGER.info(
                    "Music bonus star scheduled track=%s lane=%s gesture=%s progress=%.3f ribbon_evidence=%s deadline=%.3f",
                    track.track_id,
                    track.lane,
                    track.gesture.value,
                    last.progress,
                    track.hold_evidence_frames,
                    track.predicted_hit_time - advance_ms / 1000.0,
                )
            if track.gesture == NoteGesture.HOLD_START:
                LOGGER.info(
                    "Music hold start track=%s lane=%s tail_samples=%s release=%s",
                    track.track_id,
                    track.lane,
                    len(track.hold_tail_observations),
                    f"{track.hold_release_time:.3f}" if track.hold_release_time is not None else "fallback",
                )
            if urgent_rescue:
                self.urgent_rescue_count += 1
                LOGGER.info(
                    "Music urgent fresh-head rescue track=%s lane=%s gesture=%s progress=%.3f deadline_delta_ms=%.1f",
                    track.track_id,
                    track.lane,
                    track.gesture.value,
                    last.progress,
                    (events[-1].deadline - frame.midpoint) * 1000.0,
                )
            track.action_executed = True
            if track.gesture == NoteGesture.TAP:
                self.tap_trace.add('scheduled', frame=frame.sequence, time=frame.midpoint,
                                   event=event_id, track=track.track_id, lane=track.lane,
                                   raw_hit=track.predicted_hit_time, deadline=events[-1].deadline,
                                   observations=[(o.timestamp, o.progress) for o in track.observations][-6:])
        if self.config.enable_holds:
            self._update_active_hold_tails(frame)
            self._synchronize_linked_hold_releases(frame)
        self._prune_terminal_tracks(frame)
        return events

    def refine_pending(self, pending: list[MusicActionEvent], now: float) -> list[MusicActionEvent]:
        """Refresh every not-yet-due deadline from the latest per-track predictions."""
        maximum_age = self.config.max_schedule_horizon_ms / 1000.0 + 0.5
        refined: list[MusicActionEvent] = []
        for event in pending:
            if event.source_capture_finished > 0.0 and now - event.source_capture_finished > maximum_age:
                track = self.tracks.get(event.track_id)
                if track is not None and track.state in {TrackState.TAP_PENDING, TrackState.FLICK_PENDING}:
                    track.state = TrackState.RELEASED
                continue
            if (
                event.event_id.startswith("center-color-")
                and self.center_color_track is not None
                and self.center_color_track.track_id == event.track_id
                and self.center_color_track.predicted_hit_time is not None
                and event.deadline > now + 0.02
            ):
                event = replace(
                    event,
                    deadline=self.center_color_track.predicted_hit_time
                    - self.config.center_tap_action_advance_ms / 1000.0,
                )
                refined.append(event)
                continue
            track = self.tracks.get(event.track_id)
            if track is not None:
                if (event.gesture == NoteGesture.TAP and track.gesture == NoteGesture.HOLD_START
                        and track.state == TrackState.HOLD_PENDING and track.tap_input_started is None):
                    event = replace(event, gesture=NoteGesture.HOLD_START, contact_policy='persistent',
                                    tap_group_id=None, tap_frozen=False)
                advance = self._action_advance_ms(track) / 1000.0
                # Hold deadlines may move later when a late cap or a still-live
                # ribbon is observed.  Refresh them even inside the 20-ms tap
                # freeze window; ordinary notes retain the proven freeze rule.
                if event.event_id.startswith("release-") and track.hold_release_time is not None:
                    # The release event is created as soon as the terminal is
                    # locked, often long before the ribbon-tip flick sprite
                    # becomes visible.  Keep upgrading the gesture and pulling
                    # the deadline toward the tip arrival until it fires.
                    release_deadline = self._hold_release_deadline(track, track.hold_release_time) or track.hold_release_time
                    direction = self._hold_release_direction(track, release_deadline, log=False)
                    if direction in FLICK_GESTURES:
                        if event.gesture != direction:
                            LOGGER.info(
                                "Music hold end flick armed track=%s lane=%s direction=%s deadline=%.3f",
                                track.track_id,
                                track.lane,
                                direction.value,
                                release_deadline,
                            )
                        event = replace(event, deadline=release_deadline, gesture=direction,
                                        direction=direction, contact_policy="held_flick")
                    else:
                        event = replace(event, deadline=release_deadline)
                elif event.gesture == NoteGesture.HOLD_END and track.hold_release_time is not None:
                    event = replace(event, deadline=self._hold_release_deadline(track, track.hold_release_time))
                elif (
                    event.gesture == NoteGesture.HOLD_CONTINUE
                    and not event.event_id.startswith("route-")
                    and track.hold_release_time is not None
                ):
                    target_lane = track.hold_target_lane if track.hold_target_lane is not None else track.lane
                    target = self.calibration.points[target_lane]
                    event = replace(
                        event,
                        deadline=track.hold_release_time - self.config.hold_move_advance_ms / 1000.0,
                        coordinate=(int(target[0]), int(target[1])),
                    )
                elif (
                    event.deadline > now + 0.02
                    and event.gesture in {NoteGesture.HOLD_START, *FLICK_GESTURES}
                    and track.predicted_hit_time is not None
                ):
                    hit_time = (
                        self.tap_policy.hit_time(track)
                        if event.gesture == NoteGesture.TAP
                        else track.predicted_hit_time
                    )
                    if event.gesture in {NoteGesture.TAP, NoteGesture.HOLD_START} and track.linked_partner_id is not None:
                        partner = self.tracks.get(track.linked_partner_id)
                        if (
                            partner is not None
                            and partner.gesture == event.gesture
                            and partner.predicted_hit_time is not None
                        ):
                            hit_time = (hit_time + partner.predicted_hit_time) / 2.0
                    event = replace(event, deadline=hit_time - advance)
            refined.append(event)
        return self.tap_chords.refine(refined, self.tracks, now, self.last_frame_sequence)

    def release_events(self, now: float) -> list[MusicActionEvent]:
        events: list[MusicActionEvent] = []
        for track in self.tracks.values():
            if track.state != TrackState.HOLDING:
                continue
            release_time = track.hold_release_time
            if release_time is None and track.predicted_hit_time is not None:
                fallback_ms = min(self.config.hold_fallback_duration_ms, self.config.hold_max_tail_ms)
                release_time = track.predicted_hit_time + fallback_ms / 1000.0
                track.hold_release_time = release_time
            if release_time is None:
                continue
            release_time = self._hold_release_deadline(track, release_time) or release_time
            target_lane = track.hold_target_lane
            if (
                track.hold_fold_route_confirmed
                and not track.hold_fold_move_scheduled
                and target_lane is not None
                and target_lane != track.lane
            ):
                target = self.calibration.points[target_lane]
                events.append(MusicActionEvent(
                    event_id=f"route-fold-{track.track_id}",
                    track_id=track.track_id,
                    lane=track.lane,
                    gesture=NoteGesture.HOLD_CONTINUE,
                    deadline=now,
                    coordinate=(int(target[0]), int(target[1])),
                    contact_policy="persistent",
                ))
                track.hold_fold_move_scheduled = True
                track.hold_move_scheduled = True
                track.hold_route_steps_completed = 3
                LOGGER.info(
                    "Music folded hold moved onto straight target segment track=%s lane=%s->%s point=%s",
                    track.track_id,
                    track.lane,
                    target_lane,
                    (int(target[0]), int(target[1])),
                )
            if target_lane is not None and target_lane != self._hold_segment_source_lane(track) and track.hold_tail_observations:
                route_steps = ((0.32, 0.35), (0.55, 0.65), (0.76, 0.88))
                step_index = track.hold_route_steps_completed
                if step_index < len(route_steps):
                    progress_threshold, route_fraction = route_steps[step_index]
                    tail_progress = track.hold_tail_observations[-1].progress
                    if tail_progress >= progress_threshold:
                        start = self.calibration.points[self._hold_segment_source_lane(track)]
                        target = self.calibration.points[target_lane]
                        coordinate = (
                            int(round(start[0] + (target[0] - start[0]) * route_fraction)),
                            int(round(start[1] + (target[1] - start[1]) * route_fraction)),
                        )
                        events.append(MusicActionEvent(
                            event_id=f"route-{track.track_id}-{step_index + 1}",
                            track_id=track.track_id,
                            lane=track.lane,
                            gesture=NoteGesture.HOLD_CONTINUE,
                            deadline=now,
                            coordinate=coordinate,
                            contact_policy="persistent",
                        ))
                        track.hold_route_steps_completed += 1
                        LOGGER.info(
                            "Music hold route step track=%s lane=%s->%s step=%s progress=%.3f point=%s",
                            track.track_id,
                            track.lane,
                            target_lane,
                            track.hold_route_steps_completed,
                            tail_progress,
                            coordinate,
                        )
            if not track.hold_move_scheduled and target_lane is not None and target_lane != track.lane:
                target = self.calibration.points[target_lane]
                events.append(MusicActionEvent(
                    event_id=f"move-{track.track_id}",
                    track_id=track.track_id,
                    lane=track.lane,
                    gesture=NoteGesture.HOLD_CONTINUE,
                    deadline=release_time - self.config.hold_move_advance_ms / 1000.0,
                    coordinate=(int(target[0]), int(target[1])),
                    contact_policy="persistent",
                ))
                track.hold_move_scheduled = True
                LOGGER.info("Music hold route track=%s lane=%s->%s", track.track_id, track.lane, target_lane)
            if track.hold_release_scheduled:
                if now >= release_time:
                    track.state = TrackState.RELEASED
                    self._retire_hold_end_flick(track)
                continue
            point = self.calibration.points[track.lane]
            end_direction = self._hold_release_direction(track, release_time)
            gesture = end_direction if end_direction in FLICK_GESTURES else NoteGesture.HOLD_END
            events.append(MusicActionEvent(
                event_id=f"release-{track.track_id}",
                track_id=track.track_id,
                lane=track.lane,
                gesture=gesture,
                deadline=release_time,
                coordinate=(int(point[0]), int(point[1])),
                direction=gesture if gesture in FLICK_GESTURES else NoteGesture.UNKNOWN,
                contact_policy="held_flick" if gesture in FLICK_GESTURES else "persistent",
            ))
            track.hold_release_scheduled = True
            if now >= release_time:
                track.state = TrackState.RELEASED
                self._retire_hold_end_flick(track)
        return events

    def _retire_hold_end_flick(self, hold: NoteTrack) -> None:
        """Retire the bound ribbon-tip track once its hold has released."""
        if hold.hold_end_flick_track is None:
            return
        bound = self.tracks.get(hold.hold_end_flick_track)
        if bound is not None and bound.state not in {TrackState.RELEASED, TrackState.LOST}:
            bound.state = TrackState.RELEASED
