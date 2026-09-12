from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque

from agent.common import parse_custom_param


BASE_WIDTH = 1280
BASE_HEIGHT = 720
SUPPORTED_LANE_COUNTS = (7, 9)
VISUAL_BASELINE_VERSION = "maes-music-v4-2026-08"


class NoteGesture(str, Enum):
    TAP = "Tap"
    HOLD_START = "HoldStart"
    HOLD_CONTINUE = "HoldContinue"
    HOLD_END = "HoldEnd"
    FLICK_LEFT = "FlickLeft"
    FLICK_RIGHT = "FlickRight"
    FLICK_UP = "FlickUp"
    FLICK_DOWN = "FlickDown"
    UNKNOWN = "Unknown"


# Every gesture that must be executed as a directional swipe.  Membership is
# the single source of truth for the executor, runtime dispatch, tap-identity
# exclusions and the hold-end release path.
FLICK_GESTURES = frozenset({
    NoteGesture.FLICK_LEFT,
    NoteGesture.FLICK_RIGHT,
    NoteGesture.FLICK_UP,
    NoteGesture.FLICK_DOWN,
})

# User-confirmed colour table (matches the white-arrow direction seen in the
# validated capture): blue/right, red/left, violet/up, pink/down.  Used by
# ``flick_direction_source == "color"`` and as conflict diagnostics.
FLICK_COLOR_DIRECTIONS = {
    "blue": NoteGesture.FLICK_RIGHT,
    "red": NoteGesture.FLICK_LEFT,
    "violet": NoteGesture.FLICK_UP,
    "pink": NoteGesture.FLICK_DOWN,
}


class TrackState(str, Enum):
    APPROACHING = "Approaching"
    TAP_PENDING = "TapPending"
    # The visual head has been scheduled, but the framework has not yet
    # acknowledged TouchDown.  Tail tracking and route events must not run in
    # this state: the live failure at track 345 proved that treating a queued
    # head as an active contact can manufacture owner-less tail movement.
    HOLD_PENDING = "HoldPending"
    HOLDING = "Holding"
    FLICK_PENDING = "FlickPending"
    RELEASED = "Released"
    LOST = "Lost"


class MusicFailureCode(str, Enum):
    NONE = "none"
    PAUSED_BEFORE_START = "paused_before_start"
    INVALID_LIVE_SCREEN = "invalid_live_screen"
    INVALID_CALIBRATION = "invalid_calibration"
    PERFORMANCE_REJECTED = "performance_rejected"
    CAPTURE_FAILURE = "capture_failure"
    CANDIDATE_FAILURE = "candidate_failure"
    TOUCH_BACKEND_FUSED = "touch_backend_fused"
    CANCELLED = "cancelled"
    SONG_TIMEOUT = "song_timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class MusicConfig:
    lane_count: int = 0
    experimental_only: bool = False
    max_duration_seconds: float = 210.0
    sample_interval_ms: int = 0
    pause_check_interval_ms: int = 250
    end_check_interval_ms: int = 2500
    terminal_initial_delay_ms: int = 2000
    # Expensive result OCR is allowed only after real chart activity has been
    # seen and the chart has then remained completely quiet.  The giant LIVE
    # detector remains a cheap per-frame path, so normal completion is not
    # delayed by this guard.
    terminal_min_runtime_ms: int = 15000
    terminal_quiet_ms: int = 1500
    # MaaFramework's Job.wait() has no timeout.  Poll the asynchronous status
    # instead so a broken EmulatorExtras surface cannot strand cancellation or
    # application shutdown indefinitely.
    capture_timeout_ms: int = 1200
    max_capture_failures: int = 3
    max_provider_failures: int = 3
    preflight_warmup_samples: int = 5
    preflight_samples: int = 30
    max_capture_p95_ms: float = 25.0
    max_provider_p95_ms: float = 15.0
    max_loop_p95_ms: float = 33.0
    max_click_touch_ms: float = 20.0
    # Legacy/default timing remains available to flicks and old calibration
    # records.  Tap and HoldStart own explicit copies below so tuning either
    # subsystem cannot alter the other one by accident.
    action_advance_ms: float = 125.0
    tap_action_advance_ms: float = 125.0
    hold_start_action_advance_ms: float = 125.0
    deadline_execution_window_ms: float = 15.0
    # Live runs on this controller have a 76--79 ms loop p95.  Enter precision
    # waiting for every already-scheduled tap before the next capture can strand
    # it one frame late.  This changes neither visual classification nor the
    # predicted deadline; every gesture still uses a bounded window below.
    tap_execution_window_ms: float = 90.0
    special_tap_execution_window_ms: float = 100.0
    dense_tap_execution_window_ms: float = 90.0
    dense_tap_neighbour_ms: float = 750.0
    # Ordinary heads in a dense passage are fitted independently and their
    # short regressions tend to overestimate the next head's speed.  Delay
    # only clusters of at least three non-linked taps by 30 ms; isolated taps,
    # white-arc chords, bonus notes and every hold retain the proven 125-ms
    # advance.
    dense_tap_action_advance_ms: float = 95.0
    dense_tap_cluster_horizon_ms: float = 420.0
    dense_tap_min_tracks: int = 3
    # Two-note bursts were absent from the former three-track cluster rule even
    # though same/adjacent-lane pairs are the most common early-hit failure.
    # Keep this narrower than the general cluster horizon and inside tap policy.
    dense_tap_pair_horizon_ms: float = 320.0
    # Same-lane equal-cadence streams in the live chart are roughly 0.45--0.55s
    # apart.  They need a wider, same-lane-only window than adjacent-lane bursts.
    dense_same_lane_pair_horizon_ms: float = 700.0
    tap_cadence_min_ms: float = 180.0
    tap_cadence_max_ms: float = 800.0
    tap_cadence_tolerance_ratio: float = 0.28
    tap_cadence_max_correction_ms: float = 350.0
    dense_same_lane_min_gap_ms: float = 45.0
    # A real head can first become separable one capture too late in the
    # 47--66 ms live loop.  Three samples may schedule Tap or HoldStart only
    # inside this narrow deadline window.  The normal path still requires four
    # samples, while track-age/state guards reject long-lived stage effects.
    # Three-frame fits farther than 180 ms from dispatch correlated strongly
    # with the early Bad/Miss-heavy runs on 2026-09-04.  Far heads have time
    # to obtain the reliable fourth sample; reserve rescue for genuinely
    # imminent notes.  The centre lane keeps a lower progress gate because
    # one compact centre head was otherwise never scheduled.
    urgent_tap_min_progress: float = 0.48
    center_urgent_tap_min_progress: float = 0.42
    urgent_tap_deadline_horizon_ms: float = 180.0
    # A newly separated head can already be slightly past its ideal
    # action-advance deadline.  It is still recoverable near the judgement
    # line, so do not force it to wait for a fourth frame merely because the
    # 90-ms precision window has just elapsed.
    urgent_tap_late_horizon_ms: float = 220.0
    # HoldStart rescue intentionally mirrors the currently proven live values,
    # but is no longer wired to the ordinary-tap controls above.
    urgent_hold_start_min_progress: float = 0.48
    urgent_hold_start_deadline_horizon_ms: float = 180.0
    urgent_hold_start_late_horizon_ms: float = 220.0
    # The large centre note is tracked by its centre, unlike compact notes.  A
    # smaller visual advance avoids firing while its expanding disc is still
    # well above the judgement point (three of six recorded samples were Bad).
    center_tap_action_advance_ms: float = 50.0

    # Judge-text occlusion coasting for ordinary taps: a track with healthy
    # forward motion may keep its prediction-driven scheduling alive for a
    # bounded age while the judgement text tints/fragments the lane pixels.
    coast_enabled: bool = True
    coast_min_speed: float = 0.25
    coast_min_progress: float = 0.45
    coast_max_age_ms: float = 1600.0

    # The bonus-star head is identical for taps and holds, so one confirmed
    # ribbon frame must commit a bonus note to a hold (evidence never decays
    # for bonus stars within the track).
    bonus_hold_min_evidence: int = 1
    # Live capture loops measure 78--93 ms p95 on the user's controller.  The
    # deadline itself is unchanged; hold heads merely enter precision waiting
    # early enough that a complete capture cannot strand TouchDown one frame
    # late.  Ordinary taps retain their independent bounded window above.
    hold_start_execution_window_ms: float = 100.0
    # TouchDown precedes visual arrival.  Only this short post-arrival grace may
    # keep associating the visible head to an active hold; later same-lane heads
    # must create independent tracks.
    hold_head_association_grace_ms: float = 120.0
    event_late_tolerance_ms: float = 45.0
    association_progress_delta: float = 0.14
    min_downward_progress: float = 0.015
    track_lost_frames: int = 8
    static_speed_threshold: float = 0.04
    max_schedule_horizon_ms: float = 3000.0
    flick_distance_px: int = 56
    # Swipe timing: the direct touch events must be spread over real time or
    # the emulator coalesces them into a tap.  ``flick_steps`` interpolated
    # moves over ``flick_duration_ms`` plus an end hold before release.
    flick_duration_ms: int = 60
    flick_steps: int = 3
    flick_end_hold_ms: float = 16.0
    flick_direction_ratio: float = 1.25
    flick_min_pixels: int = 8
    # A contact is still released by the runtime's finally block and by the
    # song timeout.  Keep the watchdog beyond the longest accepted song so a
    # legitimate very long hold is not cut off at the former 15-second limit.
    max_contact_ms: int = 610000
    max_contacts: int = 10
    hold_tail_missing_frames: int = 2
    # Tail/ribbon evidence may legitimately span almost the whole song.  This
    # is a corruption guard, not a duration prior; the evidence-free fallback
    # below remains short and conservative.
    hold_max_tail_ms: float = 600000.0
    # Ribbon-only or incomplete-cap evidence is intentionally bounded at the
    # former proven limit.  The extended limit above is unlocked only by a
    # stable moving tail, preventing a pale lane/background from holding a
    # contact for the rest of the song.
    hold_unverified_max_tail_ms: float = 14000.0
    hold_fallback_duration_ms: float = 1800.0
    hold_ribbon_extension_ms: float = 700.0
    hold_cap_acquire_delay_ms: float = 250.0
    hold_cap_lock_progress: float = 0.90
    hold_same_lane_guard_ms: float = 450.0
    hold_head_color_ratio: float = 0.55
    hold_tail_min_score: float = 0.14
    hold_tail_min_pixels: int = 18
    hold_tail_min_gap_progress: float = 0.12
    hold_tail_min_motion: float = 0.01
    hold_tail_min_frames: int = 2
    # Very long tails can move by less than two pixels between live frames.
    # Keep acquisition below that speed while still requiring a multi-frame
    # downward streak before a cap is trusted.
    hold_tail_frame_min_motion_px: float = 0.75
    # A provisional tail lane near the common spawn point can wobble by one
    # lane.  Require a later, three-frame consensus before moving a held finger.
    hold_target_min_progress: float = 0.38
    hold_target_confirm_samples: int = 3
    # Folded (ribbon-visible) lane changes: Normal charts can chain A~B~A
    # style folds.  A new hop is only registered after the current segment's
    # ribbon has visibly arrived on its target.
    hold_max_route_hops: int = 3
    # Confirm that a lane change has completed and the remaining ribbon follows
    # one target lane.  This is the extra state needed by folded holds.
    hold_fold_target_confirm_frames: int = 2
    # Some short curved tails merge into their pale ribbon and disappear before
    # a 90%-progress cap lock.  If a mature, static cap and the ribbon both
    # disappear for three frames, infer only this short release instead of
    # retaining the 1.8-s evidence-free fallback.  Verified long holds bypass it.
    hold_tail_loss_min_progress: float = 0.52
    hold_tail_loss_confirm_frames: int = 3
    hold_tail_loss_release_delay_ms: float = 80.0
    hold_tail_loss_max_age_ms: float = 3500.0
    hold_long_verify_samples: int = 4
    hold_long_verify_progress_span: float = 0.006
    # A short cap moving unusually slowly can extrapolate to 14--20 seconds.
    # The new recordings contain two such false positives, while the verified
    # ultra-long sample predicts roughly 40 seconds.  Keep the extended
    # watchdog locked until the fit is unambiguously beyond the false range.
    hold_long_min_predicted_duration_ms: float = 22000.0
    hold_long_reacquire_gap_ms: float = 900.0
    hold_long_reacquire_after_ms: float = 13000.0
    hold_long_reacquire_min_progress: float = 0.62
    # A two-sided hourglass marker is a sustain checkpoint, not a tail.  It may
    # refresh the contact watchdog briefly but can never predict or lock TouchUp.
    hold_checkpoint_extension_ms: float = 1200.0
    hold_linked_release_extension_ms: float = 700.0
    # The newest paired-hold recording left one mature side on its 0.62-s-late
    # fallback even though the other side had already locked the shared cap.
    # Seven hundred milliseconds is still bounded by the existing linked-hold
    # observation window and is used only after both terminal paths are proven.
    hold_linked_sync_tolerance_ms: float = 700.0
    hold_min_duration_ms: float = 180.0
    # Offline autoplay ground truth shows that the visual cap disappears about
    # 0.1s after the last stable 90%-progress fit.  The release delay absorbs
    # that judgement-window remainder instead of letting every route end early.
    hold_release_delay_ms: float = 120.0
    hold_long_duration_ms: float = 2500.0
    # The measured live loop p95 is roughly 83 ms.  Enter precision waiting
    # early for every cap-verified release so its final TouchUp is not stranded
    # one capture late.  Evidence-free fallback holds retain the base window.
    hold_release_execution_window_ms: float = 110.0
    # Evidence-free fallback releases previously kept the generic 15-ms
    # window.  With a 65--77 ms live loop that made otherwise correct TouchUp
    # events one frame late.  This only widens precision dispatch; it does not
    # move the release deadline itself.
    hold_fallback_release_execution_window_ms: float = 80.0
    # Final route movement used the 15-ms generic window even though the live
    # loop is normally 60--90 ms.  Enter precision dispatch early enough to
    # prevent the observed 70--88 ms late final lane moves.
    hold_move_execution_window_ms: float = 110.0
    # Service an already predicted near action before beginning the next
    # controller screenshot.  The isolated preview controller raised capture
    # p95 to roughly 30 ms, so a 45-ms guard prevents the capture itself from
    # carrying a tap/head/tail across its deadline.
    pre_capture_deadline_guard_ms: float = 45.0
    hold_move_advance_ms: float = 280.0
    enable_holds: bool = False
    # Four-direction flick support.  Direction is classified purely by the
    # sprite's colour family (blue=right, red=left, violet=up, pink=down).
    enable_flicks: bool = True
    # A long-press chart may end with a flick sprite riding the ribbon tip.
    # A same-lane flick track at or beyond this progress can bind to an
    # active hold and turn its release into a held flick.
    hold_end_flick_min_progress: float = 0.45
    double_press_p95_ms: float = 25.0
    candidate_min_pixels: int = 12
    candidate_min_size: int = 18
    candidate_iou_threshold: float = 0.55
    provider: str = "auto"

    # Compatibility classifier settings retained for direct unit-level use.
    white_threshold: int = 238
    flick_brightness_threshold: int = 150
    flick_chroma_threshold: int = 35
    flick_min_signal_pixels: int = 8
    flick_max_signal_ratio: float = 0.42
    flick_direction_margin: float = 0.25
    flick_centroid_margin: float = 0.12
    hold_min_bands: int = 2

    @classmethod
    def from_param(cls, value: Any) -> "MusicConfig":
        params = parse_custom_param(value)
        lane_value = params.get("lane_count", 0)
        lane_count = 0 if lane_value in (None, "", "auto", "Auto") else int(lane_value)
        values: dict[str, Any] = {}
        for field_name, default_field in cls.__dataclass_fields__.items():
            if field_name not in params:
                continue
            default = default_field.default
            raw = params[field_name]
            if isinstance(default, bool):
                values[field_name] = bool(raw)
            elif isinstance(default, int):
                values[field_name] = int(raw)
            elif isinstance(default, float):
                values[field_name] = float(raw)
            else:
                values[field_name] = str(raw)
        values.update({
            "lane_count": lane_count,
            "experimental_only": bool(params.get("experimental_only", lane_count == 9)),
        })
        values["provider"] = str(params.get("provider", values.get("provider", "auto"))).lower()
        config = cls(**values)
        if config.lane_count not in (0, *SUPPORTED_LANE_COUNTS):
            raise ValueError("lane_count must be auto, 7, or 9")
        if not 5 <= config.max_duration_seconds <= 600:
            raise ValueError("max_duration_seconds must be between 5 and 600")
        if not 0 <= config.sample_interval_ms <= 100:
            raise ValueError("sample_interval_ms must be between 0 and 100")
        if not config.deadline_execution_window_ms <= config.hold_start_execution_window_ms <= 100:
            raise ValueError("hold_start_execution_window_ms is invalid")
        if not config.deadline_execution_window_ms <= config.tap_execution_window_ms <= 120:
            raise ValueError("tap_execution_window_ms is invalid")
        if not config.deadline_execution_window_ms <= config.special_tap_execution_window_ms <= 120:
            raise ValueError("special_tap_execution_window_ms is invalid")
        if not config.deadline_execution_window_ms <= config.dense_tap_execution_window_ms <= 120:
            raise ValueError("dense_tap_execution_window_ms is invalid")
        if not 100 <= config.dense_tap_neighbour_ms <= 1500:
            raise ValueError("dense_tap_neighbour_ms is invalid")
        if not 0 <= config.tap_action_advance_ms <= 250:
            raise ValueError("tap_action_advance_ms is invalid")
        if not 0 <= config.hold_start_action_advance_ms <= 250:
            raise ValueError("hold_start_action_advance_ms is invalid")
        if not 0 <= config.dense_tap_action_advance_ms <= config.tap_action_advance_ms:
            raise ValueError("dense_tap_action_advance_ms is invalid")
        if not 100 <= config.dense_tap_cluster_horizon_ms <= 800:
            raise ValueError("dense_tap_cluster_horizon_ms is invalid")
        if not 3 <= config.dense_tap_min_tracks <= 6:
            raise ValueError("dense_tap_min_tracks is invalid")
        if not 100 <= config.dense_tap_pair_horizon_ms <= config.dense_tap_cluster_horizon_ms:
            raise ValueError("dense_tap_pair_horizon_ms is invalid")
        if not config.dense_tap_pair_horizon_ms <= config.dense_same_lane_pair_horizon_ms <= 1000:
            raise ValueError("dense_same_lane_pair_horizon_ms is invalid")
        if not 100 <= config.tap_cadence_min_ms < config.tap_cadence_max_ms <= 1200:
            raise ValueError("tap cadence interval is invalid")
        if not 0.10 <= config.tap_cadence_tolerance_ratio <= 0.50:
            raise ValueError("tap_cadence_tolerance_ratio is invalid")
        if not 50 <= config.tap_cadence_max_correction_ms <= 500:
            raise ValueError("tap_cadence_max_correction_ms is invalid")
        if not 20 <= config.dense_same_lane_min_gap_ms <= 100:
            raise ValueError("dense_same_lane_min_gap_ms is invalid")
        if not 0 <= config.center_tap_action_advance_ms <= config.tap_action_advance_ms:
            raise ValueError("center_tap_action_advance_ms is invalid")
        if not 100 <= config.pause_check_interval_ms <= 1000:
            raise ValueError("pause_check_interval_ms is invalid")
        if not 500 <= config.end_check_interval_ms <= 5000:
            raise ValueError("end_check_interval_ms is invalid")
        if not 5000 <= config.terminal_min_runtime_ms <= 120000:
            raise ValueError("terminal_min_runtime_ms is invalid")
        if not 750 <= config.terminal_quiet_ms <= 5000:
            raise ValueError("terminal_quiet_ms is invalid")
        if not 250 <= config.capture_timeout_ms <= 5000:
            raise ValueError("capture_timeout_ms is invalid")
        if config.provider not in {"auto", "maa", "numpy"}:
            raise ValueError("provider must be auto, maa, or numpy")
        if not 0.0 < config.hold_head_color_ratio <= 1.0:
            raise ValueError("hold_head_color_ratio must be between 0 and 1")
        if not 0.0 < config.hold_tail_min_score <= 1.0:
            raise ValueError("hold_tail_min_score must be between 0 and 1")
        if config.hold_tail_min_frames < 2:
            raise ValueError("hold_tail_min_frames must be at least 2")
        if not 0.25 <= config.hold_tail_frame_min_motion_px <= 4.0:
            raise ValueError("hold_tail_frame_min_motion_px is invalid")
        if not 0.25 <= config.hold_target_min_progress <= 0.7:
            raise ValueError("hold_target_min_progress is invalid")
        if not 3 <= config.hold_target_confirm_samples <= 6:
            raise ValueError("hold_target_confirm_samples is invalid")
        if not 1 <= config.hold_max_route_hops <= 4:
            raise ValueError("hold_max_route_hops is invalid")
        if not 2 <= config.hold_fold_target_confirm_frames <= 5:
            raise ValueError("hold_fold_target_confirm_frames is invalid")
        if not 0.4 <= config.hold_tail_loss_min_progress < config.hold_cap_lock_progress:
            raise ValueError("hold_tail_loss_min_progress is invalid")
        if not 2 <= config.hold_tail_loss_confirm_frames <= 6:
            raise ValueError("hold_tail_loss_confirm_frames is invalid")
        if not 0 <= config.hold_tail_loss_release_delay_ms <= 250:
            raise ValueError("hold_tail_loss_release_delay_ms is invalid")
        if not config.hold_fallback_duration_ms <= config.hold_tail_loss_max_age_ms <= config.hold_unverified_max_tail_ms:
            raise ValueError("hold_tail_loss_max_age_ms is invalid")
        if config.hold_long_verify_samples < 4:
            raise ValueError("hold_long_verify_samples must be at least 4")
        if not 0.001 <= config.hold_long_verify_progress_span <= config.hold_tail_min_motion:
            raise ValueError("hold_long_verify_progress_span is invalid")
        if not config.hold_unverified_max_tail_ms < config.hold_long_min_predicted_duration_ms <= config.hold_max_tail_ms:
            raise ValueError("hold_long_min_predicted_duration_ms is invalid")
        if not 300 <= config.hold_long_reacquire_gap_ms <= 3000:
            raise ValueError("hold_long_reacquire_gap_ms is invalid")
        if not config.hold_long_duration_ms <= config.hold_long_reacquire_after_ms <= config.hold_unverified_max_tail_ms:
            raise ValueError("hold_long_reacquire_after_ms is invalid")
        if not 0.5 <= config.hold_long_reacquire_min_progress <= config.hold_cap_lock_progress:
            raise ValueError("hold_long_reacquire_min_progress is invalid")
        if not 100 <= config.hold_linked_release_extension_ms <= config.hold_fallback_duration_ms:
            raise ValueError("hold_linked_release_extension_ms is invalid")
        if not 300 <= config.hold_checkpoint_extension_ms <= config.hold_unverified_max_tail_ms:
            raise ValueError("hold_checkpoint_extension_ms is invalid")
        if not 50 <= config.hold_linked_sync_tolerance_ms <= config.hold_linked_release_extension_ms:
            raise ValueError("hold_linked_sync_tolerance_ms is invalid")
        if not 0.35 <= config.urgent_tap_min_progress <= 0.80:
            raise ValueError("urgent_tap_min_progress is invalid")
        if not 0.35 <= config.center_urgent_tap_min_progress <= config.urgent_tap_min_progress:
            raise ValueError("center_urgent_tap_min_progress is invalid")
        if not 50 <= config.urgent_tap_deadline_horizon_ms <= config.max_schedule_horizon_ms:
            raise ValueError("urgent_tap_deadline_horizon_ms is invalid")
        if not config.tap_execution_window_ms <= config.urgent_tap_late_horizon_ms <= 500:
            raise ValueError("urgent_tap_late_horizon_ms is invalid")
        if not 0.35 <= config.urgent_hold_start_min_progress <= 0.80:
            raise ValueError("urgent_hold_start_min_progress is invalid")
        if not 50 <= config.urgent_hold_start_deadline_horizon_ms <= config.max_schedule_horizon_ms:
            raise ValueError("urgent_hold_start_deadline_horizon_ms is invalid")
        if not config.hold_start_execution_window_ms <= config.urgent_hold_start_late_horizon_ms <= 500:
            raise ValueError("urgent_hold_start_late_horizon_ms is invalid")
        if not 0 <= config.hold_head_association_grace_ms <= 300:
            raise ValueError("hold_head_association_grace_ms is invalid")
        if config.max_contact_ms <= config.hold_max_tail_ms:
            raise ValueError("max_contact_ms must exceed hold_max_tail_ms")
        if not config.hold_fallback_duration_ms <= config.hold_unverified_max_tail_ms <= config.hold_max_tail_ms:
            raise ValueError("hold_unverified_max_tail_ms is invalid")
        if config.hold_min_duration_ms < 0 or config.hold_max_tail_ms <= config.hold_min_duration_ms:
            raise ValueError("hold duration limits are invalid")
        if not config.hold_min_duration_ms <= config.hold_fallback_duration_ms <= config.hold_max_tail_ms:
            raise ValueError("hold_fallback_duration_ms is invalid")
        if not 100 <= config.hold_ribbon_extension_ms <= config.hold_fallback_duration_ms:
            raise ValueError("hold_ribbon_extension_ms is invalid")
        if not 0 <= config.hold_move_advance_ms <= config.hold_fallback_duration_ms:
            raise ValueError("hold_move_advance_ms is invalid")
        if not 0 <= config.hold_cap_acquire_delay_ms <= config.hold_fallback_duration_ms:
            raise ValueError("hold_cap_acquire_delay_ms is invalid")
        if not 0.7 <= config.hold_cap_lock_progress <= 1.0:
            raise ValueError("hold_cap_lock_progress is invalid")
        if not 0 <= config.hold_same_lane_guard_ms <= 1000:
            raise ValueError("hold_same_lane_guard_ms is invalid")
        if config.hold_long_duration_ms < config.hold_fallback_duration_ms:
            raise ValueError("hold_long_duration_ms is invalid")
        if not config.deadline_execution_window_ms <= config.hold_release_execution_window_ms <= 150:
            raise ValueError("hold_release_execution_window_ms is invalid")
        if not config.deadline_execution_window_ms <= config.hold_fallback_release_execution_window_ms <= config.hold_release_execution_window_ms:
            raise ValueError("hold_fallback_release_execution_window_ms is invalid")
        if not config.deadline_execution_window_ms <= config.hold_move_execution_window_ms <= 150:
            raise ValueError("hold_move_execution_window_ms is invalid")
        if not 10 <= config.pre_capture_deadline_guard_ms <= 100:
            raise ValueError("pre_capture_deadline_guard_ms is invalid")
        return config


@dataclass(frozen=True)
class MusicFrame:
    sequence: int
    capture_started: float
    capture_finished: float
    midpoint: float
    image: Any


@dataclass(frozen=True)
class MusicCandidate:
    box: tuple[int, int, int, int]
    pixel_count: int
    fill_ratio: float
    center: tuple[float, float]
    variant: str = ""
    # Direction resolved by the dedicated flick channel's white-arrow
    # classifier.  Only ``variant == "flick"`` candidates carry a real value.
    flick_direction: NoteGesture = NoteGesture.UNKNOWN
    # Colour family of the sprite (blue/violet/pink/red) for the user's
    # colour table and for conflict diagnostics.
    flick_color: str = ""


@dataclass(frozen=True)
class TrackObservation:
    frame_sequence: int
    timestamp: float
    center: tuple[float, float]
    progress: float
    candidate: MusicCandidate


@dataclass(frozen=True)
class HoldTailObservation:
    frame_sequence: int
    timestamp: float
    progress: float
    score: float
    pixel_count: int
    lane: int
    center: tuple[float, float]
    ribbon_exit_count: int = 1


@dataclass
class NoteTrack:
    track_id: int
    lane: int
    # ``observations`` is a bounded deque and therefore cannot reveal that a
    # stage highlight has been tracked for tens of seconds.  Preserve the
    # original birth time so an unexecuted head cannot become schedulable late
    # in the song after its oldest samples have rolled out of the deque.
    first_seen_time: float | None = None
    observations: Deque[TrackObservation] = field(default_factory=lambda: deque(maxlen=12))
    speed: float = 0.0
    predicted_hit_time: float | None = None
    gesture: NoteGesture = NoteGesture.TAP
    state: TrackState = TrackState.APPROACHING
    direction_evidence: Deque[NoteGesture] = field(default_factory=lambda: deque(maxlen=2))
    hold_evidence_frames: int = 0
    hold_tail_observations: Deque[HoldTailObservation] = field(default_factory=lambda: deque(maxlen=12))
    hold_target_lane: int | None = None
    hold_release_time: float | None = None
    hold_release_locked: bool = False
    hold_long_verified: bool = False
    hold_tail_reacquired: bool = False
    hold_terminal_confirmed: bool = False
    hold_terminal_conflicts: int = 0
    hold_last_continuity_time: float | None = None
    hold_release_scheduled: bool = False
    hold_move_scheduled: bool = False
    hold_route_steps_completed: int = 0
    # Folded-route history: [start_lane, hop1, hop2, ...] and the index of the
    # active segment (source = route_lanes[segment_index]).
    hold_route_lanes: list[int] = field(default_factory=list)
    hold_segment_index: int = 0
    hold_fold_target_frames: int = 0
    hold_fold_route_confirmed: bool = False
    hold_fold_move_scheduled: bool = False
    hold_tail_loss_frames: int = 0
    bonus_star: bool = False
    # Born from a colour-classified flick sprite; carries its direction.
    flick: bool = False
    flick_direction: NoteGesture = NoteGesture.UNKNOWN
    flick_color: str = ""
    # A flick track bound as a hold's ribbon-tip terminal: it keeps its motion
    # prediction but never emits a standalone swipe.
    hold_end_owner: int | None = None
    # Hold side of the binding: the terminal flick track and its direction.
    hold_end_flick_track: int | None = None
    hold_end_flick_direction: NoteGesture = NoteGesture.UNKNOWN
    dense_tap: bool = False
    tap_cadence_hit_time: float | None = None
    # Compatibility only: TapTimingPolicy no longer consumes cadence overrides.
    linked_partner_id: int | None = None
    linked_evidence_frames: int = 0
    linked_release_source_id: int | None = None
    tail_missing_frames: int = 0
    missed_frames: int = 0
    action_event_id: str = ""
    action_executed: bool = False
    # Hold-end flick: the bound ribbon-tip track and its live arrival.
    hold_end_flick_arrival: float | None = None
    # Tap-only physical-input acknowledgement. action_executed remains the
    # legacy scheduled flag used by hold/flick code.
    tap_input_started: float | None = None
    tap_input_completed: float | None = None
    tap_executed_hit_time: float | None = None


@dataclass
class LaneInputState:
    lane: int
    contact: int | None = None
    hold_track_id: int | None = None
    cooldown_until: float = 0.0
    contact_started: float = 0.0


@dataclass(frozen=True)
class MusicActionEvent:
    event_id: str
    track_id: int
    lane: int
    gesture: NoteGesture
    deadline: float
    coordinate: tuple[int, int]
    direction: NoteGesture = NoteGesture.UNKNOWN
    contact_policy: str = "auto"
    source_capture_started: float = 0.0
    source_capture_finished: float = 0.0
    tap_group_id: str | None = None
    tap_frozen: bool = False
    tap_reference_hit_time: float | None = None


@dataclass(frozen=True)
class FlickRequest:
    lane: int
    x: int
    y: int
    direction: NoteGesture
    already_down: bool = False


@dataclass
class MusicCalibrationData:
    version: int
    lane_count: int
    width: int
    height: int
    points: list[list[int]]
    lane_centerlines: list[list[list[float]]]
    corridor_widths: list[float]
    trigger_progress: float
    candidate_roi: list[int]
    exclusion_rois: list[list[int]]
    baseline_version: str
    action_advance_ms: float
    color_lower: list[list[int]]
    color_upper: list[list[int]]
    candidate_min_pixels: int
    hold_min_length: float
    created_at: str

    @property
    def profile_key(self) -> str:
        return f"{self.lane_count}@{self.width}x{self.height}"

    @property
    def scan_rois(self) -> list[list[int]]:
        return [[x - 3, y - 3, 7, 7] for x, y in self.points]

    @property
    def flick_rois(self) -> list[list[int]]:
        width = max(24, int(min(self.corridor_widths) * 1.4))
        return [[x - width // 2, y - 54, width, 48] for x, y in self.points]

    @property
    def approach_rois(self) -> list[list[int]]:
        width = max(24, int(min(self.corridor_widths) * 1.3))
        return [[x - width // 2, max(0, y - 220), width, min(180, y)] for x, y in self.points]

    @property
    def tail_rois(self) -> list[list[int]]:
        width = max(18, int(min(self.corridor_widths)))
        return [[x - width // 2, max(0, y - 150), width, min(120, y)] for x, y in self.points]


@dataclass
class MusicRunResult:
    status: str
    failure_code: MusicFailureCode = MusicFailureCode.NONE
    reason: str = ""
    provider: str = ""
    input_mode: str = "compatibility"
    controller_signature: str = ""
    profile: str = ""
    task_id: int | None = None
    metrics_ms: dict[str, dict[str, float | int]] = field(default_factory=dict)
    schema_version: int = 2
    time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failure_code"] = self.failure_code.value
        return value
