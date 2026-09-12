from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agent.common import LOGGER, capture_image, is_stopping, recognition_hit, recognition_results

from .calibration import image_size, load_calibration
from .executor import MusicActionExecutor, MusicTouchError
from .hold_policy import HoldTimingPolicy
from .models import (
    BASE_HEIGHT,
    BASE_WIDTH,
    FLICK_GESTURES,
    FlickRequest,
    MusicActionEvent,
    MusicCalibrationData,
    MusicConfig,
    MusicFailureCode,
    MusicFrame,
    MusicRunResult,
    NoteGesture,
    TrackState,
)
from .storage import (
    atomic_write_json,
    controller_signature,
    metric_summary,
    percentile_95,
    preflight_path,
    read_json,
    write_result,
)
from .tap_policy import TapTimingPolicy
from .tap_dispatch import due_tap_batches
from .tap_trace import TapTrace, VERSION
from .tracking import MusicVisionEngine
from .vision import MaaCandidateProvider, NumpyCandidateProvider, VisualMask, giant_live_title_present


@dataclass
class RuntimeMetrics:
    capture: list[float] = field(default_factory=list)
    provider: list[float] = field(default_factory=list)
    perception_to_action: list[float] = field(default_factory=list)
    capture_to_action: list[float] = field(default_factory=list)
    terminal: list[float] = field(default_factory=list)
    loop: list[float] = field(default_factory=list)
    tracks_created: int = 0
    tracks_retained_peak: int = 0
    tracks_expired: int = 0
    tracks_pruned: int = 0
    urgent_rescues: int = 0
    events_scheduled: int = 0
    isolated_same_lane_heads: int = 0
    unscheduled_head_losses: int = 0

    def summaries(self, action: list[float]) -> dict[str, dict[str, float | int]]:
        return {
            "capture": metric_summary(self.capture[-60:]),
            "provider": metric_summary(self.provider[-60:]),
            "perception_to_action": metric_summary(self.perception_to_action[-60:]),
            "capture_to_action": metric_summary(self.capture_to_action[-60:]),
            "action": metric_summary(action[-60:]),
            "terminal": metric_summary(self.terminal[-60:]),
            "loop": metric_summary(self.loop[-60:]),
            "tracking": {
                "tracks_created": self.tracks_created,
                "tracks_retained_peak": self.tracks_retained_peak,
                "tracks_expired": self.tracks_expired,
                "tracks_pruned": self.tracks_pruned,
                "urgent_rescues": self.urgent_rescues,
                "events_scheduled": self.events_scheduled,
                "isolated_same_lane_heads": self.isolated_same_lane_heads,
                "unscheduled_head_losses": self.unscheduled_head_losses,
            },
        }


def _task_id(argv: Any) -> int | None:
    detail = getattr(argv, "task_detail", None)
    value = getattr(detail, "task_id", None)
    return int(value) if value is not None else None


def _capture_frame(
    context: Any,
    sequence: int,
    clock: Callable[[], float],
    timeout_ms: int = 1200,
) -> tuple[MusicFrame | None, float]:
    started = clock()
    image = capture_image(context, timeout_ms=timeout_ms)
    finished = clock()
    elapsed_ms = (finished - started) * 1000.0
    if image is None:
        return None, elapsed_ms
    return MusicFrame(
        sequence=sequence,
        capture_started=started,
        capture_finished=finished,
        midpoint=(started + finished) / 2.0,
        image=image,
    ), elapsed_ms


def _recognize(context: Any, node: str, image: Any) -> bool:
    detail = context.run_recognition(node, image)
    if detail is None:
        raise RuntimeError(f"Recognition {node} did not start")
    return recognition_hit(detail)


def _live_screen(context: Any, image: Any) -> bool:
    return _recognize(context, "MusicLiveScreen", image) or _recognize(context, "MusicLiveClearScreen", image)


def terminal_state(context: Any, image: Any) -> str:
    # A normal automatic finish is deliberately restricted to the two visual
    # transitions requested for temporary play.  Heartbeat loss, score text,
    # reward panels and the stop dialog are not completion evidence.
    if _recognize(context, "MusicResultLoading", image) or _recognize(context, "MusicResultLive", image):
        return "result"
    return "unknown"


def _resolve_calibration(context: Any, config: MusicConfig, image: Any) -> MusicCalibrationData:
    detail = context.run_recognition("MusicLaneTargets", image)
    centers: set[tuple[int, int]] = set()
    for item in recognition_results(detail):
        box = getattr(item, "box", None)
        if box is None:
            continue
        values = list(box)
        if len(values) == 4:
            centers.add((int(values[0]) + int(values[2]) // 2, int(values[1]) + int(values[3]) // 2))
    lane_count = len(centers)
    expected = config.lane_count if config.lane_count in (7, 9) else 0
    if expected and lane_count == expected:
        return load_calibration(expected, image)
    if expected and lane_count >= 4:
        LOGGER.warning(
            "Lane target recognition found %s targets but configuration expects %s lanes; "
            "proceeding with the configured calibration",
            lane_count,
            expected,
        )
        return load_calibration(expected, image)
    if not expected and lane_count in (7, 9):
        return load_calibration(lane_count, image)
    if lane_count in (7, 9):
        raise ValueError(f"Configured for {config.lane_count} lanes but the current live screen has {lane_count}")
    raise ValueError(f"Unable to confirm a 7- or 9-lane screen; recognized {lane_count} targets")


class MusicRuntime:
    def __init__(
        self,
        context: Any,
        config: MusicConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.context = context
        self.config = config
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.metrics = RuntimeMetrics()
        self.calibration: MusicCalibrationData | None = None
        self.provider: MaaCandidateProvider | NumpyCandidateProvider | None = None
        self.signature = ""
        self.input_mode = "compatibility"
        self.action_durations: list[float] = []
        # Kept in memory during play and emitted once after input cleanup.  This
        # gives a stable chart-head ordinal without adding per-note log I/O to
        # the real-time loop.
        self.head_action_trace: list[str] = []
        self.cleanup_failure = ""
        self.tap_policy = TapTimingPolicy(config, config.lane_count or 7)
        self.hold_policy = HoldTimingPolicy(config)
        self.tap_trace = TapTrace(config)

    def _failure(self, code: MusicFailureCode, reason: str, task_id: int | None = None) -> MusicRunResult:
        return MusicRunResult(
            status="cancelled" if code == MusicFailureCode.CANCELLED else "failed",
            failure_code=code,
            reason=reason,
            provider=getattr(self.provider, "name", ""),
            input_mode=self.input_mode,
            controller_signature=self.signature,
            profile=self.calibration.profile_key if self.calibration else "",
            task_id=task_id,
        )

    def _record_head_action(
        self,
        event: MusicActionEvent,
        action_started: float,
        engine: MusicVisionEngine | None,
    ) -> None:
        track = engine.tracks.get(event.track_id) if engine is not None else None
        flags = ""
        if track is not None:
            flags = "".join(("D" if track.dense_tap else "", "B" if track.bonus_star else "", "L" if track.linked_partner_id is not None else ""))
        self.head_action_trace.append(
            f"{len(self.head_action_trace) + 1}:{event.track_id}:{event.lane}:{event.gesture.value}:"
            f"{action_started:.3f}:{(action_started - event.deadline) * 1000.0:.1f}:{flags or '-'}"
        )

    def _acknowledge_taps(self, receipts, events, engine, metrics):
        by_id = {event.event_id: event for event in events}
        for receipt in receipts:
            event = by_id.get(receipt.event_id)
            if event is None:
                continue
            self.tap_trace.add('input', event=event.event_id, group=event.tap_group_id,
                               deadline=event.deadline, receipt=asdict(receipt))
            track = engine.tracks.get(event.track_id) if engine else None
            if track is not None and event.gesture == NoteGesture.TAP and receipt.down_call_started is not None:
                track.tap_input_started = receipt.down_call_started
                track.tap_input_completed = receipt.up_call_finished
                track.tap_executed_hit_time = (event.tap_reference_hit_time if event.tap_reference_hit_time is not None
                                               else event.deadline + self.tap_policy.action_advance_ms(track) / 1000.)
            if receipt.down_call_finished is not None and receipt.down_call_started is not None:
                self._record_head_action(event, receipt.down_call_started, engine)
                metrics.perception_to_action.append(max(0., (receipt.down_call_started - event.source_capture_started) * 1000.))
                metrics.capture_to_action.append(max(0., (receipt.down_call_started - event.source_capture_finished) * 1000.))

    def _flush_due_taps(
        self,
        executor: MusicActionExecutor,
        pending: list[MusicActionEvent],
        engine: MusicVisionEngine | None,
        metrics: RuntimeMetrics,
    ) -> None:
        """Dispatch input that comes due while a blocking flick gesture runs.

        The stepped swipe spends real time between its motion events; without
        this hook a tap or hold due in that window would fire tens of
        milliseconds late (recorded as Bad/Miss).  Taps are released with the
        usual chord rules; due hold starts/routes are dispatched through the
        same owner-guarded path as the main loop.  Flicks intentionally stay
        in ``pending`` so gestures never nest.
        """
        now = self.clock()
        tap_batches = due_tap_batches(pending, now, engine)
        fallback_holds = [
            event
            for event in pending
            if event.deadline <= now
            and event.gesture == NoteGesture.HOLD_START
            and not executor.supports_holds
        ]
        if tap_batches or fallback_holds:
            dispatched = [event for batch in tap_batches for event in batch] + fallback_holds
            dispatched_ids = {event.event_id for event in dispatched}
            pending[:] = [event for event in pending if event.event_id not in dispatched_ids]
            late = [
                event
                for event in dispatched
                if (now - event.deadline) * 1000.0 > self.config.event_late_tolerance_ms
            ]
            if late:
                LOGGER.warning(
                    "Music tap deadlines were missed during a flick: %s",
                    [(event.track_id, round((now - event.deadline) * 1000.0, 1)) for event in late],
                )
            for batch in [*tap_batches, ([fallback_holds] if fallback_holds else [])]:
                if not batch:
                    continue
                try:
                    receipts = executor.tap_many(
                        [(event.lane, *event.coordinate) for event in batch],
                        event_ids=[event.event_id for event in batch],
                    )
                except MusicTouchError as error:
                    self._acknowledge_taps(error.receipts, batch, engine, metrics)
                    raise
                self._acknowledge_taps(receipts or [], batch, engine, metrics)
            if engine is not None and not executor.supports_holds:
                for event in fallback_holds:
                    track = engine.tracks.get(event.track_id)
                    if track is not None:
                        track.state = TrackState.RELEASED
        if not executor.may_open_contact_during_gesture():
            return
        due_holds = [
            event
            for event in pending
            if event.deadline <= now and event.gesture not in FLICK_GESTURES
        ]
        if not due_holds:
            return
        due_ids = {event.event_id for event in due_holds}
        pending[:] = [event for event in pending if event.event_id not in due_ids]
        suppressed: set[int] = set()
        for event in due_holds:
            self._dispatch_due_event(event, executor, engine, metrics, pending, suppressed)

    def startup_gate(self, first_frame: MusicFrame, task_id: int | None = None) -> MusicRunResult | None:
        try:
            if image_size(first_frame.image) != (BASE_WIDTH, BASE_HEIGHT):
                return self._failure(
                    MusicFailureCode.INVALID_CALIBRATION,
                    f"Framework screenshot must be 1280x720, got {image_size(first_frame.image)[0]}x{image_size(first_frame.image)[1]}",
                    task_id,
                )
            if _recognize(self.context, "MusicPauseDialog", first_frame.image):
                return self._failure(
                    MusicFailureCode.PAUSED_BEFORE_START,
                    "请先点击继续演唱会；暂停页启动不会处理轨道或发送触控",
                    task_id,
                )
            if not _live_screen(self.context, first_frame.image):
                return self._failure(
                    MusicFailureCode.INVALID_LIVE_SCREEN,
                    "当前画面不是可靠的演唱会进行页",
                    task_id,
                )
            self.calibration = _resolve_calibration(self.context, self.config, first_frame.image)
            self.signature = controller_signature(self.context, first_frame.image)
            return None
        except (FileNotFoundError, ValueError) as error:
            return self._failure(MusicFailureCode.INVALID_CALIBRATION, str(error), task_id)
        except Exception as error:
            return self._failure(MusicFailureCode.INTERNAL_ERROR, f"Startup gate failed: {error}", task_id)

    def _candidate_provider(self, name: str) -> MaaCandidateProvider | NumpyCandidateProvider:
        if self.calibration is None:
            raise RuntimeError("Calibration is not loaded")
        if name == "maa":
            return MaaCandidateProvider(self.context, self.calibration, self.config.candidate_iou_threshold, self.config.candidate_min_size)
        return NumpyCandidateProvider(self.calibration, self.config.candidate_iou_threshold, self.config.candidate_min_size)

    def provider_gate(self, first_frame: MusicFrame, task_id: int | None = None) -> MusicRunResult | None:
        if self.calibration is None:
            return self._failure(MusicFailureCode.INVALID_CALIBRATION, "Calibration was not loaded", task_id)
        capture_samples: list[float] = [max(0.0, (first_frame.capture_finished - first_frame.capture_started) * 1000.0)]
        frames: list[MusicFrame] = [first_frame]
        total = self.config.preflight_warmup_samples + self.config.preflight_samples
        try:
            for sequence in range(1, total):
                frame, elapsed = _capture_frame(
                    self.context,
                    sequence,
                    self.clock,
                    self.config.capture_timeout_ms,
                )
                if frame is None:
                    raise RuntimeError("Screenshot failed during provider preflight")
                capture_samples.append(elapsed)
                frames.append(frame)
            measured_capture = capture_samples[self.config.preflight_warmup_samples :]
            if percentile_95(measured_capture) > self.config.max_capture_p95_ms:
                return self._failure(
                    MusicFailureCode.PERFORMANCE_REJECTED,
                    f"Screenshot P95 {percentile_95(measured_capture):.1f} ms exceeds {self.config.max_capture_p95_ms:.1f} ms",
                    task_id,
                )
            names = [self.config.provider] if self.config.provider != "auto" else ["maa", "numpy"]
            selected = None
            provider_samples: list[float] = []
            for name in names:
                provider = self._candidate_provider(name)
                readings: list[float] = []
                try:
                    for index, frame in enumerate(frames):
                        visual = VisualMask.from_image(frame.image, self.calibration)
                        started = self.clock()
                        provider.detect(frame, visual)
                        elapsed = (self.clock() - started) * 1000.0
                        if index >= self.config.preflight_warmup_samples:
                            readings.append(elapsed)
                        if len(readings) >= 5 and percentile_95(readings[-5:]) > self.config.max_provider_p95_ms:
                            break
                except Exception:
                    LOGGER.exception("Music candidate provider %s failed preflight", name)
                    continue
                if percentile_95(readings) <= self.config.max_provider_p95_ms:
                    selected, provider_samples = provider, readings
                    break
            if selected is None:
                return self._failure(
                    MusicFailureCode.PERFORMANCE_REJECTED,
                    "Neither the MAA nor NumPy candidate Provider met the preflight quality/performance gate",
                    task_id,
                )
            estimated_loop_p95 = percentile_95(measured_capture) + percentile_95(provider_samples)
            if estimated_loop_p95 > self.config.max_loop_p95_ms:
                return self._failure(
                    MusicFailureCode.PERFORMANCE_REJECTED,
                    f"Estimated capture and candidate loop P95 {estimated_loop_p95:.1f} ms exceeds {self.config.max_loop_p95_ms:.1f} ms",
                    task_id,
                )
            self.provider = selected
            self.metrics.capture.extend(measured_capture)
            self.metrics.provider.extend(provider_samples)
            atomic_write_json(preflight_path(), {
                "schema_version": 1,
                "provider": selected.name,
                "controller_signature": self.signature,
                "profile": self.calibration.profile_key,
                "capture_ms": metric_summary(measured_capture),
                "provider_ms": metric_summary(provider_samples),
                "estimated_loop_p95_ms": round(estimated_loop_p95, 3),
                "samples": len(provider_samples),
                "warmup_samples": self.config.preflight_warmup_samples,
                "time_monotonic": self.monotonic(),
            })
            return None
        except Exception as error:
            return self._failure(MusicFailureCode.CANDIDATE_FAILURE, f"Provider preflight failed: {error}", task_id)

    def reuse_provider_gate(self) -> bool:
        if self.calibration is None:
            return False
        state = read_json(preflight_path())
        if state.get("controller_signature") != self.signature or state.get("profile") != self.calibration.profile_key:
            return False
        provider_name = str(state.get("provider", ""))
        if provider_name not in {"maa", "numpy"}:
            return False
        completed = float(state.get("time_monotonic", 0.0))
        age = self.monotonic() - completed
        if completed <= 0 or age < 0 or age > 30.0:
            return False
        self.provider = self._candidate_provider(provider_name)
        return True

    def activate_play_provider(self) -> None:
        """Select a provider without consuming live-song frames in a benchmark.

        The standalone preflight used to capture and analyze up to 35 frames
        before the tracker existed.  When a song had already started those
        frames contained the opening notes, so the benchmark guaranteed early
        misses.  Runtime metrics still monitor the selected provider; they no
        longer block the start of a seven-lane song.
        """
        if self.reuse_provider_gate():
            return
        provider_name = self.config.provider if self.config.provider in {"maa", "numpy"} else "numpy"
        self.provider = self._candidate_provider(provider_name)
        LOGGER.info("Music runtime activated %s provider without a blocking frame benchmark", provider_name)

    def preflight(self, first_frame: MusicFrame, task_id: int | None = None) -> MusicRunResult:
        failed = self.startup_gate(first_frame, task_id) or self.provider_gate(first_frame, task_id)
        if failed:
            failed.metrics_ms = self.metrics.summaries([])
            return failed
        mode = "preflight-only" if self.config.lane_count == 9 or self.config.experimental_only else "compatibility"
        return MusicRunResult(
            status="succeeded",
            reason="Music V4 preflight passed",
            provider=getattr(self.provider, "name", ""),
            input_mode=mode,
            controller_signature=self.signature,
            profile=self.calibration.profile_key if self.calibration else "",
            task_id=task_id,
            metrics_ms=self.metrics.summaries([]),
        )

    def _create_executor(self) -> MusicActionExecutor:
        if self.calibration is None:
            raise RuntimeError("Calibration is not loaded")
        advanced = True
        multi_touch = True
        self.input_mode = "advanced"
        return MusicActionExecutor(
            self.context,
            self.calibration.width,
            self.calibration.height,
            self.config,
            controller_signature=self.signature,
            advanced=advanced,
            multi_touch=multi_touch,
            sleeper=self.sleeper,
            clock=self.clock,
        )

    def _poll_async_inputs(
        self,
        executor: MusicActionExecutor,
        engine: MusicVisionEngine | None,
        metrics: RuntimeMetrics,
    ) -> None:
        """Resolve asynchronously posted gestures without blocking the loop."""
        if not (executor.async_input or executor.async_flicks):
            return
        receipts = executor.poll_inputs()
        if not receipts:
            return
        events = []
        for receipt in receipts:
            event = self._in_flight_taps.pop(receipt.event_id, None)
            if event is not None:
                events.append(event)
        if events:
            self._acknowledge_taps(receipts, events, engine, metrics)
        failed = [receipt for receipt in receipts if receipt.error]
        if failed:
            raise MusicTouchError(f"Async music input failed: {failed[0].error}", receipts=receipts)

    def _wait_until_resumed(
        self,
        executor: MusicActionExecutor,
        sequence: int,
        task_id: int | None,
    ) -> tuple[MusicFrame | None, int, float, MusicRunResult | None]:
        """Release input and wait for a paused song to return to a live screen."""
        pause_started = self.monotonic()
        executor.release_all()
        LOGGER.info("Music pause detected; contacts released and tracking suspended until resume")
        capture_failures = 0
        live_confirmations = 0
        while True:
            if is_stopping(self.context):
                return (
                    None,
                    sequence,
                    self.monotonic() - pause_started,
                    self._failure(MusicFailureCode.CANCELLED, "User cancelled music play while paused", task_id),
                )
            frame, capture_ms = _capture_frame(
                self.context,
                sequence,
                self.clock,
                self.config.capture_timeout_ms,
            )
            sequence += 1
            self.metrics.capture.append(capture_ms)
            if frame is None:
                capture_failures += 1
                if capture_failures >= self.config.max_capture_failures:
                    return (
                        None,
                        sequence,
                        self.monotonic() - pause_started,
                        self._failure(MusicFailureCode.CAPTURE_FAILURE, f"{self.config.max_capture_failures} consecutive paused-screen screenshots failed", task_id),
                    )
            else:
                capture_failures = 0
                if _recognize(self.context, "MusicPauseDialog", frame.image):
                    live_confirmations = 0
                elif _live_screen(self.context, frame.image):
                    live_confirmations += 1
                    if live_confirmations >= 2:
                        paused_seconds = self.monotonic() - pause_started
                        LOGGER.info("Music live screen returned after %.2f paused seconds; rebuilding note tracks", paused_seconds)
                        return frame, sequence, paused_seconds, None
                else:
                    live_confirmations = 0
            self.sleeper(max(0.05, self.config.pause_check_interval_ms / 1000.0))

    def _record_hold_call(self, call, event, engine):
        started = self.clock()
        error = ''
        try:
            return call()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if engine is not None:
                engine.tap_trace.add('hold_input', event=event.event_id, track=event.track_id,
                                     lane=event.lane, gesture=event.gesture.value,
                                     coordinate=event.coordinate, deadline=event.deadline,
                                     call_started=started, call_finished=self.clock(), error=error)

    def _execute_due(
        self,
        executor: MusicActionExecutor,
        pending: list[MusicActionEvent],
        now: float,
        metrics: RuntimeMetrics,
        engine: MusicVisionEngine | None = None,
    ) -> None:
        # Candidate/hold-route refinement may have consumed tens of milliseconds
        # after the caller sampled ``now``.  Never precision-sleep from that stale
        # timestamp: doing so can turn an otherwise on-time isolated tap into the
        # single late action of a song.
        now = max(now, self.clock())
        base_window = self.config.deadline_execution_window_ms / 1000.0
        lookahead = base_window
        for event in pending:
            event_window = base_window
            track = engine.tracks.get(event.track_id) if engine is not None else None
            if event.gesture in {NoteGesture.HOLD_START, NoteGesture.HOLD_CONTINUE, NoteGesture.HOLD_END}:
                event_window = self.hold_policy.execution_window_ms(event, track) / 1000.0
            elif event.gesture == NoteGesture.TAP:
                event_window = self.tap_policy.execution_window_ms(event, track, pending) / 1000.0
            if event.deadline <= now + event_window:
                lookahead = max(lookahead, event_window)
        near = [event for event in pending if event.deadline <= now + lookahead]
        if not near:
            return
        earliest = min(event.deadline for event in near)
        if earliest > now:
            delay = earliest - now
            if delay > 0.008:
                self.sleeper(delay - 0.006)
            while self.clock() < earliest:
                pass
            now = self.clock()
        tap_batches = due_tap_batches(pending, now, engine)
        allowed_taps = {e.event_id for batch in tap_batches for e in batch}
        due = [event for event in near if event.deadline <= now
               and (event.gesture != NoteGesture.TAP or event.event_id in allowed_taps)]
        if not due:
            return
        due_ids = {event.event_id for event in due}
        pending[:] = [event for event in pending if event.event_id not in due_ids]
        late = [event for event in due if (now - event.deadline) * 1000.0 > self.config.event_late_tolerance_ms]
        if late:
            LOGGER.warning(
                "Music action deadlines were missed: %s; executing immediately and recording latency",
                [
                    (event.track_id, event.gesture.value, round((now - event.deadline) * 1000.0, 1))
                    for event in late
                ],
            )
        taps = [event for event in due if event.gesture == NoteGesture.TAP or (event.gesture == NoteGesture.HOLD_START and not executor.supports_holds)]
        if taps:
            fallback_holds = [e for e in taps if e.gesture == NoteGesture.HOLD_START]
            batches = tap_batches + ([fallback_holds] if fallback_holds else [])
            for batch in batches:
                try:
                    receipts = executor.tap_many(
                        [(event.lane, *event.coordinate) for event in batch],
                        event_ids=[event.event_id for event in batch],
                    )
                except MusicTouchError as error:
                    self._acknowledge_taps(error.receipts, batch, engine, metrics)
                    raise
                if executor.async_input:
                    # Ack is deferred until the controller jobs complete.
                    for event in batch:
                        self._in_flight_taps[event.event_id] = event
                else:
                    self._acknowledge_taps(receipts or [], batch, engine, metrics)
            if engine is not None and not executor.supports_holds:
                for event in taps:
                    if event.gesture != NoteGesture.HOLD_START:
                        continue
                    fallback_track = engine.tracks.get(event.track_id)
                    if fallback_track is not None:
                        fallback_track.state = TrackState.RELEASED
        suppressed_tracks: set[int] = set()
        for event in due:
            if event in taps:
                continue
            self._dispatch_due_event(event, executor, engine, metrics, pending, suppressed_tracks)

    def _dispatch_due_event(
        self,
        event: MusicActionEvent,
        executor: MusicActionExecutor,
        engine: MusicVisionEngine | None,
        metrics: RuntimeMetrics,
        pending: list[MusicActionEvent],
        suppressed_tracks: set[int],
    ) -> None:
        """Owner-guarded dispatch of one due non-tap event (flicks included)."""
        if event.track_id in suppressed_tracks:
            return
        owner = executor.hold_owner(event.lane)
        occupied = event.lane in executor.active_contacts
        if (
            event.gesture == NoteGesture.HOLD_START
            and occupied
            and owner is not None
            and owner != event.track_id
            and engine is not None
        ):
            new_track = engine.tracks.get(event.track_id)
            owner_track = engine.tracks.get(owner)
            distinct_later_head = (
                new_track is not None
                and owner_track is not None
                and new_track.predicted_hit_time is not None
                and owner_track.predicted_hit_time is not None
                and new_track.predicted_hit_time - owner_track.predicted_hit_time
                > self.config.hold_same_lane_guard_ms / 1000.0
            )
            if distinct_later_head:
                LOGGER.info(
                    "Music same-lane hold handoff lane=%s prior_track=%s new_track=%s hit_gap_ms=%.0f",
                    event.lane,
                    owner,
                    event.track_id,
                    (new_track.predicted_hit_time - owner_track.predicted_hit_time) * 1000.0,
                )
                release_hint = owner_track.hold_release_time or new_track.predicted_hit_time
                direction = engine._hold_release_direction(owner_track, release_hint, log=False)
                if direction in FLICK_GESTURES:
                    coordinate = engine.calibration.points[owner_track.lane]
                    LOGGER.info(
                        "Music same-lane hold handoff honours end flick lane=%s prior_track=%s direction=%s",
                        event.lane,
                        owner,
                        direction.value,
                    )
                    executor.hold_flick(
                        event.lane,
                        int(coordinate[0]),
                        int(coordinate[1]),
                        direction,
                        track_id=owner,
                        tick=lambda: self._flush_due_taps(executor, pending, engine, metrics),
                    )
                else:
                    executor.touch_up(event.lane, track_id=owner)
                owner_track.state = TrackState.RELEASED
                owner_track.hold_release_scheduled = True
                suppressed_tracks.add(owner)
                pending[:] = [item for item in pending if item.track_id != owner]
                owner = executor.hold_owner(event.lane)
                occupied = event.lane in executor.active_contacts
        persistent_followup = (
            event.gesture in {NoteGesture.HOLD_CONTINUE, NoteGesture.HOLD_END}
            or (event.gesture in FLICK_GESTURES and event.contact_policy == "held_flick")
        )
        conflict = event.gesture == NoteGesture.HOLD_START and occupied
        stale_followup = persistent_followup and owner != event.track_id
        if conflict or stale_followup:
            reason = "occupied" if conflict else "owner-mismatch"
            LOGGER.warning(
                "Music suppressed stale hold event=%s gesture=%s lane=%s track=%s owner=%s reason=%s",
                event.event_id,
                event.gesture.value,
                event.lane,
                event.track_id,
                owner,
                reason,
            )
            suppressed_tracks.add(event.track_id)
            pending[:] = [item for item in pending if item.track_id != event.track_id]
            if engine is not None:
                track = engine.tracks.get(event.track_id)
                if track is not None:
                    track.state = TrackState.LOST
            return
        action_started = self.clock()
        if event.gesture in FLICK_GESTURES:
            LOGGER.info(
                "Music flick executed track=%s lane=%s gesture=%s source=%s",
                event.track_id,
                event.lane,
                event.gesture.value,
                "held_flick" if event.contact_policy == "held_flick" else "standalone",
            )
            executor.swipe(
                FlickRequest(
                    event.lane,
                    *event.coordinate,
                    event.gesture,
                    already_down=event.contact_policy == "held_flick",
                ),
                event_id=event.event_id,
                track_id=event.track_id if event.contact_policy == "held_flick" else None,
                tick=lambda: self._flush_due_taps(executor, pending, engine, metrics),
            )
            self._record_head_action(event, action_started, engine)
        elif event.gesture == NoteGesture.HOLD_START:
            self._record_hold_call(lambda: executor.touch_down(event.lane, *event.coordinate, track_id=event.track_id), event, engine)
            self._record_head_action(event, action_started, engine)
            if engine is not None:
                started_track = engine.tracks.get(event.track_id)
                if started_track is not None:
                    started_track.state = TrackState.HOLDING
        elif event.gesture == NoteGesture.HOLD_CONTINUE:
            self._record_hold_call(lambda: executor.touch_move(event.lane, *event.coordinate, track_id=event.track_id), event, engine)
        elif event.gesture == NoteGesture.HOLD_END:
            self._record_hold_call(lambda: executor.touch_up(event.lane, track_id=event.track_id), event, engine)
        if event.source_capture_started > 0.0:
            metrics.perception_to_action.append(max(0.0, (action_started - event.source_capture_started) * 1000.0))
        if event.source_capture_finished > 0.0:
            metrics.capture_to_action.append(max(0.0, (action_started - event.source_capture_finished) * 1000.0))

    @staticmethod
    def pending_within(pending: list[MusicActionEvent], now: float, guard_ms: float) -> bool:
        limit = now + guard_ms / 1000.0
        return any(event.deadline <= limit for event in pending)

    def _service_imminent_before_capture(
        self,
        executor: MusicActionExecutor,
        pending: list[MusicActionEvent],
        metrics: RuntimeMetrics,
        engine: MusicVisionEngine,
    ) -> None:
        now = self.clock()
        if self.pending_within(pending, now, self.config.pre_capture_deadline_guard_ms):
            self._execute_due(executor, pending, now, metrics, engine)

    @staticmethod
    def chart_activity_present(
        engine: MusicVisionEngine,
        executor: MusicActionExecutor,
        pending: list[MusicActionEvent],
        frame_sequence: int,
    ) -> bool:
        """Return whether result OCR would currently endanger live input.

        Pending actions and physical contacts are unambiguous activity.  A
        fresh approaching trajectory also counts while it is acquiring its
        four scheduling samples or has measurable downward speed.  Executed
        TAP_PENDING tracks intentionally do not count; they retain their state
        only for deadline refinement and must not block end detection forever.
        """
        if pending or executor.active_contacts:
            return True
        for track in engine.tracks.values():
            if track.state in {TrackState.HOLD_PENDING, TrackState.HOLDING}:
                return True
            if track.state != TrackState.APPROACHING or not track.observations:
                continue
            last = track.observations[-1]
            if last.frame_sequence < frame_sequence - 1:
                continue
            if len(track.observations) < 4 or track.speed > engine.config.static_speed_threshold:
                return True
        return False

    def terminal_ocr_ready(
        self,
        *,
        now: float,
        schedule_started: float,
        next_terminal_check: float,
        note_activity_seen: bool,
        last_chart_activity: float,
        chart_active: bool,
    ) -> bool:
        return (
            note_activity_seen
            and not chart_active
            and now >= next_terminal_check
            and now - schedule_started >= self.config.terminal_min_runtime_ms / 1000.0
            and now - last_chart_activity >= self.config.terminal_quiet_ms / 1000.0
        )

    def play(self, first_frame: MusicFrame, task_id: int | None = None) -> MusicRunResult:
        failed = self.startup_gate(first_frame, task_id)
        if failed is None:
            try:
                self.activate_play_provider()
            except Exception as error:
                failed = self._failure(
                    MusicFailureCode.CANDIDATE_FAILURE,
                    f"Unable to initialize the live candidate Provider: {error}",
                    task_id,
                )
        if failed:
            return failed
        if self.config.lane_count == 9 or self.config.experimental_only:
            return MusicRunResult(
                status="succeeded",
                reason="9-lane V4 candidate recognition and compatibility preflight passed; formal actions remain disabled",
                provider=getattr(self.provider, "name", ""),
                input_mode="preflight-only",
                controller_signature=self.signature,
                profile=self.calibration.profile_key if self.calibration else "",
                task_id=task_id,
                metrics_ms=self.metrics.summaries([]),
            )
        if self.calibration is None or self.provider is None:
            return self._failure(MusicFailureCode.INTERNAL_ERROR, "Music runtime was not initialized", task_id)
        executor: MusicActionExecutor | None = None
        pending: list[MusicActionEvent] = []
        try:
            executor = self._create_executor()
            self.action_durations = executor.action_durations
            self._in_flight_taps: dict[str, MusicActionEvent] = {}
            engine = MusicVisionEngine(self.calibration, self.config, tap_trace=self.tap_trace)
            LOGGER.info('Music candidate version=%s run_id=%s config=%s', VERSION, self.tap_trace.run_id, self.tap_trace.config_hash)
            started = self.monotonic()
            schedule_started = self.clock()
            next_pause_check = schedule_started
            next_terminal_check = schedule_started + self.config.terminal_initial_delay_ms / 1000.0
            note_activity_seen = False
            last_chart_activity = schedule_started
            sequence = 0
            capture_failures = 0
            provider_failures = 0
            performance_warned = False
            while self.monotonic() - started < self.config.max_duration_seconds:
                loop_started = self.clock()
                if is_stopping(self.context):
                    return self._failure(MusicFailureCode.CANCELLED, "User cancelled music play", task_id)
                # A controller screencap now costs about 25--35 ms while the
                # independent preview is active.  Do not start it when a known
                # tap/head/tail is already closer than that capture boundary.
                self._service_imminent_before_capture(executor, pending, self.metrics, engine)
                self._poll_async_inputs(executor, engine, self.metrics)
                frame, capture_ms = _capture_frame(
                    self.context,
                    sequence,
                    self.clock,
                    self.config.capture_timeout_ms,
                )
                sequence += 1
                self.metrics.capture.append(capture_ms)
                if frame is None:
                    capture_failures += 1
                    if capture_failures >= self.config.max_capture_failures:
                        return self._failure(MusicFailureCode.CAPTURE_FAILURE, f"{self.config.max_capture_failures} consecutive screenshots failed", task_id)
                    continue
                capture_failures = 0
                now = self.clock()
                executor.enforce_contact_limits()
                # Service deadlines already predicted by prior frames before any
                # relatively expensive UI recognition can block the action loop.
                self._execute_due(executor, pending, now, self.metrics, engine)
                now = self.clock()
                if now >= schedule_started + self.config.terminal_initial_delay_ms / 1000.0 and giant_live_title_present(frame.image):
                    LOGGER.info("Music terminal detected by strict giant LIVE visual confirmation")
                    return MusicRunResult(
                        status="succeeded",
                        reason="Giant central LIVE transition title detected",
                        provider=self.provider.name,
                        input_mode=self.input_mode,
                        controller_signature=self.signature,
                        profile=self.calibration.profile_key,
                        task_id=task_id,
                    )
                if now >= next_pause_check:
                    next_pause_check = now + self.config.pause_check_interval_ms / 1000.0
                    if _recognize(self.context, "MusicPauseDialog", frame.image):
                        _resume_frame, sequence, paused_seconds, resume_failure = self._wait_until_resumed(executor, sequence, task_id)
                        if resume_failure is not None:
                            return resume_failure
                        # Old deadlines and hold contacts describe the frozen
                        # pre-pause frame.  Reusing them would fire stale notes as
                        # soon as the countdown disappears, so resume from a clean
                        # tracker while preserving every ordinary tap parameter.
                        pending.clear()
                        self.tap_trace.add('pause_reset', time=self.clock())
                        engine = MusicVisionEngine(self.calibration, self.config, tap_trace=self.tap_trace)
                        started += paused_seconds
                        now = self.clock()
                        next_pause_check = now + self.config.pause_check_interval_ms / 1000.0
                        next_terminal_check = now + self.config.terminal_initial_delay_ms / 1000.0
                        last_chart_activity = now
                        capture_failures = 0
                        provider_failures = 0
                        continue
                try:
                    visual = VisualMask.from_image(frame.image, self.calibration)
                    provider_started = self.clock()
                    candidates = self.provider.detect(frame, visual)
                    self.metrics.provider.append((self.clock() - provider_started) * 1000.0)
                    provider_failures = 0
                except Exception as error:
                    provider_failures += 1
                    if provider_failures >= self.config.max_provider_failures:
                        return self._failure(MusicFailureCode.CANDIDATE_FAILURE, f"Candidate Provider failed three times: {error}", task_id)
                    continue
                new_events = engine.update(frame, candidates, visual)
                self.metrics.tracks_created = max(self.metrics.tracks_created, engine.next_track_id - 1)
                self.metrics.tracks_retained_peak = max(self.metrics.tracks_retained_peak, len(engine.tracks))
                self.metrics.tracks_expired = engine.expired_track_count
                self.metrics.tracks_pruned = engine.pruned_track_count
                self.metrics.urgent_rescues = engine.urgent_rescue_count
                self.metrics.isolated_same_lane_heads = engine.isolated_same_lane_head_count
                self.metrics.unscheduled_head_losses = engine.unscheduled_head_loss_count
                self.metrics.events_scheduled += len(new_events)
                pending.extend(new_events)
                # Candidate, hold-tail and route analysis can consume 10--40 ms.
                # Reusing the pre-analysis timestamp makes a near event appear
                # outside its precision window, permits one more screenshot,
                # and then executes the head/tail 45--136 ms late.  Refresh the
                # clock immediately before release generation and dispatch.
                dispatch_now = self.clock()
                pending.extend(engine.release_events(dispatch_now))
                pending = engine.refine_pending(pending, dispatch_now)
                self._execute_due(executor, pending, dispatch_now, self.metrics, engine)
                terminal_now = self.clock()
                chart_active = self.chart_activity_present(engine, executor, pending, frame.sequence)
                if chart_active:
                    note_activity_seen = True
                    last_chart_activity = terminal_now
                if self.terminal_ocr_ready(
                    now=terminal_now,
                    schedule_started=schedule_started,
                    next_terminal_check=next_terminal_check,
                    note_activity_seen=note_activity_seen,
                    last_chart_activity=last_chart_activity,
                    chart_active=chart_active,
                ):
                    # OCR remains the strict confirmation for bottom-right
                    # “载入中”, but it can no longer consume 125--166 ms while
                    # a head, route or physical hold is active.  Giant LIVE is
                    # still checked cheaply before perception on every frame.
                    terminal_started = self.clock()
                    state = terminal_state(self.context, frame.image)
                    self.metrics.terminal.append((self.clock() - terminal_started) * 1000.0)
                    next_terminal_check = terminal_now + self.config.end_check_interval_ms / 1000.0
                    if state == "result":
                        return MusicRunResult(
                            status="succeeded",
                            reason="Bottom-right loading text or central LIVE title detected",
                            provider=self.provider.name,
                            input_mode=self.input_mode,
                            controller_signature=self.signature,
                            profile=self.calibration.profile_key,
                            task_id=task_id,
                        )
                self.metrics.loop.append((self.clock() - loop_started) * 1000.0)
                if not performance_warned and len(self.metrics.loop) >= 30:
                    loop_p95 = percentile_95(self.metrics.loop[-30:])
                    if loop_p95 > self.config.max_loop_p95_ms:
                        LOGGER.warning(
                            "Music runtime loop P95 %.1f ms exceeds the %.1f ms target; continuing with live tracking",
                            loop_p95,
                            self.config.max_loop_p95_ms,
                        )
                    performance_warned = True
                if self.config.sample_interval_ms:
                    remaining = self.config.sample_interval_ms / 1000.0 - (self.clock() - loop_started)
                    if remaining > 0:
                        self.sleeper(remaining)
            return self._failure(MusicFailureCode.SONG_TIMEOUT, "Maximum song duration exceeded", task_id)
        except MusicTouchError as error:
            return self._failure(MusicFailureCode.TOUCH_BACKEND_FUSED, str(error), task_id)
        except Exception as error:
            LOGGER.exception("Music V4 runtime failed")
            return self._failure(MusicFailureCode.INTERNAL_ERROR, f"Music runtime exception: {error}", task_id)
        finally:
            if executor is not None:
                try:
                    executor.release_all()
                except Exception as error:
                    self.cleanup_failure = str(error)
                    LOGGER.exception("Music contact cleanup failed")
                self.tap_trace.add('input_fallbacks', reasons=executor.tap_fallbacks)
            try:
                path = self.tap_trace.write()
                LOGGER.info('Music tap diagnostics run_id=%s path=%s dropped=%s', self.tap_trace.run_id, path, self.tap_trace.dropped)
            except Exception:
                LOGGER.exception('Unable to write tap diagnostics; gameplay result unchanged')
            if self.head_action_trace:
                LOGGER.info(
                    "Music head action trace count=%s format=ordinal:track:lane:gesture:clock:late_ms:flags entries=%s",
                    len(self.head_action_trace),
                    "|".join(self.head_action_trace),
                )


def run_preflight_action(context: Any, config: MusicConfig, argv: Any) -> MusicRunResult:
    runtime = MusicRuntime(context, config)
    frame, capture_ms = _capture_frame(context, 0, runtime.clock, config.capture_timeout_ms)
    if frame is None:
        result = runtime._failure(MusicFailureCode.CAPTURE_FAILURE, "Screenshot unavailable before preflight", _task_id(argv))
    else:
        runtime.metrics.capture.append(capture_ms)
        result = runtime.preflight(frame, _task_id(argv))
    write_result(result)
    return result


def run_play_action(context: Any, config: MusicConfig, argv: Any) -> MusicRunResult:
    runtime = MusicRuntime(context, config)
    frame, capture_ms = _capture_frame(context, 0, runtime.clock, config.capture_timeout_ms)
    if frame is None:
        result = runtime._failure(MusicFailureCode.CAPTURE_FAILURE, "Screenshot unavailable before start", _task_id(argv))
    else:
        runtime.metrics.capture.append(capture_ms)
        result = runtime.play(frame, _task_id(argv))
    if runtime.cleanup_failure:
        result = runtime._failure(
            MusicFailureCode.TOUCH_BACKEND_FUSED,
            f"Touch cleanup failed: {runtime.cleanup_failure}",
            _task_id(argv),
        )
    result.metrics_ms = runtime.metrics.summaries(runtime.action_durations)
    write_result(result)
    return result
