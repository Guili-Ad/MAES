from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from agent.common import LOGGER

from .models import FlickRequest, LaneInputState, MusicConfig, NoteGesture
from .storage import fuse_touch_backend

try:
    from maa.pipeline import JActionType, JClick, JSwipe, JTouch, JTouchUp
except ModuleNotFoundError:  # pragma: no cover - MaaFramework is bundled in production
    JActionType = JClick = JSwipe = JTouch = JTouchUp = None  # type: ignore[assignment]


class MusicTouchError(RuntimeError):
    def __init__(self, message, *, receipts=()):
        super().__init__(message)
        self.receipts = list(receipts)


@dataclass
class TapInputReceipt:
    event_id: str
    lane: int
    contact: int
    down_call_started: float | None = None
    down_call_finished: float | None = None
    up_call_finished: float | None = None
    error: str = ''


@dataclass(frozen=True)
class SimpleAction:
    contact: int = 0
    target: tuple[int, int, int, int] | None = None


@dataclass
class PendingInput:
    """One asynchronously posted gesture waiting for its controller jobs."""

    kind: str  # "tap" | "flick" | "hold_flick"
    lane: int
    contact: int
    event_id: str
    job_down: Any = None
    job_up: Any = None
    receipt: TapInputReceipt | None = None


class MusicActionExecutor:
    """All music input goes through synchronous Framework direct actions."""

    def __init__(
        self,
        context: Any,
        width: int,
        height: int,
        config: MusicConfig,
        *,
        controller_signature: str = "",
        advanced: bool = False,
        multi_touch: bool = False,
        async_input: bool = False,
        async_flicks: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.context = context
        self.width = width
        self.height = height
        self.config = config
        self.controller_signature = controller_signature
        self.supports_holds = bool(advanced and config.enable_holds)
        self.supports_multi_touch = bool(advanced and multi_touch)
        self.async_input = bool(async_input)
        self.async_flicks = bool(async_flicks)
        self.mode = "advanced" if advanced else "compatibility"
        self.sleeper = sleeper
        self.clock = clock
        self.healthy = True
        self.fuse_reason = ""
        self.action_durations: list[float] = []
        self.lanes: dict[int, LaneInputState] = {}
        self._used_event_ids: set[str] = set()
        self._temporary_contacts: set[int] = set()
        self.tap_fallbacks: list[str] = []
        self._pending_inputs: list[PendingInput] = []

    @property
    def active_contacts(self) -> dict[int, int]:
        return {lane: state.contact for lane, state in self.lanes.items() if state.contact is not None}

    def hold_owner(self, lane: int) -> int | None:
        state = self.lanes.get(lane)
        if state is None or state.contact is None:
            return None
        return state.hold_track_id

    def _lane_state(self, lane: int) -> LaneInputState:
        return self.lanes.setdefault(lane, LaneInputState(lane=lane))

    def _target(self, x: int, y: int) -> tuple[int, int, int, int]:
        return (
            max(0, min(self.width - 1, int(x))),
            max(0, min(self.height - 1, int(y))),
            1,
            1,
        )

    def _allocate_contact(self, lane: int) -> int:
        state = self._lane_state(lane)
        if state.contact is not None:
            raise MusicTouchError(f"Lane {lane} already owns contact {state.contact}")
        used = {item.contact for item in self.lanes.values() if item.contact is not None} | self._temporary_contacts
        for contact in range(self.config.max_contacts):
            if contact not in used:
                state.contact = contact
                return contact
        raise MusicTouchError("No free touch contacts")

    def _allocate_temporary_contact(self) -> int:
        used = {item.contact for item in self.lanes.values() if item.contact is not None} | self._temporary_contacts
        for contact in range(self.config.max_contacts):
            if contact not in used:
                self._temporary_contacts.add(contact)
                return contact
        raise MusicTouchError("No free temporary touch contacts")

    def _mark_fused(self, reason: str) -> None:
        self.healthy = False
        self.fuse_reason = reason
        if self.controller_signature:
            try:
                fuse_touch_backend(self.controller_signature, reason)
            except Exception:
                LOGGER.exception("Unable to persist music touch fuse")

    def _run(self, action_type: Any, param: Any, description: str, budget_ms: float, *, force: bool = False) -> float:
        if not self.healthy and not force:
            raise MusicTouchError("触控后端已熔断；请重新连接模拟器后重试")
        started = self.clock()
        try:
            detail = self.context.run_action_direct(action_type, param)
            if detail is None:
                raise RuntimeError("Framework returned None")
            success = getattr(detail, "success", None)
            if callable(success):
                success = success()
            if success is False:
                raise RuntimeError("Framework reported success=false")
        except Exception as error:
            reason = f"{description} failed: {error}"
            if not force:
                self._mark_fused(reason)
            raise MusicTouchError(reason) from error
        elapsed_ms = (self.clock() - started) * 1000.0
        self.action_durations.append(elapsed_ms)
        if elapsed_ms > budget_ms:
            LOGGER.warning(
                "%s exceeded soft budget: %.1f ms > %.1f ms; action succeeded, keeping touch backend active",
                description,
                elapsed_ms,
                budget_ms,
            )
        return elapsed_ms

    def _bindings(self) -> None:
        if JActionType is None or JClick is None or JSwipe is None or JTouch is None or JTouchUp is None:
            raise MusicTouchError("MaaFramework direct action bindings are unavailable")

    def _controller(self) -> Any:
        controller = getattr(getattr(self.context, "tasker", None), "controller", None)
        if controller is None:
            raise MusicTouchError("Async music input requires context.tasker.controller")
        return controller

    def _tasker(self) -> Any:
        tasker = getattr(self.context, "tasker", None)
        if tasker is None:
            raise MusicTouchError("Async music input requires context.tasker")
        return tasker

    @staticmethod
    def _job_flag(job: Any, name: str) -> bool:
        if job is None:
            return False
        value = getattr(job, name, None)
        if value is None:
            return False
        return bool(value() if callable(value) else value)

    @staticmethod
    def _valid_job(job: Any) -> bool:
        """Reject unbacked controller jobs before any native status query.

        The Python binding does not guard ``MaaInvalidId`` (see MaaFramework
        issue #408: querying an unbacked job id dereferences -1 in native
        code).  A failed post must fail here, never at poll time.
        """
        if job is None:
            return False
        job_id = getattr(job, "job_id", None)
        if job_id is None:
            return False
        try:
            return int(job_id) >= 0
        except (TypeError, ValueError):
            return False

    def _require_valid_job(self, job: Any, description: str) -> Any:
        if not self._valid_job(job):
            reason = f"{description} returned an invalid controller job"
            self._mark_fused(reason)
            raise MusicTouchError(reason)
        return job

    def _tap_many_async(self, fresh: list[tuple[tuple[int, int, int], str]]) -> list[TapInputReceipt]:
        """Post tap down/up pairs to the controller queue and return at once.

        Chord semantics are preserved: for a multi-touch batch every down is
        posted before any up, so the contacts overlap on the device.
        """
        controller = self._controller()
        duplicate_lane = len({request[0] for request, _ in fresh}) < len(fresh)
        available = self.config.max_contacts - len(self.active_contacts) - len(self._temporary_contacts)
        serial = len(fresh) > 1 and (duplicate_lane or not self.supports_multi_touch or len(fresh) > available)
        batches = [[item] for item in fresh] if serial else [fresh]
        if serial:
            reason = "same-lane" if duplicate_lane else ("no-multitouch" if not self.supports_multi_touch else "contact-capacity")
            if len(self.tap_fallbacks) < 256:
                self.tap_fallbacks.append(reason)
        receipts: list[TapInputReceipt] = []
        for batch in batches:
            contacts: list[tuple[TapInputReceipt, int, int, int]] = []
            for (lane, x, y), event_id in batch:
                contact = self._allocate_temporary_contact()
                receipt = TapInputReceipt(event_id, lane, contact, down_call_started=self.clock())
                contacts.append((receipt, contact, x, y))
                receipts.append(receipt)
                if event_id:
                    self._used_event_ids.add(event_id)
            posted: list[tuple[TapInputReceipt, int, Any]] = []
            for receipt, contact, x, y in contacts:
                try:
                    job_down = self._require_valid_job(
                        controller.post_touch_down(x, y, contact=contact, pressure=1),
                        f"async tap down lane {receipt.lane}",
                    )
                except Exception as error:
                    reason = f"async tap post lane {receipt.lane} failed: {error}"
                    self._mark_fused(reason)
                    receipt.error = reason
                    self._temporary_contacts.discard(contact)
                    raise MusicTouchError(reason, receipts=receipts) from error
                posted.append((receipt, contact, job_down))
            for receipt, contact, job_down in posted:
                try:
                    job_up = self._require_valid_job(
                        controller.post_touch_up(contact),
                        f"async tap up lane {receipt.lane}",
                    )
                except Exception as error:
                    reason = f"async tap post lane {receipt.lane} failed: {error}"
                    self._mark_fused(reason)
                    receipt.error = reason
                    self._temporary_contacts.discard(contact)
                    raise MusicTouchError(reason, receipts=receipts) from error
                self._pending_inputs.append(
                    PendingInput("tap", receipt.lane, contact, receipt.event_id, job_down, job_up, receipt)
                )
        return receipts

    def poll_inputs(self, now: float | None = None) -> list[TapInputReceipt]:
        """Resolve completed asynchronous gestures; never blocks.

        Returns the receipts that have finished (or failed) since the last
        poll.  Contacts are released as soon as their job completes.
        """
        if not self._pending_inputs:
            return []
        moment = self.clock() if now is None else now
        completed: list[TapInputReceipt] = []
        for pending in list(self._pending_inputs):
            receipt = pending.receipt
            down_done = self._job_flag(pending.job_down, "done")
            up_done = self._job_flag(pending.job_up, "done")
            failed = self._job_flag(pending.job_down, "failed") or self._job_flag(pending.job_up, "failed")
            if receipt is not None:
                if pending.job_down is not None and down_done and receipt.down_call_finished is None:
                    receipt.down_call_finished = moment
                if pending.job_up is not None and up_done and receipt.up_call_finished is None:
                    receipt.up_call_finished = moment
            if failed:
                if receipt is not None:
                    receipt.error = "framework gesture job failed"
                self._mark_fused(f"async {pending.kind} job failed on lane {pending.lane}")
                self._finish_input(pending)
                self._pending_inputs.remove(pending)
                if receipt is not None:
                    completed.append(receipt)
                continue
            finished = (pending.job_up is not None and up_done) or (
                pending.job_up is None and pending.job_down is not None and down_done
            )
            if finished:
                self._finish_input(pending)
                self._pending_inputs.remove(pending)
                if receipt is not None:
                    completed.append(receipt)
        return completed

    def _finish_input(self, pending: PendingInput) -> None:
        if pending.kind == "flick":
            self._temporary_contacts.discard(pending.contact)
            return
        state = self.lanes.get(pending.lane)
        if state is not None and state.contact == pending.contact:
            state.contact = None
            state.hold_track_id = None
        self._temporary_contacts.discard(pending.contact)

    def tap(self, lane: int, x: int, y: int, *, event_id: str = "") -> list[TapInputReceipt]:
        return self.tap_many([(lane, x, y)], event_ids=[event_id])

    def tap_many(self, requests: Iterable[tuple[int, int, int]], *, event_ids: Iterable[str] | None = None) -> list[TapInputReceipt]:
        requests = list(requests)
        ids = list(event_ids) if event_ids is not None else [""] * len(requests)
        if len(ids) != len(requests):
            raise ValueError("Tap request/event ID counts differ")
        fresh = []
        seen = set(self._used_event_ids)
        for request, event_id in zip(requests, ids):
            if event_id and event_id in seen:
                continue
            fresh.append((request, event_id))
            if event_id:
                seen.add(event_id)
        if not fresh:
            return []
        self._bindings()
        if self.async_input:
            return self._tap_many_async(fresh)
        # Defense in depth: never overlap contacts on one lane, even when a
        # caller bypasses the scheduler. Valid chords have distinct lanes.
        duplicate_lane = len({request[0] for request, _ in fresh}) < len(fresh)
        available = self.config.max_contacts - len(self.active_contacts) - len(self._temporary_contacts)
        serial = len(fresh) > 1 and (duplicate_lane or not self.supports_multi_touch or len(fresh) > available)
        batches = [[item] for item in fresh] if serial else [fresh]
        if serial:
            reason = "same-lane" if duplicate_lane else ("no-multitouch" if not self.supports_multi_touch else "contact-capacity")
            if len(self.tap_fallbacks) < 256:
                self.tap_fallbacks.append(reason)
        receipts: list[TapInputReceipt] = []
        for batch in batches:
            contacts = []
            try:
                # Allocate all chord contacts and targets before the first down.
                for (lane, x, y), event_id in batch:
                    contact = self._allocate_temporary_contact()
                    receipt = TapInputReceipt(event_id, lane, contact)
                    contacts.append((receipt, JTouch(contact=contact, target=self._target(x, y), pressure=1)))
                    receipts.append(receipt)
                for receipt, target in contacts:
                    receipt.down_call_started = self.clock()
                    # A failed/ambiguous down must not be retried automatically.
                    if receipt.event_id:
                        self._used_event_ids.add(receipt.event_id)
                    self._run(JActionType.TouchDown, target, f"tap down lane {receipt.lane}", self.config.max_click_touch_ms)
                    receipt.down_call_finished = self.clock()
                for receipt, _ in reversed(contacts):
                    self._run(JActionType.TouchUp, JTouchUp(contact=receipt.contact),
                              f"tap up lane {receipt.lane}", self.config.max_click_touch_ms)
                    receipt.up_call_finished = self.clock()
                    self._temporary_contacts.discard(receipt.contact)
            except Exception as error:
                for receipt, _ in reversed(contacts):
                    receipt.error = str(error)
                    if receipt.down_call_started is not None and receipt.up_call_finished is None:
                        try:
                            self._run(JActionType.TouchUp, JTouchUp(contact=receipt.contact),
                                      f"tap cleanup lane {receipt.lane}", self.config.max_click_touch_ms, force=True)
                            receipt.up_call_finished = self.clock()
                        except Exception:
                            # Keep the contact reserved for release_all to retry.
                            continue
                    self._temporary_contacts.discard(receipt.contact)
                raise MusicTouchError(str(error), receipts=receipts) from error
        return receipts

    def swipe(
        self,
        request: FlickRequest,
        *,
        event_id: str = "",
        track_id: int | None = None,
        tick: Callable[[], None] | None = None,
    ) -> None:
        if event_id and event_id in self._used_event_ids:
            return
        self._bindings()
        if self.async_flicks:
            self._swipe_async(request, event_id)
            if event_id:
                self._used_event_ids.add(event_id)
            return
        if request.already_down:
            self.hold_flick(request.lane, request.x, request.y, request.direction, track_id=track_id, tick=tick)
            if event_id:
                self._used_event_ids.add(event_id)
            return
        contact = self._allocate_temporary_contact()
        end_x, end_y = self._flick_target(request.x, request.y, request.direction)
        waypoints = self._flick_waypoints(request.x, request.y, end_x, end_y)
        step_sleep = self._flick_step_sleep(len(waypoints))
        aborted = False
        try:
            self._run(
                JActionType.TouchDown,
                JTouch(contact=contact, target=self._target(request.x, request.y), pressure=1),
                f"flick down lane {request.lane}",
                self.config.max_click_touch_ms,
            )
            for point_x, point_y in waypoints:
                # Interpolated moves with real spacing: a single jump (or a
                # down+move+up inside one frame) is coalesced by the emulator
                # and the game only ever samples a tap.
                self._run(
                    JActionType.TouchMove,
                    JTouch(contact=contact, target=self._target(point_x, point_y), pressure=1),
                    f"flick move lane {request.lane}",
                    self.config.max_click_touch_ms,
                )
                self._flick_pause(step_sleep, tick)
            if self.config.flick_end_hold_ms > 0:
                self._flick_pause(self.config.flick_end_hold_ms / 1000.0, tick)
            self._run(
                JActionType.TouchUp,
                JTouchUp(contact=contact),
                f"flick up lane {request.lane}",
                self.config.max_click_touch_ms,
            )
        except Exception:
            aborted = True
            raise
        finally:
            self._temporary_contacts.discard(contact)
            if aborted:
                try:
                    self._run(
                        JActionType.TouchUp,
                        JTouchUp(contact=contact),
                        f"flick abort cleanup lane {request.lane}",
                        self.config.max_click_touch_ms,
                        force=True,
                    )
                except Exception:
                    LOGGER.exception("Failed to clean up an aborted flick on lane %s", request.lane)
        if event_id:
            self._used_event_ids.add(event_id)

    def _flick_pause(self, duration: float, tick: Callable[[], None] | None) -> None:
        """Pause between flick steps, first giving the caller a chance to
        dispatch input that comes due during the swipe (e.g. taps)."""
        if tick is not None:
            tick()
        if duration > 0:
            self.sleeper(duration)

    def _flick_target(self, x: int, y: int, direction: NoteGesture) -> tuple[int, int]:
        """Screen-axis swipe endpoint for one of the four flick directions.

        The sprite arrows point along screen axes even on the radial lane
        geometry, so vertical flicks move in Y while left/right keep the
        legacy horizontal behaviour.  Unknown directions keep the legacy
        rightward fallback.
        """
        distance = self.config.flick_distance_px
        delta_x, delta_y = distance, 0
        if direction == NoteGesture.FLICK_LEFT:
            delta_x, delta_y = -distance, 0
        elif direction == NoteGesture.FLICK_UP:
            delta_x, delta_y = 0, -distance
        elif direction == NoteGesture.FLICK_DOWN:
            delta_x, delta_y = 0, distance
        end_x = max(0, min(self.width - 1, x + delta_x))
        end_y = max(0, min(self.height - 1, y + delta_y))
        return end_x, end_y

    def flick_many(self, requests: Iterable[FlickRequest]) -> None:
        for request in requests:
            self.swipe(request)

    def touch_down(self, lane: int, x: int, y: int, *, track_id: int | None = None) -> None:
        if not self.supports_holds:
            raise MusicTouchError("当前配置未启用持续触点")
        self._bindings()
        contact = self._allocate_contact(lane)
        state = self._lane_state(lane)
        try:
            self._run(
                JActionType.TouchDown,
                JTouch(contact=contact, target=self._target(x, y), pressure=1),
                f"touch down lane {lane}",
                self.config.max_click_touch_ms,
            )
        except Exception:
            try:
                self._run(
                    JActionType.TouchUp,
                    JTouchUp(contact=contact),
                    f"failed touch down cleanup lane {lane}",
                    self.config.max_click_touch_ms,
                    force=True,
                )
            except Exception:
                LOGGER.exception("Failed to clean up a partially started contact on lane %s", lane)
            state.contact = None
            raise
        state.hold_track_id = track_id
        state.contact_started = time.monotonic()

    def touch_move(self, lane: int, x: int, y: int, *, track_id: int | None = None) -> None:
        state = self._lane_state(lane)
        if track_id is not None and state.hold_track_id != track_id:
            LOGGER.warning(
                "Ignored stale hold move lane=%s track=%s owner=%s",
                lane,
                track_id,
                state.hold_track_id,
            )
            return
        if state.contact is None:
            raise MusicTouchError(f"Lane {lane} has no active contact")
        self._bindings()
        self._run(
            JActionType.TouchMove,
            JTouch(contact=state.contact, target=self._target(x, y), pressure=1),
            f"touch move lane {lane}",
            self.config.max_click_touch_ms,
        )

    def _touch_up(self, lane: int, *, force: bool = False) -> None:
        self._bindings()
        state = self._lane_state(lane)
        contact = state.contact
        if contact is None:
            return
        try:
            self._run(
                JActionType.TouchUp,
                JTouchUp(contact=contact),
                f"touch up lane {lane}",
                self.config.max_click_touch_ms,
                force=force,
            )
        finally:
            state.contact = None
            state.hold_track_id = None
            state.contact_started = 0.0

    def touch_up(self, lane: int, *, track_id: int | None = None) -> None:
        state = self._lane_state(lane)
        if track_id is not None and state.hold_track_id != track_id:
            LOGGER.warning(
                "Ignored stale hold release lane=%s track=%s owner=%s",
                lane,
                track_id,
                state.hold_track_id,
            )
            return
        self._touch_up(lane)

    def hold_flick(
        self,
        lane: int,
        x: int,
        y: int,
        direction: NoteGesture,
        *,
        track_id: int | None = None,
        tick: Callable[[], None] | None = None,
    ) -> None:
        """Swipe an already-pressed hold contact and release it.

        The hold contact is moved through the same interpolated waypoints as a
        standalone flick so the game samples real motion before the release.
        """
        end_x, end_y = self._flick_target(x, y, direction)
        waypoints = self._flick_waypoints(x, y, end_x, end_y)
        step_sleep = self._flick_step_sleep(len(waypoints))
        for point_x, point_y in waypoints:
            self.touch_move(lane, point_x, point_y, track_id=track_id)
            self._flick_pause(step_sleep, tick)
        if self.config.flick_end_hold_ms > 0:
            self._flick_pause(self.config.flick_end_hold_ms / 1000.0, tick)
        self.touch_up(lane, track_id=track_id)

    def _flick_waypoints(self, x: int, y: int, end_x: int, end_y: int) -> list[tuple[int, int]]:
        steps = max(2, int(self.config.flick_steps))
        points: list[tuple[int, int]] = []
        for index in range(1, steps + 1):
            ratio = index / steps
            points.append((int(round(x + (end_x - x) * ratio)), int(round(y + (end_y - y) * ratio))))
        return points

    def _flick_step_sleep(self, steps: int) -> float:
        return max(0.004, self.config.flick_duration_ms / 1000.0 / max(1, steps))

    def _swipe_async(self, request: FlickRequest, event_id: str) -> None:
        """Post a flick to the controller queue without blocking.

        Standalone flicks use the native interpolated swipe (10 ms grid); held
        flicks issue a hover-only swipe on the already-pressed hold contact
        (moves without down/up) followed by an explicit touch-up.
        """
        controller = self._controller()
        end_x, end_y = self._flick_target(request.x, request.y, request.direction)
        if request.already_down:
            lane_state = self._lane_state(request.lane)
            if lane_state.contact is None:
                raise MusicTouchError(f"Lane {request.lane} has no held contact for a held flick")
            contact = lane_state.contact
            job_move = None
            post_action = getattr(self._tasker(), "post_action", None)
            if post_action is not None and JActionType is not None and JSwipe is not None:
                param = JSwipe(
                    begin=self._target(request.x, request.y),
                    end=[self._target(end_x, end_y)],
                    duration=[self.config.flick_duration_ms],
                    only_hover=True,
                    contact=contact,
                    pressure=1,
                )
                try:
                    job_move = self._require_valid_job(
                        post_action(JActionType.Swipe, param),
                        f"held flick post lane {request.lane}",
                    )
                except Exception as error:
                    reason = f"held flick post lane {request.lane} failed: {error}"
                    self._mark_fused(reason)
                    raise MusicTouchError(reason) from error
            else:
                mid_x = (request.x + end_x) // 2
                mid_y = (request.y + end_y) // 2
                controller.post_touch_move(mid_x, mid_y, contact=contact, pressure=1)
                controller.post_touch_move(end_x, end_y, contact=contact, pressure=1)
            job_up = self._require_valid_job(
                controller.post_touch_up(contact),
                f"held flick up lane {request.lane}",
            )
            self._pending_inputs.append(
                PendingInput("hold_flick", request.lane, contact, event_id, job_move, job_up, None)
            )
            return
        contact = self._allocate_temporary_contact()
        try:
            job = self._require_valid_job(
                controller.post_swipe(
                    request.x,
                    request.y,
                    end_x,
                    end_y,
                    duration=self.config.flick_duration_ms,
                    contact=contact,
                    pressure=1,
                ),
                f"flick post lane {request.lane}",
            )
        except Exception as error:
            self._temporary_contacts.discard(contact)
            reason = f"flick post lane {request.lane} failed: {error}"
            self._mark_fused(reason)
            raise MusicTouchError(reason) from error
        self._pending_inputs.append(PendingInput("flick", request.lane, contact, event_id, job, None, None))

    def may_open_contact_during_gesture(self) -> bool:
        """True when another contact may be opened while a gesture is airborne.

        Multi-touch controllers always allow it.  Single-touch controllers may
        only dispatch unrelated holds when no contact (temporary swipe or held
        lane) is currently down.
        """
        if self.supports_multi_touch:
            return True
        return not self._temporary_contacts and not self.active_contacts

    def enforce_contact_limits(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        expired = [
            lane
            for lane, state in self.lanes.items()
            if state.contact is not None
            and state.contact_started > 0
            and (current - state.contact_started) * 1000.0 > self.config.max_contact_ms
        ]
        if not expired:
            return
        for lane in expired:
            self.touch_up(lane)
        LOGGER.warning("Released contacts that exceeded the maximum duration on lanes %s", expired)

    def release_all(self) -> None:
        cleanup_failed = False
        self._pending_inputs.clear()
        for contact in sorted(self._temporary_contacts, reverse=True):
            try:
                self._run(JActionType.TouchUp, JTouchUp(contact=contact), 'temporary tap cleanup',
                          self.config.max_click_touch_ms, force=True)
                self._temporary_contacts.discard(contact)
            except Exception:
                cleanup_failed = True
        for lane in sorted(list(self.active_contacts), reverse=True):
            try:
                self._touch_up(lane, force=not self.healthy)
            except Exception:
                cleanup_failed = True
                LOGGER.exception("Failed to clean up music contact on lane %s", lane)
        if cleanup_failed and self.healthy:
            self._mark_fused("Touch cleanup failed")
            raise MusicTouchError("Touch cleanup failed")

    def close(self) -> None:
        self.release_all()


# Compatibility names now use the same native executor; neither calls Controller jobs.
CompatibilityMusicInput = MusicActionExecutor
MusicTouchSession = MusicActionExecutor
