"""Single-threaded, interleaved direct-touch flick execution.

This is an input resource batch, not evidence that notes form a chart chord.
Members retain their own deadlines and motion spacing. No native calls run
concurrently, and no controller jobs or new screenshots are posted here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import FlickRequest


@dataclass(frozen=True)
class FlickSubmission:
    request: FlickRequest
    event_id: str = ''
    track_id: int | None = None
    deadline: float | None = None


@dataclass
class FlickInputReceipt:
    event_id: str
    lane: int
    contact: int
    down_call_started: float | None = None
    down_call_finished: float | None = None
    move_call_started: list[float] = field(default_factory=list)
    move_call_finished: list[float] = field(default_factory=list)
    up_call_started: float | None = None
    up_call_finished: float | None = None
    error: str = ''
    cleanup_errors: list[str] = field(default_factory=list)
    run_id: str = ''
    segment_id: int | None = None
    deferred_reasons: list[str] = field(default_factory=list)


@dataclass
class _ActiveFlick:
    submission: FlickSubmission
    receipt: FlickInputReceipt
    points: list[tuple[int, int]]
    step_sleep: float
    next_due: float
    index: int = 0
    up_attempts: int = 0


class SyncFlickSession:
    def __init__(self, executor, *, collect_due=None, tick=None,
                 check_cancelled=None, on_started=None):
        self.executor = executor
        self.collect_due = collect_due
        self.tick = tick
        self.check_cancelled = check_cancelled
        self.on_started = on_started
        self.active: list[_ActiveFlick] = []
        self.receipts: list[FlickInputReceipt] = []
        self.waiting = {}
        self.deferred = {}
        self._deferral_reasons = {}
        self.closed = False

    def _key(self, submission):
        if submission.event_id:
            return ('event', self.executor._event_key(submission.event_id))
        return ('anonymous', id(submission))

    def _check(self):
        if self.check_cancelled is not None:
            self.check_cancelled()
        if self.closed:
            self.executor._raise_flick_error('Flick session was released during execution', self.receipts)

    def _defer(self, submission, reason):
        key = self._key(submission)
        self.deferred[key] = (submission, reason)
        reasons = self._deferral_reasons.setdefault(key, [])
        if reason not in reasons:
            reasons.append(reason)

    def offer(self, submissions: Iterable[FlickSubmission]):
        now = self.executor.clock()
        active_keys = {self._key(item.submission) for item in self.active}
        for submission in submissions:
            key = self._key(submission)
            if key in active_keys or key in self.waiting:
                continue
            if (submission.event_id
                    and self.executor._event_key(submission.event_id) in self.executor._used_event_ids):
                self._defer(submission, 'duplicate-event')
                continue
            if submission.request.already_down:
                self._defer(submission, 'legacy-held-flick')
                continue
            if submission.deadline is not None and submission.deadline > now:
                self._defer(submission, 'not-due')
                continue
            self.waiting[key] = submission

    def _admit(self):
        executor = self.executor
        prepared = []
        for key, submission in list(self.waiting.items()):
            self._check()
            request = submission.request
            if request.lane in executor.active_flick_lanes or request.lane in executor.active_contacts:
                self._defer(submission, 'occupied-lane')
                continue
            if (not executor.supports_multi_touch
                    and (executor._temporary_contacts or executor.active_contacts
                         or executor.release_unconfirmed)):
                self._defer(submission, 'no-multitouch')
                continue
            if executor.available_flick_contacts <= 0:
                self._defer(submission, 'contact-capacity')
                continue
            contact = executor._allocate_temporary_contact()
            executor._flick_lane_contacts[request.lane] = contact
            end = executor._flick_target(request.x, request.y, request.direction)
            points = executor._flick_waypoints(request.x, request.y, *end)
            receipt = FlickInputReceipt(submission.event_id, request.lane, contact,
                run_id=executor.run_id, segment_id=executor.segment_id,
                deferred_reasons=self._deferral_reasons.pop(key, []))
            item = _ActiveFlick(submission, receipt, points,
                executor._flick_step_sleep(len(points)), executor.clock())
            self.active.append(item)
            self.receipts.append(receipt)
            prepared.append(item)
            del self.waiting[key]
            self.deferred.pop(key, None)
        # Reserve all selected contacts/targets before starting any member.
        for item in prepared:
            self._check()
            receipt, submission = item.receipt, item.submission
            receipt.down_call_started = executor.clock()
            if submission.event_id:
                executor._used_event_ids.add(executor._event_key(submission.event_id))
            if self.on_started is not None:
                self.on_started(submission, receipt)
            # A callback may observe cancellation or release_all; never press
            # a contact after that cleanup has already relinquished its lock.
            self._check()
            executor._send_flick_touch('down', receipt.contact,
                (submission.request.x, submission.request.y), receipt.lane)
            receipt.down_call_finished = executor.clock()
            item.next_due = receipt.down_call_finished

    def _release_confirmed(self, item):
        executor = self.executor
        contact, lane = item.receipt.contact, item.receipt.lane
        executor._temporary_contacts.discard(contact)
        executor.release_unconfirmed.discard(contact)
        if executor._flick_lane_contacts.get(lane) == contact:
            del executor._flick_lane_contacts[lane]

    def _advance(self, item):
        self._check()
        executor, receipt = self.executor, item.receipt
        if item.index < len(item.points):
            receipt.move_call_started.append(executor.clock())
            executor._send_flick_touch('move', receipt.contact, item.points[item.index], receipt.lane)
            receipt.move_call_finished.append(executor.clock())
            item.index += 1
            item.next_due = receipt.move_call_finished[-1] + item.step_sleep
            if item.index == len(item.points):
                item.next_due += max(0., executor.config.flick_end_hold_ms / 1000.)
        else:
            receipt.up_call_started = executor.clock()
            item.up_attempts += 1
            executor._send_flick_touch('up', receipt.contact, None, receipt.lane)
            receipt.up_call_finished = executor.clock()
            self._release_confirmed(item)
            self.active.remove(item)

    def _cleanup(self, error):
        failures = {}
        for item in reversed(self.active):
            receipt = item.receipt
            receipt.error = str(error)
            if receipt.down_call_started is None:
                self._release_confirmed(item)
                continue  # An unstarted reserved contact was never pressed.
            while item.up_attempts < 2 and receipt.up_call_finished is None:
                item.up_attempts += 1
                if receipt.up_call_started is None:
                    receipt.up_call_started = self.executor.clock()
                try:
                    self.executor._send_flick_touch('up', receipt.contact, None,
                                                   receipt.lane, force=True, cleanup=True)
                    receipt.up_call_finished = self.executor.clock()
                    self._release_confirmed(item)
                except Exception as cleanup_error:
                    receipt.cleanup_errors.append(str(cleanup_error))
            if receipt.up_call_finished is None:
                self.executor.release_unconfirmed.add(receipt.contact)
                failures[receipt.contact] = receipt.cleanup_errors
        self.active.clear()
        if failures:
            reason = f'Flick cleanup unconfirmed: {failures}; original={error}'
            self.executor._mark_fused(reason)
            self.executor._raise_flick_error(reason, self.receipts, cause=error)

    def released_externally(self):
        # release_all has already confirmed each Up. Do not send it again
        # when the session unwinds after a pause/cancellation cleanup.
        for item in self.active:
            receipt = item.receipt
            if (receipt.contact not in self.executor._temporary_contacts
                    and receipt.contact not in self.executor.release_unconfirmed):
                if receipt.down_call_started is not None:
                    # release_all records the actual individual native call,
                    # not the later time at which the whole cleanup finished.
                    if receipt.up_call_finished is None:
                        receipt.up_call_finished = self.executor.clock()
                receipt.error = 'Flick was released by input cleanup'
        self.active.clear()
        self.closed = True

    def external_up_started(self, contact, moment):
        for item in self.active:
            if item.receipt.contact == contact:
                item.up_attempts += 1
                if item.receipt.up_call_started is None:
                    item.receipt.up_call_started = moment

    def external_up_finished(self, contact, moment, error=None):
        for item in self.active:
            if item.receipt.contact == contact:
                if error is None:
                    item.receipt.up_call_finished = moment
                else:
                    item.receipt.cleanup_errors.append(str(error))

    def run(self, submissions):
        try:
            self.offer(submissions)
            while True:
                self._check()
                self._admit()
                if self.tick is not None:
                    self.tick()
                self._check()
                if self.collect_due is not None:
                    self.offer(self.collect_due(self.executor.active_flick_lanes,
                                                self.executor.available_flick_contacts) or ())
                    self._admit()
                if not self.active:
                    break
                item = min(self.active, key=lambda member: member.next_due)
                delay = item.next_due - self.executor.clock()
                if delay > 0.:
                    self.executor.sleeper(min(.010, delay))
                    continue
                self._advance(item)
        except Exception as error:
            self._cleanup(error)
            # Preserve cancellation's own type, but attach truthful partial
            # input receipts just as native MusicTouchError does.
            error.receipts = list(self.receipts)
            raise
        finally:
            self.executor.last_flick_deferred = list(self.deferred.values())
        return self.receipts
