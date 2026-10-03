"""Tap-mode physical gold tracks and virtual ribbon anchors.

No persistent-contact release, route or watchdog fields are consulted here.
The legacy sustained/drag modes keep their original tracker and lifecycle.
"""
from __future__ import annotations
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from .models import FLICK_GESTURES, NoteGesture, TrackState
from .sustain import SustainMarkerTracker
from .hold_topology import ribbon_at_judgement


class AnchorState(str, Enum):
    ACTIVE = 'active'
    QUIESCENT = 'quiescent'
    CLOSED = 'closed'


class OwnerEvidence(str, Enum):
    UNKNOWN = 'unknown'
    CONFIRMED = 'confirmed'
    CONTRADICTED = 'contradicted'


@dataclass
class TapHoldAnchor:
    track_id: int
    head_event_id: str
    started: float
    exit_lane: int
    partner_id: int | None = None
    state: AnchorState = AnchorState.ACTIVE
    last_evidence: float | None = None
    terminal_completed: float | None = None
    terminal_marker: int | None = None
    end_flick_id: int | None = None
    evidence_frame: int = -1
    connection_epoch: int = 0
    connection_lane: int | None = None
    route_confirmed: bool = False


class GoldMarkerTracker(SustainMarkerTracker):
    """One batch identity pass, retaining the first observation for warm-up."""
    def __init__(self, chain, trigger_progress):
        super().__init__(trigger_progress)
        self.chain = chain
        self.descriptors = {}
        self.rejected = {}

    def observe(self, marker_id, detection, frame, owner=None):
        state = super().observe(marker_id, detection, frame, owner)
        self.descriptors[state.marker_id] = detection
        if owner is not None:
            self.chain.note_evidence(owner, frame, detection.lane)
        return state

    def associate_detections(self, detections, frame):
        prior = [m for m in self.markers.values() if m.observations
                 and 0 <= frame.midpoint-m.last_seen_time <= .6]
        edges = {}
        for i, d in enumerate(detections):
            if d.physical_ring is False or (d.ring_coverage is not None and d.ring_coverage < .50):
                continue
            for j, m in enumerate(prior):
                history = list(m.observations)
                last = history[-1]
                dt = frame.midpoint-last.timestamp
                if frame.sequence == last.frame_sequence or d.progress < last.progress-.03:
                    continue
                descriptor = self.descriptors.get(m.marker_id)
                if (descriptor and descriptor.box and d.box
                        and (min(d.box[2:]) < min(descriptor.box[2:])*.55
                             or min(d.box[2:]) > min(descriptor.box[2:])*2.0)):
                    continue
                expected = last.center
                if len(history) >= 2:
                    before = history[-2]
                    elapsed = last.timestamp-before.timestamp
                    if elapsed > 1e-6:
                        scale = min(.6, dt)/elapsed
                        expected = (last.center[0]+(last.center[0]-before.center[0])*scale,
                                    last.center[1]+(last.center[1]-before.center[1])*scale)
                residual = math.dist(expected, d.center)
                if residual > 80. or abs(last.lane-d.lane) > 1:
                    continue
                expected_progress = last.progress+max(0., m.speed())*dt
                if len(history) >= 3 and abs(d.progress-expected_progress) > .12:
                    continue
                # A known ribbon cannot be measured by a contradictory HUD blob.
                if (m.owner is not None and d.owner_lanes
                        and not self.chain.owner_allowed(m.owner, d)):
                    if not (self.chain.migration_allowed(m.owner, d, m, frame, residual)
                            or self.chain.route_continuity(m.owner, d, m, frame)):
                        self.rejected[m.marker_id] = (frame.sequence, OwnerEvidence.CONTRADICTED)
                        continue
                edges[i, j] = residual
        # Solve connected conflicts only. Most rings have one possible edge.
        matches, visited, ambiguous_overflow = {}, set(), set()
        by_i, by_j = {}, {}
        for i, j in edges:
            by_i.setdefault(i, set()).add(j)
            by_j.setdefault(j, set()).add(i)
        for seed in by_i:
            if seed in visited:
                continue
            ci, cj, todo = {seed}, set(), [seed]
            while todo:
                i = todo.pop()
                for j in by_i[i]-cj:
                    cj.add(j)
                    added = by_j[j]-ci
                    ci.update(added)
                    todo.extend(added)
            visited.update(ci)
            current = sorted(ci, key=lambda i: (detections[i].center[1], i))
            previous = sorted(cj, key=lambda j: (prior[j].observations[-1].center[1], j))
            if len(previous) > 16:
                # An unbounded HUD conflict is not permission to swap physical
                # rings. Keep only mutually unique edges, defer the ambiguous
                # observations, and report the bounded-solver overflow.
                for i in current:
                    if len(by_i[i]) == 1:
                        j = next(iter(by_i[i]))
                        if len(by_j[j]) == 1:
                            matches[i] = prior[j].marker_id
                self.chain.trace.add('gold_association_overflow', frame=frame.sequence,
                    time=frame.midpoint, candidates=len(current), tracks=len(previous))
                ambiguous_overflow.update(i for i in current if i not in matches)
                visited.update(current)
                continue
            @lru_cache(None)
            def solve(pos, used):
                if pos == len(current):
                    return (0, 0., ())
                i = current[pos]
                choices = [solve(pos+1, used)]
                for bit, j in enumerate(previous):
                    if not used & (1 << bit) and (i,j) in edges:
                        # Candidates are traversed in screen order. For each
                        # confirmed ribbon, its previously observed order is
                        # also monotone. The used-state carries that constraint
                        # into recursion, rather than rejecting only its single
                        # already-selected unconstrained best solution.
                        if prior[j].owner is not None and any(
                            used & (1 << k) and prior[previous[k]].owner == prior[j].owner
                            and prior[previous[k]].last_seen_frame == prior[j].last_seen_frame
                            and prior[previous[k]].observations[-1].center[1]
                                > prior[j].observations[-1].center[1]+2.
                            for k in range(len(previous))):
                            continue
                        n, cost, pairs = solve(pos+1, used | (1 << bit))
                        choices.append((n+1, cost+edges[i,j], ((i,j),)+pairs))
                return min(choices, key=lambda x: (-x[0], x[1], x[2]))
            for i, j in solve(0, 0)[2]:
                matches[i] = prior[j].marker_id
        for i, d in enumerate(detections):
            if i in ambiguous_overflow:
                continue
            if d.physical_ring is False or (d.ring_coverage is not None and d.ring_coverage < .50):
                continue
            mid = matches.get(i)
            if mid is None:
                mid = self.allocate_id()
            m = self.markers.get(mid)
            route_pending = False
            if (m is not None and m.owner is not None and d.owner_lanes
                    and not self.chain.owner_allowed(m.owner, d)
                    and self.chain.migration_allowed(m.owner, d, m, frame)):
                anchor = self.chain.anchors[m.owner]
                self.chain.trace.add('tap_hold_route_evidence', time=frame.midpoint,
                    frame=frame.sequence, owner=m.owner, marker=mid,
                    previous_lane=anchor.exit_lane, lane=d.lane)
                anchor.exit_lane = d.lane
                anchor.route_confirmed = True
            elif (m is not None and m.owner is not None and d.owner_lanes
                    and not self.chain.owner_allowed(m.owner, d)):
                route_pending = self.chain.route_continuity(m.owner, d, m, frame)
            owner, evidence = ((None, OwnerEvidence.UNKNOWN) if route_pending
                               else self.chain.choose_owner(d, m, frame))
            if evidence == OwnerEvidence.CONTRADICTED:
                self.rejected[mid] = (frame.sequence, evidence)
                continue
            # A positively observed circular gold sprite may keep its physical
            # motion through a hidden ribbon. It does not manufacture ownership
            # evidence or turn a missing connection into a terminal vote.
            if d.topology == 'unknown' and m is not None and (
                    d.physical_ring is not True):
                continue
            if m is not None and m.observations:
                last = m.observations[-1]
                if d.center == last.center and abs(d.progress-last.progress) <= 1e-6:
                    continue  # repeated pixels are not another velocity sample
            self.rejected.pop(mid, None)
            state = (super().observe(mid, d, frame, None) if evidence == OwnerEvidence.UNKNOWN
                     else self.observe(mid, d, frame, owner))
            self.descriptors[state.marker_id] = d
            self.chain.trace.add('gold_observation', time=frame.midpoint, frame=frame.sequence,
                marker=mid, owner=state.owner, evidence=evidence.value,
                center=d.center, lane=d.lane, progress=d.progress, topology=d.topology,
                ring=d.ring_coverage, physical_ring=d.physical_ring, owner_lanes=d.owner_lanes,
                owner_scores=d.owner_scores)
        self.prune(frame.midpoint)
        alive = set(self.markers)
        self.descriptors = {k:v for k,v in self.descriptors.items() if k in alive}
        self.rejected = {k:v for k,v in self.rejected.items() if k in alive}


class TapHoldChain:
    def __init__(self, engine):
        self.engine, self.trace = engine, engine.tap_trace
        self.anchors = {}
        self.tracker = GoldMarkerTracker(self, engine.calibration.trigger_progress)
        self.last_frame_sequence = -1
        self.last_frame_time = None
        self.periods = deque(maxlen=8)
        self.connection_epochs = {}
        self.flick_frames = {}

    @property
    def coast_budget(self):
        period = statistics.median(self.periods) if self.periods else .10
        return min(.6, 2*period)

    def acknowledge_head(self, event, receipt):
        if (event.gesture != NoteGesture.HOLD_START or receipt.down_call_finished is None
                or receipt.up_call_finished is None or receipt.error):
            return
        track = self.engine.tracks.get(event.track_id)
        if track is None or track.track_id in self.anchors:
            return
        self.anchors[track.track_id] = TapHoldAnchor(track.track_id, event.event_id,
            receipt.down_call_started, track.lane, track.linked_partner_id,
            last_evidence=receipt.up_call_finished)
        epoch = self.connection_epochs.get(track.lane, 0)+1
        self.connection_epochs[track.lane] = epoch
        self.anchors[track.track_id].connection_epoch = epoch
        self.anchors[track.track_id].connection_lane = track.lane
        track.state = TrackState.HOLDING  # compatibility label; not ownership authority
        self.trace.add('tap_hold_anchor', time=receipt.up_call_finished,
                       track=track.track_id, state='active', reason='head-input-completed')

    def note_evidence(self, owner, frame, lane=None):
        anchor = self.anchors.get(owner)
        if anchor is not None and anchor.state != AnchorState.CLOSED:
            anchor.last_evidence = frame.midpoint
            anchor.evidence_frame = frame.sequence
            anchor.state = AnchorState.ACTIVE

    def owner_allowed(self, owner, detection):
        a = self.anchors.get(owner)
        return bool(a is not None and a.state != AnchorState.CLOSED
                    and (not detection.owner_lanes or a.exit_lane in detection.owner_lanes))

    def current_connection(self, anchor):
        lane = anchor.connection_lane if anchor.connection_lane is not None else anchor.exit_lane
        return anchor.connection_epoch == self.connection_epochs.get(lane, 0)

    def migration_allowed(self, owner, detection, marker, frame, residual=None):
        a = self.anchors.get(owner)
        if (a is None or a.state == AnchorState.CLOSED or not self.current_connection(a)
                or detection.physical_ring is not True or not detection.owner_lanes
                or not self.route_continuity(owner, detection, marker, frame)
                or len(marker.observations) < 2 or marker.speed() <= self.engine.config.hold_sustain_marker_min_speed):
            return False
        previous, last = list(marker.observations)[-2:]
        age = frame.midpoint-last.timestamp
        dt = last.timestamp-previous.timestamp
        if not 0 < age <= self.coast_budget or dt <= 1e-6:
            return False
        if residual is None:
            predicted = tuple(last.center[k]+(last.center[k]-previous.center[k])*age/dt for k in (0, 1))
            residual = math.dist(predicted, detection.center)
        extent = min(detection.box[2:]) if detection.box else 50.
        return (abs(last.lane-detection.lane) <= 1 and residual <= min(40., extent*.65)
                and abs(detection.progress-(last.progress+marker.speed()*age)) <= .05)

    def route_continuity(self, owner, detection, marker, frame):
        a = self.anchors.get(owner)
        descriptor = self.tracker.descriptors.get(marker.marker_id)
        if (a is None or a.state == AnchorState.CLOSED or not self.current_connection(a)
                or detection.physical_ring is not True or descriptor is None
                or descriptor.physical_ring is not True or not detection.owner_lanes
                or detection.topology == 'unknown' or not marker.observations):
            return False
        last = marker.observations[-1]
        # The connector scorer follows the curved white body and can project
        # onto an intermediate ray, distinct from both head and circle rays.
        # Strong physical continuity keeps the identity while that ownership
        # evidence matures; a stationary same-ray circle with an unrelated
        # connector is still a contradiction, not a route migration.
        return (0 < frame.midpoint-last.timestamp <= .6
                and abs(last.lane-detection.lane) <= 1
                and detection.progress > last.progress+.002
                and (a.route_confirmed or detection.lane != a.exit_lane
                     or last.lane != a.exit_lane))

    def terminal(self, marker):
        # A one-frame judgement flash can hide the downstream ribbon of an
        # internal ring. Established checkpoint evidence is not overwritten.
        known = [o for o in marker.observations if o.topology != 'unknown']
        return (marker.sustain_votes == 0 and marker.terminal_votes >= 2
                and any(o.topology == 'terminal' for o in known))

    def choose_owner(self, detection, marker, frame):
        if detection.topology == 'unknown' or detection.owner_lanes == ():
            return None, OwnerEvidence.UNKNOWN
        if marker is not None and marker.owner is not None and self.owner_allowed(marker.owner, detection):
            return marker.owner, OwnerEvidence.CONFIRMED
        eligible = [a for a in self.anchors.values() if a.state != AnchorState.CLOSED
                    and self.current_connection(a)
                    and a.started <= frame.midpoint and self.owner_allowed(a.track_id, detection)
                    and (a.state == AnchorState.ACTIVE or bool(detection.owner_lanes))]
        if not detection.owner_lanes:
            eligible = [a for a in eligible if abs(a.exit_lane-detection.lane) <= 1]
        if len(eligible) == 1:
            return eligible[0].track_id, OwnerEvidence.CONFIRMED
        # Multiple connected anchors are ambiguous, never resolved by dict age.
        return None, OwnerEvidence.UNKNOWN

    def refresh(self, frame, detections=None):
        if frame.sequence == self.last_frame_sequence:
            return
        if self.last_frame_time is not None and frame.midpoint > self.last_frame_time:
            self.periods.append(frame.midpoint-self.last_frame_time)
        self.last_frame_sequence, self.last_frame_time = frame.sequence, frame.midpoint
        ribbon_lanes = {}
        for a in self.anchors.values():
            if a.state == AnchorState.CLOSED:
                continue
            if a.exit_lane not in ribbon_lanes:
                ribbon_lanes[a.exit_lane] = ribbon_at_judgement(frame.image, self.engine.calibration, a.exit_lane)
            if self.current_connection(a) and ribbon_lanes[a.exit_lane]:
                self.note_evidence(a.track_id, frame)
            elif a.last_evidence is None or frame.midpoint-a.last_evidence > self.coast_budget:
                if a.state != AnchorState.QUIESCENT:
                    a.state = AnchorState.QUIESCENT
                    self.trace.add('tap_hold_anchor', time=frame.midpoint, track=a.track_id,
                                   state='quiescent', reason='no-current-ribbon-evidence')
        if detections is None:
            from .holds import detect_hold_tails
            detections = detect_hold_tails(frame.image, self.engine.calibration, self.engine.config)
        self.tracker.associate_detections(detections, frame)
        self.bind_flicks(frame)

    def eligible(self, marker, now):
        return self.eligibility_reason(marker, now) is None

    def eligibility_reason(self, marker, now):
        anchor = self.anchors.get(marker.owner)
        track = self.engine.tracks.get(marker.owner)
        if anchor is None or anchor.state == AnchorState.CLOSED or track is None:
            return 'owner-unconfirmed-or-closed'
        if not self.current_connection(anchor):
            return 'superseded-head-connection'
        if track.state != TrackState.HOLDING or not marker.stable(self.engine.config, now, self.engine.calibration.trigger_progress):
            return 'owner-state-or-motion-unqualified'
        if now-marker.last_seen_time > self.coast_budget:
            return 'gold-observation-age-exceeded'
        if self.last_frame_sequence-marker.last_seen_frame > 2:
            return 'gold-observation-frame-budget-exceeded'
        rejected = self.tracker.rejected.get(marker.marker_id)
        if rejected is not None and rejected[0] >= marker.last_seen_frame:
            return 'contradictory-owner-evidence'
        return None

    def linked(self, left_id, right_id):
        if left_id == right_id:
            return False
        a, b = self.anchors.get(left_id), self.anchors.get(right_id)
        if a is None or b is None or a.state == AnchorState.CLOSED or b.state == AnchorState.CLOSED:
            return False
        ta, tb = self.engine.tracks.get(left_id), self.engine.tracks.get(right_id)
        return bool(ta and tb and ta.linked_partner_id == right_id and tb.linked_partner_id == left_id)

    def bind_flicks(self, frame):
        for track in self.engine.tracks.values():
            if track.gesture not in FLICK_GESTURES or not track.observations:
                continue
            if track.state in {TrackState.RELEASED, TrackState.LOST}:
                continue
            if self.flick_frames.get(track.track_id) == frame.sequence:
                continue
            self.flick_frames[track.track_id] = frame.sequence
            last = track.observations[-1]
            if frame.midpoint-last.timestamp > self.coast_budget:
                continue
            if (len(track.observations) < 3 or track.speed <= self.engine.config.min_downward_progress
                    or last.progress < .30):
                continue
            if any(a.end_flick_id == track.track_id and a.state != AnchorState.CLOSED
                   for a in self.anchors.values()):
                continue  # verified physical binding survives later occlusion
            available = [a for a in self.anchors.values() if a.state != AnchorState.CLOSED
                         and self.current_connection(a) and a.last_evidence is not None
                         and frame.midpoint-a.last_evidence <= self.coast_budget]
            if not available:
                continue
            candidates = []
            from .holds import bonus_hold_ribbon_present
            from .tracking import project_to_polyline
            from .hold_topology import marker_evidence
            tangent = project_to_polyline(last.center, self.engine.calibration.lane_centerlines[track.lane])[2]
            if not bonus_hold_ribbon_present(frame.image, last.candidate, tangent):
                continue
            _, lanes, _, _ = marker_evidence(frame.image, last.center,
                max(last.candidate.box[2:])/2., track.lane, self.engine.calibration)
            for a in available:
                # A tail must visibly connect to this current ribbon. Mere lane
                # or old release-time equality is not terminal evidence.
                if a.last_evidence is not None and frame.midpoint-a.last_evidence <= self.coast_budget:
                    if a.exit_lane in lanes if lanes else a.exit_lane == track.lane:
                        candidates.append(a)
            if len(candidates) == 1:
                if candidates[0].end_flick_id != track.track_id:
                    candidates[0].end_flick_id = track.track_id
                    self.trace.add('tap_hold_flick_bound', time=frame.midpoint,
                                   owner=candidates[0].track_id, flick=track.track_id)
        self.flick_frames = {tid: seq for tid, seq in self.flick_frames.items()
                             if tid in self.engine.tracks}

    def completed_marker(self, event, completed, registry):
        a = self.anchors.get(event.owner_id)
        if a is None:
            return
        a.exit_lane = event.lane
        if event.marker_terminal:
            a.terminal_completed, a.terminal_marker = completed, event.marker_id
        self.finalize(completed, registry)

    def completed_flick(self, event, receipt, registry):
        if receipt.up_call_finished is None or receipt.error:
            return
        for a in self.anchors.values():
            if a.end_flick_id == event.track_id:
                a.terminal_completed = receipt.up_call_finished
                self.finalize(receipt.up_call_finished, registry)

    def finalize(self, now, registry):
        for a in self.anchors.values():
            if a.state == AnchorState.CLOSED or a.terminal_completed is None or registry.has_pending(a.track_id):
                continue
            future = any(m.owner == a.track_id and m.marker_id != a.terminal_marker
                and m.observations and (self.eligible(m, now) or
                    (now-m.last_seen_time <= .6 and self.last_frame_sequence-m.last_seen_frame <= 2))
                and (m.predicted_hit(self.engine.calibration.trigger_progress) or -math.inf)
                > a.terminal_completed+.02 for m in self.tracker.markers.values())
            if future:
                continue
            a.state = AnchorState.CLOSED
            track = self.engine.tracks.get(a.track_id)
            if track is not None:
                track.state = TrackState.RELEASED
                track.hold_sustain_final_emitted = True
            self.trace.add('hold_note_anchor_done', time=now, track=a.track_id,
                           marker=a.terminal_marker, reason='physical-terminal-input-completed')
