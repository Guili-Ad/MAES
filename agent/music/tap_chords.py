"""Only TAP/TAP groups live here. Persistent hold links are untouched."""
from __future__ import annotations

from dataclasses import dataclass, replace
from .models import NoteGesture, TrackState
from .tap_identity import discontinuity


@dataclass
class TapChord:
    group_id: str
    members: tuple[int, int]
    deadline: float
    frozen: bool = False


def valid_tap_pair(left, right, sequence: int) -> bool:
    return bool(left is not None and right is not None
                and left.gesture == right.gesture == NoteGesture.TAP
                and left.lane != right.lane
                and left.linked_partner_id == right.track_id
                and right.linked_partner_id == left.track_id
                and all(t.state not in {TrackState.RELEASED, TrackState.LOST}
                        and t.tap_input_started is None and t.predicted_hit_time is not None
                        and t.observations and sequence - t.observations[-1].frame_sequence <= 2
                        for t in (left, right)))


def coherent_tap_predictions(left, right):
    # Same-flight arc members cannot legitimately disagree by 414 ms.
    # This is a group eligibility check, not a new timing correction. Each
    # valid solo event keeps its own latest prediction on rejection.
    return (abs(left.predicted_hit_time - right.predicted_hit_time) <= .12
            and all(discontinuity(t, t.observations[-1].progress,
                                  t.observations[-1].timestamp) is None for t in (left, right)))


class TapChordManager:
    def __init__(self, policy, trace):
        self.policy, self.trace = policy, trace
        self.groups: dict[str, TapChord] = {}

    def refine(self, events, tracks, now, sequence):
        taps = {e.track_id: e for e in events if e.gesture == NoteGesture.TAP}
        grouped = {}
        live_groups = set()
        for tid, event in taps.items():
            left = tracks.get(tid)
            right = tracks.get(left.linked_partner_id) if left is not None else None
            if not valid_tap_pair(left, right, sequence) or right.track_id not in taps:
                continue
            members = tuple(sorted((tid, right.track_id)))
            gid = 'tap-' + '-'.join(map(str, members))
            if gid in live_groups:
                continue
            existing = self.groups.get(gid)
            if not (existing is not None and existing.frozen) and not coherent_tap_predictions(left, right):
                # Log once per pair/refinement (not separately for each side).
                if tid == members[0]:
                    self.trace.add('chord_rejected', time=now, group=gid,
                                   reason='incoherent-predictions',
                                   raw_hits=[left.predicted_hit_time, right.predicted_hit_time])
                continue
            live_groups.add(gid)
            member_events = [taps[m] for m in members]
            group = self.groups.get(gid)
            if group is None:
                group = TapChord(gid, members, min(e.deadline for e in member_events),
                                 any(e.tap_frozen for e in member_events))
                self.groups[gid] = group
            old = (group.deadline, group.frozen)
            if group.deadline <= now + 0.02:
                group.frozen = True
            if not group.frozen:
                group.deadline = (left.predicted_hit_time + right.predicted_hit_time) / 2. - self.policy.config.tap_action_advance_ms / 1000.
                group.frozen = group.deadline <= now + 0.02
            for member in member_events:
                grouped[member.event_id] = replace(member, deadline=group.deadline,
                                                   tap_group_id=gid, tap_frozen=group.frozen,
                                                   tap_reference_hit_time=group.deadline + self.policy.config.tap_action_advance_ms / 1000.)
            if old != (group.deadline, group.frozen) or any(e.tap_group_id != gid for e in member_events):
                self.trace.add('chord', time=now, group=gid, members=members,
                               deadline=group.deadline, frozen=group.frozen)
        for gid in set(self.groups) - live_groups:
            self.trace.add('chord_dissolved', time=now, group=gid)
            del self.groups[gid]
        refined = []
        for event in events:
            if event.gesture != NoteGesture.TAP:
                refined.append(event)
                continue
            track = tracks.get(event.track_id)
            if track is not None and track.state in {TrackState.LOST, TrackState.RELEASED}:
                self.trace.add('cancelled', time=now, event=event.event_id, reason='terminal-track')
                continue
            if track is not None and track.tap_input_started is not None:
                continue  # never re-enqueue a partially/fully dispatched input
            updated = grouped.get(event.event_id)
            if updated is None:
                frozen = event.tap_frozen or event.deadline <= now + 0.02
                updated = replace(event, tap_group_id=None, tap_frozen=frozen)
                if not frozen and track is not None and track.predicted_hit_time is not None:
                    hit = self.policy.hit_time(track)
                    updated = replace(updated, deadline=hit - self.policy.action_advance_ms(track) / 1000.,
                                      tap_reference_hit_time=hit)
                    updated = replace(updated, tap_frozen=updated.deadline <= now + 0.02)
            if event != updated:
                self.trace.add('refine', time=now, event=event.event_id,
                               raw_hit=track.predicted_hit_time if track else None,
                               before=event.deadline, deadline=updated.deadline,
                               group=updated.tap_group_id, frozen=updated.tap_frozen,
                               last_visual_time=track.observations[-1].timestamp if track and track.observations else None,
                               last_box=track.observations[-1].candidate.box if track and track.observations else None,
                               progress=track.observations[-1].progress if track and track.observations else None)
            refined.append(updated)
        return refined
