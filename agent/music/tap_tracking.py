"""Conservative tap-only replacement for ambiguous greedy associations."""
from __future__ import annotations

from functools import lru_cache
import math
from .models import FLICK_GESTURES, NoteGesture, TrackState
from .head_identity import identity_continuity_allowed
from .tap_identity import discontinuity


def confirmed_tap_motion(track):
    if (track.gesture != NoteGesture.TAP or track.bonus_star or track.hold_evidence_frames
            or len(track.observations) < 3):
        return False
    recent = list(track.observations)[-3:]
    return all(b.progress - a.progress > .002 and 0 < b.frame_sequence - a.frame_sequence <= 3
               for a, b in zip(recent, recent[1:]))


def physical_order(tracks):
    """Recover front-to-back order at a *shared* observation time, not by ID."""
    if not tracks or any(not t.observations for t in tracks):
        return None
    shared = set(o.frame_sequence for o in tracks[0].observations)
    for track in tracks[1:]:
        shared.intersection_update(o.frame_sequence for o in track.observations)
    if not shared:
        return None
    sequence = max(shared)
    positions = {t.track_id: next(o.progress for o in reversed(t.observations) if o.frame_sequence == sequence)
                 for t in tracks}
    ordered = sorted(tracks, key=lambda t: -positions[t.track_id])
    if any(abs(positions[a.track_id] - positions[b.track_id]) < 1e-5 for a, b in zip(ordered, ordered[1:])):
        return None
    return ordered


def retire_converged_shadows(tracks, frame, trace):
    """Cancel a corrupted duplicate only with containment + contour history.

    This does NOT discard close-in-time notes: two regular moving heads or
    uncontained neighbouring circles always survive. Executed input can only
    be the survivor, never the cancelled track.
    """
    current = [t for t in tracks.values() if t.gesture == NoteGesture.TAP
               and t.state in {TrackState.APPROACHING, TrackState.TAP_PENDING}
               and not t.bonus_star and not t.hold_evidence_frames and t.linked_partner_id is None
               and len(t.observations) >= 4 and t.observations[-1].frame_sequence == frame.sequence
               and t.observations[-1].progress >= .78
               and not any(g in FLICK_GESTURES for g in t.direction_evidence)]

    def collapsed(track):
        obs = list(track.observations)
        return any(b.progress >= .35 and b.candidate.box[2] * b.candidate.box[3]
                   < .35 * a.candidate.box[2] * a.candidate.box[3] for a, b in zip(obs, obs[1:]))

    by_lane: dict[int, list] = {}
    for track in current:
        by_lane.setdefault(track.lane, []).append(track)

    for lane_tracks in by_lane.values():
        for index, left in enumerate(lane_tracks):
            for right in lane_tracks[index + 1:]:
                if left.state == TrackState.LOST or right.state == TrackState.LOST:
                    continue
                a, b = left.observations[-1], right.observations[-1]
                if abs(a.progress - b.progress) > .025:
                    continue
                ax, ay, aw, ah = a.candidate.box
                bx, by, bw, bh = b.candidate.box
                overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
                if overlap < .9 * min(aw * ah, bw * bh) or math.dist(a.center, b.center) > .2 * min(aw, ah, bw, bh):
                    continue
                for shadow, real in ((left, right), (right, left)):
                    # A scheduled or physically started input may never be
                    # cancelled as a "shadow": its event is already in (or has
                    # left) the dispatch queue and cancelling the track would
                    # silently drop a real note.
                    if shadow.tap_input_started is not None or shadow.action_event_id:
                        continue
                    if collapsed(shadow) and not collapsed(real) and confirmed_tap_motion(real):
                        shadow.state = TrackState.LOST
                        trace.add('duplicate_shadow', frame=frame.sequence, time=frame.midpoint,
                                  cancelled=shadow.track_id, kept=real.track_id, lane=real.lane,
                                  reason='contained-contour-with-prior-size-collapse')
                        break


def associate_taps(tracks, entries, frame, config, safe_candidate, trace, owned=None):
    """Return entry-index -> existing track; leave births to the legacy caller.

    First reproduce the old greedy result exactly. Only wholly ordinary,
    mature connected conflict components may replace that result. Hold/bonus/
    flick candidates and immature tracks keep their original assignments.
    """
    costs = {}
    by_candidate = {}
    by_track = {}
    lookup = {t.track_id: t for t in tracks}
    owned = owned or {}
    reserved = set(owned.values())
    safe = {i: safe_candidate(*entry) for i, entry in enumerate(entries)}
    rejected = set()
    for index, (candidate, projection) in enumerate(entries):
        if index in owned:
            continue
        for track in tracks:
            if track.track_id in reserved:
                continue
            if not track.observations or (candidate.variant == 'bonus_star') != track.bonus_star:
                continue
            # Flick candidates may only extend flick tracks and vice versa; a
            # colour-classified sprite can never hijack an ordinary tap track.
            if (candidate.variant == 'flick') != track.flick:
                continue
            if not identity_continuity_allowed(track, candidate, projection, frame):
                continue
            previous = track.observations[-1]
            expected = previous.progress + max(0., track.speed) * max(0., frame.midpoint - previous.timestamp)
            residual = projection.progress - expected
            if not (-0.04 <= residual <= max(0.04, config.association_progress_delta)
                    and projection.progress - previous.progress >= -0.03):
                continue
            reason = discontinuity(track, projection.progress, frame.midpoint) if safe[index] else None
            if reason:
                if track.track_id not in rejected:
                    trace.add('tap_association_rejected', time=frame.midpoint, frame=frame.sequence,
                              track=track.track_id, reason=reason, box=candidate.box,
                              observations=[(o.timestamp, o.progress) for o in track.observations])
                    rejected.add(track.track_id)
                continue
            costs[index, track.track_id] = abs(residual)
            by_candidate.setdefault(index, set()).add(track.track_id)
            by_track.setdefault(track.track_id, set()).add(index)
    unmatched = set(lookup) - reserved
    result = {index: lookup[tid] for index, tid in owned.items()}
    for index in range(len(entries)):
        if index in owned:
            continue
        eligible = by_candidate.get(index, set()) & unmatched
        if eligible:
            tid = min(eligible, key=lambda tid: (costs[index, tid], lookup[tid].missed_frames, tid))
            result[index] = lookup[tid]
            unmatched.remove(tid)

    visited = set()
    for seed in by_candidate:
        if seed in visited:
            continue
        candidates, ids = {seed}, set()
        todo = [seed]
        while todo:
            index = todo.pop()
            for tid in by_candidate[index] - ids:
                ids.add(tid)
                additions = by_track[tid] - candidates
                candidates.update(additions)
                todo.extend(additions)
        visited.update(candidates)
        if len(ids) < 2 or len(ids) > 16 or len(candidates) > 16:
            continue
        members = [lookup[tid] for tid in ids]
        if any(not confirmed_tap_motion(t)
               or t.state not in {TrackState.APPROACHING, TrackState.TAP_PENDING}
               or any(g in FLICK_GESTURES for g in t.direction_evidence)
               for t in members):
            continue
        if not all(safe_candidate(*entries[index]) for index in candidates):
            continue
        ordered = physical_order(members)
        if ordered is None:
            continue
        positions = sorted(candidates)  # entries are already sorted by progress

        @lru_cache(None)
        def solve(i, j):
            if i == len(ordered) or j == len(positions):
                return (0, 0., ())
            options = [solve(i + 1, j), solve(i, j + 1)]
            tid, index = ordered[i].track_id, positions[j]
            if (index, tid) in costs:
                n, cost, pairs = solve(i + 1, j + 1)
                options.append((n + 1, cost + costs[index, tid], ((index, tid),) + pairs))
            return min(options, key=lambda value: (-value[0], value[1], value[2]))

        pairs = solve(0, 0)[2]
        before = [(index, result[index].track_id) for index in positions if index in result]
        if len(pairs) < len(before):
            # Crossed/ambiguous history is not reliable enough to create a new
            # identity by throwing away an otherwise admissible old match.
            trace.add('association_fallback', frame=frame.sequence, lane=ordered[0].lane,
                      reason='order_would_reduce_matches')
            continue
        for index in positions:
            result.pop(index, None)
        for index, tid in pairs:
            result[index] = lookup[tid]
        if before != list(pairs):
            trace.add('association', frame=frame.sequence, time=frame.midpoint,
                      lane=ordered[0].lane, before=before, after=pairs)
    return result
