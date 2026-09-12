"""Dependency-aware TAP batches. Different deadlines never create a chord."""
from .models import NoteGesture
from .tap_tracking import physical_order, confirmed_tap_motion


def due_tap_batches(pending, now, engine=None):
    events = [e for e in pending if e.gesture == NoteGesture.TAP]
    units = {}
    membership = {}
    for event in events:
        key = event.tap_group_id or event.event_id
        units.setdefault(key, []).append(event)
        membership[event.event_id] = key
    dependencies = {key: set() for key in units}
    lane_index: dict[int, list] = {}
    for event in events:
        lane_index.setdefault(event.lane, []).append(event)
    current_sequence = getattr(engine, "last_frame_sequence", None) if engine is not None else None
    for lane, lane_events in lane_index.items():
        track_events = {e.track_id: e for e in lane_events}
        tracks = [engine.tracks.get(e.track_id) for e in lane_events] if engine else []
        if engine is not None:
            # Only tracks whose visual observations are still current may order
            # other taps.  A stale fragment (judgement-text or effect residue)
            # must never hold a confirmed note behind its drifting deadline.
            chain = [
                event for event, track in zip(lane_events, tracks)
                if track is not None
                and track.observations
                and (current_sequence is None
                     or current_sequence - track.observations[-1].frame_sequence <= 3)
            ]
            chain_tracks = [engine.tracks.get(event.track_id) for event in chain]
        else:
            chain = list(lane_events)
            chain_tracks = []
        ordered = physical_order(chain_tracks) if chain_tracks and all(confirmed_tap_motion(t) for t in chain_tracks) else None
        sequence = ([track_events[t.track_id] for t in ordered] if ordered else
                    sorted(chain, key=lambda e: (e.deadline, e.event_id)))
        for previous, current in zip(sequence, sequence[1:]):
            before, after = membership[previous.event_id], membership[current.event_id]
            if before != after:
                dependencies[after].add(before)
    completed = set()
    result = []
    while True:
        ready = [key for key, values in units.items() if key not in completed
                 and dependencies[key] <= completed and all(e.deadline <= now for e in values)]
        if not ready:
            # Two mis-associated chord histories can imply opposite orders on
            # their two lanes. Break only a closed, fully overdue cycle; never
            # bypass an actual future predecessor or split either chord.
            remaining = set(units) - completed

            def reachable(start):
                found, todo = set(), [start]
                while todo:
                    key = todo.pop()
                    if key in found:
                        continue
                    found.add(key)
                    todo.extend((dependencies[key] & remaining) - found)
                return found

            reach = {key: reachable(key) for key in remaining}
            for key in sorted(remaining):
                cycle = {other for other in reach[key] if key in reach[other]}
                if (len(cycle) > 1 and all(dependencies[k] <= completed | cycle for k in cycle)
                        and all(e.deadline <= now for k in cycle for e in units[k])):
                    ready = list(cycle)
                    trace = getattr(engine, 'tap_trace', None)
                    if trace is not None:
                        trace.add('tap_order_conflict', time=now, groups=sorted(cycle), fallback='due-deadline-order')
                    break
            if not ready:
                break
        key = min(ready, key=lambda key: (min(e.deadline for e in units[key]), key))
        result.append(units[key])
        completed.add(key)
    return result
