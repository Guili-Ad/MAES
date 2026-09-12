"""Shared head identity validation; does not choose Tap/Hold timing."""
from __future__ import annotations
import math


def identity_continuity_allowed(track, candidate, projection, frame):
    observations = list(track.observations)
    if not observations:
        return True
    last = observations[-1]
    recent = observations[-4:]
    if (len(recent) >= 2 and frame.midpoint - recent[0].timestamp >= .085
            and min(o.progress for o in recent) >= .5
            and max(o.progress for o in recent) - min(o.progress for o in recent) < .007
            and projection.progress - last.progress > .012):
        # A stationary score glyph cannot turn into a falling note merely
        # because the note passes its position. Keep the moving identity.
        return False
    old_area = last.candidate.box[2] * last.candidate.box[3]
    new_area = candidate.box[2] * candidate.box[3]
    if (len(observations) >= 3 and last.progress >= .35
            and new_area < old_area * .4
            and track.speed > .1):
        # Perspective grows a head; a sudden tiny inner component/letter is
        # missing evidence, not a new position measurement of that head.
        return False
    if (len(observations) >= 3 and .35 <= last.progress < .85 and track.speed > .1
            and new_area < old_area*.7 and projection.distance > max(12., min(last.candidate.box[2:])*.25)):
        # A shrinking blob off the note's radial corridor is not its centre.
        return False
    return True


def unique_head_candidates(entries, trace, frame):
    """Collapse concentric inner/outer components, never neighbouring circles."""
    kept = []
    for candidate, projection in sorted(entries, key=lambda e: e[0].box[2]*e[0].box[3], reverse=True):
        x, y, w, h = candidate.box
        duplicate = None
        for outer, _ in kept:
            ox, oy, ow, oh = outer.box
            intersection = max(0, min(x+w, ox+ow)-max(x, ox)) * max(0, min(y+h, oy+oh)-max(y, oy))
            if (intersection >= .9*w*h and math.dist(candidate.center, outer.center) <= .25*min(ow, oh)
                    and max(ow, oh) <= 2.1*min(ow, oh)):
                duplicate = outer
                break
        if duplicate is None:
            kept.append((candidate, projection))
        else:
            trace.add('head_component_merged', frame=frame.sequence, time=frame.midpoint,
                      lane=projection.lane, kept=duplicate.box, removed=candidate.box)
    return kept
