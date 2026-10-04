"""Shared head identity validation; does not choose Tap/Hold timing."""
from __future__ import annotations
import math
from .models import NoteGesture, TrackState


def stationary_from_birth(track):
    """Persist an origin-specific HUD verdict, never a recent freeze verdict.

    The original first samples are checked while still available. Once the
    ring buffer has lost the birth history, a late plateau cannot manufacture
    this verdict. A previously moving real head may therefore freeze briefly
    and resume on exactly the same identity.
    """
    if not getattr(track, 'point_mode', False):
        return False
    if getattr(track, 'point_static_origin', False):
        return True
    observations = list(track.observations)
    if len(observations) < 3:
        return False
    born = track.first_seen_time
    if born is not None and observations[0].timestamp > born + 1e-6:
        return False
    # Fast capture can need six samples to span the evidence budget. Choose
    # the earliest birth prefix reaching it, never a later recent plateau.
    ending = next((i for i in range(2, len(observations))
                   if observations[i].timestamp-observations[0].timestamp >= .085), None)
    if ending is None:
        return False
    initial = observations[:ending+1]
    if (min(o.progress for o in initial) < .35
            or max(o.progress for o in initial)-min(o.progress for o in initial) >= .007):
        return False
    extent = min(min(o.candidate.box[2:]) for o in initial)
    if max(math.dist(initial[0].center, o.center) for o in initial) > max(2., extent*.04):
        return False
    track.point_static_origin = True
    return True


def identity_continuity_allowed(track, candidate, projection, frame):
    observations = list(track.observations)
    if not observations:
        return True
    last = observations[-1]
    if stationary_from_birth(track):
        # A real moving head can initially differ by less than one pixel from
        # a score glyph (the recorded .0017-progress step). Waiting for a big
        # progress jump lets that glyph absorb the whole flight. Preserve exact
        # repeats for the stationary identity, but leave every changed contour
        # available to a new/healthy physical owner rather than deleting it.
        return (candidate.box == last.candidate.box
                and candidate.center == last.candidate.center)
    recent = observations[-4:]
    if (not getattr(track, 'point_mode', False)
            and len(recent) >= 2 and frame.midpoint - recent[0].timestamp >= .085
            and min(o.progress for o in recent) >= .35
            and max(o.progress for o in recent) - min(o.progress for o in recent) < .007
            and projection.progress - last.progress > .012):
        # A stationary score glyph or upper blue stage decoration cannot turn
        # into a falling note merely because the note passes its position.
        # This rejects the old identity, never the moving candidate itself.
        return False
    old_area = last.candidate.box[2] * last.candidate.box[3]
    new_area = candidate.box[2] * candidate.box[3]
    w, h = candidate.box[2:]
    if (len(observations) >= 3 and .35 <= last.progress < .985 and track.speed > .1
            and not track.flick and candidate.variant != 'flick'
            and max(w, h) > 2.1 * min(w, h)
            and projection.distance > max(12., min(last.candidate.box[2:])*.25)
            and sum(.72 <= o.candidate.box[2] / max(o.candidate.box[3], 1) <= 1.40
                    for o in observations[-3:]) >= 2):
        # The recorded 76x34 score/ribbon fragment retained 85% of the
        # previous head's area, so a size-collapse guard could not catch it.
        # A flat, off-corridor blob is not a confirmed round head measurement.
        # Keep the healthy head fit; the unmatched candidate may still form
        # its own track. In-corridor clipping is deliberately unaffected.
        return False
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


def duplicate_head_evidence(shadow, real, frame):
    """Prove a corrupt concentric identity; never infer by timing alone.

    This head-only contract can be used by Tap or HoldStart orchestration.
    It neither chooses an owner nor changes hold lifecycle/route state. A
    detached ribbon fragment or neighbouring dense head yields no proof.
    """
    if (shadow.track_id == real.track_id or shadow.lane != real.lane
            or shadow.gesture not in {NoteGesture.TAP, NoteGesture.HOLD_START}
            or real.gesture not in {NoteGesture.TAP, NoteGesture.HOLD_START}
            or ((shadow.bonus_star or real.bonus_star)
                and not (getattr(shadow, 'point_mode', False) and getattr(real, 'point_mode', False)))
            or shadow.flick or real.flick
            or shadow.tap_input_started is not None
            or shadow.state in {TrackState.HOLDING, TrackState.RELEASED, TrackState.LOST}
            or real.state in {TrackState.RELEASED, TrackState.LOST}
            or len(shadow.observations) < 3 or len(real.observations) < 3):
        return None
    a, b = shadow.observations[-1], real.observations[-1]
    if a.frame_sequence != frame.sequence or b.frame_sequence != frame.sequence:
        return None
    if not .35 <= a.progress or abs(a.progress-b.progress) > .025:
        return None
    ax, ay, aw, ah = a.candidate.box
    bx, by, bw, bh = b.candidate.box
    overlap = max(0, min(ax+aw, bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    if (overlap < .9 * min(aw*ah, bw*bh)
            or math.dist(a.center,b.center) > .2*min(aw,ah,bw,bh)):
        return None

    def collapsed(track):
        observations = list(track.observations)
        return any(after.progress >= .35
                   and after.candidate.box[2]*after.candidate.box[3]
                   < .35*before.candidate.box[2]*before.candidate.box[3]
                   for before, after in zip(observations, observations[1:]))

    recent = list(real.observations)[-3:]
    moving = all(after.progress-before.progress > .002
                 and 0 < after.frame_sequence-before.frame_sequence <= 3
                 for before, after in zip(recent, recent[1:]))
    if moving and collapsed(shadow) and not collapsed(real):
        return 'contained-contour-with-prior-size-collapse'
    if (getattr(shadow, 'point_mode', False) and getattr(real, 'point_mode', False)
            and moving and stationary_from_birth(shadow) and not stationary_from_birth(real)):
        return 'contained-contour-with-stationary-origin'
    return None
