"""Ordinary Tap physical evidence, separate from timing and hold ownership.

A component rectangle can contain a real head plus a judgement glyph. It is
not the physical head's contour. Current pixels have three outcomes: a proven
white-core/teal-disc head, a proven narrow HUD stroke, or unknown. Unknown is
never absence. Aliasing additionally needs actual shared-contour evidence or
bidirectional motion continuity; deadlines and temporal proximity are unused.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .components import connected_components
from .models import FLICK_GESTURES, MusicCandidate, NoteGesture, TrackState


@dataclass(frozen=True)
class TapContourEvidence:
    verdict: str  # positive / negative / unknown
    candidate: MusicCandidate | None = None
    reason: str = ''


# Only the current immutable captured frame is cached. Never carry visual
# results across sequences (including pause/restart), and cap candidate keys.
# The runtime processes one controller synchronously; this is not an input
# controller or a cross-frame recognition cache.
_evidence_frame = None
_evidence_cache = {}


def _ordinary(track):
    return (track.gesture == NoteGesture.TAP and not track.bonus_star
            and not track.flick and not track.hold_evidence_frames
            and not any(g in FLICK_GESTURES for g in track.direction_evidence))


def _masks(patch):
    b, g, r = (patch[..., i].astype(np.int16) for i in range(3))
    teal = (b >= 70) & (g >= 110) & (g > r+35) & (b > r+25)
    minimum, maximum = np.minimum(np.minimum(b, g), r), np.maximum(np.maximum(b, g), r)
    white = (minimum >= 190) & (maximum-minimum < 55)
    return teal, white


def _disc_metrics(teal, white, cx, cy, rx, ry):
    yy, xx = np.ogrid[:teal.shape[0], :teal.shape[1]]
    d = ((xx-cx)/max(rx, 1.))**2 + ((yy-cy)/max(ry, 1.))**2
    core, disc = d < .20**2, (d > .35**2) & (d < .72**2)
    if not core.any() or not disc.any():
        return 0., 0.
    return float(white[core].mean()), float(teal[disc].mean())


def contour_evidence(candidate, frame):
    global _evidence_frame, _evidence_cache
    if _evidence_frame is not frame:
        _evidence_frame, _evidence_cache = frame, {}
    key = (candidate.box, candidate.center, candidate.variant)
    evidence = _evidence_cache.get(key)
    if evidence is None:
        evidence = _inspect_contour(candidate, frame)
        if len(_evidence_cache) < 96:
            _evidence_cache[key] = evidence
            if evidence.verdict == 'positive' and evidence.candidate is not None:
                normalized = evidence.candidate
                normalized_key = (normalized.box, normalized.center, normalized.variant)
                _evidence_cache[normalized_key] = TapContourEvidence('positive', normalized, evidence.reason)
    return evidence


def _inspect_contour(candidate, frame):
    """Inspect only this candidate's current pixels, never a spatial blacklist.

    The white core may be covered by judgement text. A round teal footprint
    without a visible core is therefore unknown, not negative. Negative is
    deliberately limited to an observed isolated narrow coloured stroke.
    Large heads use exactly the same evidence as small ones.
    """
    x, y, w, h = candidate.box
    image = frame.image
    if (candidate.variant or min(w, h) < 8 or max(w, h) > 192
            or not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] < 3 or x < 0 or y < 0
            or x+w > image.shape[1] or y+h > image.shape[0]):
        return TapContourEvidence('unknown', reason='pixels-unavailable-or-independent-variant')
    patch = image[y:y+h, x:x+w, :3]
    teal, white = _masks(patch)
    if not teal.any():
        return TapContourEvidence('unknown', reason='ordinary-colour-unavailable')
    # A known round component requires no expensive local seed search and no
    # coordinate rewrite. Avoid blessing the 122x84 compound HUD rectangle as
    # an ellipse when its actual disc is a smaller, shifted circle inside it.
    if .80 <= w/h <= 1.25:
        core, disc = _disc_metrics(teal, white, (w-1)/2., (h-1)/2., w/2., h/2.)
        if core >= .25 and disc >= .65:
            return TapContourEvidence('positive', candidate, 'white-core-teal-disc')

    # A compact white nucleus surrounded by teal proves a head even when its
    # connected component includes teal judgement strokes. Search is bounded
    # to six seeds and radii at 2px resolution in this one candidate rectangle.
    # There is no full-screen detector, cross-frame cache or new input policy.
    small_white = white[::2, ::2]
    seeds = []
    for (sx, sy, sw, sh), count in connected_components(small_white, 2):
        if min(sw, sh) < 2 or max(sw, sh) > min(w, h)*.24:
            continue
        if not .55 <= sw/max(sh, 1) <= 1.8 or count/(sw*sh) < .4:
            continue
        cx, cy = sx*2+(sw-1), sy*2+(sh-1)
        seeds.append((math.hypot(cx-(w-1)/2., cy-(h-1)/2.), cx, cy, max(sw, sh)*2))
    choices = []
    for _, cx, cy, nucleus in sorted(seeds)[:6]:
        lower = max(8., nucleus*1.65)
        upper = min(96., nucleus*3.5, min(w, h)*.65)
        yy, xx = np.ogrid[:h, :w]
        distance2 = (xx-cx)**2+(yy-cy)**2
        best = None
        for radius in np.arange(lower, upper+.01, 2.):
            # A complete physical disc must fit in the observed rectangle.
            # Clipped or covered contours remain unknown, not rejected.
            if cx-radius*.9 < 0 or cy-radius*.9 < 0 or cx+radius*.9 >= w or cy+radius*.9 >= h:
                continue
            core = distance2 < (.20*radius)**2
            disc = (distance2 > (.35*radius)**2) & (distance2 < (.72*radius)**2)
            outer = (distance2 > (.94*radius)**2) & (distance2 < (1.10*radius)**2)
            if not core.any() or not disc.any() or not outer.any():
                continue
            core_ratio, disc_ratio, outside = float(white[core].mean()), float(teal[disc].mean()), float(teal[outer].mean())
            if core_ratio < .35 or disc_ratio < .75 or outside > .25:
                continue
            # Disc evidence must surround the nucleus, not sit entirely on
            # one side like a teal font stroke next to a white letter.
            quadrants = [(xx >= cx) & (yy >= cy), (xx < cx) & (yy >= cy),
                         (xx >= cx) & (yy < cy), (xx < cx) & (yy < cy)]
            if any(not (disc & q).any() or teal[disc & q].mean() < .60 for q in quadrants):
                continue
            score = core_ratio+disc_ratio-outside
            if best is None or score > best[0]+1e-9:
                best = (score, radius)
        if best is None:
            continue
        radius = best[1]
        diameter = int(round(radius*2))
        box = (int(round(x+cx-radius+.5)), int(round(y+cy-radius+.5)), diameter, diameter)
        normalized = MusicCandidate(box, int(teal[distance2 < radius**2].sum()),
                                    float(teal[distance2 < radius**2].mean()),
                                    (float(x+cx+.5), float(y+cy+.5)))
        if not any(math.dist(normalized.center, previous.center) < min(diameter, previous.box[2])*.20
                   for previous in choices):
            choices.append(normalized)
    if len(choices) == 1:
        return TapContourEvidence('positive', choices[0], 'circle-inside-compound-component')
    if len(choices) > 1:
        # Two real heads in a merged component must not become one head.
        return TapContourEvidence('unknown', reason='multiple-physical-circles')
    rows, cols = np.nonzero(teal)
    coloured_w, coloured_h = int(cols.max()-cols.min()+1), int(rows.max()-rows.min()+1)
    if (min(coloured_w, coloured_h) <= max(coloured_w, coloured_h)*.24
            and max(coloured_w, coloured_h) >= min(w, h)*.65
            and int(teal.sum()) >= max(12, min(w, h))):
        return TapContourEvidence('negative', reason='isolated-narrow-hud-stroke')
    return TapContourEvidence('unknown', reason='partly-occluded-or-unproven-contour')


def normalize_tap_entries(entries, frame, project, trace, *, ordinary=None):
    """Normalize proven ordinary heads before association; preserve other types.

    ``ordinary(candidate, projection)`` is the engine's existing classification
    predicate, so hold ribbons and uncertain variants retain their own path.
    The original projection is preserved if normalization cannot be projected
    into the same lane/corridor. A negative finding removes this observation,
    never all future objects passing through its position.
    """
    result = []
    for candidate, projection in entries:
        if candidate.variant or (ordinary is not None and not ordinary(candidate, projection)):
            result.append((candidate, projection))
            continue
        evidence = contour_evidence(candidate, frame)
        if evidence.verdict == 'negative':
            trace.add('tap_contour_rejected', frame=frame.sequence, time=frame.midpoint,
                      lane=projection.lane, box=candidate.box, reason=evidence.reason)
            continue
        normalized = evidence.candidate
        if normalized is not None and normalized is not candidate:
            fitted = project(normalized)
            if (fitted is not None and fitted.lane == projection.lane
                    and fitted.distance <= min(16., min(normalized.box[2:])*.25)):
                trace.add('tap_contour_normalized', frame=frame.sequence, time=frame.midpoint,
                          lane=fitted.lane, old_box=candidate.box, new_box=normalized.box,
                          old_progress=projection.progress, progress=fitted.progress,
                          reason=evidence.reason)
                candidate, projection = normalized, fitted
        result.append((candidate, projection))
    return result


def _moving(observations):
    if len(observations) < 3:
        return False
    return all(b.progress-a.progress > .002 and 0 < b.frame_sequence-a.frame_sequence <= 3
               and 0 < b.timestamp-a.timestamp <= .20
               for a, b in zip(observations[-3:], observations[-3:][1:]))


def _line(observations, value):
    # Relative-time fitting also avoids long-running perf_counter cancellation.
    recent = observations[-3:]
    origin = recent[-1].timestamp
    times = [o.timestamp-origin for o in recent]
    values = [value(o) for o in recent]
    mean_t, mean_v = sum(times)/len(times), sum(values)/len(values)
    denominator = sum((t-mean_t)**2 for t in times)
    if denominator <= 1e-12:
        return None
    slope = sum((t-mean_t)*(v-mean_v) for t, v in zip(times, values))/denominator
    return lambda stamp: mean_v+slope*(stamp-origin-mean_t)


def _separated_history(a, b):
    shared = {o.frame_sequence: o for o in a}
    for right in b:
        left = shared.get(right.frame_sequence)
        if left is None or abs(left.timestamp-right.timestamp) > .002:
            continue
        # Any observed front/back separation is stronger than subsequent
        # contour convergence. Close, genuine same-lane runs retain identity.
        extent = min(*left.candidate.box[2:], *right.candidate.box[2:])
        if (abs(left.progress-right.progress) > .015
                and math.dist(left.center, right.center) > max(3., extent*.20)):
            return True
    return False


def _bridge_evidence(old, new, frame, new_contour):
    a, b = list(old.observations), list(new.observations)
    if _separated_history(a, b) or not _moving(a) or not _moving(b):
        return None
    last_old, first_new, last_new = a[-1], b[0], b[-1]
    if last_new.frame_sequence != frame.sequence or new_contour.verdict != 'positive':
        return None
    # Same currently observed physical circle is pixel proof, provided these
    # identities have never been simultaneously observed as separated heads.
    if last_old.frame_sequence == frame.sequence:
        old_contour = contour_evidence(last_old.candidate, frame)
        if old_contour.verdict != 'positive':
            return None
        extent = min(*old_contour.candidate.box[2:], *new_contour.candidate.box[2:])
        if math.dist(old_contour.candidate.center, new_contour.candidate.center) <= max(2., extent*.08):
            return 'same-observed-white-core-disc'
        return None
    if (not 0 < first_new.timestamp-last_old.timestamp <= .65
            or not .25 <= last_old.progress < .95
            or not last_old.progress < last_new.progress <= .985):
        return None
    old_extent, new_extent = min(last_old.candidate.box[2:]), min(new_contour.candidate.box[2:])
    if not .85*old_extent <= new_extent <= 2.5*old_extent:
        return None
    old_fit, new_fit = _line(a, lambda o: o.progress), _line(b, lambda o: o.progress)
    if old_fit is None or new_fit is None:
        return None
    # Two independent observed motion histories must identify the same flight
    # phase in BOTH directions. A later head merely close to an old deadline
    # does not meet this contract. No deadline participates in this check.
    if (max(abs(old_fit(o.timestamp)-o.progress) for o in b[-3:]) > .018
            or max(abs(new_fit(o.timestamp)-o.progress) for o in a[-3:]) > .018):
        return None
    old_xy = [_line(a, lambda o, axis=axis: o.center[axis]) for axis in (0, 1)]
    new_xy = [_line(b, lambda o, axis=axis: o.center[axis]) for axis in (0, 1)]
    if any(f is None for f in (*old_xy, *new_xy)):
        return None
    tolerance = min(8., old_extent*.14)
    if (math.dist(tuple(f(last_new.timestamp) for f in old_xy), new_contour.candidate.center) > tolerance
            or math.dist(tuple(f(last_old.timestamp) for f in new_xy), last_old.center) > tolerance):
        return None
    return 'white-core-disc-with-bidirectional-flight-continuity'


def reconcile_tap_identities(tracks, frame, config, trace):
    """Return shadow -> original owner aliases before not-yet-started dispatch.

    Only unambiguous physical aliases are committed. Queued is not sent:
    the retained owner keeps its original event ID, with current observations
    transferred for the engine to refit. The caller cancels only unstarted
    shadow events and refines the surviving original event normally. An
    already-started owner can prove an unstarted shadow is a re-recognition;
    its input, deadline and receipt are never revoked, changed or re-sent.
    Two started identities are never merged to hide an existing duplicate.
    """
    del config  # identity does not tune any timing or coasting parameter
    eligible = [t for t in tracks.values() if _ordinary(t) and t.observations
                and (t.state in {TrackState.APPROACHING, TrackState.TAP_PENDING}
                     or (t.state == TrackState.RELEASED and t.tap_input_started is not None
                         and 0 <= frame.midpoint-t.tap_input_started <= .25))
                and t.linked_partner_id is None and _moving(list(t.observations))]
    current = [t for t in eligible if t.observations[-1].frame_sequence == frame.sequence
               and t.tap_input_started is None and t.tap_input_completed is None]
    contours = {t.track_id: contour_evidence(t.observations[-1].candidate, frame) for t in current}
    pairs = []
    for new in current:
        if contours[new.track_id].verdict != 'positive':
            continue
        for old in eligible:
            if (old.track_id == new.track_id or old.lane != new.lane
                    or old.observations[0].timestamp >= new.observations[0].timestamp):
                continue
            reason = _bridge_evidence(old, new, frame, contours[new.track_id])
            if reason is not None:
                pairs.append((old, new, reason))
    old_counts, new_counts = {}, {}
    for old, new, _ in pairs:
        old_counts[old.track_id] = old_counts.get(old.track_id, 0)+1
        new_counts[new.track_id] = new_counts.get(new.track_id, 0)+1
    aliases = {}
    for old, new, reason in pairs:
        if old_counts[old.track_id] != 1 or new_counts[new.track_id] != 1:
            continue
        if old.track_id in aliases or new.track_id in aliases or old.state == TrackState.LOST:
            continue
        observations = {o.frame_sequence: o for o in old.observations}
        observations.update({o.frame_sequence: o for o in new.observations})
        old.observations.clear()
        old.observations.extend(sorted(observations.values(), key=lambda o: (o.timestamp, o.frame_sequence)))
        old.missed_frames = 0
        old.tap_contour_seen_time = max(old.tap_contour_seen_time or 0., new.tap_contour_seen_time or 0., frame.midpoint)
        new.state = TrackState.LOST
        aliases[new.track_id] = old.track_id
        trace.add('tap_physical_alias', frame=frame.sequence, time=frame.midpoint,
                  shadow=new.track_id, owner=old.track_id, lane=old.lane,
                  kept_event=old.action_event_id, shadow_event=new.action_event_id,
                  owner_input_started=old.tap_input_started,
                  reason=reason, contour=contours[new.track_id].candidate.box)
    return aliases
