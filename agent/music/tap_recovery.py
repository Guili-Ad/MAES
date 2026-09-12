"""Local, owner-only teal tap recovery across judgement exclusion rectangles.

This is not a candidate provider: no births, bonus/hold/flick classification,
or global mask changes. Missing/ambiguous pixels leave the old fit untouched.
"""
import math
import numpy as np
from .models import MusicCandidate, TrackState
from .tap_identity import ordinary_tap
from .vision import connected_components


def recover_masked_taps(tracks, frame, calibration, project):
    recovered = {}
    if frame.image is None or not calibration.exclusion_rois:
        return recovered
    for track in tracks.values():
        if (not ordinary_tap(track) or track.tap_input_started is not None
                or track.state not in {TrackState.APPROACHING, TrackState.TAP_PENDING}
                or len(track.observations) < 3):
            continue
        # Use pre-frame history: a clipped current box must not bias the search.
        prior = []
        for o in track.observations:
            if o.frame_sequence >= frame.sequence:
                continue
            if prior and o.candidate.box == prior[-1].candidate.box and o.center == prior[-1].center:
                continue
            prior.append(o)
        if len(prior) < 3:
            continue
        recent = prior[-3:]
        if not all(b.progress - a.progress > .002 and 0 < b.frame_sequence-a.frame_sequence <= 3
                   for a, b in zip(recent, recent[1:])):
            continue
        last = prior[-1]
        dt = frame.midpoint-last.timestamp
        if not 0 < dt <= .18 or not .70 <= last.progress <= .96:
            continue
        span = recent[-1].timestamp-recent[0].timestamp
        speed = (recent[-1].progress-recent[0].progress)/max(span, 1e-6)
        expected = last.progress + speed*dt
        if not .72 <= expected <= .985:
            continue
        # Local linear displacement only chooses the search box; the recovered
        # pixels are projected onto the unchanged calibrated curved lane.
        vx = (recent[-1].center[0]-recent[0].center[0])/max(span, 1e-6)
        vy = (recent[-1].center[1]-recent[0].center[1])/max(span, 1e-6)
        cx, cy = last.center[0]+vx*dt, last.center[1]+vy*dt
        extent = max(last.candidate.box[2:])
        if not any(ex < cx+extent/2 and ex+ew > cx-extent/2
                   and ey < cy+extent/2 and ey+eh > cy-extent/2
                   for ex, ey, ew, eh in calibration.exclusion_rois):
            continue
        radius = min(90, int(extent*.8+12))
        x0, y0 = max(0, int(cx)-radius), max(0, int(cy)-radius)
        crop = np.asarray(frame.image)[y0:min(calibration.height, int(cy)+radius),
                                       x0:min(calibration.width, int(cx)+radius), :3][::2, ::2]
        b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
        teal = (b >= 70) & (g >= 110) & (g > r+35) & (b > r+25)
        choices = []
        for (x, y, w, h), count in connected_components(teal, 60):
            size = max(w, h)*2
            if not .70*extent <= size <= 1.5*extent or not .83 <= w/max(h, 1) <= 1.20:
                continue
            if count/(w*h) < .48:
                continue
            # White core inside a saturated teal disc. Judgement gold rings,
            # empty teal rings, and generic solid particles fail this topology.
            yy, xx = np.ogrid[:h, :w]
            d = ((xx-(w-1)/2)/max(w/2, 1))**2 + ((yy-(h-1)/2)/max(h/2, 1))**2
            core = d < .20**2
            disc = (d > .35**2) & (d < .72**2)
            patch = crop[y:y+h, x:x+w]
            white = (patch.min(axis=2) >= 190) & (patch.max(axis=2)-patch.min(axis=2) < 55)
            if not core.any() or white[core].mean() < .25 or teal[y:y+h, x:x+w][disc].mean() < .65:
                continue
            box = (x0+x*2, y0+y*2, w*2, h*2)
            candidate = MusicCandidate(box, count*4, count/(w*h),
                                       (box[0]+w, box[1]+h))
            projection = project(candidate)
            if (projection is None or projection.lane != track.lane
                    or projection.distance > min(16., size*.20)
                    or abs(projection.progress-expected) > .045
                    or not last.progress+.002 < projection.progress < .985):
                continue
            # Never borrow the contour owned by another independently tracked
            # head (especially an already pressed head's fading visual).
            if any(other.track_id != track.track_id and other.lane == track.lane
                   and other.state not in {TrackState.LOST, TrackState.RELEASED}
                   and other.observations and frame.midpoint-other.observations[-1].timestamp <= .10
                   and math.dist(other.observations[-1].center, candidate.center) < size*.45
                   for other in tracks.values()):
                continue
            choices.append((candidate, projection))
        if len(choices) != 1:
            continue
        candidate, projection = choices[0]
        recovered[track.track_id] = (candidate, projection)
    return recovered
