"""Ordinary tap continuity. Hold/bonus/flick ownership stays in its own policy."""
from .models import FLICK_GESTURES, NoteGesture, TrackState


def ordinary_tap(track):
    return (track.gesture == NoteGesture.TAP and not track.bonus_star
            and not track.hold_evidence_frames
            and not any(g in FLICK_GESTURES
                        for g in track.direction_evidence))


def discontinuity(track, progress, timestamp):
    """A long nearly-stationary gap is not a flight measurement.

    No minimum inter-note interval and no cancellation based on combo/count.
    Short dropouts survive; a mature head that travels during a longer gap
    also survives. Thresholds are deliberately looser than the 1--2 frame
    dropout budget, but reject Test2's 364--968 ms stationary anchors.
    """
    if not ordinary_tap(track) or not track.observations:
        return None
    values = [(o.timestamp, o.progress) for o in track.observations]
    values.append((timestamp, progress))
    for (ta, pa), (tb, pb) in zip(values, values[1:]):
        if tb - ta > .24 and pb - pa < .075:
            return 'stationary-gap'
    return None


def coastable_tap(track, config, *, now=None, sequence=None):
    """Ordinary taps with healthy forward motion may coast through occlusion.

    The judgement text tints and fragments centre-lane pixels; the track's own
    fitted velocity still predicts the arrival, so scheduling may proceed on
    prediction for a bounded age instead of declaring the head lost. Bonus,
    hold, flick and stationary tracks are not eligible; the speed and span
    gates reject static glyphs and short noise fragments.
    """
    if track is None or not config.coast_enabled or track.bonus_star:
        return False
    if track.gesture != NoteGesture.TAP or track.hold_evidence_frames:
        return False
    if any(g in FLICK_GESTURES for g in track.direction_evidence):
        return False
    observations = list(track.observations)
    if len(observations) < 2:
        return False
    last = observations[-1]
    if last.progress < config.coast_min_progress or last.progress >= 0.95:
        return False
    if track.speed < config.coast_min_speed:
        return False
    if last.progress - observations[-2].progress < 0.0:
        return False
    if last.progress - observations[0].progress < 0.02:
        return False
    if now is not None and now - last.timestamp > config.coast_max_age_ms / 1000.0:
        return False
    return True


def late_birth_ready(track, project):
    """Late ordinary births need an actual forward, in-corridor trajectory.

    Early tracked heads keep their existing scheduling path. A score particle
    first seen near the line cannot become an urgent tap after a backward
    jitter plus one jump onto a real circle (Test2 replay's duplicate bursts).
    """
    if not ordinary_tap(track) or not track.observations:
        return True
    observations = list(track.observations)
    if min(o.progress for o in observations) < .50:
        return True
    distinct = []
    for o in observations:
        if distinct and o.candidate.box == distinct[-1].candidate.box and o.center == distinct[-1].center:
            continue
        distinct.append(o)
    recent = distinct[-3:]
    # A repeated captured frame is missing evidence, not proof of a fake.
    # Reject observed backward jitter, not an otherwise valid two-point fit.
    if any(b.progress-a.progress < -.006 for a, b in zip(recent, recent[1:])):
        return False
    for o in recent:
        projection = project(o.candidate)
        if (projection is None or projection.lane != track.lane
                or projection.distance > min(16., min(o.candidate.box[2:])*.25)):
            return False
    return True


def retire_bonus_fragments(tracks, stars, frame, project, trace):
    """Retire only uncommitted ordinary fragments under a recognized star.

    Keep the existing bonus birth/evidence path unchanged, particularly its
    ribbon-to-hold promotion. No transfer of motion or hold state is needed.
    """
    for star in stars:
        projection = project(star)
        if projection is None:
            continue
        for track in tracks.values():
            if (not ordinary_tap(track) or track.state != TrackState.APPROACHING
                    or track.action_executed or len(track.observations) != 1
                    or track.lane != projection.lane):
                continue
            last = track.observations[-1]
            dt = frame.midpoint - last.timestamp
            if not 0 <= dt <= .20 or not -.01 <= projection.progress - last.progress <= .10:
                continue
            # The old centre must lie within the star's swept head footprint;
            # nearby independent heads are not retired by a time cooldown.
            x, y, w, h = star.box
            if not (x - 4 <= last.center[0] <= x + w + 4
                    and y - h*.6 <= last.center[1] <= y + h + 4):
                continue
            track.state = TrackState.LOST
            trace.add('tap_identity_retired', time=frame.midpoint, frame=frame.sequence,
                      track=track.track_id, reason='ordinary-fragment-of-bonus',
                      old_box=last.candidate.box, bonus_box=star.box)
