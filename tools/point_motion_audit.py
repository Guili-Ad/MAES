"""Read-only video motion proxy audit using original decoded frame PTS.

Explicit paths/windows avoid date guessing. This does not reproduce live Maa
recognition or the omitted 250-ms-sampled live marker histories, and never
uses an input prediction as game judgement ground truth.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
from agent.music.holds import detect_hold_tails, bonus_hold_ribbon_present
from agent.music.models import MusicCalibrationData, MusicConfig
from agent.music.motion import weighted_slope
from agent.music.tracking import assign_lane
from agent.music.vision import detect_bonus_star_notes


def strict_metadata_ribbon(image, candidate, tangent, *, bilateral=False):
    """Only audit the old strict near/far topology, without fallback fan."""
    radius = max(candidate.box[2:])/2.
    if radius < 9:
        return False
    cx, cy = candidate.center
    margin = int(math.ceil(radius*4.5))
    x0, y0 = max(0, int(round(cx))-margin), max(0, int(round(cy))-margin)
    crop = image[y0:min(image.shape[0], int(round(cy))+margin+1),
                 x0:min(image.shape[1], int(round(cx))+margin+1), :3].astype(np.int16)
    yy, xx = np.indices(crop.shape[:2])
    dx, dy = xx+x0-cx, yy+y0-cy
    tx, ty = tangent
    signed_across = -dx*ty+dy*tx
    along, across = dx*tx+dy*ty, np.abs(signed_across)
    neutral = (crop.min(axis=2) >= 165) & (crop.max(axis=2)-crop.min(axis=2) <= 65)
    near = (along <= -radius*1.05)&(along >= -radius*2.40)&(across <= radius*.38)
    far = (along <= -radius*2.40)&(along >= -radius*4.20)&(across <= radius*.50)
    near_side = (along <= -radius*1.05)&(along >= -radius*2.40)&(across >= radius*.65)&(across <= radius*1.05)
    far_side = (along <= -radius*2.40)&(along >= -radius*4.20)&(across >= radius*.80)&(across <= radius*1.25)
    if any(region.sum() < 20 for region in (near, far, near_side, far_side)):
        return False
    a, b, c, d = [float(neutral[region].mean()) for region in (near, far, near_side, far_side)]
    strict = a >= .70 and b >= .62 and b-d >= .16 and (a-c >= .18 or b-d >= .35)
    if not strict or not bilateral:
        return strict
    for region, center, contrast in ((near_side, a, .18), (far_side, b, .16)):
        for sign in (-1, 1):
            side = region & (signed_across*sign > 0)
            if side.sum() < 10 or center-float(neutral[side].mean()) < contrast:
                return False
    return True


def audit(args):
    video, calibration = Path(args.video), Path(args.calibration)
    if not video.is_file() or not calibration.is_file():
        raise FileNotFoundError('Explicit video and calibration must exist')
    payload = json.loads(calibration.read_text(encoding='utf-8'))
    if 'profiles' in payload:
        payload = payload['profiles'][args.profile]
    cal = MusicCalibrationData(**payload)
    config = MusicConfig(hold_notes_as_taps=True)
    windows = []
    for value in args.window:
        label, start, end, lane, family = value.split(',')
        windows.append(dict(label=label, start=float(start), end=float(end), lane=int(lane), family=family,
                            frames=[], history=[], previous=None))
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    probe = subprocess.check_output([args.ffprobe, '-v', 'error', '-select_streams', 'v:0',
                                    '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)],
                                   creationflags=flags)
    pts = [float(row['best_effort_timestamp_time']) for row in json.loads(probe)['frames']]
    selected_pts = [p for p in pts if any(w['start'] <= p <= w['end'] for w in windows)]
    selection = '+'.join(f'between(t\\,{w["start"]}\\,{w["end"]})' for w in windows)
    proc = subprocess.Popen([args.ffmpeg, '-v', 'error', '-i', str(video), '-vf',
                             f'select={selection},scale=1280:720', '-fps_mode', 'passthrough',
                             '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    frame_bytes = 1280*720*3
    for stamp in selected_pts:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            raise RuntimeError('Full decoded frame count does not match original PTS')
        image = np.frombuffer(raw, np.uint8).reshape(720, 1280, 3)
        active = [w for w in windows if w['start'] <= stamp <= w['end']]
        gold = detect_hold_tails(image, cal, config, physical_only=True) if any(w['family']=='gold' for w in active) else []
        stars = detect_bonus_star_notes(image, cal) if any(w['family']=='bonus' for w in active) else []
        for window in active:
            row = dict(pts=stamp)
            if window['family'] == 'bonus':
                found = []
                for candidate in stars:
                    projected = assign_lane(candidate, cal)
                    if projected is not None and projected.lane == window['lane']:
                        found.append(dict(box=candidate.box, progress=projected.progress,
                                          old_ribbon=bonus_hold_ribbon_present(image, candidate, projected.tangent),
                                          strict_metadata_ribbon=strict_metadata_ribbon(image, candidate, projected.tangent),
                                          bilateral_metadata_ribbon=strict_metadata_ribbon(image, candidate, projected.tangent, bilateral=True)))
                row['candidates'] = found
            else:
                candidates = [d for d in gold if d.lane == window['lane']]
                row['all_candidates'] = [dict(center=d.center, progress=d.progress, box=d.box) for d in candidates]
                previous = window['previous']
                if previous is None:
                    candidates = [d for d in candidates if .04 <= d.progress <= .40]
                    current = min(candidates, key=lambda d:d.progress) if candidates else None
                else:
                    candidates = [d for d in candidates if previous['progress']-.02 <= d.progress <= previous['progress']+.22]
                    current = min(candidates, key=lambda d:math.dist(d.center, previous['center'])) if candidates else None
                if current is not None:
                    row.update(center=current.center, progress=current.progress, box=current.box)
                    if previous is None or current.center != previous['center'] or current.box != previous['box']:
                        window['history'].append(SimpleNamespace(timestamp=stamp, progress=current.progress))
                    window['previous'] = row
                    for count in (12, 6):
                        slope = weighted_slope(window['history'][-count:])
                        row[f'predicted_hit_{count}'] = stamp+max(0., cal.trigger_progress-current.progress)/slope if slope > 0 else None
            window['frames'].append(row)
    if proc.stdout.read(1):
        raise RuntimeError('Selected decoder produced extra frames')
    err = proc.stderr.read().decode(errors='replace')
    if proc.wait() != 0:
        raise RuntimeError(err)
    for window in windows:
        frames = window['frames']
        if window['family'] == 'gold':
            physical = [r for r in frames if 'progress' in r]
            bracket = next(([a['pts'], b['pts']] for a,b in zip(physical, physical[1:])
                            if a['progress'] < cal.trigger_progress <= b['progress']), None)
            window['observed_geometric_crossing_interval'] = bracket
            window['crossing_is_game_judgement'] = False
            window['comparison'] = {'decision': 'retain-12',
                'reason': 'compressed 30-fps proxy is not original live history; no game timing ground truth'}
            if bracket:
                midpoint = sum(bracket)/2
                selected = [r for r in physical if .50 <= r['progress'] <= .90 and r.get('predicted_hit_12') and r.get('predicted_hit_6')]
                for count in (12,6):
                    errors = [abs(r[f'predicted_hit_{count}']-midpoint)*1000 for r in selected]
                    window['comparison'][f'proxy_{count}'] = dict(samples=len(errors),
                        p95_abs_ms=float(np.percentile(errors,95)) if errors else None,
                        max_abs_ms=max(errors) if errors else None)
        else:
            observations = [r for r in frames for r in r.get('candidates',[])]
            window['summary'] = dict(detected_frames=len(observations),
                old_ribbon_positive=sum(r['old_ribbon'] for r in observations),
                strict_metadata_positive=sum(r['strict_metadata_ribbon'] for r in observations),
                bilateral_metadata_positive=sum(r['bilateral_metadata_ribbon'] for r in observations))
        del window['history'], window['previous']
    report = dict(video=str(video.resolve()), frame_clock='original ffprobe best_effort_timestamp_time; full decode no seek/resampling',
                  selected_frames=len(selected_pts), windows=windows,
                  limitations=['Recorded live gold observations were sampled at 250ms, not a full 12-sample fit history.',
                               'NumPy and compressed-video recognition differ from live Maa capture.',
                               'Geometric crossing is not a game Bad/Miss timing window.',
                               'No runtime source, controller or input operation was changed.'])
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({w['label']:w.get('summary',w.get('comparison')) for w in windows},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('video','calibration','ffmpeg','ffprobe','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--profile',default='7@1280x720')
    parser.add_argument('--window',action='append',required=True,help='label,start,end,lane,gold|bonus')
    audit(parser.parse_args())
