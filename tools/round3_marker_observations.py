"""Capture unsampled physical-marker observations from a real loop replay.

All mutations are temporary instrumentation in this process. The replay still
uses actual head receipts, source PTS/config and production input scheduling;
no musical actions or controller connection are introduced.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root', type=Path, required=True)
    parser.add_argument('--probe-output', type=Path, required=True)
    parser.add_argument('--marker', type=int, default=5)
    parser.add_argument('--focus-start', type=float, default=115.9)
    parser.add_argument('--focus-end', type=float, default=117.4)
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    if not args.probe_output.resolve().is_relative_to((APP_ROOT/'temp').resolve()):
        parser.error('Probe output must remain in app/temp')
    branch = args.branch_root.resolve()
    def fingerprint():
        files = {str(path.relative_to(branch)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((branch/'agent').rglob('*.py'))}
        return {'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(), 'files':files}
    source_before = fingerprint()
    sys.path.insert(0,str(APP_ROOT/'tools'))
    sys.path.insert(0,str(branch))
    import tap_replay
    from agent.music.models import MusicFrame
    from agent.music.sustain import SustainMarkerTracker
    from agent.music.tap_trace import TapTrace
    image_pts, captures, observations, event_changes = {}, [], [], []
    original_video = tap_replay.video_frames
    def video_frames(options):
        for pts, image, candidates in original_video(options):
            image_pts[id(image)] = pts
            yield pts, image, candidates
    tap_replay.video_frames = video_frames
    original_frame_init = MusicFrame.__init__
    def frame_init(frame, *positional, **keywords):
        original_frame_init(frame,*positional,**keywords)
        if args.focus_start <= frame.midpoint <= args.focus_end:
            captures.append({'sequence':frame.sequence, 'source_pts':image_pts.get(id(frame.image)),
                'capture_started':frame.capture_started,'capture_finished':frame.capture_finished,
                'midpoint':frame.midpoint})
    MusicFrame.__init__ = frame_init
    original_observe = SustainMarkerTracker.observe
    def observe(tracker, marker_id, detection, frame, owner=None):
        state = original_observe(tracker,marker_id,detection,frame,owner)
        if (state.marker_id == args.marker and args.focus_start <= frame.midpoint <= args.focus_end
                and state.observations and state.observations[-1].frame_sequence == frame.sequence):
            chain = getattr(tracker,'chain',None)
            observations.append({'marker':state.marker_id,'owner':state.owner,
                'source_pts':image_pts.get(id(frame.image)), 'time':frame.midpoint,
                'observation':asdict(state.observations[-1]), 'detection':asdict(detection),
                'coast_budget':chain.coast_budget if chain else None,
                'speed':state.speed(), 'predicted_hit':state.predicted_hit()})
        return state
    SustainMarkerTracker.observe = observe
    original_add = TapTrace.add
    def add(trace,kind,**fields):
        if (args.focus_start <= fields.get('time',-1.) <= args.focus_end
                and (fields.get('marker') == args.marker
                     or fields.get('event') == f'holdnote-{args.marker}'
                     or kind in {'hold_note_cancelled','gold_recovered'})):
            event_changes.append({'kind':kind,**fields})
        return original_add(trace,kind,**fields)
    TapTrace.add = add
    sys.argv = [str(APP_ROOT/'tools/tap_replay.py'),'--branch-root',str(branch),*remaining]
    result = tap_replay.main()
    source_after = fingerprint()
    report = {'branch':str(branch),'source_before':source_before,'source_after':source_after,
        'source_changed_during_replay':source_before['sha256']!=source_after['sha256'],
        'marker':args.marker,'focus_start':args.focus_start,'focus_end':args.focus_end,
        'captures':captures,'marker_observations':observations,'last_four_observations':observations[-4:],
        'event_changes':event_changes,'meaning':'Unsampled diagnostic observations; source PTS from decoded image identity. Host capture midpoint is not exact game scene time. No game BM inferred.'}
    args.probe_output.parent.mkdir(parents=True,exist_ok=True)
    args.probe_output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.probe_output),'observations':len(observations),
        'source_changed_during_replay':report['source_changed_during_replay'],
        'last_four':[{k:row[k] for k in ('source_pts','time','coast_budget','speed','predicted_hit')}
                     for row in observations[-4:]]}))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
