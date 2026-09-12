"""Development-only, timestamped replay. Never opens a real controller.

Candidate JSONL: header {calibration: {...}, config: {...}}, followed by
{time: seconds, candidates: [{box, pixel_count, fill_ratio, center, variant?}]}.
Candidate-only frames use black images; use video replay for ribbon/arc pixels.
Annotations: {heads: [{id, lane, gesture, earliest, latest}]} (seconds).
Video results use NumPy (also used in the Test2 live logs), but compressed
video timing/pixels and mock input are not a full live validation. Annotations and
per-song timestamps belong only under temp/validation, never runtime resources.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]


class ReplayClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0., seconds)


class RecordingContext:
    def __init__(self, clock, action_ms):
        self.clock, self.action_ms, self.actions = clock, action_ms, []

    def run_action_direct(self, kind, param):
        self.actions.append({'time': self.clock(), 'action': kind.value,
                             'contact': getattr(param, 'contact', 0),
                             'target': getattr(param, 'target', None)})
        self.clock.sleep(self.action_ms / 1000.)
        return SimpleNamespace(success=True)


def video_frames(args):
    import numpy as np
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    probe = subprocess.run([str(args.ffprobe), '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(args.video)],
                           capture_output=True, check=True, creationflags=flags)
    times = [float(f['best_effort_timestamp_time']) for f in json.loads(probe.stdout)['frames']
             if 'best_effort_timestamp_time' in f]
    end = args.start + args.duration
    indices = [i for i, t in enumerate(times) if args.start <= t < end]
    if not indices:
        return
    selected = [times[i] for i in indices]
    # ffprobe rounds the decimal PTS; selecting floating-point t in FFmpeg
    # can include a different boundary frame. Decode the same frame indices.
    expression = f"select='between(n,{indices[0]},{indices[-1]})',scale=1280:720:flags=area"
    process = subprocess.Popen([str(args.ffmpeg), '-v', 'error', '-i', str(args.video),
                                '-vf', expression, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    size = 1280 * 720 * 3
    try:
        for timestamp in selected:
            chunks, remaining = [], size
            while remaining:
                chunk = process.stdout.read(remaining)
                if not chunk:
                    raise RuntimeError('Video/PTS frame count mismatch or truncated raw frame')
                chunks.append(chunk)
                remaining -= len(chunk)
            yield timestamp, np.frombuffer(b''.join(chunks), dtype=np.uint8).reshape(720, 1280, 3), None
        if process.stdout.read(1):
            raise RuntimeError('Decoder returned more frames than timestamp manifest')
        error = process.stderr.read().decode(errors='replace')
        if process.wait() != 0:
            raise RuntimeError(error)
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def candidate_frames(args):
    import numpy as np
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    previous = -float('inf')
    with args.candidates.open(encoding='utf-8') as stream:
        next(stream)
        for line in stream:
            item = json.loads(line)
            timestamp = float(item['time'])
            if timestamp <= previous:
                raise ValueError('Frame timestamps must increase strictly')
            previous = timestamp
            yield timestamp, blank, item['candidates']


def annotate(heads, path):
    expected = json.loads(path.read_text(encoding='utf-8'))['heads']
    unused = set(range(len(heads)))
    matches, missed = [], []
    for note in sorted(expected, key=lambda n: (n['earliest'], n['lane'])):
        possibilities = [i for i in unused if heads[i]['lane'] == note['lane']
                         and heads[i]['gesture'] == note.get('gesture', 'Tap')
                         and note['earliest'] <= heads[i]['time'] <= note['latest']]
        if not possibilities:
            missed.append(note['id'])
            continue
        midpoint = (note['earliest'] + note['latest']) / 2.
        index = min(possibilities, key=lambda i: abs(heads[i]['time'] - midpoint))
        unused.remove(index)
        matches.append({'id': note['id'], 'head': index, 'offset_ms': (heads[index]['time'] - midpoint) * 1000.})
    # Only count extra inputs inside the annotated interval, not outside the clip subset.
    begin = min((n['earliest'] for n in expected), default=0.)
    end = max((n['latest'] for n in expected), default=0.)
    return {'matched': matches, 'missing': missed,
            'extra': [heads[i] for i in sorted(unused) if begin <= heads[i]['time'] <= end],
            'note': 'Annotation intervals are observations, not measured game judgement windows.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--candidates', type=Path)
    source.add_argument('--report', type=Path, help='Annotate an existing replay without decoding again')
    parser.add_argument('--branch-root', type=Path, default=ROOT)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=15.)
    parser.add_argument('--action-ms', type=float, default=0.)
    parser.add_argument('--trace-observations', action='store_true', help='Developer-only candidate/track history in the output report')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    binary = WORKSPACE / '.work/ffmpeg-7.1.1-extract/ffmpeg-7.1.1-essentials_build/bin'
    parser.add_argument('--ffmpeg', type=Path, default=binary / 'ffmpeg.exe')
    parser.add_argument('--ffprobe', type=Path, default=binary / 'ffprobe.exe')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'temp'):
        parser.error('Replay output must stay in Double/temp (never runtime resources or the baseline).')
    if args.duration <= 0 or args.action_ms < 0:
        parser.error('Duration must be positive; action cost cannot be negative.')
    if args.report:
        if not args.annotations:
            parser.error('--report requires --annotations')
        report = json.loads(args.report.read_text(encoding='utf-8'))
        report['annotation_result'] = annotate(report['heads'], args.annotations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report['annotation_result'], ensure_ascii=False))
        return 0
    sys.path.insert(0, str(args.branch_root.resolve()))
    import numpy as np
    from agent.music.models import MusicCalibrationData, MusicConfig, MusicCandidate, MusicFrame
    from agent.music.vision import NumpyCandidateProvider, VisualMask
    from agent.music.tracking import MusicVisionEngine
    from agent.music.runtime import MusicRuntime, RuntimeMetrics
    from agent.music.executor import MusicActionExecutor
    from agent.music.storage import metric_summary
    logging.disable(logging.CRITICAL)
    raw_config = {}
    if args.candidates:
        with args.candidates.open(encoding='utf-8') as stream:
            header = json.loads(next(stream))
        raw_cal, raw_config = header['calibration'], header.get('config', {})
    else:
        if not args.calibration:
            parser.error('Video replay requires an explicit calibration JSON; no shared settings are created.')
        raw_cal = json.loads(args.calibration.read_text(encoding='utf-8'))
        if 'profiles' in raw_cal:
            raw_cal = raw_cal['profiles']['7@1280x720']
    cal = MusicCalibrationData(**raw_cal)
    config = MusicConfig(**({'lane_count': 7, 'enable_holds': True} | raw_config))
    provider = NumpyCandidateProvider(cal, config.candidate_iou_threshold, config.candidate_min_size)
    clock = ReplayClock()
    context = RecordingContext(clock, args.action_ms)
    engine = MusicVisionEngine(cal, config)
    runtime = MusicRuntime(context, config, clock=clock, monotonic=clock, sleeper=clock.sleep)
    executor = MusicActionExecutor(context, 1280, 720, config, advanced=True, multi_touch=True, clock=clock, sleeper=clock.sleep)
    pending, samples, frame_count, scheduled = [], [], 0, []
    observation_trace = []
    metrics = RuntimeMetrics()

    def service_until(limit):
        while pending:
            deadlines = sorted({e.deadline for e in pending if e.deadline <= limit})
            if not deadlines:
                break
            before = tuple(e.event_id for e in pending)
            clock.now = max(clock.now, deadlines[0])
            runtime._execute_due(executor, pending, clock(), metrics, engine)
            if before == tuple(e.event_id for e in pending):
                future = [t for t in deadlines if t > clock.now]
                if not future:
                    break
                clock.now = future[0]
        clock.now = max(clock.now, limit)

    stream = video_frames(args) if args.video else candidate_frames(args)
    for sequence, (timestamp, image, raw_candidates) in enumerate(stream):
        service_until(timestamp)
        frame = MusicFrame(sequence, timestamp, timestamp, timestamp, image)
        visual = VisualMask.from_image(image, cal)
        candidates = provider.detect(frame, visual) if raw_candidates is None else [
            MusicCandidate(tuple(c['box']), c['pixel_count'], c['fill_ratio'], tuple(c['center']), c.get('variant', ''))
            for c in raw_candidates]
        started = time.perf_counter()
        new = engine.update(frame, candidates, visual)
        if args.trace_observations:
            observation_trace.append({'time': timestamp, 'tracks': [
                {'id': t.track_id, 'lane': t.lane, 'gesture': t.gesture.value,
                 'state': t.state.value, 'raw_hit': t.predicted_hit_time,
                 'progress': t.observations[-1].progress, 'box': t.observations[-1].candidate.box}
                for t in engine.tracks.values() if t.observations and t.observations[-1].frame_sequence == sequence]})
        pending.extend(new)
        pending.extend(engine.release_events(clock()))
        pending[:] = engine.refine_pending(pending, clock())
        samples.append((time.perf_counter() - started) * 1000.)
        for event in new:
            scheduled.append({'track': event.track_id, 'gesture': event.gesture.value,
                              'lane': event.lane, 'deadline': event.deadline})
        service_until(clock())
        frame_count += 1
    # No blind drain of distant predictions after footage ends.
    heads = []
    for row in runtime.head_action_trace:
        ordinal, tid, lane, gesture, timestamp, late, flags = row.split(':')
        heads.append({'ordinal': int(ordinal), 'track': int(tid), 'lane': int(lane),
                      'gesture': gesture, 'time': float(timestamp), 'late_ms': float(late), 'flags': flags})
    report = {'schema': 1, 'branch': str(args.branch_root.resolve()), 'frames': frame_count,
              'source': str(args.video or args.candidates), 'provider': 'numpy-video (compressed frames, mock input)' if args.video else 'known-candidates',
              'action_ms': args.action_ms, 'timing_ms': metric_summary(samples),
              'heads': heads, 'actions': context.actions, 'scheduled': scheduled,
              'pending_at_clip_end': len(pending), 'game_bad_miss': 'unavailable: offline replay is not the game',
              'trace': list(engine.tap_trace.records) if hasattr(engine, 'tap_trace') else [],
              'observations': observation_trace}
    if args.annotations:
        report['annotation_result'] = annotate(heads, args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('branch', 'frames', 'provider', 'timing_ms', 'pending_at_clip_end')}, ensure_ascii=False))
    print(f'heads={len(heads)} report={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
